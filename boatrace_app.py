import streamlit as st
import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
from datetime import date
import re
import os
import hmac

# ===== 定数 =====
STADIUM_CODES = {
    "桐生 (ナイター)": "01",
    "戸田": "02",
    "江戸川 (波・潮注意)": "03",
    "平和島": "04",
    "多摩川 (静水面)": "05",
}

BASE_URL = "https://www.boatrace.jp/owpc/pc/race"
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "ja,en-US;q=0.9",
}
GRADE_SCORE = {"A1": 1.0, "A2": 0.8, "B1": 0.6, "B2": 0.4, "": 0.5}


# ===== スクレイピング =====

@st.cache_data(ttl=300)
def fetch_racelist(jcd: str, date_str: str, rno: int):
    """boatrace.jp 出走表から6艇分のデータを取得する（5分キャッシュ）"""
    url = f"{BASE_URL}/racelist?jcd={jcd}&hd={date_str}&rno={rno:02d}"
    try:
        resp = requests.get(url, headers=HEADERS, timeout=15)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.content, "html.parser")
        racers = _parse_racers(soup)
        return racers, url, None
    except requests.RequestException as e:
        return [], url, str(e)


def _boat_num_from_td(td) -> int | None:
    """is-boatColor[1-6] クラスから艇番を返す。該当なければ None。"""
    for cls in td.get("class", []):
        m = re.fullmatch(r"is-boatColor([1-6])", cls)
        if m:
            return int(m.group(1))
    return None


def _is_anchor_tr(tr) -> bool:
    """tr の直接の子 td に is-boatColor[1-6] があれば True。
    再帰検索を使わないことで、ネストされた内部テーブルの影響を排除する。"""
    return any(
        _boat_num_from_td(td) is not None
        for td in tr.children
        if getattr(td, "name", None) == "td"
    )


def _parse_racers(soup: BeautifulSoup) -> list:
    """soup から全艇のデータを解析する。

    各艇の先頭 tr（anchor tr）を起点に、次の anchor tr の直前まで
    sibling の tr を収集して行グループを作る。
    - 艇番はクラス名から取得（テキスト内容に依存しない）
    - anchor 判定は直接の子 td のみ対象（ネスト table を誤検知しない）
    - 全艇が1つの tbody でも艇ごとに tbody が分かれていても動作する
    """
    racer_row_map: dict = {}

    for td in soup.find_all("td", class_=re.compile(r"is-boatColor[1-6]")):
        boat_num = _boat_num_from_td(td)
        if boat_num is None or boat_num in racer_row_map:
            continue

        first_tr = td.find_parent("tr")
        if first_tr is None:
            continue

        # first_tr の直接親（tbody / table）から直接子 tr だけを列挙
        container = first_tr.parent
        if container is None:
            racer_row_map[boat_num] = [first_tr]
            continue

        sibling_trs = [
            child for child in container.children
            if getattr(child, "name", None) == "tr"
        ]

        # first_tr の開始インデックスを特定
        start_idx = next(
            (i for i, t in enumerate(sibling_trs) if t is first_tr), None
        )
        if start_idx is None:
            racer_row_map[boat_num] = [first_tr]
            continue

        # first_tr から次の anchor tr の直前まで収集
        rows = []
        for tr in sibling_trs[start_idx:]:
            if tr is not first_tr and _is_anchor_tr(tr):
                break  # 次の艇の先頭行に到達
            rows.append(tr)

        racer_row_map[boat_num] = rows

    return [_extract_racer(n, rows) for n, rows in sorted(racer_row_map.items())]


def _extract_racer(boat_num: int, rows: list) -> dict:
    """1艇分の行リストからデータを抽出する。
    選手名は最初に見つかった1件のみ採用（上書きしない）。"""
    name = ""
    grade = ""
    float_vals = []

    for row in rows:
        for td in row.find_all("td"):
            # 選手名: 日本語テキストを含む大きめフォントの div（最初の1件で確定）
            if not name:
                for div in td.find_all("div"):
                    classes = " ".join(div.get("class", []))
                    text = div.get_text(strip=True)
                    if (
                        ("is-fs18" in classes or "is-fBold" in classes)
                        and re.search(r"[　-鿿]", text)
                        and not re.fullmatch(r"[AB][12]", text)
                    ):
                        name = text
                        break

            cell_text = td.get_text(strip=True)

            # 級別: A1/A2/B1/B2
            if not grade and re.fullmatch(r"[AB][12]", cell_text):
                grade = cell_text

            # 数値 (x.xx 形式の勝率・2連率)
            for m in re.finditer(r"\b\d+\.\d{2}\b", cell_text):
                val = float(m.group())
                if val <= 100.0:
                    float_vals.append(val)

    def _get(idx):
        return float_vals[idx] if idx < len(float_vals) else 0.0

    return {
        "艇番": boat_num,
        "選手名": name or f"{boat_num}号艇",
        "級別": grade,
        "全国勝率": _get(0),
        "全国2連率": _get(1),
        "当地勝率": _get(2),
        "当地2連率": _get(3),
        "モーター2連率": _get(4),
        "ボート2連率": _get(5),
    }


# ===== 予測ロジック =====

def base_probs(stadium: str) -> list:
    if "戸田" in stadium:
        return [0.35, 0.18, 0.22, 0.15, 0.06, 0.04]
    elif "江戸川" in stadium:
        return [0.32, 0.17, 0.18, 0.16, 0.10, 0.07]
    elif "多摩川" in stadium:
        return [0.48, 0.22, 0.13, 0.09, 0.05, 0.03]
    elif "桐生" in stadium:
        return [0.42, 0.20, 0.16, 0.12, 0.06, 0.04]
    else:  # 平和島
        return [0.40, 0.20, 0.17, 0.12, 0.07, 0.04]


def generate_prediction(stadium: str, wind: float, racers: list) -> list:
    probs = base_probs(stadium)[:]

    if "江戸川" in stadium and wind > 4.0:
        probs = [0.30, 0.15, 0.15, 0.15, 0.13, 0.12]

    if racers:
        # スコア = 当地勝率×0.5 + 全国勝率×0.3 + モーター2連率×0.1 + 級別補正×0.1
        scores = []
        for r in racers:
            s = (
                r["当地勝率"] * 0.5
                + r["全国勝率"] * 0.3
                + (r["モーター2連率"] / 100.0) * 0.1
                + GRADE_SCORE.get(r["級別"], 0.5) * 0.1
            )
            scores.append(max(s, 0.01))

        # 6艇未満の場合は残り枠を最小スコアで埋めて常に6要素にする
        while len(scores) < 6:
            scores.append(0.01)

        total_score = sum(scores)
        score_probs = [s / total_score for s in scores]
        # 会場バイアスとスコアを 50:50 でブレンド
        probs = [0.5 * p + 0.5 * sp for p, sp in zip(probs, score_probs)]

    total = sum(probs)
    return [p / total for p in probs]


def recommend_bets(stadium: str, probs: list) -> list:
    ranking = sorted(range(6), key=lambda i: -probs[i])
    a, b, c = ranking[0] + 1, ranking[1] + 1, ranking[2] + 1

    if "戸田" in stadium:
        return [f"{a}-{b}-全", f"3-1-全", f"4-1-全"]
    elif "多摩川" in stadium:
        return [f"{a}-{b}-{c}", f"{a}-{b}-全", f"{a}-全-{b}"]
    else:
        return [f"{a}-{b}-全", f"{a}-全-{b}", f"{b}-{a}-全"]


# ===== UI =====

st.set_page_config(page_title="ボートレース予測AI", page_icon="⚓", layout="wide")

# --- パスワード認証 ---
_CORRECT_PASSWORD = os.environ.get("BOATRACE_APP_PASSWORD", "")

if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

if not st.session_state.authenticated:
    st.title("⚓ ボートレース予測AI")
    st.markdown("### ログイン")
    pw = st.text_input("パスワードを入力してください", type="password", key="pw_input")
    if st.button("ログイン", type="primary"):
        if _CORRECT_PASSWORD and hmac.compare_digest(pw.encode(), _CORRECT_PASSWORD.encode()):
            st.session_state.authenticated = True
            st.rerun()
        else:
            st.error("パスワードが正しくありません。")
    st.stop()

st.title("⚓ 関東限定 ボートレース予測AIアプリ")
st.caption("桐生・戸田・江戸川・平和島・多摩川の5会場に特化した予測システム")

# サイドバー
st.sidebar.header("レース情報の入力")
stadium = st.sidebar.selectbox("開催会場を選択してください", list(STADIUM_CODES.keys()))
race_date = st.sidebar.date_input("開催日", value=date.today())
race_num = st.sidebar.slider("レース番号", 1, 12, 12)
weather_condition = st.sidebar.selectbox("天候", ["晴", "曇", "雨", "雪"])
wind_speed = st.sidebar.number_input("風速 (m/s)", min_value=0.0, max_value=15.0, value=2.0, step=0.5)
fetch_btn = st.sidebar.button("データ取得", type="primary", use_container_width=True)

# セッション状態
if "racers" not in st.session_state:
    st.session_state.racers = []
if "fetch_url" not in st.session_state:
    st.session_state.fetch_url = ""

# データ取得処理
if fetch_btn:
    jcd = STADIUM_CODES[stadium]
    date_str = race_date.strftime("%Y%m%d")
    with st.spinner("boatrace.jp からデータを取得中..."):
        racers, url, error = fetch_racelist(jcd, date_str, race_num)
    st.session_state.fetch_url = url
    if error:
        st.sidebar.error(f"取得失敗: {error}")
        st.session_state.racers = []
    elif not racers:
        st.sidebar.warning("出走表が取得できませんでした。日付・会場・レース番号を確認してください。")
        st.session_state.racers = []
    else:
        st.session_state.racers = racers
        st.sidebar.success(f"{len(racers)}艇分のデータを取得しました！")

racers = st.session_state.racers

# メイン表示
col1, col2 = st.columns([1.3, 1])

with col1:
    st.subheader(f"【{stadium}】 第{race_num}レース")

    if racers:
        st.markdown("##### 出走表（boatrace.jp より取得）")
        df = pd.DataFrame(racers)
        display_cols = ["艇番", "選手名", "級別", "当地勝率", "全国勝率", "モーター2連率", "ボート2連率"]
        df_display = df[[c for c in display_cols if c in df.columns]].set_index("艇番")
        st.dataframe(df_display, use_container_width=True)
        if st.session_state.fetch_url:
            st.caption(f"取得元: {st.session_state.fetch_url}")
    else:
        st.info("サイドバーの「データ取得」を押すと boatrace.jp から出走表を取得し、予測精度が向上します。")

with col2:
    st.subheader("AI予測結果")

    if "江戸川" in stadium and wind_speed > 4.0:
        st.warning("⚠️ 難水面の江戸川で強風。万舟の可能性あり。")

    try:
        probabilities = generate_prediction(stadium, wind_speed, racers)
        if len(probabilities) != 6:
            raise ValueError(f"予測値が6艇分ありません（{len(probabilities)}艇分）")
        predict_df = pd.DataFrame({
            "コース (艇番)": [f"{i}号艇" for i in range(1, 7)],
            "AI勝率予測": [f"{p * 100:.1f}%" for p in probabilities],
        })
        st.table(predict_df)
    except Exception as e:
        st.error(f"予測の生成中にエラーが発生しました: {e}")
        probabilities = [1 / 6] * 6

    st.subheader("おすすめ買い目（3連単）")
    for bet in recommend_bets(stadium, probabilities):
        st.success(f"**{bet}**")
