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


def _parse_racers(soup: BeautifulSoup) -> list:
    """出走表テーブルから6艇分のデータブロック（tbody）を取得し、
    インデックス順に処理する。クラス名には一切依存しない。

    構造: 直接の子 tr がちょうど4本ある tbody が各艇のブロック。
    ページ内に該当する tbody が6つ存在することを前提とする。
    艇番は enumerate で決定（1-indexed）。
    """
    boat_tbodies = [
        tb for tb in soup.find_all("tbody")
        if sum(1 for c in tb.children if getattr(c, "name", None) == "tr") == 4
    ]

    return [
        _extract_racer(
            i + 1,
            [c for c in tb.children if getattr(c, "name", None) == "tr"],
        )
        for i, tb in enumerate(boat_tbodies)
    ]


def _extract_racer(boat_num: int, rows: list) -> dict:
    """1艇分の4行から tr[0] の固定 td インデックスでデータを抽出する。
    クラス名には依存せず、すべて出現順（インデックス）で取得する。

    tr[0] の td レイアウト:
      td[0] : 艇番セル
      td[1] : 空
      td[2] : 選手情報セル — div[0]=登録番号/級別, div[1]=選手名, div[2]=属性
      td[3] : F/L/平均ST
      td[4] : 全国勝率・2連率・3連率（連結文字列）
      td[5] : 当地勝率・2連率・3連率（連結文字列）
      td[6] : モーター 出走数+2連率+3連率（連結文字列）
      td[7] : ボート 出走数+2連率+3連率（連結文字列）
    """
    empty = {
        "艇番": boat_num, "選手名": f"{boat_num}号艇", "級別": "",
        "全国勝率": 0.0, "全国2連率": 0.0,
        "当地勝率": 0.0, "当地2連率": 0.0,
        "モーター2連率": 0.0, "ボート2連率": 0.0,
    }
    if not rows:
        return empty

    tds = [c for c in rows[0].children if getattr(c, "name", None) == "td"]

    # td[2] の div を出現順に取得 — div[0]=登録番号/級別, div[1]=選手名
    name = ""
    grade = ""
    if len(tds) > 2:
        divs = tds[2].find_all("div")
        if len(divs) > 1:
            name = divs[1].get_text(strip=True)
        if divs:
            m = re.search(r"[AB][12]", divs[0].get_text(strip=True))
            if m:
                grade = m.group()

    def _floats(idx: int) -> list:
        """全国/当地セル用: '6.8554.7269.81' → [6.85, 54.72, 69.81]"""
        if len(tds) <= idx:
            return []
        return [float(v) for v in re.findall(r"\d+\.\d{2}", tds[idx].get_text(strip=True))]

    def _two_rate(idx: int) -> float:
        """モーター/ボートセル用: '3732.2654.84' → 32.26（先頭の XX.XX パターン）"""
        if len(tds) <= idx:
            return 0.0
        vals = re.findall(r"\d{2}\.\d{2}", tds[idx].get_text(strip=True))
        return float(vals[0]) if vals else 0.0

    national = _floats(4)
    local = _floats(5)

    return {
        "艇番": boat_num,
        "選手名": name or f"{boat_num}号艇",
        "級別": grade,
        "全国勝率": national[0] if len(national) > 0 else 0.0,
        "全国2連率": national[1] if len(national) > 1 else 0.0,
        "当地勝率": local[0] if len(local) > 0 else 0.0,
        "当地2連率": local[1] if len(local) > 1 else 0.0,
        "モーター2連率": _two_rate(6),
        "ボート2連率": _two_rate(7),
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

st.set_page_config(page_title="ボートレース予測AI", page_icon="⚓", layout="centered")

# --- パスワード認証 ---
_CORRECT_PASSWORD = os.environ.get("BOATRACE_APP_PASSWORD", "")

if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

if not st.session_state.authenticated:
    st.title("⚓ ボートレース予測AI")
    st.markdown("### ログイン")
    pw = st.text_input("パスワードを入力してください", type="password", key="pw_input")
    if st.button("ログイン", type="primary", use_container_width=True):
        if _CORRECT_PASSWORD and hmac.compare_digest(pw.encode(), _CORRECT_PASSWORD.encode()):
            st.session_state.authenticated = True
            st.rerun()
        else:
            st.error("パスワードが正しくありません。")
    st.stop()

st.title("⚓ ボートレース予測AI")
st.caption("関東5会場（桐生・戸田・江戸川・平和島・多摩川）専用")

# --- 入力フォーム（サイドバー廃止・メイン1カラムに統合）---
stadium = st.selectbox("開催会場", list(STADIUM_CODES.keys()))

c1, c2 = st.columns(2)
with c1:
    race_date = st.date_input("開催日", value=date.today())
with c2:
    race_num = st.number_input("レース番号", min_value=1, max_value=12, value=12, step=1)

c3, c4 = st.columns(2)
with c3:
    weather_condition = st.selectbox("天候", ["晴", "曇", "雨", "雪"])
with c4:
    wind_speed = st.number_input("風速 (m/s)", min_value=0.0, max_value=15.0, value=2.0, step=0.5)

fetch_btn = st.button("データ取得", type="primary", use_container_width=True)

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
        st.error(f"取得失敗: {error}")
        st.session_state.racers = []
    elif not racers:
        st.warning("出走表が取得できませんでした。日付・会場・レース番号を確認してください。")
        st.session_state.racers = []
    else:
        st.session_state.racers = racers
        st.success(f"{len(racers)}艇分のデータを取得しました！")

racers = st.session_state.racers

st.divider()

# --- AI予測結果 ---
st.subheader(f"【{stadium}】 第{int(race_num)}レース")

if "江戸川" in stadium and wind_speed > 4.0:
    st.warning("⚠️ 難水面の江戸川で強風。万舟の可能性あり。")

try:
    probabilities = generate_prediction(stadium, wind_speed, racers)
    if len(probabilities) != 6:
        raise ValueError(f"予測値が6艇分ありません（{len(probabilities)}艇分）")
except Exception as e:
    st.error(f"予測の生成中にエラーが発生しました: {e}")
    probabilities = [1 / 6] * 6

# 予測を2列×3行で表示
BOAT_COLORS = ["🔴", "⚫", "⬜", "🔵", "🟡", "🟢"]
for row in range(3):
    left, right = st.columns(2)
    for col, idx in zip([left, right], [row * 2, row * 2 + 1]):
        name = racers[idx]["選手名"] if idx < len(racers) else f"{idx + 1}号艇"
        grade = racers[idx]["級別"] if idx < len(racers) else ""
        with col:
            st.metric(
                label=f"{BOAT_COLORS[idx]} {idx + 1}号艇　{name}　{grade}",
                value=f"{probabilities[idx] * 100:.1f}%",
            )

st.divider()

# --- おすすめ買い目 ---
st.subheader("おすすめ買い目（3連単）")
for bet in recommend_bets(stadium, probabilities):
    st.success(f"## {bet}")

# --- 出走表（折りたたみ）---
if racers:
    st.divider()
    with st.expander("出走表（詳細）"):
        df = pd.DataFrame(racers)
        display_cols = ["艇番", "選手名", "級別", "当地勝率", "全国勝率", "モーター2連率"]
        df_display = df[[c for c in display_cols if c in df.columns]].set_index("艇番")
        st.dataframe(df_display, use_container_width=True)
        if st.session_state.fetch_url:
            st.caption(f"取得元: {st.session_state.fetch_url}")
else:
    st.info("「データ取得」を押すと boatrace.jp から出走表を取得し、予測精度が向上します。")
