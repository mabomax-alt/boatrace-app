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

# 16方位コード → 方角ラベル
WIND_DIR_LABELS = {
    1: "北", 2: "北北東", 3: "北東", 4: "東北東",
    5: "東", 6: "東南東", 7: "南東", 8: "南南東",
    9: "南", 10: "南南西", 11: "南西", 12: "西南西",
    13: "西", 14: "西北西", 15: "北西", 16: "北北西",
}

# 追い風方向コード（南〜西系 = 一般的に1コース追い風）
_TAIL_WIND_CODES = {7, 8, 9, 10, 11, 12, 13}
# 向かい風方向コード（北〜東系）
_HEAD_WIND_CODES = {1, 2, 3, 4, 5, 15, 16}


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


@st.cache_data(ttl=120)
def fetch_beforeinfo(jcd: str, date_str: str, rno: int):
    """直前情報（展示タイム・チルト）と気象情報を取得する（2分キャッシュ）"""
    url = f"{BASE_URL}/beforeinfo?jcd={jcd}&hd={date_str}&rno={rno:02d}"
    try:
        resp = requests.get(url, headers=HEADERS, timeout=15)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.content, "html.parser")
        before_data = _parse_beforeinfo(soup)
        weather = _parse_weather(soup)
        return before_data, weather, url, None
    except requests.RequestException as e:
        return [], {}, url, str(e)


def _parse_racers(soup: BeautifulSoup) -> list:
    """出走表テーブルから6艇分のデータブロック（tbody）を取得し、
    インデックス順に処理する。クラス名には一切依存しない。

    構造: 直接の子 tr がちょうど4本ある tbody が各艇のブロック。
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


def _parse_beforeinfo(soup: BeautifulSoup) -> list:
    """直前情報テーブル（page内2番目のtable）から展示タイム・チルトを艇番順に取得する。

    table[1] の構造:
      各 tbody = 1艇分（6tbody）
      tbody の tr[0]: td[4]=展示タイム, td[5]=チルト
    """
    tables = soup.find_all("table")
    if len(tables) < 2:
        return []

    result = []
    for tb in tables[1].find_all("tbody")[:6]:
        entry = {"展示タイム": 0.0, "チルト": 0.0}
        trs = tb.find_all("tr")
        if trs:
            tds = trs[0].find_all("td")
            try:
                entry["展示タイム"] = float(tds[4].get_text(strip=True)) if len(tds) > 4 else 0.0
            except (ValueError, IndexError):
                pass
            try:
                entry["チルト"] = float(tds[5].get_text(strip=True)) if len(tds) > 5 else 0.0
            except (ValueError, IndexError):
                pass
        result.append(entry)
    return result


def _parse_weather(soup: BeautifulSoup) -> dict:
    """beforeinfo ページから気象情報（風速・風向・天候）を取得する。

    weather1_bodyUnit 系要素のクラス名で項目を識別する:
      is-wind           → 風速 (LabelData に数値)
      is-windDirection  → 風向 (子 p の is-wind{N} クラスで方位コード)
      is-weather        → 天候 (LabelTitle テキスト)
    """
    result: dict = {
        "wind_speed": 0.0,
        "wind_dir_code": 0,
        "wind_dir_label": "不明",
        "weather_label": "",
    }
    for el in soup.find_all(class_="weather1_bodyUnit"):
        classes = el.get("class", [])
        if "is-wind" in classes and "is-windDirection" not in classes:
            span = el.find(class_="weather1_bodyUnitLabelData")
            if span:
                m = re.search(r"\d+", span.get_text(strip=True))
                if m:
                    result["wind_speed"] = float(m.group())
        elif "is-windDirection" in classes:
            p = el.find("p")
            if p:
                for cls in p.get("class", []):
                    m = re.fullmatch(r"is-wind(\d+)", cls)
                    if m:
                        code = int(m.group(1))
                        result["wind_dir_code"] = code
                        result["wind_dir_label"] = WIND_DIR_LABELS.get(code, f"方向{code}")
        elif "is-weather" in classes:
            sp = el.find(class_="weather1_bodyUnitLabelTitle")
            if sp:
                result["weather_label"] = sp.get_text(strip=True)
    return result


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


def generate_prediction(
    stadium: str,
    wind_speed: float,
    wind_dir_code: int,
    racers: list,
) -> list:
    """出走表・直前情報・気象情報を組み合わせてAI予測確率を生成する。

    特徴量:
      [既存] 当地勝率・全国勝率・モーター2連率・級別
      [追加] 展示タイム: 平均より速い艇を微増、遅い艇を微減
      [追加] チルト: プラスチルト→外艇（ダッシュ）有利、マイナス→内艇有利
      [追加] 風向: 追い風系→1コース微増、強い向かい風→外艇微増
    """
    probs = base_probs(stadium)[:]

    if "江戸川" in stadium and wind_speed > 4.0:
        probs = [0.30, 0.15, 0.15, 0.15, 0.13, 0.12]

    if racers:
        scores = []
        for r in racers:
            s = (
                r["当地勝率"] * 0.5
                + r["全国勝率"] * 0.3
                + (r["モーター2連率"] / 100.0) * 0.1
                + GRADE_SCORE.get(r["級別"], 0.5) * 0.1
            )
            scores.append(max(s, 0.01))

        while len(scores) < 6:
            scores.append(0.01)

        total_score = sum(scores)
        score_probs = [s / total_score for s in scores]
        probs = [0.5 * p + 0.5 * sp for p, sp in zip(probs, score_probs)]

    # --- 展示タイム補正 ---
    # 展示タイムが速い（値が小さい）艇ほど当日のエンジン状態が良い
    tenji_times = [
        racers[i].get("展示タイム", 0.0) if i < len(racers) else 0.0
        for i in range(6)
    ]
    valid_times = [t for t in tenji_times if t > 0]
    if valid_times:
        mean_t = sum(valid_times) / len(valid_times)
        for i, t in enumerate(tenji_times):
            if t > 0:
                # 差×0.5 → 0.1秒速ければ約+0.05、遅ければ-0.05
                probs[i] = max(probs[i] + (mean_t - t) * 0.5, 0.001)

    # --- チルト補正 ---
    # プラス: ダッシュ艇有利 → アウトコース(4-6)を微増
    # マイナス: スロー安定走行 → インコース(1-3)を微増
    for i in range(min(len(racers), 6)):
        tilt = racers[i].get("チルト", 0.0)
        boat = i + 1
        if tilt > 0:
            factor = 0.005 if boat >= 4 else -0.003
            probs[i] = max(probs[i] + factor * tilt, 0.001)
        elif tilt < 0:
            factor = 0.005 if boat <= 3 else -0.003
            probs[i] = max(probs[i] + factor * abs(tilt), 0.001)

    # --- 風向補正 ---
    # 追い風: 1コース（イン）の加速を後押し → 1号艇微増
    # 向かい風強風: スタート乱れやすくアウト艇有利
    if wind_speed > 3.0 and wind_dir_code > 0:
        if wind_dir_code in _TAIL_WIND_CODES:
            probs[0] *= 1.05
        elif wind_dir_code in _HEAD_WIND_CODES and wind_speed > 5.0:
            for i in range(3, 6):
                probs[i] *= 1.03

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

# --- セッション状態の初期化（ウィジェット描画より前に行う）---
_SS_DEFAULTS = {
    "racers": [],
    "before_data": [],
    "weather": {},
    "fetch_url": "",
    "wind_dir_code": 0,
    "wind_speed_val": 2.0,
}
for _k, _v in _SS_DEFAULTS.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v

st.title("⚓ ボートレース予測AI")
st.caption("関東5会場（桐生・戸田・江戸川・平和島・多摩川）専用")

# --- 入力フォーム ---
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
    wind_speed = st.number_input(
        "風速 (m/s)",
        min_value=0.0, max_value=15.0,
        value=float(st.session_state.wind_speed_val),
        step=0.5,
    )

fetch_btn = st.button("データ取得（出走表＋直前情報）", type="primary", use_container_width=True)

# --- データ取得処理 ---
if fetch_btn:
    jcd = STADIUM_CODES[stadium]
    date_str = race_date.strftime("%Y%m%d")

    with st.spinner("boatrace.jp からデータを取得中..."):
        racers_raw, url, err_race = fetch_racelist(jcd, date_str, int(race_num))
        before_data, weather, _before_url, err_before = fetch_beforeinfo(jcd, date_str, int(race_num))

    if err_race:
        st.error(f"出走表の取得失敗: {err_race}")
        st.session_state.racers = []
    elif not racers_raw:
        st.warning("出走表が取得できませんでした。日付・会場・レース番号を確認してください。")
        st.session_state.racers = []
    else:
        # キャッシュされたリストを直接変更しないようコピーする
        st.session_state.racers = [dict(r) for r in racers_raw]
        st.session_state.fetch_url = url

    st.session_state.before_data = before_data if before_data else []

    if weather:
        st.session_state.weather = weather
        st.session_state.wind_dir_code = weather.get("wind_dir_code", 0)
        fetched_ws = weather.get("wind_speed", 0.0)
        if fetched_ws > 0:
            st.session_state.wind_speed_val = fetched_ws  # 次回rerunで風速欄に反映

    if err_before or not before_data:
        st.success(f"{len(st.session_state.racers)}艇分の出走表を取得しました（直前情報は未公開）")
    else:
        st.success(f"出走表＋直前情報（展示タイム・チルト・気象）を取得しました！")

# --- 表示用に出走表と直前情報をマージ ---
racers = st.session_state.racers
before_data = st.session_state.before_data
weather = st.session_state.weather
wind_dir_code = st.session_state.wind_dir_code

merged_racers = [
    {**r, **before_data[i]} if i < len(before_data) else r
    for i, r in enumerate(racers)
]

# --- 気象情報の表示 ---
if weather:
    st.divider()
    wc = st.columns(3)
    wc[0].metric("天候", weather.get("weather_label", "-"))
    wc[1].metric("風向", weather.get("wind_dir_label", "-"))
    wc[2].metric("風速（自動取得）", f"{weather.get('wind_speed', 0):.0f} m/s")

st.divider()

# --- AI予測結果 ---
st.subheader(f"【{stadium}】 第{int(race_num)}レース")

if "江戸川" in stadium and wind_speed > 4.0:
    st.warning("⚠️ 難水面の江戸川で強風。万舟の可能性あり。")

try:
    probabilities = generate_prediction(stadium, wind_speed, wind_dir_code, merged_racers)
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
        r = merged_racers[idx] if idx < len(merged_racers) else {}
        name = r.get("選手名", f"{idx + 1}号艇")
        grade = r.get("級別", "")
        tenji = r.get("展示タイム", 0.0)
        tilt = r.get("チルト", 0.0)
        tenji_str = f"展示{tenji:.2f}" if tenji > 0 else ""
        tilt_str = f"チルト{tilt:+.1f}" if tilt != 0.0 else ""
        sub = "　".join(filter(None, [tenji_str, tilt_str]))
        with col:
            st.metric(
                label=f"{BOAT_COLORS[idx]} {idx + 1}号艇　{name}　{grade}",
                value=f"{probabilities[idx] * 100:.1f}%",
                delta=sub if sub else None,
                delta_color="off",
            )

st.divider()

# --- おすすめ買い目 ---
st.subheader("おすすめ買い目（3連単）")
for bet in recommend_bets(stadium, probabilities):
    st.success(f"## {bet}")

# --- 出走表（折りたたみ）---
if merged_racers:
    st.divider()
    with st.expander("出走表・直前情報（詳細）"):
        df = pd.DataFrame(merged_racers)
        display_cols = [
            "艇番", "選手名", "級別",
            "展示タイム", "チルト",
            "当地勝率", "全国勝率", "モーター2連率",
        ]
        df_display = df[[c for c in display_cols if c in df.columns]].set_index("艇番")
        st.dataframe(df_display, use_container_width=True)
        if st.session_state.fetch_url:
            st.caption(f"出走表URL: {st.session_state.fetch_url}")
else:
    st.info("「データ取得」を押すと boatrace.jp から出走表を取得し、予測精度が向上します。")
