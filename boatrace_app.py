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
        "平均ST": 0.18,
    }
    if not rows:
        return empty

    tds = [c for c in rows[0].children if getattr(c, "name", None) == "td"]

    # td[2] の div を出現順に取得 — div[0]=登録番号/級別, div[1]=選手名
    name = ""
    grade = ""
    avg_st = 0.18
    if len(tds) > 2:
        divs = tds[2].find_all("div")
        if len(divs) > 1:
            name = divs[1].get_text(strip=True)
        if divs:
            m = re.search(r"[AB][12]", divs[0].get_text(strip=True))
            if m:
                grade = m.group()
    if len(tds) > 3:
        m = re.search(r"0\.\d{2}", tds[3].get_text(strip=True))
        if m:
            avg_st = float(m.group())

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
        "平均ST": avg_st,
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


# ===== 今節成績 =====

@st.cache_data(ttl=3600)
def fetch_race_result(jcd: str, date_str: str, rno: int) -> list:
    """完了レースの着順・艇番・選手名を取得する（1時間キャッシュ）。"""
    url = f"{BASE_URL}/raceresult?jcd={jcd}&hd={date_str}&rno={rno:02d}"
    try:
        resp = requests.get(url, headers=HEADERS, timeout=10)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.content, "html.parser")
        return _parse_race_result(soup)
    except Exception:
        return []


def _parse_race_result(soup: BeautifulSoup) -> list:
    """raceresult ページから着順・艇番・選手名を抽出する。"""
    results: list = []
    seen_ranks: set = set()

    for table in soup.find_all("table"):
        for tr in table.find_all("tr"):
            tds = tr.find_all("td")
            if len(tds) < 4:
                continue
            rank_text = tds[0].get_text(strip=True)
            if not rank_text.isdigit():
                continue
            rank = int(rank_text)
            if rank < 1 or rank > 6 or rank in seen_ranks:
                continue
            boat_text = tds[1].get_text(strip=True)
            if not boat_text.isdigit():
                continue
            boat_num = int(boat_text)
            if not (1 <= boat_num <= 6):
                continue
            name = ""
            for td in tds[2:6]:
                text = td.get_text(strip=True)
                if re.search(r'[぀-鿿]', text) and 2 <= len(text) <= 12:
                    name = text
                    break
            results.append({"着順": rank, "艇番": boat_num, "選手名": name})
            seen_ranks.add(rank)
        if len(results) >= 6:
            break

    return sorted(results, key=lambda x: x["着順"])


def _accumulate_results(totals: dict, race_results: list) -> None:
    """1レースの結果を選手別成績に累計加算する。"""
    for r in race_results:
        name = r.get("選手名", "").strip()
        if not name:
            continue
        if name not in totals:
            totals[name] = {"1着": 0, "2着": 0, "出走": 0}
        totals[name]["出走"] += 1
        rank = r.get("着順", 9)
        if rank == 1:
            totals[name]["1着"] += 1
        elif rank == 2:
            totals[name]["2着"] += 1


@st.cache_data(ttl=300)
def fetch_setsunai_results(jcd: str, date_str: str, current_rno: int) -> dict:
    """今節の全レース（初日〜当日の直前レースまで）を集計し選手別成績を返す。

    今節期間: 当日の前レース全件 + 最大5日前まで遡り（結果が存在する日のみ）

    Returns: {選手名: {"1着": n, "2着": n, "出走": n}}
    """
    from datetime import datetime, timedelta

    totals: dict = {}
    base_date = datetime.strptime(date_str, "%Y%m%d")

    for rno in range(1, current_rno):
        _accumulate_results(totals, fetch_race_result(jcd, date_str, rno))

    for days_back in range(1, 6):
        prev_date = base_date - timedelta(days=days_back)
        prev_date_str = prev_date.strftime("%Y%m%d")
        if not fetch_race_result(jcd, prev_date_str, 1):
            break
        for rno in range(1, 13):
            race_res = fetch_race_result(jcd, prev_date_str, rno)
            if not race_res:
                break
            _accumulate_results(totals, race_res)

    return totals


def identify_hot_players(setsunai_results: dict) -> dict:
    """今節好調選手を特定し補正係数を返す。

    好調条件（出走2以上）:
      1着2以上 → 1.15 / 1着1かつトップ2率50%以上 → 1.10 / トップ2率60%以上 → 1.08
    """
    hot: dict = {}
    for name, stats in setsunai_results.items():
        runs = stats["出走"]
        wins = stats["1着"]
        top2 = wins + stats["2着"]
        if runs < 2:
            continue
        top2_rate = top2 / runs
        if wins >= 2:
            hot[name] = 1.15
        elif wins >= 1 and top2_rate >= 0.5:
            hot[name] = 1.10
        elif top2_rate >= 0.6:
            hot[name] = 1.08
    return hot


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


def _gen_formations(firsts: list, seconds: list, thirds: list) -> list:
    """3連単フォーメーションを生成する（重複・同艇除外）"""
    bets: list = []
    for f in firsts:
        for s in seconds:
            if s == f:
                continue
            for t in thirds:
                if t == f or t == s:
                    continue
                bet = f"{f}-{s}-{t}"
                if bet not in bets:
                    bets.append(bet)
    return bets


def generate_prediction(
    stadium: str,
    wind_speed: float,
    wind_dir_code: int,
    racers: list,
    hot_players: dict | None = None,
) -> tuple[list, dict]:
    """4ステップのルールベース重み付けでAI予測確率を生成する。

    ステップ1: 軸の決定（1号艇の精査）
    ステップ2: 対抗（2番手）の探索
    ステップ3: 風と展示による展開補正（波乱の警戒）
    ステップ4: レースタイプ判定（buy_recommendations と連動）

    Returns:
        probs: 6艇分の予測確率（正規化済み）
        analysis: レースタイプ・軸・対抗・一番時計などの分析メタデータ
    """
    probs = base_probs(stadium)[:]
    analysis: dict = {
        "axis_type": "normal",       # "strong" | "normal" | "upset_risk"
        "race_type": "normal",       # "favorite" | "normal" | "upset"
        "counter_boat": None,        # 2番手筆頭の艇番（1-indexed）
        "counter_boats": [],         # 対抗候補一覧（1-indexed）
        "fastest_tenji_boat": None,  # 展示一番時計の艇番（1-indexed）
        "is_tailwind": False,
        "is_strong_headwind": False,
        "hot_boats": [],             # 今節好調選手の艇番一覧（1-indexed）
    }

    is_tailwind = wind_dir_code in _TAIL_WIND_CODES
    is_headwind = wind_dir_code in _HEAD_WIND_CODES
    is_strong_headwind = is_headwind and wind_speed >= 5.0
    analysis["is_tailwind"] = is_tailwind
    analysis["is_strong_headwind"] = is_strong_headwind

    if "江戸川" in stadium and wind_speed > 4.0:
        probs = [0.30, 0.15, 0.15, 0.15, 0.13, 0.12]

    if not racers:
        return probs, analysis

    # ===== 基本スコア（当地勝率・全国勝率・モーター・級別）=====
    scores = []
    for r in racers:
        s = (
            r.get("当地勝率", 0.0) * 0.5
            + r.get("全国勝率", 0.0) * 0.3
            + (r.get("モーター2連率", 0.0) / 100.0) * 0.1
            + GRADE_SCORE.get(r.get("級別", ""), 0.5) * 0.1
        )
        scores.append(max(s, 0.01))

    while len(scores) < 6:
        scores.append(0.01)

    total_score = sum(scores)
    score_probs = [s / total_score for s in scores]
    probs = [0.5 * p + 0.5 * sp for p, sp in zip(probs, score_probs)]

    # ===== ステップ1: 軸の決定（1号艇の精査）=====
    boat1 = racers[0]
    boat1_local = boat1.get("当地勝率", 0.0)
    boat1_nat = boat1.get("全国勝率", 0.0)
    boat1_rate = boat1_local if boat1_local > 0 else boat1_nat
    boat1_st = boat1.get("平均ST", 0.18)

    others_nat = [r.get("全国勝率", 0.0) for r in racers[1:]]
    avg_others_nat = sum(others_nat) / max(len(others_nat), 1)

    # 絶対的な本命: 勝率5.5以上かつ追い風
    if boat1_rate >= 5.5 and is_tailwind:
        axis_type = "strong"
        probs[0] *= 1.25
    # 通常本命: 勝率5.5以上かつ強向かい風でない
    elif boat1_rate >= 5.5 and not is_strong_headwind:
        axis_type = "strong"
        probs[0] *= 1.18
    # イン飛び警戒: 勝率4.5未満かつSTが遅い（両条件を同時に要求）
    elif boat1_rate < 4.5 and boat1_st > 0.18:
        axis_type = "upset_risk"
        probs[0] *= 0.78
        for i in range(1, 6):
            probs[i] *= 1.04
    # 波乱含み: 他艇平均を下回る勝率かつ5.0未満
    elif boat1_rate < avg_others_nat and boat1_rate < 5.0:
        axis_type = "upset_risk"
        probs[0] *= 0.85
    else:
        axis_type = "normal"

    analysis["axis_type"] = axis_type

    # ===== ステップ2: 対抗（2番手）の探索 =====
    # 2〜4号艇（センター勢）: 全国勝率最高 + モーター2連率40%以上を別々に評価
    center_idxs = list(range(1, min(4, len(racers))))
    counter_boats: list = []

    if center_idxs:
        # 全国勝率が最も高い艇を2着筆頭候補に
        best_rate_idx = max(center_idxs, key=lambda i: racers[i].get("全国勝率", 0.0))
        if racers[best_rate_idx].get("全国勝率", 0.0) > 0:
            probs[best_rate_idx] *= 1.12
            counter_boats.append(best_rate_idx + 1)

        # モーター2連率40%以上の艇をさらに加点（2着・3着候補として評価）
        for i in center_idxs:
            if racers[i].get("モーター2連率", 0.0) >= 40.0:
                probs[i] *= 1.08
                if (i + 1) not in counter_boats:
                    counter_boats.append(i + 1)

    analysis["counter_boat"] = counter_boats[0] if counter_boats else None
    analysis["counter_boats"] = counter_boats

    # ===== ステップ3: 風と展示による展開補正（波乱の警戒）=====
    # 気象条件の判定: 風速5m/s以上かつ向かい風
    if is_strong_headwind:
        probs[0] *= 0.88  # インをさらに弱体化
        for i in range(2, 6):  # 3〜6号艇（ダッシュ勢）を強化
            probs[i] *= 1.08
    elif is_tailwind and wind_speed >= 3.0:
        probs[0] *= 1.06

    # 展示タイムの判定: 一番時計を特定し z スコアで全艇を補正
    tenji_times = [
        racers[i].get("展示タイム", 0.0) if i < len(racers) else 0.0
        for i in range(6)
    ]
    valid_times = [t for t in tenji_times if t > 0]

    if valid_times:
        mean_t = sum(valid_times) / len(valid_times)
        std_t = (sum((t - mean_t) ** 2 for t in valid_times) / len(valid_times)) ** 0.5

        for i, t in enumerate(tenji_times):
            if t > 0 and std_t > 0:
                z = (mean_t - t) / std_t  # 小さい（速い）ほど正の補正
                probs[i] = max(probs[i] * (1.0 + z * 0.09), 0.001)

        fastest_idx = min(
            (i for i, t in enumerate(tenji_times) if t > 0),
            key=lambda i: tenji_times[i],
        )
        analysis["fastest_tenji_boat"] = fastest_idx + 1

        # ダッシュ枠（3〜6号艇）の一番時計 × 向かい風 → 頭の可能性を強く加点
        if fastest_idx >= 2 and is_headwind:
            probs[fastest_idx] *= 1.15

    # チルト補正
    for i in range(min(len(racers), 6)):
        tilt = racers[i].get("チルト", 0.0)
        boat = i + 1
        if tilt > 0:
            factor = 0.005 if boat >= 4 else -0.003
            probs[i] = max(probs[i] + factor * tilt, 0.001)
        elif tilt < 0:
            factor = 0.005 if boat <= 3 else -0.003
            probs[i] = max(probs[i] + factor * abs(tilt), 0.001)

    # 江戸川特殊処理（強風時は外艇全体を底上げ）
    if "江戸川" in stadium and wind_speed > 4.0:
        for i in range(1, 6):
            probs[i] *= 1.05

    total = sum(probs)
    probs = [p / total for p in probs]

    # ===== 今節好調選手補正 =====
    hot_in_race: list = []
    if hot_players:
        for i, r in enumerate(racers):
            name = r.get("選手名", "").strip()
            boost = hot_players.get(name, 1.0)
            if boost > 1.0:
                probs[i] *= boost
                hot_in_race.append(i + 1)
        if hot_in_race:
            total = sum(probs)
            probs = [p / total for p in probs]
    analysis["hot_boats"] = hot_in_race

    # ===== ステップ4: レースタイプ判定 =====
    if axis_type == "strong" and not is_strong_headwind:
        race_type = "favorite"
    elif axis_type == "upset_risk" or is_strong_headwind:
        race_type = "upset"
    else:
        race_type = "normal"

    analysis["race_type"] = race_type
    return probs, analysis


def recommend_bets(probs: list, analysis: dict) -> list:
    """ステップ4: レースタイプに応じた買い目絞り込み（ガミり防止）

    本命レース: 4〜8点（1号艇軸固定フォーメーション）
    通常レース: 8〜12点（2軸フォーメーション）
    穴レース:  12〜20点（広フォーメーション）
    """
    race_type = analysis.get("race_type", "normal")
    fastest_tenji = analysis.get("fastest_tenji_boat")
    is_strong_headwind = analysis.get("is_strong_headwind", False)

    ranking = sorted(range(6), key=lambda i: -probs[i])
    boats = [r + 1 for r in ranking]  # 確率高い順の艇番（1-indexed）

    if race_type == "favorite":
        # 本命レース: 1号艇を1着に固定、確率上位選手で2着・3着を組む（4〜8点）
        axis = 1
        others = [b for b in boats if b != axis]
        seconds = others[:2]
        thirds = others[:3]
        bets = _gen_formations([axis], seconds, thirds)

        if len(bets) < 4:  # 最低4点保証
            seconds = others[:3]
            bets = _gen_formations([axis], seconds, thirds)[:8]
        else:
            bets = bets[:8]

    elif race_type == "upset":
        # 穴レース: 上位3艇が1着候補、上位5艇まで3着候補に広げる（12〜20点）
        firsts = boats[:3]

        # 強向かい風 × ダッシュ枠一番時計 → その艇を1着筆頭に追加
        if fastest_tenji and fastest_tenji >= 3 and is_strong_headwind:
            firsts_set = set(firsts)
            firsts_set.add(fastest_tenji)
            firsts = [fastest_tenji] + [b for b in boats if b in firsts_set and b != fastest_tenji][:2]

        seconds = boats[:4]
        thirds = boats[:5]
        bets = _gen_formations(firsts, seconds, thirds)

        if len(bets) < 12:  # 最低12点保証
            firsts = boats[:4]
            bets = _gen_formations(firsts, seconds, thirds)[:20]
        else:
            bets = bets[:20]

    else:
        # 通常レース: 2軸フォーメーション（8〜12点）
        axis = boats[0]
        others = [b for b in boats if b != axis]
        seconds = others[:3]
        thirds = others[:4]
        bets = _gen_formations([axis], seconds, thirds)

        # 2番手を1着に立てた追加フォーメーション
        if len(boats) >= 2:
            axis2 = boats[1]
            s2 = [b for b in boats[:4] if b != axis2][:3]
            t2 = [b for b in boats[:5] if b != axis2][:4]
            extra = _gen_formations([axis2], s2, t2)
            bets = list(dict.fromkeys(bets + extra))

        # 8点に満たない場合は広げて補完、12点超えたら絞る
        if len(bets) < 8:
            add = _gen_formations(boats[:2], boats[:4], boats[:5])
            bets = list(dict.fromkeys(bets + add))[:12]
        else:
            bets = bets[:12]

    return bets if bets else [f"{boats[0]}-{boats[1]}-{boats[2]}"]


def _compress_bets(bets: list) -> list[dict]:
    """買い目リストを (1着-2着) でグループ化しフォーメーション表記に圧縮する。

    例: [1-2-3, 1-2-4, 1-2-5] → {"label": "1-2-3/4/5", "count": 3}
        3着が残り全艇      → {"label": "1-3-全",     "count": 4}
    """
    parsed = []
    for b in bets:
        parts = b.split("-")
        if len(parts) == 3:
            try:
                parsed.append((int(parts[0]), int(parts[1]), int(parts[2])))
            except ValueError:
                pass

    groups: dict = {}
    for a, b, c in parsed:
        key = (a, b)
        if key not in groups:
            groups[key] = []
        groups[key].append(c)

    result = []
    for (a, b), thirds in sorted(groups.items()):
        remaining = set(range(1, 7)) - {a, b}
        if set(thirds) == remaining:
            label = f"{a}-{b}-全"
        elif len(thirds) >= 2:
            label = f"{a}-{b}-" + "/".join(str(c) for c in sorted(thirds))
        else:
            label = f"{a}-{b}-{thirds[0]}"
        result.append({"label": label, "count": len(thirds)})

    return result


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
    "setsunai_results": {},
    "hot_players": {},
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

    # 今節成績を集計（初日〜当日の直前レースまで全件）
    if not err_race and st.session_state.racers:
        with st.spinner("今節成績を集計中（過去レース解析）..."):
            setsunai = fetch_setsunai_results(jcd, date_str, int(race_num))
            st.session_state.setsunai_results = setsunai
            st.session_state.hot_players = identify_hot_players(setsunai)

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

_EMPTY_ANALYSIS: dict = {
    "axis_type": "normal", "race_type": "normal",
    "counter_boat": None, "fastest_tenji_boat": None,
    "is_tailwind": False, "is_strong_headwind": False,
    "hot_boats": [],
}

try:
    probabilities, analysis = generate_prediction(
        stadium, wind_speed, wind_dir_code, merged_racers,
        st.session_state.get("hot_players", {}),
    )
    if len(probabilities) != 6:
        raise ValueError(f"予測値が6艇分ありません（{len(probabilities)}艇分）")
except Exception as e:
    st.error(f"予測の生成中にエラーが発生しました: {e}")
    probabilities = [1 / 6] * 6
    analysis = _EMPTY_ANALYSIS.copy()

# --- レースタイプバナー ---
race_type = analysis.get("race_type", "normal")
axis_type = analysis.get("axis_type", "normal")
if race_type == "favorite":
    st.success("🎯 **本命レース** — 1号艇の軸が堅い。点数を絞って高回収を狙う。")
elif race_type == "upset":
    st.warning("⚡ **穴レース** — 波乱の可能性あり。フォーメーションを広げて高配当を狙う。")
else:
    st.info("📊 **通常レース** — 標準フォーメーションで安定を狙う。")

# 予測を2列×3行で表示（公式枠色カード）
BOAT_FRAME_COLORS = [
    {"bg": "#FFFFFF", "text": "#333333", "border": "#CCCCCC"},  # 1号艇：白
    {"bg": "#1A1A1A", "text": "#FFFFFF", "border": "#555555"},  # 2号艇：黒
    {"bg": "#E8001B", "text": "#FFFFFF", "border": "#C0001A"},  # 3号艇：赤
    {"bg": "#0047AB", "text": "#FFFFFF", "border": "#003380"},  # 4号艇：青
    {"bg": "#FFD700", "text": "#333333", "border": "#CCA800"},  # 5号艇：黄
    {"bg": "#007A33", "text": "#FFFFFF", "border": "#005522"},  # 6号艇：緑
]
fastest_tenji = analysis.get("fastest_tenji_boat")
counter_boat = analysis.get("counter_boat")
counter_boats_list = analysis.get("counter_boats", [])
axis_type = analysis.get("axis_type", "normal")
hot_boats_list = analysis.get("hot_boats", [])

for row in range(3):
    left, right = st.columns(2)
    for col, idx in zip([left, right], [row * 2, row * 2 + 1]):
        r = merged_racers[idx] if idx < len(merged_racers) else {}
        name = r.get("選手名", f"{idx + 1}号艇")
        grade = r.get("級別", "")
        tenji = r.get("展示タイム", 0.0)
        tilt = r.get("チルト", 0.0)
        avg_st = r.get("平均ST", 0.0)
        boat_no = idx + 1
        color = BOAT_FRAME_COLORS[idx]

        tags = []
        if tenji > 0:
            tags.append(f"展示 {tenji:.2f}")
        if tilt != 0.0:
            tags.append(f"チルト {tilt:+.1f}")
        if avg_st > 0:
            tags.append(f"ST {avg_st:.2f}")
        if fastest_tenji == boat_no:
            tags.append("⚡ 一番時計")
        if boat_no in counter_boats_list:
            tags.append("🥈 対抗")
        if axis_type == "strong" and boat_no == 1:
            tags.append("⚓ 本命軸")
        elif axis_type == "upset_risk" and boat_no == 1:
            tags.append("⚠️ 注意")
        if boat_no in hot_boats_list:
            tags.append("🔥 今節好調")

        sub = " ／ ".join(tags)
        prob_str = f"{probabilities[idx] * 100:.1f}"

        card_html = f"""<div style="
    background-color: {color['bg']};
    color: {color['text']};
    border: 2px solid {color['border']};
    border-radius: 10px;
    padding: 16px 12px;
    margin-bottom: 8px;
    text-align: center;
    box-shadow: 0 2px 6px rgba(0,0,0,0.15);
">
    <div style="font-size: 0.85em; font-weight: 600; margin-bottom: 4px;">
        {boat_no}号艇 &nbsp; {name} &nbsp; {grade}
    </div>
    <div style="font-size: 2.2em; font-weight: bold; line-height: 1.1; margin: 4px 0;">
        {prob_str}%
    </div>
    <div style="font-size: 0.75em; margin-top: 6px; opacity: 0.85;">
        {sub if sub else "&nbsp;"}
    </div>
</div>"""
        with col:
            st.markdown(card_html, unsafe_allow_html=True)

# --- 分析サマリー ---
counter_boats_display = analysis.get("counter_boats", [])
if counter_boats_display or fastest_tenji:
    ac1, ac2 = st.columns(2)
    with ac1:
        if counter_boats_display:
            boats_str = "・".join([f"{b}号艇" for b in counter_boats_display])
            ac1.info(f"🥈 対抗（2番手候補）: **{boats_str}**")
    with ac2:
        if fastest_tenji:
            if fastest_tenji >= 3 and analysis.get("is_strong_headwind"):
                ac2.warning(f"⚡ 展示一番時計: **{fastest_tenji}号艇**（向かい風で頭の可能性）")
            else:
                ac2.info(f"⏱️ 展示一番時計: **{fastest_tenji}号艇**")

st.divider()

# --- おすすめ買い目 ---
st.subheader("おすすめ買い目（3連単）")
bets = recommend_bets(probabilities, analysis)
bet_count = len(bets)

if race_type == "favorite":
    _bet_label, _bet_range = "🎯 本命レース", "4〜8点"
    _chip_bg, _chip_fg = "#2d8a4e", "#ffffff"
elif race_type == "upset":
    _bet_label, _bet_range = "⚡ 穴レース", "12〜20点"
    _chip_bg, _chip_fg = "#c0392b", "#ffffff"
else:
    _bet_label, _bet_range = "📊 通常レース", "8〜12点"
    _chip_bg, _chip_fg = "#1a5fa8", "#ffffff"

st.caption(f"{_bet_label} — 合計 **{bet_count}点**（目安: {_bet_range}）")

formations = _compress_bets(bets)
_chip_items = "".join(
    f'<span style="display:inline-flex;align-items:center;gap:4px;'
    f"background:{_chip_bg};color:{_chip_fg};"
    f'border-radius:999px;padding:6px 16px;'
    f'font-size:1.05em;font-weight:bold;white-space:nowrap;'
    f'box-shadow:0 1px 4px rgba(0,0,0,0.25);">'
    f'{f["label"]}'
    + (
        f'<span style="font-size:0.70em;opacity:0.80;margin-left:3px;">({f["count"]}点)</span>'
        if f["count"] > 1
        else ""
    )
    + "</span>"
    for f in formations
)
st.markdown(
    f'<div style="display:flex;flex-wrap:wrap;gap:10px;padding:8px 0;">'
    f"{_chip_items}</div>"
    f'<div style="font-size:0.82em;color:#888;margin-top:6px;">合計点数：{bet_count}点</div>',
    unsafe_allow_html=True,
)

# --- 出走表（折りたたみ）---
if merged_racers:
    st.divider()
    with st.expander("出走表・直前情報（詳細）"):
        df = pd.DataFrame(merged_racers)
        display_cols = [
            "艇番", "選手名", "級別",
            "展示タイム", "チルト", "平均ST",
            "当地勝率", "全国勝率", "モーター2連率",
        ]
        df_display = df[[c for c in display_cols if c in df.columns]].set_index("艇番")
        st.dataframe(df_display, use_container_width=True)
        if st.session_state.fetch_url:
            st.caption(f"出走表URL: {st.session_state.fetch_url}")

# --- 今節成績（折りたたみ）---
_setsunai = st.session_state.get("setsunai_results", {})
if _setsunai:
    st.divider()
    with st.expander("今節成績（節間パフォーマンス）"):
        _hot_names = set(st.session_state.get("hot_players", {}).keys())
        _rows = []
        for _name, _stats in _setsunai.items():
            _runs = _stats["出走"]
            _top2 = _stats["1着"] + _stats["2着"]
            _rows.append({
                "選手名": f"🔥 {_name}" if _name in _hot_names else _name,
                "出走": _runs,
                "1着": _stats["1着"],
                "2着": _stats["2着"],
                "トップ2率": f"{_top2 / _runs * 100:.0f}%" if _runs > 0 else "-",
            })
        if _rows:
            _df_sets = pd.DataFrame(_rows).sort_values("1着", ascending=False)
            st.dataframe(_df_sets.set_index("選手名"), use_container_width=True)
            st.caption("🔥 = 今節好調選手（AI予測に補正済み）")
else:
    st.info("「データ取得」を押すと boatrace.jp から出走表を取得し、予測精度が向上します。")
