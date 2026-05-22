import streamlit as st
import pandas as pd
import numpy as np
import os
import math

# 페이지 구성 설정 - 타이틀 및 레이아웃
st.set_page_config(page_title="KRA AI 경마 예측", layout="wide", initial_sidebar_state="collapsed")

# 1. Mock 데이터 정의 (경주 정보 및 출전마 스펙)
RACES = [
    {
        "id": "2024-11-03-1R",
        "date": "2024.11.03",
        "no": "1R",
        "dist": "1200M",
        "track": "양호 (6%)",
        "weather": "맑음",
        "horses": [
            { "name": "천하무적", "gate": 1, "age": 4, "weight": 504, "weight_diff": 2, "burd_wgt": 54, "jockey": "송재철", "trainer": "서인석", "odds": 3.1, "top3_rate_last_5": 0.80, "avg_rank_last_3": 1.3, "avg_gap_last_3": 0.2, "jockey_rate": 0.38, "trainer_rate": 0.31, "days_rest": 28, "race_count": 22, "is_debut": 0 },
            { "name": "스피드킹", "gate": 2, "age": 3, "weight": 490, "weight_diff": 3, "burd_wgt": 53, "jockey": "임다빈", "trainer": "박재우", "odds": 5.5, "top3_rate_last_5": 0.60, "avg_rank_last_3": 2.0, "avg_gap_last_3": 0.4, "jockey_rate": 0.32, "trainer_rate": 0.28, "days_rest": 30, "race_count": 12, "is_debut": 0 },
            { "name": "황금마차", "gate": 7, "age": 5, "weight": 512, "weight_diff": 1, "burd_wgt": 55, "jockey": "장추열", "trainer": "강환민", "odds": 32.0, "top3_rate_last_5": 0.50, "avg_rank_last_3": 3.5, "avg_gap_last_3": 0.8, "jockey_rate": 0.25, "trainer_rate": 0.24, "days_rest": 35, "race_count": 15, "is_debut": 0 },
            { "name": "번개돌이", "gate": 9, "age": 3, "weight": 475, "weight_diff": 0, "burd_wgt": 52, "jockey": "김아현", "trainer": "박윤규", "odds": 15.0, "top3_rate_last_5": 0.00, "avg_rank_last_3": None, "avg_gap_last_3": None, "jockey_rate": 0.15, "trainer_rate": 0.18, "days_rest": None, "race_count": 0, "is_debut": 1 },
            { "name": "쾌속질주", "gate": 4, "age": 4, "weight": 498, "weight_diff": -6, "burd_wgt": 54, "jockey": "유승완", "trainer": "박병일", "odds": 8.4, "top3_rate_last_5": 0.40, "avg_rank_last_3": 4.2, "avg_gap_last_3": 1.1, "jockey_rate": 0.22, "trainer_rate": 0.20, "days_rest": 65, "race_count": 18, "is_debut": 0 },
            { "name": "야호자이언트", "gate": 6, "age": 5, "weight": 520, "weight_diff": 0, "burd_wgt": 56, "jockey": "김태희", "trainer": "정호익", "odds": 12.0, "top3_rate_last_5": 0.30, "avg_rank_last_3": 5.1, "avg_gap_last_3": 1.5, "jockey_rate": 0.18, "trainer_rate": 0.15, "days_rest": 45, "race_count": 25, "is_debut": 0 },
            { "name": "블랙스톰", "gate": 10, "age": 4, "weight": 482, "weight_diff": 4, "burd_wgt": 54, "jockey": "다비드", "trainer": "안병기", "odds": 24.5, "top3_rate_last_5": 0.45, "avg_rank_last_3": 3.9, "avg_gap_last_3": 0.9, "jockey_rate": 0.28, "trainer_rate": 0.22, "days_rest": 25, "race_count": 10, "is_debut": 0 },
            { "name": "새벽바람", "gate": 5, "age": 3, "weight": 465, "weight_diff": 2, "burd_wgt": 51, "jockey": "이혁", "trainer": "홍대유", "odds": 4.2, "top3_rate_last_5": 0.70, "avg_rank_last_3": 2.1, "avg_gap_last_3": 0.5, "jockey_rate": 0.30, "trainer_rate": 0.29, "days_rest": 24, "race_count": 8, "is_debut": 0 }
        ]
    },
    {
        "id": "2024-11-03-5R",
        "date": "2024.11.03",
        "no": "5R",
        "dist": "1800M",
        "track": "다습 (14%)",
        "weather": "흐림",
        "horses": [
            { "name": "백록담", "gate": 3, "age": 4, "weight": 510, "weight_diff": 3, "burd_wgt": 55, "jockey": "페로비치", "trainer": "김대근", "odds": 2.8, "top3_rate_last_5": 0.90, "avg_rank_last_3": 1.2, "avg_gap_last_3": 0.1, "jockey_rate": 0.42, "trainer_rate": 0.35, "days_rest": 28, "race_count": 14, "is_debut": 0 },
            { "name": "한라산", "gate": 8, "age": 5, "weight": 525, "weight_diff": 1, "burd_wgt": 56, "jockey": "문세영", "trainer": "김동균", "odds": 4.5, "top3_rate_last_5": 0.75, "avg_rank_last_3": 2.2, "avg_gap_last_3": 0.5, "jockey_rate": 0.45, "trainer_rate": 0.33, "days_rest": 35, "race_count": 30, "is_debut": 0 },
            { "name": "설악바람", "gate": 5, "age": 3, "weight": 480, "weight_diff": -1, "burd_wgt": 52, "jockey": "조한우", "trainer": "최영주", "odds": 45.0, "top3_rate_last_5": 0.52, "avg_rank_last_3": 3.8, "avg_gap_last_3": 1.0, "jockey_rate": 0.22, "trainer_rate": 0.20, "days_rest": 28, "race_count": 11, "is_debut": 0 },
            { "name": "오대산", "gate": 1, "age": 4, "weight": 495, "weight_diff": 4, "burd_wgt": 54, "jockey": "이현종", "trainer": "지용철", "odds": 18.0, "top3_rate_last_5": 0.40, "avg_rank_last_3": 4.5, "avg_gap_last_3": 1.3, "jockey_rate": 0.20, "trainer_rate": 0.18, "days_rest": 42, "race_count": 16, "is_debut": 0 },
            { "name": "태백산맥", "gate": 11, "age": 3, "weight": 488, "weight_diff": 0, "burd_wgt": 53, "jockey": "빅투아르", "trainer": "배대선", "odds": 9.2, "top3_rate_last_5": 0.55, "avg_rank_last_3": 3.1, "avg_gap_last_3": 0.7, "jockey_rate": 0.31, "trainer_rate": 0.27, "days_rest": 32, "race_count": 7, "is_debut": 0 },
            { "name": "지리산천왕", "gate": 4, "age": 5, "weight": 515, "weight_diff": -7, "burd_wgt": 56, "jockey": "송재철", "trainer": "유재길", "odds": 15.0, "top3_rate_last_5": 0.35, "avg_rank_last_3": 4.9, "avg_gap_last_3": 1.4, "jockey_rate": 0.24, "trainer_rate": 0.22, "days_rest": 75, "race_count": 28, "is_debut": 0 },
            { "name": "금강산", "gate": 9, "age": 3, "weight": 472, "weight_diff": 1, "burd_wgt": 52, "jockey": "정정희", "trainer": "곽영효", "odds": 28.0, "top3_rate_last_5": 0.00, "avg_rank_last_3": None, "avg_gap_last_3": None, "jockey_rate": 0.18, "trainer_rate": 0.15, "days_rest": None, "race_count": 0, "is_debut": 1 },
            { "name": "무등산", "gate": 2, "age": 4, "weight": 502, "weight_diff": 2, "burd_wgt": 54, "jockey": "김효정", "trainer": "박흥진", "odds": 7.0, "top3_rate_last_5": 0.65, "avg_rank_last_3": 2.8, "avg_gap_last_3": 0.6, "jockey_rate": 0.26, "trainer_rate": 0.25, "days_rest": 25, "race_count": 12, "is_debut": 0 }
        ]
    },
    {
        "id": "2024-11-03-9R",
        "date": "2024.11.03",
        "no": "9R",
        "dist": "1400M",
        "track": "포화 (18%)",
        "weather": "맑음",
        "horses": [
            { "name": "일념통천", "gate": 5, "age": 4, "weight": 508, "weight_diff": 3, "burd_wgt": 54, "jockey": "유승완", "trainer": "서인석", "odds": 2.5, "top3_rate_last_5": 0.85, "avg_rank_last_3": 1.4, "avg_gap_last_3": 0.2, "jockey_rate": 0.38, "trainer_rate": 0.31, "days_rest": 28, "race_count": 19, "is_debut": 0 },
            { "name": "천리마", "gate": 2, "age": 5, "weight": 518, "weight_diff": 2, "burd_wgt": 56, "jockey": "장추열", "trainer": "강환민", "odds": 5.0, "top3_rate_last_5": 0.70, "avg_rank_last_3": 2.3, "avg_gap_last_3": 0.5, "jockey_rate": 0.35, "trainer_rate": 0.30, "days_rest": 30, "race_count": 24, "is_debut": 0 },
            { "name": "적토마", "gate": 8, "age": 3, "weight": 488, "weight_diff": 1, "burd_wgt": 52, "jockey": "김아현", "trainer": "박윤규", "odds": 35.0, "top3_rate_last_5": 0.52, "avg_rank_last_3": 3.2, "avg_gap_last_3": 0.6, "jockey_rate": 0.28, "trainer_rate": 0.26, "days_rest": 35, "race_count": 10, "is_debut": 0 },
            { "name": "우뢰매", "gate": 12, "age": 4, "weight": 498, "weight_diff": -8, "burd_wgt": 54, "jockey": "김태희", "trainer": "정호익", "odds": 14.0, "top3_rate_last_5": 0.45, "avg_rank_last_3": 4.0, "avg_gap_last_3": 0.9, "jockey_rate": 0.22, "trainer_rate": 0.21, "days_rest": 70, "race_count": 15, "is_debut": 0 },
            { "name": "비호", "gate": 6, "age": 3, "weight": 482, "weight_diff": 4, "burd_wgt": 53, "jockey": "임다빈", "trainer": "박재우", "odds": 8.8, "top3_rate_last_5": 0.60, "avg_rank_last_3": 2.9, "avg_gap_last_3": 0.6, "jockey_rate": 0.32, "trainer_rate": 0.28, "days_rest": 32, "race_count": 9, "is_debut": 0 },
            { "name": "야생마", "gate": 3, "age": 4, "weight": 492, "weight_diff": 10, "burd_wgt": 54, "jockey": "이혁", "trainer": "홍대유", "odds": 18.0, "top3_rate_last_5": 0.40, "avg_rank_last_3": 4.8, "avg_gap_last_3": 1.3, "jockey_rate": 0.20, "trainer_rate": 0.18, "days_rest": 45, "race_count": 13, "is_debut": 0 },
            { "name": "태풍선풍", "gate": 10, "age": 3, "weight": 470, "weight_diff": 2, "burd_wgt": 51, "jockey": "이현종", "trainer": "지용철", "odds": 40.0, "top3_rate_last_5": 0.35, "avg_rank_last_3": 5.0, "avg_gap_last_3": 1.4, "jockey_rate": 0.24, "trainer_rate": 0.20, "days_rest": 28, "race_count": 6, "is_debut": 0 },
            { "name": "화룡점정", "gate": 1, "age": 5, "weight": 505, "weight_diff": 0, "burd_wgt": 55, "jockey": "페로비치", "trainer": "김동균", "odds": 6.5, "top3_rate_last_5": 0.68, "avg_rank_last_3": 2.7, "avg_gap_last_3": 0.5, "jockey_rate": 0.30, "trainer_rate": 0.29, "days_rest": 25, "race_count": 22, "is_debut": 0 }
        ]
    }
]

# 2. 모델 상대 피처 연산 및 분류 기준 매핑 함수
def calcRelFeatures(horses):
    avg = lambda arr: sum(arr) / len(arr) if arr else 0
    
    avgTop3 = avg([h["top3_rate_last_5"] for h in horses])
    avgJockey = avg([h["jockey_rate"] for h in horses])
    avgTrainer = avg([h["trainer_rate"] for h in horses])
    
    synergies = [h["jockey_rate"] * h["trainer_rate"] for h in horses]
    avgSynergy = avg(synergies)
    
    processed = []
    for h in horses:
        synergy = h["jockey_rate"] * h["trainer_rate"]
        is_peak = 1 if (h["weight_diff"] >= 0 and h["weight_diff"] <= 5 and 
                        h["days_rest"] is not None and h["days_rest"] >= 21 and h["days_rest"] <= 42) else 0
        
        # 상대 피처
        rel_top3 = h["top3_rate_last_5"] - avgTop3
        rel_jockey = h["jockey_rate"] - avgJockey
        rel_synergy = synergy - avgSynergy
        
        # 피크폼 지수
        rank_fill = h["avg_rank_last_3"] if h["avg_rank_last_3"] is not None else 7
        gap_fill = h["avg_gap_last_3"] if h["avg_gap_last_3"] is not None else 3.0
        peak_form_index = 1 / ((rank_fill * gap_fill) + 1)
        
        # 다크호스 점수
        dark_horse_score = (h["odds"] / 10.0) * h["top3_rate_last_5"]
        
        # LightGBM 근사 점수
        score = 0
        score += rel_top3 * 2.5
        score += rel_jockey * 2.0
        score += rel_synergy * 1.8
        score += peak_form_index * 1.5
        score += is_peak * 0.4
        score += dark_horse_score * 0.3
        score += (-0.4 if h["is_debut"] else 0.1)
        
        prob = 1 / (1 + math.exp(-score))
        
        # EV 계산
        ev = (prob * h["odds"]) - (1 - prob)
        
        # 방향별 추천 여부
        recallFlag = prob >= 0.30
        precisionFlag = prob >= 0.70
        evFlag = ev > 0
        
        # 켈리 기준
        b = h["odds"] - 1
        kelly = max(0.0, (b * prob - (1 - prob)) / b) if b > 0 else 0.0
        
        h_copied = h.copy()
        h_copied.update({
            "synergy": synergy,
            "is_peak": is_peak,
            "peak_form_index": peak_form_index,
            "dark_horse_score": dark_horse_score,
            "prob": prob * 100, # %단위
            "ev": ev,
            "recallFlag": recallFlag,
            "precisionFlag": precisionFlag,
            "evFlag": evFlag,
            "kelly": kelly
        })
        processed.append(h_copied)
        
    # 확률 기준 내림차순 정렬
    processed.sort(key=lambda x: x["prob"], reverse=True)
    return processed

# UI 서브 컴포넌트 헬퍼 함수
def get_track_badge_html(track):
    if "양호" in track:
        badge_style = "background-color: rgba(59, 130, 246, 0.2); color: #60A5FA; border: 1px solid rgba(59, 130, 246, 0.4);"
    elif "다습" in track:
        badge_style = "background-color: rgba(245, 158, 11, 0.2); color: #FBBF24; border: 1px solid rgba(245, 158, 11, 0.4);"
    elif "포화" in track or "불량" in track:
        badge_style = "background-color: rgba(239, 68, 68, 0.2); color: #F87171; border: 1px solid rgba(239, 68, 68, 0.4);"
    else:
        badge_style = "background-color: rgba(16, 185, 129, 0.2); color: #34D399; border: 1px solid rgba(16, 185, 129, 0.4);"

    return f"""<span class="badge" style="padding: 6px 12px; font-size: 12px; font-weight: bold; border-radius: 9999px; {badge_style}">
      주로: {track}
    </span>"""

def get_prob_bar_html(value, width_px=96):
    if value >= 50:
        color_class = "#1D9E75"
    elif value >= 30:
        color_class = "#EF9F27"
    else:
        color_class = "#B4B2A9"

    return f"""<div style="display: flex; align-items: center; gap: 8px; justify-content: flex-end;">
      <div class="prob-bar-bg" style="width: {width_px}px; background-color: #060B13; border: 1px solid rgba(65,90,119,0.3); height: 10px; border-radius: 9999px; overflow: hidden; display: inline-block;">
        <div class="prob-bar-fill" style="height: 100%; border-radius: 9999px; background-color: {color_class}; width: {min(100.0, value)}%;"></div>
      </div>
      <span style="font-size: 12px; font-weight: 600; font-family: monospace; width: 45px; text-align: right; color: #F8FAFC;">{value:.1f}%</span>
    </div>"""

def get_condition_light_html(horse):
    if horse["is_debut"] == 1:
        return """<div style="display: inline-flex; align-items: center; gap: 6px; position: relative;" title="과거 데이터 없음 — 신예마 특성 참고">
          <span class="light" style="width: 12px; height: 12px; border-radius: 9999px; display: inline-block; background-color: #B4B2A9;"></span>
          <span style="font-size: 12px; color: #94A3B8; font-weight: 500;">첫출전</span>
        </div>"""

    cond1 = horse["weight_diff"] >= 0 and horse["weight_diff"] <= 5
    cond2 = horse["days_rest"] is not None and horse["days_rest"] >= 21 and horse["days_rest"] <= 42
    warningCond = horse["weight_diff"] <= -5 or (horse["days_rest"] is not None and horse["days_rest"] >= 60)

    if cond1 and cond2:
        color = "#1D9E75"
        label = "최적"
    elif warningCond:
        color = "#EF4444"
        label = "주의"
    else:
        color = "#EF9F27"
        label = "보통"

    return f"""<div style="display: inline-flex; align-items: center; gap: 6px;">
      <span class="light" style="width: 12px; height: 12px; border-radius: 9999px; display: inline-block; background-color: {color};"></span>
      <span style="font-size: 12px; font-weight: 500; color: #F8FAFC;">{label}</span>
    </div>"""

def get_gate_tag_html(gate):
    if gate >= 1 and gate <= 4:
        badge_style = "background-color: rgba(29, 158, 117, 0.2); color: #2CC091; border: 1px solid rgba(29, 158, 117, 0.4);"
        label = "내측"
    elif gate >= 5 and gate <= 8:
        badge_style = "background-color: rgba(239, 159, 39, 0.2); color: #EF9F27; border: 1px solid rgba(239, 159, 39, 0.4);"
        label = "중간"
    else:
        badge_style = "background-color: rgba(69, 10, 10, 0.6); color: #F87171; border: 1px solid rgba(153, 27, 27, 0.4);"
        label = "외측"

    return f"""<div style="display: inline-flex; align-items: center; gap: 6px;">
      <span style="font-weight: bold; font-size: 13px; background-color: #060B13; color: #F8D675; width: 24px; height: 24px; display: inline-flex; align-items: center; justify-content: center; rounded-sm; border: 1px solid #1B263B; border-radius: 4px;">{gate}</span>
      <span style="font-size: 10px; padding: 2px 6px; border-radius: 4px; {badge_style}">{label}</span>
    </div>"""

def get_ev_badge_html(value):
    if value > 0:
        badge_style = "background-color: rgba(29, 158, 117, 0.15); color: #2CC091; border: 1px solid rgba(29, 158, 117, 0.3);"
        sign = "+"
    else:
        badge_style = "background-color: rgba(69, 10, 10, 0.8); color: #F87171; border: 1px solid rgba(153, 27, 27, 0.4);"
        sign = ""
    return f"""<span class="ev-badge" style="font-family: monospace; font-size: 11px; font-weight: bold; padding: 2px 6px; border-radius: 4px; {badge_style}">{sign}{value:.2f}</span>"""

# CSS 스타일 주입 (다크 네이비 테마 및 커스텀 테이블 등)
st.markdown("""
    <style>
    /* 전체 앱 스타일 */
    .stApp {
        background-color: #060B13 !important;
        color: #F8FAFC !important;
    }
    
    /* 사이드바 숨김 처리 및 레이아웃 조정 */
    section[data-testid="stSidebar"] {
        display: none !important;
    }
    
    /* 폰트 설정 */
    html, body, [class*="css"] {
        font-family: 'Outfit', 'Noto Sans KR', sans-serif !important;
    }
    
    /* 글래스모피즘 카드 */
    .glassmorphism {
        background: rgba(27, 38, 59, 0.7);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(245, 200, 66, 0.15);
        border-radius: 12px;
        padding: 18px;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.3);
    }
    
    /* 탭 헤더 커스텀 */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0px;
        background-color: #0D1B2A;
        border-bottom: 1px solid #1B263B;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        background-color: transparent;
        border-radius: 0px;
        color: #94A3B8;
        font-weight: 900;
        font-size: 14px;
        flex: 1;
        text-align: center;
        border: none;
    }
    .stTabs [data-baseweb="tab"]:hover {
        color: #F8FAFC;
    }
    .stTabs [aria-selected="true"] {
        color: #F5C842 !important;
        border-bottom: 2px solid #F5C842 !important;
        background-color: rgba(13, 27, 42, 0.5) !important;
    }
    
    /* HTML 테이블 공통 스타일 */
    .dashboard-table {
        width: 100%;
        border-collapse: collapse;
        margin-top: 10px;
        font-size: 13px;
        min-width: 650px;
    }
    .dashboard-table th {
        color: #778DA9;
        font-size: 10px;
        font-weight: 800;
        text-transform: uppercase;
        border-bottom: 1px solid #1B263B;
        padding: 14px 16px;
        text-align: left;
    }
    .dashboard-table td {
        padding: 14px 16px;
        border-bottom: 1px solid rgba(27, 38, 59, 0.4);
        vertical-align: middle;
    }
    .dashboard-table tr.top3-highlight {
        background-color: rgba(245, 200, 66, 0.06);
        border-left: 4px solid #F5C842;
    }
    .dashboard-table tr:hover {
        background-color: rgba(65, 90, 119, 0.15);
    }
    
    /* 켈리/수익 포트폴리오 스타일 */
    .portfolio-grid {
        display: grid;
        grid-template-columns: 1fr;
        gap: 16px;
    }
    @media (min-width: 768px) {
        .portfolio-grid {
            grid-template-columns: 1fr 1fr 1fr;
        }
    }
    
    /* 손익분기 비교 바 */
    .compare-bar-bg {
        width: 100%;
        height: 12px;
        background-color: #060B13;
        border: 1px solid #1B263B;
        border-radius: 4px;
        position: relative;
        overflow: hidden;
        margin-top: 6px;
    }
    .compare-bar-fill {
        height: 100%;
        background-color: #F5C842;
    }
    .compare-bar-marker {
        position: absolute;
        top: 0;
        bottom: 0;
        width: 2px;
        background-color: #1D9E75;
        z-index: 10;
    }
    </style>
""".replace("\n", ""), unsafe_allow_html=True)


# ---------------------- Header ----------------------
header_cols = st.columns([2, 1])

with header_cols[0]:
    st.markdown("""
        <div style="display: flex; align-items: center; gap: 12px;">
            <span style="font-size: 32px;">🏇</span>
            <div>
                <h1 style="font-size: 24px; font-weight: 900; color: #F5C842; margin: 0; padding: 0; line-height: 1.2;">
                    🏇 경마 예측 <span style="font-size: 11px; background-color: #DC2626; color: white; padding: 2px 6px; border-radius: 4px; font-family: monospace; vertical-align: middle; margin-left: 6px;">LIVE</span>
                </h1>
                <p style="font-size: 11px; color: #778DA9; margin: 2px 0 0 0; padding: 0;">LightGBM & Relative Features 기반 실시간 의사결정 시스템</p>
            </div>
        </div>""".replace("\n", ""), unsafe_allow_html=True)

# 경주 선택 가공
race_options = [f"{r['date']} | {r['no']} ({r['dist']})" for r in RACES]
with header_cols[1]:
    selected_option = st.selectbox(
        "경주 선택",
        options=race_options,
        label_visibility="collapsed"
    )

selected_race_idx = race_options.index(selected_option)
current_race = RACES[selected_race_idx]

# 피처 계산된 출전마 리스트
calculated_horses = calcRelFeatures(current_race["horses"])

# 1번 인기마 (배당률 최저인 말)
favorite_horse = min(calculated_horses, key=lambda x: x["odds"])

# 다크호스 후보 탐지: 배당률 >= 30, 예측확률 >= 35%
dark_horses = [h for h in calculated_horses if h["odds"] >= 30 and h["prob"] >= 35]

# 이변 확률 계산 (다습/포화 여부 기반)
if "포화" in current_race["track"] or "불량" in current_race["track"]:
    upset_prob = "매우 높음 (38%)"
    upset_color = "#F87171"
elif "다습" in current_race["track"]:
    upset_prob = "높음 (24%)"
    upset_color = "#FBBF24"
else:
    upset_prob = "보통 (12%)"
    upset_color = "#34D399"

# 헤더 컬럼 아래에 바로 트랙 뱃지 띄우기
with header_cols[1]:
    st.markdown(
        f"<div style='text-align: right; margin-top: 8px;'>{get_track_badge_html(current_race['track'])}</div>",
        unsafe_allow_html=True
    )


# ---------------------- Summary 카드 영역 ----------------------
st.write("") # 스페이서
summary_cols = st.columns(4)

with summary_cols[0]:
    st.markdown(f"""
        <div class="glassmorphism" style="height: 100%;">
            <span style="font-size: 12px; font-weight: 600; color: #94A3B8;">출전 두수</span>
            <div style="font-size: 24px; font-weight: 900; color: #F8FAFC; margin-top: 4px;">{len(calculated_horses)}두</div>
        </div>""".replace("\n", ""), unsafe_allow_html=True)

with summary_cols[1]:
    st.markdown(f"""
        <div class="glassmorphism" style="height: 100%;">
            <span style="font-size: 12px; font-weight: 600; color: #94A3B8;">1번 인기마 (배당률)</span>
            <div style="font-size: 18px; font-weight: 900; color: #F5C842; margin-top: 8px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;" title="{favorite_horse['name']} ({favorite_horse['odds']:.1f}배)">
                {favorite_horse['name']} ({favorite_horse['odds']:.1f}배)
            </div>
        </div>""".replace("\n", ""), unsafe_allow_html=True)

dh_badge_html = f"<span style='position: absolute; top: 8px; right: 8px; background-color: #EF9F27; color: #060B13; font-weight: 950; font-size: 9px; padding: 2px 6px; border-radius: 2px;'>HOT</span>" if len(dark_horses) > 0 else ""
dh_color = "#EF9F27" if len(dark_horses) > 0 else "#F8FAFC"
with summary_cols[2]:
    st.markdown(f"""
        <div class="glassmorphism" style="height: 100%; position: relative;">
            <span style="font-size: 12px; font-weight: 600; color: #94A3B8;">다크호스 탐지 수</span>
            <div style="font-size: 24px; font-weight: 900; color: {dh_color}; margin-top: 4px;">{len(dark_horses)}두</div>
            {dh_badge_html}
        </div>""".replace("\n", ""), unsafe_allow_html=True)

with summary_cols[3]:
    st.markdown(f"""
        <div class="glassmorphism" style="height: 100%;">
            <span style="font-size: 12px; font-weight: 600; color: #94A3B8;">이변 확률</span>
            <div style="font-size: 18px; font-weight: 900; color: {upset_color}; margin-top: 8px;">{upset_prob}</div>
        </div>""".replace("\n", ""), unsafe_allow_html=True)


# ---------------------- 다크호스 알림 배너 ----------------------
if len(dark_horses) > 0:
    badges_str = ""
    for dh in dark_horses:
        badges_str += f"""
        <div style="background-color: rgba(6, 11, 19, 0.8); border: 1px solid rgba(239, 159, 39, 0.5); padding: 6px 12px; border-radius: 8px; display: inline-flex; align-items: center; gap: 12px; font-size: 12px;">
            <span style="font-weight: 800; color: #F8FAFC;">{dh['name']}</span>
            <span style="color: #F5C842; font-weight: bold; font-family: monospace;">{dh['odds']:.1f}배</span>
            <span style="color: #1D9E75; font-weight: bold; font-family: monospace;">{dh['prob']:.1f}%</span>
            <span style="background-color: rgba(29, 158, 117, 0.15); color: #2CC091; padding: 2px 6px; border-radius: 4px; font-size: 10px; font-family: monospace;">EV: +{dh['ev']:.2f}</span>
        </div>"""
        
    st.markdown(f"""
        <div style="background: linear-gradient(90deg, rgba(239, 159, 39, 0.15) 0%, rgba(220, 38, 38, 0.1) 50%, rgba(13, 27, 42, 0.4) 100%); border: 1px solid rgba(239, 159, 39, 0.3); border-radius: 12px; padding: 16px; display: flex; flex-direction: column; gap: 12px; box-shadow: 0 4px 15px rgba(0,0,0,0.2);">
            <div style="display: flex; align-items: center; gap: 12px;">
                <span style="font-size: 24px;">🔥</span>
                <div>
                    <h3 style="font-size: 14px; font-weight: bold; color: #EF9F27; margin: 0; padding: 0;">초고배당 다크호스 감지!</h3>
                    <p style="font-size: 12px; color: #CBD5E1; margin: 2px 0 0 0; padding: 0;">배당 30배 이상 & 입상 확률 35% 이상 조건을 만족하는 요주의 마필이 있습니다.</p>
                </div>
            </div>
            <div style="display: flex; flex-wrap: wrap; gap: 8px;">
                {badges_str}
            </div>
        </div>""".replace("\n", ""), unsafe_allow_html=True)


# ---------------------- 탭 네비게이션 ----------------------

# bet_budget 초기화 (View B에서 예상수익 연산을 위해 상단 선언)
if "budget_slider" not in st.session_state:
    st.session_state.budget_slider = 50000
bet_budget = st.session_state.budget_slider

tabs = st.tabs(["📊 View A: 전체 분석", "🎲 View B: 베팅 결정", "💼 View C: 포트폴리오"])

# ---------------------- View A — 전체 분석 탭 ----------------------
with tabs[0]:
    # 게이트x거리 인사이트 박스
    is_short_dist = int(current_race["dist"].replace("M", "")) <= 1400
    if is_short_dist:
        dist_insight = "내측 게이트 유리 — 외측 대비 약 8%p 높은 입상률"
    else:
        dist_insight = "거리 적성이 게이트보다 중요"

    wet_warn_html = ""
    if "다습" in current_race["track"] or "포화" in current_race["track"] or "불량" in current_race["track"]:
        wet_warn_html = """
        <div style="background-color: rgba(239, 159, 39, 0.15); border: 1px solid rgba(239, 159, 39, 0.3); padding: 8px 12px; border-radius: 8px; font-size: 12px; display: inline-flex; align-items: center; gap: 8px;">
            <span style="color: #EF9F27;">⚠️</span>
            <span style="color: #FBBF24; font-weight: 600;">다습 주로 경보: 이변 발생률 +0.9%p — 고배당 주목</span>
        </div>"""

    st.markdown(f"""
        <div style="background-color: rgba(13, 27, 42, 0.4); border: 1px solid #1B263B; border-radius: 12px; padding: 16px; display: flex; flex-direction: column; md-flex-direction: row; justify-content: space-between; align-items: flex-start; md-align-items: center; gap: 12px; margin-bottom: 16px;">
            <div style="display: flex; align-items: center; gap: 12px;">
                <span style="font-size: 20px;">💡</span>
                <div>
                    <h4 style="font-size: 13px; font-weight: bold; color: #F5C842; margin: 0; padding: 0;">경주로 및 게이트 조건별 AI 특수 분석</h4>
                    <p style="font-size: 12px; color: #94A3B8; margin: 2px 0 0 0; padding: 0;">{dist_insight}</p>
                </div>
            </div>
            {wet_warn_html}
        </div>""".replace("\n", ""), unsafe_allow_html=True)

    # 테이블 그리기
    table_rows_html = ""
    for idx, h in enumerate(calculated_horses):
        highlight_class = "top3-highlight" if idx < 3 else ""
        all_pass_star = "<span style='color: #F5C842; font-size: 12px;' title='3대 베팅 지표 만장일치 추천'>★</span>" if (h["recallFlag"] and h["precisionFlag"] and h["evFlag"]) else ""
        
        weight_diff_str = ""
        if h["is_debut"] == 0:
            diff_val = h["weight_diff"]
            diff_sign = "+" if diff_val >= 0 else ""
            diff_color = "#F87171" if diff_val > 0 else ("#60A5FA" if diff_val < 0 else "#94A3B8")
            weight_diff_str = f"<span style='margin-left: 4px; font-size: 10px; color: {diff_color};'>({diff_sign}{diff_val})</span>"
        else:
            weight_diff_str = "<span style='margin-left: 4px; font-size: 10px; color: #64748B;'>(신예)</span>"

        table_rows_html += f"""<tr class="{highlight_class}">
            <td style="text-align: center;">
                <span style="display: inline-flex; align-items: center; justify-content: center; width: 24px; height: 24px; border-radius: 9999px; font-weight: 900; font-size: 12px; 
                  {'background-color: #F5C842; color: #060B13;' if idx == 0 else ('background-color: #E2E8F0; color: #060B13;' if idx == 1 else ('background-color: #B45309; color: #F8FAFC;' if idx == 2 else 'background-color: #1B263B; color: #94A3B8;'))}">
                  {idx + 1}
                </span>
            </td>
            <td>
                <div style="font-weight: 800; font-size: 14px; color: #F8FAFC; display: flex; align-items: center; gap: 6px;">
                    {h['name']} {all_pass_star}
                </div>
                <div style="font-size: 10px; color: #94A3B8;">{h['jockey']} / {h['trainer']}</div>
            </td>
            <td>{get_gate_tag_html(h['gate'])}</td>
            <td style="font-weight: 600; color: #E2E8F0;">
                {h['age']}세 / {h['weight']}kg {weight_diff_str}
            </td>
            <td style="text-align: center; font-family: monospace; font-weight: bold; color: #E2E8F0;">
                {int(h['top3_rate_last_5']*100)}%
            </td>
            <td style="text-align: center; font-family: monospace; font-weight: bold; color: #E2E8F0;">
                {h['synergy']*100:.1f}%
            </td>
            <td style="text-align: center; display: flex; justify-content: center;">
                {get_condition_light_html(h)}
            </td>
            <td style="text-align: right; font-family: monospace; font-weight: 900; color: #F5C842; font-size: 14px;">
                {h['odds']:.1f}배
            </td>
            <td>{get_prob_bar_html(h['prob'])}</td>
        </tr>"""

    st.markdown(f"""
        <div class="glassmorphism" style="padding: 0px; overflow-x: auto; border: 1px solid #1B263B;">
            <table class="dashboard-table">
                <thead>
                    <tr>
                        <th style="text-align: center; width: 80px;">예측순위</th>
                        <th>마필명</th>
                        <th>게이트</th>
                        <th>연령/체중변화</th>
                        <th style="text-align: center;">최근기세(5경기 Top3%)</th>
                        <th style="text-align: center;">기수-조교사 시너지</th>
                        <th style="text-align: center; width: 100px;">컨디션신호등</th>
                        <th style="text-align: right; width: 100px;">배당률</th>
                        <th style="text-align: right; width: 180px;">예측확률(프로그레스바)</th>
                    </tr>
                </thead>
                <tbody>
                    {table_rows_html}
                </tbody>
            </table>
        </div>""".replace("\n", ""), unsafe_allow_html=True)


# ---------------------- View B — 베팅 결정 탭 ----------------------
with tabs[1]:
    # 3방향 모델
    dir_a_horses = [h for h in calculated_horses if h["recallFlag"]][:3]
    dir_b_horses = [h for h in calculated_horses if h["precisionFlag"]][:3]
    
    # EV 내림차순 정렬 후 EV > 0 인 것 상위 3개
    ev_positive_horses = [h for h in calculated_horses if h["evFlag"]]
    ev_positive_sorted = sorted(ev_positive_horses, key=lambda x: x["ev"], reverse=True)
    dir_c_horses = ev_positive_sorted[:3]

    cols_strategy = st.columns(3)

    # 방향 A - 다크호스형
    with cols_strategy[0]:
        horses_list_html = ""
        if len(dir_a_horses) > 0:
            for h in dir_a_horses:
                horses_list_html += f"""
                <div style="background-color: rgba(6, 11, 19, 0.6); padding: 10px; border-radius: 8px; border: 1px solid #1B263B; display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                    <div>
                        <span style="font-weight: 800; font-size: 13px; color: #F8FAFC;">{h['name']}</span>
                        <span style="color: #94A3B8; font-size: 11px; margin-left: 6px;">({h['odds']:.1f}배)</span>
                    </div>
                    <div style="display: flex; align-items: center; gap: 8px;">
                        {get_prob_bar_html(h['prob'], width_px=48)}
                        {get_ev_badge_html(h['ev'])}
                    </div>
                </div>"""
        else:
            horses_list_html = "<div style='font-size: 12px; color: #64748B; text-align: center; padding: 24px;'>조건을 충족하는 추천마가 없습니다.</div>"

        st.markdown(f"""
            <div class="glassmorphism" style="border-top: 4px solid #3B82F6; display: flex; flex-direction: column; justify-content: space-between; height: 100%;">
                <div>
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <span style="font-size: 11px; font-weight: 900; color: #60A5FA; tracking-wider; text-transform: uppercase;">방향 A — 다크호스형</span>
                        <span style="background-color: rgba(59, 130, 246, 0.2); color: #60A5FA; padding: 2px 6px; border-radius: 4px; font-size: 10px; font-family: monospace; font-weight: bold;">Recall 84%</span>
                    </div>
                    <h4 style="font-size: 14px; font-weight: 800; color: #E2E8F0; margin: 8px 0 0 0;">놓치지 않는다 — Recall 극대화</h4>
                    <p style="font-size: 11px; color: #64748B; margin: 4px 0 12px 0;">Threshold: 0.30. 입상 유력마를 배제하지 않고 폭넓게 추천.</p>
                    <div style="margin-top: 8px;">
                        {horses_list_html}
                    </div>
                </div>
                <div style="margin-top: 12px; padding-top: 12px; border-top: 1px solid rgba(65, 90, 119, 0.3); font-size: 10px; color: #94A3B8; text-align: center;">
                    특징: 고배당 다크호스 적중 가능성 높음, 오베팅 감수
                </div>
            </div>""".replace("\n", ""), unsafe_allow_html=True)

    # 방향 B - 안정형
    with cols_strategy[1]:
        horses_list_html = ""
        if len(dir_b_horses) > 0:
            for h in dir_b_horses:
                horses_list_html += f"""
                <div style="background-color: rgba(6, 11, 19, 0.6); padding: 10px; border-radius: 8px; border: 1px solid #1B263B; display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                    <div>
                        <span style="font-weight: 800; font-size: 13px; color: #F8FAFC;">{h['name']}</span>
                        <span style="color: #94A3B8; font-size: 11px; margin-left: 6px;">({h['odds']:.1f}배)</span>
                    </div>
                    <div style="display: flex; align-items: center; gap: 8px;">
                        {get_prob_bar_html(h['prob'], width_px=48)}
                        {get_ev_badge_html(h['ev'])}
                    </div>
                </div>"""
        else:
            horses_list_html = """
            <div style='font-size: 12px; color: #EF9F27; background-color: rgba(245, 158, 11, 0.1); border: 1px solid rgba(245, 158, 11, 0.2); border-radius: 8px; padding: 16px; text-align: center;'>
                ⚠️ 현재 경주에서 Precision 70% 이상 조건을 충족하는 말이 없습니다.
            </div>"""

        st.markdown(f"""
            <div class="glassmorphism" style="border-top: 4px solid #10B981; display: flex; flex-direction: column; justify-content: space-between; height: 100%;">
                <div>
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <span style="font-size: 11px; font-weight: 900; color: #34D399; tracking-wider; text-transform: uppercase;">방향 B — 안정형</span>
                        <span style="background-color: rgba(16, 185, 129, 0.2); color: #34D399; padding: 2px 6px; border-radius: 4px; font-size: 10px; font-family: monospace; font-weight: bold;">Precision 70%</span>
                    </div>
                    <h4 style="font-size: 14px; font-weight: 800; color: #E2E8F0; margin: 8px 0 0 0;">확실히 맞힌다 — Precision 극대화</h4>
                    <p style="font-size: 11px; color: #64748B; margin: 4px 0 12px 0;">Threshold: 0.70. 추천 건수는 적지만 높은 적중 신뢰도 보장.</p>
                    <div style="margin-top: 8px;">
                        {horses_list_html}
                    </div>
                </div>
                <div style="margin-top: 12px; padding-top: 12px; border-top: 1px solid rgba(65, 90, 119, 0.3); font-size: 10px; color: #94A3B8; text-align: center;">
                    특징: 추천 건수 적지만 적중률 높음
                </div>
            </div>""".replace("\n", ""), unsafe_allow_html=True)

    # 방향 C - 수익형
    with cols_strategy[2]:
        horses_list_html = ""
        if len(dir_c_horses) > 0:
            for h in dir_c_horses:
                horses_list_html += f"""
                <div style="background-color: rgba(6, 11, 19, 0.6); padding: 10px; border-radius: 8px; border: 1px solid #1B263B; display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                    <div>
                        <span style="font-weight: 800; font-size: 13px; color: #F8FAFC;">{h['name']}</span>
                        <span style="color: #94A3B8; font-size: 11px; margin-left: 6px;">({h['odds']:.1f}배)</span>
                    </div>
                    <div style="display: flex; align-items: center; gap: 8px;">
                        {get_prob_bar_html(h['prob'], width_px=48)}
                        {get_ev_badge_html(h['ev'])}
                    </div>
                </div>"""
        else:
            horses_list_html = "<div style='font-size: 12px; color: #64748B; text-align: center; padding: 24px;'>조건을 충족하는 추천마가 없습니다.</div>"

        st.markdown(f"""
            <div class="glassmorphism" style="border-top: 4px solid #EF9F27; display: flex; flex-direction: column; justify-content: space-between; height: 100%;">
                <div>
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <span style="font-size: 11px; font-weight: 900; color: #EF9F27; tracking-wider; text-transform: uppercase;">방향 C — 수익형</span>
                        <span style="background-color: rgba(239, 159, 39, 0.2); color: #EF9F27; padding: 2px 6px; border-radius: 4px; font-size: 10px; font-family: monospace; font-weight: bold;">EV 극대화</span>
                    </div>
                    <h4 style="font-size: 14px; font-weight: 800; color: #E2E8F0; margin: 8px 0 0 0;">수익을 최적화한다 — EV 극대화</h4>
                    <p style="font-size: 11px; color: #64748B; margin: 4px 0 12px 0;">EV > 0 기준. 배당률 대비 기대 가치 극대화 포지션.</p>
                    <div style="margin-top: 8px;">
                        {horses_list_html}
                    </div>
                </div>
                <div style="margin-top: 12px; padding-top: 12px; border-top: 1px solid rgba(65, 90, 119, 0.3); font-size: 10px; color: #94A3B8; text-align: center;">
                    특징: 배당률 × 확률 기대수익 기준
                </div>
            </div>""".replace("\n", ""), unsafe_allow_html=True)

    # 출전마 전체 베팅 비교표
    st.write("")
    st.markdown("<h4 style='font-size: 14px; font-weight: bold; margin-bottom: 8px;'>출전마 전체 베팅 비교표</h4>".replace("\n", ""), unsafe_allow_html=True)
    
    bet_compare_rows = ""
    for h in calculated_horses:
        is_all_recommended = h["recallFlag"] and h["precisionFlag"] and h["evFlag"]
        star_badge = "<span style='background-color: #F5C842; color: #060B13; font-weight: 900; font-size: 9px; padding: 2px 6px; border-radius: 2px; margin-left: 6px;'>★ ALL PASS</span>" if is_all_recommended else ""
        
        check_a = "<span style='color: #60A5FA; font-weight: 900; font-size: 14px;'>✓</span>" if h["recallFlag"] else "<span style='color: #475569;'>✕</span>"
        check_b = "<span style='color: #34D399; font-weight: 900; font-size: 14px;'>✓</span>" if h["precisionFlag"] else "<span style='color: #475569;'>✕</span>"
        check_c = "<span style='color: #EF9F27; font-weight: 900; font-size: 14px;'>✓</span>" if h["evFlag"] else "<span style='color: #475569;'>✕</span>"
        
        # 예상 수익: EV * 베팅금액
        exp_return_val = h["ev"] * bet_budget
        if h["ev"] > 0:
            exp_return_str = f"<span style='color: #34D399; font-weight: bold;'>+{int(exp_return_val):,}원</span>"
        else:
            exp_return_str = f"<span style='color: #F87171; font-weight: bold;'>{int(exp_return_val):,}원</span>"

        bet_compare_rows += f"""
        <tr>
            <td>
                <div style="font-weight: bold; font-size: 13px; color: #F8FAFC; display: flex; align-items: center;">
                    {h['name']} {star_badge}
                </div>
                <div style="font-size: 11px; color: #F5C842; font-weight: bold; font-family: monospace; margin-top: 2px;">{h['odds']:.1f}배</div>
            </td>
            <td>{get_prob_bar_html(h['prob'])}</td>
            <td style="text-align: center;">{get_ev_badge_html(h['ev'])}</td>
            <td style="text-align: center; background-color: rgba(59, 130, 246, 0.03);">{check_a}</td>
            <td style="text-align: center; background-color: rgba(16, 185, 129, 0.03);">{check_b}</td>
            <td style="text-align: center; background-color: rgba(245, 158, 11, 0.03);">{check_c}</td>
            <td style="text-align: right; font-family: monospace; font-size: 13px;">{exp_return_str}</td>
        </tr>"""

    st.markdown(f"""
        <div class="glassmorphism" style="padding: 0px; overflow-x: auto; border: 1px solid #1B263B;">
            <table class="dashboard-table">
                <thead>
                    <tr>
                        <th>마필명/배당률</th>
                        <th style="text-align: right; width: 180px;">예측확률(바)</th>
                        <th style="text-align: center; width: 100px;">EV값</th>
                        <th style="text-align: center; width: 110px; background-color: rgba(59, 130, 246, 0.05);">A추천 (Recall)</th>
                        <th style="text-align: center; width: 110px; background-color: rgba(16, 185, 129, 0.05);">B추천 (Precision)</th>
                        <th style="text-align: center; width: 110px; background-color: rgba(245, 158, 11, 0.05);">C추천 (EV)</th>
                        <th style="text-align: right; width: 110px;">예상수익</th>
                    </tr>
                </thead>
                <tbody>
                    {bet_compare_rows}
                </tbody>
            </table>
        </div>""".replace("\n", ""), unsafe_allow_html=True)


# ---------------------- View C — 포트폴리오 탭 ----------------------
with tabs[2]:
    # 베팅 금액 슬라이더
    st.markdown("<h4 style='font-size: 13px; font-weight: bold; color: #94A3B8; margin-bottom: 4px;'>베팅 설정</h4>".replace("\n", ""), unsafe_allow_html=True)
    bet_budget = st.slider(
        "베팅 금액 설정 (5,000원 ~ 100,000원)",
        min_value=5000,
        max_value=100000,
        value=50000,
        step=5000,
        key="budget_slider",
        label_visibility="collapsed"
    )
    
    st.markdown(f"""
        <div style="margin-top: 8px; margin-bottom: 24px;">
            <span style="font-size: 12px; color: #94A3B8; font-weight: bold;">설정된 투자금:</span>
            <span style="font-size: 24px; font-weight: 900; color: #F5C842; margin-left: 8px; font-family: monospace;">{bet_budget:,}원</span>
        </div>""".replace("\n", ""), unsafe_allow_html=True)

    # 포트폴리오 시나리오 3종
    st.markdown("<h4 style='font-size: 14px; font-weight: bold; margin-bottom: 12px;'>포트폴리오 시나리오 3종</h4>".replace("\n", ""), unsafe_allow_html=True)
    
    cols_portfolio = st.columns(3)

    # 1. 안정형 포트폴리오 (초록 테마)
    with cols_portfolio[0]:
        stable_body = ""
        if len(dir_b_horses) > 0:
            alloc_stable = bet_budget / len(dir_b_horses)
            for h in dir_b_horses:
                stable_body += f"""
                <div style="background-color: rgba(6, 11, 19, 0.6); padding: 10px; border-radius: 8px; border: 1px solid #1B263B; margin-bottom: 8px; font-size: 12px;">
                    <div style="display: flex; justify-content: space-between; font-weight: bold; color: #F8FAFC;">
                        <span>{h['name']} (단승)</span>
                        <span style="color: #F5C842; font-family: monospace;">{h['odds']:.1f}배</span>
                    </div>
                    <div style="display: flex; justify-content: space-between; color: #94A3B8; font-size: 11px; margin-top: 6px;">
                        <span>배분액: {int(alloc_stable):,}원</span>
                        <span style="color: #34D399; font-weight: bold; font-family: monospace;">적중시: {int(alloc_stable * h['odds']):,}원</span>
                    </div>
                    <div style="text-align: right; color: #64748B; font-size: 10px; margin-top: 4px;">
                        손익분기 배당: {(100 / h['prob']):.1f}배
                    </div>
                </div>"""
        else:
            # 대안: 1순위 예측마 올인
            top_horse = calculated_horses[0]
            stable_body = f"""
            <div style="background-color: rgba(6, 11, 19, 0.8); border: 1px solid #1B263B; border-radius: 8px; padding: 12px; font-size: 12px;">
                <div style="color: #94A3B8; font-weight: bold; text-align: center; margin-bottom: 8px;">Precision 충족 조건 부재</div>
                <div style="color: #64748B; text-align: center; font-size: 11px; margin-bottom: 8px;">대안: 1순위 예측마 {top_horse['name']} 단승 올인</div>
                <div style="background-color: rgba(13, 27, 42, 0.5); padding: 10px; border-radius: 6px;">
                    <div style="display: flex; justify-content: space-between; font-weight: bold; color: #F8FAFC;">
                        <span>{top_horse['name']}</span>
                        <span style="color: #F5C842;">{top_horse['odds']:.1f}배</span>
                    </div>
                    <span style="color: #34D399; font-weight: bold; font-family: monospace; display: block; margin-top: 4px; text-align: right;">적중시: {int(bet_budget * top_horse['odds']):,}원</span>
                </div>
            </div>"""
        
        st.markdown(f"""
            <div class="glassmorphism" style="border-top: 4px solid #10B981; display: flex; flex-direction: column; justify-content: space-between; height: 100%;">
                <div>
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px;">
                        <h4 style="font-size: 14px; font-weight: 900; color: #34D399; margin: 0; text-transform: uppercase; tracking-wider;">안정형 포트폴리오</h4>
                        <span style="background-color: rgba(16, 185, 129, 0.2); color: #34D399; font-size: 9px; font-weight: bold; padding: 2px 6px; border-radius: 4px;">로우리스크</span>
                    </div>
                    <p style="font-size: 11px; color: #64748B; margin: 0 0 12px 0;">방향 B (Precision) 추천마 단승 집중 베팅</p>
                    <div>
                        {stable_body}
                    </div>
                </div>
                <div style="margin-top: 12px; padding-top: 12px; border-top: 1px solid rgba(65, 90, 119, 0.3); font-size: 10px; color: #94A3B8;">
                    손익분기 최소 요건: 배당 {(100 / (calculated_horses[0]['prob'] or 1)):.1f}배 이상 유지
                </div>
            </div>""".replace("\n", ""), unsafe_allow_html=True)

    # 2. 다크호스 포트폴리오 (파랑 테마)
    with cols_portfolio[1]:
        # 교집합 마필 중 배당률 10배 이상인 마필 최대 2개
        dh_port_horses = [h for h in calculated_horses if h["recallFlag"] and h["evFlag"] and h["odds"] >= 10][:2]
        
        dh_body = ""
        if len(dh_port_horses) > 0:
            alloc_dh = bet_budget / len(dh_port_horses)
            for h in dh_port_horses:
                dh_body += f"""
                <div style="background-color: rgba(6, 11, 19, 0.6); padding: 10px; border-radius: 8px; border: 1px solid #1B263B; margin-bottom: 8px; font-size: 12px;">
                    <div style="display: flex; justify-content: space-between; font-weight: bold; color: #F8FAFC;">
                        <span>{h['name']} (단승)</span>
                        <span style="color: #F5C842; font-family: monospace;">{h['odds']:.1f}배</span>
                    </div>
                    <div style="display: flex; justify-content: space-between; color: #94A3B8; font-size: 11px; margin-top: 6px;">
                        <span>배분액: {int(alloc_dh):,}원</span>
                        <span style="color: #60A5FA; font-weight: bold; font-family: monospace;">적중시: {int(alloc_dh * h['odds']):,}원</span>
                    </div>
                </div>"""
        else:
            dh_body = "<div style='font-size: 12px; color: #64748B; text-align: center; padding: 32px;'>교집합 고배당 조건 부재<br><span style='font-size: 10px;'>(단거리 위주 인기 정배 발생)</span></div>"

        st.markdown(f"""
            <div class="glassmorphism" style="border-top: 4px solid #3B82F6; display: flex; flex-direction: column; justify-content: space-between; height: 100%;">
                <div>
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px;">
                        <h4 style="font-size: 14px; font-weight: 900; color: #60A5FA; margin: 0; text-transform: uppercase; tracking-wider;">다크호스 포트폴리오</h4>
                        <span style="background-color: rgba(59, 130, 246, 0.2); color: #60A5FA; font-size: 9px; font-weight: bold; padding: 2px 6px; border-radius: 4px;">하이리턴</span>
                    </div>
                    <p style="font-size: 11px; color: #64748B; margin: 0 0 12px 0;">방향 A(Recall) ∩ 방향 C(EV) 교집합 마필 소액 분산</p>
                    <div>
                        {dh_body}
                    </div>
                </div>
                <div style="margin-top: 12px; padding-top: 12px; border-top: 1px solid rgba(65, 90, 119, 0.3); font-size: 10px; color: #94A3B8;">
                    <div style="color: #60A5FA; font-weight: bold; font-size: 11px; margin-bottom: 4px;">
                        예상 최대수익: {int(max([(bet_budget / len(dh_port_horses)) * hx['odds'] for hx in dh_port_horses]) - bet_budget) if len(dh_port_horses) > 0 else 0:,}원
                    </div>
                    주의: 고배당 베팅은 환수율 분산으로 소액 투자 권장
                </div>
            </div>""".replace("\n", ""), unsafe_allow_html=True)

    # 3. 켈리 기준 포트폴리오 (주황 테마)
    with cols_portfolio[2]:
        # EV 양수인 마필들만 대상
        kelly_eligible = [h for h in calculated_horses if h["kelly"] > 0]
        sum_kelly = sum(h["kelly"] for h in kelly_eligible)
        
        kelly_body = ""
        if sum_kelly > 0:
            kelly_allocations = []
            for h in kelly_eligible:
                ratio = h["kelly"] / sum_kelly
                # 1000원 단위 반올림
                amount = int(round((bet_budget * ratio) / 1000) * 1000)
                kelly_allocations.append({
                    "name": h["name"],
                    "odds": h["odds"],
                    "ratio": ratio,
                    "amount": amount
                })
            
            # 배분금액 기준 내림차순
            kelly_allocations.sort(key=lambda x: x["amount"], reverse=True)
            
            kelly_body += """
            <div style="overflow-x: auto;">
                <table style="width: 100%; text-align: left; font-size: 12px; border-collapse: collapse;">
                    <thead>
                        <tr style="color: #64748B; border-bottom: 1px solid #1B263B; font-size: 10px; font-weight: bold;">
                            <th style="padding-bottom: 8px;">마필명</th>
                            <th style="padding-bottom: 8px; text-align: center;">비율</th>
                            <th style="padding-bottom: 8px; text-align: right;">권장배팅</th>
                            <th style="padding-bottom: 8px; text-align: right;">적중예상</th>
                        </tr>
                    </thead>
                    <tbody style="border-top: 1px solid rgba(27,38,59,0.3);">
            """
            for ka in kelly_allocations[:3]:
                if ka["amount"] > 0:
                    kelly_body += f"""
                        <tr style="border-bottom: 1px solid rgba(27,38,59,0.2);">
                            <td style="padding: 10px 0; font-weight: bold; color: #E2E8F0;">{ka['name']}</td>
                            <td style="padding: 10px 0; text-align: center; font-family: monospace; color: #EF9F27; font-weight: bold;">{ka['ratio']*100:.0f}%</td>
                            <td style="padding: 10px 0; text-align: right; font-family: monospace; color: #CBD5E1;">{ka['amount']:,}원</td>
                            <td style="padding: 10px 0; text-align: right; font-family: monospace; font-weight: bold; color: #F5C842;">{int(ka['amount'] * ka['odds']):,}원</td>
                        </tr>"""
            kelly_body += """
                    </tbody>
                </table>
            </div>"""
        else:
            kelly_body = "<div style='font-size: 12px; color: #64748B; text-align: center; padding: 32px;'>EV 양수 마필 부재로 배분값이 산출되지 않습니다.</div>"

        st.markdown(f"""
            <div class="glassmorphism" style="border-top: 4px solid #EF9F27; display: flex; flex-direction: column; justify-content: space-between; height: 100%;">
                <div>
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px;">
                        <h4 style="font-size: 14px; font-weight: 900; color: #EF9F27; margin: 0; text-transform: uppercase; tracking-wider;">켈리 기준 포트폴리오</h4>
                        <span style="background-color: rgba(239, 159, 39, 0.2); color: #EF9F27; font-size: 9px; font-weight: bold; padding: 2px 6px; border-radius: 4px;">수학적 최적화</span>
                    </div>
                    <p style="font-size: 11px; color: #64748B; margin: 0 0 12px 0;">공식 (bp - q) / b에 기반한 수학적 환수율 최적 배분</p>
                    <div>
                        {kelly_body}
                    </div>
                </div>
                <div style="margin-top: 12px; padding-top: 12px; border-top: 1px solid rgba(65, 90, 119, 0.3); font-size: 10px; color: #94A3B8;">
                    공식 설명: b=배당률-1, p=확률, q=실패율
                </div>
            </div>""".replace("\n", ""), unsafe_allow_html=True)


    # 조합 베팅 기대수익 및 손익분기 분석
    st.write("")
    cols_details = st.columns(2)

    # 1. 조합 베팅 기대수익 테이블
    with cols_details[0]:
        st.markdown("<h4 style='font-size: 14px; font-weight: bold; margin-bottom: 8px;'>베팅 승식 조합별 예상 가치</h4>".replace("\n", ""), unsafe_allow_html=True)
        
        h0 = calculated_horses[0]
        h1 = calculated_horses[1]
        
        # 단승 (Win)
        odds_win = h0["odds"]
        prob_win = h0["prob"]
        ev_win_val = ((prob_win / 100 * odds_win) - (1 - prob_win / 100)) * bet_budget
        ev_win_color = "#34D399" if ev_win_val > 0 else "#F87171"
        ev_win_sign = "+" if ev_win_val > 0 else ""
        
        # 연승 (Place)
        odds_place = 1 + (h0["odds"] - 1) * 0.35
        prob_place = min(95.0, h0["prob"] * 1.5)
        ev_place_val = ((prob_place / 100 * odds_place) - (1 - prob_place / 100)) * bet_budget
        ev_place_color = "#34D399" if ev_place_val > 0 else "#F87171"
        ev_place_sign = "+" if ev_place_val > 0 else ""
        
        # 복승 (Quinella)
        odds_quin = (h0["odds"] * h1["odds"]) / 2.8
        prob_quin = (h0["prob"] / 100 * h1["prob"] / 100) * 1.2 * 100
        ev_quin_val = ((prob_quin / 100 * odds_quin) - (1 - prob_quin / 100)) * bet_budget
        ev_quin_color = "#34D399" if ev_quin_val > 0 else "#F87171"
        ev_quin_sign = "+" if ev_quin_val > 0 else ""

        st.markdown(f"""
            <div class="glassmorphism" style="padding: 16px; border: 1px solid #1B263B; overflow-x: auto;">
                <table style="width: 100%; border-collapse: collapse; font-size: 12px; color: #F8FAFC; min-width: 300px;">
                    <thead>
                        <tr style="color: #778DA9; border-bottom: 1px solid #1B263B; text-align: left;">
                            <th style="padding: 8px 0; font-size: 10px; font-weight: bold; border: none; color: #778DA9;">베팅 유형</th>
                            <th style="padding: 8px 0; font-size: 10px; font-weight: bold; border: none; color: #778DA9;">대상마</th>
                            <th style="padding: 8px 0; font-size: 10px; font-weight: bold; border: none; color: #778DA9; text-align: right;">환산 배당</th>
                            <th style="padding: 8px 0; font-size: 10px; font-weight: bold; border: none; color: #778DA9; text-align: right;">적중 확률</th>
                            <th style="padding: 8px 0; font-size: 10px; font-weight: bold; border: none; color: #778DA9; text-align: right;">손익분기</th>
                            <th style="padding: 8px 0; font-size: 10px; font-weight: bold; border: none; color: #778DA9; text-align: right;">기대 수익 (EV)</th>
                        </tr>
                    </thead>
                    <tbody style="border-top: 1px solid rgba(27,38,59,0.3);">
                        <tr style="border-bottom: 1px solid rgba(27,38,59,0.2); {'background-color: rgba(16, 185, 129, 0.1); border-left: 4px solid #10B981;' if ev_win_val > 0 else ''}">
                            <td style="padding: 10px 0; font-weight: bold; padding-left: 8px;">단승 (Win)</td>
                            <td style="padding: 10px 0; color: #CBD5E1;">{h0['name']}</td>
                            <td style="padding: 10px 0; text-align: right; font-family: monospace; font-weight: bold; color: #F5C842;">{odds_win:.1f}배</td>
                            <td style="padding: 10px 0; text-align: right; font-family: monospace;">{prob_win:.1f}%</td>
                            <td style="padding: 10px 0; text-align: right; font-family: monospace; color: #64748B;">{(100 / (prob_win or 1)):.1f}배</td>
                            <td style="padding: 10px 0; text-align: right; font-family: monospace; font-weight: bold; color: {ev_win_color};">{ev_win_sign}{int(ev_win_val):,}원</td>
                        </tr>
                        <tr style="border-bottom: 1px solid rgba(27,38,59,0.2); {'background-color: rgba(16, 185, 129, 0.1); border-left: 4px solid #10B981;' if ev_place_val > 0 else ''}">
                            <td style="padding: 10px 0; font-weight: bold; padding-left: 8px;">연승 (Place)</td>
                            <td style="padding: 10px 0; color: #CBD5E1;">{h0['name']}</td>
                            <td style="padding: 10px 0; text-align: right; font-family: monospace; font-weight: bold; color: #F5C842;">{odds_place:.1f}배</td>
                            <td style="padding: 10px 0; text-align: right; font-family: monospace;">{prob_place:.1f}%</td>
                            <td style="padding: 10px 0; text-align: right; font-family: monospace; color: #64748B;">{(100 / (prob_place or 1)):.1f}배</td>
                            <td style="padding: 10px 0; text-align: right; font-family: monospace; font-weight: bold; color: {ev_place_color};">{ev_place_sign}{int(ev_place_val):,}원</td>
                        </tr>
                        <tr style="{'background-color: rgba(16, 185, 129, 0.1); border-left: 4px solid #10B981;' if ev_quin_val > 0 else ''}">
                            <td style="padding: 10px 0; font-weight: bold; padding-left: 8px;">복승 (Quinella)</td>
                            <td style="padding: 10px 0; color: #CBD5E1;">{h0['name']} + {h1['name']}</td>
                            <td style="padding: 10px 0; text-align: right; font-family: monospace; font-weight: bold; color: #F5C842;">{odds_quin:.1f}배</td>
                            <td style="padding: 10px 0; text-align: right; font-family: monospace;">{prob_quin:.1f}%</td>
                            <td style="padding: 10px 0; text-align: right; font-family: monospace; color: #64748B;">{(100 / (prob_quin or 1)):.1f}배</td>
                            <td style="padding: 10px 0; text-align: right; font-family: monospace; font-weight: bold; color: {ev_quin_color};">{ev_quin_sign}{int(ev_quin_val):,}원</td>
                        </tr>
                    </tbody>
                </table>
            </div>""".replace("\n", ""), unsafe_allow_html=True)

    # 2. 손익분기 역산 표시
    with cols_details[1]:
        st.markdown("<h4 style='font-size: 14px; font-weight: bold; margin-bottom: 8px;'>마필별 개별 손익분기점 역산 (Break-Even)</h4>".replace("\n", ""), unsafe_allow_html=True)
        
        horse_names = [h["name"] for h in calculated_horses]
        break_even_name = st.selectbox(
            "분석 대상 마필",
            options=horse_names,
            key="be_selectbox",
            label_visibility="collapsed"
        )
        
        bh = next(h for h in calculated_horses if h["name"] == break_even_name)
        break_even_odds = 100 / bh["prob"]
        
        # 바 너비 백분율 계산
        bar_fill_pct = min(100.0, (bh["odds"] / (break_even_odds * 1.5)) * 100)
        marker_left_pct = (break_even_odds / (break_even_odds * 1.5)) * 100
        
        if bh["odds"] >= break_even_odds:
            compare_msg = "✓ 현재 배당률이 AI 손익분기를 초과합니다. 베팅 시 장기적으로 수익이 누적됩니다."
            compare_color = "#34D399"
        else:
            compare_msg = "✕ 현재 배당률이 AI 손익분기 미만입니다. 베팅 시 장기적으로 손실을 입게 됩니다."
            compare_color = "#F87171"

        st.markdown(f"""
            <div class="glassmorphism" style="padding: 18px; border: 1px solid #1B263B; display: flex; flex-direction: column; gap: 10px;">
                <p style="font-size: 11px; color: #64748B; margin: 0;">예측 확률 기준, 장기 베팅 시 수익(EV > 0)을 내기 위한 최소 배당률 조건</p>
                
                <div style="font-size: 12px; color: #E2E8F0; font-weight: bold; margin-bottom: 4px;">
                    📢 이 말이 이기려면 배당률이 최소 {break_even_odds:.1f}배 이상이어야 합니다. (현재 시장 배당: {bh['odds']:.1f}배)
                </div>
                <div style="display: flex; justify-content: space-between; font-size: 12px; margin-top: 4px;">
                    <span style="color: #94A3B8;">현재 시장 배당률</span>
                    <span style="font-weight: 900; color: #F5C842;">{bh['odds']:.1f}배</span>
                </div>
                
                <div style="display: flex; justify-content: space-between; font-size: 12px;">
                    <span style="color: #94A3B8;">AI 손익분기 배당률</span>
                    <span style="font-weight: 900; color: #34D399;">{break_even_odds:.1f}배</span>
                </div>
                
                <div style="margin-top: 8px;">
                    <div style="display: flex; justify-content: space-between; font-size: 9px; color: #64748B; margin-bottom: 2px;">
                        <span>0배</span>
                        <span>손익분기 ({break_even_odds:.1f}배)</span>
                        <span>최대</span>
                    </div>
                    <div class="compare-bar-bg">
                        <div class="compare-bar-fill" style="width: {bar_fill_pct}%;"></div>
                        <div class="compare-bar-marker" style="left: {marker_left_pct}%;"></div>
                    </div>
                </div>
                
                <div style="background-color: rgba(6, 11, 19, 0.6); padding: 10px; border-radius: 6px; border: 1px solid #1B263B; font-size: 11px; font-weight: 600; color: {compare_color}; margin-top: 6px;">
                    {compare_msg}
                </div>
            </div>""".replace("\n", ""), unsafe_allow_html=True)


# ---------------------- Footer ----------------------
st.markdown("""
    <div style="border-top: 1px solid #1B263B; background-color: rgba(13, 27, 42, 0.3); padding: 24px 16px; margin-top: 48px; display: flex; flex-direction: column; md-flex-direction: row; justify-content: space-between; align-items: center; gap: 16px; font-size: 12px; color: #415A77; text-align: center; width: 100%;">
      <p style="margin: 0; padding: 0;">© 2026 KRA AI 경마 의사결정 대시보드. All rights reserved.</p>
      <div style="display: flex; gap: 16px; margin: 0; padding: 0;">
        <span>데이터 소스: 한국마사회(KRA) 3개년 공공데이터</span>
        <span>모형 알고리즘: LightGBM v4.3.0</span>
      </div>
    </div>
""".replace("\n", ""), unsafe_allow_html=True)
