# 🐎 경마 다크호스 예측 Streamlit 대시보드

이 프로젝트는 서울 경마장 3개년 데이터를 분석하여, 시장 배당률 대비 높은 입상 가능성을 가진 '다크호스'를 식별하고 베팅 전략의 ROI를 시뮬레이션하는 전문 대시보드입니다.

## 📁 폴더 구조
```
darkhorse_dashboard/
├── data/
│   └── race_results_seoul_3years_preprocessed_민수정.csv  # 전처리된 원본 데이터
├── src/
│   ├── preprocessing.py      # 데이터 클리닝 및 시간 변환
│   ├── feature_engineering.py # 모델 피처 정의
│   ├── train_regression.py   # 시간 예측 회귀 모델
│   ├── train_classification.py # 입상 확률 분류 모델
│   ├── darkhorse_score.py    # 다크호스 점수 산출 로직
│   ├── roi_backtest.py       # ROI 시뮬레이션
│   └── utils.py              # 한글 폰트 및 공통 유틸
├── models/                   # 학습된 모델 파일 (.pkl)
├── reports/                  # 평가 결과 리포트
├── app.py                    # Streamlit 엔트리포인트
├── requirements.txt          # 필요 라이브러리
└── README.md                 # 프로젝트 문서
```

## 🚀 실행 방법
1. **의존성 설치**:
   ```bash
   pip install -r requirements.txt
   ```
2. **대시보드 실행**:
   ```bash
   streamlit run app.py
   ```
   *최초 실행 시 모델 학습이 자동으로 진행됩니다.*

## 💡 주요 구현 및 멘토 피드백 반영
- **회귀 모델 기반 순위 산출**: 경주 시간을 예측하고 이를 정렬하여 예상 순위를 산출하는 구조를 충실히 구현했습니다.
- **데이터 분할 전략**: 최신 10% 데이터를 Test 셋으로 고정하고, 나머지를 Train/Val로 나누어 시계열 누수를 방지했습니다.
- **다크호스 스코어링**: 단순 확률이 아닌, 시장 확률과의 차이(Edge)와 개별 마필의 컨디션 적합도(Condition Fit)를 결합한 점수를 도입했습니다.
- **ROI 시뮬레이션**: 실제 베팅 시의 수익률과 최대 낙폭(MDD)을 산출하여 모델의 실전 가치를 검증합니다.
- **한글화 및 시각화**: `koreanize-matplotlib`를 사용하여 한글 깨짐 문제를 해결했으며, Plotly를 활용한 인터랙티브 차트를 제공합니다.

---
**Senior ML + Data Visualization Engineer**
**Antigravity AI Assistant**
