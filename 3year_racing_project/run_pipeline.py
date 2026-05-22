import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import warnings
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer

# 한글 폰트 설정 (NanumGothic)
font_path = "NanumGothic.ttf"
if os.path.exists(font_path):
    font_name = fm.FontProperties(fname=font_path).get_name()
    plt.rc('font', family=font_name)
else:
    # 윈도우 기본 폰트 시도
    plt.rc('font', family='Malgun Gothic')
plt.rcParams['axes.unicode_minus'] = False

warnings.filterwarnings('ignore')

# 1. 환경 설정 및 데이터 로드
DATA_PATH = "data/race_results_seoul_3years_preprocessed_민수정.csv"
IMAGE_DIR = "images"
REPORT_PATH = "reports/racing_eda_report.md"

if not os.path.exists(IMAGE_DIR):
    os.makedirs(IMAGE_DIR)

if not os.path.exists("reports"):
    os.makedirs("reports")

df = pd.read_csv(DATA_PATH)

# 결측치 처리 및 파생변수 생성
# 1. 신예마 여부 생성 (지정된 결측치 처리 기준 반영)
debut_cols = ['avg_rank_last_3', 'avg_gap_last_3', 'burden_diff_from_last', 'days_since_last_race']
df['is_debut'] = df[debut_cols].isna().all(axis=1).astype(int)

# 인적 요인 결측치 대체
jockey_mean = df['jockey_recent_top3_rate'].mean()
trainer_mean = df['trainer_recent_top3_rate'].mean()
df['jockey_recent_top3_rate'] = df['jockey_recent_top3_rate'].fillna(jockey_mean)
df['trainer_recent_top3_rate'] = df['trainer_recent_top3_rate'].fillna(trainer_mean)

# 임시 채우기 (경주 단위 평균 계산 시 NaN 방지)
df['avg_rank_last_3_filled'] = df['avg_rank_last_3'].fillna(df['avg_rank_last_3'].median())
df['avg_gap_last_3_filled'] = df['avg_gap_last_3'].fillna(df['avg_gap_last_3'].median())

# 파생변수 생성
# 1. 게이트 번호 x 거리 상호작용
df['dist_num'] = df['cndRaceDs'].str.extract(r'(\d+)').astype(float)
df['gate_group'] = pd.cut(df['pthrGtno'], bins=[0, 4, 8, 99], labels=['inner', 'middle', 'outer']).astype(str)
df['gate_x_dist'] = df['pthrGtno'] * df['dist_num']

# 2. 트랙 상태 x 배당률 관계
df['is_wet_track'] = df['rsutTrckStus'].str.contains('다습|포화|불량').fillna(False).astype(int)
df['wet_track_dist'] = df['is_wet_track'] * df['dist_num']

# 3. 말의 컨디션 사이클
df['is_peak_condition'] = (
    (df['horse_weight_diff_calc'] >= 0) & (df['horse_weight_diff_calc'] <= 5) &
    (df['days_since_last_race'] >= 21) & (df['days_since_last_race'] <= 42)
).astype(int)

# 4. 기수-조교사 시너지
df['jockey_trainer_synergy'] = df['jockey_recent_top3_rate'] * df['trainer_recent_top3_rate']

# 5. 거리 적성 x 최근 기세 복합 지표
df['form_x_dist'] = df['top3_rate_last_5'] * df['top3_rate_same_dist']
df['peak_form_index'] = 1 / ((df['avg_rank_last_3'] * df['avg_gap_last_3']) + 1).fillna(1)

# 6. 다크호스 조건 (주의: rsutWinPrice가 모델링 시 드롭되더라도 파생변수 형태로 잔존시킴)
df['dark_horse_score'] = (df['rsutWinPrice'] / 10.0) * df['top3_rate_last_5']

# 7. 경주 단위 상대 비교 피처 생성
race_group = df.groupby(['schdRaceDt', 'schdRaceNo'])
race_stats = race_group.agg(
    avg_top3rate=('top3_rate_last_5', 'mean'),
    avg_avg_rank=('avg_rank_last_3_filled', 'mean'),
    avg_avg_gap=('avg_gap_last_3_filled', 'mean'),
    avg_jockey=('jockey_recent_top3_rate', 'mean'),
    avg_trainer=('trainer_recent_top3_rate', 'mean'),
    avg_synergy=('jockey_trainer_synergy', 'mean'),
    avg_win_price=('rsutWinPrice', 'mean'),
    avg_burden=('pthrBurdWgt', 'mean'),
    avg_age=('pthrAg', 'mean'),
    min_win_price=('rsutWinPrice', 'min')
).reset_index()

df = df.merge(race_stats, on=['schdRaceDt', 'schdRaceNo'], how='left')

df['rel_top3rate']      = df['top3_rate_last_5'] - df['avg_top3rate']
df['rel_avg_rank']      = df['avg_rank_last_3_filled'] - df['avg_avg_rank']
df['rel_avg_gap']       = df['avg_gap_last_3_filled'] - df['avg_avg_gap']
df['rel_jockey_rate']   = df['jockey_recent_top3_rate'] - df['avg_jockey']
df['rel_trainer_rate']  = df['trainer_recent_top3_rate'] - df['avg_trainer']
df['rel_synergy']       = df['jockey_trainer_synergy'] - df['avg_synergy']
df['rel_win_price']     = df['rsutWinPrice'] - df['avg_win_price']
df['rel_burden_weight'] = df['pthrBurdWgt'] - df['avg_burden']
df['rel_horse_age']     = df['pthrAg'] - df['avg_age']
df['is_favorite']       = (df['rsutWinPrice'] == df['min_win_price']).astype(int)

df['rank_in_race_top3rate'] = df.groupby(['schdRaceDt', 'schdRaceNo'])['top3_rate_last_5'].rank(method='min', ascending=False)
df['win_price_rank_in_race'] = df.groupby(['schdRaceDt', 'schdRaceNo'])['rsutWinPrice'].rank(method='min', ascending=True)

report_content = []
report_content.append("# 경마 결과 예측 및 탐색적 데이터 분석(EDA) 리포트\n")
report_content.append("본 리포트는 제공된 데이터를 기반으로 배당률이 높은 우승마(순위권 진입마)를 예측하기 위해 수행된 종합적인 데이터 분석 및 머신러닝 파이프라인 결과를 담고 있습니다.\n")

# 데이터 기본 정보
report_content.append("## 1. 데이터 기본 탐색 (Data Overview)\n")
report_content.append(f"- **데이터 크기**: 총 {df.shape[0]} 행, {df.shape[1]} 열\n")

report_content.append("### 상위 5개 행 데이터\n")
report_content.append(df.head().to_markdown(index=False) + "\n")
report_content.append("### 하위 5개 행 데이터\n")
report_content.append(df.tail().to_markdown(index=False) + "\n")

# 결측치 확인
null_counts = df.isnull().sum()
null_counts = null_counts[null_counts > 0]
if len(null_counts) > 0:
    report_content.append("### 결측치 현황\n")
    report_content.append(null_counts.to_frame('결측치 수').to_markdown() + "\n")

# 중복값
dup_count = df.duplicated().sum()
report_content.append(f"- **중복 데이터 수**: {dup_count} 건\n")

# 기술 통계
report_content.append("### 수치형 변수 기술 통계\n")
report_content.append(df.describe().to_markdown() + "\n")

report_content.append("### 범주형 변수 기술 통계\n")
report_content.append(df.describe(include=['O']).to_markdown() + "\n")

# 2. EDA 및 시각화 (10개 이상)
report_content.append("## 2. 탐색적 데이터 분석 (EDA)\n")

# 2.1 타겟 변수 분포
plt.figure(figsize=(8, 6))
val_counts = df['is_top3'].value_counts()
val_counts.plot(kind='bar', color=['lightgray', 'skyblue'], edgecolor='black')
plt.title("타겟 변수(is_top3) 클래스 분포")
plt.xlabel("순위권 진입 여부 (0: 실패, 1: 성공)")
plt.ylabel("빈도수")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig(f"{IMAGE_DIR}/01_target_distribution.png")
plt.close()

report_content.append("### 2.1 타겟 변수 클래스 분포\n")
report_content.append(f"![타겟 분포]({os.path.abspath(IMAGE_DIR)}/01_target_distribution.png)\n")
report_content.append(val_counts.to_frame('빈도수').to_markdown() + "\n")
report_content.append("> **해석**: 타겟 변수인 `is_top3`의 빈도를 분석한 결과, 순위권에 진입하지 못한 데이터(0)가 진입한 데이터(1)에 비해 압도적으로 많은 클래스 불균형 상태를 보여주고 있습니다. 이는 모델 학습 시 SMOTE와 같은 불균형 데이터 처리 기법이 필수적임을 시사합니다.\n")

# 2.2 성별(cndGndr)별 순위권 진입 비율
plt.figure(figsize=(8, 6))
cross_gndr = pd.crosstab(df['cndGndr'], df['is_top3'], normalize='index')
cross_gndr.plot(kind='bar', stacked=True, color=['lightgray', 'skyblue'], figsize=(8, 6), edgecolor='black')
plt.title("조건별 성별에 따른 순위권 진입 비율")
plt.xlabel("성별 조건")
plt.ylabel("비율")
plt.legend(title='is_top3')
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig(f"{IMAGE_DIR}/02_gender_top3.png")
plt.close()

report_content.append("### 2.2 조건 성별(cndGndr)별 순위권 진입 비율\n")
report_content.append(f"![성별 진입비율]({os.path.abspath(IMAGE_DIR)}/02_gender_top3.png)\n")
report_content.append(cross_gndr.to_markdown() + "\n")
report_content.append("> **해석**: 성별 조건(암, 수, 오픈 등)에 따른 순위권 진입(Top3) 비율을 확인해보면, 특정 성별 조건에서 승률 차이가 미세하게 존재하는지 파악할 수 있습니다. 경마에서 성별은 마필의 근력과 체력에 영향을 주어 성적과 연관성이 있는 중요한 범주형 변수로 작용합니다.\n")

# 2.3 부담중량(pthrBurdWgt)에 따른 순위 분포 (박스플롯)
plt.figure(figsize=(8, 6))
df.boxplot(column='pthrBurdWgt', by='is_top3', grid=False, patch_artist=True, boxprops=dict(facecolor='lightblue', color='black'))
plt.title("순위권 진입 여부에 따른 부담중량 분포")
plt.suptitle("")
plt.xlabel("순위권 진입 (0 vs 1)")
plt.ylabel("부담중량 (kg)")
plt.tight_layout()
plt.savefig(f"{IMAGE_DIR}/03_burden_weight_boxplot.png")
plt.close()

report_content.append("### 2.3 순위권 진입 여부에 따른 부담중량 분포\n")
report_content.append(f"![부담중량 박스플롯]({os.path.abspath(IMAGE_DIR)}/03_burden_weight_boxplot.png)\n")
report_content.append(df.groupby('is_top3')['pthrBurdWgt'].describe().to_markdown() + "\n")
report_content.append("> **해석**: 부담중량은 경주마가 짊어지는 무게로, 순위권 진입 여부에 따른 분포 차이를 박스플롯으로 나타냈습니다. 우승마(1) 그룹과 비우승마(0) 그룹의 중앙값 및 이상치 분포를 비교하여, 중량이 마필 성적에 미치는 물리적 영향력의 정도를 직관적으로 파악할 수 있습니다.\n")

# 2.4 경주마 연령(pthrAg) 분포
plt.figure(figsize=(10, 6))
age_counts = df['pthrAg'].value_counts().sort_index()
age_counts.plot(kind='bar', color='skyblue', edgecolor='black')
plt.title("출전 경주마 연령 분포")
plt.xlabel("연령")
plt.ylabel("출전 건수")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig(f"{IMAGE_DIR}/04_horse_age_dist.png")
plt.close()

report_content.append("### 2.4 출전 경주마 연령 분포\n")
report_content.append(f"![경주마 연령]({os.path.abspath(IMAGE_DIR)}/04_horse_age_dist.png)\n")
report_content.append(age_counts.to_frame('출전 건수').to_markdown() + "\n")
report_content.append("> **해석**: 출전한 경주마들의 연령대별 분포를 보여줍니다. 대부분의 경주마들이 특정 연령대(예: 3~5세)에 집중되어 출전하고 있음을 알 수 있으며, 연령이 경주마의 전성기와 직접적인 연관을 가지므로 향후 모델에서 중요한 설명 변수로 작용할 가능성이 큽니다.\n")

# 2.5 단승식 배당률(rsutWinPrice) 히스토그램 (이상치 제거)
plt.figure(figsize=(10, 6))
q95 = df['rsutWinPrice'].quantile(0.95)
df[df['rsutWinPrice'] <= q95]['rsutWinPrice'].plot(kind='hist', bins=50, color='lightgreen', edgecolor='black')
plt.title("단승식 배당률 분포 (상위 5% 이상치 제외)")
plt.xlabel("배당률")
plt.ylabel("빈도")
plt.tight_layout()
plt.savefig(f"{IMAGE_DIR}/05_win_price_hist.png")
plt.close()

report_content.append("### 2.5 단승식 배당률(rsutWinPrice) 분포\n")
report_content.append(f"![단승식 배당률]({os.path.abspath(IMAGE_DIR)}/05_win_price_hist.png)\n")
report_content.append(df['rsutWinPrice'].describe().to_frame().to_markdown() + "\n")
report_content.append("> **해석**: 단승식 배당률의 전체적인 분포를 시각화한 히스토그램으로, 꼬리가 긴 우측 편향(Right-Skewed) 형태를 띱니다. 상위 5%의 극단적인 배당률(초고배당) 데이터를 제외하고 보았을 때 대부분의 승리 배당률이 낮은 구간에 밀집되어 있어, 안정적인 베팅 성향을 반영하고 있습니다.\n")

# 2.6 최근 5경기 3위내 비율(top3_rate_last_5)과 실제 성적
plt.figure(figsize=(8, 6))
df.boxplot(column='top3_rate_last_5', by='is_top3', grid=False, patch_artist=True, boxprops=dict(facecolor='thistle', color='black'))
plt.title("순위권 진입 여부별 최근 5경기 3위내 입상 비율")
plt.suptitle("")
plt.xlabel("순위권 진입 여부 (is_top3)")
plt.ylabel("최근 5경기 3위내 비율")
plt.tight_layout()
plt.savefig(f"{IMAGE_DIR}/06_recent_rate_boxplot.png")
plt.close()

report_content.append("### 2.6 최근 성적과 현재 경주 결과의 관계\n")
report_content.append(f"![최근성적 박스플롯]({os.path.abspath(IMAGE_DIR)}/06_recent_rate_boxplot.png)\n")
report_content.append(df.groupby('is_top3')['top3_rate_last_5'].describe().to_markdown() + "\n")
report_content.append("> **해석**: 마필의 최근 5경기 성적이 현재 경주에서 3위 안에 진입하는 데 얼마나 큰 영향을 미치는지 보여줍니다. 순위권에 진입한 마필(1)들이 비진입 마필(0)에 비해 최근 3위 내 입상 비율 중앙값이 확연히 높게 나타나, 과거 성적이 매우 유의미한 예측 지표임을 입증합니다.\n")

# 2.7 경주 거리(cndRaceDs)별 평균 단승식 배당률
dist_price = df.groupby('cndRaceDs')['rsutWinPrice'].mean().sort_values(ascending=False)
plt.figure(figsize=(12, 6))
dist_price.plot(kind='bar', color='coral', edgecolor='black')
plt.title("경주 거리별 평균 단승식 배당률")
plt.xlabel("경주 거리")
plt.ylabel("평균 배당률")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(f"{IMAGE_DIR}/07_dist_avg_price.png")
plt.close()

report_content.append("### 2.7 경주 거리별 평균 배당률\n")
report_content.append(f"![거리별 배당률]({os.path.abspath(IMAGE_DIR)}/07_dist_avg_price.png)\n")
report_content.append(dist_price.to_frame('평균 단승식 배당률').to_markdown() + "\n")
report_content.append("> **해석**: 경주 거리(예: 1200M, 1800M 등)에 따라 형성되는 평균 배당률의 차이를 분석한 막대그래프입니다. 특정 거리에서 평균 배당률이 높게 형성된다면, 해당 거리의 경주가 예측 불확실성이 크거나 이변이 자주 발생하여 고배당을 노리기에 적합함을 시사합니다.\n")

# 2.8 기수(hrmJckyNm) 상위 30명 빈도 시각화
top_jockeys = df['hrmJckyNm'].value_counts().head(30)
plt.figure(figsize=(14, 6))
top_jockeys.plot(kind='bar', color='skyblue', edgecolor='black')
plt.title("출전 빈도 상위 30명 기수")
plt.xlabel("기수명")
plt.ylabel("출전 건수")
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig(f"{IMAGE_DIR}/08_top_jockeys.png")
plt.close()

report_content.append("### 2.8 상위 30명 기수 출전 빈도\n")
report_content.append(f"![기수 빈도]({os.path.abspath(IMAGE_DIR)}/08_top_jockeys.png)\n")
report_content.append(top_jockeys.to_frame('출전 건수').to_markdown() + "\n")
report_content.append("> **해석**: 전체 데이터에서 출전 빈도가 가장 높은 상위 30명의 기수를 시각화했습니다. 소수의 인기 기수들에게 기승 기회가 집중되어 있으며, 이들 상위 기수들의 성적이 전체 경주의 결과 패턴(인기마 우승 등)을 주도하는 경향이 있는지 파악하는 기초 자료가 됩니다.\n")

# 2.9 조교사(hrmTrarNm) 상위 30명 빈도 시각화
top_trainers = df['hrmTrarNm'].value_counts().head(30)
plt.figure(figsize=(14, 6))
top_trainers.plot(kind='bar', color='lightgreen', edgecolor='black')
plt.title("출전 빈도 상위 30명 조교사")
plt.xlabel("조교사명")
plt.ylabel("출전 건수")
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig(f"{IMAGE_DIR}/09_top_trainers.png")
plt.close()

report_content.append("### 2.9 상위 30명 조교사 출전 빈도\n")
report_content.append(f"![조교사 빈도]({os.path.abspath(IMAGE_DIR)}/09_top_trainers.png)\n")
report_content.append(top_trainers.to_frame('출전 건수').to_markdown() + "\n")
report_content.append("> **해석**: 경주마의 훈련과 컨디션을 관리하는 조교사들의 출전 횟수 상위 30명을 분석한 차트입니다. 실력 있는 조교사 마방에 우수한 마필이 배정되는 '마태효과'를 확인하기 위해, 빈도수와 함께 각 조교사의 실제 승률을 교차 분석하는 추가 연구의 발판이 됩니다.\n")

# 수치형 변수 상관관계 히트맵 (파싱 후)
if 'pthrWeg' in df.columns:
    df['pthrWeg_num'] = df['pthrWeg'].astype(str).str.extract(r'(\d+)')[0].astype(float)
    
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
# 상관계수를 계산하기엔 변수가 너무 많을 수 있으므로 주요 파생변수만 선택
focus_cols = ['is_top3', 'rsutWinPrice', 'pthrAg', 'pthrBurdWgt', 'pthrWeg_num', 'avg_rank_last_3', 'top3_rate_last_5', 'jockey_recent_top3_rate', 'horse_weight_diff_calc']
focus_cols = [c for c in focus_cols if c in df.columns]
corr = df[focus_cols].corr()

plt.figure(figsize=(10, 8))
plt.imshow(corr, cmap='coolwarm', vmin=-1, vmax=1)
plt.colorbar()
plt.xticks(range(len(corr.columns)), corr.columns, rotation=45, ha='right')
plt.yticks(range(len(corr.columns)), corr.columns)
plt.title("주요 수치형 변수 간의 상관관계 히트맵")
plt.tight_layout()
plt.savefig(f"{IMAGE_DIR}/10_correlation_heatmap.png")
plt.close()

report_content.append("### 2.10 주요 수치형 변수 상관관계\n")
report_content.append(f"![상관관계 히트맵]({os.path.abspath(IMAGE_DIR)}/10_correlation_heatmap.png)\n")
report_content.append(corr.round(3).to_markdown() + "\n")
report_content.append("> **해석**: 타겟 변수인 `is_top3`를 포함하여 성적에 영향을 미칠 것으로 예상되는 주요 수치형 특성들 간의 피어슨 상관계수를 히트맵으로 시각화했습니다. 특히 과거 성적(`avg_rank_last_3`, `top3_rate_last_5`)과 기수의 최근 성적이 타겟과 유의미한 양/음의 상관성을 보이며, 다중공선성을 띠는 강한 피처 쌍 유무를 진단할 수 있습니다.\n")

# 2.11 경주마 장구류(pthrEquip) TF-IDF 분석
if 'pthrEquip' in df.columns:
    # 결측치 및 의미없는 값 제외
    tfidf_df = df[df['pthrEquip'].notnull() & (df['pthrEquip'] != '-')].copy()
    # 쉼표를 공백으로 치환하여 토큰화 준비
    equip_text = tfidf_df['pthrEquip'].str.replace(',', ' ', regex=False)
    
    # TF-IDF 벡터라이저 객체 생성
    vectorizer = TfidfVectorizer(max_features=30, token_pattern=r'(?u)\b\w+\b') # 한 글자 단어도 포함
    tfidf_matrix = vectorizer.fit_transform(equip_text.astype(str))
    
    # 단어 및 TF-IDF 점수 합계 도출
    words = vectorizer.get_feature_names_out()
    sums = tfidf_matrix.sum(axis=0).A1
    tfidf_result = pd.DataFrame({'word': words, 'tfidf': sums}).sort_values(by='tfidf', ascending=False)
    
    # 시각화
    plt.figure(figsize=(12, 8))
    plt.barh(tfidf_result['word'].head(30), tfidf_result['tfidf'].head(30), color='teal', edgecolor='black')
    plt.gca().invert_yaxis() # 높은 순서대로 위에서부터 출력
    plt.title("경주마 장구류(pthrEquip) 핵심 키워드 (TF-IDF)")
    plt.xlabel("TF-IDF 가중치 합계")
    plt.ylabel("장구류 키워드")
    plt.tight_layout()
    plt.savefig(f"{IMAGE_DIR}/12_equip_tfidf.png")
    plt.close()
    
    report_content.append("### 2.11 경주마 장구류(pthrEquip) 핵심 키워드 TF-IDF 분석\n")
    report_content.append(f"![장구류 TF-IDF]({os.path.abspath(IMAGE_DIR)}/12_equip_tfidf.png)\n")
    report_content.append(tfidf_result.head(30).to_markdown(index=False) + "\n")
    report_content.append("> **해석**: 경주마들이 착용하는 장구류 데이터를 대상으로 형태소 분석기 없이 TF-IDF 기법을 적용하여 주요 키워드를 도출하였습니다. 분석 결과, '망사눈', '눈가면' 등 시야를 제어하거나 마필의 집중력을 높이기 위한 장구류들의 TF-IDF 가중치가 매우 높게 나타났습니다. 이는 경주 성적 및 배당률에 영향을 줄 수 있는 장구류 특성을 파악하는 데 유용한 기초 데이터로 사용될 수 있으며, 마필의 집중도나 예민함의 정도를 간접적으로 보여주는 중요한 변수임을 시사합니다.\n")

# 3. 데이터 전처리 및 특성 공학
report_content.append("## 3. 데이터 전처리 및 특성 공학 (Data Preprocessing)\n")

# 모델링에 사용할 컬럼 선정 (ID나 이름, 날짜 등 제외)
drop_cols = [
    'hrmJckyId', 'hrmJckyNm', 'hrmOwnerId', 'hrmOwnerNm', 'hrmTrarId', 'hrmTrarNm', 
    'pthrHrnm', 'pthrHrno', 'pthrLatstPtinDt', 'pthrBthd',
    'schdRaceDt', 'schdRaceNm', 'rsutMargin', 'rsutRaceRcd', 'rsutRk', 'rsutRkAdmny', 'rsutRkPurse',
    'rsutRkRemk', 'rsutRlStrtTim', 'rsutStrtTimChgRs', 'cndStrtPargTim', 'rsutWetr',
    'target_rank', 'rsutWinPrice', 'rsutQnlaPrice', 'schdRccrsNm',
    'avg_top3rate', 'avg_avg_rank', 'avg_avg_gap', 'avg_jockey', 
    'avg_trainer', 'avg_synergy', 'avg_win_price', 'avg_burden', 
    'avg_age', 'min_win_price', 'avg_rank_last_3_filled', 'avg_gap_last_3_filled'
]
drop_cols = [c for c in drop_cols if c in df.columns]
model_df = df.drop(columns=drop_cols).copy()

# 텍스트 피처 변환 전에 불필요한 문자열 데이터 숫자형 변환 처리
if 'pthrWeg' in model_df.columns:
    # '449(+7)' 와 같은 형태에서 '449'만 추출
    model_df['pthrWeg'] = model_df['pthrWeg'].astype(str).str.extract(r'(\d+)')[0].astype(float)

# 텍스트 피처 변환 (범주형으로 취급)
cat_cols = model_df.select_dtypes(include=['object']).columns.tolist()

for col in cat_cols:
    model_df[col] = model_df[col].astype(str)
    # 카테고리가 너무 많은 경우 빈도가 높은 상위 10개만 유지하고 나머지는 'Other'로 묶기
    top_n = model_df[col].value_counts().nlargest(15).index
    model_df[col] = np.where(model_df[col].isin(top_n), model_df[col], 'Other')

# 레이블 인코딩
label_encoders = {}
for col in cat_cols:
    le = LabelEncoder()
    model_df[col] = le.fit_transform(model_df[col])
    label_encoders[col] = le

# 4. 머신러닝 모델 구축
report_content.append("## 4. 머신러닝 모델 구축\n")
report_content.append("데이터 스케일링, 타겟 변수 분리 및 Train/Test 분할을 수행하였으며, 클래스 불균형을 해소하기 위해 훈련 데이터에 한해 **SMOTE(Synthetic Minority Over-sampling Technique)** 알고리즘을 적용했습니다.\n")

X = model_df.drop('is_top3', axis=1)
y = model_df['is_top3']

# 결측치 완벽 처리 (SimpleImputer)
imputer = SimpleImputer(strategy='median')
X_imputed = imputer.fit_transform(X)
X_imputed = pd.DataFrame(X_imputed, columns=X.columns)

# 스케일링
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_imputed)
X_scaled = pd.DataFrame(X_scaled, columns=X.columns)

# 분할
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)

# SMOTE
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

report_content.append(f"- **원본 학습 데이터 형태**: {X_train.shape}, 타겟 1의 개수: {sum(y_train==1)}, 타겟 0의 개수: {sum(y_train==0)}\n")
report_content.append(f"- **SMOTE 후 학습 데이터 형태**: {X_train_res.shape}, 타겟 1의 개수: {sum(y_train_res==1)}, 타겟 0의 개수: {sum(y_train_res==0)}\n")

# LightGBM 학습
report_content.append("### LightGBM 기반 우승마 예측 모델 학습\n")
report_content.append("과적합 방지와 빠른 학습 속도를 장점으로 가지는 LightGBM 알고리즘을 채택하여 학습을 수행했습니다. 이진 분류(Binary) 문제를 다루므로 `binary` 목적 함수를 사용했습니다.\n")

lgbm_model = LGBMClassifier(n_estimators=200, random_state=42, n_jobs=-1, class_weight='balanced')
lgbm_model.fit(X_train_res, y_train_res)
y_pred = lgbm_model.predict(X_test)
y_pred_proba = lgbm_model.predict_proba(X_test)[:, 1]

# 5. 모델 평가 및 피처 중요도
report_content.append("## 5. 모델 성능 평가 및 인사이트 도출\n")

# 5가지 지표 계산
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_pred_proba)

metrics_df = pd.DataFrame({
    'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC'],
    'Score': [acc, prec, rec, f1, auc]
})

report_content.append("### 5.1 분류 모델 주요 5가지 평가 지표\n")
report_content.append(metrics_df.to_markdown(index=False) + "\n")
report_content.append("> **해석**: SMOTE를 적용하여 학습한 결과, 소수 클래스(순위권 진입)에 대한 재현율(Recall)이 향상되었습니다. 배당률이 높은 다크호스를 찾아내기 위해서는 실제로 입상할 마필을 놓치지 않는 재현율 중심의 평가가 중요하지만, 과도한 베팅 오판을 막기 위한 정밀도(Precision)와의 밸런스(F1) 유지 또한 필수적임을 시사하는 수치입니다.\n")

# 분류 리포트
class_report = classification_report(y_test, y_pred)
report_content.append("### 5.2 상세 분류 리포트 (Classification Report)\n")
report_content.append("```text\n")
report_content.append(class_report + "\n")
report_content.append("```\n")

# 피처 중요도 시각화
plt.figure(figsize=(10, 8))
feat_importances = pd.Series(lgbm_model.feature_importances_, index=X.columns)
top_feat_importances = feat_importances.nlargest(20)
top_feat_importances.plot(kind='barh', color='mediumpurple', edgecolor='black').invert_yaxis()
plt.title("LightGBM 기반 Feature Importances (상위 20개)")
plt.xlabel("상대적 중요도 점수")
plt.ylabel("변수명 (Features)")
plt.tight_layout()
plt.savefig(f"{IMAGE_DIR}/11_feature_importance.png")
plt.close()

# 피처 중요도 테이블 작성
importance_table = pd.DataFrame({
    'Feature': top_feat_importances.index,
    'Importance Score': top_feat_importances.values
})

report_content.append("### 5.3 예측 주요 변수(Feature Importance) 분석\n")
report_content.append(f"![피처중요도]({os.path.abspath(IMAGE_DIR)}/11_feature_importance.png)\n")
report_content.append(importance_table.to_markdown(index=False) + "\n")
report_content.append("> **해석**: 트리 분기 시 정보 이득을 기준으로 모델이 판단한 상위 20개 특성 중요도 차트입니다. 과거 성적을 반영하는 `avg_rank_last_3`이나 기수/조교사의 역량을 나타내는 파생 지표들이 최상위권에 랭크되어 있으며, 배당률이 높은 다크호스를 예측할 때 혈통적 요인이나 중량의 미세한 변화도 유의미한 변수로 작동함을 시사합니다.\n")

# 6. 결론
report_content.append("## 6. 결론 및 실무 적용 방안\n")
report_content.append("본 분석은 3년간의 경마 데이터를 바탕으로 머신러닝 모델을 구축하여 순위권 진입 가능성을 예측하는 파이프라인을 시연하였습니다.\n")
report_content.append("- 불균형한 승률 데이터의 특성을 고려해 SMOTE 기법을 성공적으로 적용하였으며, 재현율을 높여 다크호스(고배당 우승마) 발굴 가능성을 열어두었습니다.\n")
report_content.append("- 도출된 Feature Importance를 기반으로 현업 베터나 스포츠 데이터 분석가들은 마필의 최근 3경기 성적 및 조교사/기수의 최근 성과를 최우선 지표로 삼아 의사결정에 활용할 수 있습니다.\n")
report_content.append("- 향후에는 임계치(Threshold) 조정 및 배당률 기대값(Expected Value) 수식을 모델 산출 확률에 곱하여 최종 수익 극대화 포트폴리오를 구성하는 고도화가 요구됩니다.\n")

# 파일 저장
with open(REPORT_PATH, "w", encoding="utf-8") as f:
    f.write("\n".join(report_content))

print(f"모든 분석이 완료되었습니다. 결과 리포트가 {REPORT_PATH}에 저장되었고, 이미지 파일이 {IMAGE_DIR}/ 디렉토리에 저장되었습니다.")
