import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

def parse_time_to_sec(val):
    if pd.isna(val) or not isinstance(val, str):
        return np.nan
    parts = val.split(':')
    try:
        if len(parts) == 3:
            return float(parts[0]) * 3600 + float(parts[1]) * 60 + float(parts[2])
        elif len(parts) == 2:
            return float(parts[0]) * 60 + float(parts[1])
        else:
            return float(val)
    except ValueError:
        return np.nan

def get_condition_warning(row):
    if row.get('is_debut', 0) == 1:
        return "첫출전"
    weight_diff = row.get('horse_weight_diff_calc', 0)
    days = row.get('days_since_last_race', np.nan)
    if pd.isna(days) or pd.isna(weight_diff):
        return "보통"
    if (0 <= weight_diff <= 5) and (21 <= days <= 42):
        return "최적"
    elif abs(weight_diff) >= 15 or days > 100 or days < 14:
        return "주의"
    else:
        return "보통"

def preprocess_data(df):
    df = df.copy()
    
    # 1. 신예마 여부 생성
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
    df['dist_num'] = df['cndRaceDs'].str.extract(r'(\d+)').astype(float)
    df['race_dist_m'] = df['dist_num']
    df['gate_group'] = pd.cut(df['pthrGtno'], bins=[0, 4, 8, 99], labels=['inner', 'middle', 'outer']).astype(str)
    df['gate_x_dist'] = df['pthrGtno'] * df['dist_num']
    df['is_wet_track'] = df['rsutTrckStus'].str.contains('다습|포화|불량').fillna(False).astype(int)
    df['wet_track_dist'] = df['is_wet_track'] * df['dist_num']
    
    # 함수율(track_moisture_pct) 추출 (예: '양호 (6%)' -> 6.0)
    df['track_moisture_pct'] = df['rsutTrckStus'].str.extract(r'(\d+)').astype(float)
    df['track_moisture_pct'] = df['track_moisture_pct'].fillna(8.0) # 기본값
    
    # 시간 변환
    df['race_time_sec'] = df['rsutRaceRcd'].apply(parse_time_to_sec)
    
    # 컨디션 사이클
    df['is_peak_condition'] = (
        (df['horse_weight_diff_calc'] >= 0) & (df['horse_weight_diff_calc'] <= 5) &
        (df['days_since_last_race'] >= 21) & (df['days_since_last_race'] <= 42)
    ).astype(int)
    
    # 컨디션 경고
    df['condition_warning'] = df.apply(get_condition_warning, axis=1)

    df['jockey_trainer_synergy'] = df['jockey_recent_top3_rate'] * df['trainer_recent_top3_rate']
    df['form_x_dist'] = df['top3_rate_last_5'] * df['top3_rate_same_dist']
    df['peak_form_index'] = 1 / ((df['avg_rank_last_3'] * df['avg_gap_last_3']) + 1).fillna(1)
    df['dark_horse_score'] = (df['rsutWinPrice'] / 10.0) * df['top3_rate_last_5']

    # 상대 비교 피처 생성
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

    df['schdRaceDt'] = pd.to_datetime(df['schdRaceDt'], format='%Y.%m.%d')
    
    le = LabelEncoder()
    df['gate_group'] = le.fit_transform(df['gate_group'].astype(str))
    
    return df

def split_data(df):
    # 시계열 분할
    train_mask = df['schdRaceDt'] < '2024-07-01'
    test_mask = df['schdRaceDt'] >= '2024-07-01'
    
    train_full = df[train_mask].copy()
    test = df[test_mask].copy()
    
    train_full = train_full.sort_values('schdRaceDt')
    val_size = int(len(train_full) * 0.15)
    train = train_full.iloc[:-val_size].copy()
    val = train_full.iloc[-val_size:].copy()
    
    # 지정된 결측치 처리 기준 반영: avg_rank_last_3, avg_gap_last_3 → is_debut=1 그룹 평균 대체
    mean_rank_debut = train.loc[train['is_debut'] == 0, 'avg_rank_last_3'].mean()
    mean_gap_debut = train.loc[train['is_debut'] == 0, 'avg_gap_last_3'].mean()

    train['avg_rank_last_3'] = train['avg_rank_last_3'].fillna(mean_rank_debut)
    train['avg_gap_last_3'] = train['avg_gap_last_3'].fillna(mean_gap_debut)
    val['avg_rank_last_3'] = val['avg_rank_last_3'].fillna(mean_rank_debut)
    val['avg_gap_last_3'] = val['avg_gap_last_3'].fillna(mean_gap_debut)
    test['avg_rank_last_3'] = test['avg_rank_last_3'].fillna(mean_rank_debut)
    test['avg_gap_last_3'] = test['avg_gap_last_3'].fillna(mean_gap_debut)

    return train, val, test
