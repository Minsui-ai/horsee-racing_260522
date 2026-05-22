from lightgbm import LGBMRegressor
import numpy as np
import pandas as pd

def train_regression(train, val, features):
    X_train = train[features].copy()
    y_train = train['race_time_sec'].copy()
    X_val = val[features].copy()
    y_val = val['race_time_sec'].copy()
    
    # 타겟 결측치 제외
    train_mask = y_train.notna()
    X_train = X_train[train_mask]
    y_train = y_train[train_mask]
    
    val_mask = y_val.notna()
    X_val = X_val[val_mask]
    y_val = y_val[val_mask]
    
    # 결측치 중앙값 대체
    for col in features:
        median_val = X_train[col].median()
        X_train[col] = X_train[col].fillna(median_val)
        X_val[col] = X_val[col].fillna(median_val)
        
    model = LGBMRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], callbacks=[])
    return model

def evaluate_hit_rate(df):
    grouped = df.groupby(['schdRaceDt', 'schdRaceNo'])
    total_races = 0
    top1_hits = 0
    top3_hits = 0
    
    for name, group in grouped:
        if len(group) < 2:
            continue
        group_clean = group.dropna(subset=['race_time_sec', 'pred_time'])
        if len(group_clean) < 2:
            continue
            
        total_races += 1
        actual_winner_idx = group_clean['race_time_sec'].idxmin()
        pred_winner_idx = group_clean['pred_time'].idxmin()
        
        # 실제 순위 계산
        actual_ranks = group_clean['race_time_sec'].rank(method='min')
        
        if pred_winner_idx == actual_winner_idx:
            top1_hits += 1
        if actual_ranks.loc[pred_winner_idx] <= 3:
            top3_hits += 1
            
    hit_rate_top1 = top1_hits / total_races if total_races > 0 else 0
    hit_rate_top3 = top3_hits / total_races if total_races > 0 else 0
    return hit_rate_top1, hit_rate_top3
