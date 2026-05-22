from lightgbm import LGBMClassifier
from imblearn.over_sampling import SMOTE
import pandas as pd

def train_classification(train, val, features):
    X_train = train[features].copy()
    y_train = train['is_top3'].copy()
    X_val = val[features].copy()
    y_val = val['is_top3'].copy()
    
    # 결측치 중앙값 대체
    for col in features:
        median_val = X_train[col].median()
        X_train[col] = X_train[col].fillna(median_val)
        X_val[col] = X_val[col].fillna(median_val)
        
    # SMOTE
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
    
    model = LGBMClassifier(n_estimators=100, random_state=42, n_jobs=-1, class_weight='balanced')
    model.fit(X_train_res, y_train_res)
    return model
