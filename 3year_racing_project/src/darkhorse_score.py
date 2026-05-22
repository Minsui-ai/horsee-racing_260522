import numpy as np

def calculate_darkhorse_score(df):
    df = df.copy()
    
    # edge 계산: 예측 확률 - (1 / 배당률)
    # 시장 확률 대비 예측 확률의 우위
    df['edge'] = df['p_top3'] - (1.0 / df['rsutWinPrice'].replace(0, np.nan).fillna(100.0))
    
    # 다크호스 스코어: 예측확률 * 배당률
    df['darkhorse_score'] = df['p_top3'] * df['rsutWinPrice']
    
    # 다크호스 후보 여부: 배당률이 10배 이상이면서 예측확률이 0.15 이상인 경우 등
    # app.py 121행에서 'is_darkhorse_candidate' 컬럼을 참고한다.
    df['is_darkhorse_candidate'] = (df['rsutWinPrice'] >= 10.0) & (df['p_top3'] >= 0.15)
    
    return df
