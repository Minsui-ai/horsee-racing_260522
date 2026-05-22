import numpy as np
import pandas as pd

def run_roi_simulation(df):
    df = df.copy()
    
    races = df.groupby(['schdRaceDt', 'schdRaceNo'])
    
    cum_1 = [0.0]
    cum_3 = [0.0]
    cum_rnd = [0.0]
    
    hit_1 = 0
    hit_3 = 0
    hit_rnd = 0
    total_bets = 0
    
    for name, group in races:
        if len(group) == 0:
            continue
        total_bets += 1
        
        # 실제 우승마(is_top3 == 1)
        winners = group[group['is_top3'] == 1]
        
        # 전략 1: Top 1 Darkhorse
        top1_horse = group.sort_values('darkhorse_score', ascending=False).iloc[0]
        bet_amt = 10000
        return_amt = bet_amt * top1_horse['rsutWinPrice'] if top1_horse['is_top3'] == 1 else 0.0
        cum_1.append(cum_1[-1] - bet_amt + return_amt)
        if top1_horse['is_top3'] == 1:
            hit_1 += 1
            
        # 전략 2: Top 3 Darkhorse
        top3_horses = group.sort_values('darkhorse_score', ascending=False).head(3)
        bet_amt_3 = 10000
        return_amt_3 = sum((bet_amt_3 / 3) * row['rsutWinPrice'] if row['is_top3'] == 1 else 0.0 for idx, row in top3_horses.iterrows())
        cum_3.append(cum_3[-1] - bet_amt_3 + return_amt_3)
        if any(top3_horses['is_top3'] == 1):
            hit_3 += 1
            
        # 전략 3: Random
        rnd_horse = group.sample(1).iloc[0]
        bet_amt_rnd = 10000
        return_amt_rnd = bet_amt_rnd * rnd_horse['rsutWinPrice'] if rnd_horse['is_top3'] == 1 else 0.0
        cum_rnd.append(cum_rnd[-1] - bet_amt_rnd + return_amt_rnd)
        if rnd_horse['is_top3'] == 1:
            hit_rnd += 1
            
    # ROI 계산
    roi_1 = (cum_1[-1] / (total_bets * 10000)) * 100 if total_bets > 0 else 0
    roi_3 = (cum_3[-1] / (total_bets * 10000)) * 100 if total_bets > 0 else 0
    roi_rnd = (cum_rnd[-1] / (total_bets * 10000)) * 100 if total_bets > 0 else 0
    
    # MDD 계산
    def get_mdd(cum_profit):
        cum_profit = np.array(cum_profit)
        peaks = np.maximum.accumulate(cum_profit)
        drawdowns = peaks - cum_profit
        return np.max(drawdowns) if len(drawdowns) > 0 else 0.0
        
    mdd_1 = get_mdd(cum_1)
    mdd_3 = get_mdd(cum_3)
    mdd_rnd = get_mdd(cum_rnd)
    
    stats = {
        'Top1 Darkhorse': (hit_1 / total_bets * 100 if total_bets > 0 else 0, roi_1, mdd_1),
        'Top3 Darkhorse': (hit_3 / total_bets * 100 if total_bets > 0 else 0, roi_3, mdd_3),
        'Random (Baseline)': (hit_rnd / total_bets * 100 if total_bets > 0 else 0, roi_rnd, mdd_rnd)
    }
    
    cum_data = [cum_1, cum_3, cum_rnd]
    return df, stats, cum_data
