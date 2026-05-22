def get_feature_columns():
    return [
        'rel_top3rate', 'rel_avg_rank', 'rel_avg_gap', 'rank_in_race_top3rate',
        'rel_jockey_rate', 'rel_trainer_rate', 'rel_synergy',
        'rel_win_price', 'win_price_rank_in_race', 'is_favorite',
        'rel_burden_weight', 'rel_horse_age',
        'is_debut', 'is_peak_condition', 'jockey_trainer_synergy',
        'form_x_dist', 'peak_form_index', 'dark_horse_score',
        'gate_group', 'gate_x_dist', 'is_wet_track',
        'top3_rate_last_5', 'avg_rank_last_3', 'avg_gap_last_3',
        'jockey_recent_top3_rate', 'trainer_recent_top3_rate',
        'top3_rate_same_dist', 'pthrBurdWgt', 'pthrAg',
        'horse_weight_diff_calc', 'days_since_last_race',
        'fe_horse_race_count', 'rsutWinPrice', 'pthrGtno',
        'dist_num'
    ]

def add_synergy_features(df):
    return df
