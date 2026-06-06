import numpy as np
import pandas as pd
import scipy.stats as stats

def apply_elo_features_inference(df, elo_state):
    home_elos = []
    away_elos = []
    for _, row in df.iterrows():
        h_r = elo_state.get(row['HomeTeam'], 1500)
        a_r = elo_state.get(row['AwayTeam'], 1500)
        home_elos.append(h_r)
        away_elos.append(a_r)
    df['Home_PreMatch_Elo'] = home_elos
    df['Away_PreMatch_Elo'] = away_elos
    df['Net_Elo_Advantage'] = df['Home_PreMatch_Elo'] - df['Away_PreMatch_Elo']
    return df

def compute_xg_features_inference(df, state, window=5, default_league_xg=1.35):
    if 'xg_history' not in state or not state['xg_history']:
        league_xg = default_league_xg
    else:
        all_recent = [v[0] for h in state['xg_history'].values() for v in h[-window:]]
        league_xg = np.mean(all_recent) if all_recent else default_league_xg

    h_for, h_ag, a_for, a_ag = [], [], [], []
    for _, row in df.iterrows():
        h_hist = state.get('xg_history', {}).get(row['HomeTeam'], [])
        a_hist = state.get('xg_history', {}).get(row['AwayTeam'], [])
        
        h_f = np.mean([x[0] for x in h_hist[-window:]]) if h_hist else league_xg
        h_a = np.mean([x[1] for x in h_hist[-window:]]) if h_hist else league_xg
        a_f = np.mean([x[0] for x in a_hist[-window:]]) if a_hist else league_xg
        a_a = np.mean([x[1] for x in a_hist[-window:]]) if a_hist else league_xg
        
        h_for.append(h_f); h_ag.append(h_a); a_for.append(a_f); a_ag.append(a_a)

    df['Home_Rolling_xG_For'], df['Home_Rolling_xG_Against'] = h_for, h_ag
    df['Away_Rolling_xG_For'], df['Away_Rolling_xG_Against'] = a_for, a_ag
    df['Net_xG_Form'] = df['Home_Rolling_xG_For'] - df['Away_Rolling_xG_For']
    df['Attack_Defence_Mismatch'] = df['Home_Rolling_xG_For'] - df['Away_Rolling_xG_Against']
    df['Reverse_Mismatch'] = df['Away_Rolling_xG_For'] - df['Home_Rolling_xG_Against']
    return df

def compute_variance_inference(df, state, window=5):
    h_ev, h_xv, a_ev, a_xv = [], [], [], []
    for _, row in df.iterrows():
        h_h = state.get('team_history', {}).get(row['HomeTeam'], [])
        a_h = state.get('team_history', {}).get(row['AwayTeam'], [])
        
        h_ev.append(np.var([x[0] for x in h_h[-window:]]) if len(h_h) > 1 else 0.0)
        h_xv.append(np.var([x[1] for x in h_h[-window:]]) if len(h_h) > 1 else 0.0)
        a_ev.append(np.var([x[0] for x in a_h[-window:]]) if len(a_h) > 1 else 0.0)
        a_xv.append(np.var([x[1] for x in a_h[-window:]]) if len(a_h) > 1 else 0.0)

    df['Home_Elo_Var_5pt'], df['Home_xG_Var_5pt'] = h_ev, h_xv
    df['Away_Elo_Var_5pt'], df['Away_xG_Var_5pt'] = a_ev, a_xv
    df['Net_Elo_Variance_Diff'] = df['Home_Elo_Var_5pt'] - df['Away_Elo_Var_5pt']
    df['Net_xG_Variance_Diff'] = df['Home_xG_Var_5pt'] - df['Away_xG_Var_5pt']
    return df
