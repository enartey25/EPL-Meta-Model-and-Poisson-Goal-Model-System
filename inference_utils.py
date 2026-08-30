import numpy as np
import pandas as pd
import scipy.stats as stats

# Standard Elo assigned to newly promoted clubs as established in development notebook (Cell 63)
PROMOTED_START_ELO = 1420.0
DEFAULT_LEAGUE_XG = 1.35

def apply_elo_features_inference(df: pd.DataFrame, elo_state: dict, promoted_default_elo: float = PROMOTED_START_ELO) -> pd.DataFrame:
    """
    Applies pre-match Elo features to an inference dataframe without state mutation.
    Uses PROMOTED_START_ELO (1420) for newly promoted or untracked clubs.
    """
    if df.empty:
        return df

    home_elos = []
    away_elos = []
    for _, row in df.iterrows():
        h_r = float(elo_state.get(row['HomeTeam'], promoted_default_elo))
        a_r = float(elo_state.get(row['AwayTeam'], promoted_default_elo))
        home_elos.append(h_r)
        away_elos.append(a_r)

    df['Home_PreMatch_Elo'] = np.array(home_elos, dtype=np.float64)
    df['Away_PreMatch_Elo'] = np.array(away_elos, dtype=np.float64)
    df['Net_Elo_Advantage'] = df['Home_PreMatch_Elo'] - df['Away_PreMatch_Elo']
    return df

def compute_xg_features_inference(df: pd.DataFrame, state: dict, window: int = 5, default_league_xg: float = DEFAULT_LEAGUE_XG) -> pd.DataFrame:
    """
    Computes leakage-free rolling xG features for upcoming unplayed fixtures
    using the historical state dictionary. Does NOT update the state.
    For newly promoted clubs with no previous match history, defaults cleanly
    to the league baseline rolling xG.
    """
    if df.empty:
        return df

    xg_history = state.get('xg_history', {})
    if not xg_history:
        league_xg = default_league_xg
    else:
        all_recent = [v[0] for h in xg_history.values() for v in h[-window:]]
        league_xg = float(np.mean(all_recent)) if all_recent else default_league_xg

    h_for, h_ag, a_for, a_ag = [], [], [], []
    for _, row in df.iterrows():
        h_hist = xg_history.get(row['HomeTeam'], [])
        a_hist = xg_history.get(row['AwayTeam'], [])

        h_slice = h_hist[-window:]
        if h_slice:
            h_f = float(np.mean([x[0] for x in h_slice]))
            h_a = float(np.mean([x[1] for x in h_slice]))
        else:
            h_f = league_xg
            h_a = league_xg

        a_slice = a_hist[-window:]
        if a_slice:
            a_f = float(np.mean([x[0] for x in a_slice]))
            a_a = float(np.mean([x[1] for x in a_slice]))
        else:
            a_f = league_xg
            a_a = league_xg

        h_for.append(h_f)
        h_ag.append(h_a)
        a_for.append(a_f)
        a_ag.append(a_a)

    df['Home_Rolling_xG_For'] = np.array(h_for, dtype=np.float64)
    df['Home_Rolling_xG_Against'] = np.array(h_ag, dtype=np.float64)
    df['Away_Rolling_xG_For'] = np.array(a_for, dtype=np.float64)
    df['Away_Rolling_xG_Against'] = np.array(a_ag, dtype=np.float64)

    df['Net_xG_Form'] = df['Home_Rolling_xG_For'] - df['Away_Rolling_xG_For']
    df['Attack_Defence_Mismatch'] = df['Home_Rolling_xG_For'] - df['Away_Rolling_xG_Against']
    df['Reverse_Mismatch'] = df['Away_Rolling_xG_For'] - df['Home_Rolling_xG_Against']
    return df

def compute_variance_inference(df: pd.DataFrame, state: dict, window: int = 5) -> pd.DataFrame:
    """
    Computes rolling Elo and xG variance features for upcoming unplayed fixtures.
    For newly promoted or unranked clubs with < 2 matches in history, defaults cleanly to 0.0.
    """
    if df.empty:
        return df

    team_history = state.get('team_history', {})
    h_ev, h_xv, a_ev, a_xv = [], [], [], []

    for _, row in df.iterrows():
        h_h = team_history.get(row['HomeTeam'], [])
        a_h = team_history.get(row['AwayTeam'], [])

        h_slice = h_h[-window:]
        if len(h_slice) > 1:
            h_ev.append(float(np.var([x[0] for x in h_slice])))
            h_xv.append(float(np.var([x[1] for x in h_slice])))
        else:
            h_ev.append(0.0)
            h_xv.append(0.0)

        a_slice = a_h[-window:]
        if len(a_slice) > 1:
            a_ev.append(float(np.var([x[0] for x in a_slice])))
            a_xv.append(float(np.var([x[1] for x in a_slice])))
        else:
            a_ev.append(0.0)
            a_xv.append(0.0)

    df['Home_Elo_Var_5pt'] = np.array(h_ev, dtype=np.float64)
    df['Home_xG_Var_5pt'] = np.array(h_xv, dtype=np.float64)
    df['Away_Elo_Var_5pt'] = np.array(a_ev, dtype=np.float64)
    df['Away_xG_Var_5pt'] = np.array(a_xv, dtype=np.float64)

    df['Net_Elo_Variance_Diff'] = df['Home_Elo_Var_5pt'] - df['Away_Elo_Var_5pt']
    df['Net_xG_Variance_Diff'] = df['Home_xG_Var_5pt'] - df['Away_xG_Var_5pt']
    return df
