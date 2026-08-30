"""
Feature engineering module for the EPL Match Predictor pipeline.
Reconstructs chronological Elo, rolling xG, and variance features with zero lookahead bias.
"""
import logging
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

FINAL_FEATURES = [
    "Net_Elo_Advantage",
    "HomeTeam",
    "AwayTeam",
    "Net_Elo_Variance_Diff",
    "Net_xG_Form",
    "Net_xG_Variance_Diff",
    "Attack_Defence_Mismatch",
    "Reverse_Mismatch",
    "Home_Rolling_xG_For",
    "Home_Rolling_xG_Against",
    "Away_Rolling_xG_For",
    "Away_Rolling_xG_Against",
    "Home_PreMatch_Elo",
    "Away_PreMatch_Elo",
    "Home_Elo_Var_5pt",
    "Away_Elo_Var_5pt",
]

PROMOTED_START_ELO = 1420.0
INITIAL_ELO = 1500.0
K_FACTOR = 32.0
HOME_ADVANTAGE = 50.0
REGRESSION_FACTOR = 0.33


def compute_elo_features(df: pd.DataFrame, state: dict) -> tuple[pd.DataFrame, dict]:
    """Sequential Elo rating updates match by match with season regression."""
    if df.empty:
        return df, state

    if "elo" not in state:
        state["elo"] = {}
    current_elo = state["elo"]

    home_teams = df["HomeTeam"].to_numpy()
    away_teams = df["AwayTeam"].to_numpy()
    results = df["FTR"].to_numpy()
    seasons = df["season"].to_numpy() if "season" in df.columns else np.array(["2022/23"] * len(df))

    n = len(df)
    home_pre_elo = np.empty(n, dtype=np.float64)
    away_pre_elo = np.empty(n, dtype=np.float64)
    last_season = None

    for i in range(n):
        current_season = seasons[i]
        if last_season is not None and current_season != last_season:
            # Regress towards mean between seasons
            for t in list(current_elo.keys()):
                current_elo[t] = (1.0 - REGRESSION_FACTOR) * current_elo[t] + REGRESSION_FACTOR * INITIAL_ELO
        last_season = current_season

        h_team = home_teams[i]
        a_team = away_teams[i]

        r_home = current_elo.get(h_team, PROMOTED_START_ELO)
        r_away = current_elo.get(a_team, PROMOTED_START_ELO)

        home_pre_elo[i] = r_home
        away_pre_elo[i] = r_away

        # Expected score with home advantage
        e_home = 1.0 / (1.0 + 10.0 ** ((r_away - (r_home + HOME_ADVANTAGE)) / 400.0))
        e_away = 1.0 - e_home

        res = results[i]
        if res == "H" or res == 2 or res == "HomeWin":
            s_home, s_away = 1.0, 0.0
        elif res == "A" or res == 1 or res == "AwayWin":
            s_home, s_away = 0.0, 1.0
        else:
            s_home, s_away = 0.5, 0.5

        current_elo[h_team] = r_home + K_FACTOR * (s_home - e_home)
        current_elo[a_team] = r_away + K_FACTOR * (s_away - e_away)

    df = df.copy()
    df["Home_PreMatch_Elo"] = home_pre_elo
    df["Away_PreMatch_Elo"] = away_pre_elo
    df["Net_Elo_Advantage"] = home_pre_elo - away_pre_elo
    return df, state


def compute_xg_features(df: pd.DataFrame, state: dict, window: int = 5) -> tuple[pd.DataFrame, dict]:
    """Sequential rolling xG computation (5-match window)."""
    if df.empty:
        return df, state

    if "xg_history" not in state:
        state["xg_history"] = {}
    xg_history = state["xg_history"]

    league_xg = (df["home_xG"].mean() + df["away_xG"].mean()) / 2.0
    home_teams = df["HomeTeam"].to_numpy()
    away_teams = df["AwayTeam"].to_numpy()
    home_xGs = df["home_xG"].to_numpy()
    away_xGs = df["away_xG"].to_numpy()

    n = len(df)
    h_roll_for = np.empty(n, dtype=np.float64)
    h_roll_against = np.empty(n, dtype=np.float64)
    a_roll_for = np.empty(n, dtype=np.float64)
    a_roll_against = np.empty(n, dtype=np.float64)

    for i in range(n):
        h_team = home_teams[i]
        a_team = away_teams[i]

        h_hist_for, h_hist_against = xg_history.get(h_team, ([], []))
        a_hist_for, a_hist_against = xg_history.get(a_team, ([], []))

        h_roll_for[i] = np.mean(h_hist_for[-window:]) if h_hist_for else league_xg
        h_roll_against[i] = np.mean(h_hist_against[-window:]) if h_hist_against else league_xg
        a_roll_for[i] = np.mean(a_hist_for[-window:]) if a_hist_for else league_xg
        a_roll_against[i] = np.mean(a_hist_against[-window:]) if a_hist_against else league_xg

        # Update historical queues
        if h_team not in xg_history:
            xg_history[h_team] = ([home_xGs[i]], [away_xGs[i]])
        else:
            h_hist_for.append(home_xGs[i])
            h_hist_against.append(away_xGs[i])

        if a_team not in xg_history:
            xg_history[a_team] = ([away_xGs[i]], [home_xGs[i]])
        else:
            a_hist_for.append(away_xGs[i])
            a_hist_against.append(home_xGs[i])

    df = df.copy()
    df["Home_Rolling_xG_For"] = h_roll_for
    df["Home_Rolling_xG_Against"] = h_roll_against
    df["Away_Rolling_xG_For"] = a_roll_for
    df["Away_Rolling_xG_Against"] = a_roll_against

    df["Net_xG_Form"] = (h_roll_for - h_roll_against) - (a_roll_for - a_roll_against)
    df["Attack_Defence_Mismatch"] = h_roll_for - a_roll_against
    df["Reverse_Mismatch"] = a_roll_for - h_roll_against
    return df, state


def compute_variance(df: pd.DataFrame, state: dict, window: int = 5) -> tuple[pd.DataFrame, dict]:
    """Sequential volatility/variance metrics over rolling match windows."""
    if df.empty:
        return df, state

    if "team_history" not in state:
        state["team_history"] = {}
    team_history = state["team_history"]

    home_teams = df["HomeTeam"].to_numpy()
    away_teams = df["AwayTeam"].to_numpy()
    h_elos = df["Home_PreMatch_Elo"].to_numpy()
    h_xgs = df["Home_Rolling_xG_For"].to_numpy()
    a_elos = df["Away_PreMatch_Elo"].to_numpy()
    a_xgs = df["Away_Rolling_xG_For"].to_numpy()

    n = len(df)
    h_elo_var = np.zeros(n, dtype=np.float64)
    h_xg_var = np.zeros(n, dtype=np.float64)
    a_elo_var = np.zeros(n, dtype=np.float64)
    a_xg_var = np.zeros(n, dtype=np.float64)

    for i in range(n):
        h_team = home_teams[i]
        a_team = away_teams[i]

        h_hist = team_history.get(h_team, [])
        a_hist = team_history.get(a_team, [])

        if len(h_hist) >= 2:
            sub = h_hist[-window:]
            h_elo_var[i] = np.var([x[0] for x in sub])
            h_xg_var[i] = np.var([x[1] for x in sub])

        if len(a_hist) >= 2:
            sub = a_hist[-window:]
            a_elo_var[i] = np.var([x[0] for x in sub])
            a_xg_var[i] = np.var([x[1] for x in sub])

        # Append state
        if h_team not in team_history:
            team_history[h_team] = [(h_elos[i], h_xgs[i])]
        else:
            team_history[h_team].append((h_elos[i], h_xgs[i]))

        if a_team not in team_history:
            team_history[a_team] = [(a_elos[i], a_xgs[i])]
        else:
            team_history[a_team].append((a_elos[i], a_xgs[i]))

    df = df.copy()
    df["Home_Elo_Var_5pt"] = h_elo_var
    df["Home_xG_Var_5pt"] = h_xg_var
    df["Away_Elo_Var_5pt"] = a_elo_var
    df["Away_xG_Var_5pt"] = a_xg_var

    df["Net_Elo_Variance_Diff"] = h_elo_var - a_elo_var
    df["Net_xG_Variance_Diff"] = h_xg_var - a_xg_var
    return df, state


def build_pipeline_features(df: pd.DataFrame) -> tuple[pd.DataFrame, dict, dict]:
    """Execute end-to-end sequential feature generation and categorical encoding."""
    state = {}
    df = df.sort_values("Date").reset_index(drop=True)

    # 1. Build canonical team map
    all_teams = sorted(list(set(df["HomeTeam"].unique()).union(set(df["AwayTeam"].unique()))))
    team_map = {team: idx for idx, team in enumerate(all_teams)}

    # 2. Sequential feature generation
    df, state = compute_elo_features(df, state)
    df, state = compute_xg_features(df, state)
    df, state = compute_variance(df, state)

    # 3. Integer encoding for tree models
    df["HomeTeam_orig"] = df["HomeTeam"]
    df["AwayTeam_orig"] = df["AwayTeam"]
    df["HomeTeam"] = df["HomeTeam"].map(team_map).fillna(0).astype(int)
    df["AwayTeam"] = df["AwayTeam"].map(team_map).fillna(0).astype(int)

    # 4. Map FTR target: Draw=0, AwayWin=1, HomeWin=2
    label_map = {0: "Draw", 1: "AwayWin", 2: "HomeWin"}
    ftr_map = {"D": 0, "A": 1, "H": 2, "Draw": 0, "AwayWin": 1, "HomeWin": 2}
    df["FTR_label"] = df["FTR"].map(ftr_map)

    logger.info(f"Features built successfully for {len(df)} fixtures across {len(team_map)} clubs.")
    return df, state, team_map
