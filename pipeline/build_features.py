"""Training-time feature engineering: sequential Elo, rolling xG form and
volatility, computed match by match so every feature is pre-match.

``inference_utils.py`` holds the mirror-image code for unplayed fixtures;
``pipeline/validate.py`` checks the two agree exactly.
"""
import logging

import numpy as np
import pandas as pd

from pipeline import config

logger = logging.getLogger(__name__)


def build_team_map(played: pd.DataFrame, fixtures: pd.DataFrame) -> dict:
    """Codes for every club that has played or is scheduled (promoted clubs
    included before their first match), in a fixed alphabetical order."""
    teams = set(played["HomeTeam"]) | set(played["AwayTeam"]) | set(fixtures["HomeTeam"]) | set(fixtures["AwayTeam"])
    return {t: i for i, t in enumerate(sorted(teams))}


def encode(df: pd.DataFrame, team_map: dict) -> pd.DataFrame:
    df = df.copy()
    cats = list(range(len(team_map)))
    for col in ("HomeTeam", "AwayTeam"):
        codes = df[col].map(team_map)
        if codes.isna().any():
            raise ValueError(f"Unknown teams in {col}: {sorted(df.loc[codes.isna(), col].unique())}")
        df[col] = pd.Categorical(codes.astype(int), categories=cats)
    return df


def compute_elo_features(df, state):
    p = config.ELO_PARAMS
    current_elo = state.setdefault("elo", {})
    home, away = df["HomeTeam"].to_numpy(), df["AwayTeam"].to_numpy()
    ftrs, seasons = df["FTR"].to_numpy(), df["season"].to_numpy()
    n = len(df)
    h_pre, a_pre = np.empty(n), np.empty(n)
    active_season = state.get("last_season", seasons[0] if n else None)

    for i in range(n):
        ht, at = int(home[i]), int(away[i])
        if seasons[i] != active_season:   # summer regression toward the mean
            for t in current_elo:
                current_elo[t] = current_elo[t] * (1 - p["REGRESSION_FACTOR"]) + p["INITIAL_ELO"] * p["REGRESSION_FACTOR"]
            active_season = seasons[i]
        rh = current_elo.get(ht, p["PROMOTED_START_ELO"])
        ra = current_elo.get(at, p["PROMOTED_START_ELO"])
        h_pre[i], a_pre[i] = rh, ra
        exp_home = 1.0 / (1.0 + 10.0 ** ((ra - (rh + p["HOME_ADVANTAGE"])) / 400.0))
        s_home = 1.0 if ftrs[i] == 2 else (0.0 if ftrs[i] == 1 else 0.5)
        current_elo[ht] = rh + p["K_FACTOR"] * (s_home - exp_home)
        current_elo[at] = ra + p["K_FACTOR"] * ((1.0 - s_home) - (1.0 - exp_home))

    df["Home_PreMatch_Elo"], df["Away_PreMatch_Elo"] = h_pre, a_pre
    df["Net_Elo_Advantage"] = h_pre - a_pre
    state["last_season"] = active_season
    return df, state


def compute_xg_features(df, state, window=config.FORM_WINDOW):
    hist = state.setdefault("xg_history", {})
    state.setdefault("xg_sum", 0.0)
    state.setdefault("xg_count", 0)
    home, away = df["HomeTeam"].to_numpy(), df["AwayTeam"].to_numpy()
    hx, ax = df["home_xG"].to_numpy(), df["away_xG"].to_numpy()
    n = len(df)
    hf, ha, af, aa = np.empty(n), np.empty(n), np.empty(n), np.empty(n)

    for i in range(n):
        ht, at = int(home[i]), int(away[i])
        # Fallback for a team with no history uses only matches played before this one
        league_xg = state["xg_sum"] / state["xg_count"] if state["xg_count"] else config.DEFAULT_LEAGUE_XG
        s = hist.get(ht, [])[-window:]
        hf[i], ha[i] = (np.mean([x[0] for x in s]), np.mean([x[1] for x in s])) if s else (league_xg, league_xg)
        s = hist.get(at, [])[-window:]
        af[i], aa[i] = (np.mean([x[0] for x in s]), np.mean([x[1] for x in s])) if s else (league_xg, league_xg)
        hist.setdefault(ht, []).append((hx[i], ax[i]))
        hist.setdefault(at, []).append((ax[i], hx[i]))
        state["xg_sum"] += hx[i] + ax[i]
        state["xg_count"] += 2

    df["Home_Rolling_xG_For"], df["Home_Rolling_xG_Against"] = hf, ha
    df["Away_Rolling_xG_For"], df["Away_Rolling_xG_Against"] = af, aa
    df["Net_xG_Form"] = hf - af
    df["Attack_Defence_Mismatch"] = hf - aa
    df["Reverse_Mismatch"] = af - ha
    return df, state


def compute_variance(df, state, window=config.FORM_WINDOW):
    hist = state.setdefault("team_history", {})
    home, away = df["HomeTeam"].to_numpy(), df["AwayTeam"].to_numpy()
    he, ae = df["Home_PreMatch_Elo"].to_numpy(), df["Away_PreMatch_Elo"].to_numpy()
    hx, ax = df["home_xG"].to_numpy(), df["away_xG"].to_numpy()   # per-match xG
    n = len(df)
    hev, hxv, aev, axv = np.zeros(n), np.zeros(n), np.zeros(n), np.zeros(n)

    for i in range(n):
        ht, at = int(home[i]), int(away[i])
        s = hist.get(ht, [])[-window:]
        if len(s) > 1:
            hev[i], hxv[i] = np.var([v[0] for v in s]), np.var([v[1] for v in s])
        s = hist.get(at, [])[-window:]
        if len(s) > 1:
            aev[i], axv[i] = np.var([v[0] for v in s]), np.var([v[1] for v in s])
        hist.setdefault(ht, []).append((he[i], hx[i]))
        hist.setdefault(at, []).append((ae[i], ax[i]))

    df["Home_Elo_Var_5pt"], df["Home_xG_Var_5pt"] = hev, hxv
    df["Away_Elo_Var_5pt"], df["Away_xG_Var_5pt"] = aev, axv
    df["Net_Elo_Variance_Diff"] = hev - aev
    df["Net_xG_Variance_Diff"] = hxv - axv
    return df, state


def build_features(df_encoded: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """df_encoded: chronologically sorted, team columns encoded with ``encode``."""
    df = df_encoded.sort_values("Date", kind="stable").reset_index(drop=True)
    state = {}
    df, state = compute_elo_features(df, state)
    df, state = compute_xg_features(df, state)
    df, state = compute_variance(df, state)
    return df, state
