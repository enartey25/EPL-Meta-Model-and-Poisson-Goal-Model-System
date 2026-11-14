"""Inference-time feature engineering and prediction for the EPL forecasting model.

Mirrors the training-time feature code in ``pipeline/build_features.py``. The
pipeline's parity check (``pipeline/validate.py``) rebuilds the last block of
played matches with these functions and fails the run unless the features are
identical to the ones the models were trained on.
"""
import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import least_squares

# Alternative / full club names -> football-data.co.uk names used in training
TEAM_NAME_ALIASES = {
    "AFC Bournemouth": "Bournemouth", "Arsenal FC": "Arsenal", "Aston Villa FC": "Aston Villa",
    "Brentford FC": "Brentford", "Brighton & Hove Albion": "Brighton",
    "Brighton and Hove Albion": "Brighton", "Burnley FC": "Burnley", "Chelsea FC": "Chelsea",
    "Coventry City": "Coventry", "Crystal Palace FC": "Crystal Palace", "Everton FC": "Everton",
    "Fulham FC": "Fulham", "Hull City": "Hull", "Ipswich Town": "Ipswich",
    "Leeds United": "Leeds", "Leicester City": "Leicester", "Liverpool FC": "Liverpool",
    "Luton Town": "Luton", "Manchester City": "Man City", "Manchester United": "Man United",
    "Middlesbrough FC": "Middlesbrough", "Newcastle United": "Newcastle",
    "Norwich City": "Norwich", "Nottingham Forest": "Nott'm Forest", "Nottm Forest": "Nott'm Forest",
    "Sheffield Utd": "Sheffield United", "Sheffield United FC": "Sheffield United",
    "Southampton FC": "Southampton", "Stoke City": "Stoke", "Sunderland AFC": "Sunderland",
    "Swansea City": "Swansea", "Tottenham Hotspur": "Tottenham", "Spurs": "Tottenham",
    "Watford FC": "Watford", "West Bromwich Albion": "West Brom", "West Ham United": "West Ham",
    "Wolverhampton Wanderers": "Wolves", "Wolverhampton": "Wolves",
    "Queens Park Rangers": "QPR", "Cardiff City": "Cardiff", "Derby County": "Derby",
    "Preston North End": "Preston", "Blackburn Rovers": "Blackburn", "Plymouth Argyle": "Plymouth",
}


def normalize_team_names(names):
    return pd.Series(list(names), dtype=object).map(
        lambda n: TEAM_NAME_ALIASES.get(str(n).strip(), str(n).strip())).to_numpy()


def encode_teams(names, team_map):
    """Names -> categorical codes with the FIXED training category set."""
    names = pd.Series(normalize_team_names(names))
    codes = names.map(team_map)
    if codes.isna().any():
        raise ValueError(f"Unknown team name(s): {sorted(names[codes.isna()].unique())}")
    return pd.Categorical(codes.astype(int), categories=list(range(len(team_map))))


# ---------------------------------------------------------------- features
def add_elo_features(df, state, elo_params):
    elo = dict(state["elo"])
    season = df["season"].iloc[0] if "season" in df.columns else state.get("last_season")
    if season != state.get("last_season"):
        # First fixtures of a new season: apply the same summer regression as training
        r, base = elo_params["REGRESSION_FACTOR"], elo_params["INITIAL_ELO"]
        elo = {t: v * (1 - r) + base * r for t, v in elo.items()}
    promoted = elo_params["PROMOTED_START_ELO"]
    h = np.array([elo.get(int(t), promoted) for t in df["HomeTeam"]], dtype=float)
    a = np.array([elo.get(int(t), promoted) for t in df["AwayTeam"]], dtype=float)
    df["Home_PreMatch_Elo"], df["Away_PreMatch_Elo"] = h, a
    df["Net_Elo_Advantage"] = h - a
    return df


def add_xg_features(df, state, window=5, default_league_xg=1.35):
    hist = state.get("xg_history", {})
    league_xg = state["xg_sum"] / state["xg_count"] if state.get("xg_count") else default_league_xg

    def form(team):
        s = hist.get(int(team), [])[-window:]
        if not s:
            return league_xg, league_xg
        return np.mean([x[0] for x in s]), np.mean([x[1] for x in s])

    hf = np.array([form(t) for t in df["HomeTeam"]], dtype=float)
    af = np.array([form(t) for t in df["AwayTeam"]], dtype=float)
    df["Home_Rolling_xG_For"], df["Home_Rolling_xG_Against"] = hf[:, 0], hf[:, 1]
    df["Away_Rolling_xG_For"], df["Away_Rolling_xG_Against"] = af[:, 0], af[:, 1]
    df["Net_xG_Form"] = hf[:, 0] - af[:, 0]
    df["Attack_Defence_Mismatch"] = hf[:, 0] - af[:, 1]
    df["Reverse_Mismatch"] = af[:, 0] - hf[:, 1]
    return df


def add_variance_features(df, state, window=5):
    hist = state.get("team_history", {})

    def var(team):
        s = hist.get(int(team), [])[-window:]
        if len(s) <= 1:
            return 0.0, 0.0
        return np.var([v[0] for v in s]), np.var([v[1] for v in s])

    hv = np.array([var(t) for t in df["HomeTeam"]], dtype=float)
    av = np.array([var(t) for t in df["AwayTeam"]], dtype=float)
    df["Home_Elo_Var_5pt"], df["Home_xG_Var_5pt"] = hv[:, 0], hv[:, 1]
    df["Away_Elo_Var_5pt"], df["Away_xG_Var_5pt"] = av[:, 0], av[:, 1]
    df["Net_Elo_Variance_Diff"] = hv[:, 0] - av[:, 0]
    df["Net_xG_Variance_Diff"] = hv[:, 1] - av[:, 1]
    return df


def build_feature_frame(fixtures, assets):
    """fixtures: DataFrame with HomeTeam, AwayTeam (names) and optionally season."""
    f = fixtures.copy().reset_index(drop=True)
    f["HomeTeam"] = encode_teams(f["HomeTeam"], assets["team_map"])
    f["AwayTeam"] = encode_teams(f["AwayTeam"], assets["team_map"])
    f = add_elo_features(f, assets["state"], assets["elo_params"])
    f = add_xg_features(f, assets["state"], window=assets["window"])
    f = add_variance_features(f, assets["state"], window=assets["window"])
    return f


# ---------------------------------------------------------------- goal model
def fixture_lambdas(home_code, away_code, pstate):
    """Opponent-adjusted Poisson rates from the latest team strengths."""
    lh_bar, la_bar = pstate["league_home"], pstate["league_away"]
    lam_h = pstate["home_attack"].get(home_code, lh_bar) * pstate["away_defence"].get(away_code, lh_bar) / lh_bar
    lam_a = pstate["away_attack"].get(away_code, la_bar) * pstate["home_defence"].get(home_code, la_bar) / la_bar
    return float(np.clip(lam_h, *pstate["clip"])), float(np.clip(lam_a, *pstate["clip"]))


def score_matrix(lam_home, lam_away, max_goals=10):
    """Rows = home goals, columns = away goals; renormalised for the truncated tail."""
    g = np.arange(max_goals + 1)
    m = np.outer(stats.poisson.pmf(g, lam_home), stats.poisson.pmf(g, lam_away))
    return m / m.sum()


def outcome_probs(m):
    """[Draw, AwayWin, HomeWin]."""
    return np.array([np.trace(m), np.triu(m, 1).sum(), np.tril(m, -1).sum()])


def implied_lambdas(p, lam_h0, lam_a0, reg=0.05, max_goals=10):
    """Rates whose score grid reproduces outcome probabilities p = [Draw, Away, Home]."""
    log0 = np.log([lam_h0, lam_a0])

    def resid(theta):
        q = outcome_probs(score_matrix(*np.exp(theta), max_goals=max_goals))
        return np.r_[q[2] - p[2], q[1] - p[1], reg * (theta - log0)]

    return np.exp(least_squares(resid, log0, bounds=(np.log(0.05), np.log(8.0))).x)


# ---------------------------------------------------------------- outcome model
def meta_features(parts, transform):
    X = np.hstack(parts)
    return np.log(np.clip(X, 1e-6, 1)) if transform == "log" else X


def predict_outcomes(fixtures, assets):
    """Returns (DataFrame with P_Draw / P_Away / P_Home and the raw Poisson rates,
    engineered feature frame)."""
    f = build_feature_frame(fixtures, assets)
    X = f[assets["final_features"]]
    lam = np.array([fixture_lambdas(int(h), int(a), assets["poisson_state"])
                    for h, a in zip(f["HomeTeam"], f["AwayTeam"])]).reshape(-1, 2)
    parts = {"poisson": np.array([outcome_probs(score_matrix(h, a)) for h, a in lam]).reshape(-1, 3)}
    if assets.get("xgb_model") is not None:
        parts["xgb"] = assets["xgb_model"].predict_proba(X)
    if assets.get("rf_model") is not None:
        parts["rf"] = assets["rf_model"].predict_proba(X)
    p = assets["meta_model"].predict_proba(
        meta_features([parts[k] for k in assets["meta_inputs"]], assets["meta_transform"]))
    out = pd.DataFrame({"HomeTeam": normalize_team_names(fixtures["HomeTeam"]),
                        "AwayTeam": normalize_team_names(fixtures["AwayTeam"]),
                        "P_Draw": p[:, 0], "P_Away": p[:, 1], "P_Home": p[:, 2],
                        "Raw_Lambda_Home": lam[:, 0], "Raw_Lambda_Away": lam[:, 1]})
    return out, f


def predict_scores(outcomes, assets, max_goals=10):
    """Adds meta-implied goal expectations and goal markets to predict_outcomes() output."""
    rows, mats = [], []
    g = np.arange(max_goals + 1)
    for _, r in outcomes.iterrows():
        lh, la = implied_lambdas(r[["P_Draw", "P_Away", "P_Home"]].to_numpy(float),
                                 r["Raw_Lambda_Home"], r["Raw_Lambda_Away"], max_goals=max_goals)
        m = score_matrix(lh, la, max_goals)
        hs, as_ = np.unravel_index(m.argmax(), m.shape)
        rows.append({"xG_Home": lh, "xG_Away": la,
                     "P_Over_2.5": m[np.add.outer(g, g) > 2.5].sum(),
                     "P_BTTS": m[1:, 1:].sum(),
                     "P_Home_CleanSheet": m[:, 0].sum(), "P_Away_CleanSheet": m[0, :].sum(),
                     "Most_Likely_Score": f"{hs}-{as_}",
                     "P_Most_Likely_Score": float(m[hs, as_])})
        mats.append(m)
    return pd.concat([outcomes.reset_index(drop=True), pd.DataFrame(rows)], axis=1), mats
