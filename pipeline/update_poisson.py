"""Leakage-free, opponent-adjusted Poisson goal model.

lambda_home = HomeAttack(home team) * AwayDefence(away team) / league home average
lambda_away = AwayAttack(away team) * HomeDefence(home team) / league away average

"Performance" blends goals and xG: (goals + xG) / 2. Strengths are
venue-specific exponentially weighted means, shifted one match so a fixture
only sees earlier results; league averages are expanding means of past matches.
"""
import numpy as np
import pandas as pd

from inference_utils import outcome_probs, score_matrix
from pipeline import config


def add_poisson_strengths(df: pd.DataFrame, halflife=config.POISSON_HALFLIFE):
    d = df.copy()  # same index as df, so rows stay aligned with the feature frame
    d["Home_Perf_For"] = (d["FTHG"] + d["home_xG"]) / 2
    d["Away_Perf_For"] = (d["FTAG"] + d["away_xG"]) / 2

    prev_ewm = lambda s: s.shift().ewm(halflife=halflife).mean()
    gh = d.groupby("HomeTeam", observed=True)
    ga = d.groupby("AwayTeam", observed=True)
    d["Home_Attack"] = gh["Home_Perf_For"].transform(prev_ewm)
    d["Home_Defence"] = gh["Away_Perf_For"].transform(prev_ewm)
    d["Away_Attack"] = ga["Away_Perf_For"].transform(prev_ewm)
    d["Away_Defence"] = ga["Home_Perf_For"].transform(prev_ewm)

    d["League_Home"] = d["Home_Perf_For"].shift().expanding().mean().fillna(config.PRIOR_HOME)
    d["League_Away"] = d["Away_Perf_For"].shift().expanding().mean().fillna(config.PRIOR_AWAY)
    d["Home_Attack"] = d["Home_Attack"].fillna(d["League_Home"])
    d["Away_Defence"] = d["Away_Defence"].fillna(d["League_Home"])
    d["Away_Attack"] = d["Away_Attack"].fillna(d["League_Away"])
    d["Home_Defence"] = d["Home_Defence"].fillna(d["League_Away"])

    d["Lambda_Home"] = (d["Home_Attack"] * d["Away_Defence"] / d["League_Home"]).clip(*config.LAMBDA_CLIP)
    d["Lambda_Away"] = (d["Away_Attack"] * d["Home_Defence"] / d["League_Away"]).clip(*config.LAMBDA_CLIP)

    # Latest strengths (including each team's most recent match) for upcoming fixtures
    last_ewm = lambda s: s.ewm(halflife=halflife).mean().iloc[-1]
    poisson_state = {
        "home_attack": {int(k): float(v) for k, v in gh["Home_Perf_For"].apply(last_ewm).items()},
        "home_defence": {int(k): float(v) for k, v in gh["Away_Perf_For"].apply(last_ewm).items()},
        "away_attack": {int(k): float(v) for k, v in ga["Away_Perf_For"].apply(last_ewm).items()},
        "away_defence": {int(k): float(v) for k, v in ga["Home_Perf_For"].apply(last_ewm).items()},
        "league_home": float(d["Home_Perf_For"].mean()),
        "league_away": float(d["Away_Perf_For"].mean()),
        "clip": config.LAMBDA_CLIP,
    }
    return d, poisson_state


def poisson_probs(lam_h, lam_a) -> np.ndarray:
    return np.array([outcome_probs(score_matrix(h, a, config.MAX_GOALS))
                     for h, a in zip(lam_h, lam_a)]).reshape(-1, 3)
