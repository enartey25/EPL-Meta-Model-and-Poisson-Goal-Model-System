"""App-facing prediction API over the pipeline's exported assets."""
import os
import pickle
from functools import lru_cache

import numpy as np
import pandas as pd
from scipy import stats

import inference_utils as iu

ROOT = os.path.dirname(os.path.abspath(__file__))
ASSETS_PATH = os.path.join(ROOT, "epl_predictor_assets.pkl")
OUTCOMES = ("Home Win", "Draw", "Away Win")
OUTCOME_COLORS = {"Home Win": "#00d4aa", "Draw": "#f59e0b", "Away Win": "#f43f5e"}


@lru_cache(maxsize=1)
def load_assets() -> dict:
    with open(ASSETS_PATH, "rb") as f:
        return pickle.load(f)


def teams() -> list[str]:
    return sorted(load_assets()["team_map"])


def is_cold_start(team: str) -> bool:
    """True when a club has no top-flight matches in the model's history yet,
    so its features come from priors (1420 Elo, league-average xG)."""
    a = load_assets()
    code = a["team_map"].get(iu.normalize_team_names([team])[0])
    return code is None or code not in a["state"].get("elo", {})


def predict_match(home_team: str, away_team: str) -> dict:
    a = load_assets()
    fx = pd.DataFrame({"HomeTeam": [home_team], "AwayTeam": [away_team], "season": [a["current_season"]]})
    outcomes, feats = iu.predict_outcomes(fx, a)
    forecast, mats = iu.predict_scores(outcomes, a)
    r, m, f = forecast.iloc[0], mats[0], feats.iloc[0]
    proba = {"Home Win": float(r["P_Home"]), "Draw": float(r["P_Draw"]), "Away Win": float(r["P_Away"])}
    predicted = max(proba, key=proba.get)
    hs, as_ = (int(x) for x in r["Most_Likely_Score"].split("-"))
    return {
        "home_team": r["HomeTeam"], "away_team": r["AwayTeam"],
        "proba": proba, "predicted_outcome": predicted, "confidence": proba[predicted],
        "matrix": m, "most_likely_score": (hs, as_), "most_likely_prob": float(m[hs, as_]),
        "lambda_h": float(r["xG_Home"]), "lambda_a": float(r["xG_Away"]),
        "raw_lambda_h": float(r["Raw_Lambda_Home"]), "raw_lambda_a": float(r["Raw_Lambda_Away"]),
        "markets": {"Over 2.5 goals": float(r["P_Over_2.5"]), "Both teams score": float(r["P_BTTS"]),
                    f"{r['HomeTeam']} clean sheet": float(r["P_Home_CleanSheet"]),
                    f"{r['AwayTeam']} clean sheet": float(r["P_Away_CleanSheet"])},
        "features": {"home_elo": float(f["Home_PreMatch_Elo"]), "away_elo": float(f["Away_PreMatch_Elo"]),
                     "home_xg_for": float(f["Home_Rolling_xG_For"]), "home_xg_against": float(f["Home_Rolling_xG_Against"]),
                     "away_xg_for": float(f["Away_Rolling_xG_For"]), "away_xg_against": float(f["Away_Rolling_xG_Against"])},
        "home_cold_start": is_cold_start(home_team), "away_cold_start": is_cold_start(away_team),
        # Poisson goal distributions for display
        "h_pmf8": stats.poisson.pmf(np.arange(9), float(r["xG_Home"])),
        "a_pmf8": stats.poisson.pmf(np.arange(9), float(r["xG_Away"])),
        # Keys kept for the legacy app layout
        "stacked_proba": proba,
        "home_fallback": is_cold_start(home_team), "away_fallback": is_cold_start(away_team),
    }


if __name__ == "__main__":  # smoke test used by CI
    t = teams()
    res = predict_match(t[0], t[1])
    p = np.array(list(res["proba"].values()))
    assert np.isclose(p.sum(), 1) and (p > 0).all(), res["proba"]
    print(f"Smoke test OK: {res['home_team']} vs {res['away_team']} -> "
          + ", ".join(f"{k} {v:.1%}" for k, v in res["proba"].items()))
