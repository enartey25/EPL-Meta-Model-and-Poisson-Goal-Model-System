"""Pre-export checks. Any failure stops the pipeline before assets are written."""
import logging

import numpy as np
import pandas as pd

import inference_utils as iu
from pipeline import config
from pipeline.build_features import compute_elo_features, compute_variance, compute_xg_features

logger = logging.getLogger(__name__)


def parity_check(df: pd.DataFrame, team_map: dict, block: int = 10) -> float:
    """Rebuild the state from every match before the last block of played matches,
    generate that block's features with inference_utils, and require them to equal
    the training features exactly."""
    cutoff = df["Date"].iloc[-block]
    hist, day = df[df["Date"] < cutoff].copy(), df[df["Date"] >= cutoff]
    teams = pd.concat([day["HomeTeam"], day["AwayTeam"]]).astype(int)
    if not teams.is_unique:  # a team plays twice in the block: shrink to the last date
        cutoff = df["Date"].iloc[-1]
        hist, day = df[df["Date"] < cutoff].copy(), df[df["Date"] >= cutoff]

    state = {}
    hist, state = compute_elo_features(hist, state)
    hist, state = compute_xg_features(hist, state)
    hist, state = compute_variance(hist, state)

    fixtures = pd.DataFrame({"HomeTeam": day["HomeTeam_name"].values, "AwayTeam": day["AwayTeam_name"].values,
                             "season": day["season"].values})
    assets = {"team_map": team_map, "state": state, "elo_params": config.ELO_PARAMS, "window": config.FORM_WINDOW}
    feat = iu.build_feature_frame(fixtures, assets)
    numeric = [c for c in config.FINAL_FEATURES if c not in ("HomeTeam", "AwayTeam")]
    diff = float(np.abs(feat[numeric].to_numpy() - day[numeric].to_numpy()).max())
    if diff > 1e-9:
        raise AssertionError(f"Inference features differ from training features (max abs diff {diff})")
    if not (feat["HomeTeam"].astype(int).to_numpy() == day["HomeTeam"].astype(int).to_numpy()).all():
        raise AssertionError("Team encoding differs between training and inference")
    logger.info(f"Parity check passed on {len(day)} matches (max abs diff {diff:.1e})")
    return diff


def sanity_check_assets(assets: dict) -> None:
    """Predict a fixed fixture and check the output is a valid distribution."""
    teams = sorted(assets["team_map"])
    fx = pd.DataFrame({"HomeTeam": [teams[0]], "AwayTeam": [teams[1]], "season": [assets["current_season"]]})
    out, _ = iu.predict_outcomes(fx, assets)
    p = out[["P_Draw", "P_Away", "P_Home"]].to_numpy()
    if not (np.isfinite(p).all() and np.allclose(p.sum(1), 1) and (p > 0).all()):
        raise AssertionError(f"Invalid probabilities from exported assets: {p}")
