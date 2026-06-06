import numpy as np
import pandas as pd
import pickle
import warnings
import os
warnings.filterwarnings("ignore")

from inference_utils import (
    apply_elo_features_inference,
    compute_xg_features_inference,
    compute_variance_inference,
)

_ASSETS_PATH = os.path.join(os.path.dirname(__file__), "epl_predictor_assets.pkl")

_assets = None

def load_assets():
    global _assets
    if _assets is None:
        with open(_ASSETS_PATH, "rb") as f:
            _assets = pickle.load(f)
    return _assets


def get_teams() -> list[str]:
    """Return canonical sorted team list."""
    assets = load_assets()
    return sorted(assets["team_map"].keys())


def get_elo_state() -> dict:
    """Return current ELO state keyed by team name."""
    assets = load_assets()
    team_map = assets["team_map"]
    inv = {v: k for k, v in team_map.items()}
    elo_raw = assets["state"]["elo"]
    return {inv[k]: v for k, v in elo_raw.items() if k in inv}


def build_inference_row(home_team: str, away_team: str) -> pd.DataFrame:
    """Build a single-row DataFrame with all 16 features for inference."""
    assets = load_assets()
    team_map = assets["team_map"]
    state_raw = assets["state"]

    # Map names to indices for the state dicts (state is keyed by team index)
    h_idx = team_map.get(home_team)
    a_idx = team_map.get(away_team)

    # Reconstruct a name-keyed state for inference_utils
    inv_map = {v: k for k, v in team_map.items()}

    def remap_state(raw_dict: dict) -> dict:
        return {inv_map[k]: v for k, v in raw_dict.items() if k in inv_map}

    state = {
        "elo": remap_state(state_raw["elo"]),
        "xg_history": remap_state(state_raw["xg_history"]),
        "team_history": remap_state(state_raw["team_history"]),
    }

    # Build row
    df = pd.DataFrame([{"HomeTeam": home_team, "AwayTeam": away_team}])
    df = apply_elo_features_inference(df, state["elo"])
    df = compute_xg_features_inference(df, state)
    df = compute_variance_inference(df, state)

    # Encode team names
    df["HomeTeam"] = team_map.get(home_team, 0)
    df["AwayTeam"] = team_map.get(away_team, 0)

    return df


def get_xg_features(home_team: str, away_team: str) -> dict:
    """Return raw xG features (before encoding) for display purposes."""
    assets = load_assets()
    team_map = assets["team_map"]
    state_raw = assets["state"]
    inv_map = {v: k for k, v in team_map.items()}

    def remap(d):
        return {inv_map[k]: v for k, v in d.items() if k in inv_map}

    state = {
        "elo": remap(state_raw["elo"]),
        "xg_history": remap(state_raw["xg_history"]),
        "team_history": remap(state_raw["team_history"]),
    }

    df = pd.DataFrame([{"HomeTeam": home_team, "AwayTeam": away_team}])
    df = apply_elo_features_inference(df, state["elo"])
    df = compute_xg_features_inference(df, state)
    df = compute_variance_inference(df, state)

    return df.iloc[0].to_dict()
