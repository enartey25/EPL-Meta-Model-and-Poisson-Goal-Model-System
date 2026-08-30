import numpy as np
import pandas as pd
import pickle
import warnings
import os

warnings.filterwarnings("ignore")

from inference_utils import (
    PROMOTED_START_ELO,
    DEFAULT_LEAGUE_XG,
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


DEFAULT_PROMOTED_TEAMS = ["Coventry", "Hull", "Ipswich"]

PROMOTED_CLUBS = {
    "Coventry", "Coventry City",
    "Ipswich", "Ipswich Town",
    "Hull", "Hull City",
    "Burnley", "Leeds", "Leicester",
    "Luton", "Sheffield United", "Southampton", "Sunderland",
}


def get_teams(extra_teams: list[str] | None = None) -> list[str]:
    """
    Return sorted list of all canonical EPL teams, merging assets team_map,
    promoted teams (Coventry City, Ipswich Town, Hull City), and any dynamically
    discovered clubs from live data.
    """
    assets = load_assets()
    all_teams = set(assets["team_map"].keys()) | set(DEFAULT_PROMOTED_TEAMS)
    if extra_teams:
        all_teams.update([t for t in extra_teams if t and isinstance(t, str)])
    return sorted(all_teams)


def get_elo_state(extra_teams: list[str] | None = None) -> dict[str, float]:
    """
    Return current ELO state keyed by team name.
    Newly promoted or unranked clubs default to PROMOTED_START_ELO (1420.0).
    """
    assets = load_assets()
    team_map = assets["team_map"]
    inv = {v: k for k, v in team_map.items()}
    elo_raw = assets["state"].get("elo", {})

    elo_dict = {inv[k]: float(v) for k, v in elo_raw.items() if k in inv}

    # Ensure all teams in canonical list have an Elo rating
    all_teams = get_teams(extra_teams)
    for team in all_teams:
        if team not in elo_dict:
            elo_dict[team] = PROMOTED_START_ELO

    return elo_dict


def is_promoted_or_new(team: str) -> bool:
    """
    Check if a team is a promoted club or lacks a full top-flight historical record.
    """
    if team in PROMOTED_CLUBS:
        return True
    assets = load_assets()
    team_map = assets["team_map"]
    t_id = team_map.get(team)
    if t_id is None:
        return True
    
    xg_hist = assets["state"].get("xg_history", {}).get(t_id, [])
    return len(xg_hist) < 100


def build_inference_row(home_team: str, away_team: str) -> pd.DataFrame:
    """
    Build a single-row DataFrame with all 16 features for inference.
    Handles newly promoted clubs seamlessly using notebook baseline priors.
    """
    assets = load_assets()
    team_map = assets["team_map"]
    state_raw = assets["state"]

    inv_map = {v: k for k, v in team_map.items()}

    def remap_state(raw_dict: dict) -> dict:
        return {inv_map[k]: v for k, v in raw_dict.items() if k in inv_map}

    state = {
        "elo": remap_state(state_raw.get("elo", {})),
        "xg_history": remap_state(state_raw.get("xg_history", {})),
        "team_history": remap_state(state_raw.get("team_history", {})),
    }

    # If home_team or away_team is newly discovered, assign default promoted Elo
    if home_team not in state["elo"]:
        state["elo"][home_team] = PROMOTED_START_ELO
    if away_team not in state["elo"]:
        state["elo"][away_team] = PROMOTED_START_ELO

    # Build initial row
    df = pd.DataFrame([{"HomeTeam": home_team, "AwayTeam": away_team}])
    df = apply_elo_features_inference(df, state["elo"])
    df = compute_xg_features_inference(df, state)
    df = compute_variance_inference(df, state)

    # Encode team names (using existing team_map index, or a neutral category fallback)
    # Defaulting to middle neutral index if completely unknown to prevent categorical failure
    default_code = len(team_map) // 2
    df["HomeTeam"] = team_map.get(home_team, default_code)
    df["AwayTeam"] = team_map.get(away_team, default_code)

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
        "elo": remap(state_raw.get("elo", {})),
        "xg_history": remap(state_raw.get("xg_history", {})),
        "team_history": remap(state_raw.get("team_history", {})),
    }

    if home_team not in state["elo"]:
        state["elo"][home_team] = PROMOTED_START_ELO
    if away_team not in state["elo"]:
        state["elo"][away_team] = PROMOTED_START_ELO

    df = pd.DataFrame([{"HomeTeam": home_team, "AwayTeam": away_team}])
    df = apply_elo_features_inference(df, state["elo"])
    df = compute_xg_features_inference(df, state)
    df = compute_variance_inference(df, state)

    return df.iloc[0].to_dict()
