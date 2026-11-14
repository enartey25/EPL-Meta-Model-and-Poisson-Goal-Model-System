"""Team registry and Elo state for the app, read from the pipeline's assets."""
from predictor import is_cold_start, load_assets, teams

PROMOTED_START_ELO = 1420.0


def get_teams(extra_teams=None) -> list[str]:
    return teams()


def get_elo_state(extra_teams=None) -> dict[str, float]:
    a = load_assets()
    inv = {v: k for k, v in a["team_map"].items()}
    elo = {inv[k]: float(v) for k, v in a["state"].get("elo", {}).items() if k in inv}
    for t in teams():
        elo.setdefault(t, a["elo_params"]["PROMOTED_START_ELO"])
    return elo


def is_promoted_or_new(team: str) -> bool:
    return is_cold_start(team)
