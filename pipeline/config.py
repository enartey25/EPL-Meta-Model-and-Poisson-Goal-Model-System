"""Shared constants for the EPL forecasting pipeline.

Model settings come from the development notebook: hyperparameters were tuned
with time-series cross-validation on 2022/23-2023/24, and the meta-learner
inputs were selected out-of-fold on 2024/25.
"""
from datetime import date
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
ASSETS_PATH = os.path.join(PROJECT_ROOT, "epl_predictor_assets.pkl")

ASSET_FORMAT_VERSION = 2  # bump when the asset layout changes; forces a retrain

FIRST_SEASON_START = 2022


def season_label(start_year: int) -> str:
    return f"{start_year}/{str(start_year + 1)[2:]}"


def current_season_start(today: date | None = None) -> int:
    today = today or date.today()
    return today.year if today.month >= 7 else today.year - 1


def tracked_seasons(today: date | None = None) -> list[str]:
    return [season_label(y) for y in range(FIRST_SEASON_START, current_season_start(today) + 1)]


def football_data_url(start_year: int) -> str:
    code = f"{str(start_year)[2:]}{str(start_year + 1)[2:]}"
    return f"https://www.football-data.co.uk/mmz4281/{code}/E0.csv"


def fotmob_season(start_year: int) -> str:
    return f"{start_year}/{start_year + 1}"


FOTMOB_LEAGUE_ID = 47
HTTP_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/130.0 Safari/537.36"
}

# ---------------------------------------------------------------- features
ELO_PARAMS = dict(INITIAL_ELO=1500, K_FACTOR=32, HOME_ADVANTAGE=50,
                  REGRESSION_FACTOR=0.33, PROMOTED_START_ELO=1420)
FORM_WINDOW = 5
DEFAULT_LEAGUE_XG = 1.35

FINAL_FEATURES = [
    "Net_Elo_Advantage", "HomeTeam", "AwayTeam",
    "Net_Elo_Variance_Diff", "Net_xG_Form", "Net_xG_Variance_Diff",
    "Attack_Defence_Mismatch", "Reverse_Mismatch",
    "Home_Rolling_xG_For", "Home_Rolling_xG_Against",
    "Away_Rolling_xG_For", "Away_Rolling_xG_Against",
    "Home_PreMatch_Elo", "Away_PreMatch_Elo",
    "Home_Elo_Var_5pt", "Away_Elo_Var_5pt",
]

# ---------------------------------------------------------------- Poisson
POISSON_HALFLIFE = 19          # venue-specific matches, ~ one season
LAMBDA_CLIP = (0.2, 5.0)
PRIOR_HOME, PRIOR_AWAY = 1.5, 1.2
MAX_GOALS = 10

# ---------------------------------------------------------------- models
RF_PARAMS = dict(n_estimators=443, max_depth=3, min_samples_split=10, min_samples_leaf=7,
                 max_features="sqrt", class_weight=None, criterion="log_loss",
                 random_state=42, n_jobs=-1)
META_INPUTS = ("rf", "poisson")   # selected out-of-fold on 2024/25 (0.9803 vs 0.9933 for xgb+rf)
META_TRANSFORM = "prob"
MIN_PRIOR_SEASONS_FOR_META = 2    # walk-forward: base model needs >= 2 earlier seasons

RETRAIN_BATCH_THRESHOLD = 50      # full retrain every ~5 gameweeks
