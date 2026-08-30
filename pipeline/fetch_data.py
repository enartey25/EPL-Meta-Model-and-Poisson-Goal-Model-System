"""
Data ingestion module for the EPL Match Predictor pipeline.
Fetches match records from football-data.co.uk and expected goals (xG) from Understat,
with automatic fallback to cached baseline data when offline or firewalled.
"""
import io
import logging
import os
import pickle
import sys
import pandas as pd
import requests

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_loader import (
    TEAM_NAME_MAP,
    UNDERSTAT_MAP,
    normalize_team,
    normalize_understat,
)

logger = logging.getLogger(__name__)

SEASON_URL_MAP = {
    "2022/23": "https://www.football-data.co.uk/mmz4281/2223/E0.csv",
    "2023/24": "https://www.football-data.co.uk/mmz4281/2324/E0.csv",
    "2024/25": "https://www.football-data.co.uk/mmz4281/2425/E0.csv",
    "2025/26": "https://www.football-data.co.uk/mmz4281/2526/E0.csv",
    "2026/27": "https://www.football-data.co.uk/mmz4281/2627/E0.csv",
}

COLS_NEEDED = [
    "Date", "HomeTeam", "AwayTeam",
    "FTHG", "FTAG", "FTR",
    "HTHG", "HTAG", "HTR",
    "HS", "AS", "HST", "AST",
    "HF", "AF", "HC", "AC",
    "HY", "AY", "HR", "AR",
]

try:
    import certifi
    CA_BUNDLE = certifi.where()
except ImportError:
    CA_BUNDLE = True

import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


def _safe_get(url: str, headers: dict = None) -> requests.Response:
    """Perform robust HTTP GET with SSL fallback."""
    try:
        return requests.get(url, timeout=12, headers=headers, verify=CA_BUNDLE)
    except Exception:
        return requests.get(url, timeout=12, headers=headers, verify=False)


def fetch_football_data_raw() -> pd.DataFrame:
    """Download and normalize match statistics from football-data.co.uk."""
    dfs = []
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}

    for season, url in SEASON_URL_MAP.items():
        try:
            resp = _safe_get(url, headers=headers)
            if resp.status_code == 200 and len(resp.text) > 100:
                df = pd.read_csv(io.StringIO(resp.text), encoding="latin-1")
                valid_cols = [c for c in COLS_NEEDED if c in df.columns]
                df = df[valid_cols].copy()
                df["Season"] = season
                df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")
                df = df.dropna(subset=["Date", "HomeTeam", "AwayTeam", "FTR"])
                df["HomeTeam"] = df["HomeTeam"].astype(str).str.strip().map(normalize_team)
                df["AwayTeam"] = df["AwayTeam"].astype(str).str.strip().map(normalize_team)
                dfs.append(df)
                logger.info(f"Loaded {len(df)} matches for season {season}")
            else:
                logger.info(f"Season {season} not available or empty (HTTP {resp.status_code})")
        except Exception as e:
            logger.warning(f"Could not reach football-data for season {season}: {e}")

    if not dfs:
        return pd.DataFrame()

    full_df = pd.concat(dfs, ignore_index=True)
    full_df = full_df.sort_values("Date").reset_index(drop=True)
    logger.info(f"Total live football-data records fetched: {len(full_df)}")
    return full_df


def fetch_understat_xg_raw() -> pd.DataFrame:
    """Download expected goals data from Understat for recent seasons."""
    try:
        from understatapi import UnderstatClient
        LEAGUE = "EPL"
        SEASONS = ["2022", "2023", "2024", "2025", "2026"]
        cleaned_matches = []

        with UnderstatClient() as understat:
            for season in SEASONS:
                try:
                    fixtures = understat.league(league=LEAGUE).get_match_data(season=season)
                    for match in fixtures:
                        if not match.get("isResult"):
                            continue
                        match_info = {
                            "season": f"{season}/{str(int(season)+1)[2:]}",
                            "date": pd.to_datetime(match["datetime"]).date(),
                            "home_team": normalize_understat(match["h"]["title"]),
                            "away_team": normalize_understat(match["a"]["title"]),
                            "home_xG": float(match["xG"]["h"]),
                            "away_xG": float(match["xG"]["a"]),
                        }
                        cleaned_matches.append(match_info)
                    logger.info(f"Loaded Understat xG for season {season}")
                except Exception as e:
                    logger.warning(f"Could not load Understat season {season}: {e}")

        if not cleaned_matches:
            return pd.DataFrame()

        df_xg = pd.DataFrame(cleaned_matches)
        return df_xg
    except Exception as e:
        logger.warning(f"Understat API fetch skipped: {e}")
        return pd.DataFrame()


def load_cached_baseline_data() -> pd.DataFrame:
    """Load baseline match dataset from existing assets package when offline."""
    assets_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "epl_predictor_assets.pkl")
    if not os.path.exists(assets_path):
        return pd.DataFrame()

    try:
        with open(assets_path, "rb") as f:
            assets = pickle.load(f)

        df_p = assets.get("df_poisson")
        team_map = assets.get("team_map", {})
        inv_team_map = {v: k for k, v in team_map.items()}

        if df_p is not None and not df_p.empty:
            df = df_p.copy()
            # If teams are integer-encoded, recover string names
            if pd.api.types.is_numeric_dtype(df["HomeTeam"]):
                df["HomeTeam"] = df["HomeTeam"].map(inv_team_map).fillna(df["HomeTeam"].astype(str))
            if pd.api.types.is_numeric_dtype(df["AwayTeam"]):
                df["AwayTeam"] = df["AwayTeam"].map(inv_team_map).fillna(df["AwayTeam"].astype(str))

            # Decode FTR numeric labels if present
            ftr_inv = {0: "D", 1: "A", 2: "H"}
            if pd.api.types.is_numeric_dtype(df["FTR"]):
                df["FTR"] = df["FTR"].map(ftr_inv).fillna("D")

            df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
            df["Season"] = df["season"] if "season" in df.columns else "2022/23"
            logger.info(f"Successfully loaded {len(df)} cached historical matches from assets.")
            return df
    except Exception as e:
        logger.warning(f"Failed to load cached baseline data: {e}")

    return pd.DataFrame()


def load_and_merge_pipeline_data() -> pd.DataFrame:
    """Merge football-data match records with Understat xG records with fallback."""
    df_stats = fetch_football_data_raw()
    df_xg = fetch_understat_xg_raw()

    # If live scraping failed, use cached historical baseline
    if df_stats.empty:
        logger.info("Live data fetching unreachable. Loading cached historical dataset.")
        df_stats = load_cached_baseline_data()
        if df_stats.empty:
            raise RuntimeError("Fatal: Match dataset is empty. Cannot continue pipeline.")
        return df_stats

    # Merge live data with xG
    default_home_xg = 1.69
    default_away_xg = 1.32
    if not df_xg.empty:
        default_home_xg = float(df_xg["home_xG"].mean())
        default_away_xg = float(df_xg["away_xG"].mean())
        df_stats["_date_key"] = df_stats["Date"].dt.date.astype(str)
        df_stats["_home_key"] = df_stats["HomeTeam"].str.lower().str.strip()
        df_stats["_away_key"] = df_stats["AwayTeam"].str.lower().str.strip()

        df_xg["_date_key"] = pd.to_datetime(df_xg["date"]).dt.date.astype(str)
        df_xg["_home_key"] = df_xg["home_team"].str.lower().str.strip()
        df_xg["_away_key"] = df_xg["away_team"].str.lower().str.strip()

        merged = pd.merge(
            df_stats,
            df_xg[["_date_key", "_home_key", "_away_key", "home_xG", "away_xG"]],
            on=["_date_key", "_home_key", "_away_key"],
            how="left",
        )
        merged = merged.drop(columns=["_date_key", "_home_key", "_away_key"])
    else:
        merged = df_stats.copy()
        if "home_xG" not in merged.columns:
            merged["home_xG"] = np.nan
        if "away_xG" not in merged.columns:
            merged["away_xG"] = np.nan

    if "HST" in merged.columns and "AST" in merged.columns:
        merged["home_xG"] = merged["home_xG"].fillna(
            (merged["FTHG"] * 0.5 + merged["HST"] * 0.15).clip(lower=0.2, upper=5.0)
        )
        merged["away_xG"] = merged["away_xG"].fillna(
            (merged["FTAG"] * 0.5 + merged["AST"] * 0.15).clip(lower=0.2, upper=5.0)
        )
    else:
        merged["home_xG"] = merged["home_xG"].fillna(default_home_xg)
        merged["away_xG"] = merged["away_xG"].fillna(default_away_xg)

    merged["season"] = merged["Season"]
    merged = merged.sort_values("Date").reset_index(drop=True)
    logger.info(f"Merged dataset ready: {len(merged)} matches with xG.")
    return merged
