"""Data ingestion: match results from football-data.co.uk, match-level xG and
fixture lists from FotMob.

Understat (the original xG source, via ``understatapi``) has been offline
site-wide, so xG now comes from FotMob's public match API. Fetched xG is cached
in ``data/xg_fotmob.csv`` (committed), so a weekly run only requests matches
played since the last run.
"""
import io
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import requests

from inference_utils import normalize_team_names
from pipeline import config

logger = logging.getLogger(__name__)

XG_CACHE = os.path.join(config.DATA_DIR, "xg_fotmob.csv")
RESULT_COLS = ["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR",
               "HS", "AS", "HST", "AST", "HC", "AC", "HY", "AY", "HR", "AR"]
MAX_MISSING_XG_SHARE = 0.02


def _get(url, params=None, retries=4, as_json=True):
    last = None
    for attempt in range(retries):
        try:
            r = requests.get(url, params=params, headers=config.HTTP_HEADERS, timeout=30)
            if r.status_code == 200:
                return r.json() if as_json else r.text
            last = f"HTTP {r.status_code}"
        except (requests.RequestException, ValueError) as e:
            last = str(e)
        time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"Request failed after {retries} attempts: {url} {params or ''} ({last})")


# ---------------------------------------------------------------- results
def fetch_results(seasons_start_years) -> pd.DataFrame:
    frames = []
    for y in seasons_start_years:
        season = config.season_label(y)
        text = _get(config.football_data_url(y), as_json=False)
        df = pd.read_csv(io.StringIO(text), encoding="latin-1")
        if df.empty or "FTR" not in df.columns:
            logger.info(f"{season}: no results published yet")
            continue
        df = df[[c for c in RESULT_COLS if c in df.columns]].dropna(subset=["Date", "HomeTeam", "AwayTeam", "FTR"])
        df["Date"] = pd.to_datetime(df["Date"], dayfirst=True)
        df["HomeTeam"] = normalize_team_names(df["HomeTeam"])
        df["AwayTeam"] = normalize_team_names(df["AwayTeam"])
        df["season"] = season
        frames.append(df)
        logger.info(f"{season}: {len(df)} results")
    if not frames:
        raise RuntimeError("No match results could be loaded from football-data.co.uk")
    return pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------------- FotMob
def _find_xg(node):
    """Depth-first search for the first populated 'expected_goals' stat."""
    if isinstance(node, dict):
        if node.get("key") == "expected_goals":
            raw = node.get("rawStats")
            if isinstance(raw, list) and len(raw) == 2 and all(
                    isinstance(v, dict) and v.get("value") is not None for v in raw):
                return float(raw[0]["value"]), float(raw[1]["value"])
        for v in node.values():
            found = _find_xg(v)
            if found:
                return found
    elif isinstance(node, list):
        for v in node:
            found = _find_xg(v)
            if found:
                return found
    return None


def fetch_fotmob_fixtures(seasons_start_years) -> pd.DataFrame:
    """Every fixture (played and scheduled) FotMob lists for the given seasons."""
    rows = []
    for y in seasons_start_years:
        data = _get("https://www.fotmob.com/api/data/leagues",
                    {"id": config.FOTMOB_LEAGUE_ID, "season": config.fotmob_season(y)})
        for m in data["fixtures"]["allMatches"]:
            st = m["status"]
            if st.get("cancelled"):
                continue
            hg = ag = np.nan
            if st.get("finished") and st.get("scoreStr"):
                try:
                    hg, ag = (int(x) for x in st["scoreStr"].replace(" ", "").split("-"))
                except ValueError:
                    pass
            rows.append({"fotmob_id": int(m["id"]), "season": config.season_label(y),
                         "round": m.get("round"), "kickoff_utc": st["utcTime"],
                         "home_team_fotmob": m["home"]["name"], "away_team_fotmob": m["away"]["name"],
                         "finished": bool(st.get("finished")), "fotmob_home_goals": hg,
                         "fotmob_away_goals": ag})
    fx = pd.DataFrame(rows)
    fx["HomeTeam"] = normalize_team_names(fx["home_team_fotmob"])
    fx["AwayTeam"] = normalize_team_names(fx["away_team_fotmob"])
    fx["kickoff_utc"] = pd.to_datetime(fx["kickoff_utc"], utc=True)
    # A fixture can be listed twice (e.g. a postponed match); keep the latest entry
    fx = fx.sort_values("kickoff_utc").drop_duplicates(["season", "HomeTeam", "AwayTeam"], keep="last")
    return fx.reset_index(drop=True)


def update_xg_cache(fixtures: pd.DataFrame) -> pd.DataFrame:
    """Fetch match xG for finished fixtures missing from the cache; return the cache."""
    cache = pd.read_csv(XG_CACHE) if os.path.exists(XG_CACHE) else pd.DataFrame(
        columns=["fotmob_id", "home_xG", "away_xG"])
    have = set(cache["fotmob_id"].astype(int)) if len(cache) else set()
    todo = fixtures[fixtures["finished"] & ~fixtures["fotmob_id"].isin(have)]["fotmob_id"].tolist()
    if todo:
        logger.info(f"Fetching xG for {len(todo)} new matches from FotMob")

        def one(mid):
            d = _get("https://www.fotmob.com/api/data/matchDetails", {"matchId": mid})
            xg = _find_xg(d.get("content", {}).get("stats", {}))
            return {"fotmob_id": mid, "home_xG": xg[0] if xg else np.nan, "away_xG": xg[1] if xg else np.nan}

        with ThreadPoolExecutor(max_workers=6) as pool:
            new = pd.DataFrame(list(pool.map(one, todo)))
        # Don't cache "no xG yet" (FotMob can publish stats a little after full time)
        new = new.dropna(subset=["home_xG", "away_xG"])
        cache = new if cache.empty else pd.concat([cache, new], ignore_index=True)
        os.makedirs(config.DATA_DIR, exist_ok=True)
        cache.sort_values("fotmob_id").to_csv(XG_CACHE, index=False)
    return cache


# ---------------------------------------------------------------- merge
def load_match_data(today=None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (played matches with xG, all FotMob fixtures)."""
    years = list(range(config.FIRST_SEASON_START, config.current_season_start(today) + 1))
    results = fetch_results(years)
    fixtures = fetch_fotmob_fixtures(years)

    unknown = (set(fixtures["HomeTeam"]) | set(fixtures["AwayTeam"])) - (set(results["HomeTeam"]) | set(results["AwayTeam"]))
    current = config.season_label(years[-1])
    unknown_played = unknown & set(fixtures.loc[fixtures["finished"] & (fixtures["season"] != current), "HomeTeam"])
    if unknown_played:
        raise RuntimeError(f"FotMob team names with no football-data match (add to TEAM_NAME_ALIASES): {sorted(unknown_played)}")

    xg = update_xg_cache(fixtures)
    fx = fixtures.merge(xg, on="fotmob_id", how="left")
    df = results.merge(fx[["season", "HomeTeam", "AwayTeam", "fotmob_id", "round", "home_xG", "away_xG"]],
                       on=["season", "HomeTeam", "AwayTeam"], how="left", validate="one_to_one")

    missing = df["home_xG"].isna() | df["away_xG"].isna()
    if missing.mean() > MAX_MISSING_XG_SHARE:
        raise RuntimeError(f"{missing.sum()} of {len(df)} matches have no xG; refusing to train on a proxy. "
                           f"Examples:\n{df.loc[missing, ['Date', 'HomeTeam', 'AwayTeam']].head(10)}")
    if missing.any():
        # A handful of very recent matches can lack published xG; goals are the stand-in
        logger.warning(f"{missing.sum()} matches without xG; using goals as their xG")
        df.loc[missing, "home_xG"] = df.loc[missing, "FTHG"].astype(float)
        df.loc[missing, "away_xG"] = df.loc[missing, "FTAG"].astype(float)
    df["xg_is_proxy"] = missing

    df["FTR"] = df["FTR"].map({"H": 2, "A": 1, "D": 0})
    if df["FTR"].isna().any():
        raise RuntimeError("Unexpected FTR values in results")
    df["FTR"] = df["FTR"].astype(int)
    df = df.sort_values("Date", kind="stable").reset_index(drop=True)
    logger.info(f"Loaded {len(df)} played matches ({df['season'].nunique()} seasons) with xG")
    return df, fixtures
