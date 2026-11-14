"""Read the data files the pipeline exports (data/*.csv, data/metrics.json).

The app never scrapes at request time: the weekly pipeline fetches results
(football-data.co.uk) and xG (FotMob) and commits them here.
"""
import json
import os

import pandas as pd

try:
    import streamlit as st
    cache = st.cache_data(ttl=3600, show_spinner=False)
except ImportError:  # usable outside Streamlit
    def cache(f):
        return f

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


def _csv(name: str, dates=()) -> pd.DataFrame:
    path = os.path.join(DATA_DIR, name)
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return pd.DataFrame()
    df = pd.read_csv(path)
    for c in dates:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], utc=c.endswith("_utc"))
    return df


@cache
def load_full_dataset() -> pd.DataFrame:
    df = _csv("matches.csv", dates=("Date",))
    if not df.empty:
        df["Season"] = df["season"]
    return df


@cache
def fetch_xg_data() -> pd.DataFrame:
    df = load_full_dataset()
    if df.empty:
        return df
    return df.rename(columns={"Date": "date", "HomeTeam": "home_team", "AwayTeam": "away_team"})[
        ["date", "season", "home_team", "away_team", "home_xG", "away_xG"]]


@cache
def load_upcoming() -> pd.DataFrame:
    return _csv("upcoming.csv", dates=("kickoff_utc",))


@cache
def load_track_record() -> pd.DataFrame:
    df = _csv("track_record.csv", dates=("Date",))
    if not df.empty and ("round" not in df.columns or df["round"].isna().all()):
        # Older exports had no gameweek: approximate with blocks of 10 matches per season
        df = df.sort_values("Date", kind="stable")
        df["round"] = df.groupby("season").cumcount() // 10 + 1
    return df


@cache
def load_forecast_log() -> pd.DataFrame:
    return _csv("forecast_log.csv", dates=("kickoff_utc",))


@cache
def load_metrics() -> dict:
    path = os.path.join(DATA_DIR, "metrics.json")
    if not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def compute_league_table(df: pd.DataFrame, season: str = None) -> pd.DataFrame:
    """League table from match results (FTR as H / D / A)."""
    if season and season != "All Seasons":
        df = df[df["Season"] == season]
    if df.empty:
        return pd.DataFrame()
    rows = []
    for team in sorted(set(df["HomeTeam"]) | set(df["AwayTeam"])):
        h, a = df[df["HomeTeam"] == team], df[df["AwayTeam"] == team]
        w = (h["FTR"] == "H").sum() + (a["FTR"] == "A").sum()
        d = (h["FTR"] == "D").sum() + (a["FTR"] == "D").sum()
        l = (h["FTR"] == "A").sum() + (a["FTR"] == "H").sum()
        gf = int(h["FTHG"].sum() + a["FTAG"].sum())
        ga = int(h["FTAG"].sum() + a["FTHG"].sum())
        xgf = float(h["home_xG"].sum() + a["away_xG"].sum())
        xga = float(h["away_xG"].sum() + a["home_xG"].sum())
        rows.append({"Team": team, "P": int(w + d + l), "W": int(w), "D": int(d), "L": int(l),
                     "GF": gf, "GA": ga, "GD": gf - ga, "xGF": round(xgf, 1), "xGA": round(xga, 1),
                     "Pts": int(3 * w + d)})
    table = pd.DataFrame(rows).sort_values(["Pts", "GD", "GF"], ascending=False)
    table.insert(0, "#", range(1, len(table) + 1))
    return table.reset_index(drop=True)
