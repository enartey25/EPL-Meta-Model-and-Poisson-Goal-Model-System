import pandas as pd
import numpy as np
import streamlit as st
import requests
import io
import warnings
warnings.filterwarnings("ignore")

# ─── Team name normalization map ──────────────────────────────────────────────
TEAM_NAME_MAP = {
    # football-data.co.uk → canonical name (same as team_map keys)
    "Arsenal": "Arsenal",
    "Aston Villa": "Aston Villa",
    "Bournemouth": "Bournemouth",
    "Brentford": "Brentford",
    "Brighton": "Brighton",
    "Brighton & Hove Albion": "Brighton",
    "Burnley": "Burnley",
    "Chelsea": "Chelsea",
    "Crystal Palace": "Crystal Palace",
    "Everton": "Everton",
    "Fulham": "Fulham",
    "Ipswich": "Ipswich",
    "Ipswich Town": "Ipswich",
    "Leeds": "Leeds",
    "Leeds United": "Leeds",
    "Leicester": "Leicester",
    "Leicester City": "Leicester",
    "Liverpool": "Liverpool",
    "Luton": "Luton",
    "Luton Town": "Luton",
    "Man City": "Man City",
    "Manchester City": "Man City",
    "Man United": "Man United",
    "Manchester United": "Man United",
    "Newcastle": "Newcastle",
    "Newcastle United": "Newcastle",
    "Nott'm Forest": "Nott'm Forest",
    "Nottingham Forest": "Nott'm Forest",
    "Nottm Forest": "Nott'm Forest",
    "Sheffield United": "Sheffield United",
    "Sheffield Utd": "Sheffield United",
    "Southampton": "Southampton",
    "Sunderland": "Sunderland",
    "Tottenham": "Tottenham",
    "Tottenham Hotspur": "Tottenham",
    "West Ham": "West Ham",
    "West Ham United": "West Ham",
    "Wolves": "Wolves",
    "Wolverhampton Wanderers": "Wolves",
    # Understat names
    "Manchester City": "Man City",
    "Manchester United": "Man United",
    "Newcastle United": "Newcastle",
    "Nottingham Forest": "Nott'm Forest",
    "Brighton": "Brighton",
    "Ipswich": "Ipswich",
}

# Understat → canonical
UNDERSTAT_MAP = {
    "Manchester City": "Man City",
    "Manchester United": "Man United",
    "Newcastle United": "Newcastle",
    "Nottingham Forest": "Nott'm Forest",
    "Brighton": "Brighton",
    "Wolverhampton Wanderers": "Wolves",
    "Tottenham Hotspur": "Tottenham",
    "Leicester City": "Leicester",
    "Leeds United": "Leeds",
    "Ipswich": "Ipswich",
    "Luton": "Luton",
    "Sheffield United": "Sheffield United",
    "West Ham": "West Ham",
}

SEASON_URL_MAP = {
    "2022/23": "https://www.football-data.co.uk/mmz4281/2223/E0.csv",
    "2023/24": "https://www.football-data.co.uk/mmz4281/2324/E0.csv",
    "2024/25": "https://www.football-data.co.uk/mmz4281/2425/E0.csv",
    "2025/26": "https://www.football-data.co.uk/mmz4281/2526/E0.csv",
}


def normalize_team(name: str) -> str:
    return TEAM_NAME_MAP.get(name, UNDERSTAT_MAP.get(name, name))


def normalize_understat(name: str) -> str:
    return UNDERSTAT_MAP.get(name, TEAM_NAME_MAP.get(name, name))


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_football_data() -> pd.DataFrame:
    """Fetch match stats from football-data.co.uk for 4 seasons."""
    dfs = []
    cols_needed = [
        "Date", "HomeTeam", "AwayTeam",
        "FTHG", "FTAG", "FTR",
        "HTHG", "HTAG", "HTR",
        "HS", "AS", "HST", "AST",
        "HF", "AF", "HC", "AC",
        "HY", "AY", "HR", "AR",
    ]
    for season, url in SEASON_URL_MAP.items():
        try:
            resp = requests.get(url, timeout=15, verify=True)
            resp.raise_for_status()
            df = pd.read_csv(io.StringIO(resp.text), encoding="latin-1")
            df = df[[c for c in cols_needed if c in df.columns]].copy()
            df["Season"] = season
            df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")
            df = df.dropna(subset=["Date", "HomeTeam", "AwayTeam", "FTR"])
            df["HomeTeam"] = df["HomeTeam"].map(normalize_team).fillna(df["HomeTeam"])
            df["AwayTeam"] = df["AwayTeam"].map(normalize_team).fillna(df["AwayTeam"])
            dfs.append(df)
        except Exception:
            st.warning(f"Could not load season {season} data. Please check your connection.")

    if not dfs:
        return pd.DataFrame()
    return pd.concat(dfs, ignore_index=True)


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_xg_data() -> pd.DataFrame:
    """Fetch xG data from understat for 2022-2025 seasons."""
    try:
        from understatapi import UnderstatClient
        LEAGUE = "EPL"
        SEASONS = ["2022", "2023", "2024", "2025"]
        cleaned_matches = []

        with UnderstatClient() as understat:
            for season in SEASONS:
                try:
                    fixtures = understat.league(league=LEAGUE).get_match_data(season=season)
                    for match in fixtures:
                        if not match.get("isResult"):
                            continue
                        match_info = {
                            "season": season,
                            "date": pd.to_datetime(match["datetime"]).date(),
                            "home_team": normalize_understat(match["h"]["title"]),
                            "away_team": normalize_understat(match["a"]["title"]),
                            "home_xG": float(match["xG"]["h"]),
                            "away_xG": float(match["xG"]["a"]),
                        }
                        cleaned_matches.append(match_info)
                except Exception:
                    pass

        if not cleaned_matches:
            return pd.DataFrame()

        df_xg = pd.DataFrame(cleaned_matches)
        df_xg.loc[df_xg["season"] == "2022", "season"] = "2022/23"
        df_xg.loc[df_xg["season"] == "2023", "season"] = "2023/24"
        df_xg.loc[df_xg["season"] == "2024", "season"] = "2024/25"
        df_xg.loc[df_xg["season"] == "2025", "season"] = "2025/26"
        cutoff = pd.to_datetime("2026-05-24").date()
        df_xg = df_xg[df_xg["date"] < cutoff].copy()
        return df_xg

    except Exception as e:
        st.warning(f"Could not load xG data: {e}")
        return pd.DataFrame()


@st.cache_data(ttl=3600, show_spinner=False)
def load_full_dataset() -> pd.DataFrame:
    """Merge football-data stats with understat xG data."""
    df_stats = fetch_football_data()
    df_xg = fetch_xg_data()

    if df_stats.empty:
        return pd.DataFrame()

    df_stats["_date_key"] = df_stats["Date"].dt.date.astype(str)
    df_stats["_home_key"] = df_stats["HomeTeam"].str.lower().str.strip()
    df_stats["_away_key"] = df_stats["AwayTeam"].str.lower().str.strip()

    if not df_xg.empty:
        df_xg["_date_key"] = df_xg["date"].astype(str)
        df_xg["_home_key"] = df_xg["home_team"].str.lower().str.strip()
        df_xg["_away_key"] = df_xg["away_team"].str.lower().str.strip()

        df_merged = df_stats.merge(
            df_xg[["_date_key", "_home_key", "_away_key", "home_xG", "away_xG"]],
            on=["_date_key", "_home_key", "_away_key"],
            how="left",
        )
    else:
        df_merged = df_stats.copy()
        df_merged["home_xG"] = np.nan
        df_merged["away_xG"] = np.nan

    df_merged.drop(columns=["_date_key", "_home_key", "_away_key"], inplace=True)
    return df_merged


def compute_league_table(df: pd.DataFrame, season: str = None) -> pd.DataFrame:
    """Compute a league table from match results."""
    if season and season != "All Seasons":
        df = df[df["Season"] == season].copy()

    if df.empty:
        return pd.DataFrame()

    teams = set(df["HomeTeam"].tolist() + df["AwayTeam"].tolist())
    records = []

    for team in sorted(teams):
        home = df[df["HomeTeam"] == team]
        away = df[df["AwayTeam"] == team]

        hw = len(home[home["FTR"] == "H"])
        hd = len(home[home["FTR"] == "D"])
        hl = len(home[home["FTR"] == "A"])
        aw = len(away[away["FTR"] == "A"])
        ad = len(away[away["FTR"] == "D"])
        al = len(away[away["FTR"] == "H"])

        gf_h = home["FTHG"].sum() if "FTHG" in home.columns else 0
        ga_h = home["FTAG"].sum() if "FTAG" in home.columns else 0
        gf_a = away["FTAG"].sum() if "FTAG" in away.columns else 0
        ga_a = away["FTHG"].sum() if "FTHG" in away.columns else 0

        p = hw + hd + hl + aw + ad + al
        w = hw + aw
        d = hd + ad
        l = hl + al
        gf = int(gf_h + gf_a)
        ga = int(ga_h + ga_a)
        pts = w * 3 + d

        records.append({
            "Team": team,
            "P": p, "W": w, "D": d, "L": l,
            "GF": gf, "GA": ga, "GD": gf - ga, "Pts": pts,
        })

    table = pd.DataFrame(records).sort_values(["Pts", "GD", "GF"], ascending=False)
    table.insert(0, "#", range(1, len(table) + 1))
    return table.reset_index(drop=True)
