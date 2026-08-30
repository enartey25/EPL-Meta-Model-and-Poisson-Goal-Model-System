import pandas as pd
import numpy as np
import requests
import io
import warnings

try:
    import streamlit as st
except ImportError:
    class _DummyStreamlit:
        def cache_data(self, *args, **kwargs):
            def decorator(f):
                return f
            return decorator
    st = _DummyStreamlit()

warnings.filterwarnings("ignore")

# ─── Team name normalization map ──────────────────────────────────────────────
TEAM_NAME_MAP = {
    # Canonical 25 teams
    "Arsenal": "Arsenal",
    "Arsenal FC": "Arsenal",
    "Aston Villa": "Aston Villa",
    "Aston Villa FC": "Aston Villa",
    "Bournemouth": "Bournemouth",
    "AFC Bournemouth": "Bournemouth",
    "Brentford": "Brentford",
    "Brentford FC": "Brentford",
    "Brighton": "Brighton",
    "Brighton & Hove Albion": "Brighton",
    "Brighton and Hove Albion": "Brighton",
    "Burnley": "Burnley",
    "Burnley FC": "Burnley",
    "Chelsea": "Chelsea",
    "Chelsea FC": "Chelsea",
    "Crystal Palace": "Crystal Palace",
    "Crystal Palace FC": "Crystal Palace",
    "Everton": "Everton",
    "Everton FC": "Everton",
    "Fulham": "Fulham",
    "Fulham FC": "Fulham",
    "Ipswich": "Ipswich",
    "Ipswich Town": "Ipswich",
    "Ipswich Town FC": "Ipswich",
    "Leeds": "Leeds",
    "Leeds United": "Leeds",
    "Leeds United FC": "Leeds",
    "Leicester": "Leicester",
    "Leicester City": "Leicester",
    "Leicester City FC": "Leicester",
    "Liverpool": "Liverpool",
    "Liverpool FC": "Liverpool",
    "Luton": "Luton",
    "Luton Town": "Luton",
    "Luton Town FC": "Luton",
    "Man City": "Man City",
    "Manchester City": "Man City",
    "Manchester City FC": "Man City",
    "Man United": "Man United",
    "Manchester United": "Man United",
    "Manchester United FC": "Man United",
    "Newcastle": "Newcastle",
    "Newcastle United": "Newcastle",
    "Newcastle United FC": "Newcastle",
    "Nott'm Forest": "Nott'm Forest",
    "Nottingham Forest": "Nott'm Forest",
    "Nottm Forest": "Nott'm Forest",
    "Sheffield United": "Sheffield United",
    "Sheffield Utd": "Sheffield United",
    "Sheffield United FC": "Sheffield United",
    "Southampton": "Southampton",
    "Southampton FC": "Southampton",
    "Sunderland": "Sunderland",
    "Sunderland AFC": "Sunderland",
    "Tottenham": "Tottenham",
    "Tottenham Hotspur": "Tottenham",
    "Tottenham Hotspur FC": "Tottenham",
    "Spurs": "Tottenham",
    "West Ham": "West Ham",
    "West Ham United": "West Ham",
    "West Ham United FC": "West Ham",
    "Wolves": "Wolves",
    "Wolverhampton": "Wolves",
    "Wolverhampton Wanderers": "Wolves",
    "Wolverhampton Wanderers FC": "Wolves",
    # Additional clubs (Championship / promoted)
    "Norwich": "Norwich",
    "Norwich City": "Norwich",
    "Watford": "Watford",
    "Watford FC": "Watford",
    "West Brom": "West Brom",
    "West Bromwich": "West Brom",
    "West Bromwich Albion": "West Brom",
    "Middlesbrough": "Middlesbrough",
    "Coventry": "Coventry",
    "Coventry City": "Coventry",
    "Hull": "Hull",
    "Hull City": "Hull",
    "Stoke": "Stoke",
    "Stoke City": "Stoke",
    "Blackburn": "Blackburn",
    "Blackburn Rovers": "Blackburn",
    "Preston": "Preston",
    "Preston North End": "Preston",
    "Derby": "Derby",
    "Derby County": "Derby",
    "QPR": "QPR",
    "Queens Park Rangers": "QPR",
    "Millwall": "Millwall",
    "Bristol City": "Bristol City",
    "Swansea": "Swansea",
    "Swansea City": "Swansea",
    "Cardiff": "Cardiff",
    "Cardiff City": "Cardiff",
    "Portsmouth": "Portsmouth",
    "Oxford United": "Oxford United",
    "Plymouth": "Plymouth",
    "Plymouth Argyle": "Plymouth",
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
    "Ipswich Town": "Ipswich",
    "Luton": "Luton",
    "Luton Town": "Luton",
    "Sheffield United": "Sheffield United",
    "West Ham": "West Ham",
    "West Ham United": "West Ham",
    "Burnley": "Burnley",
    "Southampton": "Southampton",
    "Sunderland": "Sunderland",
    "Coventry": "Coventry",
    "Coventry City": "Coventry",
    "Hull": "Hull",
    "Hull City": "Hull",
}

SEASON_URL_MAP = {
    "2022/23": "https://www.football-data.co.uk/mmz4281/2223/E0.csv",
    "2023/24": "https://www.football-data.co.uk/mmz4281/2324/E0.csv",
    "2024/25": "https://www.football-data.co.uk/mmz4281/2425/E0.csv",
    "2025/26": "https://www.football-data.co.uk/mmz4281/2526/E0.csv",
}


def normalize_team(name: str) -> str:
    if not isinstance(name, str):
        return str(name)
    cleaned = name.strip()
    return TEAM_NAME_MAP.get(cleaned, UNDERSTAT_MAP.get(cleaned, cleaned))


def normalize_understat(name: str) -> str:
    if not isinstance(name, str):
        return str(name)
    cleaned = name.strip()
    return UNDERSTAT_MAP.get(cleaned, TEAM_NAME_MAP.get(cleaned, cleaned))


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_football_data() -> pd.DataFrame:
    """Fetch match stats from football-data.co.uk for multiple seasons."""
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
            if resp.status_code == 200:
                df = pd.read_csv(io.StringIO(resp.text), encoding="latin-1")
                df = df[[c for c in cols_needed if c in df.columns]].copy()
                df["Season"] = season
                df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")
                df = df.dropna(subset=["Date", "HomeTeam", "AwayTeam", "FTR"])
                df["HomeTeam"] = df["HomeTeam"].astype(str).str.strip().map(normalize_team)
                df["AwayTeam"] = df["AwayTeam"].astype(str).str.strip().map(normalize_team)
                dfs.append(df)
        except Exception:
            pass

    if not dfs:
        return pd.DataFrame()
    return pd.concat(dfs, ignore_index=True)


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_xg_data() -> pd.DataFrame:
    """Fetch xG data from understat for recent seasons."""
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
        df_xg = df_xg[df_xg["date"] <= cutoff].copy()
        return df_xg

    except Exception:
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

    teams = set(df["HomeTeam"].dropna().tolist() + df["AwayTeam"].dropna().tolist())
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
