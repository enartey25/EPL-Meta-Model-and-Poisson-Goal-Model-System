"""
Leakage-free Poisson goal expectancy calculation module.
Computes expanding-mean home/away goal expectancies for the generative Poisson score grid.
"""
import logging
import pandas as pd

logger = logging.getLogger(__name__)


def compute_leakage_free_poisson(df_input: pd.DataFrame) -> pd.DataFrame:
    """Calculate leakage-free rolling goal expectancy strengths using blended actual goals and xG."""
    temp_df = df_input.sort_values(["Date"]).copy()

    # Blended goal and xG metrics
    temp_df["Home_Perf_For"] = (temp_df["FTHG"] + temp_df["home_xG"]) / 2.0
    temp_df["Home_Perf_Against"] = (temp_df["FTAG"] + temp_df["away_xG"]) / 2.0
    temp_df["Away_Perf_For"] = (temp_df["FTAG"] + temp_df["away_xG"]) / 2.0
    temp_df["Away_Perf_Against"] = (temp_df["FTHG"] + temp_df["home_xG"]) / 2.0

    # Group by team and compute expanding prior means (shift excludes current match)
    home_col = "HomeTeam_orig" if "HomeTeam_orig" in temp_df.columns else "HomeTeam"
    away_col = "AwayTeam_orig" if "AwayTeam_orig" in temp_df.columns else "AwayTeam"

    temp_df["Home_Avg_Scored"] = temp_df.groupby(home_col)["Home_Perf_For"].transform(
        lambda x: x.shift().expanding().mean()
    )
    temp_df["Home_Avg_Conceded"] = temp_df.groupby(home_col)["Home_Perf_Against"].transform(
        lambda x: x.shift().expanding().mean()
    )
    temp_df["Away_Avg_Scored"] = temp_df.groupby(away_col)["Away_Perf_For"].transform(
        lambda x: x.shift().expanding().mean()
    )
    temp_df["Away_Avg_Conceded"] = temp_df.groupby(away_col)["Away_Perf_Against"].transform(
        lambda x: x.shift().expanding().mean()
    )

    # Fill initial matches with empirical league priors
    league_avg_home = (temp_df["FTHG"].mean() + temp_df["home_xG"].mean()) / 2.0
    league_avg_away = (temp_df["FTAG"].mean() + temp_df["away_xG"].mean()) / 2.0

    temp_df["Home_Avg_Scored"] = temp_df["Home_Avg_Scored"].fillna(league_avg_home)
    temp_df["Home_Avg_Conceded"] = temp_df["Home_Avg_Conceded"].fillna(league_avg_away)
    temp_df["Away_Avg_Scored"] = temp_df["Away_Avg_Scored"].fillna(league_avg_away)
    temp_df["Away_Avg_Conceded"] = temp_df["Away_Avg_Conceded"].fillna(league_avg_home)

    logger.info(f"Poisson goal expectancy computed. League priors: Home={league_avg_home:.2f}, Away={league_avg_away:.2f}")
    return temp_df
