"""Weekly pipeline: ingest results and xG, rebuild features and Poisson strengths,
retrain the stack every ~5 gameweeks (or when the club list or asset format
changes), score the track record, forecast upcoming fixtures, and export.

    python -m pipeline.run_pipeline               # weekly run
    python -m pipeline.run_pipeline --force-retrain
    python -m pipeline.run_pipeline --dry-run     # everything except writing files
"""
import argparse
import logging
import os
import sys
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import inference_utils as iu  # noqa: E402
from pipeline import config  # noqa: E402
from pipeline.build_features import build_features, build_team_map, encode  # noqa: E402
from pipeline.export_assets import load_existing_assets, write_assets, write_csv, write_json  # noqa: E402
from pipeline.fetch_data import load_match_data  # noqa: E402
from pipeline.train_models import fit_stack, holdout_evaluation, scores, season_record  # noqa: E402
from pipeline.update_poisson import add_poisson_strengths  # noqa: E402
from pipeline.validate import parity_check, sanity_check_assets  # noqa: E402

logger = logging.getLogger("pipeline")
FORECAST_LOG = "forecast_log.csv"
UPCOMING_WINDOW_DAYS = 10


def _needs_retrain(existing, team_map, n_matches, force, threshold):
    if force:
        return True, "forced"
    if existing is None:
        return True, "no existing assets"
    if existing.get("format_version") != config.ASSET_FORMAT_VERSION:
        return True, "asset format changed"
    if existing.get("team_map") != team_map:
        return True, "club list changed (team codes shift)"
    new = n_matches - existing.get("last_trained_count", 0)
    if new >= threshold:
        return True, f"{new} new matches since last retrain (threshold {threshold})"
    return False, f"{new} new matches since last retrain; next full retrain in {threshold - new}"


def _record_summary(rec: pd.DataFrame) -> dict:
    y = rec["FTR"].to_numpy()
    model = scores(y, rec[["P_Draw", "P_Away", "P_Home"]].to_numpy())
    null = scores(y, rec[["Null_Draw", "Null_Away", "Null_Home"]].to_numpy())
    fav_hit = float((rec[["P_Draw", "P_Away", "P_Home"]].to_numpy().argmax(1) == y).mean())
    return {"model": model, "null_baseline": null, "favourite_hit_rate": fav_hit}


def _upcoming(fixtures, current_season, now):
    up = fixtures[(~fixtures["finished"]) & (fixtures["season"] == current_season)
                  & (fixtures["kickoff_utc"] >= now - pd.Timedelta(hours=3))].sort_values("kickoff_utc")
    if up.empty:
        return up
    first = up["kickoff_utc"].iloc[0]
    return up[up["kickoff_utc"] <= first + pd.Timedelta(days=UPCOMING_WINDOW_DAYS)]


def run(force_retrain=False, dry_run=False, retrain_threshold=config.RETRAIN_BATCH_THRESHOLD) -> bool:
    t0 = time.time()
    now = pd.Timestamp.now(tz="UTC")
    current_season = config.season_label(config.current_season_start())
    try:
        # 1. Data
        logger.info("[1/6] Ingesting results (football-data.co.uk) and xG (FotMob)")
        played, fixtures = load_match_data()

        # 2. Features
        logger.info("[2/6] Building pre-match features")
        team_map = build_team_map(played, fixtures)
        enc = encode(played, team_map)
        enc["HomeTeam_name"], enc["AwayTeam_name"] = played["HomeTeam"].values, played["AwayTeam"].values
        df, state = build_features(enc)
        dfp, poisson_state = add_poisson_strengths(df)
        parity_check(df, team_map)

        seasons = list(dict.fromkeys(df["season"]))  # chronological
        complete = [s for s in seasons if s != current_season]

        # 3. Models
        existing = load_existing_assets()
        retrain, why = _needs_retrain(existing, team_map, len(df), force_retrain, retrain_threshold)
        if retrain:
            logger.info(f"[3/6] Full retrain: {why}")
            rf, meta, n_meta = fit_stack(df, dfp, seasons)
            trained_at, trained_count = now.isoformat(), len(df)
        else:
            logger.info(f"[3/6] Weekly state update: {why}")
            rf, meta = existing["rf_model"], existing["meta_model"]
            n_meta = existing.get("meta_training_rows")
            trained_at, trained_count = existing["trained_at"], existing["last_trained_count"]

        assets = {
            "format_version": config.ASSET_FORMAT_VERSION,
            "meta_model": meta, "meta_inputs": list(config.META_INPUTS), "meta_transform": config.META_TRANSFORM,
            "rf_model": rf, "xgb_model": None,
            "team_map": team_map, "state": state, "poisson_state": poisson_state,
            "elo_params": config.ELO_PARAMS, "window": config.FORM_WINDOW,
            "final_features": config.FINAL_FEATURES, "current_season": current_season,
            "trained_at": trained_at, "last_trained_count": trained_count, "meta_training_rows": n_meta,
            "state_updated_at": now.isoformat(), "data_through": df["Date"].max().date().isoformat(),
        }
        sanity_check_assets(assets)

        # 4. Evaluation and track record
        logger.info("[4/6] Scoring the holdout season and the walk-forward track record")
        holdout = holdout_evaluation(df, dfp, complete)
        record_seasons = [s for s in seasons if seasons.index(s) >= config.MIN_PRIOR_SEASONS_FOR_META + 1]
        record = pd.concat([season_record(df, dfp, s, seasons) for s in record_seasons], ignore_index=True)
        record_summary = {s: _record_summary(record[record["season"] == s]) for s in record_seasons}

        # 5. Upcoming fixtures
        logger.info("[5/6] Forecasting upcoming fixtures")
        up = _upcoming(fixtures, current_season, now)
        upcoming = pd.DataFrame()
        if not up.empty:
            fx = up[["HomeTeam", "AwayTeam", "season"]].reset_index(drop=True)
            outcomes, _ = iu.predict_outcomes(fx, assets)
            upcoming, _ = iu.predict_scores(outcomes, assets, config.MAX_GOALS)
            upcoming.insert(0, "kickoff_utc", up["kickoff_utc"].values)
            upcoming.insert(0, "round", up["round"].values)
            upcoming.insert(0, "fotmob_id", up["fotmob_id"].values)

        # 6. Export
        keep = [c for c in ["Date", "season", "FTHG", "FTAG", "FTR", "HS", "AS", "HST", "AST", "HC", "AC",
                            "HY", "AY", "HR", "AR", "home_xG", "away_xG", "xg_is_proxy",
                            "Home_PreMatch_Elo", "Away_PreMatch_Elo"] if c in df.columns]
        played_out = df[keep].copy()
        played_out.insert(1, "HomeTeam", df["HomeTeam_name"].values)
        played_out.insert(2, "AwayTeam", df["AwayTeam_name"].values)
        played_out["FTR"] = played_out["FTR"].map({2: "H", 1: "A", 0: "D"})
        played_out["Lambda_Home"] = dfp["Lambda_Home"].round(3).values
        played_out["Lambda_Away"] = dfp["Lambda_Away"].round(3).values
        inv = {v: k for k, v in team_map.items()}
        elo_now = {inv[k]: round(float(v), 1) for k, v in state["elo"].items()}
        metrics = {
            "generated_at": now.isoformat(), "data_through": assets["data_through"],
            "current_season": current_season, "matches": len(df), "trained_at": trained_at,
            "retrain_reason": why, "meta_inputs": list(config.META_INPUTS),
            "holdout": holdout, "track_record": record_summary, "elo": elo_now,
            "xg_proxy_matches": int(played["xg_is_proxy"].sum()),
        }
        if dry_run:
            logger.info("[6/6] Dry run: nothing written")
        else:
            logger.info("[6/6] Exporting")
            write_assets(assets)
            write_csv(played_out, "matches.csv")
            write_csv(record, "track_record.csv")
            write_csv(upcoming, "upcoming.csv")
            write_json(metrics, "metrics.json")
            if not upcoming.empty:
                log_path = os.path.join(config.DATA_DIR, FORECAST_LOG)
                log = pd.read_csv(log_path) if os.path.exists(log_path) else pd.DataFrame(columns=["fotmob_id"])
                new = upcoming[~upcoming["fotmob_id"].isin(log["fotmob_id"])].copy()
                if len(new):
                    new.insert(0, "logged_at", now.isoformat())
                    new["model_trained_at"] = trained_at
                    write_csv(pd.concat([log, new], ignore_index=True), FORECAST_LOG)
        logger.info(f"Pipeline finished in {time.time() - t0:.1f}s")
        return True
    except Exception as e:
        logger.exception(f"Pipeline failed: {e}")
        return False


def main():
    ap = argparse.ArgumentParser(description="EPL forecasting pipeline")
    ap.add_argument("--force-retrain", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--retrain-threshold", type=int, default=config.RETRAIN_BATCH_THRESHOLD)
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args()
    logging.basicConfig(level=args.log_level.upper(), format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
                        datefmt="%H:%M:%S")
    sys.exit(0 if run(args.force_retrain, args.dry_run, args.retrain_threshold) else 1)


if __name__ == "__main__":
    main()
