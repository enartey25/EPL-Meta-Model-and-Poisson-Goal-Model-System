"""
Master orchestrator for the EPL Match Predictor pipeline.
Features a dual-cadence architecture:
  - Weekly: Ingests completed matches, updates sequential Elo and rolling xG state.
  - Every 5 Gameweeks (~50 fixtures): Fully retrains XGBoost, Random Forest, and Stacking Meta-Learner.
"""
import argparse
import logging
import os
import pickle
import sys
import time

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from pipeline.fetch_data import load_and_merge_pipeline_data
from pipeline.build_features import build_pipeline_features
from pipeline.update_poisson import compute_leakage_free_poisson
from pipeline.train_models import train_stacking_pipeline
from pipeline.export_assets import export_assets_bundle

RETRAIN_BATCH_THRESHOLD = 50  # 5 Gameweeks = 50 fixtures


def setup_logging(log_level: str = "INFO") -> None:
    """Configure structured logging."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def load_existing_models_and_count(asset_path: str) -> tuple[dict, int]:
    """Load existing base models and tracking metadata if available."""
    if not os.path.exists(asset_path):
        return None, 0
    try:
        with open(asset_path, "rb") as f:
            assets = pickle.load(f)
        models_dict = {
            "stacked_meta_model": assets.get("stacked_meta_model"),
            "best_xgb_model": assets.get("best_xgb_model"),
            "best_rf_model": assets.get("best_rf_model"),
        }
        last_count = assets.get("last_trained_count", len(assets.get("df_poisson", [])))
        return models_dict, last_count
    except Exception:
        return None, 0


def run(
    output_path: str = "epl_predictor_assets.pkl",
    dry_run: bool = False,
    force_retrain: bool = False,
    retrain_threshold: int = RETRAIN_BATCH_THRESHOLD,
) -> bool:
    """Execute the dual-cadence state update & 5-gameweek model retraining pipeline."""
    start_time = time.time()
    logger = logging.getLogger("pipeline")
    logger.info("=" * 75)
    logger.info("STARTING EPL MATCH PREDICTOR PIPELINE (DUAL-CADENCE)")
    logger.info(f"Output: {output_path} | Force Retrain: {force_retrain} | Retrain Threshold: {retrain_threshold} matches")
    logger.info("=" * 75)

    try:
        # Step 1: Ingest match and xG data
        logger.info("[1/5] Ingesting match results and expected goals data...")
        df_raw = load_and_merge_pipeline_data()
        current_match_count = len(df_raw)
        logger.info(f"Ingested {current_match_count} total fixtures.")

        # Step 2: Build sequential features and state
        logger.info("[2/5] Constructing chronological features (Elo, rolling xG, variance)...")
        df_feat, state, team_map = build_pipeline_features(df_raw)

        # Step 3: Compute expanding Poisson goal expectancies
        logger.info("[3/5] Computing expanding Poisson goal expectancies...")
        df_poisson = compute_leakage_free_poisson(df_feat)

        # Step 4: Check if full model retraining is required
        full_out_path = os.path.join(PROJECT_ROOT, output_path) if not os.path.isabs(output_path) else output_path
        existing_models, last_trained_count = load_existing_models_and_count(full_out_path)

        new_matches_since_train = current_match_count - last_trained_count
        should_retrain = (
            force_retrain
            or existing_models is None
            or existing_models.get("stacked_meta_model") is None
            or new_matches_since_train >= retrain_threshold
        )

        if should_retrain:
            logger.info(
                f"[4/5] FULL RETRAINING TRIGGERED: {new_matches_since_train} new fixtures accumulated "
                f"(Threshold: {retrain_threshold}) or force flag active."
            )
            models_dict = train_stacking_pipeline(df_feat)
            new_last_trained_count = current_match_count
        else:
            logger.info(
                f"[4/5] WEEKLY STATE UPDATE: {new_matches_since_train} new matches since last full retrain. "
                f"Updating state, Elo ratings & Poisson matrix (Full retrain in {retrain_threshold - new_matches_since_train} matches)."
            )
            models_dict = existing_models
            new_last_trained_count = last_trained_count

        # Step 5: Export assets bundle
        if not dry_run:
            logger.info(f"[5/5] Serializing updated asset bundle to {output_path}...")
            export_assets_bundle(
                models_dict=models_dict,
                state=state,
                team_map=team_map,
                df_poisson=df_poisson,
                output_path=full_out_path,
                last_trained_count=new_last_trained_count,
            )

            # Step 6: Smoke test inference
            logger.info("Running smoke test inference...")
            from predictor import predict_match
            test_res = predict_match("Arsenal", "Chelsea")
            logger.info(f"Smoke test result: Arsenal vs Chelsea -> {test_res['predicted_outcome']} ({test_res['confidence']*100:.1f}%)")
        else:
            logger.info("[5/5] DRY RUN active: Skipping asset export.")

        elapsed = time.time() - start_time
        logger.info("=" * 75)
        logger.info(f"PIPELINE COMPLETED SUCCESSFULLY IN {elapsed:.2f}s")
        logger.info("=" * 75)
        return True

    except Exception as e:
        logger.exception(f"Pipeline execution failed with error: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="EPL Match Predictor Dual-Cadence Pipeline")
    parser.add_argument("--output", type=str, default="epl_predictor_assets.pkl", help="Path to save epl_predictor_assets.pkl")
    parser.add_argument("--dry-run", action="store_true", help="Run without overwriting assets")
    parser.add_argument("--force-retrain", action="store_true", help="Force full model retraining regardless of match threshold")
    parser.add_argument("--retrain-threshold", type=int, default=RETRAIN_BATCH_THRESHOLD, help="Match count threshold for full retraining (default: 50)")
    parser.add_argument("--log-level", type=str, default="INFO", help="Logging level (DEBUG, INFO, WARNING, ERROR)")
    args = parser.parse_args()

    setup_logging(args.log_level)
    success = run(
        output_path=args.output,
        dry_run=args.dry_run,
        force_retrain=args.force_retrain,
        retrain_threshold=args.retrain_threshold,
    )
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
