"""
Master orchestrator for the EPL Match Predictor weekly pipeline.
Runs data fetch, feature engineering, model retraining, and asset export.
"""
import argparse
import logging
import os
import sys
import time

# Ensure project root is in path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from pipeline.fetch_data import load_and_merge_pipeline_data
from pipeline.build_features import build_pipeline_features
from pipeline.update_poisson import compute_leakage_free_poisson
from pipeline.train_models import train_stacking_pipeline
from pipeline.export_assets import export_assets_bundle


def setup_logging(log_level: str = "INFO") -> None:
    """Configure structured logging."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def run(output_path: str = "epl_predictor_assets.pkl", dry_run: bool = False) -> bool:
    """Execute the full end-to-end retraining pipeline."""
    start_time = time.time()
    logger = logging.getLogger("pipeline")
    logger.info("=" * 70)
    logger.info("STARTING EPL MATCH PREDICTOR WEEKLY RETRAINING PIPELINE")
    logger.info(f"Target Output: {output_path} | Dry Run: {dry_run}")
    logger.info("=" * 70)

    try:
        # Step 1: Ingest match and xG data
        logger.info("[1/5] Ingesting match results and expected goals data...")
        df_raw = load_and_merge_pipeline_data()
        logger.info(f"Ingested {len(df_raw)} total fixtures.")

        # Step 2: Build sequential features and state
        logger.info("[2/5] Constructing chronological features (Elo, rolling xG, variance)...")
        df_feat, state, team_map = build_pipeline_features(df_raw)

        # Step 3: Compute expanding Poisson goal expectancies
        logger.info("[3/5] Computing expanding Poisson goal expectancies...")
        df_poisson = compute_leakage_free_poisson(df_feat)

        # Step 4: Train models and meta-learner
        logger.info("[4/5] Training XGBoost, Random Forest, and Stacking Meta-Learner...")
        models_dict = train_stacking_pipeline(df_feat)
        metrics = models_dict.get("metrics", {})
        logger.info(f"Validation Metrics: Log Loss = {metrics.get('log_loss'):.4f}, Brier = {metrics.get('brier_score'):.4f}")

        # Sanity check validation threshold
        if metrics.get("log_loss", 999.0) > 1.15:
            logger.warning(f"Log loss {metrics.get('log_loss')} exceeds expected threshold (1.15). Inspecting data.")

        # Step 5: Export assets bundle
        if not dry_run:
            logger.info(f"[5/5] Serializing and exporting asset bundle to {output_path}...")
            full_out_path = os.path.join(PROJECT_ROOT, output_path) if not os.path.isabs(output_path) else output_path
            export_assets_bundle(
                models_dict=models_dict,
                state=state,
                team_map=team_map,
                df_poisson=df_poisson,
                output_path=full_out_path,
            )

            # Step 6: Smoke test inference with predictor.py
            logger.info("Running smoke test inference...")
            from predictor import predict_match
            test_res = predict_match("Arsenal", "Chelsea")
            logger.info(f"Smoke test result: Arsenal vs Chelsea -> {test_res['predicted_outcome']} ({test_res['confidence']*100:.1f}%)")
        else:
            logger.info("[5/5] DRY RUN active: Skipping asset serialization and smoke test.")

        elapsed = time.time() - start_time
        logger.info("=" * 70)
        logger.info(f"PIPELINE COMPLETED SUCCESSFULLY IN {elapsed:.2f}s")
        logger.info("=" * 70)
        return True

    except Exception as e:
        logger.exception(f"Pipeline execution failed with fatal error: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="EPL Weekly Model Retraining Pipeline")
    parser.add_argument("--output", type=str, default="epl_predictor_assets.pkl", help="Path to save epl_predictor_assets.pkl")
    parser.add_argument("--dry-run", action="store_true", help="Run without overwriting assets")
    parser.add_argument("--log-level", type=str, default="INFO", help="Logging level (DEBUG, INFO, WARNING, ERROR)")
    args = parser.parse_args()

    setup_logging(args.log_level)
    success = run(output_path=args.output, dry_run=args.dry_run)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
