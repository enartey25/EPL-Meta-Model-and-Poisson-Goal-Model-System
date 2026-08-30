"""
Asset serialization and verification module for the EPL Match Predictor pipeline.
Packages trained models, state objects, and Poisson tables atomically into epl_predictor_assets.pkl.
"""
import logging
import os
import pickle
import shutil
import tempfile
import pandas as pd

from pipeline.build_features import FINAL_FEATURES

logger = logging.getLogger(__name__)

LABEL_MAP = {0: "Draw", 1: "AwayWin", 2: "HomeWin"}


def export_assets_bundle(
    models_dict: dict,
    state: dict,
    team_map: dict,
    df_poisson: pd.DataFrame,
    output_path: str = "epl_predictor_assets.pkl",
    last_trained_count: int = None,
) -> str:
    """Pack models and state into a single pickle artifact atomically."""
    bundle = {
        "stacked_meta_model": models_dict["stacked_meta_model"],
        "best_xgb_model": models_dict["best_xgb_model"],
        "best_rf_model": models_dict["best_rf_model"],
        "team_map": team_map,
        "label_map": LABEL_MAP,
        "state": state,
        "final_features": FINAL_FEATURES,
        "df_poisson": df_poisson,
        "last_trained_count": last_trained_count or len(df_poisson),
    }

    dir_name = os.path.dirname(os.path.abspath(output_path))
    with tempfile.NamedTemporaryFile("wb", dir=dir_name, delete=False) as tf:
        pickle.dump(bundle, tf, protocol=pickle.HIGHEST_PROTOCOL)
        temp_path = tf.name

    if os.path.exists(output_path):
        backup_path = output_path + ".bak"
        shutil.copy2(output_path, backup_path)
        logger.info(f"Created backup of previous assets at {backup_path}")

    shutil.move(temp_path, output_path)
    logger.info(f"Assets bundle successfully exported to {output_path} ({os.path.getsize(output_path):,} bytes).")
    return output_path
