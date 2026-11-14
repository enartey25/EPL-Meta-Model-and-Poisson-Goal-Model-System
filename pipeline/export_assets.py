"""Atomic writes for the model bundle and the data files the app reads."""
import json
import logging
import os
import pickle
import tempfile

import pandas as pd

from pipeline import config

logger = logging.getLogger(__name__)


def _atomic_write(path: str, write_fn, mode: str = "w") -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=os.path.dirname(path) or ".", suffix=".tmp")
    try:
        with os.fdopen(fd, mode, **({} if "b" in mode else {"encoding": "utf-8", "newline": ""})) as f:
            write_fn(f)
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise


def write_assets(assets: dict, path: str = config.ASSETS_PATH) -> None:
    _atomic_write(path, lambda f: pickle.dump(assets, f, protocol=pickle.HIGHEST_PROTOCOL), "wb")
    logger.info(f"Wrote {path} ({os.path.getsize(path):,} bytes)")


def write_csv(df: pd.DataFrame, name: str) -> None:
    path = os.path.join(config.DATA_DIR, name)
    _atomic_write(path, lambda f: df.to_csv(f, index=False))
    logger.info(f"Wrote data/{name} ({len(df)} rows)")


def write_json(obj, name: str) -> None:
    path = os.path.join(config.DATA_DIR, name)
    _atomic_write(path, lambda f: json.dump(obj, f, indent=2, default=str))
    logger.info(f"Wrote data/{name}")


def load_existing_assets(path: str = config.ASSETS_PATH):
    if not os.path.exists(path):
        return None
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except Exception as e:  # unreadable / old format: treat as absent
        logger.warning(f"Could not read existing assets ({e}); a full retrain will run")
        return None
