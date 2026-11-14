"""Model training, evaluation and the walk-forward track record.

The stack: a Random Forest on the engineered features and the opponent-adjusted
Poisson model's outcome probabilities, combined by a multinomial logistic
regression meta-learner. The meta-learner is always trained on out-of-sample
inputs: for each season, a Random Forest trained only on earlier seasons.
"""
import logging

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss

from inference_utils import meta_features
from pipeline import config
from pipeline.update_poisson import poisson_probs

logger = logging.getLogger(__name__)
LABELS = [0, 1, 2]  # Draw, AwayWin, HomeWin


def scores(y, p) -> dict:
    y = np.asarray(y)
    onehot = np.column_stack([(y == c).astype(int) for c in LABELS])
    return {"log_loss": float(log_loss(y, p, labels=LABELS)),
            "brier": float(np.mean(np.sum((np.asarray(p) - onehot) ** 2, axis=1))),
            "n": int(len(y))}


def make_rf():
    return RandomForestClassifier(**config.RF_PARAMS)


def make_meta():
    return LogisticRegression(max_iter=1000)


def class_prior(y) -> np.ndarray:
    return np.bincount(np.asarray(y, dtype=int), minlength=3) / len(y)


def _parts(rf, rows: pd.DataFrame, dfp: pd.DataFrame) -> dict:
    return {"rf": rf.predict_proba(rows[config.FINAL_FEATURES]),
            "poisson": poisson_probs(dfp.loc[rows.index, "Lambda_Home"], dfp.loc[rows.index, "Lambda_Away"])}


def _meta_X(parts: dict) -> np.ndarray:
    return meta_features([parts[k] for k in config.META_INPUTS], config.META_TRANSFORM)


def fit_stack(df: pd.DataFrame, dfp: pd.DataFrame, seasons: list[str]):
    """Fit the stack using only ``seasons`` (in chronological order).

    The meta-learner sees walk-forward out-of-sample Random Forest probabilities
    for every season that has at least MIN_PRIOR_SEASONS_FOR_META earlier ones."""
    X_parts, y_parts = [], []
    for i, s in enumerate(seasons):
        if i < config.MIN_PRIOR_SEASONS_FOR_META:
            continue
        past = df[df["season"].isin(seasons[:i])]
        cur = df[df["season"] == s]
        rf_wf = make_rf().fit(past[config.FINAL_FEATURES], past["FTR"])
        X_parts.append(_meta_X(_parts(rf_wf, cur, dfp)))
        y_parts.append(cur["FTR"].to_numpy())
    if not X_parts:
        raise ValueError(f"Need more than {config.MIN_PRIOR_SEASONS_FOR_META} seasons to train the meta-learner")
    meta = make_meta().fit(np.vstack(X_parts), np.concatenate(y_parts))
    used = df[df["season"].isin(seasons)]
    rf = make_rf().fit(used[config.FINAL_FEATURES], used["FTR"])
    return rf, meta, int(sum(len(y) for y in y_parts))


def predict_rows(rf, meta, rows: pd.DataFrame, dfp: pd.DataFrame) -> np.ndarray:
    return meta.predict_proba(_meta_X(_parts(rf, rows, dfp)))


def holdout_evaluation(df: pd.DataFrame, dfp: pd.DataFrame, complete_seasons: list[str]) -> dict:
    """The notebook protocol on the latest complete season: Random Forest trained on
    the seasons before validation, meta-learner trained on the validation season,
    test season scored once."""
    if len(complete_seasons) < 3:
        return {}
    test_s, val_s, train_s = complete_seasons[-1], complete_seasons[-2], complete_seasons[:-2]
    tr, va, te = (df[df["season"].isin(train_s)], df[df["season"] == val_s], df[df["season"] == test_s])
    rf = make_rf().fit(tr[config.FINAL_FEATURES], tr["FTR"])
    meta = make_meta().fit(_meta_X(_parts(rf, va, dfp)), va["FTR"])
    parts_te = _parts(rf, te, dfp)
    stack = meta.predict_proba(_meta_X(parts_te))
    prior = class_prior(tr["FTR"])
    out = {
        "train_seasons": train_s, "validation_season": val_s, "test_season": test_s,
        "null_baseline": scores(te["FTR"], np.tile(prior, (len(te), 1))),
        "random_forest": scores(te["FTR"], parts_te["rf"]),
        "poisson": scores(te["FTR"], parts_te["poisson"]),
        "stack": scores(te["FTR"], stack),
    }
    logger.info(f"Holdout {test_s}: stack log loss {out['stack']['log_loss']:.4f} "
                f"vs null {out['null_baseline']['log_loss']:.4f}")
    return out


def season_record(df: pd.DataFrame, dfp: pd.DataFrame, season: str, seasons: list[str]) -> pd.DataFrame:
    """Forecasts for every played match of ``season`` from a stack frozen at that
    season's start (trained only on earlier seasons). Features and Poisson rates
    are pre-match, so each forecast uses only information available before kickoff."""
    prior_seasons = seasons[:seasons.index(season)]
    rf, meta, _ = fit_stack(df, dfp, prior_seasons)
    rows = df[df["season"] == season]
    p = predict_rows(rf, meta, rows, dfp)
    prior = class_prior(df[df["season"].isin(prior_seasons)]["FTR"])
    rec = pd.DataFrame({
        "season": season, "round": rows["round"].values if "round" in rows else np.nan,
        "fotmob_id": rows["fotmob_id"].values if "fotmob_id" in rows else np.nan,
        "Date": rows["Date"].values,
        "HomeTeam": rows["HomeTeam_name"].values, "AwayTeam": rows["AwayTeam_name"].values,
        "FTHG": rows["FTHG"].values, "FTAG": rows["FTAG"].values, "FTR": rows["FTR"].values,
        "P_Draw": p[:, 0], "P_Away": p[:, 1], "P_Home": p[:, 2],
        "Null_Draw": prior[0], "Null_Away": prior[1], "Null_Home": prior[2],
        "source": "walk-forward",
    })
    return rec
