"""
Model training and stacking module for the EPL Match Predictor pipeline.
Trains XGBoost, Random Forest, and the Level-1 Logistic Regression Stacking Meta-Learner.
"""
import logging
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier

from pipeline.build_features import FINAL_FEATURES

logger = logging.getLogger(__name__)


def multiclass_brier_score(y_true, y_prob):
    """Compute multiclass Brier score."""
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    n_samples, n_classes = y_prob.shape
    y_true_one_hot = np.zeros((n_samples, n_classes))
    for i, label in enumerate(y_true):
        y_true_one_hot[i, int(label)] = 1.0
    return float(np.mean(np.sum((y_prob - y_true_one_hot) ** 2, axis=1)))


def get_base_models():
    """Instantiate calibrated, regularized base models."""
    xgb_model = XGBClassifier(
        n_estimators=541,
        max_depth=3,
        learning_rate=0.0297,
        subsample=0.985,
        colsample_bytree=0.729,
        gamma=0.259,
        reg_alpha=4.544,
        reg_lambda=5.047,
        eval_metric="logloss",
        random_state=42,
    )

    rf_model = RandomForestClassifier(
        n_estimators=443,
        max_depth=3,
        min_samples_split=10,
        min_samples_leaf=7,
        class_weight={0: 1.4, 1: 1.0, 2: 1.0},
        random_state=42,
        n_jobs=-1,
    )

    return xgb_model, rf_model


def train_stacking_pipeline(df: pd.DataFrame) -> dict:
    """Train XGBoost, Random Forest, and Level-1 Stacking Logistic Regression Meta-Learner."""
    valid_mask = df["FTR_label"].notna()
    df_valid = df[valid_mask].copy()

    X = df_valid[FINAL_FEATURES]
    y = df_valid["FTR_label"].astype(int).values

    logger.info(f"Training pipeline with {len(X)} fixtures and {X.shape[1]} features.")

    # 1. Out-of-fold probability generation for the meta-learner
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof_xgb = np.zeros((len(X), 3))
    oof_rf = np.zeros((len(X), 3))

    for train_idx, val_idx in skf.split(X, y):
        X_tr, y_tr = X.iloc[train_idx], y[train_idx]
        X_va = X.iloc[val_idx]

        m_xgb, m_rf = get_base_models()
        m_xgb.fit(X_tr, y_tr)
        m_rf.fit(X_tr, y_tr)

        oof_xgb[val_idx] = m_xgb.predict_proba(X_va)
        oof_rf[val_idx] = m_rf.predict_proba(X_va)

    # 2. Fit Level-1 Stacking Meta-Learner on OOF probabilities
    X_meta = np.hstack([oof_xgb, oof_rf])
    meta_model = LogisticRegression(
        max_iter=1000,
        class_weight={0: 1.4, 1: 1.0, 2: 1.0},
        solver="lbfgs",
        random_state=42,
    )
    meta_model.fit(X_meta, y)

    # 3. Fit base models on the entire dataset
    best_xgb_model, best_rf_model = get_base_models()
    best_xgb_model.fit(X, y)
    best_rf_model.fit(X, y)

    # 4. In-sample / OOF evaluation metrics
    oof_meta_probs = meta_model.predict_proba(X_meta)
    brier = multiclass_brier_score(y, oof_meta_probs)
    ll = float(log_loss(y, oof_meta_probs))

    logger.info(f"Stacking Meta-Model OOF Log Loss: {ll:.4f} | Brier Score: {brier:.4f}")

    return {
        "best_xgb_model": best_xgb_model,
        "best_rf_model": best_rf_model,
        "stacked_meta_model": meta_model,
        "metrics": {"log_loss": ll, "brier_score": brier, "n_samples": len(X)},
    }
