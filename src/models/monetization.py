"""
Monetization / purchase-propensity model.

Predicts the probability that a user completes a purchase event within a
configurable window after a reference date.  The pipeline mirrors
``retention.py`` in structure but targets conversion rather than return.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from xgboost import XGBClassifier

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Feature schema
# ---------------------------------------------------------------------------

MONETIZATION_FEATURES: List[str] = [
    # engagement / session context
    "sessions_7d",
    "sessions_30d",
    "avg_session_duration",
    "session_duration_avg_7d",
    "session_duration_avg_30d",
    "session_duration_trend",
    "days_active_30d",
    "events_per_session_7d",
    "avg_screens_viewed_7d",
    "avg_actions_completed_7d",
    # monetization signals
    "purchase_count_total",
    "purchase_count_30d",
    "revenue_7d",
    "revenue_30d",
    "avg_purchase_value",
    "purchase_conversion_rate",
    "ad_revenue_7d",
    "ad_revenue_30d",
    # behavioural / lifecycle
    "content_diversity",
    "challenge_completion_rate",
    "notification_open_rate",
    "notification_click_rate",
    "days_since_signup",
    "new_user_flag",
    "returning_user_flag",
    "avg_level",
    "highest_level",
    "level_progression_rate",
    "engagement_change_7d",
]

# Event types that signal purchase intent / completion
PURCHASE_EVENT_TYPES: List[str] = [
    "purchase",
    "payment_success",
    "checkout_complete",
    "subscription_created",
]


# ---------------------------------------------------------------------------
# Dataset construction
# ---------------------------------------------------------------------------


def build_monetization_dataset(
    features_df: pd.DataFrame,
    events_df: pd.DataFrame,
    target_window_days: int = 7,
    reference_date: Optional[pd.Timestamp] = None,
    purchase_event_types: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Build a labelled dataset for purchase-propensity prediction.

    Parameters
    ----------
    features_df:
        One row per ``user_id`` with monetization features **plus**
        ``user_id``.
    events_df:
        Event log with at least ``user_id``, ``event_type``, ``timestamp``.
    target_window_days:
        Prediction window length in days.
    reference_date:
        Anchor point.  Defaults to the latest timestamp in *events_df*.
    purchase_event_types:
        Which event types count as a purchase.  Falls back to
        ``PURCHASE_EVENT_TYPES``.

    Returns
    -------
    X, y, reference_dates
        Same semantics as :func:`retention.build_retention_dataset`.
    """

    if purchase_event_types is None:
        purchase_event_types = PURCHASE_EVENT_TYPES

    if reference_date is None:
        reference_date = pd.Timestamp(events_df["timestamp"].max())
    else:
        reference_date = pd.Timestamp(reference_date)

    window_end = reference_date + pd.Timedelta(days=target_window_days)

    logger.info(
        "Building monetization dataset  reference_date=%s  window=%d days",
        reference_date,
        target_window_days,
    )

    # Label: any purchase event inside (reference_date, window_end]
    future_purchases = events_df.loc[
        (events_df["timestamp"] > reference_date)
        & (events_df["timestamp"] <= window_end)
        & (events_df["event_type"].isin(purchase_event_types))
    ]
    converted_users = set(future_purchases["user_id"].unique())

    df = features_df.copy()
    df["label"] = df["user_id"].isin(converted_users).astype(int)

    # Feature matrix
    available = [f for f in MONETIZATION_FEATURES if f in df.columns]
    missing = set(MONETIZATION_FEATURES) - set(available)
    if missing:
        logger.warning("Missing features filled with 0: %s", sorted(missing))
        for col in missing:
            df[col] = 0.0

    X = df[MONETIZATION_FEATURES].copy()
    y = df["label"].copy()
    reference_dates = pd.Series(
        reference_date, index=df.index, name="reference_date"
    )

    pos_rate = y.mean()
    logger.info(
        "Dataset ready  n=%d  positive_rate=%.4f  purchases_found=%d",
        len(y),
        pos_rate,
        len(converted_users),
    )
    return X, y, reference_dates


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def train_monetization_model(
    X: pd.DataFrame,
    y: pd.Series,
    reference_dates: Optional[pd.Series] = None,
) -> Tuple[Dict[str, Any], Dict[str, Dict[str, float]]]:
    """Train three classifiers with a time-aware 70/15/15 split.

    Parameters
    ----------
    X, y:
        Feature matrix and binary labels (1 = converted).
    reference_dates:
        Chronological sort key.  If ``None`` a random stratified split
        is used.

    Returns
    -------
    models : dict
        ``{"logistic": …, "random_forest": …, "xgboost": …}``
    metrics : dict
        Per-model train-set summary metrics.
    """

    if reference_dates is not None:
        order = reference_dates.values.argsort()
        X = X.iloc[order].reset_index(drop=True)
        y = y.iloc[order].reset_index(drop=True)

    n = len(X)
    train_end = int(n * 0.70)
    val_end = int(n * 0.85)

    X_train, y_train = X.iloc[:train_end], y.iloc[:train_end]
    X_val, y_val = X.iloc[train_end:val_end], y.iloc[train_end:val_end]

    logger.info(
        "Split sizes  train=%d  val=%d  test=%d",
        len(X_train),
        len(X_val),
        len(X) - val_end,
    )

    # Compute class weight for imbalanced purchase labels
    n_pos = int(y_train.sum())
    n_neg = len(y_train) - n_pos
    pos_weight = n_neg / max(n_pos, 1)

    candidates: Dict[str, Any] = {
        "logistic": LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            n_jobs=-1,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=300,
            max_depth=12,
            class_weight="balanced",
            n_jobs=-1,
            random_state=42,
        ),
        "xgboost": XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.1,
            scale_pos_weight=pos_weight,
            eval_metric="logloss",
            n_jobs=-1,
            random_state=42,
            use_label_encoder=False,
        ),
    }

    trained: Dict[str, Any] = {}
    metrics: Dict[str, Dict[str, float]] = {}

    for name, model in candidates.items():
        logger.info("Training %s …", name)
        model.fit(X_train, y_train)

        train_proba = (
            model.predict_proba(X_train)[:, 1]
            if hasattr(model, "predict_proba")
            else model.predict(X_train).astype(float)
        )
        train_pred = model.predict(X_train)

        trained[name] = model
        metrics[name] = {
            "roc_auc": float(roc_auc_score(y_train, train_proba)),
            "pr_auc": float(average_precision_score(y_train, train_proba)),
            "f1": float(f1_score(y_train, train_pred)),
        }
        logger.info(
            "  %s  ROC-AUC=%.4f  PR-AUC=%.4f",
            name,
            metrics[name]["roc_auc"],
            metrics[name]["pr_auc"],
        )

    return trained, metrics


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def evaluate_monetization_models(
    models: Dict[str, Any],
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> Tuple[pd.DataFrame, Dict[str, Dict[str, Any]]]:
    """Evaluate every model on a held-out test set.

    Returns
    -------
    metrics_df : pd.DataFrame
        Comparison table with ROC-AUC, PR-AUC, precision, recall, F1,
        Brier score, and confusion-matrix components per model.
    details : dict
        ``{model_name: {"confusion_matrix": …, "probabilities": …}}``
        for downstream analysis.
    """

    rows: List[Dict[str, Any]] = []
    details: Dict[str, Dict[str, Any]] = {}

    for name, model in models.items():
        y_pred = model.predict(X_test)
        y_proba = (
            model.predict_proba(X_test)[:, 1]
            if hasattr(model, "predict_proba")
            else y_pred.astype(float)
        )

        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

        row: Dict[str, Any] = {
            "model": name,
            "roc_auc": round(roc_auc_score(y_test, y_proba), 4),
            "pr_auc": round(average_precision_score(y_test, y_proba), 4),
            "precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
            "recall": round(recall_score(y_test, y_pred, zero_division=0), 4),
            "f1": round(f1_score(y_test, y_pred, zero_division=0), 4),
            "brier": round(brier_score_loss(y_test, y_proba), 4),
            "true_pos": int(tp),
            "false_pos": int(fp),
            "true_neg": int(tn),
            "false_neg": int(fn),
        }
        rows.append(row)
        details[name] = {
            "confusion_matrix": confusion_matrix(y_test, y_pred),
            "probabilities": y_proba,
        }

    metrics_df = pd.DataFrame(rows).set_index("model")
    return metrics_df, details


# ---------------------------------------------------------------------------
# Feature importance
# ---------------------------------------------------------------------------


def feature_importance(
    model: Any,
    features: List[str],
) -> pd.DataFrame:
    """Model-agnostic feature importance extraction.

    * Logistic regression  → absolute coefficient values.
    * Tree-based models    → ``feature_importances_`` attribute.
    * Fallback             → uniform importance with a warning.

    Returns a DataFrame sorted descending by importance.
    """

    imp: Optional[np.ndarray] = None

    if hasattr(model, "coef_"):
        coefs = np.array(model.coef_)
        imp = np.abs(coefs).flatten()
    elif hasattr(model, "feature_importances_"):
        imp = np.array(model.feature_importances_)
    else:
        logger.warning(
            "Model %s exposes no importance attribute; returning uniform.",
            type(model).__name__,
        )
        imp = np.ones(len(features)) / len(features)

    df = (
        pd.DataFrame({"feature": features, "importance": imp})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )
    return df


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def save_models(
    models: Dict[str, Any],
    path_prefix: str = "data/models/monetization",
) -> Dict[str, str]:
    """Persist every model to ``<path_prefix>/<name>.joblib``.

    Returns ``{name: file_path}``.
    """

    out_dir = Path(path_prefix)
    out_dir.mkdir(parents=True, exist_ok=True)

    saved: Dict[str, str] = {}
    for name, model in models.items():
        fp = out_dir / f"{name}.joblib"
        joblib.dump(model, fp)
        saved[name] = str(fp)
        logger.info("Saved %s → %s", name, fp)

    return saved


def load_models(
    path_prefix: str = "data/models/monetization",
    names: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Load models previously saved with :func:`save_models`."""

    if names is None:
        names = ["logistic", "random_forest", "xgboost"]

    base = Path(path_prefix)
    loaded: Dict[str, Any] = {}
    for name in names:
        fp = base / f"{name}.joblib"
        if fp.exists():
            loaded[name] = joblib.load(fp)
            logger.info("Loaded %s ← %s", name, fp)
        else:
            logger.warning("Model file not found: %s", fp)
    return loaded


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    )

    import sys

    project_root = Path(__file__).resolve().parents[2]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    # ---- load data ----
    features_path = project_root / "data" / "processed" / "user_features.parquet"
    events_path = project_root / "data" / "raw" / "events.parquet"

    if not features_path.exists():
        raise FileNotFoundError(f"Missing features file: {features_path}")
    if not events_path.exists():
        raise FileNotFoundError(f"Missing events file: {events_path}")

    logger.info("Loading features from %s", features_path)
    features_df = pd.read_parquet(features_path)

    logger.info("Loading events from %s", events_path)
    events_df = pd.read_parquet(events_path)

    # ---- build dataset ----
    X, y, ref_dates = build_monetization_dataset(features_df, events_df)

    # ---- time-aware split ----
    order = ref_dates.values.argsort()
    X = X.iloc[order].reset_index(drop=True)
    y = y.iloc[order].reset_index(drop=True)
    ref_dates = ref_dates.iloc[order].reset_index(drop=True)

    n = len(X)
    train_end = int(n * 0.70)
    val_end = int(n * 0.85)

    X_train, y_train = X.iloc[:train_end], y.iloc[:train_end]
    X_test, y_test = X.iloc[val_end:], y.iloc[val_end:]

    logger.info(
        "Final split  train=%d  test=%d",
        len(X_train),
        len(X_test),
    )

    # ---- train ----
    models, train_metrics = train_monetization_model(X_train, y_train)

    # ---- evaluate ----
    metrics_df, details = evaluate_monetization_models(models, X_test, y_test)

    print("\n=== Monetization Model – Test-set Comparison ===\n")
    print(metrics_df.to_string())

    # ---- feature importance (best model) ----
    best_model_name = metrics_df["roc_auc"].idxmax()
    best_model = models[best_model_name]
    importance_df = feature_importance(best_model, MONETIZATION_FEATURES)
    print(f"\n=== Feature Importance ({best_model_name}) ===\n")
    print(importance_df.head(15).to_string(index=False))

    # ---- save ----
    models_dir = project_root / "data" / "models" / "monetization"
    saved = save_models(models, path_prefix=str(models_dir))
    print(f"\nModels saved to {models_dir}")
