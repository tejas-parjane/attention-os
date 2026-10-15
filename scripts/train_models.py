"""Train retention and monetization models, save models + metrics to data/models/."""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from src.models.retention import (
    RETENTION_FEATURES,
    build_retention_dataset,
    evaluate_retention_models,
    feature_importance,
    save_models as save_retention_models,
    train_retention_model,
)
from src.models.monetization import (
    MONETIZATION_FEATURES,
    build_monetization_dataset,
    evaluate_monetization_models,
    feature_importance as monetization_importance,
    save_models as save_monetization_models,
    train_monetization_model,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent


def _split(X: pd.DataFrame, y: pd.Series, ref: pd.Series, seed: int = 42) -> tuple:
    """Stratified 70/15/15 user split at a fixed prediction date.

    All rows share one prediction date ``T0``: every feature is computed
    strictly before ``T0`` and every label strictly after it, so temporal
    separation holds for every user by construction and no future
    information can leak across the split.  Within that guarantee, users
    are assigned to train/val/test with a stratified shuffle split
    (``random_state=42``) so each fold preserves the positive-class rate.
    Returns ref slices aligned to each fold.
    """
    from sklearn.model_selection import StratifiedShuffleSplit

    splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.30, random_state=seed)
    train_idx, temp_idx = next(splitter.split(X, y))
    splitter2 = StratifiedShuffleSplit(n_splits=1, test_size=0.50, random_state=seed)
    rel_val, rel_test = next(splitter2.split(X.iloc[temp_idx], y.iloc[temp_idx]))
    val_idx = temp_idx[rel_val]
    test_idx = temp_idx[rel_test]

    def take(frame: pd.DataFrame | pd.Series, idx):
        return frame.iloc[idx].reset_index(drop=True)

    return (
        take(X, train_idx), take(y, train_idx), take(ref, train_idx),
        take(X, val_idx), take(y, val_idx), take(ref, val_idx),
        take(X, test_idx), take(y, test_idx), take(ref, test_idx),
    )


def main() -> None:
    features_path = ROOT / "data" / "processed" / "user_features.parquet"
    events_path = ROOT / "data" / "raw" / "events.parquet"

    if not features_path.exists():
        logger.error("Missing %s — run `python scripts/build_features.py` first.", features_path)
        sys.exit(1)
    if not events_path.exists():
        logger.error("Missing %s — run `python scripts/generate_data.py` first.", events_path)
        sys.exit(1)

    features_df = pd.read_parquet(features_path)
    events_df = pd.read_parquet(events_path)
    logger.info("Loaded %d users, %d events", len(features_df), len(events_df))

    # Choose a reference date well inside the data range so that the 7-day
    # label window (reference -> reference + 7d) contains real future events.
    max_ts = pd.to_datetime(events_df["timestamp"]).max()
    reference_date = max_ts - pd.Timedelta(days=14)
    logger.info("Using reference_date=%s for label construction", reference_date)

    # ── Retention model ──────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("TRAINING RETENTION MODEL")
    logger.info("=" * 60)

    X_r, y_r, ref_r = build_retention_dataset(features_df, events_df, reference_date=reference_date)
    X_tr, y_tr, ref_tr, X_v, y_v, ref_v, X_te, y_te, ref_te = _split(X_r, y_r, ref_r)

    ret_models, ret_train_metrics = train_retention_model(X_tr, y_tr, ref_tr)
    ret_metrics, _ = evaluate_retention_models(ret_models, X_te, y_te)

    ret_models_dir = ROOT / "data" / "models" / "retention"
    save_retention_models(ret_models, path_prefix=str(ret_models_dir))
    (ret_models_dir / "metrics.csv").parent.mkdir(parents=True, exist_ok=True)
    ret_metrics.to_csv(ret_models_dir / "metrics.csv")

    best_ret = ret_metrics["roc_auc"].idxmax()
    ret_importance = feature_importance(ret_models[best_ret], RETENTION_FEATURES)

    # ── Monetization model ───────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("TRAINING MONETIZATION MODEL")
    logger.info("=" * 60)

    X_m, y_m, ref_m = build_monetization_dataset(features_df, events_df, reference_date=reference_date)
    X_trm, y_trm, ref_trm, X_vm, y_vm, ref_vm, X_tem, y_tem, ref_tem = _split(X_m, y_m, ref_m)

    mon_models, mon_train_metrics = train_monetization_model(X_trm, y_trm, ref_trm)
    mon_metrics, _ = evaluate_monetization_models(mon_models, X_tem, y_tem)

    mon_models_dir = ROOT / "data" / "models" / "monetization"
    save_monetization_models(mon_models, path_prefix=str(mon_models_dir))
    mon_metrics.to_csv(mon_models_dir / "metrics.csv")

    best_mon = mon_metrics["roc_auc"].idxmax()
    mon_importance = monetization_importance(mon_models[best_mon], MONETIZATION_FEATURES)

    # ── Comparison ───────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("RETENTION MODEL - Test-set Comparison")
    print("=" * 60)
    print(ret_metrics.to_string())

    print("\n" + "=" * 60)
    print("MONETIZATION MODEL - Test-set Comparison")
    print("=" * 60)
    print(mon_metrics.to_string())

    print(f"\n=== Best Retention Model: {best_ret}  (ROC-AUC={ret_metrics.loc[best_ret, 'roc_auc']:.4f}) ===")
    print(ret_importance.head(10).to_string(index=False))

    print(f"\n=== Best Monetization Model: {best_mon}  (ROC-AUC={mon_metrics.loc[best_mon, 'roc_auc']:.4f}) ===")
    print(mon_importance.head(10).to_string(index=False))

    # ── Side-by-side best-model comparison ───────────────────────────────
    comparison = pd.DataFrame({
        "model": [best_ret, best_mon],
        "task": ["retention", "monetization"],
        "roc_auc": [ret_metrics.loc[best_ret, "roc_auc"], mon_metrics.loc[best_mon, "roc_auc"]],
        "pr_auc": [ret_metrics.loc[best_ret, "pr_auc"], mon_metrics.loc[best_mon, "pr_auc"]],
        "f1": [ret_metrics.loc[best_ret, "f1"], mon_metrics.loc[best_mon, "f1"]],
    })
    print("\n=== Best Model Comparison ===")
    print(comparison.to_string(index=False))


if __name__ == "__main__":
    main()
