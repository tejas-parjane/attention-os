"""Run retention + monetization predictions for all users and save results."""

from __future__ import annotations

import logging
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent


def _load_best_model(model_dir: Path) -> object:
    """Load the best model (by filename convention) from a model directory.

    Tries xgboost.joblib, random_forest.joblib, logistic.joblib in order,
    falling back to pickle (.pkl) files.
    """
    import joblib

    model_dir = Path(model_dir)
    for name in ("xgboost", "random_forest", "logistic"):
        joblib_path = model_dir / f"{name}.joblib"
        if joblib_path.exists():
            return joblib.load(joblib_path)
    for name in ("xgboost", "random_forest", "logistic"):
        pkl_path = model_dir / f"{name}.pkl"
        if pkl_path.exists():
            with open(pkl_path, "rb") as fh:
                return pickle.load(fh)
    return None


def _get_feature_names(model: object, fallback: list[str]) -> list[str]:
    """Resolve the feature names a model expects."""
    if hasattr(model, "feature_names_in_"):
        return list(model.feature_names_in_)
    return fallback


def main() -> None:
    features_path = ROOT / "data" / "processed" / "user_features.parquet"
    ret_model_dir = ROOT / "data" / "models" / "retention"
    mon_model_dir = ROOT / "data" / "models" / "monetization"
    out_path = ROOT / "data" / "processed" / "predictions.parquet"

    if not features_path.exists():
        logger.error("Missing %s — run `python scripts/build_features.py` first.", features_path)
        sys.exit(1)
    if not ret_model_dir.exists() or not mon_model_dir.exists():
        logger.error("Missing model directory. Run `python scripts/train_models.py` first.")
        sys.exit(1)

    features_df = pd.read_parquet(features_path)
    logger.info("Loaded features for %d users", len(features_df))

    ret_model = _load_best_model(ret_model_dir)
    mon_model = _load_best_model(mon_model_dir)

    if ret_model is None:
        logger.error("No retention model found in %s — run train_models.py first.", ret_model_dir)
        sys.exit(1)
    if mon_model is None:
        logger.error("No monetization model found in %s — run train_models.py first.", mon_model_dir)
        sys.exit(1)

    logger.info("Loaded retention model: %s", type(ret_model).__name__)
    logger.info("Loaded monetization model: %s", type(mon_model).__name__)

    # Import canonical feature lists
    from src.models.retention import RETENTION_FEATURES
    from src.models.monetization import MONETIZATION_FEATURES

    ret_feature_names = _get_feature_names(ret_model, RETENTION_FEATURES)
    mon_feature_names = _get_feature_names(mon_model, MONETIZATION_FEATURES)

    # Fill missing columns with 0
    for col in set(ret_feature_names + mon_feature_names):
        if col not in features_df.columns:
            features_df[col] = 0.0

    predictions: list[dict] = []
    for _, row in features_df.iterrows():
        uid = row["user_id"]
        feat_dict = row.to_dict()

        ret_row = [[feat_dict.get(c, 0.0) for c in ret_feature_names]]
        mon_row = [[feat_dict.get(c, 0.0) for c in mon_feature_names]]

        ret_prob = 0.5
        mon_prob = 0.0
        try:
            ret_prob = float(ret_model.predict_proba(ret_row)[0][1])
        except Exception as exc:
            logger.warning("Retention prediction failed for %s: %s", uid, exc)

        try:
            if hasattr(mon_model, "predict_proba"):
                mon_prob = float(mon_model.predict_proba(mon_row)[0][1])
            else:
                mon_prob = float(mon_model.predict(mon_row)[0])
        except Exception as exc:
            logger.warning("Monetization prediction failed for %s: %s", uid, exc)

        predictions.append({
            "user_id": uid,
            "retention_probability": round(ret_prob, 6),
            "churn_probability": round(1.0 - ret_prob, 6),
            "purchase_probability": round(mon_prob, 6),
            "predicted_ltv": round(mon_prob * float(feat_dict.get("total_ltv", 0.0)), 4),
        })

    pred_df = pd.DataFrame(predictions)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pred_df.to_parquet(out_path, index=False)
    logger.info("Saved predictions -> %s", out_path)

    print("\n=== Prediction Summary ===")
    print(pred_df.describe().to_string())
    print(f"\nRetention probability distribution:")
    print(pred_df["retention_probability"].describe().to_string())
    print(f"\nPurchase probability distribution:")
    print(pred_df["purchase_probability"].describe().to_string())


if __name__ == "__main__":
    main()
