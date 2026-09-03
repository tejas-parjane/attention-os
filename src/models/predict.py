"""Functions to run retention and monetization predictions for a user."""

import os
import pickle
from pathlib import Path
from typing import Any


_MODEL_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "models"


def load_models() -> dict[str, Any]:
    """
    Load saved retention + monetization models from data/models/.

    Expected files:
        data/models/retention_model.pkl
        data/models/monetization_model.pkl

    Returns:
        dict with keys "retention" and "monetization", each mapping to the
        deserialized model object (pickle).

    Raises:
        FileNotFoundError: if either model file is missing.
    """
    models: dict[str, Any] = {}

    retention_path = _MODEL_DIR / "retention_model.pkl"
    monetization_path = _MODEL_DIR / "monetization_model.pkl"

    if not retention_path.exists():
        raise FileNotFoundError(f"Retention model not found at {retention_path}")
    if not monetization_path.exists():
        raise FileNotFoundError(f"Monetization model not found at {monetization_path}")

    with open(retention_path, "rb") as fh:
        models["retention"] = pickle.load(fh)
    with open(monetization_path, "rb") as fh:
        models["monetization"] = pickle.load(fh)

    return models


def _fill_missing_features(
    features: dict[str, float],
    feature_names: list[str],
    fill_value: float = 0.0,
) -> list[list[float]]:
    """
    Align a raw feature dict to the expected feature list, filling missing keys.

    Parameters:
        features      : raw feature dict from the feature pipeline.
        feature_names : ordered list of feature names the model expects.
        fill_value    : value to insert for missing features (default 0.0).

    Returns:
        A single-row 2-D list ready for model.predict().
    """
    return [[features.get(name, fill_value) for name in feature_names]]


def predict_user(
    user_id: str,
    features: dict[str, float],
    models: dict[str, Any],
) -> dict[str, float]:
    """
    Run both retention and monetization models on a user's feature dict.

    Handles missing features by filling 0 for any key not present in
    the features dict.  Model-specific feature names are read from
    ``model.feature_names_in_`` when available, otherwise fallback
    columns are used.

    Returns:
        dict with keys:
            - retention_probability  : P(user retained over next window)
            - purchase_probability   : P(user makes a purchase next period)
            - churn_probability      : 1 - retention_probability
            - predicted_ltv          : monetization model's LTV estimate
    """
    results: dict[str, float] = {
        "retention_probability": 0.5,
        "purchase_probability": 0.0,
        "churn_probability": 0.5,
        "predicted_ltv": 0.0,
    }

    # --- Retention prediction ---
    retention_model = models.get("retention")
    if retention_model is not None:
        try:
            feature_names = _get_feature_names(retention_model, features)
            X = _fill_missing_features(features, feature_names)
            prob = float(retention_model.predict_proba(X)[0][1])
            results["retention_probability"] = round(prob, 6)
            results["churn_probability"] = round(1.0 - prob, 6)
        except Exception:
            # Graceful fallback: keep defaults
            pass

    # --- Monetization / LTV prediction ---
    monetization_model = models.get("monetization")
    if monetization_model is not None:
        try:
            feature_names = _get_feature_names(monetization_model, features)
            X = _fill_missing_features(features, feature_names)

            # Some LTV models predict probability of purchase; others predict value
            if hasattr(monetization_model, "predict_proba") and _is_classifier(monetization_model):
                prob = float(monetization_model.predict_proba(X)[0][1])
                results["purchase_probability"] = round(prob, 6)
            else:
                ltv = float(monetization_model.predict(X)[0])
                results["predicted_ltv"] = round(max(ltv, 0.0), 2)
        except Exception:
            pass

    return results


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _get_feature_names(model: Any, features: dict[str, float]) -> list[str]:
    """
    Resolve the ordered feature list a model expects.

    Priority:
        1. ``model.feature_names_in_``  (scikit-learn convention)
        2. keys of the provided features dict (best-effort alignment)
    """
    if hasattr(model, "feature_names_in_"):
        return list(model.feature_names_in_)
    return sorted(features.keys())


def _is_classifier(model: Any) -> bool:
    """Heuristic: return True if the model looks like a classifier."""
    if hasattr(model, "predict_proba"):
        return True
    if hasattr(model, "_estimator_type"):
        return model._estimator_type == "classifier"
    return False
