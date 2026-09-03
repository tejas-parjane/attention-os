"""Lightweight model monitoring: predictions, drift, and summary reporting."""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


class PredictionRecord(BaseModel):
    """Single prediction logged for monitoring purposes."""

    user_id: str
    model_name: str
    prediction_type: str
    prediction_value: float
    feature_values: dict[str, float]
    predicted_at: datetime
    recommended_action: str = ""
    expected_value: float = 0.0


class DriftMetrics(BaseModel):
    """Per-feature drift summary compared against a reference window."""

    feature: str
    reference_mean: float
    current_mean: float
    drift_score: float
    status: str  # "normal" | "warning" | "drift"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _calculate_psi(
    expected: np.ndarray,
    actual: np.ndarray,
    buckets: int = 10,
) -> float:
    """Compute the Population Stability Index (PSI) between two arrays.

    PSI bins the *expected* distribution into ``buckets`` quantile-based
    intervals, then measures how much the *actual* distribution deviates.

    Interpretation
    --------------
    * PSI < 0.10  – no significant drift
    * 0.10 <= PSI < 0.25 – moderate drift (warning)
    * PSI >= 0.25 – significant drift

    Parameters
    ----------
    expected:
        Reference (training / baseline) values.
    actual:
        Current production values.
    buckets:
        Number of quantile bins (default 10 = deciles).

    Returns
    -------
    float
        The computed PSI value.
    """
    expected = np.asarray(expected, dtype=np.float64).flatten()
    actual = np.asarray(actual, dtype=np.float64).flatten()

    # Build breakpoints from the expected distribution (quantile-based).
    quantiles = np.linspace(0, 100, buckets + 1)
    breakpoints = np.percentile(expected, quantiles)
    # Ensure breakpoints are strictly increasing for digitize.
    breakpoints = np.unique(breakpoints)

    # Clip edges so every value falls into a bin.
    breakpoints[0] = -np.inf
    breakpoints[-1] = np.inf

    expected_counts = np.histogram(expected, bins=breakpoints)[0].astype(np.float64)
    actual_counts = np.histogram(actual, bins=breakpoints)[0].astype(np.float64)

    # Convert to proportions, avoiding division-by-zero.
    expected_pct = expected_counts / expected_counts.sum() if expected_counts.sum() else np.zeros_like(expected_counts)
    actual_pct = actual_counts / actual_counts.sum() if actual_counts.sum() else np.zeros_like(actual_counts)

    # Shift tiny proportions away from zero to prevent log(0).
    epsilon = 1e-6
    expected_pct = np.clip(expected_pct, epsilon, None)
    actual_pct = np.clip(actual_pct, epsilon, None)

    psi = float(np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct)))
    return psi


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def log_prediction(
    record: PredictionRecord,
    db_session: Any | None = None,
) -> None:
    """Persist a ``PredictionRecord``.

    Parameters
    ----------
    record:
        The prediction event to store.
    db_session:
        Optional database session / connection.  When *None* the record is
        only emitted to the logger – useful during development or tests.
    """
    if db_session is not None:
        db_session.add(record)
        db_session.commit()
        logger.info("Persisted prediction record for user %s", record.user_id)
    else:
        logger.info(
            "Prediction logged (no DB): user=%s model=%s value=%.4f action=%s",
            record.user_id,
            record.model_name,
            record.prediction_value,
            record.recommended_action,
        )


def compute_prediction_distribution(records: list[float]) -> dict[str, float]:
    """Return distribution statistics for a list of prediction values.

    Returns a dict with keys: ``mean``, ``std``, ``min``, ``max``,
    and percentile values ``p5``, ``p25``, ``p50``, ``p75``, ``p95``.
    """
    arr = np.asarray(records, dtype=np.float64)
    if arr.size == 0:
        return {k: 0.0 for k in ("mean", "std", "min", "max", "p5", "p25", "p50", "p75", "p95")}

    percentiles = np.percentile(arr, [5, 25, 50, 75, 95])
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "p5": float(percentiles[0]),
        "p25": float(percentiles[1]),
        "p50": float(percentiles[2]),
        "p75": float(percentiles[3]),
        "p95": float(percentiles[4]),
    }


def compute_feature_drift(
    reference_features: pd.DataFrame,
    current_features: pd.DataFrame,
    threshold: float = 0.1,
) -> list[DriftMetrics]:
    """Compare reference vs current feature distributions using PSI.

    Parameters
    ----------
    reference_features:
        Baseline dataframe (training / healthy window).
    current_features:
        Recent production dataframe.
    threshold:
        PSI threshold at which a feature transitions from *normal* to
        *warning*.  A second threshold of ``threshold * 2.5`` is used to
        flag *drift*.

    Returns
    -------
    list[DriftMetrics]
        One entry per overlapping column present in both dataframes.
    """
    common_cols = sorted(set(reference_features.columns) & set(current_features.columns))
    drift_warning_threshold = threshold
    drift_critical_threshold = threshold * 2.5

    results: list[DriftMetrics] = []
    for col in common_cols:
        ref_data = reference_features[col].dropna().values.astype(np.float64)
        cur_data = current_features[col].dropna().values.astype(np.float64)

        if ref_data.size == 0 or cur_data.size == 0:
            continue

        psi = _calculate_psi(ref_data, cur_data)
        ref_mean = float(np.mean(ref_data))
        cur_mean = float(np.mean(cur_data))

        if psi >= drift_critical_threshold:
            status = "drift"
        elif psi >= drift_warning_threshold:
            status = "warning"
        else:
            status = "normal"

        results.append(
            DriftMetrics(
                feature=col,
                reference_mean=round(ref_mean, 6),
                current_mean=round(cur_mean, 6),
                drift_score=round(psi, 6),
                status=status,
            )
        )

    return results


def check_missing_feature_rate(features_df: pd.DataFrame) -> dict[str, float]:
    """Return the missing-value rate per feature (0.0 – 1.0)."""
    total = len(features_df)
    if total == 0:
        return {col: 0.0 for col in features_df.columns}
    return {col: float(features_df[col].isna().sum() / total) for col in features_df.columns}


def compute_recommendation_distribution(actions: list[str]) -> dict[str, int]:
    """Count occurrence of each recommended action."""
    distribution: dict[str, int] = {}
    for action in actions:
        distribution[action] = distribution.get(action, 0) + 1
    return distribution


def summary_report(
    records: list[PredictionRecord],
    reference_features: pd.DataFrame,
    current_features: pd.DataFrame,
) -> dict[str, Any]:
    """Return a full monitoring summary dictionary.

    Sections
    --------
    * **prediction_distribution** – statistics over logged prediction values.
    * **feature_drift** – per-feature PSI drift metrics.
    * **missing_rates** – fraction of missing values in the current window.
    * **action_distribution** – counts of recommended actions.
    * **record_count** – total number of records examined.
    """
    pred_values = [r.prediction_value for r in records]
    action_list = [r.recommended_action for r in records if r.recommended_action]

    return {
        "record_count": len(records),
        "prediction_distribution": compute_prediction_distribution(pred_values),
        "feature_drift": [m.model_dump() for m in compute_feature_drift(reference_features, current_features)],
        "missing_rates": check_missing_feature_rate(current_features),
        "action_distribution": compute_recommendation_distribution(action_list),
    }
