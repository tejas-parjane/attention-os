"""Service layer functions for AttentionOS API.

Reads feature/prediction/recommendation/experiment data primarily from
parquet files stored in ``data/processed/`` for the demo, with an optional
PostgreSQL/SQLite fallback. This keeps the API fully functional before any
database seeding.

The decision pipeline glue (features -> predictions -> state -> decision ->
explanation) is assembled here, delegating to the decision engine and the
LLM explanation provider.
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data source resolution
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[2]
_PROCESSED_DIR = ROOT / "data" / "processed"
_RAW_DIR = ROOT / "data" / "raw"


def _load_parquet(name: str) -> pd.DataFrame | None:
    """Load a parquet file from ``data/processed/`` by bare stem name."""
    path = _PROCESSED_DIR / f"{name}.parquet"
    if path.exists():
        return pd.read_parquet(path)
    return None


def _load_users() -> pd.DataFrame | None:
    """Load user metadata (from the generator's CSV, falling back to DB)."""
    csv_path = _RAW_DIR / "users.csv"
    if csv_path.exists():
        return pd.read_csv(csv_path)
    return _load_parquet("users")


def _user_row_from_parquet(name: str, user_id: str) -> dict[str, Any] | None:
    """Return the most-recent row for *user_id* from a parquet file, or None."""
    df = _load_parquet(name)
    if df is None or df.empty or "user_id" not in df.columns:
        return None
    user_df = df[df["user_id"] == user_id]
    if user_df.empty:
        return None
    return user_df.iloc[-1].to_dict()


# ---------------------------------------------------------------------------
# Optional DB helpers (graceful fallback)
# ---------------------------------------------------------------------------

try:
    from sqlalchemy.exc import OperationalError, ProgrammingError  # noqa: F401
except ImportError:  # pragma: no cover
    OperationalError = ProgrammingError = type("DummyException", (), {})


def _db_session_safe():
    """Yield a DB session if reachable, else None."""
    try:
        from src.data.database import get_database_url, get_engine, get_session_ctx
        engine = get_engine(get_database_url())
        with engine.connect() as conn:
            conn.execute(__import__("sqlalchemy").text("SELECT 1"))
        for session in get_session_ctx(engine):
            yield session
    except Exception as exc:  # noqa: BLE001
        logger.debug("DB unavailable, falling back to parquet: %s", exc)
        yield None


# ---------------------------------------------------------------------------
# Feature / prediction / recommendation services
# ---------------------------------------------------------------------------

def get_features_for_user(user_id: str) -> dict[str, Any]:
    """Get the latest feature vector for a user.

    Tries ``data/processed/user_features.parquet`` first, then the DB.
    """
    row = _user_row_from_parquet("user_features", user_id)
    if row is not None:
        features = {k: float(v) if isinstance(v, (int, float)) else v for k, v in row.items() if k != "user_id"}
        return {"user_id": user_id, "features": features, "feature_timestamp": None}

    for session in _db_session_safe():
        if session is None:
            break
        try:
            from sqlalchemy import text
            result = session.execute(
                text(
                    "SELECT * FROM user_features WHERE user_id = :uid "
                    "ORDER BY computed_at DESC LIMIT 1"
                ),
                {"uid": user_id},
            )
            row_dict = result.mappings().first()
            if row_dict:
                d = dict(row_dict)
                features = {k: v for k, v in d.items() if k not in ("id", "user_id", "computed_at")}
                return {"user_id": user_id, "features": features, "feature_timestamp": str(d.get("computed_at", ""))}
        except Exception as exc:  # noqa: BLE001
            logger.debug("DB feature query failed: %s", exc)

    return {"user_id": user_id, "features": {}, "feature_timestamp": None}


def get_predictions_for_user(user_id: str) -> dict[str, Any]:
    """Get the latest predictions (retention, purchase, LTV) for a user."""
    row = _user_row_from_parquet("predictions", user_id)
    if row is not None:
        return {
            "user_id": user_id,
            "retention_probability": row.get("retention_probability"),
            "churn_probability": row.get("churn_probability"),
            "purchase_probability": row.get("purchase_probability"),
            "predicted_ltv": row.get("predicted_ltv"),
            "retention_risk": row.get("churn_probability")
            if row.get("churn_probability") is not None
            else None,
        }

    for session in _db_session_safe():
        if session is None:
            break
        try:
            from sqlalchemy import text
            result = session.execute(
                text(
                    "SELECT * FROM predictions WHERE user_id = :uid "
                    "ORDER BY predicted_at DESC LIMIT 1"
                ),
                {"uid": user_id},
            )
            row_dict = result.mappings().first()
            if row_dict:
                d = dict(row_dict)
                return {
                    "user_id": user_id,
                    "retention_probability": d.get("retention_probability"),
                    "churn_probability": d.get("churn_probability"),
                    "purchase_probability": d.get("purchase_probability"),
                    "predicted_ltv": d.get("predicted_ltv"),
                    "retention_risk": d.get("churn_probability"),
                }
        except Exception as exc:  # noqa: BLE001
            logger.debug("DB prediction query failed: %s", exc)

    return {
        "user_id": user_id,
        "retention_probability": None,
        "churn_probability": None,
        "purchase_probability": None,
        "predicted_ltv": None,
        "retention_risk": None,
    }


def get_recommendation_for_user(user_id: str) -> dict[str, Any]:
    """Get the latest recommendation for a user."""
    row = _user_row_from_parquet("recommendations", user_id)
    if row is not None:
        return {
            "user_id": user_id,
            "recommended_action": (row.get("action") or "NO_ACTION"),
            "expected_value": row.get("expected_value"),
            "confidence": row.get("confidence"),
            "objective": row.get("objective"),
            "reasons": row.get("reasons", ""),
            "guardrails": row.get("guardrails", ""),
            "recommendation_timestamp": None,
        }

    for session in _db_session_safe():
        if session is None:
            break
        try:
            from sqlalchemy import text
            result = session.execute(
                text(
                    "SELECT * FROM recommendations WHERE user_id = :uid "
                    "ORDER BY recommended_at DESC LIMIT 1"
                ),
                {"uid": user_id},
            )
            row_dict = result.mappings().first()
            if row_dict:
                d = dict(row_dict)
                return {
                    "user_id": user_id,
                    "recommended_action": d.get("recommended_action") or "NO_ACTION",
                    "expected_value": d.get("expected_value"),
                    "confidence": d.get("confidence"),
                    "objective": d.get("objective"),
                    "reasons": str(d.get("explanation", "")),
                    "guardrails": str(d.get("guardrails_applied", "")),
                    "recommendation_timestamp": str(d.get("recommended_at", "")),
                }
        except Exception as exc:  # noqa: BLE001
            logger.debug("DB recommendation query failed: %s", exc)

    return {
        "user_id": user_id,
        "recommended_action": "NO_ACTION",
        "expected_value": None,
        "confidence": None,
        "objective": None,
        "reasons": "",
        "guardrails": "",
        "recommendation_timestamp": None,
    }


# ---------------------------------------------------------------------------
# Decision pipeline
# ---------------------------------------------------------------------------

def run_full_decision(user_id: str) -> dict[str, Any]:
    """Run the full pipeline: features -> predictions -> state -> decision -> explanation.

    Every computation is delegated to the decision engine and LLM provider;
    the LLM only *explains* a decision the deterministic engine already made.
    """
    features_data = get_features_for_user(user_id)
    predictions_data = get_predictions_for_user(user_id)

    features = features_data.get("features", {})
    preds = {
        k: predictions_data.get(k)
        for k in (
            "retention_probability",
            "churn_probability",
            "purchase_probability",
            "predicted_ltv",
        )
    }

    # Build a feature dict that satisfies the decision engine's UserState.
    feature_vector = {
        "user_id": user_id,
        **features,
        **preds,
        "engagement_trend": features.get("engagement_change_7d", 0.0),
        "days_since_last_session": int(features.get("days_since_last_session", 0) or 0),
        "notification_open_rate": float(features.get("notification_open_rate", 0.0) or 0.0),
        "revenue_30d": float(features.get("revenue_30d", 0.0) or 0.0),
        "sessions_7d": int(features.get("sessions_7d", 0) or 0),
        "active_days_30d": int(features.get("days_active_30d", 0) or 0),
    }

    # --- Decision engine ---
    user_state: dict[str, Any] = {}
    decision: dict[str, Any] = {"action": "NO_ACTION", "confidence": 0.0, "expected_value": 0.0}
    guardrails: list[str] = []
    reasons: list[str] = []
    try:
        from src.decision_engine import build_user_state as build_decision_state, decide

        state_obj = build_decision_state(feature_vector, preds)
        decision_obj = decide(state_obj, business_objective="retention")
        decision = decision_obj.model_dump()
        guardrails = decision_obj.guardrails
        reasons = decision_obj.reasons
    except ImportError as exc:  # pragma: no cover
        logger.warning("Decision engine unavailable: %s", exc)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Decision engine failed: %s", exc)

    # Structured, portfolio-facing user state (engagement label, retention risk,
    # purchase propensity, LTV, value segment, notification response).
    try:
        from src.models.state import build_user_state as build_structured_state
        structured = build_structured_state(user_id, features, preds)
        user_state = structured.model_dump()
    except ImportError:  # pragma: no cover
        user_state = {"user_id": user_id}
    except Exception as exc:  # noqa: BLE001
        logger.warning("Structured user state failed: %s", exc)
        user_state = {"user_id": user_id}

    # --- LLM explanation ---
    explanation: str | None = None
    try:
        from src.llm import ExplanationInput, get_explanation_provider

        provider = get_explanation_provider()
        explanation_input = ExplanationInput(
            user_id=user_id,
            retention_probability=float(preds.get("retention_probability") or 0.0),
            purchase_probability=float(preds.get("purchase_probability") or 0.0),
            engagement_trend=float(features.get("engagement_change_7d", 0.0) or 0.0),
            days_since_last_session=int(features.get("days_since_last_session", 0) or 0),
            recommended_action=decision.get("action", "NO_ACTION"),
            reasons=reasons,
            guardrails=guardrails,
        )
        explanation_out = provider.explain_decision(explanation_input)
        explanation = explanation_out.summary or explanation_out.reasoning
    except ImportError:  # pragma: no cover
        explanation = None
    except Exception as exc:  # noqa: BLE001
        logger.warning("LLM explanation failed: %s", exc)
        explanation = None

    return {
        "user_id": user_id,
        "features": features,
        "predictions": {k: preds[k] for k in preds},
        "user_state": user_state,
        "decision": decision,
        "explanation": explanation,
        "decision_timestamp": datetime.now().isoformat(),
    }


# ---------------------------------------------------------------------------
# Experiments
# ---------------------------------------------------------------------------

def list_experiments() -> list[dict[str, Any]]:
    """List all experiments (from the experiment_results metadata or DB)."""
    df = _load_parquet("experiment_results")
    if df is not None and not df.empty and "experiment_id" in df.columns:
        ids = df["experiment_id"].unique()
        return [
            {
                "experiment_id": eid,
                "name": f"Attention Experiment ({eid})",
                "status": "completed",
                "created_at": None,
            }
            for eid in ids
        ]

    for session in _db_session_safe():
        if session is None:
            break
        try:
            from sqlalchemy import text
            result = session.execute(text("SELECT * FROM experiments"))
            rows = result.mappings().all()
            return [
                {
                    "experiment_id": r["id"] if "id" in r.keys() else r.get("experiment_id", ""),
                    "name": r.get("experiment_name") or r.get("name") or "",
                    "status": r.get("status", "unknown"),
                    "created_at": str(r.get("created_at", "")) if r.get("created_at") else None,
                }
                for r in rows
            ]
        except Exception as exc:  # noqa: BLE001
            logger.debug("DB experiments query failed: %s", exc)

    return []


def get_experiment_results(experiment_id: str) -> dict[str, Any]:
    """Return per-variant analysis for an experiment."""
    df = _load_parquet("experiment_results")
    if df is not None and not df.empty and "experiment_id" in df.columns:
        match = df[df["experiment_id"] == experiment_id]
        if not match.empty:
            return _summarize_experiment(match)

    for session in _db_session_safe():
        if session is None:
            break
        try:
            from sqlalchemy import text
            result = session.execute(
                text(
                    "SELECT * FROM outcomes WHERE experiment_id = :eid"
                ),
                {"eid": experiment_id},
            )
            rows = result.mappings().all()
            if rows:
                return _summarize_experiment(pd.DataFrame([dict(r) for r in rows]))
        except Exception as exc:  # noqa: BLE001
            logger.debug("DB experiment results failed: %s", exc)

    return {
        "experiment_id": experiment_id,
        "name": "",
        "status": "unknown",
        "variant_assignments": {},
        "conversion_rates": {},
        "statistical_significance": {},
        "recommendation": None,
    }


def _summarize_experiment(df: pd.DataFrame) -> dict[str, Any]:
    """Aggregate per-variant outcome stats into the results schema."""
    variant_assignments: dict[str, int] = {}
    conversion_rates: dict[str, float] = {}
    statistical_significance: dict[str, Any] = {}

    if "variant" in df.columns and "outcome" in df.columns:
        variant_col = "variant"
        outcome_col = "outcome"
    elif "intervention_type" in df.columns:
        variant_col = "intervention_type"
        outcome_col = "metric_value"
    else:
        variant_col = outcome_col = None

    if variant_col and outcome_col:
        groups = df.groupby(variant_col)
        variants = list(groups.groups.keys())

        # Determine control
        control = "control" if "control" in variants else variants[0]

        for variant in variants:
            grp = groups.get_group(variant)[outcome_col].astype(float)
            n = int(grp.count())
            mean = float(grp.mean())
            std = float(grp.std(ddof=0)) if n > 1 else 0.0
            variant_assignments[variant] = n
            conversion_rates[variant] = mean

        # Simple statistical comparison vs control
        try:
            from scipy import stats as _stats
            control_grp = groups.get_group(control)[outcome_col].astype(float)
            for variant in variants:
                if variant == control:
                    statistical_significance[variant] = {
                        "mean": conversion_rates[variant],
                        "sample_size": variant_assignments[variant],
                        "p_value": None,
                        "is_significant": False,
                        "is_control": True,
                    }
                    continue
                grp = groups.get_group(variant)[outcome_col].astype(float)
                t, p = _stats.ttest_ind(control_grp, grp, equal_var=False)
                lift = conversion_rates[variant] - conversion_rates[control]
                statistical_significance[variant] = {
                    "mean": conversion_rates[variant],
                    "sample_size": variant_assignments[variant],
                    "lift": round(float(lift), 4),
                    "p_value": round(float(p), 4),
                    "is_significant": bool(p < 0.05),
                    "is_control": False,
                }
        except Exception as exc:  # noqa: BLE001
            logger.debug("Stats calc failed (may need scipy): %s", exc)

    best_variant = max(
        variant_assignments,
        key=lambda v: (conversion_rates.get(v, 0.0), 0.0),
        default=None,
    )
    recommendation = None
    if best_variant and best_variant != "control":
        recommendation = (
            f"Variant '{best_variant}' shows the highest mean outcome "
            f"({conversion_rates[best_variant]:.3f}). "
        )
        sig = statistical_significance.get(best_variant, {})
        if sig.get("is_significant"):
            recommendation += "Difference vs control is statistically significant (p < 0.05)."
        else:
            recommendation += "Difference vs control is NOT statistically significant."

    return {
        "experiment_id": str(df["experiment_id"].iloc[0]) if "experiment_id" in df.columns else "",
        "name": "Attention Experiment",
        "status": "completed",
        "variant_assignments": variant_assignments,
        "conversion_rates": conversion_rates,
        "statistical_significance": statistical_significance,
        "recommendation": recommendation,
    }


# ---------------------------------------------------------------------------
# Overview metrics
# ---------------------------------------------------------------------------

def get_overview_metrics() -> dict[str, Any]:
    """Overall product metrics: totals, at-risk, revenue, conversion rate."""
    users_df = _load_users()
    predictions_df = _load_parquet("predictions")
    recommendations_df = _load_parquet("recommendations")

    total_users = int(len(users_df)) if users_df is not None else 0

    active_users = 0
    if users_df is not None and not users_df.empty and "total_sessions" in users_df.columns:
        active_users = int((users_df["total_sessions"] > 0).sum())

    at_risk_users = 0
    retention_risk_avg = 0.0
    if predictions_df is not None and not predictions_df.empty:
        risk_series = None
        if "churn_probability" in predictions_df.columns:
            risk_series = predictions_df["churn_probability"]
        elif "retention_risk" in predictions_df.columns:
            risk_series = predictions_df["retention_risk"]
        if risk_series is not None:
            retention_risk_avg = float(risk_series.mean())
            at_risk_users = int((risk_series > 0.7).sum())

    total_revenue = 0.0
    if users_df is not None and not users_df.empty and "total_spend" in users_df.columns:
        total_revenue = float(users_df["total_spend"].sum())

    conversion_rate = 0.0
    if users_df is not None and not users_df.empty and "purchase_count" in users_df.columns:
        total_users_safe = max(total_users, 1)
        conversion_rate = float((users_df["purchase_count"] > 0).sum()) / total_users_safe

    return {
        "total_users": total_users,
        "active_users": active_users,
        "at_risk_users": at_risk_users,
        "retention_risk_avg": retention_risk_avg,
        "total_revenue": total_revenue,
        "conversion_rate": conversion_rate,
        "timestamp": datetime.now().isoformat(),
    }
