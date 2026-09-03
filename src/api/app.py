"""FastAPI application for AttentionOS.

All route handlers are thin — business logic lives in src/api/services.py.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from src.api.schemas import (
    DecisionOut,
    ExperimentOut,
    ExperimentResultsOut,
    FeaturesOut,
    OverviewMetricsOut,
    PredictionsOut,
    RecommendationOut,
    UserOut,
)
from src.api.services import (
    get_experiment_results,
    get_features_for_user,
    get_overview_metrics,
    get_predictions_for_user,
    get_recommendation_for_user,
    list_experiments,
    run_full_decision,
)


# ---------------------------------------------------------------------------
# Lifespan (startup / shutdown)
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):  # noqa: ANN201
    """Initialize DB on startup if configured."""
    try:
        from src.data.database import init_db
        init_db()
    except Exception:  # noqa: BLE001
        pass
    yield


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="AttentionOS",
    version="1.0.0",
    description="AI User Retention & Monetization Decision Engine",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/health", response_model=dict[str, Any])
def health() -> dict[str, Any]:
    """Check DB connectivity + app status."""
    status = "healthy"
    db_ok = True
    try:
        from src.data.database import get_engine
        engine = get_engine()
        with engine.connect() as conn:
            conn.execute(__import__("sqlalchemy").text("SELECT 1"))
    except Exception:  # noqa: BLE001
        status = "degraded"
        db_ok = False
    return {"status": status, "db_connected": db_ok, "version": "1.0.0"}


@app.get("/users/{user_id}", response_model=UserOut)
def get_user(user_id: str) -> dict[str, Any]:
    """Get user profile (from the generator's user metadata)."""
    try:
        from src.api.services import _load_users
        users_df = _load_users()
        if users_df is None or users_df.empty or "user_id" not in users_df.columns:
            raise HTTPException(status_code=404, detail=f"User {user_id} not found")
        row = users_df[users_df["user_id"] == user_id]
        if row.empty:
            raise HTTPException(status_code=404, detail=f"User {user_id} not found")
        d = row.iloc[0].to_dict()
        return_meta = {k: (str(v) if k in ("first_seen", "last_seen") else v) for k, v in d.items() if k != "user_id"}
        return {"user_id": user_id, "metadata": return_meta}
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/users/{user_id}/features", response_model=FeaturesOut)
def get_user_features(user_id: str) -> dict[str, Any]:
    """Get the most recent feature vector for a user."""
    return get_features_for_user(user_id)


@app.get("/users/{user_id}/predictions", response_model=PredictionsOut)
def get_user_predictions(user_id: str) -> dict[str, Any]:
    """Get the latest predictions (retention, purchase, ltv) for a user."""
    return get_predictions_for_user(user_id)


@app.get("/users/{user_id}/recommendation", response_model=RecommendationOut)
def get_user_recommendation(user_id: str) -> dict[str, Any]:
    """Get the latest recommendation for a user."""
    return get_recommendation_for_user(user_id)


@app.post("/users/{user_id}/decision", response_model=DecisionOut)
def make_decision(user_id: str) -> dict[str, Any]:
    """Run the full decision pipeline: features -> predictions -> state -> decision -> explanation."""
    return run_full_decision(user_id)


@app.get("/experiments", response_model=list[ExperimentOut])
def list_experiments_endpoint() -> list[dict[str, Any]]:
    """List all experiments."""
    return list_experiments()


@app.get("/experiments/{experiment_id}/results", response_model=ExperimentResultsOut)
def experiment_results(experiment_id: str) -> dict[str, Any]:
    """Get experiment analysis results."""
    return get_experiment_results(experiment_id)


@app.get("/metrics/overview", response_model=OverviewMetricsOut)
def metrics_overview() -> dict[str, Any]:
    """Overall product metrics: total users, active, at-risk, retention risk avg, revenue, conversion rate."""
    return get_overview_metrics()
