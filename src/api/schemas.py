"""Pydantic response models for AttentionOS API endpoints."""

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class UserOut(BaseModel):
    """User profile response."""

    user_id: str
    signup_date: str | None = None
    plan: str | None = None
    status: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class FeaturesOut(BaseModel):
    """Feature vector for a user."""

    user_id: str
    features: dict[str, Any]
    feature_timestamp: str | None = None


class PredictionsOut(BaseModel):
    """Latest predictions for a user."""

    user_id: str
    retention_risk: float | None = None
    purchase_probability: float | None = None
    predicted_ltv: float | None = None
    prediction_timestamp: str | None = None


class RecommendationOut(BaseModel):
    """Recommendation for a user."""

    user_id: str
    recommended_action: str
    segment: str | None = None
    intervention_type: str | None = None
    recommendation_timestamp: str | None = None


class DecisionOut(BaseModel):
    """Full decision pipeline output."""

    user_id: str
    features: dict[str, Any]
    predictions: dict[str, Any]
    user_state: dict[str, Any]
    decision: dict[str, Any]
    explanation: str | None = None
    decision_timestamp: str


class ExperimentOut(BaseModel):
    """Experiment summary."""

    experiment_id: str
    name: str
    status: str
    created_at: str | None = None


class ExperimentResultsOut(BaseModel):
    """Experiment analysis results."""

    experiment_id: str
    name: str
    status: str
    variant_assignments: dict[str, Any] = Field(default_factory=dict)
    conversion_rates: dict[str, Any] = Field(default_factory=dict)
    statistical_significance: dict[str, Any] = Field(default_factory=dict)
    recommendation: str | None = None


class OverviewMetricsOut(BaseModel):
    """Overall product metrics."""

    total_users: int
    active_users: int
    at_risk_users: int
    retention_risk_avg: float
    total_revenue: float
    conversion_rate: float
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat())
