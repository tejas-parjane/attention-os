from pydantic import BaseModel
from typing import Optional


class UserState(BaseModel):
    """Structured representation of a user's current state for the decision engine."""

    user_id: str
    engagement: str  # "declining" | "stable" | "improving"
    retention_risk: float
    purchase_propensity: float
    predicted_ltv: float
    recent_sessions: int
    days_since_last_session: int
    notification_response: str  # "positive" | "neutral" | "negative" | "none"
    value_segment: str  # "low" | "medium" | "high"
    revenue_30d: float = 0.0
    sessions_7d: int = 0
    engagement_trend: float = 0.0


def engagement_label(engagement_change: float, duration_trend: float) -> str:
    """
    Derive an engagement label from two trend signals.

    Thresholds:
        - declining : engagement_change_7d < -0.2
        - improving : engagement_change_7d >  0.2
        - stable    : everything in between
    """
    combined = (engagement_change + duration_trend) / 2.0
    if combined < -0.2:
        return "declining"
    if combined > 0.2:
        return "improving"
    return "stable"


def notification_label(open_rate: float, has_opens: bool) -> str:
    """
    Derive notification response label from historical open-rate.

    Thresholds:
        - positive : open_rate > 0.2
        - negative : open_rate < 0.05 and user has at least some opens
        - neutral  : everything else
        - none     : no notification history (has_opens is False)
    """
    if not has_opens:
        return "none"
    if open_rate > 0.2:
        return "positive"
    if open_rate < 0.05:
        return "negative"
    return "neutral"


def value_segment_label(revenue_30d: float) -> str:
    """
    Bucket a user into a value segment based on 30-day revenue.

    Thresholds:
        - low    : revenue_30d < 5
        - medium : revenue_30d < 30
        - high   : revenue_30d >= 30
    """
    if revenue_30d < 5.0:
        return "low"
    if revenue_30d < 30.0:
        return "medium"
    return "high"


def build_user_state(
    user_id: str,
    features: dict[str, float],
    predictions: dict[str, float],
) -> UserState:
    """
    Transform raw features + model predictions into a structured UserState.

    Derivation logic:
        - engagement        : combined trend from engagement_change_7d & session_duration_trend
        - retention_risk    : 1 - retention_probability (from predictions)
        - purchase_propensity: purchase_probability (from monetization model)
        - predicted_ltv     : total_ltv feature if present, else 0
        - notification_response : from notification_open_rate feature
        - value_segment     : from revenue_30d feature
    """
    engagement_change = features.get("engagement_change_7d", 0.0)
    duration_trend = features.get("session_duration_trend", 0.0)
    revenue_30d = features.get("revenue_30d", 0.0)
    notification_open_rate = features.get("notification_open_rate", 0.0)
    has_opens = notification_open_rate > 0.0 or features.get("total_notifications_opened", 0) > 0

    retention_prob = predictions.get("retention_probability", 0.5)
    purchase_prob = predictions.get("purchase_probability", 0.0)
    predicted_ltv = features.get("total_ltv", predictions.get("predicted_ltv", 0.0))

    return UserState(
        user_id=user_id,
        engagement=engagement_label(engagement_change, duration_trend),
        retention_risk=round(1.0 - retention_prob, 4),
        purchase_propensity=round(purchase_prob, 4),
        predicted_ltv=round(predicted_ltv, 2),
        recent_sessions=int(features.get("recent_sessions", 0)),
        days_since_last_session=int(features.get("days_since_last_session", 0)),
        notification_response=notification_label(notification_open_rate, has_opens),
        value_segment=value_segment_label(revenue_30d),
        revenue_30d=round(revenue_30d, 2),
        sessions_7d=int(features.get("sessions_7d", 0)),
        engagement_trend=round(engagement_change, 4),
    )
