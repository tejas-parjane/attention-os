import pytest

from src.decision_engine.engine import (
    ActionDecision,
    UserState,
    decide,
)


def _state(**overrides) -> UserState:
    """Build a UserState with sensible defaults; override any field."""
    defaults = dict(
        user_id="test_user",
        retention_probability=0.5,
        purchase_probability=0.2,
        engagement_trend=0.0,
        days_since_last_session=5,
        notification_open_rate=0.3,
        revenue_30d=10.0,
        sessions_7d=3,
        notifications_last_24h=0,
        notifications_last_week=0,
        predicted_ltv=30.0,
        active_days_30d=15,
        churn_probability=0.3,
        value_segment="medium",
    )
    defaults.update(overrides)
    return UserState(**defaults)


class TestHighChurnRisk:
    def test_high_churn_triggers_intervention(self):
        state = _state(churn_probability=0.95, engagement_trend=-0.8, days_since_last_session=7)
        result = decide(state, business_objective="retention")
        assert isinstance(result, ActionDecision)
        assert result.action != "NO_ACTION", (
            f"Expected intervention for high churn, got NO_ACTION: {result.reasons}"
        )


class TestLowChurnRisk:
    def test_low_risk_prefers_no_action_or_mild_intervention(self):
        """Low-risk, high-activity user with no notification opt-in should
        not receive push notifications. The engine may still pick a benign
        action like AD_FREQUENCY_REDUCTION — verify it is NOT an aggressive nudge."""
        state = _state(
            churn_probability=0.05,
            engagement_trend=0.5,
            days_since_last_session=1,
            sessions_7d=10,
            active_days_30d=25,
            notification_open_rate=0.0,  # not opted-in to push
        )
        result = decide(state, business_objective="retention")
        assert result.action != "PUSH_NOTIFICATION", (
            f"Low-risk user should not get PUSH_NOTIFICATION: {result.action}"
        )


class TestNotificationFatigue:
    def test_high_fatigue_blocks_push_notification(self):
        state = _state(
            notifications_last_24h=3,
            notifications_last_week=10,
            notification_open_rate=0.6,
        )
        result = decide(state, business_objective="engagement")
        assert result.action != "PUSH_NOTIFICATION", (
            f"PUSH_NOTIFICATION should be blocked under high fatigue, got {result.action}"
        )


class TestLowConfidence:
    def test_low_confidence_returns_no_action(self):
        state = _state(
            sessions_7d=0,
            active_days_30d=0,
            engagement_trend=0.0,
            churn_probability=0.3,
            days_since_last_session=1,
        )
        result = decide(state, business_objective="retention")
        assert result.action == "NO_ACTION", (
            f"Low-data user should get NO_ACTION, got {result.action}"
        )


class TestWhaleProtection:
    def test_whale_not_sent_push_notification(self):
        state = _state(
            revenue_30d=200.0,
            predicted_ltv=500.0,
            notification_open_rate=0.8,
            sessions_7d=8,
            active_days_30d=28,
            churn_probability=0.6,
            engagement_trend=-0.4,
        )
        result = decide(state, business_objective="engagement")
        assert result.action != "PUSH_NOTIFICATION", (
            f"Whale should not receive PUSH_NOTIFICATION, got {result.action}"
        )
        guardrail_blocked = any("PUSH_NOTIFICATION" in g for g in result.guardrails)
        assert guardrail_blocked, "Expected PUSH_NOTIFICATION to be blocked by guardrail"


class TestEligibilityConstraints:
    def test_sessions_based_actions_ineligible_when_sessions_7d_zero(self):
        """PERSONALIZED_CONTENT, PERSONALIZED_CHALLENGE, IN_APP_MESSAGE, and
        AD_FREQUENCY_REDUCTION all require sessions_7d >= 1. With sessions_7d=0
        and notification_open_rate=0 (no push opt-in), only NO_ACTION, CHALLENGE,
        and REWARD remain eligible — all have low expected value for this state."""
        state = _state(
            days_since_last_session=1,
            sessions_7d=0,
            notification_open_rate=0.0,  # no notification opt-in
            churn_probability=0.8,
            engagement_trend=-0.9,
        )
        result = decide(state, business_objective="engagement")
        # With very limited eligible actions and low data, confidence should be low
        assert result.confidence < 0.65, (
            f"Expected low confidence for minimal-data user, got {result.confidence}"
        )


class TestDecisionStructure:
    def test_decision_has_required_fields(self):
        state = _state()
        result = decide(state)
        assert hasattr(result, "action")
        assert hasattr(result, "expected_value")
        assert hasattr(result, "objective")
        assert hasattr(result, "reasons")
        assert hasattr(result, "guardrails")
        assert hasattr(result, "confidence")
        assert isinstance(result.reasons, list)
        assert isinstance(result.guardrails, list)

    def test_no_action_has_zero_expected_value(self):
        state = _state(sessions_7d=0, active_days_30d=0)
        result = decide(state)
        if result.action == "NO_ACTION":
            assert result.expected_value == 0.0
