"""Deterministic decision engine for attention-optimized interventions.

Scores candidate actions against a user's current state, applies guardrails
to filter unsafe or counterproductive interventions, and returns the single
highest-value action — or NO_ACTION when confidence is insufficient.

Scoring formula
---------------
    expected_value = predicted_impact * objective_weight * user_value
                     - action_cost - fatigue_penalty - risk_penalty

Design principles
-----------------
* Deterministic: same inputs always produce the same output.
* Conservative: NO_ACTION is preferred when confidence < 0.5 or EV is negative.
* High-value user protection: users above a revenue threshold are shielded
  from high-fatigue / low-value interventions.
"""

from __future__ import annotations

import logging
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, field_validator

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Guardrails
# ---------------------------------------------------------------------------

MAX_NOTIFICATIONS_PER_DAY: int = 2
MAX_NOTIFICATIONS_PER_WEEK: int = 8
MIN_TIME_BETWEEN_INTERVENTIONS_HOURS: int = 6
HIGH_VALUE_USER_MIN_REVENUE: float = 50.0


# ---------------------------------------------------------------------------
# Action catalogue
# ---------------------------------------------------------------------------


class Action(BaseModel):
    """Immutable definition of a single intervention action."""

    name: str
    cost: float
    risk: float
    expected_objective: str
    eligibility: dict[str, Any] = Field(default_factory=dict)
    frequency_constraint: int = Field(
        default=999, description="Maximum times the action may be shown per 24 h."
    )
    description: str = ""


ACTION_CATALOG: dict[str, Action] = {
    "NO_ACTION": Action(
        name="NO_ACTION",
        cost=0,
        risk=0,
        expected_objective="baseline",
        eligibility={},
        frequency_constraint=999,
        description="Do nothing — serve organic experience.",
    ),
    "PERSONALIZED_CONTENT": Action(
        name="PERSONALIZED_CONTENT",
        cost=1,
        risk=0.1,
        expected_objective="engagement",
        eligibility={"sessions_7d__gte": 1},
        frequency_constraint=4,
        description="Serve AI-curated content matched to user interests.",
    ),
    "PERSONALIZED_CHALLENGE": Action(
        name="PERSONALIZED_CHALLENGE",
        cost=1,
        risk=0.15,
        expected_objective="engagement",
        eligibility={"sessions_7d__gte": 1},
        frequency_constraint=3,
        description="Present a personalised micro-challenge.",
    ),
    "CHALLENGE": Action(
        name="CHALLENGE",
        cost=1,
        risk=0.1,
        expected_objective="engagement",
        eligibility={},
        frequency_constraint=3,
        description="Generic engagement challenge.",
    ),
    "DISCOUNT": Action(
        name="DISCOUNT",
        cost=3,
        risk=0.2,
        expected_objective="monetization",
        eligibility={"purchase_probability__gte": 0.3},
        frequency_constraint=2,
        description="Offer a time-limited discount to encourage purchase.",
    ),
    "REWARD": Action(
        name="REWARD",
        cost=2,
        risk=0.15,
        expected_objective="retention",
        eligibility={},
        frequency_constraint=3,
        description="Grant an in-app reward to reinforce loyalty.",
    ),
    "PUSH_NOTIFICATION": Action(
        name="PUSH_NOTIFICATION",
        cost=1,
        risk=0.25,
        expected_objective="engagement",
        eligibility={"notification_opt_in": True},
        frequency_constraint=2,
        description="Send a push notification (fatigue-sensitive).",
    ),
    "IN_APP_MESSAGE": Action(
        name="IN_APP_MESSAGE",
        cost=1,
        risk=0.1,
        expected_objective="engagement",
        eligibility={"sessions_7d__gte": 1},
        frequency_constraint=4,
        description="Display an in-app message or tooltip.",
    ),
    "CROSS_SELL": Action(
        name="CROSS_SELL",
        cost=2,
        risk=0.3,
        expected_objective="monetization",
        eligibility={"purchase_probability__gte": 0.4},
        frequency_constraint=1,
        description="Recommend a complementary product or feature.",
    ),
    "AD_FREQUENCY_REDUCTION": Action(
        name="AD_FREQUENCY_REDUCTION",
        cost=1,
        risk=0.05,
        expected_objective="retention",
        eligibility={"sessions_7d__gte": 1},
        frequency_constraint=999,
        description="Temporarily reduce ad frequency for the user.",
    ),
}


# ---------------------------------------------------------------------------
# Business objective weights
# ---------------------------------------------------------------------------

OBJECTIVE_WEIGHTS: dict[str, float] = {
    "retention": 1.0,
    "engagement": 0.8,
    "monetization": 0.7,
    "revenue": 0.7,
}


# ---------------------------------------------------------------------------
# User state
# ---------------------------------------------------------------------------


class UserState(BaseModel):
    """Snapshot of everything the engine needs to know about a user."""

    user_id: str
    retention_probability: float
    purchase_probability: float
    engagement_trend: float
    days_since_last_session: int
    notification_open_rate: float
    revenue_30d: float
    sessions_7d: int
    notifications_last_24h: int = 0
    notifications_last_week: int = 0
    predicted_ltv: float = 0.0
    active_days_30d: int = 0
    churn_probability: float = 0.0
    value_segment: str = "medium"

    @field_validator("retention_probability", "purchase_probability", "churn_probability")
    @classmethod
    def _clamp_probability(cls, v: float) -> float:
        return max(0.0, min(1.0, v))

    @field_validator("engagement_trend")
    @classmethod
    def _clamp_trend(cls, v: float) -> float:
        return max(-1.0, min(1.0, v))


# ---------------------------------------------------------------------------
# Decision artefacts
# ---------------------------------------------------------------------------


class ActionScore(BaseModel):
    """Intermediate scoring result for a single candidate action."""

    action_name: str
    predicted_impact: float
    objective_weight: float
    user_value: float
    action_cost: float
    fatigue_penalty: float
    risk_penalty: float
    expected_value: float
    confidence: float


class ActionDecision(BaseModel):
    """Final decision returned to the caller."""

    action: str
    expected_value: float
    objective: str
    reasons: list[str] = Field(default_factory=list)
    guardrails: list[str] = Field(default_factory=list)
    confidence: float = 0.0


# ---------------------------------------------------------------------------
# Eligibility helpers
# ---------------------------------------------------------------------------


def _check_eligibility(eligibility: dict[str, Any], state: UserState) -> tuple[bool, str]:
    """Return (eligible, reason) by evaluating every rule in *eligibility*."""
    state_dict = state.model_dump()
    for feature, condition in eligibility.items():
        if feature == "notification_opt_in":
            if state.notification_open_rate <= 0:
                return False, "notification_open_rate <= 0"
            continue
        # Feature suffix patterns like sessions_7d__gte
        parts = feature.split("__")
        base_feature = parts[0]
        operator = parts[1] if len(parts) > 1 else "eq"
        value = state_dict.get(base_feature)
        if value is None:
            return False, f"feature {base_feature} not in state"
        if operator == "gte" and value < condition:
            return False, f"{base_feature}={value} < {condition}"
        elif operator == "lte" and value > condition:
            return False, f"{base_feature}={value} > {condition}"
        elif operator == "eq" and value != condition:
            return False, f"{base_feature}={value} != {condition}"
        elif operator == "gt" and value <= condition:
            return False, f"{base_feature}={value} <= {condition}"
        elif operator == "lt" and value >= condition:
            return False, f"{base_feature}={value} >= {condition}"
    return True, ""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def build_user_state(features: dict[str, Any], predictions: dict[str, Any]) -> UserState:
    """Construct a :class:`UserState` from raw feature and prediction dicts.

    Parameters
    ----------
    features:
        Keyed raw features (e.g. from a feature store).
    predictions:
        Model outputs such as ``predicted_ltv`` or ``churn_probability``.

    Returns
    -------
    UserState
        A validated, clamped user state ready for scoring.
    """
    merged: dict[str, Any] = {**features, **predictions}
    state = UserState(**merged)
    logger.debug("Built UserState for user %s", state.user_id)
    return state


def apply_guardrails(state: UserState, candidates: list[Action]) -> list[Action]:
    """Filter *candidates* through hard guardrails.

    Guardrails are non-negotiable safety rules that prevent the engine from
    selecting interventions that could harm long-term retention or annoy
    high-value users.

    Returns the filtered list (may be empty, in which case the engine should
    fall back to ``NO_ACTION``).
    """
    filtered: list[Action] = []
    blocked: list[str] = []

    for action in candidates:
        if action.name == "NO_ACTION":
            filtered.append(action)
            continue

        block_reason: str | None = None

        # --- Notification fatigue guardrails ---
        if action.name == "PUSH_NOTIFICATION":
            if state.notifications_last_24h >= MAX_NOTIFICATIONS_PER_DAY:
                block_reason = (
                    f"notifications_last_24h ({state.notifications_last_24h}) "
                    f">= MAX_NOTIFICATIONS_PER_DAY ({MAX_NOTIFICATIONS_PER_DAY})"
                )
            elif state.notifications_last_week >= MAX_NOTIFICATIONS_PER_WEEK:
                block_reason = (
                    f"notifications_last_week ({state.notifications_last_week}) "
                    f">= MAX_NOTIFICATIONS_PER_WEEK ({MAX_NOTIFICATIONS_PER_WEEK})"
                )

        # --- High-value user protection ---
        if (
            state.revenue_30d > HIGH_VALUE_USER_MIN_REVENUE
            and action.name in {"PUSH_NOTIFICATION", "CROSS_SELL"}
        ):
            block_reason = (
                f"high-value user (revenue_30d={state.revenue_30d:.2f} "
                f"> {HIGH_VALUE_USER_MIN_REVENUE}) — blocking {action.name}"
            )

        if block_reason:
            blocked.append(f"{action.name}: {block_reason}")
            logger.info("Guardrail blocked %s for user %s: %s", action.name, state.user_id, block_reason)
            continue

        filtered.append(action)

    if blocked:
        logger.debug("Blocked actions for user %s: %s", state.user_id, blocked)

    return filtered


def get_candidate_actions(state: UserState) -> list[Action]:
    """Return every action the user is eligible for, plus ``NO_ACTION``.

    Eligibility is determined by the ``eligibility`` rules on each
    :class:`Action`.  ``NO_ACTION`` is always included.
    """
    candidates: list[Action] = [ACTION_CATALOG["NO_ACTION"]]

    for name, action in ACTION_CATALOG.items():
        if name == "NO_ACTION":
            continue
        eligible, reason = _check_eligibility(action.eligibility, state)
        if eligible:
            candidates.append(action)
        else:
            logger.debug("Action %s ineligible for user %s: %s", name, state.user_id, reason)

    logger.debug(
        "Candidate actions for user %s: %s",
        state.user_id,
        [a.name for a in candidates],
    )
    return candidates


def _compute_predicted_impact(action: Action, state: UserState) -> float:
    """Derive a [0, 1] predicted impact score from user state + action type.

    The heuristic is intentionally transparent: each action maps to the user
    signal it most directly influences.
    """
    if action.name == "NO_ACTION":
        return 0.0

    # Engagement-oriented actions are boosted when engagement is trending down
    if action.expected_objective == "engagement":
        # Negative trend = higher potential lift from engagement actions
        trend_component = max(0.0, -state.engagement_trend)
        open_rate_component = state.notification_open_rate if action.name == "PUSH_NOTIFICATION" else 0.0
        return min(1.0, 0.3 + 0.5 * trend_component + 0.2 * open_rate_component)

    # Retention actions matter most when churn risk is high
    if action.expected_objective == "retention":
        churn_component = state.churn_probability
        recency_component = min(1.0, state.days_since_last_session / 14.0)
        return min(1.0, 0.2 + 0.5 * churn_component + 0.3 * recency_component)

    # Monetization actions depend on purchase propensity
    if action.expected_objective == "monetization":
        return min(1.0, 0.2 + 0.8 * state.purchase_probability)

    return 0.3  # fallback baseline


def _compute_fatigue_penalty(action: Action, state: UserState) -> float:
    """Fatigue penalty scales with recent notification volume.

    Returns a value in [0, 1] that is subtracted from expected value.
    """
    if action.name not in {"PUSH_NOTIFICATION", "IN_APP_MESSAGE"}:
        return 0.0

    daily_pressure = state.notifications_last_24h / MAX_NOTIFICATIONS_PER_DAY
    weekly_pressure = state.notifications_last_week / MAX_NOTIFICATIONS_PER_WEEK
    return min(1.0, 0.4 * daily_pressure + 0.6 * weekly_pressure)


def score_action(
    action: Action,
    state: UserState,
    business_objective: str = "retention",
) -> ActionScore:
    """Compute a full :class:`ActionScore` for a single action.

    Parameters
    ----------
    action:
        The action to evaluate.
    state:
        Current user state snapshot.
    business_objective:
        One of ``retention``, ``engagement``, ``monetization``, ``revenue``.

    Returns
    -------
    ActionScore
        All intermediate values plus the final ``expected_value``.
    """
    objective_weight = OBJECTIVE_WEIGHTS.get(business_objective, 0.5)
    user_value = max(state.predicted_ltv, state.revenue_30d * 0.3)
    predicted_impact = _compute_predicted_impact(action, state)
    action_cost = action.cost / 10.0  # normalise to ~[0, 0.3]
    fatigue_penalty = _compute_fatigue_penalty(action, state) * 0.5
    risk_penalty = action.risk * min(1.0, user_value / 100.0)

    expected_value = (
        predicted_impact * objective_weight * (1.0 + user_value / 100.0)
        - action_cost
        - fatigue_penalty
        - risk_penalty
    )

    # Confidence heuristic: higher when we have enough data and impact is clear.
    # The data-volume terms saturate quickly; the model / impact signal is the
    # principal driver that separates "act now" from "wait".
    data_sufficiency = 0.7 * min(1.0, state.sessions_7d / 4.0) + 0.3 * min(
        1.0, state.active_days_30d / 12.0
    )
    confidence = 0.15 + 0.60 * data_sufficiency + 0.25 * predicted_impact

    score = ActionScore(
        action_name=action.name,
        predicted_impact=round(predicted_impact, 4),
        objective_weight=objective_weight,
        user_value=round(user_value, 4),
        action_cost=round(action_cost, 4),
        fatigue_penalty=round(fatigue_penalty, 4),
        risk_penalty=round(risk_penalty, 4),
        expected_value=round(expected_value, 4),
        confidence=round(confidence, 4),
    )

    logger.debug(
        "Scored %s for user %s: EV=%.4f conf=%.4f",
        action.name,
        state.user_id,
        expected_value,
        confidence,
    )
    return score


def decide(state: UserState, business_objective: str = "retention") -> ActionDecision:
    """Select the best action for *state* under the given business objective.

    The decision pipeline is:

    1. Enumerate candidate actions (eligibility filter).
    2. Apply hard guardrails (fatigue, high-value protection).
    3. Score every surviving candidate.
    4. Select the highest-expected-value action — **unless** confidence is
       below 0.5 or the best EV is negative, in which case ``NO_ACTION``
       is returned.

    Parameters
    ----------
    state:
        Current user state snapshot.
    business_objective:
        One of ``retention``, ``engagement``, ``monetization``, ``revenue``.

    Returns
    -------
    ActionDecision
        The recommended action with full reasoning.
    """
    if business_objective not in OBJECTIVE_WEIGHTS:
        logger.warning(
            "Unknown business_objective '%s' — defaulting to 'retention'",
            business_objective,
        )
        business_objective = "retention"

    # 1. Candidates
    candidates = get_candidate_actions(state)

    # 2. Guardrails
    eligible = apply_guardrails(state, candidates)

    # 3. Score
    scores: list[ActionScore] = []
    for action in eligible:
        scores.append(score_action(action, state, business_objective))

    # 4. Select
    if not scores:
        logger.info("No eligible actions for user %s — returning NO_ACTION", state.user_id)
        return ActionDecision(
            action="NO_ACTION",
            expected_value=0.0,
            objective=business_objective,
            reasons=["No eligible actions after guardrails"],
            guardrails=["All candidates blocked"],
            confidence=0.0,
        )

    scores.sort(key=lambda s: s.expected_value, reverse=True)
    best = scores[0]

    reasons: list[str] = [
        f"Best expected value: {best.expected_value:.4f}",
        f"Predicted impact: {best.predicted_impact:.4f}",
        f"Objective weight ({business_objective}): {best.objective_weight}",
        f"User value: {best.user_value:.4f}",
        f"Fatigue penalty: {best.fatigue_penalty:.4f}",
        f"Risk penalty: {best.risk_penalty:.4f}",
    ]

    guardrails_applied: list[str] = []
    blocked_names = {a.name for a in candidates} - {a.name for a in eligible}
    for name in sorted(blocked_names):
        guardrails_applied.append(f"Blocked by guardrail: {name}")

    # Conservative fallback: NO_ACTION wins when confidence or EV is too low
    if best.confidence < 0.5:
        reasons.append(
            f"Confidence {best.confidence:.4f} < 0.50 threshold — falling back to NO_ACTION"
        )
        logger.info(
            "Low confidence (%.4f) for user %s — selecting NO_ACTION",
            best.confidence,
            state.user_id,
        )
        return ActionDecision(
            action="NO_ACTION",
            expected_value=0.0,
            objective=business_objective,
            reasons=reasons,
            guardrails=guardrails_applied,
            confidence=best.confidence,
        )

    if best.expected_value < 0:
        reasons.append(
            f"Best EV {best.expected_value:.4f} is negative — selecting NO_ACTION"
        )
        logger.info(
            "Negative EV (%.4f) for user %s — selecting NO_ACTION",
            best.expected_value,
            state.user_id,
        )
        return ActionDecision(
            action="NO_ACTION",
            expected_value=0.0,
            objective=business_objective,
            reasons=reasons,
            guardrails=guardrails_applied,
            confidence=best.confidence,
        )

    # High-value user protection: if the best action is a low-value nudge,
    # prefer NO_ACTION for users above the revenue threshold.
    low_value_actions = {"PUSH_NOTIFICATION", "CROSS_SELL"}
    if (
        state.revenue_30d > HIGH_VALUE_USER_MIN_REVENUE
        and best.action_name in low_value_actions
    ):
        reasons.append(
            f"High-value user (revenue_30d={state.revenue_30d:.2f}) — "
            f"preferred non-nudging action over {best.action_name}"
        )
        # Check if there's a non-nudging alternative with positive EV
        safe_alternatives = [
            s for s in scores
            if s.action_name not in low_value_actions and s.expected_value > 0
        ]
        if safe_alternatives:
            best = safe_alternatives[0]
            reasons.append(f"Switched to safer alternative: {best.action_name}")
        else:
            reasons.append("No safe alternative with positive EV — falling back to NO_ACTION")
            return ActionDecision(
                action="NO_ACTION",
                expected_value=0.0,
                objective=business_objective,
                reasons=reasons,
                guardrails=guardrails_applied,
                confidence=best.confidence,
            )

    logger.info(
        "Decision for user %s: %s (EV=%.4f, conf=%.4f, objective=%s)",
        state.user_id,
        best.action_name,
        best.expected_value,
        best.confidence,
        business_objective,
    )

    return ActionDecision(
        action=best.action_name,
        expected_value=best.expected_value,
        objective=business_objective,
        reasons=reasons,
        guardrails=guardrails_applied,
        confidence=best.confidence,
    )
