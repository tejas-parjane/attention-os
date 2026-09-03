# Attention OS — Decision Engine

## Overview

The decision engine is the business logic layer that transforms ML predictions into actionable recommendations. It takes a user state (features + model scores) as input and produces a single recommended action — or explicitly decides that no action should be taken.

The engine is designed to be **configurable, auditable, and safe**. Every recommendation comes with a full scoring breakdown, guardrails evaluated, and an explanation of why this action was selected over alternatives.

---

## Business Objective Configuration

The decision engine is driven by a configurable objective function. Two primary objectives are supported:

| Objective | Description | Typical Use |
|---|---|---|
| **Retention** | Maximize the number of users who remain active over the next 7 days. | When the primary business risk is user churn. |
| **Monetization** | Maximize revenue per user over the next 7 days. | When the primary business goal is revenue growth. |
| **Balanced** | Weighted combination of retention and monetization. | The default — balances user health with business sustainability. |

**Configuration:**

```python
objective_config = {
    "mode": "balanced",
    "weights": {
        "retention": 0.6,
        "revenue": 0.4,
    },
    "time_horizon_days": 7,
    "risk_tolerance": "moderate",
}
```

The weights determine how much the scoring formula values retention impact versus revenue impact. A retention-focused configuration would weight `retention: 0.8, revenue: 0.2`.

---

## Candidate Actions Catalog

Each candidate action is defined in a catalog with metadata that the scoring engine uses to evaluate it.

### Action Definitions

| Action ID | Name | Target Segment | Estimated Cost | Risk Level | Description |
|---|---|---|---|---|---|
| `notify_reengage` | Re-engagement Notification | Users with declining activity | $0.01 | Low | Push notification reminding user of recent activity or content. |
| `notify_feature_highlight` | Feature Spotlight | Users who haven't tried key features | $0.01 | Low | Notification highlighting an underused feature. |
| `offer_discount_10` | 10% Discount Offer | Users with purchase propensity > 0.3 | $2.00 | Medium | Small discount to nudge a conversion. |
| `offer_discount_20` | 20% Discount Offer | Users with purchase propensity > 0.5 | $5.00 | Medium | Larger discount for higher-propensity users. |
| `offer_premium_trial` | Premium Trial Unlock | High-engagement free users | $3.00 | Medium | 7-day trial of premium features. |
| `send_case_study` | Success Story Share | Enterprise segment users | $0.50 | Low | Share a relevant case study or testimonial. |
| `schedule_checkin` | Personal Check-in | High-value at-risk users | $5.00 | Low | Personal message from success team. |
| `do_nothing` | No Action | All users | $0.00 | None | Explicitly choose not to intervene. |

### Eligibility Rules

Each action has eligibility rules that must be satisfied before it enters the candidate set:

```python
eligibility_rules = {
    "notify_reengage": {
        "min_days_since_last_session": 3,
        "max_notifications_7d": 5,
        "not_in_trial": True,
    },
    "offer_discount_10": {
        "monetization_propensity_min": 0.3,
        "days_since_last_purchase_min": 7,
        "max_discounts_30d": 1,
        "not_in_discount_cooldown": 7,  # days
    },
    "offer_discount_20": {
        "monetization_propensity_min": 0.5,
        "days_since_last_purchase_min": 14,
        "max_discounts_30d": 1,
        "not_in_discount_cooldown": 14,
    },
    "offer_premium_trial": {
        "is_premium": False,
        "retention_risk_min": 0.4,
        "sessions_last_30d_min": 5,
        "no_active_trial": True,
    },
    "schedule_checkin": {
        "retention_risk_min": 0.7,
        "lifetime_value_percentile_min": 90,
        "not_contacted_14d": True,
    },
    "do_nothing": {
        "always_eligible": True,
    },
}
```

### Action Risk Assessment

| Risk Level | Implication | Examples |
|---|---|---|
| **None** | No downside. Always safe to execute. | `do_nothing` |
| **Low** | Minimal cost, unlikely to annoy. Can be sent frequently. | Notifications, case studies |
| **Medium** | Moderate cost or potential for user fatigue. Requires cooldown enforcement. | Discounts, premium trials |
| **High** | Not currently in the catalog. Reserved for actions with significant cost or reputational risk. | (none — excluded by design) |

---

## Scoring Formula

For each eligible candidate action *a* and user *u*, the engine computes:

```
score(a, u) = w_r × Δ_retention(a, u)
            + w_rev × Δ_revenue(a, u)
            + w_ltv × Δ_ltv(a, u)
            − w_cost × normalized_cost(a)
            − w_risk × risk_penalty(a)
            − w_fatigue × fatigue_penalty(a, u)
```

### Term Definitions

| Term | Definition | Computation |
|---|---|---|
| `Δ_retention(a, u)` | Estimated change in retention probability if action *a* is applied to user *u*. | `model_retention(action_applied) − model_retention(baseline)`. In practice, estimated from historical action-response data or model-derived treatment effect estimates. |
| `Δ_revenue(a, u)` | Estimated change in 7-day revenue if action *a* is applied. | `p_purchase(action_applied) × avg_order_value − p_purchase(baseline) × avg_order_value`. |
| `Δ_ltv(a, u)` | Estimated change in lifetime value. | Long-term extension of the revenue estimate; simplified in this implementation as `Δ_revenue × expected_remaining_lifetime_months / 7`. |
| `normalized_cost(a)` | Cost of executing action *a*, normalized to [0, 1] across the action catalog. | `action_cost / max_action_cost`. |
| `risk_penalty(a)` | Penalty for action risk level. | `{None: 0, Low: 0.05, Medium: 0.15, High: 0.5}`. |
| `fatigue_penalty(a, u)` | Penalty for notification/offer fatigue. | Increases with recent notification count and recent discount count. Computed as `min(1.0, notifications_7d / max_notifications)`. |

### Weight Configuration

```python
scoring_weights = {
    "w_r": 0.40,       # retention impact
    "w_rev": 0.30,     # revenue impact
    "w_ltv": 0.10,     # lifetime value impact
    "w_cost": 0.10,    # cost sensitivity
    "w_risk": 0.05,    # risk aversion
    "w_fatigue": 0.05, # fatigue aversion
}
```

The weights are sum-normalized and can be adjusted to shift the engine's priorities. Increasing `w_fatigue` makes the engine more conservative about notification frequency. Increasing `w_risk` penalizes higher-cost actions more heavily.

---

## Guardrails

Guardrails are **hard constraints** that override the scoring formula. An action that scores highest but violates a guardrail is rejected, and the next-best action is selected.

### Guardrail Rules

| Guardrail | Rule | Rationale |
|---|---|---|
| **Notification Cap** | No user receives more than 5 notifications in a rolling 7-day window. | Prevents notification fatigue, which causes users to disable notifications or uninstall. |
| **Discount Frequency Cap** | No user receives more than 1 discount offer in a rolling 30-day window. | Prevents discount dependency — users should not learn to wait for discounts. |
| **High-Value User Protection** | Users in the top 10% by lifetime value receive only low-risk actions unless retention risk > 0.8. | High-value users are most expensive to lose but also most expensive to irritate. |
| **New User Protection** | Users with < 7 days of tenure receive only onboarding-related actions (feature highlights, no discounts). | New users need to learn the product, not be sold to. |
| **No-Action Threshold** | If no candidate action scores above the minimum expected value threshold (0.02), `do_nothing` is selected. | An action with negligible expected value is not worth the execution cost or fatigue cost. |
| **Cooldown Enforcement** | Actions with cooldowns (e.g., discount offers) are excluded from candidates if the user is in cooldown. | Prevents repeated intervention on the same user in a short window. |

### Guardrail Evaluation Order

```
1. Filter candidates by eligibility rules
2. Apply hard guardrails (notification cap, cooldown, etc.)
3. Score remaining candidates
4. Apply post-scoring guardrails (high-value protection, no-action threshold)
5. Select highest-scoring candidate
```

If the selected candidate is blocked by a post-scoring guardrail, the engine falls through to the next candidate. If all candidates are blocked, `do_nothing` is returned.

---

## Harmful Optimization Avoidance

The decision engine is explicitly designed to avoid **short-term optimization that causes long-term harm**:

| Anti-Pattern | How the Engine Avoids It |
|---|---|
| **Notification spam** | Hard cap of 5 notifications per 7 days per user. Fatigue penalty increases linearly with notification count. |
| **Discount addiction** | 30-day cooldown between discount offers. `max_discounts_30d` eligibility rule. No discount action for users with >3 discounts in the past 90 days. |
| **Over-servicing high-value users** | High-value protection guardrail limits intervention frequency. The engine does not assume that spending more on high-value users always improves retention. |
| **Exploiting new users** | New user protection prevents monetization actions during the onboarding window. The first 7 days are reserved for engagement actions only. |
| **Optimizing for a single metric** | The balanced objective with multiple weighted terms prevents extreme optimization on any single dimension. |

---

## `do_nothing` Logic

The "no action" option is a first-class candidate, not a fallback. It is always eligible and has zero cost and zero risk.

**When `do_nothing` wins:**

1. All other candidates are blocked by guardrails.
2. No candidate exceeds the minimum expected value threshold (0.02).
3. The user is already on a positive trajectory (high retention probability, recent activity, no decline signals).
4. The cost of any action outweighs its expected benefit.

**Why this matters:**

In many recommendation systems, the system is forced to recommend *something*. This creates perverse incentives — the system recommends low-value actions just to appear active. Attention OS explicitly values restraint. **Doing nothing is often the best decision**, and the system should be confident enough to say so.

---

## Example Decision Walk-Through

### User Profile

```
User ID: u_7842
Tenure: 45 days
Last session: 4 days ago
Sessions last 7d: 2 (down from 6 in prior 7d)
Purchases last 30d: 0
Retention risk: 0.72
Monetization propensity: 0.28
Lifetime value percentile: 85th
Notifications received 7d: 3
Plan: Free tier
```

### Step 1: Eligibility Filtering

| Action | Eligible? | Reason |
|---|---|---|
| `notify_reengage` | Yes | 4 days since last session (>3), 3 notifications (<5), not in trial. |
| `notify_feature_highlight` | Yes | Same base eligibility as reengage. |
| `offer_discount_10` | No | Monetization propensity 0.28 < 0.30 threshold. |
| `offer_discount_20` | No | Monetization propensity 0.28 < 0.50 threshold. |
| `offer_premium_trial` | Yes | Not premium, retention risk 0.72 > 0.4, 5+ sessions in 30d. |
| `send_case_study` | No | Not enterprise segment. |
| `schedule_checkin` | No | LTV percentile 85 < 90 threshold. |
| `do_nothing` | Yes | Always eligible. |

### Step 2: Guardrail Check

| Guardrail | Status |
|---|---|
| Notification cap (3 < 5) | OK |
| Discount cooldown | N/A (no discount candidates) |
| High-value protection | 85th percentile — low-risk actions only. All remaining candidates are low-risk. OK. |
| New user protection | 45 days > 7 days. OK. |

### Step 3: Scoring

| Action | Δ_ret | Δ_rev | Δ_ltv | Cost | Risk | Fatigue | **Score** |
|---|---|---|---|---|---|---|---|
| `notify_reengage` | +0.08 | +0.01 | +0.02 | 0.001 | 0.05 | 0.30 | **0.041** |
| `notify_feature_highlight` | +0.05 | +0.00 | +0.01 | 0.001 | 0.05 | 0.30 | **0.022** |
| `offer_premium_trial` | +0.12 | −0.02 | +0.05 | 0.15 | 0.15 | 0.00 | **0.019** |
| `do_nothing` | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | **0.000** |

### Step 4: Selection

**Winner: `notify_reengage`** with score 0.041.

The re-engagement notification is selected because:
- The user shows clear churn risk signals (declining sessions, 4 days inactive).
- A notification has minimal cost and risk.
- The estimated retention impact (+0.08) justifies the fatigue cost.
- Higher-impact actions (discounts, premium trial) are either ineligible or blocked.

### Step 5: Explanation

```json
{
  "action": "notify_reengage",
  "score": 0.041,
  "rationale": "User shows declining engagement (sessions dropped from 6 to 2 week-over-week) with 4 days of inactivity. A re-engagement notification has the highest expected retention impact at minimal cost.",
  "guardrails_applied": ["notification_cap_ok", "not_high_value_protection_triggered"],
  "alternatives_rejected": {
    "offer_discount_10": "ineligible: monetization propensity below threshold",
    "offer_premium_trial": "lower expected value after cost adjustment",
    "do_nothing": "retention risk (0.72) warrants intervention"
  }
}
```

---

## Extension Points

| Extension | How |
|---|---|
| **Add a new action** | Append to the candidate actions catalog with eligibility rules, cost, and risk metadata. No scoring engine changes needed. |
| **Adjust priorities** | Modify `scoring_weights` in the objective configuration. |
| **Add a guardrail** | Implement as a boolean check in the guardrail evaluation pipeline. Guardrails are evaluated in order and short-circuit on failure. |
| **Switch objectives** | Change `objective_config.mode` from `balanced` to `retention` or `monetization`. Weights adjust automatically. |

---

## Design Principles

1. **Transparency**: Every decision is auditable. The full scoring breakdown and guardrail evaluation log are returned with every recommendation.
2. **Restraint**: The engine is designed to say "no action" when appropriate, not to force intervention.
3. **Safety**: Guardrails are hard constraints, not soft penalties. They cannot be overridden by scoring.
4. **Configurability**: All weights, thresholds, and rules are externalized. No code changes needed to adjust behavior.
5. **Extensibility**: New actions, guardrails, and objectives can be added without modifying the core scoring logic.
