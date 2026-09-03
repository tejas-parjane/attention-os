# Attention OS — Experimentation Framework

## Overview

The experimentation framework provides controlled, statistically rigorous evaluation of intervention strategies. Every recommendation produced by the decision engine is assigned to an experiment variant, enabling the system to measure whether interventions actually work — not just whether the model predicts they will.

This document covers assignment methodology, variant design, metrics, statistical analysis, counterfactual reasoning, and the critical distinction between correlation and causation.

---

## Deterministic Assignment

### Why Deterministic?

Non-deterministic (random-at-serve-time) assignment creates problems:
- **Reproducibility**: The same user might be assigned to different variants across requests, making results uninterpretable.
- **Auditability**: You cannot verify that a user was assigned correctly without logs.
- **Consistency**: The dashboard, API, and analysis pipeline must agree on assignment — deterministic hashing guarantees this.

### Implementation: Stable Hashing

Assignment is computed as:

```python
def assign_variant(user_id: str, experiment_id: str, variants: list[str]) -> str:
    hash_input = f"{user_id}:{experiment_id}"
    hash_value = int(hashlib.sha256(hash_input.encode()).hexdigest(), 16)
    variant_index = hash_value % len(variants)
    return variants[variant_index]
```

**Properties:**
- The same `(user_id, experiment_id)` pair always produces the same variant.
- The assignment is distributed approximately uniformly across variants (within ±2% for large user bases).
- No random number generator is involved — the assignment is fully reproducible.
- Changing the variant list for an experiment does **not** reassign existing users (because the hash is over the user-experiment pair, not the variant list). New variants are only assigned to new users.

### Assignment Table

| Experiment ID | Variants | Assignment |
|---|---|---|
| `exp_reengage_v1` | `["control", "notify_reengage", "notify_feature_highlight"]` | Users hashed into one of three variants. |
| `exp_discount_v1` | `["control", "discount_10", "discount_20"]` | Users hashed into one of three variants. |
| `exp_premium_v1` | `["control", "premium_trial"]` | Simple A/B — control or treatment. |

---

## Control vs. Variants

### Control Group

The control group receives **business-as-usual** treatment — typically no intervention or the existing intervention strategy. The control is essential because without it, you cannot distinguish the effect of your intervention from natural user behavior.

| Variant | Treatment |
|---|---|
| `control` | No intervention (or existing default behavior). |
| `treatment_A` | New intervention strategy A. |
| `treatment_B` | New intervention strategy B. |

### Variant Design Principles

1. **One variable per experiment**: Each experiment should test one intervention change. Testing multiple changes simultaneously makes it impossible to attribute effects.
2. **Sufficient variant size**: Each variant needs enough users to achieve statistical power. The framework includes a minimum sample size calculator.
3. **Mutually exclusive variants**: No user can be in more than one variant of the same experiment. The hashing mechanism guarantees this.
4. **Consistent duration**: All variants run for the same time period to avoid temporal confounds.

---

## Metrics

### Primary Metrics

These metrics directly measure whether the intervention achieves its business objective.

| Metric | Definition | Why It Matters |
|---|---|---|
| **Retention Rate** | Fraction of users in the variant who are active on Day 7. | Directly measures the retention impact of the intervention. This is the headline metric for retention-focused experiments. |
| **Revenue Per User (RPU)** | Total revenue in the variant divided by the number of users in the variant. | Measures the monetization impact. Important because an intervention that improves retention but destroys revenue is not a success. |
| **Conversion Rate** | Fraction of users in the variant who make a purchase within 7 days. | Granular monetization signal — separate from revenue because it captures intent, not just spend magnitude. |

### Guardrail Metrics

These metrics ensure that the intervention does not cause unintended harm. An experiment that improves retention but degrades these metrics is a failure.

| Metric | Definition | Why It Matters |
|---|---|---|
| **Churn Rate** | Fraction of users who do not have any session in the last 7 days. | Inverse of retention — included explicitly because stakeholders sometimes focus on churn rate rather than retention rate, and they can diverge if the user population changes. |
| **Notification Fatigue Rate** | Fraction of users who received > 5 notifications in 7 days and had zero opens. | Detects whether the intervention is over-notifying users to the point of disengagement. |
| **Discount Redemption Rate** | Fraction of discount offers that were redeemed. | A low redemption rate suggests the offer was poorly targeted — money spent with no return. |
| **Unsubscribe Rate** | Fraction of users who disabled notifications during the experiment period. | A leading indicator of long-term harm — users who unsubscribe are harder to re-engage later. |

### Why Guardrail Metrics Are Non-Negotiable

A common failure mode in experimentation is **local optimization**: the experiment shows a positive result on the primary metric, but the intervention causes side effects that are worse than the benefit. Guardrail metrics catch this.

**Example**: A push notification campaign increases 7-day retention by 3%, but the unsubscribe rate doubles. The short-term retention gain is offset by a long-term communication channel loss. Without guardrail metrics, this experiment would be declared a success.

---

## Statistical Analysis

### Two-Sample t-Test

The primary statistical test compares the mean of each metric between the control and treatment variants.

```python
from scipy import stats

def analyze_experiment(control_values, treatment_values, alpha=0.05):
    t_stat, p_value = stats.ttest_ind(treatment_values, control_values)
    ci = stats.ttest_ind(treatment_values, control_values).confidence_interval()
    
    return {
        "t_statistic": t_stat,
        "p_value": p_value,
        "significant": p_value < alpha,
        "control_mean": np.mean(control_values),
        "treatment_mean": np.mean(treatment_values),
        "effect_size": np.mean(treatment_values) - np.mean(control_values),
        "confidence_interval": ci,
    }
```

### Confidence Intervals

| Concept | Explanation |
|---|---|
| **What it is** | A range of values within which the true treatment effect is likely to fall, with a specified probability (typically 95%). |
| **How to read it** | If the 95% CI for the effect size is `[0.01, 0.05]`, we are 95% confident that the true effect is between +1% and +5%. If the CI includes 0, the effect is not statistically significant. |
| **Why it matters** | A p-value alone tells you whether the result is "significant." A confidence interval tells you **how big the effect is** and how uncertain you are about it. A result can be statistically significant but practically meaningless (e.g., a 0.1% improvement with p < 0.01). |

### Significance Threshold

| Parameter | Value | Rationale |
|---|---|---|
| **Alpha** | 0.05 | Standard threshold — 5% chance of a false positive. |
| **Power** | 0.80 | 80% chance of detecting a true effect of the minimum detectable size. |
| **Minimum detectable effect** | Configured per experiment | Typically set to the smallest effect that would be practically meaningful (e.g., 2% absolute change in retention rate). |

### Interpreting Results

| Outcome | Interpretation | Action |
|---|---|---|
| p < 0.05, effect > 0, guardrails OK | **Positive result** | Roll out the treatment variant. |
| p < 0.05, effect > 0, guardrail violated | **Mixed result** | Investigate the guardrail violation. Do not roll out without addressing the harm. |
| p < 0.05, effect < 0 | **Negative result** | The treatment is harmful. Do not roll out. Document why it failed. |
| p ≥ 0.05 | **Inconclusive** | The experiment did not detect a meaningful effect. This does not mean there is no effect — it may mean the sample size was too small or the effect is smaller than the minimum detectable size. |

---

## Known Treatment Effects Simulation

### The Problem

In a real A/B test, you observe the outcome for each user under exactly one treatment. You never observe what *would have happened* to a treatment user if they had been in the control group. This is the **fundamental problem of causal inference**.

### Simulation Approach

The experimentation framework includes a simulation mode that uses the trained models as a proxy for the true response function:

```python
def simulate_treatment_effect(user_state, action, model):
    """
    Estimate what would happen if we applied `action` to this user,
    using the model as a proxy for the true response function.
    """
    baseline_score = model.predict(user_state)
    treated_state = apply_action_features(user_state, action)
    treated_score = model.predict(treated_state)
    
    return treated_score - baseline_score
```

**What this gives us:**
- An estimate of the **individual treatment effect** for each user.
- A way to compare strategies without running a live experiment.
- A sanity check — if the simulated effect is much larger than what the A/B test shows, the model may be overestimating the intervention's impact.

**What this does NOT give us:**
- **Causal truth**. The model is trained on observational data. It can predict what *typically happens* to users who receive an intervention, but it cannot guarantee that the intervention *caused* the outcome. Users who receive notifications may be systematically different from those who don't.

### Counterfactual Thinking in Practice

```
Observed: User received notification → User stayed active
Model estimate: User would have stayed active anyway (base probability 0.85)
Conclusion: The notification likely had minimal causal effect.
```

```
Observed: User received notification → User churned
Model estimate: User would have churned regardless (base probability 0.78)
Conclusion: The notification did not cause the churn — the user was already at risk.
```

The framework explicitly reports both the **observed outcome** and the **counterfactual estimate** to prevent misattribution.

---

## Correlation vs. Causation

This is the most important conceptual point in the experimentation framework.

### The Trap

Consider this observation from historical data:

> "Users who received re-engagement notifications had a 15% higher retention rate than users who did not."

**This does not mean notifications cause retention.** It could mean:
- Users who were already more engaged were more likely to open notifications (and more likely to stay active).
- The decision engine sends notifications to users it *already thinks* will retain — selection bias.
- Some third factor (e.g., product quality perception) drives both notification engagement and retention.

### How the Framework Addresses This

| Technique | Purpose |
|---|---|
| **Randomized assignment** | The A/B test randomly (via hashing) assigns users to treatment and control, breaking the link between user characteristics and treatment assignment. If the experiment is well-designed, the only systematic difference between variants is the treatment itself. |
| **Guardrail metrics** | Detects when an intervention is correlated with a metric improvement but actually causing harm elsewhere. |
| **Counterfactual simulation** | Provides an estimate of what would have happened without the intervention, helping distinguish "the intervention helped" from "the user would have been fine anyway." |
| **Explicit caveats** | The experiment dashboard displays a disclaimer: *"Observed effects measure the difference between variants, not the causal impact of the intervention. Causal claims require additional assumptions about the experiment design."* |

### What We Can Claim

| Scenario | Valid Claim |
|---|---|
| **Randomized A/B test** | "Users randomly assigned to variant A had a X% higher retention rate than those assigned to control (p < 0.05). We attribute this difference to the intervention, assuming no confounding factors affected assignment." |
| **Observational analysis** | "Users who received notifications had a Y% higher retention rate. This is a correlational finding and does not establish causation." |
| **Model-based simulation** | "The model estimates a Z% improvement in retention if the intervention is applied. This estimate is based on historical patterns and may not reflect the true causal effect." |

### Why This Matters for a Portfolio Project

Demonstrating awareness of the correlation-causation distinction is a signal of **maturity in ML thinking**. Many production ML systems make causal claims from observational data without proper caveats. This framework explicitly separates:
- What we **observed** (correlation in historical data)
- What we **estimate** (model-based treatment effects)
- What we **conclude** (statistically validated causal claims from experiments)

---

## Experiment Dashboard Metrics Summary

The experiment dashboard displays the following for each active experiment:

```
┌─────────────────────────────────────────────────────┐
│ Experiment: exp_reengage_v1                         │
│ Status: Active | Duration: 14 days | Days elapsed: 7│
│                                                     │
│ Variant        | N     | Retention | RPU    | Churn │
│ ───────────────|───────|───────────|────────|───────│
│ control        | 1,204 | 72.3%     | $4.21  | 27.7% │
│ notify_reengage| 1,198 | 75.8%     | $4.35  | 24.2% │
│ notify_feature | 1,201 | 74.1%     | $4.28  | 25.9% │
│                                                     │
│ Effect (reengage vs control):                       │
│   Retention: +3.5pp (95% CI: [+1.2pp, +5.8pp])     │
│   p-value: 0.003                                    │
│   Significant: Yes ✓                                │
│   Guardrails: All OK ✓                              │
│                                                     │
│ Effect (feature vs control):                        │
│   Retention: +1.8pp (95% CI: [-0.5pp, +4.1pp])     │
│   p-value: 0.12                                     │
│   Significant: No ✗                                 │
│   Guardrails: All OK ✓                              │
└─────────────────────────────────────────────────────┘
```

---

## Experiment Lifecycle

```
1. Define hypothesis: "Re-engagement notifications improve 7-day retention"
2. Design experiment: 3 variants, primary metric = retention rate
3. Calculate minimum sample size: ~1,000 per variant for 3pp effect at α=0.05, power=0.80
4. Launch: Assign users to variants via deterministic hashing
5. Run: Collect metrics for 14 days (2 full 7-day windows)
6. Analyze: Run t-tests, check guardrails, review confidence intervals
7. Decide: Roll out winner, iterate, or stop
8. Document: Record results, learnings, and next steps
```

---

## Limitations

| Limitation | Mitigation |
|---|---|
| **Novelty effects** | Users may respond differently to a new intervention initially than they do long-term. Run experiments for at least 2 weeks. |
| **Sample ratio mismatch** | If variant sizes diverge significantly (>5%), investigate for assignment bugs. |
| **Multiple testing** | Running many experiments increases false positive risk. Apply Bonferroni correction or false discovery rate control when needed. |
| **Network effects** | If users interact with each other, treatment effects can spill over between variants. The framework assumes user-level independence. |
| **Long-term effects** | 7-day experiments may miss longer-term effects. Supplement with cohort analysis for high-stakes decisions. |
