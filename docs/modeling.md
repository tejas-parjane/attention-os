# Attention OS — Modeling Methodology

## Overview

The ML layer in Attention OS serves two prediction tasks that directly inform downstream decisions: **retention risk** and **monetization propensity**. This document covers target definitions, feature engineering, temporal validation, model selection, evaluation, calibration, and known limitations.

---

## Target Definitions

Both targets use a **7-day forward-looking window**, which balances signal strength with operational relevance — 7 days is long enough to capture meaningful behavior but short enough to act on.

### Retention Target

| Field | Value |
|---|---|
| **Name** | `will_not_churn_7d` |
| **Definition** | 1 if the user has at least one session in the next 7 days; 0 otherwise. |
| **Positive class** | Retained (did not churn) |
| **Rationale** | We want to intervene on users who are *about* to churn, so we predict the positive outcome (retention) and intervene when it is low. |

### Monetization Target

| Field | Value |
|---|---|
| **Name** | `will_purchase_7d` |
| **Definition** | 1 if the user completes at least one purchase in the next 7 days; 0 otherwise. |
| **Positive class** | Converted (made a purchase) |
| **Rationale** | Identifies users with latent purchase intent who can be nudged with targeted offers. |

---

## Feature Engineering

All features are computed using **rolling windows** with strict temporal boundaries. No feature at time *t* may reference any data from *t + 1* onward.

### Feature Catalog

#### Recency Features

| Feature | Window | Rationale |
|---|---|---|
| `days_since_last_session` | Current time − last session timestamp | Single strongest churn predictor. A user who was active yesterday is fundamentally different from one active 30 days ago. |
| `days_since_last_purchase` | Current time − last purchase timestamp | Recency of spending is a strong conversion signal; recent buyers are more likely to buy again. |
| `days_since_signup` | Current time − registration date | Tenure correlates with both retention (longer-tenured users churn less) and monetization (newer users are in the onboarding-to-conversion funnel). |

#### Frequency Features

| Feature | Window | Rationale |
|---|---|---|
| `sessions_last_7d` | 7-day lookback | Short-term activity level — the most immediate signal of current engagement. |
| `sessions_last_30d` | 30-day lookback | Medium-term activity level — smooths out weekly variance. |
| `purchases_last_30d` | 30-day lookback | Spending frequency indicates purchase habit formation. |
| `notifications_received_7d` | 7-day lookback | Notification load affects fatigue; high values may indicate diminishing returns. |
| `notifications_opened_7d` | 7-day lookback | Open rate as a fraction of received is an engagement quality signal. |

#### Intensity Features

| Feature | Window | Rationale |
|---|---|---|
| `avg_session_duration_7d` | 7-day lookback | Depth of engagement per session — a user with 10 short sessions behaves differently from one with 3 long sessions. |
| `total_screen_time_7d` | 7-day lookback | Aggregate time investment; high screen time suggests strong product habit. |
| `avg_actions_per_session_7d` | 7-day lookback | Interaction density — how much the user does per visit. |

#### Trend Features

| Feature | Window | Rationale |
|---|---|---|
| `session_trend_7d_vs_30d` | Ratio of 7d rate to 30d rate | Directional momentum. A ratio < 1 means the user is slowing down — an early churn signal that recency alone misses. |
| `engagement_slope_7d` | Linear slope of daily sessions over 7 days | Formalizes the trend into a continuous value. Negative slope = declining engagement. |
| `purchase_frequency_trend` | 30d purchase rate vs. lifetime rate | Accelerating or decelerating purchase behavior — informs monetization propensity. |

#### Diversity Features

| Feature | Window | Rationale |
|---|---|---|
| `distinct_features_used_7d` | 7-day lookback | Breadth of product exploration. Users who try more features are less likely to churn. |
| `days_active_last_14d` | 14-day lookback | Consistency of engagement — 7 out of 14 active days is very different from 2 out of 14. |
| `feature_breadth_score` | Computed over available features | Normalized count of distinct feature categories touched; a proxy for product stickiness. |

---

## Temporal Split Strategy

### Why Not Random Split?

A random train/test split would place sessions from the same user (and even the same day) across both sets. This creates **data leakage** — the model learns patterns that would not be available at inference time.

Specifically, random splits allow:
- **Same-user leakage**: The model sees user *i*'s future behavior in training and memorizes it, leading to artificially high validation metrics.
- **Temporal leakage**: Features computed from data after the prediction point contaminate the training set.

### How We Split

```
Training set:   users/sessions from Day 1 → Day 60
Validation set: users/sessions from Day 61 → Day 75
```

- The split is **hard temporal** — no overlap in time between training and validation.
- Users who appear in both periods have their earlier sessions in training and later sessions in validation. This is correct and desirable — it mimics the real deployment scenario where we train on historical data and predict on future users.
- **No user is fully excluded from training just because they appear in validation** — their historical sessions still contribute to feature computation and label generation for training users.

### Implementation

```python
cutoff_date = training_end_date  # e.g., Day 60
train_mask = df["event_date"] <= cutoff_date
val_mask = df["event_date"] > cutoff_date

X_train, y_train = df.loc[train_mask, features], df.loc[train_mask, target]
X_val, y_val = df.loc[val_mask, features], df.loc[val_mask, target]
```

---

## Leakage Prevention Techniques

| Technique | Description |
|---|---|
| **Forward-fill prevention** | Features are computed *only* from historical data. The feature engineering pipeline enforces a strict cutoff — no future data is accessible during computation. |
| **Label timing** | Labels are assigned based on events occurring *after* the feature computation timestamp, ensuring the model cannot "see" the label at prediction time. |
| **No target leakage in features** | Features like `purchases_last_30d` use a window that ends at the prediction point, not at the end of the dataset. |
| **Temporal ordering** | Data is sorted by timestamp before splitting. No shuffling. |
| **Audit checks** | The pipeline includes validation assertions that no feature value correlates suspiciously with the target beyond expected levels (>0.9 correlation triggers a warning). |

---

## Model Comparison

Three models are trained and compared on the validation set:

### Logistic Regression (Baseline)

| Aspect | Detail |
|---|---|
| **Role** | Interpretable baseline — establishes the floor for model performance. |
| **Strengths** | Coefficients are directly interpretable. Training is fast and deterministic. No hyperparameter tuning needed. |
| **Weaknesses** | Assumes linear decision boundary. Cannot capture feature interactions without manual feature engineering. |
| **When it wins** | When the relationship between features and targets is approximately linear (often the case for recency features). |

### Random Forest

| Aspect | Detail |
|---|---|
| **Role** | Non-linear baseline — captures interactions without explicit feature engineering. |
| **Strengths** | Handles non-linear relationships. Robust to outliers. Low risk of overfitting with reasonable hyperparameters. |
| **Weaknesses** | Less calibrated than boosting methods. Feature importance can be misleading with correlated features. Slower inference than a single tree. |
| **When it wins** | When feature interactions are important but the dataset is small enough that boosting overfits. |

### XGBoost (Primary)

| Aspect | Detail |
|---|---|
| **Role** | Production model — selected for the best balance of performance, calibration, and inference speed. |
| **Strengths** | Excellent predictive performance. Built-in regularization reduces overfitting. Handles missing values natively. Fast inference. |
| **Weaknesses** | Less interpretable than logistic regression (though SHAP values can help). More hyperparameters to tune. |
| **Selection criteria** | Selected based on: (1) highest AUC on temporal validation, (2) best calibration (Brier score), (3) reasonable feature importance stability across folds. |

### Hyperparameters (XGBoost)

```python
xgb_params = {
    "objective": "binary:logistic",
    "eval_metric": "auc",
    "max_depth": 4,
    "learning_rate": 0.1,
    "n_estimators": 200,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 5,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "scale_pos_weight": "auto",  # adjusted for class imbalance
}
```

The conservative `max_depth=4` and regularization terms (`reg_alpha`, `reg_lambda`) prevent overfitting on the synthetic data while still capturing meaningful non-linearities.

---

## Evaluation Metrics

### Primary Metrics

| Metric | Formula | Why It Matters |
|---|---|---|
| **AUC-ROC** | Area under the ROC curve | Threshold-independent ranking quality. Measures how well the model separates positive and negative classes regardless of the decision threshold. Robust to class imbalance. |
| **Brier Score** | Mean squared error of probability predictions | Measures **calibration** — how close predicted probabilities are to actual observed frequencies. Critical for the decision engine, which multiplies probabilities by dollar values. A model with high AUC but poor calibration will produce unreliable expected value scores. |
| **Log Loss** | Cross-entropy of predictions | Penalizes confident wrong predictions more heavily than AUC. Sensitive to calibration in a different way than Brier — it amplifies the cost of overconfident errors. |

### Secondary Metrics

| Metric | Formula | Why It Matters |
|---|---|---|
| **Precision @ top-K** | Fraction of top-K predictions that are true positives | Directly relevant to the decision engine: if we intervene on the top-K highest-risk users, how many actually needed intervention? |
| **Recall @ top-K** | Fraction of true positives captured in top-K | Measures coverage — what fraction of at-risk users are we catching? |
| **F1 Score** | Harmonic mean of precision and recall | Balances precision and recall into a single number for model comparison. |
| **Calibration Plot** | Predicted probability vs. observed frequency bin | Visual diagnostic — perfect calibration is a diagonal line. Deviations reveal where the model is over- or under-confident. |

### Why Brier Score Matters More Than You Might Think

In a typical classification task, AUC is often sufficient. But in Attention OS, the model output is not just a label — it is a **probability that gets multiplied by a dollar value** to compute expected action value:

```
expected_value = p_conversion × revenue_per_conversion − cost_of_action
```

If the model says `p = 0.8` but the true rate is `0.4`, the decision engine will over-allocate resources to that user. **Calibration is not optional — it is a business requirement.**

---

## Calibration

### What Calibration Tells Us

A perfectly calibrated model produces predictions where, among all users predicted with probability *p*, approximately *p* fraction actually exhibit the positive outcome.

```
User predicted p=0.7 → 70% of such users should actually convert
User predicted p=0.2 → 20% of such users should actually convert
```

### Calibration Assessment

| Method | Description |
|---|---|
| **Brier Score** | Lower is better. Decomposes into reliability, resolution, and uncertainty. |
| **Calibration curve** | Plot predicted probability bins against observed positive rate. Ideal: diagonal line y = x. |
| **Expected Calibration Error (ECE)** | Weighted average of bin-level calibration errors. A single number summary of calibration quality. |

### Post-Hoc Calibration

If the primary model is well-calibrated (as XGBoost typically is with sufficient data), no post-hoc calibration is needed. If calibration degrades, **Platt scaling** (logistic regression on model outputs) or **isotonic regression** can be applied as a calibration layer between the model and the decision engine.

---

## Model Limitations

### Synthetic Data Constraints

| Limitation | Impact | Mitigation |
|---|---|---|
| **Synthetic distributions may not match real data** | Models may learn patterns that do not exist in production. | The pipeline is designed with clear interfaces so real data can be substituted. Feature engineering and validation logic transfer directly. |
| **No cold-start problem represented** | All users in the synthetic data have history. Real deployments face new users with no features. | The system flags users with insufficient history and falls back to segment-level defaults. |
| **No concept drift** | Synthetic data has stationary distributions. Real user behavior shifts over time. | The temporal split partially addresses this. A production deployment would add drift detection and periodic retraining. |

### Algorithmic Limitations

| Limitation | Impact | Mitigation |
|---|---|---|
| **No sequential modeling** | Features are aggregated over windows, losing temporal granularity within the window. | A future enhancement could add LSTM/Transformer-based sequence models as an additional signal. |
| **No causal inference** | Models predict correlation, not causation. "If we send a notification, will the user stay?" is not the same as "Users who receive notifications tend to stay." | The experimentation layer and counterfactual simulation partially address this by estimating treatment effects. |
| **Class imbalance** | Churned users may be a minority class, leading to biased predictions. | `scale_pos_weight` is set automatically. Evaluation metrics (AUC, F1) are chosen to be robust to imbalance. |
| **Feature availability at inference** | Some features require historical data that may not be available for new users. | The feature pipeline includes fallback values for missing features. The model handles NaN natively. |

### Scope Limitations

- The models do not account for external factors (competitor actions, seasonality, marketing campaigns outside the system).
- The 7-day horizon is a design choice; different business contexts may require different horizons.
- The models are trained jointly but do not share information — a multi-task model could potentially improve both targets.

---

## Summary

The modeling approach prioritizes **calibration over raw accuracy**, **temporal validity over convenience**, and **interpretability over complexity where possible**. The three-model comparison ensures that the production model is not just the best performer but also the most reliable for downstream decision-making.
