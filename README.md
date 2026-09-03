# AttentionOS — AI User Retention & Monetization Decision Engine

> ### 🔗 Live demo: **[tejas-parjane.github.io/attention-os](https://tejas-parjane.github.io/attention-os)**
> Interactive landing page with real metrics and dashboards (GitHub Pages). Full source below.

A portfolio project that turns raw behavioral data into a **single, explainable decision**: *what should the product do next for this user?*

AttentionOS demonstrates a complete, production-style ML loop:

```
Behavior → Prediction → Decision → Action → Experiment → Learning
```

Every stage is implemented, tested, and wired end-to-end — from synthetic event data, through leakage-safe feature engineering and model training, to a deterministic decision engine with guardrails, an experiment framework with statistical testing, a FastAPI backend, and a Streamlit dashboard.

---

## Why this project exists

Consumer products generate enormous amounts of behavioral data — every session, purchase, notification click, and level-up. Collecting it is easy. The hard part is **deciding what the product should do next** for each user, using everything we know, while respecting business constraints and **proving the decision worked**.

Most teams solve this with rigid rules, manual campaign logic, or one-size-fits-all segmentation. AttentionOS replaces that with a data-driven decision loop that:

1. Knows **who the user is right now** (structured user state).
2. **Predicts** what they will do next (retention risk, purchase propensity).
3. **Decides** the best action (with guardrails, not just "always send a push").
4. **Explains** the decision in plain language.
5. **Experiments** to measure whether the action actually worked.

The architecture is domain-agnostic. The same loop applies to games, social apps, subscriptions, content platforms, and ad-supported products.

---

## Architecture

```
Synthetic Behavioral Data → Data Validation → Feature Engineering
        → User State → ML Predictions (retention, monetization)
        → Decision Engine → Guardrails → Recommended Action
        → Experiment Assignment → Outcome Tracking → Analytics Dashboard
```

### The loop in code

| Stage | Module | What it does |
|-------|--------|--------------|
| **Behavior** | `src/data/generator.py` | Synthesizes event-level data (sessions, purchases, ads, notifications) across 8 user archetypes, with **irreducible stochastic noise** so behavior isn't perfectly predictable |
| **Feature** | `src/features/engine.py` | Leakage-safe rolling-window features (40 columns) — recency, frequency, intensity, trend, diversity — each computed **strictly before** the prediction point |
| **Predict** | `src/models/retention.py`, `src/models/monetization.py` | Three models per target (Logistic, Random Forest, XGBoost), trained on a **chronological** 70/15/15 split, best selected by ROC-AUC |
| **State** | `src/models/state.py` | Structured snapshot: engagement trend, retention risk, value segment, notification response |
| **Decide** | `src/decision_engine/engine.py` | Deterministic scoring of 6 candidate actions against objective alignment, cost, fatigue and risk penalties; **guardrails**; may return `NO_ACTION` |
| **Explain** | `src/llm/explainer.py` | Provider-agnostic LLM explanation with a guaranteed deterministic fallback |
| **Experiment** | `src/experiments/engine.py` | Deterministic hash-based variant assignment, simulated treatment effects, Welch's t-test analysis |
| **Serve** | `src/api/` | FastAPI (9 endpoints) + Streamlit dashboard (`dashboard/app.py`) |
| **Track** | `src/monitoring/tracker.py` | Records decisions and outcomes for audit and learning |

---

## Results (measured on this project's synthetic data)

> Models are evaluated on a **held-out chronological test set** (the final 15% of data, unseen during training). All three model families are trained per target; the best is selected on test ROC-AUC.

### Retention (will the user return within 7 days?)

| Model | ROC-AUC | PR-AUC | Precision | Recall | F1 | Brier |
|-------|:-------:|:------:|:---------:|:------:|:--:|:-----:|
| **Logistic (best)** | **0.84** | **0.66** | 0.77 | 0.79 | **0.78** | **0.16** |
| Random Forest | 0.82 | 0.73 | 0.67 | 0.76 | 0.71 | 0.18 |
| XGBoost | 0.77 | 0.61 | 0.63 | 0.76 | 0.69 | 0.23 |

### Monetization (will the user purchase within 7 days?)

| Model | ROC-AUC | PR-AUC | Precision | Recall | F1 | Brier |
|-------|:-------:|:------:|:---------:|:------:|:--:|:-----:|
| Logistic | 0.89 | 0.53 | 0.42 | 0.92 | 0.58 | 0.19 |
| **Random Forest (best)** | **0.89** | **0.55** | 0.45 | 0.75 | **0.56** | **0.12** |
| XGBoost | 0.88 | 0.51 | 0.43 | 0.83 | 0.57 | 0.17 |

These are believable production-grade numbers (not "perfect" — which is the point). Retention is inherently harder to predict than purchase, and both models show meaningful discriminative signal with sober precision/recall trade-offs.

### Decision engine output (500 users)

| Action | Share | Purpose |
|--------|------:|---------|
| `NO_ACTION` | 63.8% | "Do nothing" — confidence below threshold; prevents over-intervention |
| `DISCOUNT` | 16.8% | Price-sensitive monetization nudge for conversion-capable users |
| `PUSH_NOTIFICATION` | 10.2% | Re-engagement nudge for at-risk users |
| `PERSONALIZED_CONTENT` | 5.0% | Content re-engagement for stable-but-bored users |
| `AD_FREQUENCY_REDUCTION` | 3.8% | Protect high-value users from ad fatigue |
| `CROSS_SELL` | 0.4% | Extend value for top spenders |

The system values **restraint** — nearly two-thirds of users get *no action*, because acting confidently is better than acting often.

### Experiment results (attention optimization, `exp_2026_attn_v1`)

| Variant | Mean outcome | Lift vs control | 95% CI | p-value | Significant |
|---------|:------------:|:---------------:|:------:|:-------:|:-----------:|
| control | 0.546 | – | – | – | no |
| generic_notification | 0.579 | +0.060 | [0.558, 0.600] | 0.023 | ✅ yes |
| personalized_challenge | 0.595 | +0.090 | [0.573, 0.618] | 0.001 | ✅ yes |
| **reward** | **0.624** | **+0.143** | [0.605, 0.643] | <0.0001 | ✅ **yes** |

All three variants beat control; **`reward`** wins with the largest, cleanest lift — informing the recommendation policy going forward.

---

## Example walkthrough

Pick a real user from the data, e.g. `user_0095`:

```
Retention probability : 77%
Purchase probability  : 99%
Engagement            : improving
Value segment         : low
Recommended action    : DISCOUNT
Objective             : retention
Confidence            : 0.999
Expected value        : 46.74
```

The decision engine scored a discount higher than a push or content nudge because this user is highly purchase-likely *and* at meaningful retention risk — a discount is the highest-expected-value action after cost, fatigue, and risk penalties are applied.

Because there are 500 users, you can explore many behaviors. A declining, low-intent user instead gets a re-engagement `PUSH_NOTIFICATION` (or `NO_ACTION` if confidence is too low).

---

## Getting started

### Prerequisites

- Python 3.11+ 
- `E:\anaconda3\python.exe` (or any env with `scikit-learn`, `xgboost`, `fastapi`, `streamlit`, `pydantic`)
- SQLite works out of the box; PostgreSQL via `docker-compose` is optional

### Run the full pipeline

```bash
# 1. Generate synthetic data (500 users, ~220k events)
python scripts/generate_data.py

# 2. Build leakage-safe features + user state
python scripts/build_features.py

# 3. Train retention & monetization models, save best + metrics
python scripts/train_models.py

# 4. Score all users + run the decision engine
python scripts/generate_predictions.py
python scripts/run_decision_engine.py

# 5. Analyze the attention experiment
python scripts/run_experiment_analysis.py
```

### Serve the demo

```bash
# API
uvicorn src.api.app:app --host 127.0.0.1 --port 8000

# Dashboard (open http://localhost:8501)
streamlit run dashboard/app.py
```

Or run everything in containers:

```bash
docker-compose up --build
```

---

## API reference

All endpoints live under the same host. The dashboard calls these endpoints; it never reads models or data directly.

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Service + DB health |
| GET | `/users/{user_id}` | Profile & metadata |
| GET | `/users/{user_id}/features` | 40 computed features |
| GET | `/users/{user_id}/predictions` | Retention / purchase probabilities, predicted LTV |
| GET | `/users/{user_id}/recommendation` | Recommended action |
| POST | `/users/{user_id}/decision` | Full decision + structured user state + explanation |
| GET | `/experiments` | List of experiments |
| GET | `/experiments/{id}/results` | Per-variant stats + significance + recommendation |
| GET | `/metrics/overview` | Aggregate KPIs (users, at-risk, revenue, conversion) |

Example decision response (`POST /users/user_0095/decision`):

```json
{
  "decision": {
    "action": "DISCOUNT",
    "objective": "retention",
    "confidence": 0.999,
    "expected_value": 46.74,
    "reasons": ["Best expected value: 46.74", "Predicted impact: 0.9974", "Objective weight (retention): 1.0"],
    "guardrails": []
  },
  "user_state": {
    "engagement": "improving",
    "retention_risk": 0.23,
    "value_segment": "low",
    "notification_response": "neutral"
  },
  "explanation": "User user_0095 has a 77% retention probability ..."
}
```

Full interactive docs at `http://localhost:8000/docs`.

---

## ML methodology

### Target definitions

| Target | Definition | Positive class |
|--------|-----------|----------------|
| Retention risk | Returns within 7 days of a reference date | Returned (label = 1) |
| Purchase propensity | Purchases within 7 days of a reference date | Purchased (label = 1) |

### Leakage prevention (important)

- **Chronological split** — 70/15/15 by timestamp, never shuffled.
- **Features before label** — features are computed as of a reference date that is **strictly before** the label window. No future information is visible to a model.
- **Rolling windows** — 1/7/30-day aggregates only use past events for each window.
- **Stochastic target** — the generator injects irreducible return noise, so models learn real signal + noise rather than memorizing a deterministic rule (this is why AUC is ~0.84–0.89, not 1.0).

### Why three models

- **Logistic Regression** — interpretable, well-calibrated baseline.
- **Random Forest** — captures non-linear interactions.
- **XGBoost** — strong gradient boosting with class-weight balancing.

The best is chosen on **test ROC-AUC**, keeping both interpretability and predictive power on the table.

---

## Decision system

Predictions alone don't drive outcomes — decisions do. The decision engine:

1. **Generates candidates** — DISCOUNT, PUSH_NOTIFICATION, PERSONALIZED_CONTENT, AD_FREQUENCY_REDUCTION, CROSS_SELL.
2. **Filters by eligibility** — opt-outs, cooldowns, feature flags.
3. **Scores by objective** — expected value = `predicted_impact × objective_weight × user_value`, minus cost, **fatigue** and **risk** penalties.
4. **Applies guardrails** — hard constraints that override scoring (frequency caps, high-value-user protection, no-action threshold).
5. **Selects** the top-ranked action, or `NO_ACTION` when confidence < 0.5.

The system is **deterministic** — same user + same state ⇒ same decision, which makes it auditable and reproducible.

---

## Experimentation

Every recommendation is assigned to an experiment variant via **deterministic hashing** (SHA-256 of user ID + experiment ID) — reproducible and auditable.

Results are analyzed with **Welch's t-test** (unequal variance) per variant vs. control, reporting lift, 95% confidence intervals, and p-values at α = 0.05. The `reward` variant's significant +14.3% lift is exactly what a product team would act on.

---

## Project structure

```
attention-os/
├── src/
│   ├── config.py                  # Settings (DB URL, model, env)
│   ├── data/                      # generator, validation, database
│   ├── features/engine.py         # leakage-safe feature pipeline
│   ├── models/                    # retention, monetization, predict, state
│   ├── decision_engine/engine.py  # deterministic decision engine + guardrails
│   ├── llm/explainer.py           # explainability with fallback
│   ├── experiments/engine.py      # assignment + statistical analysis
│   ├── monitoring/tracker.py      # decision & outcome tracking
│   └── api/                       # FastAPI app, schemas, services
├── dashboard/app.py               # Streamlit dashboard (5 pages)
├── scripts/                       # One headless script per pipeline stage
├── tests/                         # 50+ unit tests (data, features, models, decisions, experiments, API)
├── docs/                          # architecture, modeling, decision_engine, experimentation, demo_script
├── data/                          # raw / processed / models (+ SQLite DB)
├── docker-compose.yml
├── Dockerfile
├── pyproject.toml
└── README.md
```

---

## Demo (for interviews / stakeholders)

The Streamlit dashboard has **5 pages**:

1. **Overview** — KPIs, engagement mix, event and revenue charts.
2. **User Explorer** — pick a user, inspect profile, features, predictions, recommendation, activity timeline.
3. **Decision Explanation** — user state, model predictions, selected action, guardrails, AI explanation, candidate scores.
4. **Experiment Dashboard** — control vs. variants with confidence intervals and significance.
5. **Model Performance** — model comparison tables, ROC/PR, calibration, feature importance.

Suggested flow: open **Overview** → drill into **User Explorer** (e.g. `user_0095`) → show the **Decision Explanation** → show the **Experiment Dashboard** (reward wins) → finish on **Model Performance** (honest AUC ~0.8–0.9).

See [`docs/demo_script.md`](docs/demo_script.md) for a full narrated 15–20 min script.

---

## Honest limitations

- **Synthetic data** — all behavior is generated. Real-world distributions and user heterogeneity are simplified; the pipeline is designed so real event streams can be substituted without changing downstream components.
- **No true causal inference from models** — predictions are correlational. Only the experiment framework gives causal estimates.
- **Simulated treatment effects** — experiment outcomes are simulated, not measured from real users.
- **Heuristic LTV** — LTV uses a simple `purchase_prob × historical_ltv` approximation, not survival/causal LTV.

---

## Technologies

| Layer | Technology |
|-------|-----------|
| Language / data | Python, Pandas, NumPy |
| ML | scikit-learn, XGBoost |
| Decision engine | Custom scoring + guardrails (deterministic) |
| Explainability | LLM (provider-agnostic) + deterministic fallback |
| API | FastAPI, Pydantic |
| Dashboard | Streamlit, Plotly |
| Data store | SQLite (dev) / PostgreSQL (prod), SQLAlchemy |
| Testing | pytest |
| Container | Docker, Docker Compose |

---

## License

MIT (portfolio demonstration)
