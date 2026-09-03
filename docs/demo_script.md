# AttentionOS — Demo Script

Narrated walkthrough for presenting AttentionOS to an interviewer, HR, or stakeholder. Approximately 15 minutes.

**Before you start:**
- API running: `uvicorn src.api.app:app --host 127.0.0.1 --port 8000`
- Dashboard running: `streamlit run dashboard/app.py`
- Browser open to `http://localhost:8501`

---

## Act 1 — The problem (2 min)

> "AttentionOS converts raw behavioral data into a single decision: **what should the product do next for this user?** Most teams use gut instinct or rigid rules for things like notifications and discounts. This system makes the decision data-driven: it predicts behavior, scores candidate actions against business objectives and guardrails, explains its reasoning, and *experiments* to prove the action worked."

**Key point:** this is about **decision quality**, not just model accuracy.

---

## Act 2 — Overview (2 min)

Open the **Overview** page.

> "The dashboard is just a visualization layer. It calls the FastAPI backend — it never touches models or data directly. The API is the system of record."

Point at the KPIs (500 users, at-risk count, revenue, conversion) and the engagement mix.

---

## Act 3 — User Explorer (3 min)

Go to **User Explorer**, pick a user, e.g. `user_0095`.

> "This user is scored on ~40 leakage-safe features — recency, frequency, intensity, trend, diversity. Every feature is computed only from events **before** the prediction point. Nothing future leaks in."

Show the **Predictions**: retention probability and purchase probability.

> "The retention model says this user is 77% likely to return — meaningful risk. The monetization model sees ~99% purchase likelihood. High buying intent plus retention risk is exactly the profile for a discount."

Show the **Recommended Action** (`DISCOUNT`) and the activity timeline.

---

## Act 4 — Decision explanation (3 min)

Go to **Decision Explanation** for the same user.

Walk through:
1. **Current user state** — engagement trend, value segment, notification response.
2. **Model predictions** — the two probabilities.
3. **Selected action** — `DISCOUNT`, objective `retention`, confidence, expected value.
4. **Reasoning / guardrails** — cost, fatigue, risk penalties; any caps applied.
5. **AI explanation** — plain-language rationale (with a deterministic fallback if no LLM).

> "The engine scores every eligible action as an expected value, then applies guardrails as hard constraints. It's **deterministic** — the same user and state always produce the same decision, which makes it auditable."

Pick a second user with declining engagement to show a `PUSH_NOTIFICATION` and the **"do nothing"** principle:

> "Notice the system returns `NO_ACTION` for the majority of users. It values restraint — acting confidently is better than acting often. Over-intervention causes notification fatigue."

---

## Act 5 — Experiment dashboard (3 min)

Go to **Experiment Dashboard**.

> "Every recommendation is assigned to an experiment via deterministic hashing — the same user always gets the same variant. That's reproducible and auditable."

Show the **control vs. variants** table:

> "We ran an attention-optimization experiment with 500 users. All three variants beat control. The `reward` variant improved the outcome by +0.143 with a p-value < 0.0001 — statistically very significant. The confidence interval [0.605, 0.643] doesn't include control, so we're confident it's a real lift. This is what we'd roll out."

Mention `personalized_challenge` (+0.090, p=0.001) also beat control; `generic_notification` barely did (+0.060, p=0.023).

---

## Act 6 — Model performance (2 min)

Go to **Model Performance**.

> "We train three families per target — Logistic, Random Forest, XGBoost — on a chronological 70/15/15 split, and pick the best on a held-out test set. No shuffling, so models are evaluated on their ability to predict the *future*."

> "Retention: ROC-AUC ~0.84. Monetization: ~0.89. These are believable production numbers — **not** 1.0. That's deliberate: real churn has irreducible stochasticity, and our generator includes return-noise so the models learn real signal plus noise rather than memorizing the data."

Show the feature-importance chart and the model comparison table.

---

## Act 7 — Architecture & technical depth (2 min)

> "The loop is **Behavior → Prediction → Decision → Action → Experiment → Learning**:
> - Clean interfaces — data, features, models, decisions, API, dashboard are independent and testable.
> - Leakage prevention — chronological splits, features before labels, strict windows.
> - Guardrails as hard constraints, not soft penalties.
> - Experimentation built in — every decision is assigned to a variant.
> - Deterministic, auditable decisions.
> - ~50 unit tests covering data, features, models, decisions, experiments, and the API."

---

## Closing

> "This demonstrates end-to-end ML systems thinking: I can build not just models, but systems that **make decisions with models, validate those decisions, and explain them** — structured for production."

---

## Backup questions

**"Why synthetic data?"**
> "To exercise every component without PII concerns. The generator embeds causal signals so models learn meaningful patterns, and it adds irreducible noise so metrics stay realistic. Real event streams can be substituted without changing downstream code."

**"Why not deep learning?"**
> "For tabular features at this size, gradient-boosting (XGBoost) and random forests give strong, well-calibrated, fast predictions with feature importance. Deep learning is a possible enhancement for sequential data but isn't the right baseline here."

**"How would this run in production?"**
> "Swap the generated events for a real event stream (e.g. Kafka), move storage to a columnar warehouse, schedule training with drift detection, and deploy the API behind a load balancer. The dashboard already talks to the API, so it wouldn't change."

**"Biggest limitation?"**
> "Models predict correlation, not causation. The experiment layer gives causal estimates, but treatment effects here are simulated. A production system would add proper A/B infrastructure and causal-inference methods."
