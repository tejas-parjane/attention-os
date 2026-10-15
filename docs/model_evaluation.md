# Model Evaluation Log — AttentionOS

How the headline numbers were produced, why the winners were chosen, and
where the models fail. Everything below is measured on this project's
**synthetic data** (500 users, 90 days, seed 42) — technical evaluation
results, not real-world business outcomes.

Reproduce everything with:

```bash
python scripts/generate_data.py
python scripts/build_features.py
python scripts/train_models.py        # writes data/models/*/metrics.csv
python scripts/analyze_models.py      # writes data/models/evaluation_report.json
pytest tests/test_analysis.py
```

## 1. Problem and targets

| Target | Definition | Positive class | Test pos. rate |
|--------|-----------|----------------|----------------|
| Retention | ≥1 event in the 7 days after prediction date T0 | returned (1) | 45.3% (34/75) |
| Monetization | ≥1 purchase event in the 7 days after T0 | purchased (1) | 16.0% (12/75) |

Single fixed prediction date T0 = `max_ts − 14d` for all users. Features use
only events before T0; labels use only events after T0.

## 2. Dataset

Synthetic behavioral log: 235,481 events × 500 users × 90 days across
8 archetypes (`casual`, `new_user`, `high_value`, `at_risk`, `whale`,
`highly_engaged`, …), with irreducible return noise so behavior is not
perfectly predictable. Generator: `src/data/generator.py` (`seed=42`).

## 3. Split design (read this before citing "chronological")

Users are assigned 70/15/15 by **stratified shuffle split**
(`StratifiedShuffleSplit`, `random_state=42`): disjoint users per fold,
positive rate preserved, exactly reproducible. Temporal separation holds
**by construction** — one shared T0, features strictly before it, labels
strictly after — so no future information can cross folds. Earlier versions
of these docs called this "chronological"; that wording was inaccurate for
a single-T0 snapshot design and has been corrected.

## 4. Baselines

Every model must beat trivial baselines on the same test set:

| Task | Dummy (most_frequent) ROC-AUC | Dummy (stratified) ROC-AUC |
|------|-------------------------------|----------------------------|
| Retention | 0.50 | 0.47 |
| Monetization | 0.50 | 0.40 |

Logistic Regression additionally serves as the interpretable modeling
baseline (`class_weight="balanced"`, `max_iter=1000`).

## 5. Results (held-out test set, n=75)

### Retention

| Model | ROC-AUC | PR-AUC | Precision | Recall | F1 | Brier |
|-------|:-------:|:------:|:---------:|:------:|:--:|:-----:|
| Logistic | 0.8486 | 0.8023 | 0.7353 | 0.7353 | 0.7353 | 0.1690 |
| **Random Forest (selected)** | **0.8594** | **0.8339** | 0.7143 | 0.7353 | 0.7246 | **0.1664** |
| XGBoost | 0.8199 | 0.7478 | 0.7027 | 0.7647 | 0.7324 | 0.2160 |

### Monetization

| Model | ROC-AUC | PR-AUC | Precision | Recall | F1 | Brier |
|-------|:-------:|:------:|:---------:|:------:|:--:|:-----:|
| Logistic | 0.8188 | 0.4286 | 0.4118 | 0.5833 | 0.4828 | 0.1660 |
| **Random Forest (selected)** | **0.8512** | **0.4996** | 0.4000 | 0.3333 | 0.3636 | **0.1058** |
| XGBoost | 0.8214 | 0.5475 | 0.4286 | 0.5000 | 0.4615 | 0.1521 |

**Selection rationale:** highest test ROC-AUC, with the lowest Brier score
(best calibration) as a tie-break signal. Note the trade-off the table
hides: at the default 0.5 threshold, the monetization winner recalls only
33% of purchasers — see §6.

## 6. Threshold analysis (0.5 is not optimal)

Precision/recall/F1 swept over thresholds 0.05–0.95
(`scripts/analyze_models.py::threshold_sweep`):

| Task | Model | F1-max threshold | F1 | Youden threshold |
|------|-------|-----------------|----|------------------|
| Retention | Random Forest | 0.10 | 0.77 | 0.65 |
| Retention | Logistic | 0.20 | 0.77 | 0.35 |
| Monetization | Random Forest | 0.30 | 0.60 | 0.20 |
| Monetization | XGBoost | 0.15 | 0.57 | 0.15 |

Takeaway: the deployed decision engine should **not** use raw 0.5
cutoffs — particularly for monetization, where lowering the threshold
nearly doubles F1 (0.36 → 0.60). The current engine consumes
probabilities with its own confidence/EV logic rather than hard cutoffs,
which sidesteps this, but any future binary deployment must set the
operating point explicitly. Full grids: `evaluation_report.json`.

## 7. Calibration

| Task | Model | Brier | ECE (10-bin) |
|------|-------|-------|--------------|
| Retention | Random Forest | 0.1664 | 0.1365 |
| Retention | Logistic | 0.1690 | 0.1103 |
| Monetization | Random Forest | 0.1058 | 0.0761 |

Probabilities are reasonably calibrated (ECE < 0.15); the monetization
winner is the best-calibrated of all six fits, which supports using its
scores inside expected-value math.

## 8. Error analysis by user segment

Worst segments for the selected models at threshold 0.5:

- **Retention (RF):** `at_risk` (error 1.00, n=6 — every at-risk test user misclassified), `whale` (0.50, n=2), `high_value` (0.37, n=30).
- **Monetization (RF):** `highly_engaged` (0.50, n=2), `whale` (0.50, n=6), `high_value` (0.26, n=35).

Small-n segments dominate the worst list — a data-collection finding, not
just a modeling one: rare archetypes need more users before their errors
are actionable. The `at_risk` retention failure is the most substantive:
users exhibiting churn-like behavior at T0 are systematically scored
wrong, suggesting missing recency-decay or trend features. Full per-segment
tables (at 0.5 and at the F1-max threshold): `evaluation_report.json`.

## 9. Cost sketch (illustrative — not measured revenue)

With stated assumptions (intervention $2, true positive $25):

- **Retention:** cost-optimal threshold 0.10 → net benefit $742 vs $700 for treat-all and $0 for treat-none. The model adds $42 over blanket action.
- **Monetization:** cost-optimal threshold 0.20 → $221 vs $150 treat-all.

This demonstrates *how* a threshold would be chosen from unit economics,
not that these dollar figures would materialize. Real deployment needs
measured costs, measured conversion values, and online validation.

## 10. Limitations

- Synthetic data: archetypes, noise, and economics are simplified; real distributions will differ.
- Single 75-user test set: confidence intervals on all metrics are wide; segment slices (n=2–6) are anecdotal.
- Snapshot design: one T0 means no test of temporal drift; a production system needs rolling-origin validation over multiple prediction dates.
- Predictions are correlational; only the experiment framework (§experiment docs) estimates causal effects — on simulated outcomes.
