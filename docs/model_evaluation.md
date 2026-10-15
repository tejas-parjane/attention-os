# Model Evaluation Log — AttentionOS

How the headline numbers were produced, why the winners were chosen, and
where the models fail. Everything below is measured on this project's
**synthetic data** (500 users, 90 days, seed 42) — technical evaluation
results, not real-world business outcomes.

Reproduce everything with:

```bash
python scripts/generate_data.py
python scripts/build_features.py
python scripts/train_models.py        # writes data/models/*/metrics.csv + val_metrics.csv
python scripts/analyze_models.py      # writes data/models/evaluation_report.json
pytest tests/test_analysis.py
```

## 1. Problem and targets

| Target | Definition | Positive class | Test pos. rate |
|--------|-----------|----------------|----------------|
| Retention | ≥1 event in the 7 days after prediction date T0 | returned (1) | 45.3% (34/75) |
| Monetization | ≥1 purchase event in the 7 days after T0 | purchased (1) | 17.3% (13/75) |

Single fixed prediction date T0 = `max_ts − 14d` for all users. Features use
only events before T0; labels use only events after T0.

## 2. Dataset

Synthetic behavioral log: 235,481 events × 500 users × 90 days across
8 archetypes (`casual`, `new_user`, `high_value`, `at_risk`, `whale`,
`highly_engaged`, …), with irreducible return noise so behavior is not
perfectly predictable. Generator: `src/data/generator.py` (`seed=42`).

## 3. Split design (read this before citing "chronological")

One definition shared by training and analysis (`src/models/splits.py`,
`random_state=42`):

1. **Outer split (85/15):** 85% training pool / 15% held-out test (n=75).
   The test fold is locked away — never used for fitting, tuning, or
   selection.
2. **Inner split (~82/18 of the pool):** fit/validation. **Model selection
   happens here, on validation ROC-AUC.** Test numbers are reported
   afterward as the final unbiased estimate.

Temporal separation holds **by construction** — one shared T0, features
strictly before it, labels strictly after — so no future information can
cross folds. Earlier versions of these docs called the split
"chronological" and selected on test ROC-AUC; both were fixed (see
`git log`).

## 4. Baselines

Every model must beat trivial baselines on the same test set:

| Task | Dummy (most_frequent) ROC-AUC | Dummy (stratified) ROC-AUC |
|------|-------------------------------|----------------------------|
| Retention | 0.50 | 0.51 |
| Monetization | 0.50 | 0.46 |

Logistic Regression additionally serves as the interpretable modeling
baseline (`class_weight="balanced"`, `max_iter=1000`).

## 5. Results

Validation ROC-AUC (selection criterion):

| Task | Logistic | Random Forest | XGBoost | Selected |
|------|---------:|--------------:|--------:|----------|
| Retention | 0.8615 | 0.8508 | 0.8407 | Logistic |
| Monetization | 0.8524 | 0.8710 | 0.8759 | XGBoost |

Held-out test set (n=75) — final unbiased estimate for the selected model:

### Retention (selected: Logistic)

| Model | ROC-AUC | PR-AUC | Precision | Recall | F1 | Brier |
|-------|:-------:|:------:|:---------:|:------:|:--:|:-----:|
| **Logistic (selected)** | **0.88** | **0.86** | 0.82 | 0.79 | **0.81** | **0.14** |
| Random Forest | 0.88 | 0.86 | 0.80 | 0.82 | 0.81 | 0.14 |
| XGBoost | 0.83 | 0.80 | 0.70 | 0.76 | 0.73 | 0.21 |

### Monetization (selected: XGBoost)

| Model | ROC-AUC | PR-AUC | Precision | Recall | F1 | Brier |
|-------|:-------:|:------:|:---------:|:------:|:--:|:-----:|
| Logistic | 0.82 | 0.54 | 0.67 | 0.62 | 0.64 | 0.11 |
| Random Forest | 0.82 | 0.63 | 0.63 | 0.38 | 0.48 | 0.10 |
| **XGBoost (selected)** | **0.80** | **0.67** | 0.75 | 0.46 | **0.57** | **0.09** |

**Selection rationale:** highest validation ROC-AUC. Note the honest
consequences: on retention, Logistic and Random Forest are tied on test
(0.88) and validation picks the simpler model — a good outcome. On
monetization, the winner's validation score (0.88) exceeds its test score
(0.80); with only ~13 positives per 75-user fold, that gap is expected
sampling noise, and reporting both numbers is exactly why the test fold
exists.

## 6. Threshold analysis (0.5 is not optimal)

Precision/recall/F1 swept over thresholds 0.05–0.95
(`scripts/analyze_models.py::threshold_sweep`):

| Task | Model | F1-max threshold | F1 | Youden threshold |
|------|-------|-----------------|----|------------------|
| Retention | Logistic | 0.50 | 0.81 | 0.50 |
| Retention | Random Forest | 0.65 | 0.83 | 0.65 |
| Monetization | XGBoost | 0.35 | 0.70 | 0.15 |
| Monetization | Logistic | 0.30 | 0.67 | 0.30 |

Takeaway: for monetization, moving off the 0.5 default substantially helps
(XGBoost F1 0.57 → 0.70 at threshold 0.35). The deployed decision engine
consumes probabilities with its own confidence/EV logic rather than hard
cutoffs, which sidesteps this — but any future binary deployment must set
the operating point explicitly. Full grids: `evaluation_report.json`.

## 7. Calibration

| Task | Model | Brier | ECE (10-bin) |
|------|-------|-------|--------------|
| Retention | Logistic | 0.1373 | 0.0751 |
| Retention | Random Forest | 0.1372 | 0.1288 |
| Monetization | XGBoost | 0.0894 | 0.0873 |
| Monetization | Random Forest | 0.1039 | 0.0547 |

Probabilities are reasonably calibrated (ECE < 0.13 except XGBoost
retention at 0.19); the selected models' scores are suitable inputs for
expected-value math.

## 8. Error analysis by user segment

Worst segments for the selected models at threshold 0.5:

- **Retention (Logistic):** `at_risk` (error 0.50, n=6), `whale` (0.50, n=2), `casual` (0.33, n=6).
- **Monetization (XGBoost):** `highly_engaged` (1.00, n=2), `whale` (0.25, n=4), `at_risk` (0.20, n=5).

Small-n segments dominate the worst list — a data-collection finding, not
just a modeling one: rare archetypes need more users before their errors
are actionable. Full per-segment tables (at 0.5 and at the F1-max
threshold): `evaluation_report.json`.

## 9. Cost sketch (illustrative — not measured revenue)

With stated assumptions (intervention $2, true positive $25):

- **Retention:** cost-optimal threshold 0.05 → net benefit $730 vs $700 for treat-all and $0 for treat-none.
- **Monetization:** cost-optimal threshold 0.15 → $216 vs $175 treat-all.

This demonstrates *how* a threshold would be chosen from unit economics,
not that these dollar figures would materialize. Real deployment needs
measured costs, measured conversion values, and online validation.

## 10. Limitations

- Synthetic data: archetypes, noise, and economics are simplified; real distributions will differ.
- Single 75-user test set (and 75-user validation fold): confidence intervals on all metrics are wide; segment slices (n=2–6) are anecdotal. The monetization val/test gap (0.88 → 0.80) reflects this noise.
- Snapshot design: one T0 means no test of temporal drift; a production system needs rolling-origin validation over multiple prediction dates.
- Predictions are correlational; only the experiment framework (§experiment docs) estimates causal effects — on simulated outcomes.
