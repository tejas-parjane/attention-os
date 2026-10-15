"""Post-training analysis: baselines, thresholds, calibration, segments, cost.

Reads the trained models and the held-out stratified test set (reproduced
with the same seed as ``scripts/train_models.py``) and answers the questions
a reviewer asks after seeing ROC-AUC:

1. Do the models beat trivial baselines (majority-class, stratified dummy)?
2. Is 0.5 the right operating threshold? (precision/recall sweep, F1-max,
   Youden's J, cost-optimal)
3. Are the probabilities calibrated? (Brier score, expected calibration error)
4. Where does the best model fail? (per-archetype error analysis)
5. Is acting on the model worth it? (illustrative cost/benefit sketch with
   stated assumptions — NOT measured revenue)

Run after ``scripts/train_models.py``::

    python scripts/analyze_models.py

Writes ``data/models/evaluation_report.json`` and prints a summary.
All results are on this project's synthetic data.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedShuffleSplit

from src.models.monetization import build_monetization_dataset
from src.models.retention import build_retention_dataset
from src.models.splits import inner_split, outer_split

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s - %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
SEED = 42

# --- Illustrative cost assumptions (documented, not measured revenue) --------
# A push/discount nudge costs ~$2 to send; retaining (or converting) one user
# is worth ~$25 in this synthetic economy. These numbers exist so the
# threshold discussion has business units — they are NOT observed P&L.
COST_PER_INTERVENTION = 2.0
VALUE_PER_TRUE_POSITIVE = 25.0


def reproduce_test_split(
    X: pd.DataFrame, y: pd.Series, seed: int = SEED
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (pool_fit_idx, pool_val_idx, test_idx) identical to training.

    Mirrors ``scripts/train_models.py`` + the inner split in the train
    functions: outer 85/15 via :func:`outer_split`, then :func:`inner_split`
    on the pool. All indices are positional into ``X``/``y`` as passed.
    """
    pool_idx, test_idx = outer_split(X, y, seed=seed)
    rel_fit, rel_val = inner_split(X.iloc[pool_idx], y.iloc[pool_idx], seed=seed)
    return pool_idx[rel_fit], pool_idx[rel_val], test_idx


def dummy_baselines(
    X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame, y_test: pd.Series
) -> dict[str, dict[str, float | None]]:
    """Trivial baselines every real model must beat."""
    out: dict[str, dict[str, float | None]] = {}
    for strategy in ("most_frequent", "stratified"):
        dummy = DummyClassifier(strategy=strategy, random_state=SEED)
        dummy.fit(X_train, y_train)
        try:
            proba = dummy.predict_proba(X_test)[:, 1]
            auc: float | None = round(float(roc_auc_score(y_test, proba)), 4)
            ap: float | None = round(float(average_precision_score(y_test, proba)), 4)
        except Exception:  # constant scores (e.g. all-one-class prediction)
            auc, ap = None, None
        out[strategy] = {"roc_auc": auc, "pr_auc": ap}
    return out


def threshold_sweep(y_true: pd.Series, proba: np.ndarray) -> dict:
    """Precision/recall/F1 over a threshold grid + selection rules."""
    y = np.asarray(y_true).astype(int)
    grid = [round(t, 2) for t in np.arange(0.05, 1.0, 0.05)]
    rows = []
    for t in grid:
        pred = (proba >= t).astype(int)
        rows.append(
            {
                "threshold": t,
                "precision": round(float(precision_score(y, pred, zero_division=0)), 4),
                "recall": round(float(recall_score(y, pred, zero_division=0)), 4),
                "f1": round(float(f1_score(y, pred, zero_division=0)), 4),
            }
        )
    for r in rows:
        r["youden_j"] = round(r["recall"] + _specificity(y, proba, r["threshold"]) - 1, 4)
    f1_best = max(rows, key=lambda r: r["f1"])
    youden_best = max(rows, key=lambda r: r["youden_j"])
    return {
        "grid": rows,
        "f1_max": {"threshold": f1_best["threshold"], "f1": f1_best["f1"]},
        "youden": {"threshold": youden_best["threshold"], "j": youden_best["youden_j"]},
    }


def _specificity(y: np.ndarray, proba: np.ndarray, threshold: float) -> float:
    pred = (proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return float(tn / (tn + fp)) if (tn + fp) else 0.0


def calibration_summary(y_true: pd.Series, proba: np.ndarray, n_bins: int = 10) -> dict:
    """Brier score + expected calibration error (ECE), manual uniform bins."""
    y = np.asarray(y_true).astype(int)
    p = np.asarray(proba, dtype=float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    mean_pred, frac_pos, ece = [], [], 0.0
    for b in range(n_bins):
        lo, hi = edges[b], edges[b + 1]
        mask = (p > lo) & (p <= hi) if b > 0 else (p >= lo) & (p <= hi)
        if mask.sum() == 0:
            mean_pred.append(round((lo + hi) / 2, 4))
            frac_pos.append(None)
            continue
        mean_pred.append(round(float(p[mask].mean()), 4))
        frac_pos.append(round(float(y[mask].mean()), 4))
        ece += mask.sum() / len(y) * abs(p[mask].mean() - y[mask].mean())
    return {
        "brier": round(float(brier_score_loss(y, p)), 4),
        "ece_10bin": round(float(ece), 4),
        "curve": {"mean_predicted": mean_pred, "fraction_positive": frac_pos},
    }


def segment_analysis(
    y_true: pd.Series,
    proba: np.ndarray,
    user_ids: pd.Series,
    users_df: pd.DataFrame,
    threshold: float = 0.5,
) -> list[dict]:
    """Per-archetype error profile for the chosen operating threshold."""
    frame = pd.DataFrame({"y": np.asarray(y_true).astype(int), "proba": proba, "user_id": list(user_ids)})
    frame = frame.merge(users_df[["user_id", "archetype_label"]], on="user_id", how="left")
    frame["pred"] = (frame["proba"] >= threshold).astype(int)
    out = []
    for archetype, group in frame.groupby("archetype_label", dropna=False):
        g = group
        try:
            auc = round(float(roc_auc_score(g["y"], g["proba"])), 4) if g["y"].nunique() > 1 else None
        except Exception:
            auc = None
        out.append(
            {
                "archetype": str(archetype),
                "n": int(len(g)),
                "positive_rate": round(float(g["y"].mean()), 4),
                "predicted_positive_rate": round(float(g["pred"].mean()), 4),
                "error_rate": round(float((g["pred"] != g["y"]).mean()), 4),
                "roc_auc": auc,
            }
        )
    return sorted(out, key=lambda r: r["error_rate"], reverse=True)


def cost_sketch(y_true: pd.Series, proba: np.ndarray) -> dict:
    """Illustrative net-benefit analysis with stated (not measured) unit economics."""
    y = np.asarray(y_true).astype(int)
    n = len(y)

    def benefit(pred: np.ndarray) -> float:
        tp = int(((pred == 1) & (y == 1)).sum())
        acted = int((pred == 1).sum())
        return tp * VALUE_PER_TRUE_POSITIVE - acted * COST_PER_INTERVENTION

    treat_none = 0.0
    treat_all = benefit(np.ones(n, dtype=int))
    grid = []
    for t in [round(v, 2) for v in np.arange(0.05, 1.0, 0.05)]:
        grid.append({"threshold": t, "net_benefit": round(benefit((proba >= t).astype(int)), 2)})
    best = max(grid, key=lambda r: r["net_benefit"])
    return {
        "assumptions": {
            "cost_per_intervention": COST_PER_INTERVENTION,
            "value_per_true_positive": VALUE_PER_TRUE_POSITIVE,
            "note": "Illustrative unit economics for threshold discussion, not measured revenue.",
        },
        "treat_none": treat_none,
        "treat_all": round(treat_all, 2),
        "grid": grid,
        "cost_optimal": best,
    }


def analyze_task(
    task: str,
    X: pd.DataFrame,
    y: pd.Series,
    user_ids: pd.Series,
    users_df: pd.DataFrame,
    models_dir: Path,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    test_user_ids: pd.Series,
) -> dict:
    """Full analysis for one prediction task; returns a JSON-serializable report."""
    models = {}
    for name in ("logistic", "random_forest", "xgboost"):
        fp = models_dir / f"{name}.joblib"
        if not fp.exists():
            logger.error("Missing %s — run `python scripts/train_models.py` first.", fp)
            sys.exit(1)
        models[name] = joblib.load(fp)

    per_model = {}
    val_auc: dict[str, float] = {}
    for name, model in models.items():
        proba = np.asarray(model.predict_proba(X_test)[:, 1], dtype=float)
        val_proba = np.asarray(model.predict_proba(X_val)[:, 1], dtype=float)
        val_auc[name] = round(float(roc_auc_score(y_val, val_proba)), 4)
        per_model[name] = {
            "val_roc_auc": val_auc[name],
            "roc_auc": round(float(roc_auc_score(y_test, proba)), 4),
            "pr_auc": round(float(average_precision_score(y_test, proba)), 4),
            "thresholds": threshold_sweep(y_test, proba),
            "calibration": calibration_summary(y_test, proba),
        }

    # Selection on VALIDATION ROC-AUC; test numbers are the final estimate.
    best = max(val_auc, key=lambda n: val_auc[n])
    best_proba = np.asarray(models[best].predict_proba(X_test)[:, 1], dtype=float)
    f1_threshold = per_model[best]["thresholds"]["f1_max"]["threshold"]

    return {
        "n_test": int(len(y_test)),
        "test_positive_rate": round(float(np.asarray(y_test).mean()), 4),
        "selection": {"criterion": "validation ROC-AUC (never the test set)",
                      "val_roc_auc": {k: per_model[k]["val_roc_auc"] for k in per_model}},
        "dummy_baselines": dummy_baselines(X_train, y_train, X_test, y_test),
        "models": per_model,
        "best_model": best,
        "segments_at_default_threshold": segment_analysis(y_test, best_proba, test_user_ids, users_df, 0.5),
        "segments_at_f1_threshold": segment_analysis(y_test, best_proba, test_user_ids, users_df, f1_threshold),
        "cost_sketch": cost_sketch(y_test, best_proba),
    }


def main() -> None:
    features_path = ROOT / "data" / "processed" / "user_features.parquet"
    events_path = ROOT / "data" / "raw" / "events.parquet"
    users_path = ROOT / "data" / "raw" / "users.csv"
    for p in (features_path, events_path, users_path):
        if not p.exists():
            logger.error("Missing %s — run the pipeline scripts first.", p)
            sys.exit(1)

    features_df = pd.read_parquet(features_path)
    events_df = pd.read_parquet(events_path)
    users_df = pd.read_csv(users_path)
    max_ts = pd.to_datetime(events_df["timestamp"]).max()
    reference_date = max_ts - pd.Timedelta(days=14)

    report: dict = {"seed": SEED, "reference_date": str(reference_date), "tasks": {}}

    for task, builder, dirname in (
        ("retention", build_retention_dataset, "retention"),
        ("monetization", build_monetization_dataset, "monetization"),
    ):
        X, y, _ref = builder(features_df, events_df, reference_date=reference_date)
        user_ids = features_df["user_id"].reset_index(drop=True)
        fit_idx, val_idx, test_idx = reproduce_test_split(X, y)
        report["tasks"][task] = analyze_task(
            task, X, y, user_ids, users_df,
            ROOT / "data" / "models" / dirname,
            X.iloc[fit_idx], y.iloc[fit_idx],
            X.iloc[val_idx], y.iloc[val_idx],
            X.iloc[test_idx], y.iloc[test_idx],
            user_ids.iloc[test_idx].reset_index(drop=True),
        )

    out_path = ROOT / "data" / "models" / "evaluation_report.json"
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    logger.info("Wrote %s", out_path)

    print("\n=== MODEL ANALYSIS (synthetic test set) ===")
    for task, rep in report["tasks"].items():
        print(f"\n[{task}] n_test={rep['n_test']} pos_rate={rep['test_positive_rate']}")
        print(f"  dummy most_frequent ROC-AUC={rep['dummy_baselines']['most_frequent']['roc_auc']} | "
              f"stratified ROC-AUC={rep['dummy_baselines']['stratified']['roc_auc']}")
        for name, m in rep["models"].items():
            t = m["thresholds"]
            print(f"  {name}: VAL-AUC={m['val_roc_auc']} TEST-AUC={m['roc_auc']} PR-AUC={m['pr_auc']} "
                  f"Brier={m['calibration']['brier']} ECE={m['calibration']['ece_10bin']} | "
                  f"F1-max@{t['f1_max']['threshold']}={t['f1_max']['f1']} | "
                  f"Youden@{t['youden']['threshold']}")
        print(f"  best (by VAL-AUC)={rep['best_model']} | cost-optimal threshold={rep['cost_sketch']['cost_optimal']} "
              f"(treat-all={rep['cost_sketch']['treat_all']})")
        print("  worst segments @0.5: " + ", ".join(
            f"{s['archetype']}(err={s['error_rate']},n={s['n']})"
            for s in rep["segments_at_default_threshold"][:3]
        ))


if __name__ == "__main__":
    main()
