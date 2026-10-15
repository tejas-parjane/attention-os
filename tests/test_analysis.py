"""Tests for scripts/analyze_models.py — pure functions on synthetic inputs."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from analyze_models import (  # noqa: E402
    calibration_summary,
    cost_sketch,
    dummy_baselines,
    reproduce_test_split,
    segment_analysis,
    threshold_sweep,
)


def _toy(n: int = 200, seed: int = 7):
    rng = np.random.default_rng(seed)
    y = pd.Series(rng.integers(0, 2, n))
    proba = np.clip(y.to_numpy() * 0.6 + rng.normal(0.2, 0.15, n), 0.01, 0.99)
    X = pd.DataFrame({"f1": rng.normal(size=n), "f2": rng.normal(size=n)})
    return X, y, proba


def test_reproduce_test_split_sizes_disjoint_and_stratified():
    X, y, _ = _toy(200)
    fit, va, te = reproduce_test_split(X, y)
    assert len(te) == 30 and len(fit) + len(va) == 170  # ceil-per-class rounding
    assert len(set(fit) & set(va) & set(te)) == 0
    assert len(set(fit) | set(va) | set(te)) == 200
    # positive rate preserved approximately in each fold
    overall = y.mean()
    for idx in (fit, va, te):
        assert abs(y.iloc[idx].mean() - overall) < 0.15


def test_reproduce_test_split_deterministic():
    X, y, _ = _toy(200)
    assert reproduce_test_split(X, y)[2].tolist() == reproduce_test_split(X, y)[2].tolist()


def test_threshold_sweep_f1max_beats_default():
    _, y, proba = _toy()
    rep = threshold_sweep(y, proba)
    f1_at_05 = next(r["f1"] for r in rep["grid"] if r["threshold"] == 0.5)
    assert rep["f1_max"]["f1"] >= f1_at_05
    assert 0.05 <= rep["f1_max"]["threshold"] <= 0.95
    assert 0.05 <= rep["youden"]["threshold"] <= 0.95


def test_calibration_summary_bounds():
    _, y, proba = _toy()
    rep = calibration_summary(y, proba)
    assert 0.0 <= rep["brier"] <= 1.0
    assert 0.0 <= rep["ece_10bin"] <= 1.0
    assert len(rep["curve"]["mean_predicted"]) == 10


def test_dummy_baselines_structure():
    X, y, _ = _toy()
    tr, _va, te = reproduce_test_split(X, y)
    rep = dummy_baselines(X.iloc[tr], y.iloc[tr], X.iloc[te], y.iloc[te])
    assert set(rep) == {"most_frequent", "stratified"}
    assert rep["most_frequent"]["roc_auc"] == 0.5


def test_segment_analysis_ranks_worst_first():
    _, y, proba = _toy()
    user_ids = pd.Series([f"u{i}" for i in range(len(y))])
    users = pd.DataFrame(
        {"user_id": list(user_ids), "archetype_label": ["a" if i % 2 else "b" for i in range(len(y))]}
    )
    segs = segment_analysis(y, proba, user_ids, users, threshold=0.5)
    assert {s["archetype"] for s in segs} == {"a", "b"}
    errs = [s["error_rate"] for s in segs]
    assert errs == sorted(errs, reverse=True)
    assert all(0.0 <= s["error_rate"] <= 1.0 for s in segs)


def test_cost_sketch_best_beats_doing_nothing():
    _, y, proba = _toy()
    rep = cost_sketch(y, proba)
    assert rep["treat_none"] == 0.0
    assert rep["cost_optimal"]["net_benefit"] >= rep["treat_none"]
    assert rep["assumptions"]["cost_per_intervention"] > 0
