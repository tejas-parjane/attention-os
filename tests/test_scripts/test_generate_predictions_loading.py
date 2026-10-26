"""Batch prediction must load the validation-selected model, not a filename default."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import joblib
import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "generate_predictions.py"


@pytest.fixture()
def gp(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location("generate_predictions_under_test", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "ROOT", tmp_path)
    return module


def _write_models(root: Path, task: str, val_auc: dict[str, float]) -> Path:
    model_dir = root / "data" / "models" / task
    model_dir.mkdir(parents=True)
    for name in ("logistic", "random_forest", "xgboost"):
        joblib.dump({"name": name}, model_dir / f"{name}.joblib")
    return model_dir


def _write_report(root: Path, selection: dict[str, dict[str, float]]) -> None:
    report = {"tasks": {t: {"selection": {"val_roc_auc": v}} for t, v in selection.items()}}
    (root / "data" / "models").mkdir(parents=True, exist_ok=True)
    (root / "data" / "models" / "evaluation_report.json").write_text(json.dumps(report))


def test_loads_validation_winner_not_xgboost_by_filename(gp, tmp_path):
    retention_dir = _write_models(tmp_path, "retention", {})
    monetization_dir = _write_models(tmp_path, "monetization", {})
    _write_report(
        tmp_path,
        {
            "retention": {"logistic": 0.86, "random_forest": 0.85, "xgboost": 0.84},
            "monetization": {"logistic": 0.85, "random_forest": 0.87, "xgboost": 0.88},
        },
    )
    assert gp._load_best_model(retention_dir, "retention") == {"name": "logistic"}
    assert gp._load_best_model(monetization_dir, "monetization") == {"name": "xgboost"}


def test_falls_back_to_filename_order_without_report(gp, tmp_path):
    retention_dir = _write_models(tmp_path, "retention", {})
    assert gp._load_best_model(retention_dir, "retention") == {"name": "xgboost"}
