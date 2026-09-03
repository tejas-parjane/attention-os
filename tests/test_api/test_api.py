from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from src.api.app import app

client = TestClient(app, raise_server_exceptions=False)


class TestHealthEndpoint:
    def test_returns_200(self):
        resp = client.get("/health")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] in ("healthy", "degraded")
        assert "version" in body


class TestUserNotFound:
    def test_returns_404(self):
        resp = client.get("/users/nonexistent_user_xyz")
        assert resp.status_code == 404


class TestDecisionEndpoint:
    def test_returns_valid_structure_with_mocked_services(self):
        fake_features = {
            "sessions_7d": 3,
            "days_since_last_session": 2,
            "revenue_30d": 10.0,
        }
        fake_predictions = {
            "retention_probability": 0.6,
            "purchase_probability": 0.2,
            "predicted_ltv": 40.0,
        }
        fake_user_state = {"user_id": "mock_user", "engagement": "stable"}
        fake_decision = {"action": "NO_ACTION", "confidence": 0.4, "expected_value": 0.0}

        with (
            patch(
                "src.api.services.get_features_for_user",
                return_value={"user_id": "mock_user", "features": fake_features},
            ),
            patch(
                "src.api.services.get_predictions_for_user",
                return_value=fake_predictions,
            ),
            patch(
                "src.api.services._user_row_from_parquet",
                return_value=None,
            ),
        ):
            resp = client.post("/users/mock_user/decision")
            assert resp.status_code == 200
            body = resp.json()
            assert "action" in body.get("decision", {}) or "user_id" in body


class TestExperimentsEndpoint:
    def test_returns_list(self):
        with patch("src.api.services._load_parquet", return_value=None), patch(
            "src.api.services._db_session_safe",
            return_value=iter([None]),
        ):
            resp = client.get("/experiments")
            assert resp.status_code == 200
            assert isinstance(resp.json(), list)
