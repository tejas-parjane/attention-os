import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def sample_events_df() -> pd.DataFrame:
    """A small synthetic events DataFrame suitable for most unit tests."""
    n = 50
    rng = np.random.default_rng(0)
    timestamps = pd.date_range("2026-01-05", periods=n, freq="4h")
    return pd.DataFrame(
        {
            "user_id": "user_0001",
            "session_id": [f"user_0001_s{i // 3:05d}" for i in range(n)],
            "timestamp": timestamps,
            "event_type": rng.choice(
                ["session_start", "content_view", "purchase", "session_end"],
                size=n,
            ),
            "session_duration": rng.uniform(5, 30, size=n).round(1),
            "screens_viewed": rng.integers(0, 10, size=n),
            "actions_completed": rng.integers(0, 5, size=n),
            "level": np.ones(n, dtype=int),
            "purchase_amount": np.where(
                rng.random(n) < 0.2, rng.uniform(1, 20, size=n).round(2), 0.0
            ),
            "ad_impressions": rng.integers(0, 3, size=n),
            "ad_revenue": rng.uniform(0, 0.5, size=n).round(4),
            "notification_opened": rng.integers(0, 2, size=n),
            "notification_clicked": rng.integers(0, 1, size=n),
            "content_category": rng.choice(
                ["puzzle", "action", "social"], size=n
            ),
            "device_type": rng.choice(["ios", "android"], size=n),
            "country": "US",
            "acquisition_channel": "organic",
        }
    )


@pytest.fixture
def sample_features_df() -> pd.DataFrame:
    """Feature vector DataFrame for a single user."""
    return pd.DataFrame(
        [
            {
                "user_id": "user_0001",
                "sessions_7d": 5,
                "sessions_30d": 12,
                "days_since_last_session": 3,
                "session_duration_avg_7d": 15.0,
                "revenue_30d": 25.0,
                "total_ltv": 40.0,
                "engagement_change_7d": -1.5,
                "session_duration_trend": -0.5,
                "notification_open_rate": 0.3,
                "purchase_conversion_rate": 0.05,
            }
        ]
    )


@pytest.fixture
def sample_predictions_df() -> pd.DataFrame:
    """Model prediction outputs for a single user."""
    return pd.DataFrame(
        [
            {
                "user_id": "user_0001",
                "retention_probability": 0.45,
                "purchase_probability": 0.35,
                "predicted_ltv": 50.0,
                "churn_probability": 0.55,
            }
        ]
    )
