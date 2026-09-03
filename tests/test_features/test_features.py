import numpy as np
import pandas as pd
import pytest

from src.features.engine import build_features_for_user, build_user_features


def _make_user_events(
    user_id: str = "user_0001",
    start: str = "2026-01-01",
    n_sessions: int = 10,
    session_gap_days: float = 1.0,
    seed: int = 0,
) -> pd.DataFrame:
    """Build a minimal events DataFrame with known, ordered session structure."""
    rng = np.random.default_rng(seed)
    rows: list[dict] = []
    base = pd.Timestamp(start)
    for i in range(n_sessions):
        session_id = f"{user_id}_s{i:05d}"
        session_ts = base + pd.Timedelta(days=i * session_gap_days)
        rows.append(
            {
                "user_id": user_id,
                "session_id": session_id,
                "timestamp": session_ts,
                "event_type": "session_start",
                "session_duration": round(float(rng.uniform(5, 25)), 1),
                "screens_viewed": int(rng.integers(1, 8)),
                "actions_completed": int(rng.integers(0, 3)),
                "level": 1,
                "purchase_amount": 10.0 if i % 3 == 0 else 0.0,
                "ad_impressions": int(rng.integers(0, 3)),
                "ad_revenue": round(float(rng.uniform(0, 0.3)), 4),
                "notification_opened": int(rng.integers(0, 2)),
                "notification_clicked": int(rng.integers(0, 1)),
                "content_category": "puzzle",
            }
        )
        # extra content_view event per session
        rows.append(
            {
                "user_id": user_id,
                "session_id": session_id,
                "timestamp": session_ts + pd.Timedelta(minutes=5),
                "event_type": "content_view",
                "session_duration": 0.0,
                "screens_viewed": int(rng.integers(1, 5)),
                "actions_completed": 0,
                "level": 1,
                "purchase_amount": 0.0,
                "ad_impressions": 0,
                "ad_revenue": 0.0,
                "notification_opened": 0,
                "notification_clicked": 0,
                "content_category": "action",
            }
        )
    return pd.DataFrame(rows)


class TestBuildFeaturesForUser:
    def test_returns_feature_dict(self):
        events = _make_user_events(n_sessions=10, session_gap_days=1.0)
        ref = pd.Timestamp("2026-01-15")
        feats = build_features_for_user(events, reference_date=ref)
        assert isinstance(feats, dict)
        assert "sessions_7d" in feats
        assert "days_since_last_session" in feats

    def test_sessions_7d_count(self):
        # Place sessions at days 0,1,2,3,4,5,6,7,8,9; reference at day 10
        events = _make_user_events(n_sessions=10, session_gap_days=1.0)
        ref = pd.Timestamp("2026-01-11")  # day 10
        feats = build_features_for_user(events, reference_date=ref)
        assert feats["sessions_7d"] == 7.0  # days 3..9 inclusive within 7-day window

    def test_sessions_30d_count(self):
        events = _make_user_events(n_sessions=10, session_gap_days=1.0)
        ref = pd.Timestamp("2026-01-11")
        feats = build_features_for_user(events, reference_date=ref)
        assert feats["sessions_30d"] == 10.0  # all 10 sessions within 30-day window

    def test_days_since_last_session(self):
        events = _make_user_events(n_sessions=5, session_gap_days=2.0)
        # Last session_start is at day 8 (0-indexed: 0,2,4,6,8)
        ref = pd.Timestamp("2026-01-12")  # day 11
        feats = build_features_for_user(events, reference_date=ref)
        assert feats["days_since_last_session"] == pytest.approx(3.0, abs=0.01)


class TestLeakagePrevention:
    def test_future_event_does_not_affect_features(self):
        """Construct two event sets identical except for a future 'purchase'
        event. Features at the reference date must be identical."""
        common_rows = [
            {
                "user_id": "u1",
                "session_id": "u1_s00",
                "timestamp": pd.Timestamp("2026-01-05"),
                "event_type": "session_start",
                "session_duration": 10.0,
                "screens_viewed": 2,
                "actions_completed": 1,
                "level": 1,
                "purchase_amount": 0.0,
                "ad_impressions": 0,
                "ad_revenue": 0.0,
                "notification_opened": 0,
                "notification_clicked": 0,
                "content_category": "puzzle",
            }
        ]
        future_purchase = {
            "user_id": "u1",
            "session_id": "u1_s00",
            "timestamp": pd.Timestamp("2026-01-20"),  # AFTER reference_date
            "event_type": "purchase",
            "session_duration": 0.0,
            "screens_viewed": 0,
            "actions_completed": 0,
            "level": 1,
            "purchase_amount": 99.99,
            "ad_impressions": 0,
            "ad_revenue": 0.0,
            "notification_opened": 0,
            "notification_clicked": 0,
            "content_category": "puzzle",
        }

        events_no_future = pd.DataFrame(common_rows)
        events_with_future = pd.DataFrame(common_rows + [future_purchase])

        ref = pd.Timestamp("2026-01-10")

        feats_without = build_user_features(events_no_future, reference_date=ref)
        feats_with = build_user_features(events_with_future, reference_date=ref)

        assert feats_without.shape == feats_with.shape
        for col in feats_without.columns:
            if col == "user_id":
                continue
            v1 = feats_without[col].iloc[0]
            v2 = feats_with[col].iloc[0]
            # NaN != NaN in Python, so handle that case explicitly
            if pd.isna(v1) and pd.isna(v2):
                continue
            assert v1 == v2, f"Leakage detected in column '{col}': {v1} != {v2}"


class TestRollingWindowCorrectness:
    def test_sessions_7d_excludes_old_sessions(self):
        """Only sessions within last 7 days should count in sessions_7d."""
        # 10 sessions, one per day; reference at day 10
        events = _make_user_events(n_sessions=10, session_gap_days=1.0)
        ref = pd.Timestamp("2026-01-11")
        feats = build_features_for_user(events, reference_date=ref)
        # Days 0..9, reference day 10: window is days 3..9 = 7 sessions
        assert feats["sessions_7d"] == 7.0

    def test_sessions_1d_counts_only_today(self):
        events = _make_user_events(n_sessions=3, session_gap_days=1.0)
        # Reference exactly at day 2: window [1,2] => sessions at day 1 and day 2
        ref = pd.Timestamp("2026-01-03")
        feats = build_features_for_user(events, reference_date=ref)
        # session_start events at day 0,1,2; within 1-day window (days <=1 day ago) = day 1 and 2
        assert feats["sessions_1d"] == 2.0


class TestUserAggregation:
    def test_build_user_features_returns_one_row_per_user(self):
        ev1 = _make_user_events(user_id="u1", n_sessions=5, session_gap_days=1.0, seed=0)
        ev2 = _make_user_events(user_id="u2", n_sessions=3, session_gap_days=2.0, seed=1)
        events = pd.concat([ev1, ev2], ignore_index=True)
        ref = pd.Timestamp("2026-01-12")
        result = build_user_features(events, reference_date=ref)
        assert result["user_id"].nunique() == 2
        assert len(result) == 2

    def test_build_user_features_preserves_user_ids(self):
        ev1 = _make_user_events(user_id="alpha", n_sessions=3, session_gap_days=1.0, seed=0)
        ev2 = _make_user_events(user_id="beta", n_sessions=3, session_gap_days=1.0, seed=0)
        events = pd.concat([ev1, ev2], ignore_index=True)
        ref = pd.Timestamp("2026-01-10")
        result = build_user_features(events, reference_date=ref)
        assert set(result["user_id"]) == {"alpha", "beta"}
