import pandas as pd
import pytest

from src.data.generator import (
    ARCHETYPE_NAMES,
    ARCHETYPE_WEIGHTS,
    EVENT_TYPES,
    generate_events,
)


class TestGenerateEvents:
    def test_returns_dataframe(self):
        df = generate_events(n_users=10, days=5, seed=99)
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0

    def test_required_columns_present(self):
        required = [
            "user_id",
            "session_id",
            "timestamp",
            "event_type",
            "session_duration",
            "screens_viewed",
            "actions_completed",
            "level",
            "purchase_amount",
            "ad_impressions",
            "ad_revenue",
            "notification_opened",
            "notification_clicked",
            "content_category",
            "device_type",
            "country",
            "acquisition_channel",
        ]
        df = generate_events(n_users=5, days=3, seed=1)
        for col in required:
            assert col in df.columns, f"Missing column: {col}"

    def test_event_types_in_expected_set(self):
        df = generate_events(n_users=10, days=5, seed=2)
        observed = set(df["event_type"].unique())
        assert observed.issubset(set(EVENT_TYPES)), (
            f"Unexpected event types: {observed - set(EVENT_TYPES)}"
        )

    def test_positive_purchase_amounts(self):
        df = generate_events(n_users=20, days=7, seed=3)
        purchases = df.loc[df["event_type"] == "purchase", "purchase_amount"]
        assert len(purchases) > 0, "No purchase events generated"
        assert (purchases >= 0).all(), "Negative purchase amounts found"

    def test_timestamps_are_valid_datetimes(self):
        df = generate_events(n_users=5, days=3, seed=4)
        assert pd.api.types.is_datetime64_any_dtype(df["timestamp"]), (
            "timestamp column is not datetime"
        )
        assert df["timestamp"].isna().sum() == 0, "NaT values in timestamp"

    def test_no_duplicate_event_keys(self):
        df = generate_events(n_users=10, days=5, seed=5)
        subset = ["user_id", "session_id", "timestamp", "event_type"]
        n_dupes = df.duplicated(subset=subset, keep=False).sum()
        assert n_dupes == 0, f"Found {n_dupes} duplicate event key rows"

    def test_multiple_archetypes_present(self):
        df = generate_events(n_users=200, days=30, seed=6)
        unique_users = df["user_id"].nunique()
        assert unique_users >= 50, f"Expected many users, got {unique_users}"
        archetypes_in_config = set(ARCHETYPE_WEIGHTS.keys())
        assert archetypes_in_config == set(ARCHETYPE_NAMES)

    def test_no_future_dates_relative_to_known_start(self):
        """Generated data starts 2026-01-01; with 90 days, nothing should exceed 2026-04-11."""
        df = generate_events(n_users=10, days=90, seed=7)
        max_ts = df["timestamp"].max()
        cutoff = pd.Timestamp("2026-05-01")
        assert max_ts < cutoff, f"Latest timestamp {max_ts} is unreasonably far in the future"

    def test_user_ids_match_count(self):
        df = generate_events(n_users=25, days=3, seed=8)
        assert df["user_id"].nunique() == 25

    def test_sorted_by_timestamp(self):
        df = generate_events(n_users=10, days=5, seed=9)
        assert df["timestamp"].is_monotonic_increasing, "Events not sorted by timestamp"
