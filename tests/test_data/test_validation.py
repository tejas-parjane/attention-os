import numpy as np
import pandas as pd
import pytest

from src.data.validation import (
    DataValidationError,
    validate_events,
)


def _clean_df(n: int = 10) -> pd.DataFrame:
    """Build a minimal clean DataFrame that always passes validation."""
    timestamps = pd.date_range("2026-01-01", periods=n, freq="D")
    return pd.DataFrame(
        {
            "user_id": [f"u{i}" for i in range(n)],
            "session_id": [f"s{i}" for i in range(n)],
            "timestamp": timestamps,
            "event_type": "session_start",
            "session_duration": 10.0,
            "screens_viewed": 1,
            "actions_completed": 0,
            "level": 1,
            "purchase_amount": 0.0,
            "ad_impressions": 0,
            "ad_revenue": 0.0,
            "notification_opened": 0,
            "notification_clicked": 0,
            "content_category": "puzzle",
            "device_type": "ios",
            "country": "US",
            "acquisition_channel": "organic",
        }
    )


class TestValidateEventsClean:
    def test_clean_data_passes(self):
        result = validate_events(_clean_df(), fail_loud=False)
        assert result.passed is True
        critical = [c for c in result.checks if c["status"] == "failed"]
        assert len(critical) == 0, f"Unexpected failures: {critical}"


class TestValidateEventsMissingColumns:
    def test_missing_required_column_raises(self):
        df = _clean_df().drop(columns=["user_id"])
        with pytest.raises(DataValidationError, match="Missing required columns"):
            validate_events(df, fail_loud=True)

    def test_missing_required_column_not_raised_when_fail_loud_false(self):
        df = _clean_df().drop(columns=["user_id"])
        result = validate_events(df, fail_loud=False)
        assert result.passed is False
        col_check = [c for c in result.checks if c["name"] == "required_columns"][0]
        assert col_check["status"] == "failed"


class TestValidateNullUserId:
    def test_null_user_id_flagged(self):
        df = _clean_df()
        df.loc[0, "user_id"] = np.nan
        with pytest.raises(DataValidationError, match="null user_id"):
            validate_events(df, fail_loud=True)

    def test_null_user_id_fail_loud_false(self):
        df = _clean_df()
        df.loc[0, "user_id"] = np.nan
        result = validate_events(df, fail_loud=False)
        assert result.passed is False
        uid_check = [c for c in result.checks if c["name"] == "null_user_id"][0]
        assert uid_check["status"] == "failed"


class TestValidateTimestamps:
    def test_invalid_timestamps_flagged(self):
        df = _clean_df()
        df.loc[2, "timestamp"] = "not-a-date"
        with pytest.raises(DataValidationError, match="timestamp"):
            validate_events(df, fail_loud=True)


class TestValidateDuplicates:
    def test_duplicate_events_flagged(self):
        df = _clean_df(n=5)
        dup = df.iloc[[0]].copy()
        df = pd.concat([df, dup], ignore_index=True)
        result = validate_events(df, fail_loud=False)
        dup_check = [c for c in result.checks if c["name"] == "duplicates"][0]
        assert dup_check["status"] == "failed"


class TestValidateNegativeRevenue:
    def test_negative_revenue_flagged(self):
        df = _clean_df()
        df.loc[0, "purchase_amount"] = -5.0
        result = validate_events(df, fail_loud=False)
        rev_check = [c for c in result.checks if c["name"] == "negative_revenue"][0]
        assert rev_check["status"] == "failed"
        assert result.passed is False


class TestValidateFutureEvents:
    def test_future_events_flagged(self):
        df = _clean_df()
        df.loc[0, "timestamp"] = pd.Timestamp("2027-01-01")
        result = validate_events(df, fail_loud=False, reference_max_ts=pd.Timestamp("2026-12-31"))
        fut_check = [c for c in result.checks if c["name"] == "future_events"][0]
        assert fut_check["status"] == "failed"


class TestValidateEmptyInput:
    def test_empty_dataframe_raises(self):
        with pytest.raises(DataValidationError):
            validate_events(pd.DataFrame(), fail_loud=True)
