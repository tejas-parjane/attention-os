"""
Data validation module for the events DataFrame.

Defines reusable data-quality checks that can be run on the generated
(or ingested) events data, and a strict/fail-loud mode suitable for
pipeline gating.
"""

from __future__ import annotations

from datetime import timedelta
from typing import Any

import pandas as pd
from pydantic import BaseModel

# ---------------------------------------------------------------------------
# Configuration / allowed values
# ---------------------------------------------------------------------------

REQUIRED_COLUMNS = [
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

EVENT_TYPES = [
    "app_open",
    "session_start",
    "session_end",
    "content_view",
    "game_start",
    "level_complete",
    "challenge_started",
    "challenge_completed",
    "purchase",
    "ad_view",
    "notification_received",
    "notification_opened",
    "notification_clicked",
    "subscription_started",
    "subscription_cancelled",
]

DEVICE_TYPES = ["ios", "android", "web"]
COUNTRIES = ["US", "UK", "DE", "BR", "IN", "JP"]
ACQUISITION_CHANNELS = ["organic", "paid_social", "paid_search", "referral", "viral"]

# Maximum allowed timestamp (reference "now" used to detect future events).
# Generated data starts 2026-01-01 over 90 days; we allow a little headroom.
MAX_ALLOWED_TIMESTAMP = pd.Timestamp("2026-12-31 23:59:59")

MAX_SESSION_DURATION = timedelta(hours=12)


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------


class ValidationResult(BaseModel):
    """Outcome of a full validation pass over an events DataFrame."""

    passed: bool
    checks: list[dict]  # [{name, status, message}]


class DataValidationError(Exception):
    """Raised when critical data quality rules are violated."""


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def _check(
    checks: list[dict],
    name: str,
    ok: bool,
    message: str,
) -> None:
    checks.append(
        {
            "name": name,
            "status": "passed" if ok else "failed",
            "message": message,
        }
    )


def validate_events(
    df: pd.DataFrame,
    fail_loud: bool = True,
    reference_max_ts: pd.Timestamp | None = None,
) -> ValidationResult:
    """Perform data quality checks on an events DataFrame.

    Checks performed:
      1. Required columns present
      2. No null user_id
      3. No null session_id
      4. Timestamps are valid (parseable, no NaT)
      5. No future events (beyond max allowed / reference max timestamp)
      6. No duplicate events (user_id + session_id + timestamp + event_type)
      7. No negative revenue (purchase_amount >= 0, ad_revenue >= 0)
      8. No impossible session durations (0 <= duration <= 12h)
      9. event_type is in the allowed list
     10. device_type / country / channel in allowed lists (if provided)

    Critical checks (raised when ``fail_loud`` is True and they fail):
      missing required columns, null user_id, invalid timestamps.

    Non-critical checks (reported, never raised): duplicates, negative
    revenue, impossible durations, out-of-allowed-list values.
    """
    checks: list[dict] = []
    critical_failed = False

    # --- 1. Required columns present : CRITICAL ---------------------------
    if df is None or len(df) == 0:
        return _hard_fail(
            "empty_input",
            "The supplied DataFrame is empty or None.",
            fail_loud,
            checks,
        )

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        critical_failed = True
        _check(
            checks,
            "required_columns",
            ok=False,
            message=f"Missing required columns: {missing}",
        )
    else:
        _check(
            checks,
            "required_columns",
            ok=True,
            message="All required columns present.",
        )

    # --- 2. No null user_id : CRITICAL ------------------------------------
    n_null_user = int(df["user_id"].isna().sum()) if "user_id" in df.columns else 0
    if n_null_user:
        critical_failed = True
        _check(
            checks,
            "null_user_id",
            ok=False,
            message=f"{n_null_user} row(s) have null user_id.",
        )
    else:
        _check(checks, "null_user_id", ok=True, message="No null user_id.")

    # --- 3. No null session_id : WARN (notifications legitimately blank) ---
    if "session_id" in df.columns:
        n_null_session = int(df["session_id"].isna().sum())
        # Also treat empty-string as "no session" for notification events.
        n_blank_session = int((df["session_id"].astype(str).str.strip() == "").sum())
        if n_null_session or n_blank_session:
            _check(
                checks,
                "null_session_id",
                ok=False,
                message=(
                    f"{n_null_session} null + {n_blank_session} blank "
                    "session_id(s) (expected only for notification events)."
                ),
            )
        else:
            _check(checks, "null_session_id", ok=True, message="No null session_id.")
    else:
        _check(checks, "null_session_id", ok=False, message="session_id column missing.")

    # --- 4. Timestamps valid (parseable, no NaT) : CRITICAL ----------------
    n_bad_ts = 0
    if "timestamp" not in df.columns:
        critical_failed = True
        _check(checks, "timestamps", ok=False, message="timestamp column missing.")
    else:
        ts = df["timestamp"]
        if not pd.api.types.is_datetime64_any_dtype(ts):
            try:
                ts = pd.to_datetime(ts, errors="coerce")
                n_bad_ts = int(ts.isna().sum())
            except Exception:  # noqa: BLE001
                n_bad_ts = int(df["timestamp"].isna().sum() + len(df.index))
        else:
            n_bad_ts = int(ts.isna().sum())

        if n_bad_ts:
            critical_failed = True
            _check(
                checks,
                "timestamps",
                ok=False,
                message=f"{n_bad_ts} row(s) have unparseable/NaT timestamps.",
            )
        else:
            _check(checks, "timestamps", ok=True, message="All timestamps valid.")

    # --- 5. No future events ----------------------------------------------
    if "timestamp" in df.columns and n_bad_ts == 0:
        ref_ts = reference_max_ts if reference_max_ts is not None else MAX_ALLOWED_TIMESTAMP
        ref_ts = pd.Timestamp(ref_ts)
        ts = pd.to_datetime(df["timestamp"])
        n_future = int((ts > ref_ts).sum())
        if n_future:
            _check(
                checks,
                "future_events",
                ok=False,
                message=f"{n_future} event(s) exceed reference max {ref_ts}.",
            )
        else:
            _check(
                checks,
                "future_events",
                ok=True,
                message=f"No events beyond reference max {ref_ts}.",
            )

    # --- 6. No duplicate events --------------------------------------------
    dup_cols = [c for c in ["user_id", "session_id", "timestamp", "event_type"] if c in df.columns]
    n_dupes = 0
    if len(dup_cols) == 4:
        n_dupes = int(df.duplicated(subset=dup_cols, keep=False).sum())
        _check(
            checks,
            "duplicates",
            ok=n_dupes == 0,
            message=(
                f"{n_dupes} duplicate event(s) (same user/session/timestamp/type)."
                if n_dupes
                else "No duplicate events."
            ),
        )

    # --- 7. No negative revenue ---------------------------------------------
    n_neg_rev = 0
    for col in ("purchase_amount", "ad_revenue"):
        if col in df.columns:
            n_neg_rev += int((df[col] < 0).sum())
    if n_neg_rev:
        _check(
            checks,
            "negative_revenue",
            ok=False,
            message=f"{n_neg_rev} row(s) with negative purchase_amount/ad_revenue.",
        )
    else:
        _check(checks, "negative_revenue", ok=True, message="No negative revenue.")

    # --- 8. No impossible session durations ----------------------------------
    n_bad_dur = 0
    if "session_duration" in df.columns:
        dur = pd.to_numeric(df["session_duration"], errors="coerce")
        n_bad_dur = int(((dur < 0) | (dur > MAX_SESSION_DURATION.total_seconds() / 60)).sum())
        if n_bad_dur:
            _check(
                checks,
                "session_duration",
                ok=False,
                message=(
                    f"{n_bad_dur} row(s) with session_duration outside "
                    f"[0, {int(MAX_SESSION_DURATION.total_seconds() // 60)}] minutes."
                ),
            )
        else:
            _check(checks, "session_duration", ok=True, message="Session durations plausible.")

    # --- 9. event_type in allowed list ---------------------------------------
    invalid_types: set[Any] = set()
    if "event_type" in df.columns:
        invalid_types = set(df["event_type"].dropna().unique()) - set(EVENT_TYPES)
        if invalid_types:
            _check(
                checks,
                "event_type",
                ok=False,
                message=f"Unexpected event_type values: {sorted(map(str, invalid_types))}",
            )
        else:
            _check(checks, "event_type", ok=True, message="All event_type values allowed.")

    # --- 10. device_type / country / channel in allowed lists ------------------
    for col, allowed in (
        ("device_type", DEVICE_TYPES),
        ("country", COUNTRIES),
        ("acquisition_channel", ACQUISITION_CHANNELS),
    ):
        if col not in df.columns or df[col].dropna().empty:
            _check(checks, col, ok=True, message=f"{col}: not present/empty, skipped.")
            continue
        invalids = set(df[col].dropna().astype(str).unique()) - set(allowed)
        if invalids:
            _check(
                checks,
                col,
                ok=False,
                message=f"Unexpected {col} values: {sorted(map(str, invalids))}",
            )
        else:
            _check(checks, col, ok=True, message=f"All {col} values allowed.")

    # --- Aggregate ------------------------------------------------------------
    noncrit_failed = any(c["status"] == "failed" for c in checks if c["name"] not in _CRITICAL)
    passed = not critical_failed

    if fail_loud and critical_failed:
        _raise_critical(checks)

    return ValidationResult(
        passed=passed and not noncrit_failed,
        checks=checks,
    )


# Critical checks that should fail loudly.
_CRITICAL = {"required_columns", "null_user_id", "timestamps"}


def _hard_fail(name: str, message: str, fail_loud: bool, checks: list[dict]) -> ValidationResult:
    _check(checks, name, ok=False, message=message)
    if fail_loud:
        raise DataValidationError(message)
    return ValidationResult(passed=False, checks=checks)


def _raise_critical(checks: list[dict]) -> None:
    failed = [c for c in checks if c["name"] in _CRITICAL and c["status"] == "failed"]
    msgs = [f"[{c['name']}] {c['message']}" for c in failed]
    raise DataValidationError("Critical data validation failed:\n" + "\n".join(msgs))
