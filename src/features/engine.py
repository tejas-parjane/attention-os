"""Leakage-safe feature engineering pipeline.

Computes user-level features from raw events using rolling windows.
All features respect temporal causality: features for a user at time T
use only events with timestamps strictly before T, preventing any
information from the future leaking into historical features.

Feature categories: engagement, retention, monetization, behavioral,
lifecycle, and session quality.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

REQUIRED_EVENT_COLUMNS = [
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
]


def _safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Divide with zero-safety."""
    if denominator == 0 or np.isnan(denominator):
        return default
    return numerator / denominator


def _coefficient_of_variation(values: pd.Series) -> float:
    """Coefficient of variation; 0 if mean is 0 or insufficient data."""
    if len(values) < 2:
        return 0.0
    mean = values.mean()
    if mean == 0 or np.isnan(mean):
        return 0.0
    return float(values.std() / mean)


def _linear_slope(x: np.ndarray, y: np.ndarray) -> float:
    """Ordinary least-squares slope. Returns 0 if insufficient or degenerate."""
    if len(x) < 2:
        return 0.0
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if len(x) < 2:
        return 0.0
    n = len(x)
    sum_x = x.sum()
    sum_y = y.sum()
    sum_xy = (x * y).sum()
    sum_x2 = (x * x).sum()
    denom = n * sum_x2 - sum_x * sum_x
    if denom == 0:
        return 0.0
    return float((n * sum_xy - sum_x * sum_y) / denom)


def _validate_events(df: pd.DataFrame) -> None:
    """Raise ValueError if required columns are missing."""
    missing = set(REQUIRED_EVENT_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"Events DataFrame missing required columns: {missing}")


def build_features_for_user(
    user_events: pd.DataFrame,
    reference_date: datetime,
) -> dict[str, float]:
    """Compute all features for a single user.

    Temporal leakage prevention
    ---------------------------
    ``user_events`` must already contain only rows whose timestamps are
    strictly before ``reference_date``.  Within the function, window-based
    filters (1 d, 7 d, 30 d, etc.) further restrict to the appropriate
    subset so that every feature is computed solely from past data.

    Parameters
    ----------
    user_events:
        Events for one user, already filtered to ``timestamp < reference_date``.
    reference_date:
        The cut-off point; no event at or after this time may be used.

    Returns
    -------
    dict[str, float]
        Mapping of feature name to computed value.
    """

    feats: dict[str, Any] = {}

    if user_events.empty:
        return _empty_features()

    ts = pd.to_datetime(user_events["timestamp"])
    ed = user_events.copy()
    ed["_ts"] = ts
    ed = ed.sort_values("_ts")

    now = pd.Timestamp(reference_date)

    # ------------------------------------------------------------------
    # Derived temporals
    # ------------------------------------------------------------------
    ed["_days_ago"] = (now - ed["_ts"]).dt.total_seconds() / 86400.0
    ed["_hour"] = ed["_ts"].dt.hour
    ed["_is_weekend"] = ed["_ts"].dt.dayofweek >= 5
    ed["_date"] = ed["_ts"].dt.date

    # Window masks
    m1 = ed["_days_ago"] <= 1
    m7 = ed["_days_ago"] <= 7
    m30 = ed["_days_ago"] <= 30
    m3 = ed["_days_ago"] <= 3

    n_unique_sessions = lambda mask: ed.loc[mask, "session_id"].nunique()  # noqa: E731

    # ==================================================================
    # ENGAGEMENT
    # ==================================================================
    feats["sessions_1d"] = float(n_unique_sessions(m1))
    feats["sessions_7d"] = float(n_unique_sessions(m7))
    feats["sessions_30d"] = float(n_unique_sessions(m30))

    # Session-duration averages (across sessions in the window)
    def _avg_session_duration(mask: pd.Series) -> float:
        sub = ed.loc[mask]
        if sub.empty:
            return 0.0
        per_session = sub.groupby("session_id")["session_duration"].first()
        return float(per_session.mean())

    feats["session_duration_avg_7d"] = _avg_session_duration(m7)
    feats["session_duration_avg_30d"] = _avg_session_duration(m30)

    # Session duration trend: slope of duration over last 7 sessions
    session_durs = ed.groupby("session_id").agg(
        duration=("session_duration", "first"),
        ts=("session_id", lambda s: ed.loc[s.index[0], "_ts"]),
    )
    if len(session_durs) >= 2:
        last7 = session_durs.sort_values("ts").tail(7)
        x = np.arange(len(last7), dtype=float)
        y = last7["duration"].values.astype(float)
        feats["session_duration_trend"] = _linear_slope(x, y)
    else:
        feats["session_duration_trend"] = 0.0

    # Days active
    feats["days_active_7d"] = float(ed.loc[m7, "_date"].nunique())
    feats["days_active_30d"] = float(ed.loc[m30, "_date"].nunique())

    # Per-session aggregates
    def _per_session_avg(col: str, mask: pd.Series) -> float:
        sub = ed.loc[mask]
        if sub.empty:
            return 0.0
        ps = sub.groupby("session_id")[col].sum()
        return float(ps.mean())

    feats["events_per_session_7d"] = _per_session_avg("_days_ago", m7)
    feats["avg_screens_viewed_7d"] = _per_session_avg("screens_viewed", m7)
    feats["avg_actions_completed_7d"] = _per_session_avg("actions_completed", m7)

    # ==================================================================
    # RETENTION
    # ==================================================================
    session_firsts = ed.groupby("session_id")["_ts"].min().sort_values(ascending=False)
    days_since_last = (now - session_firsts.iloc[0]).total_seconds() / 86400.0 if len(session_firsts) >= 1 else np.nan
    days_since_second = (now - session_firsts.iloc[1]).total_seconds() / 86400.0 if len(session_firsts) >= 2 else np.nan

    feats["days_since_last_session"] = float(days_since_last) if not np.isnan(days_since_last) else np.nan
    feats["days_since_second_last_session"] = float(days_since_second) if not np.isnan(days_since_second) else np.nan

    feats["sessions_last_3d"] = float(n_unique_sessions(m3))
    feats["sessions_last_7d"] = float(n_unique_sessions(m7))

    # Engagement change: sessions in current window vs. preceding window
    m_prev7 = (ed["_days_ago"] > 7) & (ed["_days_ago"] <= 14)
    m_prev30 = (ed["_days_ago"] > 30) & (ed["_days_ago"] <= 60)

    prev_7 = float(n_unique_sessions(m_prev7))
    curr_7 = feats["sessions_7d"]
    feats["engagement_change_7d"] = curr_7 - prev_7

    prev_30 = float(n_unique_sessions(m_prev30))
    curr_30 = feats["sessions_30d"]
    feats["engagement_change_30d"] = curr_30 - prev_30

    # ==================================================================
    # MONETIZATION
    # ==================================================================
    purchases_total = ed["purchase_amount"].gt(0).sum()
    purchases_30 = ed.loc[m30, "purchase_amount"].gt(0).sum()

    feats["purchase_count_total"] = float(purchases_total)
    feats["purchase_count_30d"] = float(purchases_30)

    feats["revenue_7d"] = float(ed.loc[m7, "ad_revenue"].sum())
    feats["revenue_30d"] = float(ed.loc[m30, "ad_revenue"].sum())
    feats["total_ltv"] = float(ed["purchase_amount"].sum() + ed["ad_revenue"].sum())

    feats["avg_purchase_value"] = (
        float(ed.loc[ed["purchase_amount"] > 0, "purchase_amount"].mean())
        if purchases_total > 0
        else 0.0
    )

    feats["ad_revenue_7d"] = float(ed.loc[m7, "ad_revenue"].sum())
    feats["ad_revenue_30d"] = float(ed.loc[m30, "ad_revenue"].sum())

    total_sessions = float(n_unique_sessions(ed.index.notna()))
    feats["purchase_conversion_rate"] = _safe_divide(purchases_total, total_sessions)

    # ==================================================================
    # BEHAVIORAL
    # ==================================================================
    feats["content_diversity"] = float(ed["content_category"].nunique())

    challenge_events = ed[ed["event_type"].str.contains("challenge", case=False, na=False)]
    challenge_completed = challenge_events["event_type"].str.contains("complete", case=False, na=False)
    feats["challenge_completion_rate"] = _safe_divide(
        float(challenge_completed.sum()), float(len(challenge_events))
    )

    notif_open = ed["notification_opened"].sum()
    notif_click = ed["notification_clicked"].sum()
    notif_total = len(ed)
    feats["notification_open_rate"] = _safe_divide(float(notif_open), float(notif_total))
    feats["notification_click_rate"] = _safe_divide(float(notif_click), float(notif_total))

    feats["avg_level"] = float(ed["level"].mean())
    feats["highest_level"] = float(ed["level"].max())

    days_active_total = ed["_date"].nunique()
    level_range = ed["level"].max() - ed["level"].min() if len(ed) > 1 else 0
    feats["level_progression_rate"] = _safe_divide(float(level_range), float(days_active_total))

    # ==================================================================
    # LIFECYCLE
    # ==================================================================
    first_seen = ed["_ts"].min()
    feats["days_since_signup"] = float((now - first_seen).total_seconds() / 86400.0)
    feats["new_user_flag"] = 1.0 if feats["days_since_signup"] < 7 else 0.0

    # Returning user: had a gap > 3 days between consecutive sessions, then returned
    returning = 0.0
    if len(session_firsts) >= 2:
        sf_sorted = session_firsts.sort_values()
        gaps = sf_sorted.diff().dt.total_seconds().dropna() / 86400.0
        if (gaps > 3).any():
            first_gap_idx = gaps[gaps > 3].index[0]
            later = sf_sorted.loc[sf_sorted.index >= first_gap_idx]
            if len(later) >= 2:
                returning = 1.0
    feats["returning_user_flag"] = returning

    # ==================================================================
    # SESSION QUALITY
    # ==================================================================
    per_session_dur = ed.groupby("session_id")["session_duration"].first()
    feats["avg_session_duration"] = float(per_session_dur.mean()) if len(per_session_dur) else 0.0
    feats["session_duration_cv"] = _coefficient_of_variation(per_session_dur)

    peak_hour = ed.groupby("session_id")["_hour"].first().mode()
    feats["peak_hour"] = float(peak_hour.iloc[0]) if len(peak_hour) else 0.0

    per_session_weekend = ed.groupby("session_id")["_is_weekend"].first()
    feats["weekend_ratio"] = float(per_session_weekend.mean()) if len(per_session_weekend) else 0.0

    # Coerce any remaining object / bool to float
    return {k: float(v) if v is not None and not isinstance(v, float) else (v if isinstance(v, float) else float(v)) for k, v in feats.items()}


def _empty_features() -> dict[str, float]:
    """Return a dict of NaN/0 for every expected feature when there are no events."""
    keys = [
        "sessions_1d", "sessions_7d", "sessions_30d",
        "session_duration_avg_7d", "session_duration_avg_30d", "session_duration_trend",
        "days_active_7d", "days_active_30d", "events_per_session_7d",
        "avg_screens_viewed_7d", "avg_actions_completed_7d",
        "days_since_last_session", "days_since_second_last_session",
        "sessions_last_3d", "sessions_last_7d",
        "engagement_change_7d", "engagement_change_30d",
        "purchase_count_total", "purchase_count_30d",
        "revenue_7d", "revenue_30d", "total_ltv", "avg_purchase_value",
        "ad_revenue_7d", "ad_revenue_30d", "purchase_conversion_rate",
        "content_diversity", "challenge_completion_rate",
        "notification_open_rate", "notification_click_rate",
        "avg_level", "highest_level", "level_progression_rate",
        "days_since_signup", "new_user_flag", "returning_user_flag",
        "avg_session_duration", "session_duration_cv", "peak_hour", "weekend_ratio",
    ]
    return {k: 0.0 for k in keys}


def build_user_features(
    events_df: pd.DataFrame,
    user_metadata_df: pd.DataFrame | None = None,
    reference_date: datetime | None = None,
) -> pd.DataFrame:
    """Build one-row-per-user feature DataFrame from raw events.

    Leakage-safety
    --------------
    Each user's features are computed with ``reference_date`` as the
    temporal boundary.  ``build_features_for_user`` receives only the
    slice of events where ``timestamp < reference_date``.  Downstream
    consumers should always provide a ``reference_date`` in the past
    (e.g. the prediction target time minus any label-look-ahead window)
    to guarantee that no future information contaminates features.

    Parameters
    ----------
    events_df:
        Raw event log with the required columns listed in
        ``REQUIRED_EVENT_COLUMNS``.
    user_metadata_df:
        Optional metadata (e.g. signup_date).  If provided, it must
        contain at least ``user_id`` and ``signup_date`` columns.
    reference_date:
        Temporal cut-off.  Defaults to the maximum timestamp in the
        events plus one second (useful for batch offline scoring but
        **beware** that this still respects leakage for the *label*
        period you supply externally).

    Returns
    -------
    pd.DataFrame
        One row per unique ``user_id`` with all computed features.
    """

    _validate_events(events_df)

    if events_df.empty:
        logger.warning("Empty events DataFrame passed; returning empty features.")
        return pd.DataFrame(columns=["user_id"])

    df = events_df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    if reference_date is None:
        reference_date = (df["timestamp"].max() + pd.Timedelta(seconds=1)).to_pydatetime()
        logger.info("No reference_date supplied; defaulting to %s", reference_date)

    ref = pd.Timestamp(reference_date)

    feature_rows: list[dict[str, Any]] = []
    user_ids = df["user_id"].unique()

    for uid in user_ids:
        user_ev = df.loc[df["user_id"] == uid].copy()
        # Hard temporal filter: only events strictly before reference_date
        user_ev = user_ev[user_ev["timestamp"] < ref]

        row = build_features_for_user(user_ev, ref)
        row["user_id"] = uid
        feature_rows.append(row)

    features_df = pd.DataFrame(feature_rows)

    # Merge optional metadata
    if user_metadata_df is not None and not user_metadata_df.empty:
        meta = user_metadata_df[["user_id", "signup_date"]].copy() if "signup_date" in user_metadata_df.columns else user_metadata_df[["user_id"]].copy()
        features_df = features_df.merge(meta, on="user_id", how="left")

    # Move user_id to first column
    cols = ["user_id"] + [c for c in features_df.columns if c != "user_id"]
    features_df = features_df[cols]

    logger.info(
        "Built features for %d users (%d features).",
        len(features_df),
        len(features_df.columns) - 1,
    )
    return features_df


if __name__ == "__main__":
    from pathlib import Path

    project_root = Path(__file__).resolve().parents[2]
    raw_path = project_root / "data" / "raw" / "events.parquet"
    out_path = project_root / "data" / "processed" / "user_features.parquet"

    if not raw_path.exists():
        raise FileNotFoundError(f"No events file found at {raw_path}")

    events = pd.read_parquet(raw_path)
    logger.info("Loaded %d events from %s", len(events), raw_path)

    features = build_user_features(events, reference_date=datetime.utcnow())

    out_path.parent.mkdir(parents=True, exist_ok=True)
    features.to_parquet(out_path, index=False)
    logger.info("Saved user features (%d rows) to %s", len(features), out_path)
    print(f"Saved {len(features)} user features to {out_path}")
