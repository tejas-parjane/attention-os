"""
Synthetic data generator for a consumer app/game.

Creates realistic user behavioral data with multiple archetypes,
temporal patterns, and engagement trajectories over a configurable time window.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

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

CONTENT_CATEGORIES = ["puzzle", "action", "social", "news", "video"]
DEVICE_TYPES = ["ios", "android", "web"]
COUNTRIES = ["US", "UK", "DE", "BR", "IN", "JP"]
ACQUISITION_CHANNELS = ["organic", "paid_social", "paid_search", "referral", "viral"]

ARCHETYPE_NAMES = [
    "highly_engaged",
    "casual",
    "new_user",
    "at_risk",
    "whale",
    "ad_heavy",
    "high_value",
    "declining",
]

# Country weight distribution (US-heavy to reflect a typical western app)
_COUNTRY_WEIGHTS = [0.35, 0.12, 0.08, 0.15, 0.20, 0.10]

# Device weight distribution
_DEVICE_WEIGHTS = [0.40, 0.45, 0.15]

# Channel weight distribution
_CHANNEL_WEIGHTS = [0.30, 0.25, 0.20, 0.15, 0.10]

# ---------------------------------------------------------------------------
# Archetype parameters (NamedTuple for clarity)
# ---------------------------------------------------------------------------


class ArchetypeParams(NamedTuple):
    """Behavioral parameters that define a user archetype."""

    # how many sessions per day (mean)
    sessions_per_day_mean: float
    sessions_per_day_std: float
    # session duration in minutes (mean, std)
    session_duration_mean: float
    session_duration_std: float
    # probability of doing an action during a session
    content_view_prob: float
    game_start_prob: float
    level_complete_prob: float
    challenge_start_prob: float
    challenge_complete_prob: float
    # purchase behaviour
    purchase_prob: float
    purchase_amount_mean: float
    purchase_amount_std: float
    subscription_prob: float
    # ad behaviour
    ad_view_prob: float
    ad_views_per_session_mean: float
    # notification engagement
    notif_received_per_day: float
    notif_open_prob: float
    notif_click_prob: float
    # engagement trajectory (multiplier applied per day, <1 means declining)
    engagement_decay: float
    # level progression speed
    level_progress_speed: float
    # day offset for new users (0 = full window)
    signup_day_offset: int


ARCHETYPE_CONFIGS: dict[str, ArchetypeParams] = {
    "highly_engaged": ArchetypeParams(
        sessions_per_day_mean=2.5,
        sessions_per_day_std=0.8,
        session_duration_mean=22.0,
        session_duration_std=8.0,
        content_view_prob=0.85,
        game_start_prob=0.70,
        level_complete_prob=0.55,
        challenge_start_prob=0.50,
        challenge_complete_prob=0.40,
        purchase_prob=0.06,
        purchase_amount_mean=8.0,
        purchase_amount_std=4.0,
        subscription_prob=0.15,
        ad_view_prob=0.30,
        ad_views_per_session_mean=1.0,
        notif_received_per_day=2.5,
        notif_open_prob=0.60,
        notif_click_prob=0.35,
        engagement_decay=1.0,
        level_progress_speed=1.2,
        signup_day_offset=0,
    ),
    "casual": ArchetypeParams(
        sessions_per_day_mean=0.35,
        sessions_per_day_std=0.15,
        session_duration_mean=10.0,
        session_duration_std=5.0,
        content_view_prob=0.60,
        game_start_prob=0.40,
        level_complete_prob=0.25,
        challenge_start_prob=0.20,
        challenge_complete_prob=0.15,
        purchase_prob=0.02,
        purchase_amount_mean=4.0,
        purchase_amount_std=2.0,
        subscription_prob=0.04,
        ad_view_prob=0.50,
        ad_views_per_session_mean=1.5,
        notif_received_per_day=2.0,
        notif_open_prob=0.30,
        notif_click_prob=0.12,
        engagement_decay=1.0,
        level_progress_speed=0.6,
        signup_day_offset=0,
    ),
    "new_user": ArchetypeParams(
        sessions_per_day_mean=1.2,
        sessions_per_day_std=0.5,
        session_duration_mean=14.0,
        session_duration_std=6.0,
        content_view_prob=0.80,
        game_start_prob=0.75,
        level_complete_prob=0.50,
        challenge_start_prob=0.35,
        challenge_complete_prob=0.30,
        purchase_prob=0.03,
        purchase_amount_mean=5.0,
        purchase_amount_std=3.0,
        subscription_prob=0.05,
        ad_view_prob=0.40,
        ad_views_per_session_mean=1.2,
        notif_received_per_day=3.0,
        notif_open_prob=0.55,
        notif_click_prob=0.30,
        engagement_decay=0.97,
        level_progress_speed=1.0,
        signup_day_offset=60,
    ),
    "at_risk": ArchetypeParams(
        sessions_per_day_mean=1.8,
        sessions_per_day_std=0.6,
        session_duration_mean=18.0,
        session_duration_std=7.0,
        content_view_prob=0.70,
        game_start_prob=0.55,
        level_complete_prob=0.40,
        challenge_start_prob=0.35,
        challenge_complete_prob=0.25,
        purchase_prob=0.04,
        purchase_amount_mean=6.0,
        purchase_amount_std=3.0,
        subscription_prob=0.08,
        ad_view_prob=0.35,
        ad_views_per_session_mean=1.0,
        notif_received_per_day=2.5,
        notif_open_prob=0.40,
        notif_click_prob=0.18,
        engagement_decay=0.985,
        level_progress_speed=0.9,
        signup_day_offset=0,
    ),
    "whale": ArchetypeParams(
        sessions_per_day_mean=2.0,
        sessions_per_day_std=0.7,
        session_duration_mean=20.0,
        session_duration_std=7.0,
        content_view_prob=0.75,
        game_start_prob=0.65,
        level_complete_prob=0.50,
        challenge_start_prob=0.45,
        challenge_complete_prob=0.35,
        purchase_prob=0.25,
        purchase_amount_mean=45.0,
        purchase_amount_std=30.0,
        subscription_prob=0.40,
        ad_view_prob=0.10,
        ad_views_per_session_mean=0.3,
        notif_received_per_day=3.0,
        notif_open_prob=0.50,
        notif_click_prob=0.28,
        engagement_decay=1.0,
        level_progress_speed=1.4,
        signup_day_offset=0,
    ),
    "ad_heavy": ArchetypeParams(
        sessions_per_day_mean=1.0,
        sessions_per_day_std=0.4,
        session_duration_mean=12.0,
        session_duration_std=5.0,
        content_view_prob=0.55,
        game_start_prob=0.35,
        level_complete_prob=0.20,
        challenge_start_prob=0.15,
        challenge_complete_prob=0.10,
        purchase_prob=0.005,
        purchase_amount_mean=2.0,
        purchase_amount_std=1.0,
        subscription_prob=0.01,
        ad_view_prob=0.90,
        ad_views_per_session_mean=4.0,
        notif_received_per_day=2.0,
        notif_open_prob=0.25,
        notif_click_prob=0.10,
        engagement_decay=1.0,
        level_progress_speed=0.5,
        signup_day_offset=0,
    ),
    "high_value": ArchetypeParams(
        sessions_per_day_mean=1.5,
        sessions_per_day_std=0.5,
        session_duration_mean=16.0,
        session_duration_std=6.0,
        content_view_prob=0.70,
        game_start_prob=0.55,
        level_complete_prob=0.40,
        challenge_start_prob=0.35,
        challenge_complete_prob=0.28,
        purchase_prob=0.12,
        purchase_amount_mean=25.0,
        purchase_amount_std=15.0,
        subscription_prob=0.30,
        ad_view_prob=0.20,
        ad_views_per_session_mean=0.6,
        notif_received_per_day=2.5,
        notif_open_prob=0.50,
        notif_click_prob=0.25,
        engagement_decay=1.0,
        level_progress_speed=1.0,
        signup_day_offset=0,
    ),
    "declining": ArchetypeParams(
        sessions_per_day_mean=2.2,
        sessions_per_day_std=0.7,
        session_duration_mean=20.0,
        session_duration_std=7.0,
        content_view_prob=0.80,
        game_start_prob=0.65,
        level_complete_prob=0.45,
        challenge_start_prob=0.40,
        challenge_complete_prob=0.30,
        purchase_prob=0.05,
        purchase_amount_mean=7.0,
        purchase_amount_std=3.5,
        subscription_prob=0.08,
        ad_view_prob=0.35,
        ad_views_per_session_mean=1.0,
        notif_received_per_day=2.5,
        notif_open_prob=0.45,
        notif_click_prob=0.20,
        engagement_decay=0.975,
        level_progress_speed=0.8,
        signup_day_offset=0,
    ),
}

# Archetype distribution weights (how many users of each type)
ARCHETYPE_WEIGHTS: dict[str, float] = {
    "highly_engaged": 0.10,
    "casual": 0.25,
    "new_user": 0.12,
    "at_risk": 0.13,
    "whale": 0.05,
    "ad_heavy": 0.12,
    "high_value": 0.08,
    "declining": 0.15,
}

# ---------------------------------------------------------------------------
# Irreducible behavioural noise
# ---------------------------------------------------------------------------
# Real-world churn is only *partially* predictable from past behaviour: an
# apparently-engaged user can suddenly stop, and a quiet user can reappear.
# To model this stochasticity (and avoid the unrealistic "perfectly
# separable" datasets that would make every model score AUC = 1.0) we allow a
# random subset of users to experience an unexpected dry spell or activity
# burst in the final days of the observation window.  These final days sit
# *after* the feature-reference horizon, so past features cannot perfectly
# predict the outcome — mirroring genuine irreducible uncertainty.
RETURN_NOISE_RATE: float = 0.28   # fraction of users who get a surprise tail
TAIL_NOISE_DAYS: int = 14         # final N days perturbed (>= label window)

# ---------------------------------------------------------------------------
# Hourly activity distribution (fraction of daily sessions that start in each hour)
# Simulates typical mobile app usage curves.
# ---------------------------------------------------------------------------

_HOURLY_WEIGHTS_RAW: list[float] = [
    0.010, 0.005, 0.003, 0.002, 0.002, 0.005,  # 00-05
    0.015, 0.035, 0.055, 0.065, 0.070, 0.075,  # 06-11
    0.080, 0.070, 0.065, 0.060, 0.055, 0.060,  # 12-17
    0.070, 0.080, 0.085, 0.070, 0.045, 0.025,  # 18-23
]
_HOURLY_WEIGHTS = np.array(_HOURLY_WEIGHTS_RAW)
_HOURLY_WEIGHTS /= _HOURLY_WEIGHTS.sum()

# Day-of-week weights (0=Mon .. 6=Sun) — weekend slightly higher
_DOW_WEIGHTS = np.array([0.13, 0.13, 0.14, 0.14, 0.15, 0.16, 0.15])

# Content category base weights
_CONTENT_CATEGORY_WEIGHTS = np.array([0.30, 0.25, 0.20, 0.10, 0.15])


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------


def _hour_of_day(rng: np.random.Generator, size: int) -> np.ndarray:
    """Sample hours of day following the weighted distribution."""
    return rng.choice(24, size=size, p=_HOURLY_WEIGHTS)


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


# ---------------------------------------------------------------------------
# Core generation
# ---------------------------------------------------------------------------


def _generate_session_events(
    user_id: str,
    archetype: str,
    params: ArchetypeParams,
    start_date: datetime,
    days: int,
    rng: np.random.Generator,
    country: str,
    device: str,
    channel: str,
) -> list[dict]:
    """Generate all events for a single user across the full time window."""
    events: list[dict] = []
    current_level = 1
    total_sessions = 0

    # Daily session count trajectory
    daily_session_counts = np.maximum(
        0,
        rng.normal(params.sessions_per_day_mean, params.sessions_per_day_std, size=days),
    ).astype(int)

    # Apply engagement decay (or growth for stable archetypes)
    if params.engagement_decay != 1.0:
        decay_curve = np.array([params.engagement_decay ** d for d in range(days)])
        daily_session_counts = np.maximum(
            0,
            (daily_session_counts.astype(float) * decay_curve).astype(int),
        )

    # --- Irreducible behavioural noise in the final window ----------------
    # Independently of the archetype's signal, a random subset of users
    # experience either an unexpected dry spell (sudden churn) or an
    # unexpected activity burst (surprise return) in the tail.  Because this
    # happens after the feature-reference horizon, past features cannot
    # perfectly predict the return outcome.
    if RETURN_NOISE_RATE > 0.0 and days > TAIL_NOISE_DAYS:
        tail_start = days - TAIL_NOISE_DAYS
        if rng.random() < RETURN_NOISE_RATE:
            # surprise churn: zero out the tail (they stop using the app)
            daily_session_counts[tail_start:] = 0
        if rng.random() < RETURN_NOISE_RATE:
            # surprise return: inject a last-minute activity burst
            daily_session_counts[tail_start:] += rng.integers(0, 3, size=TAIL_NOISE_DAYS)

    # Determine signup day offset
    first_active_day = min(params.signup_day_offset, days - 1)

    # Ensure even heavily-decayed users have at least one early session,
    # otherwise they would vanish entirely with zero events.
    if daily_session_counts.sum() == 0 and days > 0:
        daily_session_counts[0] = 1

    # Pre-generate user-level preferences
    primary_content = rng.choice(CONTENT_CATEGORIES, p=_CONTENT_CATEGORY_WEIGHTS)

    subscription_active = False

    for day_offset in range(first_active_day, days):
        day_dt = start_date + timedelta(days=day_offset)
        n_sessions_today = int(daily_session_counts[day_offset])

        # New users have a welcome spike in first 3 days
        if archetype == "new_user" and day_offset - first_active_day < 3:
            n_sessions_today = max(n_sessions_today, int(rng.integers(2, 4)))

        if n_sessions_today <= 0:
            continue

        # Sample hours for today's sessions
        hours = _hour_of_day(rng, size=n_sessions_today)
        hours.sort()  # chronological order

        for hour in hours:
            session_id = f"{user_id}_s{total_sessions:05d}"
            minute = int(rng.integers(0, 60))
            second = int(rng.integers(0, 60))
            ts = day_dt.replace(hour=int(hour), minute=minute, second=second)

            session_duration = max(
                1.0,
                rng.normal(params.session_duration_mean, params.session_duration_std),
            )

            # Session start
            _session_row = dict(
                user_id=user_id,
                session_id=session_id,
                timestamp=ts,
                event_type="session_start",
                session_duration=round(session_duration, 1),
                screens_viewed=0,
                actions_completed=0,
                level=current_level,
                purchase_amount=0.0,
                ad_impressions=0,
                ad_revenue=0.0,
                notification_opened=0,
                notification_clicked=0,
                content_category=primary_content,
                device_type=device,
                country=country,
                acquisition_channel=channel,
            )
            events.append(_session_row)

            # App open fires just before a session start (opening the app)
            events.append(
                dict(
                    _session_row,
                    event_type="app_open",
                    timestamp=ts - timedelta(seconds=int(rng.integers(1, 10))),
                )
            )

            # --- In-session events ---
            screens_viewed = 0
            actions_completed = 0
            ad_impressions = 0
            ad_revenue = 0.0
            purchase_amount = 0.0

            # Content views
            if rng.random() < params.content_view_prob:
                n_views = rng.integers(1, 6)
                screens_viewed += int(n_views)
                for _ in range(n_views):
                    cat = (
                        primary_content
                        if rng.random() < 0.6
                        else rng.choice(CONTENT_CATEGORIES, p=_CONTENT_CATEGORY_WEIGHTS)
                    )
                    events.append(
                        dict(
                            _session_row,
                            event_type="content_view",
                            timestamp=ts + timedelta(seconds=int(rng.integers(10, 300))),
                            content_category=cat,
                            screens_viewed=0,
                            actions_completed=0,
                        )
                    )

            # Game start
            if rng.random() < params.game_start_prob:
                actions_completed += 1
                events.append(
                    dict(
                        _session_row,
                        event_type="game_start",
                        timestamp=ts + timedelta(seconds=int(rng.integers(20, 600))),
                        screens_viewed=0,
                        actions_completed=1,
                    )
                )

                # Level complete
                if rng.random() < params.level_complete_prob:
                    actions_completed += 1
                    current_level += 1
                    events.append(
                        dict(
                            _session_row,
                            event_type="level_complete",
                            timestamp=ts + timedelta(seconds=int(rng.integers(60, 900))),
                            level=current_level,
                            screens_viewed=0,
                            actions_completed=1,
                        )
                    )

            # Challenge events
            if rng.random() < params.challenge_start_prob:
                actions_completed += 1
                events.append(
                    dict(
                        _session_row,
                        event_type="challenge_started",
                        timestamp=ts + timedelta(seconds=int(rng.integers(30, 400))),
                        screens_viewed=0,
                        actions_completed=1,
                    )
                )
                if rng.random() < params.challenge_complete_prob:
                    actions_completed += 1
                    events.append(
                        dict(
                            _session_row,
                            event_type="challenge_completed",
                            timestamp=ts + timedelta(seconds=int(rng.integers(90, 600))),
                            screens_viewed=0,
                            actions_completed=1,
                        )
                    )

            # Ad views
            if rng.random() < params.ad_view_prob:
                n_ads = max(1, int(rng.poisson(params.ad_views_per_session_mean)))
                ad_impressions = n_ads
                ad_revenue = round(n_ads * rng.uniform(0.01, 0.08), 4)
                for _ in range(n_ads):
                    events.append(
                        dict(
                            _session_row,
                            event_type="ad_view",
                            timestamp=ts + timedelta(seconds=int(rng.integers(15, 500))),
                            ad_impressions=1,
                            ad_revenue=round(rng.uniform(0.01, 0.08), 4),
                            screens_viewed=0,
                            actions_completed=0,
                        )
                    )

            # Purchase
            if rng.random() < params.purchase_prob:
                amt = round(
                    max(
                        0.99,
                        rng.normal(params.purchase_amount_mean, params.purchase_amount_std),
                    ),
                    2,
                )
                purchase_amount = amt
                events.append(
                    dict(
                        _session_row,
                        event_type="purchase",
                        timestamp=ts + timedelta(seconds=int(rng.integers(30, 700))),
                        purchase_amount=amt,
                        screens_viewed=0,
                        actions_completed=0,
                    )
                )

            # Subscription
            if not subscription_active and rng.random() < params.subscription_prob:
                subscription_active = True
                events.append(
                    dict(
                        _session_row,
                        event_type="subscription_started",
                        timestamp=ts + timedelta(seconds=int(rng.integers(10, 200))),
                        screens_viewed=0,
                        actions_completed=0,
                    )
                )

            # Rare subscription cancellation (5% chance if subscribed)
            if subscription_active and rng.random() < 0.005:
                subscription_active = False
                events.append(
                    dict(
                        _session_row,
                        event_type="subscription_cancelled",
                        timestamp=ts + timedelta(seconds=int(rng.integers(10, 200))),
                        screens_viewed=0,
                        actions_completed=0,
                    )
                )

            # Update session row with aggregates
            _session_row["screens_viewed"] = screens_viewed
            _session_row["actions_completed"] = actions_completed
            _session_row["ad_impressions"] = ad_impressions
            _session_row["ad_revenue"] = ad_revenue
            _session_row["purchase_amount"] = purchase_amount

            # Session end
            end_ts = ts + timedelta(minutes=session_duration)
            events.append(
                dict(
                    _session_row,
                    event_type="session_end",
                    timestamp=end_ts,
                    screens_viewed=0,
                    actions_completed=0,
                )
            )

            total_sessions += 1

        # --- Notifications (outside session context) ---
        n_notifs = int(rng.poisson(params.notif_received_per_day))
        for _ in range(n_notifs):
            notif_hour = int(rng.choice(24, p=_HOURLY_WEIGHTS))
            notif_ts = day_dt.replace(
                hour=notif_hour,
                minute=int(rng.integers(0, 60)),
                second=int(rng.integers(0, 60)),
            )
            events.append(
                dict(
                    user_id=user_id,
                    session_id="",
                    timestamp=notif_ts,
                    event_type="notification_received",
                    session_duration=0.0,
                    screens_viewed=0,
                    actions_completed=0,
                    level=current_level,
                    purchase_amount=0.0,
                    ad_impressions=0,
                    ad_revenue=0.0,
                    notification_opened=0,
                    notification_clicked=0,
                    content_category=primary_content,
                    device_type=device,
                    country=country,
                    acquisition_channel=channel,
                )
            )

            if rng.random() < params.notif_open_prob:
                events.append(
                    dict(
                        user_id=user_id,
                        session_id="",
                        timestamp=notif_ts + timedelta(seconds=int(rng.integers(5, 300))),
                        event_type="notification_opened",
                        session_duration=0.0,
                        screens_viewed=0,
                        actions_completed=0,
                        level=current_level,
                        purchase_amount=0.0,
                        ad_impressions=0,
                        ad_revenue=0.0,
                        notification_opened=1,
                        notification_clicked=0,
                        content_category=primary_content,
                        device_type=device,
                        country=country,
                        acquisition_channel=channel,
                    )
                )

                if rng.random() < params.notif_click_prob:
                    events.append(
                        dict(
                            user_id=user_id,
                            session_id="",
                            timestamp=notif_ts + timedelta(seconds=int(rng.integers(10, 400))),
                            event_type="notification_clicked",
                            session_duration=0.0,
                            screens_viewed=0,
                            actions_completed=0,
                            level=current_level,
                            purchase_amount=0.0,
                            ad_impressions=0,
                            ad_revenue=0.0,
                            notification_opened=0,
                            notification_clicked=1,
                            content_category=primary_content,
                            device_type=device,
                            country=country,
                            acquisition_channel=channel,
                        )
                    )

    return events


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def generate_events(
    n_users: int = 500,
    days: int = 90,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate synthetic app/game event data for *n_users* over *days* days.

    Returns a DataFrame with one row per event, sorted by timestamp.
    """
    rng = np.random.default_rng(seed)
    start_date = datetime(2026, 1, 1)

    logger.info(
        "Generating events for %d users over %d days (seed=%d)", n_users, days, seed
    )

    # Assign archetypes to users
    archetype_names = list(ARCHETYPE_WEIGHTS.keys())
    archetype_probs = np.array(list(ARCHETYPE_WEIGHTS.values()))
    archetype_probs /= archetype_probs.sum()
    user_archetypes = rng.choice(archetype_names, size=n_users, p=archetype_probs)

    all_events: list[dict] = []

    for i, archetype in enumerate(user_archetypes):
        user_id = f"user_{i:04d}"
        params = ARCHETYPE_CONFIGS[archetype]
        country = rng.choice(COUNTRIES, p=_COUNTRY_WEIGHTS)
        device = rng.choice(DEVICE_TYPES, p=_DEVICE_WEIGHTS)
        channel = rng.choice(ACQUISITION_CHANNELS, p=_CHANNEL_WEIGHTS)

        user_events = _generate_session_events(
            user_id=user_id,
            archetype=archetype,
            params=params,
            start_date=start_date,
            days=days,
            rng=rng,
            country=country,
            device=device,
            channel=channel,
        )
        all_events.extend(user_events)

        if (i + 1) % 100 == 0:
            logger.info("  … generated events for %d / %d users", i + 1, n_users)

    logger.info("Total events generated: %d", len(all_events))

    df = pd.DataFrame(all_events)

    # Ensure consistent column order
    column_order = [
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
    df = df[column_order]
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)

    logger.info(
        "Event generation complete — %d events, %d unique users",
        len(df),
        df["user_id"].nunique(),
    )
    return df


def generate_user_metadata(events_df: pd.DataFrame) -> pd.DataFrame:
    """Derive a per-user metadata row from the generated events DataFrame.

    Includes first/last seen timestamps, total sessions, total spend,
    dominant archetype heuristics, and device/country info.
    """
    logger.info("Generating user metadata for %d users", events_df["user_id"].nunique())

    user_groups = events_df.groupby("user_id", sort=False)

    def _agg(user_id: str, g: pd.DataFrame) -> dict[str, object]:
        timestamps = g["timestamp"]
        first_seen = timestamps.min()
        last_seen = timestamps.max()
        total_days_active = (last_seen - first_seen).days + 1

        session_starts = g[g["event_type"] == "session_start"]
        total_sessions = len(session_starts)
        avg_session_duration = (
            round(session_starts["session_duration"].mean(), 1)
            if total_sessions > 0
            else 0.0
        )

        purchases = g[g["event_type"] == "purchase"]
        total_spend = round(float(purchases["purchase_amount"].sum()), 2)
        purchase_count = len(purchases)

        ads = g[g["event_type"] == "ad_view"]
        total_ad_views = len(ads)
        total_ad_revenue = round(float(ads["ad_revenue"].sum()), 4)

        sub_started = g[g["event_type"] == "subscription_started"]
        sub_cancelled = g[g["event_type"] == "subscription_cancelled"]
        has_subscription = len(sub_started) > len(sub_cancelled)

        content_views = g[g["event_type"] == "content_view"]
        top_content = (
            content_views["content_category"].mode().iloc[0]
            if len(content_views) > 0
            else "unknown"
        )

        max_level = int(g["level"].max())

        # Acquisition meta (stable per user)
        country = g["country"].iloc[0]
        device = g["device_type"].iloc[0]
        channel = g["acquisition_channel"].iloc[0]

        # Heuristic: derive a simplified archetype label
        ltv = total_spend + total_ad_revenue
        sessions_per_day = total_sessions / max(total_days_active, 1)

        if sessions_per_day > 2.0 and total_spend > 50:
            archetype_label = "whale"
        elif sessions_per_day > 1.8 and total_days_active > 60:
            archetype_label = "highly_engaged"
        elif total_spend > 30 or has_subscription:
            archetype_label = "high_value"
        elif sessions_per_day < 0.05:
            archetype_label = "declining"
        elif total_ad_views > 200:
            archetype_label = "ad_heavy"
        elif total_days_active < 20 and sessions_per_day > 0.5:
            archetype_label = "new_user"
        elif sessions_per_day < 0.5:
            archetype_label = "casual"
        else:
            archetype_label = "at_risk"

        return dict(
            user_id=user_id,
            first_seen=first_seen,
            last_seen=last_seen,
            total_days_active=total_days_active,
            total_sessions=total_sessions,
            avg_session_duration=avg_session_duration,
            total_spend=total_spend,
            purchase_count=purchase_count,
            total_ad_views=total_ad_views,
            total_ad_revenue=total_ad_revenue,
            has_subscription=has_subscription,
            top_content_category=top_content,
            max_level=max_level,
            country=country,
            device_type=device,
            acquisition_channel=channel,
            archetype_label=archetype_label,
            estimated_ltv=round(ltv, 4),
        )

    rows: list[dict] = []
    for user_id, group in user_groups:
        rows.append(_agg(user_id, group))
    meta_df = pd.DataFrame(rows)

    logger.info("User metadata complete — %d rows", len(meta_df))
    return meta_df


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Generate and save synthetic data to disk."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )

    raw_dir = Path(__file__).resolve().parents[2] / "data" / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    parquet_path = raw_dir / "events.parquet"
    csv_path = raw_dir / "users.csv"

    events_df = generate_events(n_users=500, days=90, seed=42)
    user_meta_df = generate_user_metadata(events_df)

    events_df.to_parquet(parquet_path, index=False)
    logger.info("Saved events → %s (%d rows)", parquet_path, len(events_df))

    user_meta_df.to_csv(csv_path, index=False)
    logger.info("Saved user metadata → %s (%d rows)", csv_path, len(user_meta_df))

    # Quick summary stats
    print("\n=== Data Summary ===")
    print(f"Events : {len(events_df):,} rows")
    print(f"Users  : {events_df['user_id'].nunique():,}")
    print(f"Date range: {events_df['timestamp'].min()} -> {events_df['timestamp'].max()}")
    print(f"\nEvent type breakdown:")
    print(events_df["event_type"].value_counts().to_string())
    print(f"\nArchetype distribution (metadata):")
    print(user_meta_df["archetype_label"].value_counts().to_string())
    print(f"\nDevice distribution:")
    print(events_df["device_type"].value_counts().to_string())
    print(f"\nCountry distribution:")
    print(events_df["country"].value_counts().to_string())


if __name__ == "__main__":
    main()
