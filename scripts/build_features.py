"""Build user features from raw events and save to data/processed/."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from src.features.engine import build_user_features

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    events_path = ROOT / "data" / "raw" / "events.parquet"
    users_path = ROOT / "data" / "raw" / "users.csv"
    features_path = ROOT / "data" / "processed" / "user_features.parquet"
    state_path = ROOT / "data" / "processed" / "user_state.parquet"

    if not events_path.exists():
        logger.error("Missing %s — run `python scripts/generate_data.py` first.", events_path)
        sys.exit(1)
    if not users_path.exists():
        logger.error("Missing %s — run `python scripts/generate_data.py` first.", users_path)
        sys.exit(1)

    logger.info("Loading events from %s", events_path)
    events_df = pd.read_parquet(events_path)

    logger.info("Loading user metadata from %s", users_path)
    users_df = pd.read_csv(users_path)

    # Build features as of a reference date strictly *before* the label
    # window used for training (label window = reference_date -> +7 days).
    # This prevents any look-ahead leakage: features never see the outcome
    # they are asked to predict.
    max_ts = pd.to_datetime(events_df["timestamp"]).max()
    feature_cutoff = max_ts - pd.Timedelta(days=14)
    logger.info("Feature reference date (cut-off): %s", feature_cutoff)

    logger.info("Building user features …")
    features_df = build_user_features(
        events_df,
        user_metadata_df=users_df,
        reference_date=feature_cutoff,
    )

    # Fill residual missing values. Single-session users have no
    # "second last session"; impute with their last-session recency so the
    # value stays semantically meaningful rather than a default 0.
    if "days_since_second_last_session" in features_df.columns:
        features_df["days_since_second_last_session"] = features_df[
            "days_since_second_last_session"
        ].fillna(features_df["days_since_last_session"])
    features_df = features_df.fillna(0.0)

    logger.info("Features built: %d users, %d feature columns", len(features_df), len(features_df.columns) - 1)

    # Derive structured user state from the computed features
    state_rows: list[dict] = []
    for _, row in features_df.iterrows():
        uid = row["user_id"]
        engagement_change = float(row.get("engagement_change_7d", 0.0))
        duration_trend = float(row.get("session_duration_trend", 0.0))
        revenue_30d = float(row.get("revenue_30d", 0.0))
        notif_open_rate = float(row.get("notification_open_rate", 0.0))
        days_active_30d = int(row.get("days_active_30d", 0))
        sessions_7d = int(row.get("sessions_7d", 0))
        days_since_last = row.get("days_since_last_session", 0.0)
        total_ltv = float(row.get("total_ltv", 0.0))
        purchase_count = int(row.get("purchase_count_total", 0))

        # Engagement label
        combined = (engagement_change + duration_trend) / 2.0
        if combined < -0.2:
            engagement = "declining"
        elif combined > 0.2:
            engagement = "improving"
        else:
            engagement = "stable"

        # Notification response
        if notif_open_rate <= 0.0:
            notif_response = "none"
        elif notif_open_rate > 0.2:
            notif_response = "positive"
        elif notif_open_rate < 0.05:
            notif_response = "negative"
        else:
            notif_response = "neutral"

        # Value segment
        if revenue_30d < 5.0:
            value_segment = "low"
        elif revenue_30d < 30.0:
            value_segment = "medium"
        else:
            value_segment = "high"

        state_rows.append({
            "user_id": uid,
            "engagement": engagement,
            "retention_risk": 0.0,
            "purchase_propensity": 0.0,
            "predicted_ltv": total_ltv,
            "recent_sessions": sessions_7d,
            "days_since_last_session": int(days_since_last) if pd.notna(days_since_last) else 0,
            "notification_response": notif_response,
            "value_segment": value_segment,
            "revenue_30d": revenue_30d,
            "sessions_7d": sessions_7d,
            "engagement_trend": engagement_change,
        })

    user_state_df = pd.DataFrame(state_rows)

    features_path.parent.mkdir(parents=True, exist_ok=True)
    features_df.to_parquet(features_path, index=False)
    user_state_df.to_parquet(state_path, index=False)

    logger.info("Saved user features   -> %s", features_path)
    logger.info("Saved user state      -> %s", state_path)

    print("\n=== Feature Summary ===")
    print(f"Users : {len(features_df)}")
    print(f"Columns ({len(features_df.columns) - 1} features):")
    for col in features_df.columns:
        if col == "user_id":
            continue
        dtype = features_df[col].dtype
        non_null = int(features_df[col].notna().sum())
        print(f"  {col:<35} {str(dtype):<12} {non_null:>5}/{len(features_df)} non-null")

    print("\nEngagement distribution:")
    print(user_state_df["engagement"].value_counts().to_string())
    print("\nNotification response distribution:")
    print(user_state_df["notification_response"].value_counts().to_string())
    print("\nValue segment distribution:")
    print(user_state_df["value_segment"].value_counts().to_string())


if __name__ == "__main__":
    main()
