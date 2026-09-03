"""Run the decision engine for every user and save recommendations."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from src.decision_engine.engine import UserState, build_user_state, decide

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    features_path = ROOT / "data" / "processed" / "user_features.parquet"
    predictions_path = ROOT / "data" / "processed" / "predictions.parquet"
    out_path = ROOT / "data" / "processed" / "recommendations.parquet"

    if not features_path.exists():
        logger.error("Missing %s — run `python scripts/build_features.py` first.", features_path)
        sys.exit(1)
    if not predictions_path.exists():
        logger.error("Missing %s — run `python scripts/generate_predictions.py` first.", predictions_path)
        sys.exit(1)

    features_df = pd.read_parquet(features_path)
    pred_df = pd.read_parquet(predictions_path)
    logger.info("Loaded features for %d users, predictions for %d users", len(features_df), len(pred_df))

    # Merge features and predictions on user_id
    merged = features_df.merge(pred_df, on="user_id", how="inner")
    logger.info("Merged: %d users with both features and predictions", len(merged))

    recommendations: list[dict] = []
    for _, row in merged.iterrows():
        feat = row.to_dict()

        # Map feature columns to the decision engine's UserState fields
        days_since_last = feat.get("days_since_last_session", 0.0)
        sessions_7d = feat.get("sessions_7d", 0)
        active_days_30d = feat.get("days_active_30d", 0.0)
        notif_open_rate = feat.get("notification_open_rate", 0.0)
        revenue_30d = feat.get("revenue_30d", 0.0)
        total_ltv = feat.get("total_ltv", 0.0)

        # Value segment derived from revenue
        if revenue_30d < 5.0:
            value_segment = "low"
        elif revenue_30d < 30.0:
            value_segment = "medium"
        else:
            value_segment = "high"

        predictions = {
            "retention_probability": feat.get("retention_probability", 0.5),
            "churn_probability": feat.get("churn_probability", 0.5),
            "purchase_probability": feat.get("purchase_probability", 0.0),
            "predicted_ltv": feat.get("predicted_ltv", 0.0),
            "engagement_trend": feat.get("engagement_change_7d", 0.0),
            "days_since_last_session": int(days_since_last) if pd.notna(days_since_last) else 0,
            "notification_open_rate": notif_open_rate,
            "revenue_30d": revenue_30d,
            "sessions_7d": int(sessions_7d),
            "active_days_30d": int(active_days_30d),
            "value_segment": value_segment,
        }

        state = build_user_state({"user_id": feat["user_id"]}, predictions)
        decision = decide(state, business_objective="retention")

        recommendations.append({
            "user_id": feat["user_id"],
            "action": decision.action,
            "expected_value": decision.expected_value,
            "confidence": decision.confidence,
            "objective": decision.objective,
            "reasons": "; ".join(decision.reasons),
            "guardrails": "; ".join(decision.guardrails) if decision.guardrails else "",
        })

    rec_df = pd.DataFrame(recommendations)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rec_df.to_parquet(out_path, index=False)
    logger.info("Saved recommendations -> %s (%d rows)", out_path, len(rec_df))

    print("\n=== Action Distribution ===")
    dist = rec_df["action"].value_counts()
    total = len(rec_df)
    for action, count in dist.items():
        pct = count / total * 100
        print(f"  {action:<30} {count:>5}  ({pct:.1f}%)")

    print(f"\nTotal users: {total}")
    print(f"Mean expected value: {rec_df['expected_value'].mean():.4f}")
    print(f"Mean confidence: {rec_df['confidence'].mean():.4f}")


if __name__ == "__main__":
    main()
