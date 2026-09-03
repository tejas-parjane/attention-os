"""Export per-user state for the interactive browser demo.

Produces docs/data/attentionos.json used by the static GitHub Pages demo.
Each record contains the exact UserState fields the decision engine needs
plus the precomputed Python decision (used to cross-check the JS port).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from src.decision_engine.engine import build_user_state, decide

ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    features = pd.read_parquet(ROOT / "data" / "processed" / "user_features.parquet")
    pred = pd.read_parquet(ROOT / "data" / "processed" / "predictions.parquet")
    state_df = pd.read_parquet(ROOT / "data" / "processed" / "user_state.parquet")[
        ["user_id", "engagement", "recent_sessions"]
    ]
    merged = features.merge(pred, on="user_id", how="inner").merge(state_df, on="user_id", how="left")

    users = []
    for _, row in merged.iterrows():
        feat = row.to_dict()
        days_since_last = feat.get("days_since_last_session", 0.0)
        sessions_7d = feat.get("sessions_7d", 0)
        active_days_30d = feat.get("days_active_30d", 0.0)
        notif_open_rate = feat.get("notification_open_rate", 0.0)
        revenue_30d = feat.get("revenue_30d", 0.0)

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

        users.append({
            "user_id": feat["user_id"],
            "state": state.model_dump(),
            "python_decision": {
                "action": decision.action,
                "expected_value": decision.expected_value,
                "objective": decision.objective,
                "confidence": decision.confidence,
                "reasons": decision.reasons,
                "guardrails": decision.guardrails,
            },
            "extra": {
                "engagement": str(feat.get("engagement", "")),
                "recent_sessions": int(feat.get("recent_sessions", 0)),
                "purchase_count_30d": int(feat.get("purchase_count_30d", 0)),
                "total_ltv": float(feat.get("total_ltv", 0.0)),
                "content_diversity": float(feat.get("content_diversity", 0.0)),
                "new_user_flag": int(feat.get("new_user_flag", 0)),
            },
        })

    exp = pd.read_parquet(ROOT / "data" / "processed" / "experiment_results.parquet")
    exp_rows = exp[["user_id", "variant", "outcome"]].to_dict(orient="records")
    exp_stats = (
        exp.groupby("variant")["outcome"]
        .agg(["mean", "std", "count"])
        .reset_index()
        .to_dict(orient="records")
    )

    out = {
        "meta": {
            "description": "AttentionOS interactive demo dataset (500 users).",
            "count": len(users),
            "generated_by": "scripts/export_demo_data.py",
            "engine": "src/decision_engine/engine.py",
        },
        "users": users,
        "experiment": {
            "experiment_id": "exp_2026_attn_v1",
            "rows": exp_rows,
            "stats": exp_stats,
        },
    }

    out_dir = ROOT / "docs" / "data"
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / "attentionos.json"
    target.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(f"Wrote {target} ({target.stat().st_size/1024:.0f} KB, {len(users)} users)")


if __name__ == "__main__":
    main()
