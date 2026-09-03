"""Generate interactive Plotly chart HTML files for the landing page.

Each chart is written as a self-contained HTML file (embeds its own JS),
so the landing page works as a fully-static Vercel site.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import plotly.express as px
import plotly.io as pio

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
RAW = ROOT / "data" / "raw"
OUT = ROOT / "landing" / "charts"
OUT.mkdir(parents=True, exist_ok=True)


def _theme(fig):
    fig.update_layout(
        template="plotly_white",
        font=dict(family="Segoe UI, Arial, sans-serif", size=13),
        margin=dict(l=40, r=20, t=50, b=40),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    return fig


def write(fig, name: str) -> None:
    pio.write_html(
        _theme(fig),
        file=str(OUT / f"{name}.html"),
        include_plotlyjs="cdn",
        full_html=False,
        config={"displayModeBar": False},
    )
    print(f"wrote {name}.html")


def main() -> None:
    users = pd.read_csv(RAW / "users.csv")
    events = pd.read_parquet(RAW / "events.parquet")
    predictions = pd.read_parquet(PROC / "predictions.parquet")
    recommendations = pd.read_parquet(PROC / "recommendations.parquet")
    features = pd.read_parquet(PROC / "user_features.parquet")
    state = pd.read_parquet(PROC / "user_state.parquet")
    exp = pd.read_parquet(PROC / "experiment_results.parquet")

    # 1. Daily active users
    ev = events.copy()
    ev["ts"] = pd.to_datetime(ev["timestamp"])
    daily = ev.set_index("ts").resample("D")["user_id"].nunique().reset_index()
    daily.columns = ["date", "active_users"]
    write(px.line(daily, x="date", y="active_users", title="Daily Active Users"), "daily_active")

    # 2. Engagement mix (donut)
    eng = state["engagement"].value_counts().reset_index()
    eng.columns = ["engagement", "count"]
    write(px.pie(eng, names="engagement", values="count", hole=0.55, title="Engagement Mix"), "engagement_mix")

    # 3. Model performance comparison (retention + monetization)
    ret = pd.read_csv(ROOT / "data" / "models" / "retention" / "metrics.csv").set_index("model")
    mon = pd.read_csv(ROOT / "data" / "models" / "monetization" / "metrics.csv").set_index("model")
    blocks = []
    for task, df in [("Retention", ret), ("Monetization", mon)]:
        t = df[["roc_auc", "pr_auc", "f1", "brier"]].reset_index().melt(id_vars="model")
        t["task"] = task
        blocks.append(t)
    mdf = pd.concat(blocks)
    write(px.bar(
        mdf, x="model", y="value", color="variable", facet_col="task", barmode="group",
        title="Model Comparison (test set)", labels={"value": "Score"}),
        "model_performance")

    # 4. Action distribution
    act = recommendations["action"].value_counts().reset_index()
    act.columns = ["action", "count"]
    write(px.bar(act, x="action", y="count", title="Recommended Action Distribution", labels={"count": "Users"}), "action_distribution")

    # 5. Experiment: mean outcome by variant with CI
    stats = exp.groupby("variant")["outcome"].agg(["mean", "std", "count"]).reset_index()
    stats["ci"] = 1.96 * stats["std"] / stats["count"] ** 0.5
    write(px.bar(
        stats, x="variant", y="mean", error_y="ci",
        title="Experiment: Mean Outcome by Variant (±95% CI)"), "experiment")

    # 6. Predictions distribution (retention risk)
    write(px.histogram(
        predictions, x="churn_probability", nbins=30,
        title="Churn-Risk Distribution (retention model)"), "churn_risk")

    print("Done.")


if __name__ == "__main__":
    main()
