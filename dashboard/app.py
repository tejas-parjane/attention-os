"""Streamlit dashboard for AttentionOS.

Connects to the FastAPI backend (default ``http://localhost:8000``,
configurable via the ``API_BASE_URL`` environment variable or the
right-hand settings controls).  When the API is unreachable it falls
back to reading the same parquet files from ``data/processed`` and
``data/raw`` so the dashboard still renders with locally available data.

Pages
-----
1. Overview            : KPI cards + engagement / risk / revenue charts.
2. User Explorer       : search a user and inspect their full state.
3. Decision Explanation: candidate actions, chosen action, guardrails, AI text.
4. Experiment Dashboard: control vs. variant results.
5. Model Performance   : offline model metrics from ``data/models``/``data/processed``.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]
PROCESSED_DIR: Path = PROJECT_ROOT / "data" / "processed"
RAW_DIR: Path = PROJECT_ROOT / "data" / "raw"
MODELS_DIR: Path = PROJECT_ROOT / "data" / "models"

DEFAULT_API_BASE: str = "http://localhost:8000"
API_TIMEOUT: float = 5.0

PAGES: list[str] = [
    "Overview",
    "User Explorer",
    "Decision Explanation",
    "Experiment Dashboard",
    "Model Performance",
]


# ---------------------------------------------------------------------------
# API / data-access helpers
# ---------------------------------------------------------------------------


def _api_base() -> str:
    """Return the configured API base URL (env var or session state)."""
    return st.session_state.get(
        "api_base", os.environ.get("API_BASE_URL", DEFAULT_API_BASE)
    )


def _api_get(path: str) -> dict[str, Any] | list[Any] | None:
    """GET *path* from the API; return parsed JSON or ``None`` on failure."""
    import httpx

    base = _api_base().rstrip("/")
    try:
        with httpx.Client(timeout=API_TIMEOUT) as client:
            resp = client.get(f"{base}{path}")
            resp.raise_for_status()
            return resp.json()
    except Exception:  # noqa: BLE001
        return None


def _api_post(path: str) -> dict[str, Any] | None:
    """POST *path* to the API; return parsed JSON or ``None`` on failure."""
    import httpx

    base = _api_base().rstrip("/")
    try:
        with httpx.Client(timeout=API_TIMEOUT) as client:
            resp = client.post(f"{base}{path}")
            resp.raise_for_status()
            return resp.json()
    except Exception:  # noqa: BLE001
        return None


def _api_available() -> bool:
    """Whether the API backend is reachable."""
    return _api_get("/health") is not None


def _load_parquet(stem: str, folder: Path) -> pd.DataFrame | None:
    """Load ``<folder>/<stem>.parquet`` or return ``None`` when missing."""
    path = folder / f"{stem}.parquet"
    if path.exists():
        try:
            return pd.read_parquet(path)
        except Exception:  # noqa: BLE001
            return None
    return None


def _load_csv(stem: str, folder: Path) -> pd.DataFrame | None:
    """Load ``<folder>/<stem>.csv`` or return ``None`` when missing."""
    path = folder / f"{stem}.csv"
    if path.exists():
        try:
            return pd.read_csv(path)
        except Exception:  # noqa: BLE001
            return None
    return None


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------


def _render_sidebar() -> str:
    """Render the sidebar navigation and settings; return the selected page."""
    with st.sidebar:
        st.title("AttentionOS")
        st.caption("AI User Retention & Monetization Engine")
        st.divider()
        page = st.radio("Navigate", PAGES, label_visibility="collapsed")
        st.divider()
        st.subheader("Settings")
        api_url = st.text_input(
            "API Base URL",
            value=_api_base(),
            help="Full URL of the FastAPI backend",
        )
        st.session_state["api_base"] = api_url
        if _api_available():
            st.success("API connected")
        else:
            st.warning("API unreachable — using local files")
    return page


# ---------------------------------------------------------------------------
# 1. Overview
# ---------------------------------------------------------------------------


def _overview_from_api() -> dict[str, Any] | None:
    """Pull KPI payload from the /metrics/overview endpoint."""
    return _api_get("/metrics/overview")  # type: ignore[return-value]


def _overview_from_files() -> dict[str, Any]:
    """Compute KPI payload from local data files (API-unavailable fallback)."""
    users = _load_csv("users", RAW_DIR)
    predictions = _load_parquet("predictions", PROCESSED_DIR)
    recommendations = _load_parquet("recommendations", PROCESSED_DIR)

    total_users = int(len(users)) if users is not None else 0
    active_users = 0
    total_revenue = 0.0
    if users is not None:
        if "total_sessions" in users.columns:
            active_users = int((users["total_sessions"] > 0).sum())
        if "total_spend" in users.columns:
            total_revenue = float(users["total_spend"].sum())
        elif "revenue" in users.columns:
            total_revenue = float(users["revenue"].sum())

    at_risk = 0
    risk_avg = 0.0
    if predictions is not None:
        risk_col = "churn_probability" if "churn_probability" in predictions.columns else "retention_risk"
        if risk_col in predictions.columns:
            risk_avg = float(predictions[risk_col].mean())
            at_risk = int((predictions[risk_col] > 0.7).sum())

    conversion = 0.0
    if recommendations is not None and "action" in recommendations.columns:
        total_recs = len(recommendations)
        intervened = int((recommendations["action"] != "NO_ACTION").sum())
        conversion = intervened / total_recs if total_recs else 0.0

    return {
        "total_users": total_users,
        "active_users": active_users,
        "at_risk_users": at_risk,
        "retention_risk_avg": risk_avg,
        "total_revenue": total_revenue,
        "conversion_rate": conversion,
    }


def render_overview() -> None:
    """Render the Overview page: KPI cards plus charts."""
    st.header("AttentionOS Overview")

    payload = _overview_from_api() or _overview_from_files()
    if not payload:
        st.warning("No overview data available.")
        return

    k1, k2, k3, k4, k5, k6 = st.columns(6)
    k1.metric("Total Users", f"{payload.get('total_users', 0):,}")
    k2.metric("Active Users", f"{payload.get('active_users', 0):,}")
    k3.metric("At-Risk Users", f"{payload.get('at_risk_users', 0):,}")
    k4.metric("Avg Retention Risk", f"{payload.get('retention_risk_avg', 0.0):.1%}")
    k5.metric("Revenue", f"${payload.get('total_revenue', 0.0):,.0f}")
    k6.metric("Conversion Rate", f"{payload.get('conversion_rate', 0.0):.1%}")

    st.divider()
    st.subheader("Charts")

    events_df = _load_parquet("events", RAW_DIR)
    if events_df is None:
        events_df = _load_parquet("events", PROCESSED_DIR)
    predictions_df = _load_parquet("predictions", PROCESSED_DIR)
    users_df = _load_csv("users", RAW_DIR)

    col_a, col_b = st.columns(2)

    with col_a:
        if events_df is not None and "timestamp" in events_df.columns:
            ev = events_df.copy()
            ev["timestamp"] = pd.to_datetime(ev["timestamp"])
            daily = ev.set_index("timestamp").resample("D")["user_id"].nunique().reset_index()
            daily.columns = ["date", "active_users"]
            st.plotly_chart(
                px.line(daily, x="date", y="active_users",
                        title="Daily Active Users"),
                use_container_width=True,
            )
        else:
            st.info("No engagement time-series data available.")

    with col_b:
        risk_col = "churn_probability" if (
            predictions_df is not None and "churn_probability" in predictions_df.columns
        ) else "retention_risk"
        if predictions_df is not None and risk_col in predictions_df.columns:
            st.plotly_chart(
                px.histogram(
                    predictions_df, x=risk_col, nbins=30,
                    title="Retention Risk Distribution",
                    color_discrete_sequence=["#EF553B"],
                ),
                use_container_width=True,
            )
        else:
            st.info("No retention-risk data available.")

    if users_df is not None:
        rev_col = "total_spend" if "total_spend" in users_df.columns else ("revenue" if "revenue" in users_df.columns else None)
        if rev_col:
            st.plotly_chart(
                px.histogram(
                    users_df, x=rev_col, nbins=40,
                    title="Revenue Distribution",
                    color_discrete_sequence=["#00CC96"],
                ),
                use_container_width=True,
            )


# ---------------------------------------------------------------------------
# 2. User Explorer
# ---------------------------------------------------------------------------


def _get_user_ids() -> list[str]:
    """Return the sorted list of user IDs from API or local files."""
    result = _api_get("/users/")
    if isinstance(result, list):
        return sorted(str(u.get("user_id", "")) for u in result if isinstance(u, dict))
    users_csv = _load_csv("users", RAW_DIR)
    if users_csv is not None and "user_id" in users_csv.columns:
        return sorted(users_csv["user_id"].astype(str).unique().tolist())
    return []


def _get_user_profile(user_id: str) -> dict[str, Any] | None:
    """Fetch a user profile from the API or local metadata."""
    via_api = _api_get(f"/users/{user_id}")
    if via_api is not None:
        return via_api  # type: ignore[return-value]
    users_csv = _load_csv("users", RAW_DIR)
    if users_csv is not None and "user_id" in users_csv.columns:
        row = users_csv[users_csv["user_id"] == user_id]
        if not row.empty:
            return row.iloc[0].to_dict()
    return None


def _get_user_events(user_id: str, limit: int = 20) -> pd.DataFrame | None:
    """Return recent events for *user_id* from local parquet (fallback)."""
    events = _load_parquet("events", RAW_DIR)
    if events is None or "user_id" not in events.columns:
        return None
    sub = events[events["user_id"] == user_id].copy()
    if sub.empty:
        return None
    sub["timestamp"] = pd.to_datetime(sub["timestamp"])
    return sub.sort_values("timestamp", ascending=False).head(limit)


def render_user_explorer() -> None:
    """Render the User Explorer page."""
    st.header("User Explorer")

    user_ids = _get_user_ids()
    if not user_ids:
        st.warning("No users found. Run the data pipeline first.")
        return

    user_id = st.selectbox("Select user", user_ids)

    profile = _get_user_profile(user_id)
    if profile:
        st.subheader("Profile")
        c1, c2, c3, c4 = st.columns(4)
        c1.write(f"**User ID:** {profile.get('user_id', user_id)}")
        c2.write(f"**Archetype:** {profile.get('archetype_label', profile.get('archetype', 'n/a'))}")
        c3.write(f"**Device:** {profile.get('device_type', profile.get('device', 'n/a'))}")
        c4.write(f"**Country:** {profile.get('country', 'n/a')}")
        c5, c6, c7, c8 = st.columns(4)
        c5.write(f"**Channel:** {profile.get('acquisition_channel', profile.get('channel', 'n/a'))}")
        signup_val = profile.get("signup_date", profile.get("first_seen", "n/a"))
        c6.write(f"**Signup:** {signup_val}")
        c7.write(f"**Status:** {profile.get('status', 'n/a')}")
        c8.write(f"**Total Sessions:** {profile.get('total_sessions', 'n/a')}")

    features_data = _api_get(f"/users/{user_id}/features")
    predictions = _api_get(f"/users/{user_id}/predictions")
    recommendation = _api_get(f"/users/{user_id}/recommendation")

    if features_data:
        features = features_data.get("features", {}) if isinstance(features_data, dict) else {}
        if features:
            st.subheader("Key Features")
            feat_cols = st.columns(3)
            key_feats = [
                ("sessions_7d", "Sessions (7d)"),
                ("sessions_30d", "Sessions (30d)"),
                ("days_since_last_session", "Days Since Last Session"),
                ("revenue_30d", "Revenue (30d)"),
                ("total_ltv", "Total LTV"),
                ("engagement_change_7d", "Engagement Delta (7d)"),
                ("session_duration_avg_7d", "Avg Session Duration (7d)"),
                ("notification_open_rate", "Notification Open Rate"),
            ]
            for i, (key, label) in enumerate(key_feats):
                val = features.get(key)
                if val is not None:
                    if isinstance(val, float):
                        feat_cols[i % 3].write(f"**{label}:** {val:,.2f}")
                    else:
                        feat_cols[i % 3].write(f"**{label}:** {val}")

    if predictions:
        st.subheader("Predictions")
        p1, p2, p3, p4 = st.columns(4)
        rr = predictions.get("retention_risk")
        p1.metric("Retention Risk", f"{rr:.1%}" if rr is not None else "n/a")
        pp = predictions.get("purchase_probability")
        p2.metric("Purchase Propensity", f"{pp:.1%}" if pp is not None else "n/a")
        ltv = predictions.get("predicted_ltv")
        p3.metric("Predicted LTV", f"${ltv:,.2f}" if ltv is not None else "n/a")
        p4.metric("Segment", recommendation.get("segment", "n/a") if recommendation else "n/a")

    if recommendation:
        st.subheader("Recommended Action")
        st.success(f"**{recommendation.get('recommended_action', 'n/a')}**")
        st.caption(
            f"Intervention: {recommendation.get('intervention_type', 'n/a')} "
            f"| {recommendation.get('recommendation_timestamp', '')}"
        )

    st.subheader("Recent Activity Timeline")
    activity = _get_user_events(user_id)
    if activity is not None and not activity.empty:
        display_cols = [c for c in ["timestamp", "event_type", "content_category",
                                    "session_duration", "purchase_amount", "level"]
                        if c in activity.columns]
        st.dataframe(activity[display_cols], use_container_width=True)
    else:
        st.info("No activity data available locally.")


# ---------------------------------------------------------------------------
# 3. Decision Explanation
# ---------------------------------------------------------------------------


def render_decision_explanation() -> None:
    """Render candidate actions, the chosen action, guardrails, and AI explanation."""
    st.header("Decision Explanation")

    user_ids = _get_user_ids()
    if not user_ids:
        st.warning("No users found.")
        return

    user_id = st.selectbox("Select user", user_ids, key="decision_user")

    decision = _api_post(f"/users/{user_id}/decision")
    if decision is None:
        st.error(
            "Could not fetch decision — is the API running? "
            "The decision endpoint requires the full backend."
        )
        return

    # ---- Current user state ----
    st.subheader("Current User State")
    user_state = decision.get("user_state", {})
    if user_state:
        s1, s2, s3, s4 = st.columns(4)
        s1.write(f"**Segment:** {user_state.get('segment', user_state.get('value_segment', 'n/a'))}")
        s2.write(f"**Engagement:** {user_state.get('engagement', 'n/a')}")
        rr = user_state.get("retention_risk", user_state.get("retention_probability", "n/a"))
        s3.write(f"**Retention Risk:** {rr}")
        s4.write(f"**Value:** {user_state.get('value_segment', 'n/a')}")

    # ---- Predictions ----
    st.subheader("Model Predictions")
    predictions = decision.get("predictions", {})
    if predictions:
        pr1, pr2, pr3 = st.columns(3)
        rr_val = predictions.get("retention_risk")
        pr1.metric("Retention Risk", f"{rr_val:.1%}" if rr_val is not None else "n/a")
        pp_val = predictions.get("purchase_probability")
        pr2.metric("Purchase Propensity", f"{pp_val:.1%}" if pp_val is not None else "n/a")
        lt_val = predictions.get("predicted_ltv")
        pr3.metric("Predicted LTV", f"${lt_val:,.2f}" if lt_val is not None else "n/a")

    # ---- Candidate actions table ----
    decision_body = decision.get("decision", {})
    action_name = decision_body.get("action", "n/a")
    ev = decision_body.get("expected_value", 0.0)
    objective = decision_body.get("objective", "n/a")
    confidence = decision_body.get("confidence", 0.0)
    reasons = decision_body.get("reasons", [])

    st.subheader("Selected Action")
    st.markdown(
        f"### {action_name}\n"
        f"**Expected Value:** {ev:.4f} · "
        f"**Objective:** {objective} · "
        f"**Confidence:** {confidence:.3f}"
    )

    if reasons:
        st.subheader("Reasoning")
        for reason in reasons:
            st.markdown(f"- {reason}")

    # ---- Guardrails ----
    guardrails = decision_body.get("guardrails", [])
    st.subheader("Guardrails Applied")
    if guardrails:
        for g in guardrails:
            st.markdown(f"- :warning: {g}")
    else:
        st.write("No guardrails triggered.")

    # ---- AI Explanation ----
    st.subheader("AI Explanation")
    explanation = decision.get("explanation")
    if explanation:
        st.info(explanation)
    else:
        st.info(
            f"User {user_id} is recommended **{action_name}** "
            f"(expected value {ev:.4f}). "
            "No LLM explanation was generated — set OPENAI_API_KEY "
            "to enable richer reasoning."
        )

    # ---- Candidate scores ----
    st.subheader("Candidate Action Scores")
    scores = decision.get("candidate_scores")
    if scores and isinstance(scores, list):
        st.dataframe(pd.DataFrame(scores), use_container_width=True)
    else:
        st.caption("Candidate-level scores are available when the decision engine exposes them.")


# ---------------------------------------------------------------------------
# 4. Experiment Dashboard
# ---------------------------------------------------------------------------


def _get_experiments() -> list[dict[str, Any]]:
    """Return experiment records from the API or local parquet."""
    via_api = _api_get("/experiments")
    if isinstance(via_api, list):
        return [e for e in via_api if isinstance(e, dict)]
    exp_df = _load_parquet("experiments", PROCESSED_DIR)
    if exp_df is not None and not exp_df.empty:
        return exp_df.to_dict(orient="records")
    return []


def render_experiment_dashboard() -> None:
    """Render the Experiment Dashboard with variant comparison and charts."""
    st.header("Experiment Dashboard")

    experiments = _get_experiments()
    if not experiments:
        st.info("No experiments available. Create experiments via the API to see results.")
        return

    exp_names = [e.get("name", e.get("experiment_id", "?")) for e in experiments]
    exp_idx = st.selectbox("Experiment", range(len(experiments)), format_func=lambda i: exp_names[i])
    exp = experiments[exp_idx]
    exp_id = exp.get("experiment_id", "")

    st.write(f"**Status:** {exp.get('status', 'n/a')} | **Created:** {exp.get('created_at', 'n/a')}")

    results = _api_get(f"/experiments/{exp_id}/results")
    if results is None:
        st.warning("No results available for this experiment.")
        return

    variant_assignments = results.get("variant_assignments", {})
    conversion_rates = results.get("conversion_rates", {})
    significance = results.get("statistical_significance", {})
    recommendation = results.get("recommendation")

    if recommendation:
        st.info(f"**Recommendation:** {recommendation}")

    # ---- Comparison table ----
    if variant_assignments and isinstance(variant_assignments, dict):
        st.subheader("Control vs. Variants")
        rows: list[dict[str, Any]] = []
        for variant, assignments in variant_assignments.items():
            conv = conversion_rates.get(variant, 0.0)
            mean = conv.get("mean", 0.0) if isinstance(conv, dict) else conv
            n_val = assignments.get("n", assignments) if isinstance(assignments, dict) else assignments
            sig = significance.get(variant, {}) if isinstance(significance, dict) else {}
            rows.append({
                "variant": variant,
                "n": n_val,
                "mean": mean,
                "ci_lower": conv.get("ci_lower") if isinstance(conv, dict) else None,
                "ci_upper": conv.get("ci_upper") if isinstance(conv, dict) else None,
                "lift": sig.get("lift", 0.0),
                "p_value": sig.get("p_value"),
                "significant": sig.get("is_significant", sig.get("significant", False)),
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True)

        # Bar chart of means
        chart_df = pd.DataFrame(rows)
        has_ci = "ci_lower" in chart_df.columns and chart_df["ci_lower"].notna().any()
        fig = px.bar(
            chart_df, x="variant", y="mean",
            error_y=(
                [
                    chart_df["mean"] - chart_df["ci_lower"].fillna(chart_df["mean"]),
                    chart_df["ci_upper"].fillna(chart_df["mean"]) - chart_df["mean"],
                ]
                if has_ci
                else None
            ),
            title="Mean Outcome by Variant",
            color="variant",
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No variant-level results for this experiment.")

    # ---- Outcome-level charts from local data ----
    outcomes = _load_parquet("experiment_results", PROCESSED_DIR)
    if outcomes is not None and "variant" in outcomes.columns:
        st.subheader("Outcome Metrics by Variant")
        metric_candidates = ["d1_retention", "d7_retention", "conversion",
                             "conversion_rate", "revenue_per_user", "outcome"]
        available = [m for m in metric_candidates if m in outcomes.columns]
        if available:
            chosen = st.selectbox("Metric", available, key="exp_metric")
            chart_data = outcomes.groupby("variant", as_index=False)[chosen].agg(["mean", "std"]).reset_index()
            fig = px.bar(
                chart_data, x="variant", y="mean", error_y="std",
                title=f"Mean {chosen} by Variant",
                color="variant",
            )
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.caption("Outcome-level experiment data not present locally.")


# ---------------------------------------------------------------------------
# 5. Model Performance
# ---------------------------------------------------------------------------


def _find_model_artifacts() -> dict[str, Path]:
    """Scan data/models/ (recursively) and data/processed/ for model files."""
    artifacts: dict[str, Path] = {}
    for folder in [MODELS_DIR, PROCESSED_DIR]:
        if not folder.exists():
            continue
        for f in folder.rglob("*"):
            if f.is_file() and f.suffix in {".joblib", ".pkl", ".pickle"}:
                key = f"{f.parent.name}/{f.stem}" if f.parent.name not in ("models", "processed") else f.stem
                artifacts[key] = f
            elif f.is_file() and f.suffix == ".csv" and "metric" in f.stem.lower():
                key = f"{f.parent.name}/{f.stem}" if f.parent.name not in ("models", "processed") else f.stem
                artifacts[key] = f
    return artifacts


def _load_joblib_model(path: Path) -> Any:
    """Load a joblib/pickle model from *path*."""
    import pickle

    with open(path, "rb") as fh:
        try:
            import joblib
            return joblib.load(fh)
        except ImportError:
            fh.seek(0)
            return pickle.load(fh)


def render_model_performance() -> None:
    """Render offline model performance metrics from saved artifacts."""
    st.header("Model Performance")

    artifacts = _find_model_artifacts()
    if not artifacts:
        st.info(
            "No model artifacts found in data/models/ or data/processed/. "
            "Run the model training scripts first."
        )
        return

    model_names = sorted(artifacts.keys())
    chosen = st.selectbox("Model / Artifact", model_names, key="perf_model")
    model_path = artifacts[chosen]

    # Try loading as a scikit-learn / xgboost model with predict_proba
    model = None
    try:
        model = _load_joblib_model(model_path)
    except Exception:  # noqa: BLE001
        pass

    has_proba = model is not None and hasattr(model, "predict_proba")
    has_importances = model is not None and (
        hasattr(model, "feature_importances_") or hasattr(model, "coef_")
    )

    if has_proba:
        st.subheader("Model Summary")
        st.write(f"**Type:** {type(model).__name__}")
        st.write(f"**File:** `{model_path.name}`")

        # Show feature importances if available
        if has_importances:
            st.subheader("Feature Importance")
            if hasattr(model, "feature_importances_"):
                imp = np.array(model.feature_importances_)
            else:
                imp = np.abs(np.array(model.coef_)).flatten()

            feature_names = (
                list(model.feature_names_in_)
                if hasattr(model, "feature_names_in_")
                else [f"feature_{i}" for i in range(len(imp))]
            )
            imp_df = pd.DataFrame({"feature": feature_names, "importance": imp})
            imp_df = imp_df.sort_values("importance", ascending=False).head(20)
            fig = px.bar(imp_df, x="importance", y="feature", orientation="h",
                         title="Top 20 Feature Importances")
            fig.update_layout(yaxis={"categoryorder": "total ascending"})
            st.plotly_chart(fig, use_container_width=True)

        # Synthentic evaluation with random data for calibration / curve demos
        st.subheader("Model Evaluation (Synthetic Data)")
        n_features = len(model.feature_names_in_) if hasattr(model, "feature_names_in_") else 24
        rng = np.random.default_rng(42)
        X_demo = rng.standard_normal((1000, n_features))
        y_demo = rng.integers(0, 2, size=1000)

        try:
            y_proba = model.predict_proba(X_demo)[:, 1]
            y_pred = model.predict(X_demo)

            # ROC / PR data
            from sklearn.metrics import (
                average_precision_score,
                confusion_matrix,
                precision_recall_curve,
                roc_auc_score,
                roc_curve,
            )

            roc_auc = roc_auc_score(y_demo, y_proba)
            pr_auc = average_precision_score(y_demo, y_proba)
            cm = confusion_matrix(y_demo, y_pred)
            precision_arr, recall_arr, _ = precision_recall_curve(y_demo, y_proba)
            fpr, tpr, _ = roc_curve(y_demo, y_proba)

            m1, m2 = st.columns(2)
            m1.metric("ROC-AUC", f"{roc_auc:.4f}")
            m2.metric("PR-AUC", f"{pr_auc:.4f}")

            c1, c2 = st.columns(2)

            with c1:
                roc_fig = go.Figure()
                roc_fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name="ROC"))
                roc_fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines",
                                             line_dash="dash", name="Random"))
                roc_fig.update_layout(title="ROC Curve", xaxis_title="FPR", yaxis_title="TPR")
                st.plotly_chart(roc_fig, use_container_width=True)

            with c2:
                pr_fig = go.Figure()
                pr_fig.add_trace(go.Scatter(x=recall_arr, y=precision_arr, mode="lines", name="PR"))
                pr_fig.update_layout(title="PR Curve", xaxis_title="Recall", yaxis_title="Precision")
                st.plotly_chart(pr_fig, use_container_width=True)

            # Confusion matrix heatmap
            st.subheader("Confusion Matrix")
            cm_fig = px.imshow(
                cm, text_auto=True,
                labels=dict(x="Predicted", y="Actual", color="Count"),
                x=["Negative", "Positive"], y=["Negative", "Positive"],
                title="Confusion Matrix",
            )
            st.plotly_chart(cm_fig, use_container_width=True)

            # Calibration
            st.subheader("Calibration Curve")
            from sklearn.calibration import calibration_curve

            prob_true, prob_pred = calibration_curve(y_demo, y_proba, n_bins=10)
            cal_fig = go.Figure()
            cal_fig.add_trace(go.Scatter(x=prob_pred, y=prob_true, mode="markers+lines",
                                         name="Model"))
            cal_fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines",
                                         line_dash="dash", name="Perfect"))
            cal_fig.update_layout(title="Calibration Curve",
                                  xaxis_title="Mean Predicted Probability",
                                  yaxis_title="Fraction of Positives")
            st.plotly_chart(cal_fig, use_container_width=True)

        except Exception as exc:  # noqa: BLE001
            st.warning(f"Could not run synthetic evaluation: {exc}")

    else:
        st.info(
            f"Loaded `{model_path.name}` but it does not expose `predict_proba`. "
            "Showing file info only."
        )
        st.write(f"**File:** `{model_path}`")
        st.write(f"**Size:** {model_path.stat().st_size:,} bytes")
        if model is not None:
            st.write(f"**Type:** {type(model).__name__}")


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------


def main() -> None:
    """Run the AttentionOS Streamlit dashboard."""
    st.set_page_config(
        page_title="AttentionOS Dashboard",
        page_icon="🧠",
        layout="wide",
    )

    page = _render_sidebar()

    renderers: dict[str, Any] = {
        "Overview": render_overview,
        "User Explorer": render_user_explorer,
        "Decision Explanation": render_decision_explanation,
        "Experiment Dashboard": render_experiment_dashboard,
        "Model Performance": render_model_performance,
    }

    renderers[page]()


if __name__ == "__main__":
    main()
