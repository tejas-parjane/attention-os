import hashlib
import numpy as np
import pandas as pd
from typing import List, Dict, Optional
from datetime import datetime
from pydantic import BaseModel

import scipy.stats as stats


class Experiment(BaseModel):
    experiment_id: str
    name: str
    variants: list[str]  # e.g. ["control", "personalized_challenge", "generic_notification", "reward"]
    primary_metrics: list[str] = ["d1_retention", "d7_retention", "session_frequency", "conversion_rate", "revenue_per_user"]
    guardrail_metrics: list[str] = ["notification_frequency", "unsubscribe_rate", "negative_engagement"]
    created_at: datetime


class Assignment(BaseModel):
    experiment_id: str
    user_id: str
    variant: str
    assignment_timestamp: datetime


def assign_variant(experiment_id: str, user_id: str, variants: list[str], salt: str = "attention-os") -> str:
    """
    Deterministic stable assignment using MD5 hash of (experiment_id, user_id, salt).
    Same user always gets same variant regardless of when called.
    """
    raw = f"{experiment_id}|{user_id}|{salt}".encode("utf-8")
    digest = hashlib.md5(raw).hexdigest()
    index = int(digest, 16) % len(variants)
    return variants[index]


def create_assignment_table(experiment: Experiment, user_ids: list[str]) -> pd.DataFrame:
    """Create assignment records for all users."""
    timestamp = datetime.now()
    records = [
        {
            "experiment_id": experiment.experiment_id,
            "user_id": user_id,
            "variant": assign_variant(experiment.experiment_id, user_id, experiment.variants),
            "assignment_timestamp": timestamp,
        }
        for user_id in user_ids
    ]
    return pd.DataFrame(records)


def simulate_treatment_effects(
    assignments: pd.DataFrame,
    baseline_outcome: float,
    treatment_effects: Dict[str, float],
    noise_std: float = 0.1,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Create simulated outcomes where treatment effects are KNOWN.
    control gets baseline_outcome + noise.
    variant v gets baseline_outcome + treatment_effects[v] + noise.
    Returns DataFrame with experiment_id, user_id, variant, outcome.
    This is a CONTROLLED simulation demonstrating the analysis method.
    """
    rng = np.random.default_rng(seed)
    noise = rng.normal(0, noise_std, size=len(assignments))

    outcomes = []
    for row in assignments.itertuples(index=False):
        effect = treatment_effects.get(row.variant, 0.0)
        outcome = baseline_outcome + effect + noise[len(outcomes)]
        outcomes.append(
            {
                "experiment_id": row.experiment_id,
                "user_id": row.user_id,
                "variant": row.variant,
                "outcome": outcome,
            }
        )

    return pd.DataFrame(outcomes)


def run_ab_analysis(
    data: pd.DataFrame,
    metric_column: str = "outcome",
    variant_column: str = "variant",
    control_variant: str = "control",
    alpha: float = 0.05,
) -> pd.DataFrame:
    """
    Compute per-variant: mean, std, sample_size, CI (95%), lift vs control, p-value (Welch's t-test or Mann-Whitney U).
    Returns a summary DataFrame.
    """
    variants = data[variant_column].unique()
    summary_rows = []

    control_values = data.loc[data[variant_column] == control_variant, metric_column].to_numpy(dtype=float)

    for variant in variants:
        variant_values = data.loc[data[variant_column] == variant, metric_column].to_numpy(dtype=float)
        mean = float(np.mean(variant_values))
        std = float(np.std(variant_values, ddof=1)) if len(variant_values) > 1 else 0.0
        n = len(variant_values)
        ci_margin = 1.96 * (std / np.sqrt(n)) if n > 0 else 0.0

        p_value = None
        if variant == control_variant:
            lift = 0.0
        else:
            if control_values.var() > 0 or variant_values.var() > 0:
                t_stat, p_value = stats.ttest_ind(variant_values, control_values, equal_var=False)
                p_value = float(p_value)
            else:
                try:
                    _, p_value = stats.mannwhitneyu(variant_values, control_values, alternative="two-sided")
                    p_value = float(p_value)
                except ValueError:
                    p_value = None
            lift = (mean - float(np.mean(control_values))) / float(np.mean(control_values)) if float(np.mean(control_values)) != 0 else float("nan")

        summary_rows.append(
            {
                "variant": variant,
                "mean": mean,
                "std": std,
                "sample_size": n,
                "ci_lower": mean - ci_margin,
                "ci_upper": mean + ci_margin,
                "lift": lift,
                "p_value": p_value,
                "significant": (p_value is not None) and (p_value < alpha),
            }
        )

    return pd.DataFrame(summary_rows)


def analyze_experiment(assignments: pd.DataFrame, outcomes: pd.DataFrame, experiment_id: str) -> dict:
    """
    Full pipeline: given assignments + outcomes, return summary with per-variant stats and significance.
    """
    data = outcomes[outcomes["experiment_id"] == experiment_id]

    variants = sorted(data["variant"].unique())
    detailed = {}

    for metric in [c for c in data.columns if c not in {"experiment_id", "user_id", "variant"}]:
        metric_data = data[["user_id", "variant", metric]].copy()
        detailed[metric] = run_ab_analysis(metric_data, metric_column=metric)

    has_control = "control" in data["variant"].values
    control_summary = None
    if has_control:
        for metric, df in detailed.items():
            row = df[df["variant"] == "control"]
            if not row.empty:
                control_summary = row.iloc[0].to_dict()
                control_summary["metric"] = metric
                break

    summary = {
        "experiment_id": experiment_id,
        "variants": variants,
        "control_variant": "control" if has_control else None,
        "sample_size": int(len(data)),
        "detailed": detailed,
    }

    return summary
