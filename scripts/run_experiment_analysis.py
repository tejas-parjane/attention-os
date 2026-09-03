"""Create a synthetic A/B experiment, simulate treatment effects, run analysis."""

from __future__ import annotations

import logging
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from src.experiments.engine import (
    Experiment,
    analyze_experiment,
    create_assignment_table,
    run_ab_analysis,
    simulate_treatment_effects,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    events_path = ROOT / "data" / "raw" / "events.parquet"
    if not events_path.exists():
        logger.error("Missing %s — run `python scripts/generate_data.py` first.", events_path)
        sys.exit(1)

    events_df = pd.read_parquet(events_path)
    user_ids = sorted(events_df["user_id"].unique().tolist())
    logger.info("Loaded %d user IDs from events", len(user_ids))

    # ── 1. Create experiment ─────────────────────────────────────────────
    experiment = Experiment(
        experiment_id="exp_2026_attn_v1",
        name="Attention Optimization Experiment",
        variants=["control", "personalized_challenge", "generic_notification", "reward"],
        primary_metrics=["d1_retention", "d7_retention", "session_frequency", "conversion_rate", "revenue_per_user"],
        created_at=datetime.now(),
    )
    logger.info("Created experiment: %s (%s)", experiment.name, experiment.experiment_id)

    # ── 2. Assign users deterministically ────────────────────────────────
    assignments = create_assignment_table(experiment, user_ids)
    logger.info("Assigned %d users to variants", len(assignments))

    print("\nVariant assignment counts:")
    print(assignments["variant"].value_counts().to_string())

    # ── 3. Simulate treatment effects ────────────────────────────────────
    treatment_effects = {
        "control": 0.0,
        "personalized_challenge": 0.05,
        "generic_notification": 0.02,
        "reward": 0.08,
    }
    baseline_retention = 0.55
    noise_std = 0.12

    outcomes = simulate_treatment_effects(
        assignments,
        baseline_outcome=baseline_retention,
        treatment_effects=treatment_effects,
        noise_std=noise_std,
        seed=42,
    )
    logger.info(
        "Simulated outcomes: baseline=%.2f, effects=%s, noise=%.2f",
        baseline_retention,
        treatment_effects,
        noise_std,
    )

    # ── 4. Run A/B analysis ─────────────────────────────────────────────
    analysis = analyze_experiment(assignments, outcomes, experiment.experiment_id)
    summary_df = run_ab_analysis(outcomes, metric_column="outcome", control_variant="control")

    # ── 5. Save results ─────────────────────────────────────────────────
    results_df = outcomes.merge(
        assignments[["experiment_id", "user_id", "variant"]].drop_duplicates(),
        on=["experiment_id", "user_id", "variant"],
        how="left",
    )
    results_df["timestamp"] = datetime.now()
    out_path = ROOT / "data" / "processed" / "experiment_results.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_parquet(out_path, index=False)
    logger.info("Saved experiment results -> %s (%d rows)", out_path, len(results_df))

    # ── 6. Print per-variant comparison with CIs and significance ───────
    print("\n" + "=" * 70)
    print(f"EXPERIMENT: {experiment.name}")
    print(f"ID: {experiment.experiment_id}")
    print(f"Total users: {len(assignments)}")
    print("=" * 70)

    print(f"\n{'Variant':<28} {'Mean':>8} {'Std':>8} {'N':>6} {'95% CI':>22} {'Lift':>8} {'p-value':>10} {'Sig.':>6}")
    print("-" * 100)
    for _, row in summary_df.iterrows():
        ci_str = f"[{row['ci_lower']:.4f}, {row['ci_upper']:.4f}]"
        p_str = f"{row['p_value']:.4f}" if row["p_value"] is not None else "N/A"
        sig_str = "YES" if row["significant"] else "no"
        lift_str = f"{row['lift']:.4f}" if row["variant"] != "control" else "  -  "
        print(
            f"{row['variant']:<28} {row['mean']:>8.4f} {row['std']:>8.4f} "
            f"{int(row['sample_size']):>6} {ci_str:>22} {lift_str:>8} {p_str:>10} {sig_str:>6}"
        )

    print("\nTreatment effects (true vs estimated):")
    for variant in ["personalized_challenge", "generic_notification", "reward"]:
        true_effect = treatment_effects[variant]
        est_row = summary_df[summary_df["variant"] == variant]
        if not est_row.empty:
            est_lift = est_row.iloc[0]["lift"]
            est_mean = est_row.iloc[0]["mean"]
            sig = est_row.iloc[0]["significant"]
            print(
                f"  {variant:<30} true={true_effect:+.3f}  "
                f"estimated_lift={est_lift:+.4f}  "
                f"est_mean={est_mean:.4f}  "
                f"significant={'yes' if sig else 'no'}"
            )


if __name__ == "__main__":
    main()

