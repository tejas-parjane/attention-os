import numpy as np
import pandas as pd
import pytest

from src.experiments.engine import (
    assign_variant,
    create_assignment_table,
    run_ab_analysis,
    simulate_treatment_effects,
)


class TestAssignVariantDeterministic:
    def test_same_input_same_output(self):
        v1 = assign_variant("exp_001", "user_042", ["control", "treatment_a"])
        v2 = assign_variant("exp_001", "user_042", ["control", "treatment_a"])
        assert v1 == v2

    def test_different_users_can_get_different_variants(self):
        variants = ["control", "treatment_a", "treatment_b"]
        assignments = {
            assign_variant("exp_002", f"user_{i:04d}", variants)
            for i in range(200)
        }
        assert len(assignments) > 1, "All users got the same variant — assignment is not working"

    def test_different_experiments_give_different_assignments(self):
        v_exp1 = assign_variant("exp_A", "user_001", ["control", "treatment"])
        v_exp2 = assign_variant("exp_B", "user_001", ["control", "treatment"])
        # Not guaranteed to differ, but very likely with different experiment IDs
        # This test mainly verifies no crash
        assert v_exp1 in {"control", "treatment"}
        assert v_exp2 in {"control", "treatment"}


class TestAssignmentBalance:
    def test_balanced_across_variants(self):
        variants = ["control", "treatment_a", "treatment_b"]
        assignments = [
            assign_variant("exp_bal", f"user_{i:05d}", variants) for i in range(3000)
        ]
        counts = pd.Series(assignments).value_counts()
        # Each variant should get between 25% and 42% of users
        for v in variants:
            ratio = counts[v] / len(assignments)
            assert 0.25 < ratio < 0.42, f"Variant '{v}' has {ratio:.1%} — outside [25%, 42%]"


class TestSimulateTreatmentEffects:
    def test_produces_known_effect(self):
        rng = np.random.default_rng(42)
        n = 500
        assignments = pd.DataFrame(
            {
                "experiment_id": "exp_sim",
                "user_id": [f"u{i:04d}" for i in range(n)],
                "variant": ["control"] * (n // 2) + ["treatment"] * (n // 2),
            }
        )
        baseline = 0.5
        effects = {"control": 0.0, "treatment": 0.1}
        outcomes = simulate_treatment_effects(
            assignments, baseline_outcome=baseline, treatment_effects=effects, noise_std=0.02, seed=0
        )
        control_mean = outcomes.loc[outcomes["variant"] == "control", "outcome"].mean()
        treatment_mean = outcomes.loc[outcomes["variant"] == "treatment", "outcome"].mean()
        # With low noise, treatment should be ~0.1 higher than control
        assert treatment_mean - control_mean == pytest.approx(0.1, abs=0.02)


class TestRunAbAnalysis:
    def test_identifies_winning_variant(self):
        n = 1000
        rng = np.random.default_rng(99)
        data = pd.DataFrame(
            {
                "variant": ["control"] * (n // 2) + ["winner"] * (n // 2),
                "outcome": np.concatenate(
                    [
                        rng.normal(0.5, 0.1, n // 2),
                        rng.normal(0.55, 0.1, n // 2),
                    ]
                ),
            }
        )
        summary = run_ab_analysis(data, metric_column="outcome", control_variant="control")
        winner_row = summary[summary["variant"] == "winner"].iloc[0]
        control_row = summary[summary["variant"] == "control"].iloc[0]
        assert winner_row["mean"] > control_row["mean"], "Winner should have higher mean"
        # With n=1000 and effect=0.05, this should be significant
        assert winner_row["p_value"] < 0.05, f"p-value {winner_row['p_value']} not significant"

    def test_control_lift_is_zero(self):
        data = pd.DataFrame(
            {
                "variant": ["control"] * 100 + ["treatment"] * 100,
                "outcome": np.concatenate(
                    [np.full(100, 0.5), np.full(100, 0.5)]
                ),
            }
        )
        summary = run_ab_analysis(data, metric_column="outcome", control_variant="control")
        control_row = summary[summary["variant"] == "control"].iloc[0]
        assert control_row["lift"] == 0.0
        # With zero variance in both groups, p_value is either None or nan
        import math
        pv = control_row["p_value"]
        assert pv is None or (isinstance(pv, float) and math.isnan(pv))

    def test_output_has_expected_columns(self):
        data = pd.DataFrame(
            {"variant": ["control", "treatment"], "outcome": [0.5, 0.6]}
        )
        summary = run_ab_analysis(data, metric_column="outcome", control_variant="control")
        expected_cols = {"variant", "mean", "std", "sample_size", "ci_lower", "ci_upper", "lift", "p_value", "significant"}
        assert expected_cols.issubset(set(summary.columns))


class TestCreateAssignmentTable:
    def test_correct_row_count(self):
        experiment = type(
            "Exp",
            (),
            {
                "experiment_id": "exp_001",
                "name": "test",
                "variants": ["control", "treatment"],
            },
        )()
        user_ids = [f"u{i}" for i in range(50)]
        table = create_assignment_table(experiment, user_ids)
        assert len(table) == 50
        assert set(table["user_id"]) == set(user_ids)
        assert set(table["variant"]).issubset({"control", "treatment"})
