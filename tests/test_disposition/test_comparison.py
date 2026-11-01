"""Disposition comparison tests: unit, integration, CLI, report (spec §4, §8).

The go/no-go rails for the engine itself live in ``test_evaluation.py``; the
H1 scaffold lives in ``tests/test_baseline_comparison.py``. This suite covers
the H3-shaped disposition comparison — the risk-adjusted optimizer against a
simple pass/fail resale policy (and a perfect-information oracle) on
identical held-out simulated data under a synthetic economy. Every assertion
is a pipeline health check on simulated ground truth, never a real-world
reliability claim and never an H3 outcome.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest

from rechip01.disposition import (
    OPTIMIZER_POLICY_NAME,
    ORACLE_POLICY_NAME,
    PASS_FAIL_POLICY_NAME,
    PATH_MANUAL_REVIEW,
    DispositionComparison,
    DispositionConfig,
    compare_disposition_policies,
    comparison_economy,
    render_comparison_report,
)
from rechip01.disposition.comparison import (
    _realize_decision,
    main,
    optimizer_actions,
    pass_fail_actions,
    realize_action,
)
from rechip01.engine.harness import HarnessResult, run_harness
from rechip01.schemas import Provenance, ReliabilityEstimate
from rechip01.simulator import default_fleet_config, generate_fleet
from tests.test_disposition.helpers import make_estimate
from tests.test_engine.helpers import EVAL_SEED

if TYPE_CHECKING:
    from pathlib import Path

    from rechip01.simulator import SimulatedFleet


def _estimates_for(
    fleet: SimulatedFleet, p_fail: float = 0.45, ci: tuple[float, float] = (0.40, 0.50)
) -> list[ReliabilityEstimate]:
    """Estimates aligned with the fleet, keyed to each unit's assigned workload."""
    return [
        ReliabilityEstimate(
            asset_id=scan.asset_id,
            horizon_days=365,
            provenance=Provenance.SIMULATED,
            p_fail=p_fail,
            ci=ci,
            workload_fitness={label.assigned_workload: 0.80},
            ood_flag=False,
            model_version="test-engine-v0",
        )
        for scan, label in zip(fleet.scans, fleet.labels, strict=True)
    ]


@pytest.fixture(scope="session")
def small_fleet() -> SimulatedFleet:
    """A small fleet for fast unit-level tests."""
    config = default_fleet_config(seed=321, n_units=12)
    return generate_fleet(config)


@pytest.fixture(scope="session")
def harness_result() -> HarnessResult:
    """One full default-harness run shared by the integration tests."""
    return run_harness()


@pytest.fixture(scope="session")
def comparison(harness_result: HarnessResult) -> DispositionComparison:
    return compare_disposition_policies(
        harness_result.estimates, harness_result.eval_fleet, comparison_economy()
    )


# ------------------------------------------------------------------ unit


def test_pass_fail_actions_gate_and_monotonicity(small_fleet: SimulatedFleet) -> None:
    labels = small_fleet.labels
    estimates = _estimates_for(small_fleet, p_fail=0.45, ci=(0.45, 0.45))
    at_loose = pass_fail_actions(estimates, labels, threshold=0.5)
    at_tight = pass_fail_actions(estimates, labels, threshold=0.4)
    assert all(
        action == f"reuse_{label.assigned_workload}"
        for action, label in zip(at_loose, labels, strict=True)
    )  # p=0.45 <= 0.5
    assert all(action == "recycle" for action in at_tight)  # p=0.45 > 0.4


def test_pass_fail_alignment_errors(small_fleet: SimulatedFleet) -> None:
    estimates = _estimates_for(small_fleet)
    with pytest.raises(ValueError, match="misaligned"):
        pass_fail_actions(estimates[:-1], small_fleet.labels)
    swapped = list(estimates)
    swapped[0], swapped[1] = swapped[1], swapped[0]
    with pytest.raises(ValueError, match="misaligned"):
        pass_fail_actions(swapped, small_fleet.labels)


def test_optimizer_actions_alignment_errors(small_fleet: SimulatedFleet) -> None:
    economy = comparison_economy()
    estimates = _estimates_for(small_fleet)
    with pytest.raises(ValueError, match="misaligned"):
        optimizer_actions(estimates[:-1], small_fleet.labels, economy)
    swapped = list(estimates)
    swapped[0], swapped[1] = swapped[1], swapped[0]
    with pytest.raises(ValueError, match="misaligned"):
        optimizer_actions(swapped, small_fleet.labels, economy)


def test_realize_action_hand_computed(small_fleet: SimulatedFleet) -> None:
    label = small_fleet.labels[0]
    economy = comparison_economy()
    price = economy.resale_price_by_workload[label.assigned_workload].midpoint()

    good = label.model_copy(update={"failed_within_horizon": False})
    assert realize_action(f"reuse_{label.assigned_workload}", good, economy) == pytest.approx(
        -economy.test_cost + price
    )
    failed = label.model_copy(update={"failed_within_horizon": True})
    assert realize_action(f"reuse_{label.assigned_workload}", failed, economy) == pytest.approx(
        -economy.test_cost - economy.downtime_liability
    )
    best_price = max(pr.midpoint() for pr in economy.resale_price_by_workload.values())
    assert realize_action("repair", label, economy) == pytest.approx(
        -economy.test_cost
        - economy.repair_cost
        + (1 - economy.repair_residual_p_fail) * best_price
        - economy.repair_residual_p_fail * economy.downtime_liability
    )
    assert realize_action("harvest", label, economy) == pytest.approx(
        -economy.test_cost + economy.harvest_recovery.midpoint()
    )
    assert realize_action("recycle", label, economy) == pytest.approx(
        economy.recycling_value.midpoint()
    )


def test_unknown_path_rejected(small_fleet: SimulatedFleet) -> None:
    with pytest.raises(ValueError, match="unknown disposition path"):
        realize_action("shred", small_fleet.labels[0], comparison_economy())


def test_review_cost_flows_into_realized_value(small_fleet: SimulatedFleet) -> None:
    economy = comparison_economy()
    config = DispositionConfig(review_cost=42.0)
    label = small_fleet.labels[0]
    estimate = make_estimate(
        asset_id=small_fleet.scans[0].asset_id,
        ood_flag=True,
        workload_fitness={label.assigned_workload: 0.80},
    )
    decisions = optimizer_actions([estimate], [label], economy, config=config, seed=0)
    assert decisions[0].recommendation == PATH_MANUAL_REVIEW

    best = max(
        (o for o in decisions[0].options_evaluated if o.path != PATH_MANUAL_REVIEW),
        key=lambda o: o.expected_net_recovery,
    )
    realized = _realize_decision(decisions[0], label, economy, review_cost=42.0)
    assert realized == pytest.approx(realize_action(best.path, label, economy) - 42.0)


# ------------------------------------------------------------------ integration


class TestDispositionComparisonIntegration:
    def test_three_policies_on_identical_fleet(self, comparison: DispositionComparison) -> None:
        assert {p.name for p in comparison.policies} == {
            PASS_FAIL_POLICY_NAME,
            OPTIMIZER_POLICY_NAME,
            ORACLE_POLICY_NAME,
        }
        assert comparison.n_units == 2000
        assert comparison.eval_seed == EVAL_SEED
        for policy in comparison.policies:
            assert policy.n_units == 2000
            assert sum(policy.action_counts.values()) == 2000

    def test_h3_shaped_ordering(self, comparison: DispositionComparison) -> None:
        by_name = {p.name: p for p in comparison.policies}
        # The H3-shaped direction on simulated data: risk-adjusted optimization
        # realizes more per unit than the simple pass/fail policy, and the
        # oracle (perfect information) does not underperform the model.
        assert (
            by_name[OPTIMIZER_POLICY_NAME].mean_per_unit
            > by_name[PASS_FAIL_POLICY_NAME].mean_per_unit
        )
        assert (
            by_name[ORACLE_POLICY_NAME].mean_per_unit
            >= by_name[OPTIMIZER_POLICY_NAME].mean_per_unit - 1e-9
        )

    def test_differences_bracket_mean(self, comparison: DispositionComparison) -> None:
        assert {d.name for d in comparison.differences} == {
            OPTIMIZER_POLICY_NAME,
            ORACLE_POLICY_NAME,
        }
        for difference in comparison.differences:
            assert difference.ci_low <= difference.mean_diff_per_unit <= difference.ci_high

    def test_oracle_never_routes_to_review(self, comparison: DispositionComparison) -> None:
        oracle = next(p for p in comparison.policies if p.name == ORACLE_POLICY_NAME)
        assert oracle.review_count == 0

    def test_protocol_recorded(self, comparison: DispositionComparison) -> None:
        assert comparison.pass_fail_threshold == 0.5
        assert comparison.n_bootstrap == 500
        assert comparison.horizon_days == 365
        assert set(comparison.economy.resale_price_by_workload) == {
            "inference",
            "finetune",
            "memory-stress",
        }

    def test_comparison_deterministic(self, harness_result: HarnessResult) -> None:
        first = compare_disposition_policies(
            harness_result.estimates, harness_result.eval_fleet, comparison_economy()
        )
        second = compare_disposition_policies(
            harness_result.estimates, harness_result.eval_fleet, comparison_economy()
        )
        assert first == second


# ------------------------------------------------------------------ report + CLI


def test_report_contains_banner_and_protocol(comparison: DispositionComparison) -> None:
    report = render_comparison_report(comparison)
    assert "SIMULATION-BASED PIPELINE CHECK" in report
    assert "not an H1/H2/H3 outcome" in report
    assert "Pre-registered comparison protocol" in report
    assert "## Policy comparison" in report
    assert "## Paired difference vs pass/fail baseline" in report
    for policy in comparison.policies:
        assert policy.name in report
    assert "must never" in report  # honesty footer


def test_cli_writes_artifacts(tmp_path: Path) -> None:
    exit_code = main(["--out", str(tmp_path)])
    assert exit_code == 0
    report_path = tmp_path / "disposition-comparison.md"
    metrics_path = tmp_path / "disposition-metrics.json"
    assert report_path.exists()
    assert metrics_path.exists()
    payload = json.loads(metrics_path.read_text())
    assert payload["honesty_banner"]
    assert len(payload["policies"]) == 3
    assert payload["pass_fail_threshold"] == 0.5
    report_text = report_path.read_text()
    assert "SIMULATION-BASED PIPELINE CHECK" in report_text
    assert "must never" in report_text
