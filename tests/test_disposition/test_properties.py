"""Hypothesis property tests for the disposition layer (spec art_vuIPvUhR, §4, §8).

Properties pinned here, mirroring the task's test plan:

1. **Valid probability/economics bounds** — every option's ENR stays inside
   the economy's arithmetic bounds (no NaN, no impossible valuations) for
   generated economies and estimates.
2. **Monotonicity in reliability** — higher failure probability never raises
   the reuse valuation (exact on point valuations).
3. **Determinism under fixed seed** — identical inputs + seed reproduce the
   decision bit-for-bit, across generated economies.
4. **Evidence trace present on every decision** — including manual-review
   routes; every line is a non-empty string and review decisions explain
   the routing in the trace.

All inputs are synthetic; nothing asserts a real-world claim.
"""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from rechip01.disposition import (
    PATH_MANUAL_REVIEW,
    PATH_REUSE,
    EconomyAssumptions,
    optimize_disposition,
    value_reuse,
)
from rechip01.disposition.comparison import pass_fail_actions
from rechip01.schemas import ReliabilityEstimate
from rechip01.simulator import GroundTruthLabel
from tests.test_disposition.strategies import economies, estimates


@given(economies(), estimates())
@settings(max_examples=40, deadline=None)
def test_option_enr_within_economy_bounds(
    economy: EconomyAssumptions, estimate: ReliabilityEstimate
) -> None:
    decision = optimize_disposition(estimate, economy, seed=0)
    max_price = max(pr.midpoint() for pr in economy.resale_price_by_workload.values())
    upper = max_price + economy.harvest_recovery.high + 1e-6
    lower = -(economy.test_cost + economy.repair_cost + economy.downtime_liability) - 1e-6
    for option in decision.options_evaluated:
        assert lower <= option.expected_net_recovery <= upper, option.path


@given(economies(), estimates())
@settings(max_examples=40, deadline=None)
def test_regret_bound_nonnegative_and_equals_top2_gap(
    economy: EconomyAssumptions, estimate: ReliabilityEstimate
) -> None:
    decision = optimize_disposition(estimate, economy, seed=0)
    assert decision.regret_bound >= -1e-9
    enrs = sorted((o.expected_net_recovery for o in decision.options_evaluated), reverse=True)
    assert len(enrs) >= 4  # at least one reuse option + repair + harvest + recycle
    assert decision.regret_bound == pytest.approx(enrs[0] - enrs[1], abs=1e-6)


@given(economies(), estimates(), st.integers(0, 2**31 - 1))
@settings(max_examples=30, deadline=None)
def test_optimizer_deterministic_under_fixed_seed(
    economy: EconomyAssumptions, estimate: ReliabilityEstimate, seed: int
) -> None:
    first = optimize_disposition(estimate, economy, seed=seed)
    second = optimize_disposition(estimate, economy, seed=seed)
    assert first == second


@given(economies(), estimates(), st.booleans())
@settings(max_examples=30, deadline=None)
def test_evidence_trace_present_on_every_decision(
    economy: EconomyAssumptions, estimate: ReliabilityEstimate, ood: bool
) -> None:
    wide = (
        max(0.0, estimate.p_fail - 0.2),
        min(1.0, estimate.p_fail + 0.2),
    )
    candidates = [
        estimate,
        estimate.model_copy(update={"ood_flag": ood}),
        estimate.model_copy(update={"ci": wide}),  # width 0.4 > 0.35 review gate
    ]
    for candidate in candidates:
        decision = optimize_disposition(candidate, economy, seed=0)
        assert len(decision.assumption_trace) >= 1
        assert all(isinstance(line, str) and line.strip() for line in decision.assumption_trace)
        if decision.recommendation == PATH_MANUAL_REVIEW:
            assert any("manual_review" in line for line in decision.assumption_trace)


@given(economies(), estimates(), st.floats(0.01, 0.5, allow_nan=False))
@settings(max_examples=40, deadline=None)
def test_point_reuse_value_nonincreasing_in_p_fail(
    economy: EconomyAssumptions, estimate: ReliabilityEstimate, delta: float
) -> None:
    # Raw resale price draws; fitness enters the valuation through the
    # per-workload failure probability (1 - fitness).
    workload = next(iter(estimate.workload_fitness))
    price = economy.resale_price_by_workload[workload].midpoint()
    args = {"test_cost": economy.test_cost, "downtime_liability": economy.downtime_liability}
    p_low = 1.0 - estimate.workload_fitness[workload]
    p_high = min(1.0, p_low + delta)
    value_low = float(value_reuse(np.asarray([p_low]), np.asarray([price]), **args)[0])
    value_high = float(value_reuse(np.asarray([p_high]), np.asarray([price]), **args)[0])
    if p_high > p_low and price + economy.downtime_liability > 0:
        assert value_high < value_low + 1e-9
    else:
        assert value_high == pytest.approx(value_low)


@given(economies(), estimates(), st.floats(0.01, 0.3, allow_nan=False))
@settings(max_examples=30, deadline=None)
def test_reuse_enr_decreases_when_fitness_degrades(
    economy: EconomyAssumptions, estimate: ReliabilityEstimate, delta: float
) -> None:
    # Engine contract: fitness[w] is survival probability. Degrading every
    # workload's fitness must never raise any reuse option's ENR.
    degraded = {
        workload: max(0.0, fitness - delta)
        for workload, fitness in estimate.workload_fitness.items()
    }
    before = optimize_disposition(estimate, economy, seed=0)
    after = optimize_disposition(
        estimate.model_copy(update={"workload_fitness": degraded}), economy, seed=0
    )
    enr_before = {o.path: o.expected_net_recovery for o in before.options_evaluated}
    enr_after = {o.path: o.expected_net_recovery for o in after.options_evaluated}
    for path, value in enr_before.items():
        if not path.startswith(f"{PATH_REUSE}_"):
            continue
        workload = path[len(PATH_REUSE) + 1 :]
        if degraded[workload] < estimate.workload_fitness[workload]:
            assert enr_after[path] <= value + 1e-9, path


@given(economies(), estimates())
@settings(max_examples=40, deadline=None)
def test_recommendation_is_available_path(
    economy: EconomyAssumptions, estimate: ReliabilityEstimate
) -> None:
    decision = optimize_disposition(estimate, economy, seed=0)
    paths = {o.path for o in decision.options_evaluated} | {PATH_MANUAL_REVIEW}
    assert decision.recommendation in paths
    if decision.recommendation.startswith(f"{PATH_REUSE}_"):
        assert decision.recommendation[len(PATH_REUSE) + 1 :] in estimate.workload_fitness


@given(
    st.lists(estimates(), min_size=1, max_size=6),
    st.floats(0.05, 0.45, allow_nan=False),
    st.floats(0.55, 0.95, allow_nan=False),
)
@settings(max_examples=25, deadline=None)
def test_pass_fail_actions_monotone_in_threshold(
    estimates_list: list[ReliabilityEstimate], threshold_low: float, threshold_high: float
) -> None:
    labels = [
        GroundTruthLabel(
            asset_id=estimate.asset_id,
            provenance=estimate.provenance,
            assigned_workload=next(iter(estimate.workload_fitness)),
            horizon_days=365,
            has_latent_defect=False,
            failed_within_horizon=False,
            p_fail_assigned_workload=estimate.p_fail,
            p_fail_by_workload=dict.fromkeys(estimate.workload_fitness, estimate.p_fail),
        )
        for estimate in estimates_list
    ]
    actions_low = pass_fail_actions(estimates_list, labels, threshold=threshold_low)
    actions_high = pass_fail_actions(estimates_list, labels, threshold=threshold_high)
    for action_low, action_high in zip(actions_low, actions_high, strict=True):
        # A tighter threshold can only move units out of resale, never in.
        if action_low.startswith(f"{PATH_REUSE}_"):
            assert action_high.startswith(f"{PATH_REUSE}_")
