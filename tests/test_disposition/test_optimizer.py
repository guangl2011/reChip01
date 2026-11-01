"""Optimizer unit tests: valuations, argmax, review gates, traces (spec §4, §8).

Fixture hand-computations follow the engine's estimate contract
(``workload_fitness[w]`` = survival probability, headline interval tied to the
primary workload) and the optimizer's documented propagation model:
per-workload failure interval = headline CI width centered on ``1 - fitness[w]``.
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest

from rechip01.disposition import (
    PATH_HARVEST,
    PATH_MANUAL_REVIEW,
    PATH_RECYCLE,
    PATH_REPAIR,
    PATH_REUSE,
    DispositionConfig,
    EconomyAssumptions,
    PriceRange,
    optimize_disposition,
    point_option_values,
    value_harvest,
    value_recycle,
    value_repair,
    value_reuse,
    workload_failure_interval,
)
from rechip01.schemas import ReliabilityEstimate
from tests.test_disposition.helpers import make_economy, make_estimate

FIXTURE_VALUES = {
    "reuse_inference": 770.0,  # -50 + 0.9*1000 - 0.1*800
    "reuse_finetune": -50.0,  # -50 + 0.5*800 - 0.5*800
    "repair": 620.0,  # -50 - 150 + 0.9*1000 - 0.1*800
    "harvest": 200.0,  # -50 + 250
    "recycle": 75.0,
}


class TestValueFunctions:
    def test_value_reuse_hand_computed(self) -> None:
        draws = value_reuse(
            np.asarray([0.1]),
            np.asarray([1000.0]),
            test_cost=50.0,
            downtime_liability=800.0,
        )
        assert draws[0] == pytest.approx(770.0)

    def test_value_reuse_vectorized_over_draws(self) -> None:
        draws = value_reuse(
            np.asarray([0.0, 1.0]),
            np.asarray([1000.0, 1000.0]),
            test_cost=50.0,
            downtime_liability=800.0,
        )
        assert draws[0] == pytest.approx(950.0)
        assert draws[1] == pytest.approx(-850.0)

    def test_value_repair_sells_into_best_price(self) -> None:
        draws = value_repair(
            np.asarray([[1000.0, 1400.0]]),
            test_cost=50.0,
            repair_cost=150.0,
            residual_p_fail=0.1,
            downtime_liability=800.0,
        )
        assert draws[0] == pytest.approx(-50.0 - 150.0 + 0.9 * 1400.0 - 0.1 * 800.0)

    def test_value_harvest_and_recycle(self) -> None:
        assert value_harvest(np.asarray([250.0]), test_cost=50.0)[0] == pytest.approx(200.0)
        assert value_recycle(np.asarray([75.0]))[0] == pytest.approx(75.0)


class TestWorkloadFailureInterval:
    def test_degenerate_ci_gives_point_interval(self) -> None:
        estimate = make_estimate()  # fitness inference 0.90 -> p 0.10
        assert workload_failure_interval(estimate, "inference") == pytest.approx((0.10, 0.10, 0.10))

    def test_headline_interval_transfers_to_non_primary_workload(self) -> None:
        estimate = make_estimate(
            p_fail=0.50, ci=(0.45, 0.55), workload_fitness={"inference": 0.50, "finetune": 0.50}
        )
        # Headline width 0.10 centered on 1 - fitness[finetune] = 0.50.
        assert workload_failure_interval(estimate, "finetune") == pytest.approx((0.45, 0.50, 0.55))

    def test_primary_workload_reproduces_headline_interval(self) -> None:
        # Engine invariant: fitness[primary] = 1 - p_fail. Here the headline
        # p_fail is 0.20 with ci (0.15, 0.25); inference is the primary.
        estimate = make_estimate(p_fail=0.20, ci=(0.15, 0.25), workload_fitness={"inference": 0.80})
        assert workload_failure_interval(estimate, "inference") == pytest.approx((0.15, 0.20, 0.25))

    def test_interval_clipped_to_unit_range(self) -> None:
        estimate = make_estimate(ci=(0.0, 0.90), workload_fitness={"inference": 0.05})
        low, _, high = workload_failure_interval(estimate, "inference")
        assert low == pytest.approx(0.5)
        assert high == pytest.approx(1.0)

    def test_unknown_workload_rejected(self) -> None:
        with pytest.raises(KeyError):
            workload_failure_interval(make_estimate(), "memory-stress")


class TestPointOptionValues:
    def test_fixture_point_values(self) -> None:
        values = point_option_values(make_estimate(), make_economy())
        assert values == pytest.approx(FIXTURE_VALUES)

    def test_unknown_price_scenario_rejected(self) -> None:
        with pytest.raises(KeyError):
            point_option_values(make_estimate(), make_economy(), price_scenario="top")  # type: ignore[arg-type]

    def test_downtime_multiplier_stresses_failure_cost(self) -> None:
        base = point_option_values(make_estimate(), make_economy())
        stressed = point_option_values(make_estimate(), make_economy(), downtime_multiplier=2.0)
        # Only failure-exposed paths move; harvest/recycle are liability-free.
        assert stressed["harvest"] == pytest.approx(base["harvest"])
        assert stressed["recycle"] == pytest.approx(base["recycle"])
        assert stressed["reuse_inference"] == pytest.approx(770.0 - 0.1 * 800.0)
        assert stressed["repair"] == pytest.approx(620.0 - 0.1 * 800.0)

    def test_scenario_price_selection(self) -> None:
        economy = make_economy(
            resale_price_by_workload={
                "inference": PriceRange(low=800.0, high=1200.0),
                "finetune": PriceRange(low=600.0, high=1000.0),
            }
        )
        estimate = make_estimate(workload_fitness={"inference": 1.0, "finetune": 1.0})
        low = point_option_values(estimate, economy, price_scenario="low")
        mid = point_option_values(estimate, economy, price_scenario="mid")
        high = point_option_values(estimate, economy, price_scenario="high")
        assert low["reuse_inference"] == pytest.approx(-50.0 + 800.0)
        assert mid["reuse_inference"] == pytest.approx(-50.0 + 1000.0)
        assert high["reuse_inference"] == pytest.approx(-50.0 + 1200.0)


def _brute_force_values(
    estimate: ReliabilityEstimate, economy: EconomyAssumptions
) -> dict[str, float]:
    """Reference ENR table: point probabilities, insertion order as the optimizer."""
    workloads = sorted(estimate.workload_fitness)
    values: dict[str, float] = {}
    for workload in workloads:
        p_fail_workload = 1.0 - estimate.workload_fitness[workload]
        price = economy.resale_price_by_workload[workload].midpoint()
        values[f"{PATH_REUSE}_{workload}"] = (
            -economy.test_cost
            + (1.0 - p_fail_workload) * price
            - p_fail_workload * economy.downtime_liability
        )
    best_price = max(economy.resale_price_by_workload[w].midpoint() for w in workloads)
    residual = economy.repair_residual_p_fail
    values[PATH_REPAIR] = (
        -economy.test_cost
        - economy.repair_cost
        + (1.0 - residual) * best_price
        - residual * economy.downtime_liability
    )
    values[PATH_HARVEST] = -economy.test_cost + economy.harvest_recovery.midpoint()
    values[PATH_RECYCLE] = economy.recycling_value.midpoint()
    return values


class TestArgmaxVsBruteForce:
    @pytest.mark.parametrize("residual", [0.05, 0.35, 0.9])
    @pytest.mark.parametrize("harvest_recovery", [0.0, 1200.0])
    @pytest.mark.parametrize("repair_cost", [0.0, 400.0])
    @pytest.mark.parametrize("inference_price", [1000.0, 1800.0])
    def test_argmax_matches_brute_force_on_small_grid(
        self,
        inference_price: float,
        repair_cost: float,
        harvest_recovery: float,
        residual: float,
    ) -> None:
        economy = make_economy(
            resale_price_by_workload={
                "inference": PriceRange(low=inference_price, high=inference_price),
                "finetune": PriceRange(low=800.0, high=800.0),
            },
            repair_cost=repair_cost,
            repair_residual_p_fail=residual,
            harvest_recovery=PriceRange(low=harvest_recovery, high=harvest_recovery),
        )
        estimate = make_estimate()
        decision = optimize_disposition(estimate, economy, seed=0)
        values = _brute_force_values(estimate, economy)
        expected = max(values, key=lambda path: values[path])
        assert decision.recommendation == expected, (values, decision.recommendation)

    def test_hand_computed_fixture_case(self) -> None:
        decision = optimize_disposition(make_estimate(), make_economy(), seed=0)
        assert decision.recommendation == f"{PATH_REUSE}_inference"
        assert decision.expected_net_recovery == pytest.approx(770.0)
        assert decision.regret_bound == pytest.approx(150.0)  # 770 - repair 620


class TestOptionEvaluations:
    def test_all_five_options_with_regret_bounds(self) -> None:
        decision = optimize_disposition(make_estimate(), make_economy(), seed=0)
        by_path = {option.path: option for option in decision.options_evaluated}
        assert set(by_path) == set(FIXTURE_VALUES)
        for path, value in FIXTURE_VALUES.items():
            assert by_path[path].expected_net_recovery == pytest.approx(value)
            assert by_path[path].regret_bound == pytest.approx(
                max(0.0, FIXTURE_VALUES["reuse_inference"] - value)
            )

    def test_estimate_metadata_propagates(self) -> None:
        decision = optimize_disposition(make_estimate(), make_economy(), seed=0)
        assert decision.asset_id == "gpu-0001"
        assert decision.model_version == "test-engine-v0"
        assert str(decision.provenance) == "simulated"

    def test_economy_missing_workload_price_rejected(self) -> None:
        economy = make_economy(
            resale_price_by_workload={"inference": PriceRange(low=1000.0, high=1000.0)}
        )
        with pytest.raises(ValueError, match="lacks resale prices"):
            optimize_disposition(make_estimate(), economy, seed=0)

    def test_estimate_without_fitness_rejected(self) -> None:
        with pytest.raises(ValueError, match="no workload fitness"):
            optimize_disposition(make_estimate(workload_fitness={}), make_economy(), seed=0)


class TestReviewGates:
    def test_ood_estimates_route_to_manual_review(self) -> None:
        decision = optimize_disposition(make_estimate(ood_flag=True), make_economy(), seed=0)
        assert decision.recommendation == PATH_MANUAL_REVIEW
        assert any("ood_flag set" in line for line in decision.assumption_trace)

    def test_wide_ci_routes_to_manual_review(self) -> None:
        estimate = make_estimate(ci=(0.05, 0.45))  # width 0.40 > 0.35 default
        decision = optimize_disposition(estimate, make_economy(), seed=0)
        assert decision.recommendation == PATH_MANUAL_REVIEW
        assert any("exceeds threshold 0.3500" in line for line in decision.assumption_trace)

    def test_custom_threshold_tightens_review(self) -> None:
        estimate = make_estimate(ci=(0.05, 0.15))  # width 0.10
        config = DispositionConfig(max_ci_width=0.05)
        decision = optimize_disposition(estimate, make_economy(), config=config, seed=0)
        assert decision.recommendation == PATH_MANUAL_REVIEW

    def test_high_risk_with_tight_ci_still_recommends(self) -> None:
        # Review gates are uncertainty gates, not risk gates: a risky unit with
        # a calibrated tight interval still gets an automatic recommendation.
        estimate = make_estimate(
            p_fail=0.60,
            ci=(0.60, 0.60),
            workload_fitness={"inference": 0.40, "finetune": 0.05},
        )
        decision = optimize_disposition(estimate, make_economy(), seed=0)
        assert decision.recommendation == PATH_REPAIR  # 620 vs 200 harvest vs negative reuse

    def test_review_carries_review_cost(self) -> None:
        config = DispositionConfig(review_cost=42.0)
        decision = optimize_disposition(
            make_estimate(ood_flag=True), make_economy(), config=config, seed=0
        )
        assert decision.recommendation == PATH_MANUAL_REVIEW
        assert decision.expected_net_recovery == pytest.approx(770.0 - 42.0)

    def test_review_option_present_in_evaluation_table(self) -> None:
        decision = optimize_disposition(make_estimate(ood_flag=True), make_economy(), seed=0)
        paths = {option.path for option in decision.options_evaluated}
        assert PATH_MANUAL_REVIEW in paths
        assert set(FIXTURE_VALUES).issubset(paths)


class TestEvidenceTrace:
    def test_trace_cites_estimate_and_economy(self) -> None:
        decision = optimize_disposition(make_estimate(), make_economy(), seed=0)
        trace = "\n".join(decision.assumption_trace)
        assert "test-engine-v0" in trace
        assert "p_fail=0.1000" in trace
        assert "ci=[0.1000, 0.1000]" in trace
        assert "ood=False" in trace
        assert "resale_price_by_workload[inference] = [1000.00, 1000.00]" in trace
        assert "test_cost = 50.00" in trace
        assert "downtime_liability = 800.00" in trace
        assert "repair_cost = 150.00" in trace
        assert "harvest_recovery = [250.00, 250.00]" in trace
        assert "recycling_value = [75.00, 75.00]" in trace

    def test_trace_contains_sensitivity_and_regret_lines(self) -> None:
        decision = optimize_disposition(make_estimate(), make_economy(), seed=0)
        trace = decision.assumption_trace
        assert any(line.startswith("sensitivity: at price-low") for line in trace)
        assert any(
            line.startswith("sensitivity: at 2x downtime_liability") and "(holds vs" in line
            for line in trace
        )
        assert any(line.startswith("regret bound:") and "150.00" in line for line in trace)

    def test_sensitivity_flip_recorded_when_pick_changes(self) -> None:
        # Repair's residual risk (0.05) is lower than reuse_inference's
        # (0.10): doubling the downtime liability costs reuse 80 USD more
        # than repair, which closes the 35 USD mid-price margin and flips
        # the pick to repair.
        economy = make_economy(repair_cost=125.0, repair_residual_p_fail=0.05)
        decision = optimize_disposition(make_estimate(), economy, seed=0)
        assert decision.recommendation == f"{PATH_REUSE}_inference"
        assert any(
            "sensitivity: at 2x downtime_liability the recommendation is repair"
            " (changes vs mid-price pick reuse_inference)" in line
            for line in decision.assumption_trace
        )

    def test_trace_lists_every_valuation(self) -> None:
        decision = optimize_disposition(make_estimate(), make_economy(), seed=0)
        valuation_lines = [line for line in decision.assumption_trace if "ENR mean=" in line]
        assert len(valuation_lines) == len(FIXTURE_VALUES)

    def test_triggers_recorded_in_trace(self) -> None:
        decision = optimize_disposition(make_estimate(ood_flag=True), make_economy(), seed=0)
        assert any(
            "manual_review" in line and "ood_flag set" in line for line in decision.assumption_trace
        )


class TestDeterminismAndMonotonicity:
    def test_identical_seed_reproduces_decision(self) -> None:
        economy = make_economy(
            resale_price_by_workload={
                "inference": PriceRange(low=800.0, high=1200.0),
                "finetune": PriceRange(low=600.0, high=1000.0),
            }
        )
        first = optimize_disposition(make_estimate(ci=(0.05, 0.15)), economy, seed=7)
        second = optimize_disposition(make_estimate(ci=(0.05, 0.15)), economy, seed=7)
        assert first == second

    def test_different_seed_stable_recommendation(self) -> None:
        economy = make_economy(
            resale_price_by_workload={
                "inference": PriceRange(low=800.0, high=1200.0),
                "finetune": PriceRange(low=600.0, high=1000.0),
            }
        )
        estimates = [
            optimize_disposition(make_estimate(ci=(0.05, 0.15)), economy, seed=seed)
            for seed in (0, 1, 2)
        ]
        assert {decision.recommendation for decision in estimates} == {f"{PATH_REUSE}_inference"}

    def test_mc_mean_converges_to_point_expectation(self) -> None:
        # Independent uniform price draws and triangular failure draws: the MC
        # mean equals the point valuation with the interval's mean probability.
        economy = make_economy(
            resale_price_by_workload={
                "inference": PriceRange(low=800.0, high=1200.0),
                "finetune": PriceRange(low=800.0, high=800.0),
            }
        )
        estimate = make_estimate(ci=(0.05, 0.15))  # triangular mean p = 0.10
        decision = optimize_disposition(estimate, economy, seed=11)
        by_path = {option.path: option for option in decision.options_evaluated}
        assert by_path["reuse_inference"].expected_net_recovery == pytest.approx(770.0, abs=15.0)

    def test_reuse_enr_decreases_as_fitness_drops(self) -> None:
        healthy = optimize_disposition(
            make_estimate(), make_economy(), seed=0
        )  # fitness inference 0.90
        degraded = optimize_disposition(
            make_estimate(
                p_fail=0.60, ci=(0.60, 0.60), workload_fitness={"inference": 0.40, "finetune": 0.05}
            ),
            make_economy(),
            seed=0,
        )
        by_path_healthy = {o.path: o for o in healthy.options_evaluated}
        by_path_degraded = {o.path: o for o in degraded.options_evaluated}
        assert (
            by_path_degraded["reuse_inference"].expected_net_recovery
            < by_path_healthy["reuse_inference"].expected_net_recovery
        )

    def test_wider_p_spread_widens_value_spread(self) -> None:
        tight = value_reuse(
            np.asarray([0.08, 0.12]),
            np.asarray([1000.0, 1000.0]),
            test_cost=50.0,
            downtime_liability=800.0,
        )
        wide = value_reuse(
            np.asarray([0.0, 0.2]),
            np.asarray([1000.0, 1000.0]),
            test_cost=50.0,
            downtime_liability=800.0,
        )
        assert wide.max() - wide.min() > tight.max() - tight.min()


def test_grid_cells_cover_all_path_winners() -> None:
    """The parametrized grid must exercise every economic path as a winner."""
    winners: set[str] = set()
    for inference_price, repair_cost, harvest_recovery, residual in itertools.product(
        [1000.0, 1800.0], [0.0, 400.0], [0.0, 1200.0], [0.05, 0.35, 0.9]
    ):
        economy = make_economy(
            resale_price_by_workload={
                "inference": PriceRange(low=inference_price, high=inference_price),
                "finetune": PriceRange(low=800.0, high=800.0),
            },
            repair_cost=repair_cost,
            repair_residual_p_fail=residual,
            harvest_recovery=PriceRange(low=harvest_recovery, high=harvest_recovery),
        )
        estimate = make_estimate()
        values = _brute_force_values(estimate, economy)
        winners.add(max(values, key=lambda path: values[path]))
    assert {PATH_REUSE + "_inference", PATH_REPAIR, PATH_HARVEST}.issubset(winners)
