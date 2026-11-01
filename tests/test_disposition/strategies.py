"""Reusable Hypothesis strategies for disposition tests (spec §4, §8)."""

from __future__ import annotations

from typing import Any

from hypothesis import strategies as st
from hypothesis.strategies import composite

from rechip01.disposition import EconomyAssumptions, PriceRange
from rechip01.schemas import Provenance, ReliabilityEstimate

ALL_WORKLOADS = ("inference", "finetune", "memory-stress")
# CI width kept below the 0.35 manual-review threshold so generated
# estimates still exercise recommendation paths.
MAX_CI_WIDTH = 0.30


@composite
def economies(draw: Any) -> EconomyAssumptions:
    """Valid synthetic economies: bounded, varied, non-negative prices/costs."""
    resale = {}
    for workload in ALL_WORKLOADS:
        low = draw(st.floats(200.0, 1500.0, allow_nan=False, allow_infinity=False))
        high = low + draw(st.floats(0.0, 500.0, allow_nan=False, allow_infinity=False))
        resale[workload] = PriceRange(low=low, high=high)
    harvest_low = draw(st.floats(50.0, 500.0, allow_nan=False, allow_infinity=False))
    recycle_low = draw(st.floats(10.0, 60.0, allow_nan=False, allow_infinity=False))
    return EconomyAssumptions(
        resale_price_by_workload=resale,
        test_cost=draw(st.floats(0.0, 200.0, allow_nan=False, allow_infinity=False)),
        repair_cost=draw(st.floats(0.0, 600.0, allow_nan=False, allow_infinity=False)),
        repair_residual_p_fail=draw(st.floats(0.0, 0.4, allow_nan=False, allow_infinity=False)),
        downtime_liability=draw(st.floats(0.0, 2000.0, allow_nan=False, allow_infinity=False)),
        harvest_recovery=PriceRange(
            low=harvest_low,
            high=harvest_low + draw(st.floats(0.0, 400.0, allow_nan=False, allow_infinity=False)),
        ),
        recycling_value=PriceRange(
            low=recycle_low,
            high=recycle_low + draw(st.floats(0.0, 90.0, allow_nan=False, allow_infinity=False)),
        ),
    )


@composite
def estimates(draw: Any) -> ReliabilityEstimate:
    """Valid in-distribution estimates; CI brackets p_fail, width under review gate."""
    n_workloads = draw(st.integers(1, len(ALL_WORKLOADS)))
    workloads = ALL_WORKLOADS[:n_workloads]
    p = draw(st.floats(0.05, 0.95, allow_nan=False, allow_infinity=False))
    width = draw(st.floats(0.0, MAX_CI_WIDTH, allow_nan=False, allow_infinity=False))
    low = max(0.0, p - draw(st.floats(0.0, 1.0)) * width)
    high = min(1.0, low + width)
    fitness = {
        workload: draw(st.floats(0.0, 1.0, allow_nan=False, allow_infinity=False))
        for workload in workloads
    }
    return ReliabilityEstimate(
        asset_id=f"gpu-{draw(st.integers(0, 10**6)):06d}",
        horizon_days=365,
        provenance=Provenance.SIMULATED,
        p_fail=p,
        ci=(low, high),
        workload_fitness=fitness,
        ood_flag=False,
        model_version="test-engine-v0",
    )
