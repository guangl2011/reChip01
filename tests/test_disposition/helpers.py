"""Shared builders for disposition tests (spec art_vuIPvUhR, §4).

The default economy and estimate are deliberately degenerate (zero-width
price ranges and CI) so unit-test valuations are hand-checkable; the
Hypothesis strategies in ``strategies.py`` provide varied non-degenerate
inputs for the property tests.
"""

from __future__ import annotations

from typing import Any

from rechip01.disposition import EconomyAssumptions, PriceRange
from rechip01.schemas import Provenance, ReliabilityEstimate


def make_economy(**overrides: Any) -> EconomyAssumptions:
    """Two-workload degenerate economy: hand-computable option values."""
    base: dict[str, Any] = dict(
        resale_price_by_workload={
            "inference": PriceRange(low=1000.0, high=1000.0),
            "finetune": PriceRange(low=800.0, high=800.0),
        },
        test_cost=50.0,
        repair_cost=150.0,
        repair_residual_p_fail=0.10,
        downtime_liability=800.0,
        harvest_recovery=PriceRange(low=250.0, high=250.0),
        recycling_value=PriceRange(low=75.0, high=75.0),
    )
    base.update(overrides)
    return EconomyAssumptions(**base)


def make_estimate(**overrides: Any) -> ReliabilityEstimate:
    """Degenerate-CI estimate: p_fail 0.10, fitness 0.90/0.50, in-distribution.

    The fitness of the primary workload (inference) is kept engine-consistent:
    ``1 - p_fail``. The secondary workload's fitness is deliberately
    inconsistent with the headline interval — the optimizer's propagation
    model centers each workload's interval on its own ``1 - fitness``.
    """
    base: dict[str, Any] = dict(
        asset_id="gpu-0001",
        horizon_days=365,
        provenance=Provenance.SIMULATED,
        p_fail=0.10,
        ci=(0.10, 0.10),
        workload_fitness={"inference": 0.90, "finetune": 0.50},
        ood_flag=False,
        model_version="test-engine-v0",
    )
    base.update(overrides)
    return ReliabilityEstimate(**base)
