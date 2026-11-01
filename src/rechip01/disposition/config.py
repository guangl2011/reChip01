"""Disposition configuration contracts (spec art_vuIPvUhR, §4).

The economy is an explicit, validated input — never a implicit default. Every
dollar figure here is an *assumption of the synthetic economy or the operator's
declared terms*; the optimizer cites each one it consumed in the decision's
``assumption_trace``. No number in this module is a market observation.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, model_validator

from rechip01.schemas import NonNegativeFloat, UnitInterval


class _FrozenContract(BaseModel):
    """Base for disposition contracts: immutable after construction."""

    model_config = ConfigDict(frozen=True)


class PriceRange(_FrozenContract):
    """Inclusive money range for one uncertain value (e.g. resale price)."""

    low: NonNegativeFloat
    high: NonNegativeFloat = Field(description="Upper bound; must not be below low")

    @model_validator(mode="after")
    def _ordered(self) -> PriceRange:
        if self.low > self.high:
            raise ValueError(f"price range must be ordered: low {self.low} > high {self.high}")
        return self

    def midpoint(self) -> float:
        """Midpoint of the range — the deterministic realization price."""
        return (self.low + self.high) / 2.0


class EconomyAssumptions(_FrozenContract):
    """Declared economic parameters for one disposition decision (spec §4).

    Reuse resale prices are per target workload: the workload-conditioned
    price a working unit fetches when deployed into that regime. ``test_cost``
    covers functional triage and applies to reuse, repair, and harvesting;
    certified recycling needs no test. ``downtime_liability`` is the
    seller-borne cost (warranty, return, replacement) when a resold unit
    fails within the assessment horizon.
    """

    resale_price_by_workload: dict[str, PriceRange] = Field(min_length=1)
    test_cost: NonNegativeFloat
    repair_cost: NonNegativeFloat
    repair_residual_p_fail: UnitInterval = Field(
        description="Failure risk remaining after a repair attempt"
    )
    downtime_liability: NonNegativeFloat
    harvest_recovery: PriceRange = Field(description="Component recovery value range")
    recycling_value: PriceRange = Field(description="Certified recycling net value range")


class DispositionConfig(_FrozenContract):
    """Review thresholds and Monte Carlo settings for the optimizer.

    ``max_ci_width`` is a conservative calibration threshold per spec §5:
    units whose calibrated interval is wider are routed to manual review
    instead of receiving an automatic recommendation. Set conservatively at
    first (typical in-distribution engine widths are far below it) and tuned
    only against the simulator's held-out ground truth — never against
    training performance.
    """

    max_ci_width: UnitInterval = Field(
        default=0.35,
        description="Headline CI width above which a unit routes to manual review",
    )
    review_cost: NonNegativeFloat = Field(
        default=25.0, description="Cost of one manual review (deferral is not free)"
    )
    n_samples: int = Field(
        default=2048, ge=1, description="Monte Carlo draws per uncertainty propagation"
    )


__all__ = [
    "DispositionConfig",
    "EconomyAssumptions",
    "PriceRange",
]
