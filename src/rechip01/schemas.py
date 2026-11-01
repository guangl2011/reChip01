"""Core schema contracts (spec art_vuIPvUhR, §4).

These models are the integration boundary between the scanner, simulator,
engine, disposition optimizer, and passport layers. Three invariants hold by
construction:

1. Every model that carries numbers carries a required ``provenance`` field —
   simulated and measured data are structurally distinct and can never be
   merged into one undifferentiated pool.
2. Every model is frozen: provenance is stamped once at ingest and cannot be
   mutated downstream. Immutability is enforced in code, not by convention.
3. Uncertainty is mandatory wherever a probability is reported: a point
   estimate must always sit inside a calibrated interval.
"""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

# A probability or normalized percentile: finite, within [0, 1].
UnitInterval = Annotated[float, Field(ge=0.0, le=1.0, allow_inf_nan=False)]

# A finite float with no range assumption (e.g. money, which may be negative).
FiniteFloat = Annotated[float, Field(allow_inf_nan=False)]

# A non-negative finite float (e.g. hours, months, regret bounds).
NonNegativeFloat = Annotated[float, Field(ge=0.0, allow_inf_nan=False)]


class Provenance(StrEnum):
    """Data origin. The field-level honesty invariant of the whole platform."""

    SIMULATED = "simulated"
    MEASURED = "measured"


class _FrozenContract(BaseModel):
    """Base for all schema contracts: immutable after construction."""

    model_config = ConfigDict(frozen=True)


class ScanResult(_FrozenContract):
    """Per-unit normalized feature record produced at ingest (spec §4).

    Emitted by the scanner for both simulator feeds and real partner exports;
    downstream layers cannot tell the two apart except via ``provenance``.
    Optional fields (``age_months``, ``power_on_hours``) are ``None`` in
    partial mode — missingness is explicit, never silently imputed.
    """

    asset_id: str = Field(min_length=1)
    model: str = Field(min_length=1, description='GPU model, e.g. "A100-SXM4-80GB"')
    provenance: Provenance
    age_months: NonNegativeFloat | None = None
    power_on_hours: NonNegativeFloat | None = None
    ecc_events: int = Field(ge=0)
    xid_errors: int = Field(ge=0)
    retired_pages: int = Field(ge=0)
    thermal_history: list[UnitInterval] = Field(
        description="Normalized temperature percentiles, each within [0, 1]"
    )
    max_sustained_power_pct: UnitInterval = Field(
        description="Maximum sustained power draw, normalized to [0, 1]"
    )
    bench_mark_snapshot: dict[str, FiniteFloat] = Field(
        description="DCGM-class diagnostic readings keyed by metric name"
    )
    chain_of_custody: list[str] = Field(
        min_length=1,
        description="Custody events from intake onward; required for passport issuance",
    )


class ReliabilityEstimate(_FrozenContract):
    """Uncertainty-aware reliability prediction for one unit (spec §4)."""

    asset_id: str = Field(min_length=1)
    horizon_days: int = Field(ge=1)
    provenance: Provenance
    p_fail: UnitInterval = Field(description="Calibrated probability of failure within the horizon")
    ci: tuple[UnitInterval, UnitInterval] = Field(
        description="Calibrated interval (low, high) that must bracket p_fail"
    )
    workload_fitness: dict[str, UnitInterval] = Field(
        description='Fitness per target workload, e.g. {"inference": 0.83}'
    )
    ood_flag: bool = Field(
        description="True when the unit is outside the model's training support; "
        "never scored silently"
    )
    model_version: str = Field(min_length=1)

    @model_validator(mode="after")
    def _ci_brackets_p_fail(self) -> ReliabilityEstimate:
        low, high = self.ci
        if not low <= self.p_fail <= high:
            raise ValueError(f"ci must bracket p_fail: ci=({low}, {high}), p_fail={self.p_fail}")
        return self


class OptionEvaluation(_FrozenContract):
    """Risk-adjusted outcome of one disposition path considered for a unit."""

    path: str = Field(min_length=1)
    expected_net_recovery: FiniteFloat
    regret_bound: NonNegativeFloat


class DispositionDecision(_FrozenContract):
    """Risk-adjusted disposition recommendation for one unit (spec §4)."""

    asset_id: str = Field(min_length=1)
    provenance: Provenance
    recommendation: str = Field(min_length=1)
    expected_net_recovery: FiniteFloat = Field(
        description="Risk-adjusted net recovery of the recommended path"
    )
    regret_bound: NonNegativeFloat = Field(
        description="How wrong the model can be before the recommendation flips"
    )
    options_evaluated: list[OptionEvaluation] = Field(min_length=1)
    assumption_trace: list[str] = Field(
        min_length=1,
        description="Price/cost assumptions cited in the passport",
    )
    model_version: str = Field(min_length=1)

    @model_validator(mode="after")
    def _recommendation_is_an_evaluated_option(self) -> DispositionDecision:
        evaluated = {option.path for option in self.options_evaluated}
        if self.recommendation not in evaluated:
            raise ValueError(
                f"recommendation {self.recommendation!r} is not among the evaluated "
                f"options {sorted(evaluated)}"
            )
        return self


class AssetPassport(_FrozenContract):
    """Auditable asset passport: the immutable assessment record (spec §4).

    Frozen at issuance — provenance, evidence trace, uncertainty, and model
    version can never be edited after the fact.
    """

    asset_id: str = Field(min_length=1)
    run_id: str = Field(min_length=1, description="Assessment run that produced this passport")
    provenance: Provenance
    issued_at: datetime
    reliability: ReliabilityEstimate
    disposition: DispositionDecision
    evidence_trace: list[str] = Field(
        min_length=1,
        description="Lineage entries: scan batch, model version, input hashes, ...",
    )
    model_version: str = Field(min_length=1)

    @field_validator("issued_at")
    @classmethod
    def _issued_at_is_timezone_aware(cls, value: datetime) -> datetime:
        if value.tzinfo is None:
            raise ValueError("issued_at must be timezone-aware")
        return value.astimezone(UTC)


__all__ = [
    "AssetPassport",
    "DispositionDecision",
    "FiniteFloat",
    "NonNegativeFloat",
    "OptionEvaluation",
    "Provenance",
    "ReliabilityEstimate",
    "ScanResult",
    "UnitInterval",
]
