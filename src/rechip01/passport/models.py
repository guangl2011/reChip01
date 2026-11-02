"""API contracts for the passport service (spec art_vuIPvUhR, §4–5).

Request and response models for the FastAPI layer. Every response model
carries provenance (per unit or per batch) and, wherever an assessment is
reported, the calibrated uncertainty interval — the simulated-vs-measured
distinction and the uncertainty bands ride on every screen, never behind a
flag.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from rechip01.disposition.config import EconomyAssumptions
from rechip01.scanner.models import SourceFormat
from rechip01.schemas import (
    FiniteFloat,
    NonNegativeFloat,
    Provenance,
    UnitInterval,
)


class _FrozenContract(BaseModel):
    """Base for API contracts: immutable after construction."""

    model_config = ConfigDict(frozen=True)


class ScanIngestRequest(_FrozenContract):
    """One ingest batch: declared provenance plus raw rows.

    ``provenance`` is the batch-level declaration (who asserts these rows'
    origin); the scanner quarantines rows that carry a conflicting
    per-row claim rather than silently trusting either. Rows are arbitrary
    JSON values — anything the scanner cannot normalize lands in the
    report's rejection list with a machine-readable reason.
    """

    provenance: Provenance
    source_format: SourceFormat = SourceFormat.IN_MEMORY
    rows: list[object] = Field(description="Raw rows; malformed rows are quarantined, not dropped")


class UnitOverview(_FrozenContract):
    """One stored unit on the batch overview.

    Assessment fields are ``None`` until the unit has been assessed; once a
    passport exists they carry its headline uncertainty and recommendation.
    """

    asset_id: str = Field(min_length=1)
    model: str = Field(min_length=1)
    provenance: Provenance
    assessed: bool
    p_fail: UnitInterval | None = None
    ci: tuple[UnitInterval, UnitInterval] | None = None
    ood_flag: bool | None = None
    recommendation: str | None = None


class BatchOverview(_FrozenContract):
    """Batch overview: every stored unit with provenance and, where issued,
    the passport's uncertainty and recommendation."""

    total_units: int = Field(ge=0)
    provenance_counts: dict[Provenance, int] = Field(
        description="Unit counts by provenance — simulated and measured never pooled"
    )
    units: tuple[UnitOverview, ...]


class AssessRequest(_FrozenContract):
    """One assessment run over stored scans.

    ``target_workloads`` are resolved against the built-in profiles; unknown
    names are a 422, never a silent skip. ``asset_ids`` selects a subset of
    the stored batch; ``None`` means every stored unit. ``run_id`` is
    optional: retries of the same assessment run replay safely into a 409
    instead of issuing a second passport for the same (asset, run).
    """

    horizon_days: int = Field(ge=1)
    target_workloads: tuple[str, ...] = Field(min_length=1)
    economy: EconomyAssumptions
    asset_ids: tuple[str, ...] | None = None
    run_id: str | None = Field(
        default=None, description="Client-supplied run id; generated when omitted"
    )


class AssessmentSummary(_FrozenContract):
    """Per-unit outcome of one assessment run (spec §5, batch report)."""

    asset_id: str = Field(min_length=1)
    provenance: Provenance
    p_fail: UnitInterval
    ci: tuple[UnitInterval, UnitInterval]
    ood_flag: bool
    recommendation: str = Field(min_length=1)
    expected_net_recovery: FiniteFloat
    regret_bound: NonNegativeFloat


class AssessResponse(_FrozenContract):
    """Outcome of one assessment run over the requested batch."""

    run_id: str = Field(min_length=1)
    horizon_days: int = Field(ge=1)
    target_workloads: tuple[str, ...] = Field(min_length=1)
    provenance_counts: dict[Provenance, int] = Field(
        description="Assessed-unit counts by provenance — simulated and measured never pooled"
    )
    items: tuple[AssessmentSummary, ...]


__all__ = [
    "AssessRequest",
    "AssessResponse",
    "AssessmentSummary",
    "BatchOverview",
    "ScanIngestRequest",
    "UnitOverview",
]
