"""Scanner output contracts: accepted units, quarantined rows, batch report.

Every row a batch contains lands in exactly one of two buckets:

- ``AcceptedUnit`` — a schema-valid :class:`ScanResult` plus the explicit list
  of fields that were missing from the input and were default-filled (never
  silently: the flag rides on the record).
- ``RejectedUnit`` — a quarantined row with one or more machine-readable
  rejection reasons (a stable ``code``, the offending ``field`` if any, and a
  human-readable ``detail``).
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field

from rechip01.schemas import Provenance, ScanResult


class _ImmutableRecord(BaseModel):
    """Frozen scanner output: batch reports are audit records, never edited."""

    model_config = ConfigDict(frozen=True)


class RejectionCode(StrEnum):
    """Stable, machine-readable rejection categories."""

    MISSING_REQUIRED_FIELD = "missing_required_field"
    SCHEMA_VIOLATION = "schema_violation"
    PARSE_ERROR = "parse_error"
    PROVENANCE_CONFLICT = "provenance_conflict"
    DUPLICATE_ASSET_ID = "duplicate_asset_id"


class SourceFormat(StrEnum):
    """Where a batch's rows came from; recorded on the report for lineage."""

    SIMULATOR_FEED = "simulator_feed"
    CSV_EXPORT = "csv_export"
    JSONL_EXPORT = "jsonl_export"
    IN_MEMORY = "in_memory"


class RejectionReason(_ImmutableRecord):
    """One machine-readable reason a row was quarantined."""

    code: RejectionCode
    field: str | None = Field(default=None, description="Offending field, if identifiable")
    detail: str = Field(min_length=1)


class AcceptedUnit(_ImmutableRecord):
    """A schema-valid unit, with explicit missingness flags for partial input."""

    row_index: int = Field(ge=0)
    result: ScanResult
    missing_fields: tuple[str, ...] = Field(
        default=(),
        description="Schema fields absent from the input row and default-filled; never silent",
    )


class RejectedUnit(_ImmutableRecord):
    """A quarantined row. Invalid input never proceeds silently."""

    row_index: int = Field(ge=0)
    asset_id: str | None = Field(default=None, description="Best-effort ID, for operator reference")
    reasons: tuple[RejectionReason, ...] = Field(min_length=1)


class ScanBatchReport(_ImmutableRecord):
    """Outcome of scanning one ingest batch (spec §5, batch report)."""

    provenance: Provenance
    source_format: SourceFormat
    total_rows: int = Field(ge=0)
    accepted: tuple[AcceptedUnit, ...]
    rejected: tuple[RejectedUnit, ...]

    @property
    def accepted_count(self) -> int:
        return len(self.accepted)

    @property
    def rejected_count(self) -> int:
        return len(self.rejected)


__all__ = [
    "AcceptedUnit",
    "Provenance",
    "RejectedUnit",
    "RejectionCode",
    "RejectionReason",
    "ScanBatchReport",
    "ScanResult",
    "SourceFormat",
]
