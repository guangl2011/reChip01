"""Row validation, normalization, and provenance stamping (spec §4).

``scan_batch`` is the single ingest interface: fleet-simulator feeds and
partner CSV/JSONL exports arrive as the same row-mapping shape and leave as
schema-valid :class:`ScanResult` records — differing only in the declared
provenance and the recorded source format.

Invariants:

- The declared source provenance is the only authority for stamping. A row
  that claims a *different* provenance is quarantined, never re-stamped.
- Missing required fields (identity, model, chain of custody) quarantine the
  row with machine-readable reasons; no silent coercion.
- Missing non-critical fields enter partial mode: an explicit default is
  applied and the field name is recorded on the accepted unit. Nothing is
  imputed silently.
- Rows outside the schema (e.g. outcome-feedback columns) are ignored in the
  MVP; they never affect the assessment.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Mapping
from typing import Final, NamedTuple

from pydantic import ValidationError

from rechip01.scanner.models import (
    AcceptedUnit,
    RejectedUnit,
    RejectionCode,
    RejectionReason,
    ScanBatchReport,
    SourceFormat,
)
from rechip01.schemas import Provenance, ScanResult

# Fields that must be present and non-empty; their absence quarantines a row
# (spec §5: "Schema violation or missing chain-of-custody" → Rejected).
_REQUIRED_FIELDS: Final[frozenset[str]] = frozenset({"asset_id", "model", "chain_of_custody"})

# Schema field order, used to keep missingness flags deterministic.
_FIELD_ORDER: Final[tuple[str, ...]] = (
    "asset_id",
    "model",
    "age_months",
    "power_on_hours",
    "ecc_events",
    "xid_errors",
    "retired_pages",
    "thermal_history",
    "max_sustained_power_pct",
    "bench_mark_snapshot",
    "chain_of_custody",
)

# Missing non-critical fields get an explicit default (recorded, never silent).
_FIELD_DEFAULTS: Final[dict[str, object]] = {
    "ecc_events": 0,
    "xid_errors": 0,
    "retired_pages": 0,
    "thermal_history": [],
    "max_sustained_power_pct": 0.0,
    "bench_mark_snapshot": {},
}

# Optional-schema fields that stay None when missing (explicit missingness).
_MAY_BE_NONE: Final[frozenset[str]] = frozenset({"age_months", "power_on_hours"})

_KNOWN_FIELDS: Final[frozenset[str]] = frozenset(_FIELD_ORDER)

_INT_FIELDS: Final[frozenset[str]] = frozenset({"ecc_events", "xid_errors", "retired_pages"})
_FLOAT_FIELDS: Final[frozenset[str]] = frozenset(
    {"age_months", "power_on_hours", "max_sustained_power_pct"}
)
_JSON_FIELDS: Final[frozenset[str]] = frozenset({"thermal_history", "bench_mark_snapshot"})

# Accepted spellings for partner export headers, after normalization.
_HEADER_ALIASES: Final[dict[str, str]] = {
    "asset": "asset_id",
    "serial": "asset_id",
    "serial_number": "asset_id",
    "gpu_model": "model",
    "part_number": "model",
    "hours": "power_on_hours",
    "ecc": "ecc_events",
    "xid": "xid_errors",
    "thermal": "thermal_history",
    "max_power_pct": "max_sustained_power_pct",
    "benchmarks": "bench_mark_snapshot",
    "test_evidence": "bench_mark_snapshot",
    "custody": "chain_of_custody",
}


class _Prepared(NamedTuple):
    """Result of preparing one row: at most one of result/reasons is set."""

    result: ScanResult | None
    reasons: tuple[RejectionReason, ...]
    missing_fields: tuple[str, ...]


def _normalize_key(key: str) -> str:
    """Normalize an export header to a schema field name."""
    slug = key.strip().lower().replace(" ", "_").replace("-", "_")
    return _HEADER_ALIASES.get(slug, slug)


def _is_empty(value: object) -> bool:
    """Blank strings and empty containers count as absent."""
    if value is None:
        return True
    if isinstance(value, str):
        return not value.strip()
    if isinstance(value, (list, dict, tuple)):
        return len(value) == 0
    return False


def _coerce(field: str, value: object) -> tuple[object | None, RejectionReason | None]:
    """Coerce a raw cell (typically CSV text) toward its schema type."""
    if not isinstance(value, str):
        return value, None
    text = value.strip()
    try:
        if field in _INT_FIELDS:
            return int(text), None
        if field in _FLOAT_FIELDS:
            return float(text), None
    except ValueError:
        return None, RejectionReason(
            code=RejectionCode.PARSE_ERROR,
            field=field,
            detail=f"cannot parse {text!r} as {'integer' if field in _INT_FIELDS else 'number'}",
        )
    if field in _JSON_FIELDS:
        return _coerce_json(field, text)
    if field == "chain_of_custody":
        return _coerce_custody(text), None
    return text, None  # asset_id, model and any other text field


def _coerce_json(field: str, text: str) -> tuple[object | None, RejectionReason | None]:
    """Parse a structured cell: JSON first, separators as a thermal fallback."""
    try:
        return json.loads(text), None
    except json.JSONDecodeError as exc:
        if field == "thermal_history":
            parts = [part.strip() for part in re.split(r"[;|]", text) if part.strip()]
            try:
                return [float(part) for part in parts], None
            except ValueError:
                pass
        return None, RejectionReason(
            code=RejectionCode.PARSE_ERROR,
            field=field,
            detail=f"invalid JSON for {field!r}: {exc.msg}",
        )


def _coerce_custody(text: str) -> object:
    """Chain-of-custody cells accept a JSON list or ';'-/'|'-separated events."""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return [part.strip() for part in re.split(r"[;|]", text) if part.strip()]


def _provenance_conflict(row: Mapping[str, object], declared: Provenance) -> RejectionReason | None:
    """A row may agree with the declared source provenance, or stay silent —
    it may never claim a different one."""
    raw = row.get("provenance")
    if _is_empty(raw):
        return None
    try:
        claimed = Provenance(str(raw))
    except ValueError:
        return RejectionReason(
            code=RejectionCode.PARSE_ERROR,
            field="provenance",
            detail=f"unrecognized provenance value {raw!r}",
        )
    if claimed is not declared:
        return RejectionReason(
            code=RejectionCode.PROVENANCE_CONFLICT,
            field="provenance",
            detail=(
                f"row claims provenance {claimed.value!r} "
                f"but the batch is declared {declared.value!r}"
            ),
        )
    return None


def _prepare(row: Mapping[str, object], declared: Provenance) -> _Prepared:
    """Validate, normalize, and stamp one normalized row."""
    conflict = _provenance_conflict(row, declared)
    if conflict is not None:
        return _Prepared(None, (conflict,), ())

    cleaned: dict[str, object] = {}
    reasons: list[RejectionReason] = []
    for key, value in row.items():
        if key not in _KNOWN_FIELDS or _is_empty(value):
            continue  # unknown columns and blanks handled below
        coerced, error = _coerce(key, value)
        if error is not None:
            reasons.append(error)
        else:
            cleaned[key] = coerced

    for field in sorted(_REQUIRED_FIELDS):
        if _is_empty(cleaned.get(field)):
            reasons.append(
                RejectionReason(
                    code=RejectionCode.MISSING_REQUIRED_FIELD,
                    field=field,
                    detail=f"required field {field!r} is missing or empty",
                )
            )
    if reasons:
        return _Prepared(None, tuple(reasons), ())

    missing_fields: list[str] = []
    for field in _FIELD_ORDER:
        if field in cleaned or field in _REQUIRED_FIELDS:
            continue
        if field in _FIELD_DEFAULTS:
            cleaned[field] = _FIELD_DEFAULTS[field]
            missing_fields.append(field)
        elif field in _MAY_BE_NONE:
            cleaned[field] = None
            missing_fields.append(field)

    try:
        result = ScanResult.model_validate({**cleaned, "provenance": declared})
    except ValidationError as exc:
        for violation in exc.errors():
            loc = ".".join(str(part) for part in violation["loc"])
            reasons.append(
                RejectionReason(
                    code=RejectionCode.SCHEMA_VIOLATION,
                    field=loc or None,
                    detail=f"{violation['type']}: {violation['msg']}",
                )
            )
        return _Prepared(None, tuple(reasons), ())

    return _Prepared(result, (), tuple(missing_fields))


def _row_asset_id(row: Mapping[str, object]) -> str | None:
    value = row.get("asset_id")
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def scan_batch(
    rows: Iterable[object],
    provenance: Provenance,
    source_format: SourceFormat = SourceFormat.IN_MEMORY,
) -> ScanBatchReport:
    """Run one ingest batch through the scanner (spec §5 step 1).

    Accepts simulator feeds and partner exports through the same interface:
    rows are mappings keyed by schema field names (or export headers, which
    are normalized). Malformed rows are quarantined with machine-readable
    reasons; nothing invalid is ever accepted, and nothing accepted is ever
    silently incomplete.
    """
    accepted: list[AcceptedUnit] = []
    rejected: list[RejectedUnit] = []
    seen_assets: set[str] = set()
    total_rows = 0

    for index, row in enumerate(rows):
        total_rows += 1
        if not isinstance(row, Mapping):
            rejected.append(
                RejectedUnit(
                    row_index=index,
                    asset_id=None,
                    reasons=(
                        RejectionReason(
                            code=RejectionCode.PARSE_ERROR,
                            detail=f"row {index} is not an object/mapping",
                        ),
                    ),
                )
            )
            continue

        normalized = {_normalize_key(str(key)): value for key, value in row.items()}
        prepared = _prepare(normalized, provenance)
        if prepared.result is None:
            rejected.append(
                RejectedUnit(
                    row_index=index,
                    asset_id=_row_asset_id(normalized),
                    reasons=prepared.reasons,
                )
            )
            continue

        if prepared.result.asset_id in seen_assets:
            rejected.append(
                RejectedUnit(
                    row_index=index,
                    asset_id=prepared.result.asset_id,
                    reasons=(
                        RejectionReason(
                            code=RejectionCode.DUPLICATE_ASSET_ID,
                            field="asset_id",
                            detail=(
                                f"asset {prepared.result.asset_id!r} already appeared "
                                "earlier in this batch"
                            ),
                        ),
                    ),
                )
            )
            continue

        seen_assets.add(prepared.result.asset_id)
        accepted.append(
            AcceptedUnit(
                row_index=index,
                result=prepared.result,
                missing_fields=prepared.missing_fields,
            )
        )

    return ScanBatchReport(
        provenance=provenance,
        source_format=source_format,
        total_rows=total_rows,
        accepted=tuple(accepted),
        rejected=tuple(rejected),
    )


__all__ = ["scan_batch"]
