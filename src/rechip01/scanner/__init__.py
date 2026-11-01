"""Scanner /ingest: inventory and telemetry ingest, validation, normalization,
and provenance stamping (spec art_vuIPvUhR, §4).

Public surface:

- :func:`scan_batch` — the single ingest interface for simulator feeds and
  partner exports (row mappings in, batch report out)
- :func:`load_rows` — CSV/JSONL file loading into raw rows
- :func:`persist_accepted` — write accepted records to a scan repository
- report contracts: :class:`ScanBatchReport`, :class:`AcceptedUnit`,
  :class:`RejectedUnit`, :class:`RejectionReason`, :class:`RejectionCode`
"""

from __future__ import annotations

from rechip01.repository import DuplicateScanError, ScanResultRepository
from rechip01.scanner.file_io import load_rows
from rechip01.scanner.ingest import scan_batch
from rechip01.scanner.models import (
    AcceptedUnit,
    RejectedUnit,
    RejectionCode,
    RejectionReason,
    ScanBatchReport,
    SourceFormat,
)
from rechip01.schemas import Provenance, ScanResult


def persist_accepted(report: ScanBatchReport, repository: ScanResultRepository) -> None:
    """Persist every accepted record; repository duplicate rules apply.

    A :class:`~rechip01.repository.DuplicateScanError` propagates — re-running
    a batch into a non-empty store is the operator's explicit move, and
    swallowing the collision here would hide it from the batch report trail.
    """
    for unit in report.accepted:
        repository.save(unit.result)


__all__ = [
    "AcceptedUnit",
    "DuplicateScanError",
    "Provenance",
    "RejectedUnit",
    "RejectionCode",
    "RejectionReason",
    "ScanBatchReport",
    "ScanResult",
    "ScanResultRepository",
    "SourceFormat",
    "load_rows",
    "persist_accepted",
    "scan_batch",
]
