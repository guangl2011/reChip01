"""Provenance stamping tests: the source declaration is the only authority."""

from __future__ import annotations

from rechip01.scanner import (
    Provenance,
    RejectionCode,
    ScanBatchReport,
    scan_batch,
)
from tests.test_scanner.fixtures import valid_row


def test_simulated_batch_is_stamped_simulated() -> None:
    report = scan_batch([valid_row()], Provenance.SIMULATED)

    assert report.provenance is Provenance.SIMULATED
    assert report.accepted[0].result.provenance is Provenance.SIMULATED


def test_measured_batch_is_stamped_measured() -> None:
    report = scan_batch([valid_row()], Provenance.MEASURED)

    assert report.provenance is Provenance.MEASURED
    assert report.accepted[0].result.provenance is Provenance.MEASURED


def test_row_claiming_conflicting_provenance_is_quarantined() -> None:
    """A measured export can never sneak into a simulated batch (or the
    reverse): the conflict is quarantined, never silently re-stamped."""
    report = scan_batch([valid_row(provenance="measured")], Provenance.SIMULATED)

    assert report.accepted_count == 0
    reason = report.rejected[0].reasons[0]
    assert reason.code is RejectionCode.PROVENANCE_CONFLICT
    assert reason.field == "provenance"


def test_row_claiming_unrecognized_provenance_is_quarantined() -> None:
    report = scan_batch([valid_row(provenance="certified")], Provenance.MEASURED)

    assert report.accepted_count == 0
    assert report.rejected[0].reasons[0].field == "provenance"


def test_row_agreeing_with_declared_provenance_is_accepted() -> None:
    report = scan_batch([valid_row(provenance="measured")], Provenance.MEASURED)

    assert report.accepted_count == 1
    assert report.accepted[0].result.provenance is Provenance.MEASURED


def test_report_is_round_trippable_through_json() -> None:
    report = scan_batch([valid_row()], Provenance.SIMULATED)

    restored = ScanBatchReport.model_validate_json(report.model_dump_json())

    assert restored == report
    assert restored.accepted[0].result.provenance is Provenance.SIMULATED
