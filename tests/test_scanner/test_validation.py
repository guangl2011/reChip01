"""Validation unit tests: what the scanner accepts, and what it quarantines."""

from __future__ import annotations

from rechip01.scanner import Provenance, RejectionCode, scan_batch
from rechip01.schemas import ScanResult
from tests.test_scanner.fixtures import valid_row


def test_valid_row_is_accepted_with_all_fields() -> None:
    report = scan_batch([valid_row()], Provenance.SIMULATED)

    assert report.accepted_count == 1
    assert report.rejected_count == 0
    assert report.accepted[0].missing_fields == ()
    assert isinstance(report.accepted[0].result, ScanResult)


def test_missing_asset_id_is_rejected() -> None:
    row = valid_row()
    del row["asset_id"]

    report = scan_batch([row], Provenance.MEASURED)

    assert report.accepted_count == 0
    reasons = report.rejected[0].reasons
    assert any(
        reason.code is RejectionCode.MISSING_REQUIRED_FIELD and reason.field == "asset_id"
        for reason in reasons
    )


def test_blank_asset_id_is_rejected() -> None:
    report = scan_batch([valid_row(asset_id="   ")], Provenance.MEASURED)

    assert report.rejected[0].reasons[0].code is RejectionCode.MISSING_REQUIRED_FIELD


def test_missing_model_is_rejected() -> None:
    row = valid_row()
    del row["model"]

    report = scan_batch([row], Provenance.MEASURED)

    assert any(
        reason.code is RejectionCode.MISSING_REQUIRED_FIELD and reason.field == "model"
        for reason in report.rejected[0].reasons
    )


def test_missing_chain_of_custody_is_rejected() -> None:
    row = valid_row()
    del row["chain_of_custody"]

    report = scan_batch([row], Provenance.MEASURED)

    assert any(
        reason.code is RejectionCode.MISSING_REQUIRED_FIELD and reason.field == "chain_of_custody"
        for reason in report.rejected[0].reasons
    )


def test_empty_chain_of_custody_is_rejected() -> None:
    report = scan_batch([valid_row(chain_of_custody=[])], Provenance.MEASURED)

    assert any(
        reason.code is RejectionCode.MISSING_REQUIRED_FIELD and reason.field == "chain_of_custody"
        for reason in report.rejected[0].reasons
    )


def test_negative_ecc_events_is_a_schema_violation() -> None:
    report = scan_batch([valid_row(ecc_events=-1)], Provenance.SIMULATED)

    reason = report.rejected[0].reasons[0]
    assert reason.code is RejectionCode.SCHEMA_VIOLATION
    assert reason.field == "ecc_events"


def test_thermal_above_one_is_a_schema_violation() -> None:
    report = scan_batch([valid_row(thermal_history=[0.5, 1.5])], Provenance.SIMULATED)

    reason = report.rejected[0].reasons[0]
    assert reason.code is RejectionCode.SCHEMA_VIOLATION
    assert reason.field is not None
    assert reason.field.startswith("thermal_history")


def test_nan_power_reading_is_rejected() -> None:
    report = scan_batch([valid_row(max_sustained_power_pct=float("nan"))], Provenance.SIMULATED)

    assert report.accepted_count == 0
    assert report.rejected[0].reasons[0].code is RejectionCode.SCHEMA_VIOLATION


def test_non_mapping_row_is_rejected() -> None:
    report = scan_batch(["not a row", 42], Provenance.SIMULATED)

    assert report.rejected_count == 2
    assert report.rejected[0].reasons[0].code is RejectionCode.PARSE_ERROR
    assert report.rejected[0].asset_id is None


def test_unknown_keys_are_ignored() -> None:
    row = valid_row(outcome_feedback="returned-after-90-days")

    report = scan_batch([row], Provenance.MEASURED)

    assert report.accepted_count == 1
    assert report.rejected_count == 0


def test_empty_batch_is_empty() -> None:
    report = scan_batch([], Provenance.SIMULATED)

    assert report.total_rows == 0
    assert report.accepted_count == 0
    assert report.rejected_count == 0
