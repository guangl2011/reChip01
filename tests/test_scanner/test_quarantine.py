"""Quarantine-path tests: rejections are machine-readable and never leak through."""

from __future__ import annotations

from pathlib import Path

import pytest

from rechip01.repository import DuplicateScanError, JsonScanResultRepository
from rechip01.scanner import Provenance, RejectionCode, persist_accepted, scan_batch
from tests.test_scanner.fixtures import valid_row


def test_rejections_carry_machine_readable_reasons() -> None:
    rows = [
        valid_row(asset_id="   "),
        {"model": "A100-SXM4-80GB", "chain_of_custody": ["intake"]},
        valid_row(ecc_events="not-a-number"),
        valid_row(thermal_history=[2.0]),
    ]

    report = scan_batch(rows, Provenance.MEASURED)

    assert report.accepted_count == 0
    for unit in report.rejected:
        assert unit.reasons, f"row {unit.row_index} rejected with no reasons"
        for reason in unit.reasons:
            assert isinstance(reason.code, RejectionCode)
            assert reason.detail


def test_row_index_is_preserved_through_quarantine() -> None:
    rows = [valid_row(asset_id="GPU-A"), "garbage", valid_row()]

    report = scan_batch(rows, Provenance.SIMULATED)

    assert [unit.row_index for unit in report.accepted] == [0, 2]
    assert [unit.row_index for unit in report.rejected] == [1]
    assert report.total_rows == 3


def test_a_row_lands_in_exactly_one_bucket() -> None:
    rows = [valid_row(), valid_row(asset_id="GPU-B"), valid_row(model="")]

    report = scan_batch(rows, Provenance.SIMULATED)

    accepted_rows = {unit.row_index for unit in report.accepted}
    rejected_rows = {unit.row_index for unit in report.rejected}
    assert not accepted_rows & rejected_rows
    assert accepted_rows | rejected_rows == {0, 1, 2}


def test_duplicate_asset_in_batch_is_rejected_not_merged() -> None:
    first = valid_row(asset_id="GPU-DUP")
    second = valid_row(asset_id="GPU-DUP", ecc_events=99)

    report = scan_batch([first, second], Provenance.MEASURED)

    assert report.accepted_count == 1
    assert report.accepted[0].result.ecc_events == 12  # first wins, second quarantined
    reason = report.rejected[0].reasons[0]
    assert reason.code is RejectionCode.DUPLICATE_ASSET_ID


def test_quarantined_rows_never_reach_the_repository(tmp_path: Path) -> None:
    rows = [valid_row(asset_id="GPU-OK"), valid_row(asset_id="GPU-BAD", ecc_events=-5)]

    report = scan_batch(rows, Provenance.SIMULATED)
    persist_accepted(report, JsonScanResultRepository(tmp_path / "store"))

    stored = {result.asset_id for result in JsonScanResultRepository(tmp_path / "store").list_all()}
    assert stored == {"GPU-OK"}


def test_repository_rejects_duplicate_asset_save(tmp_path: Path) -> None:
    repo = JsonScanResultRepository(tmp_path / "store")
    result = scan_batch([valid_row(asset_id="GPU-DUP")], Provenance.SIMULATED).accepted[0].result

    repo.save(result)
    with pytest.raises(DuplicateScanError):
        repo.save(result)
