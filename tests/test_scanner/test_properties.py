"""Hypothesis property tests for the scanner.

The two properties the spec's testing plan calls out:

1. Invalid input is never silently accepted — every corrupted row lands in
   quarantine with machine-readable reasons, whatever the corruption.
2. Accepted records round-trip through the file-backed repository — what the
   scanner produced is exactly what the persistence layer hands back.
"""

from __future__ import annotations

import csv
import tempfile
import uuid
from pathlib import Path

from hypothesis import given, settings
from hypothesis import strategies as st

from rechip01.repository import JsonScanResultRepository
from rechip01.scanner import (
    Provenance,
    RejectionReason,
    ScanBatchReport,
    SourceFormat,
    load_rows,
    persist_accepted,
    scan_batch,
)
from tests.test_scanner.fixtures import corrupted_rows, stringify_row, valid_rows

_PROVENANCES = st.sampled_from([Provenance.SIMULATED, Provenance.MEASURED])


def _asset_id_of(row: dict[str, object]) -> str:
    return str(row["asset_id"])


@settings(max_examples=50)
@given(
    rows=st.lists(st.one_of(valid_rows(), corrupted_rows()), max_size=6),
    provenance=_PROVENANCES,
)
def test_property_rows_land_in_exactly_one_bucket(
    rows: list[object], provenance: Provenance
) -> None:
    report = scan_batch(rows, provenance)

    accepted_indexes = [unit.row_index for unit in report.accepted]
    rejected_indexes = [unit.row_index for unit in report.rejected]
    assert sorted(accepted_indexes + rejected_indexes) == list(range(len(rows)))
    assert not set(accepted_indexes) & set(rejected_indexes)
    assert report.total_rows == len(rows)

    for accepted_unit in report.accepted:
        assert accepted_unit.result.provenance is provenance
    for rejected_unit in report.rejected:
        assert rejected_unit.reasons
        for reason in rejected_unit.reasons:
            assert isinstance(reason, RejectionReason)
            assert reason.detail


@settings(max_examples=50)
@given(row=corrupted_rows(), provenance=_PROVENANCES)
def test_property_corrupted_rows_are_never_silently_accepted(
    row: dict[str, object], provenance: Provenance
) -> None:
    report = scan_batch([row], provenance)

    assert report.accepted_count == 0
    assert report.rejected_count == 1
    assert report.rejected[0].reasons


@settings(max_examples=50)
@given(rows=st.lists(valid_rows(), max_size=5, unique_by=_asset_id_of))
def test_property_accepted_records_round_trip_through_repository(
    rows: list[dict[str, object]],
) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / uuid.uuid4().hex
        report = scan_batch(rows, Provenance.MEASURED)

        persist_accepted(report, JsonScanResultRepository(root))
        repository = JsonScanResultRepository(root)

        stored = {result.asset_id: result for result in repository.list_all()}
        for unit in report.accepted:
            assert repository.get(unit.result.asset_id) == unit.result
            assert stored[unit.result.asset_id] == unit.result
        assert set(stored) == {unit.result.asset_id for unit in report.accepted}


@settings(max_examples=50)
@given(row=valid_rows(), provenance=_PROVENANCES)
def test_property_csv_encoded_row_matches_typed_row(
    row: dict[str, object], provenance: Provenance
) -> None:
    """The export path and the feed path agree for the same logical row."""
    with tempfile.TemporaryDirectory() as tmp:
        csv_path = Path(tmp) / "batch.csv"
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(row))
            writer.writeheader()
            writer.writerow(stringify_row(row))

        loaded_rows, source_format = load_rows(csv_path)
        feed_report = scan_batch([row], provenance, source_format=SourceFormat.SIMULATOR_FEED)
        export_report = scan_batch(loaded_rows, provenance, source_format=source_format)

        assert export_report.accepted_count == 1
        assert export_report.accepted[0].result == feed_report.accepted[0].result


@settings(max_examples=30)
@given(rows=st.lists(valid_rows(), max_size=4, unique_by=_asset_id_of))
def test_property_report_json_round_trip(rows: list[dict[str, object]]) -> None:
    report = scan_batch(rows, Provenance.SIMULATED)

    restored = ScanBatchReport.model_validate_json(report.model_dump_json())

    assert restored == report
