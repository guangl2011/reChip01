"""Normalization unit tests: header aliases, string coercion, partial mode."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from rechip01.scanner import Provenance, RejectionCode, SourceFormat, load_rows, scan_batch
from tests.test_scanner.fixtures import stringify_row, valid_row


def test_partner_style_headers_are_normalized() -> None:
    row = {
        "Asset ID": "GPU-0001",
        "GPU Model": "A100-SXM4-80GB",
        "Serial Number": "GPU-0001",  # collides by aliasing — normalized name wins order
        "ECC": 12,
        "XID": 1,
        "Retired Pages": 3,
        "Thermal History": "[0.5, 0.6]",
        "Max Power Pct": "0.9",
        "Test Evidence": '{"mem_bw": 0.4}',
        "Custody": "intake; erasure",
    }

    report = scan_batch([row], Provenance.MEASURED)

    # Both "Asset ID" and "Serial Number" normalize to asset_id; last one wins.
    assert report.accepted_count == 1
    result = report.accepted[0].result
    assert result.asset_id == "GPU-0001"
    assert result.ecc_events == 12
    assert result.thermal_history == [0.5, 0.6]
    assert result.max_sustained_power_pct == 0.9
    assert result.bench_mark_snapshot == {"mem_bw": 0.4}
    assert result.chain_of_custody == ["intake", "erasure"]


def test_string_counters_and_floats_are_coerced() -> None:
    row = valid_row(
        ecc_events="12",
        xid_errors="0",
        retired_pages="3",
        age_months="30.5",
        power_on_hours="18250",
        max_sustained_power_pct="0.87",
    )

    report = scan_batch([row], Provenance.MEASURED)

    assert report.accepted_count == 1
    result = report.accepted[0].result
    assert result.ecc_events == 12
    assert result.age_months == 30.5
    assert result.power_on_hours == 18250.0
    assert result.max_sustained_power_pct == 0.87


def test_unparseable_integer_is_a_parse_error() -> None:
    report = scan_batch([valid_row(ecc_events="twelve")], Provenance.MEASURED)

    reason = report.rejected[0].reasons[0]
    assert reason.code is RejectionCode.PARSE_ERROR
    assert reason.field == "ecc_events"


def test_thermal_history_fallback_separator() -> None:
    report = scan_batch([valid_row(thermal_history="0.5;0.6|0.7")], Provenance.MEASURED)

    assert report.accepted_count == 1
    assert report.accepted[0].result.thermal_history == [0.5, 0.6, 0.7]


def test_invalid_benchmark_json_is_a_parse_error() -> None:
    report = scan_batch([valid_row(bench_mark_snapshot="{not json")], Provenance.MEASURED)

    reason = report.rejected[0].reasons[0]
    assert reason.code is RejectionCode.PARSE_ERROR
    assert reason.field == "bench_mark_snapshot"


def test_partial_row_gets_explicit_defaults_and_flags() -> None:
    row = {"asset_id": "GPU-0002", "model": "H100-SXM5-80GB", "chain_of_custody": ["intake"]}

    report = scan_batch([row], Provenance.MEASURED)

    assert report.accepted_count == 1
    unit = report.accepted[0]
    assert unit.missing_fields == (
        "age_months",
        "power_on_hours",
        "ecc_events",
        "xid_errors",
        "retired_pages",
        "thermal_history",
        "max_sustained_power_pct",
        "bench_mark_snapshot",
    )
    # Explicit defaults, visibly flagged — never silent imputation.
    result = unit.result
    assert result.age_months is None
    assert result.power_on_hours is None
    assert result.ecc_events == 0
    assert result.thermal_history == []
    assert result.bench_mark_snapshot == {}
    assert result.max_sustained_power_pct == 0.0


def test_blank_optional_cell_is_treated_as_missing_not_fatal() -> None:
    row = valid_row(ecc_events="", power_on_hours=None)

    report = scan_batch([row], Provenance.MEASURED)

    assert report.accepted_count == 1
    unit = report.accepted[0]
    assert "ecc_events" in unit.missing_fields
    assert "power_on_hours" in unit.missing_fields
    assert unit.result.ecc_events == 0
    assert unit.result.power_on_hours is None


def test_simulator_feed_rows_and_exports_share_the_interface() -> None:
    """The same logical row via dict (feed) and CSV-style strings (export)
    must produce identical ScanResults apart from the declared provenance."""
    typed = valid_row()
    as_strings = stringify_row(typed)

    feed_report = scan_batch(
        [typed], Provenance.SIMULATED, source_format=SourceFormat.SIMULATOR_FEED
    )
    export_report = scan_batch(
        [as_strings], Provenance.MEASURED, source_format=SourceFormat.CSV_EXPORT
    )

    assert feed_report.accepted_count == 1
    assert export_report.accepted_count == 1
    feed_result = feed_report.accepted[0].result
    export_result = export_report.accepted[0].result
    assert feed_result.provenance is Provenance.SIMULATED
    assert export_result.provenance is Provenance.MEASURED
    # Everything except provenance — the only legitimate difference — is identical.
    assert feed_result.model_dump(exclude={"provenance"}) == export_result.model_dump(
        exclude={"provenance"}
    )


def test_csv_and_jsonl_files_round_trip_through_load_rows(tmp_path: Path) -> None:
    csv_path = tmp_path / "batch.csv"
    jsonl_path = tmp_path / "batch.jsonl"
    row = valid_row(asset_id="GPU-PARITY")
    fields = list(row)

    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerow(stringify_row(row))
    jsonl_path.write_text(json.dumps(row) + "\n", encoding="utf-8")

    csv_rows, csv_format = load_rows(csv_path)
    jsonl_rows, jsonl_format = load_rows(jsonl_path)

    csv_report = scan_batch(csv_rows, Provenance.MEASURED, source_format=csv_format)
    jsonl_report = scan_batch(jsonl_rows, Provenance.MEASURED, source_format=jsonl_format)

    assert csv_report.accepted_count == 1
    assert jsonl_report.accepted_count == 1
    assert csv_report.accepted[0].result == jsonl_report.accepted[0].result
    assert csv_format is SourceFormat.CSV_EXPORT
    assert jsonl_format is SourceFormat.JSONL_EXPORT
