"""End-to-end CLI tests: ingest files, persist, report, exit codes."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from rechip01.repository import JsonScanResultRepository
from rechip01.scanner import ScanBatchReport, load_rows
from rechip01.scanner.cli import main
from tests.test_scanner.fixtures import stringify_row, valid_row


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> Path:
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    return path


def test_cli_ingests_jsonl_persists_and_reports(tmp_path: Path) -> None:
    source = _write_jsonl(tmp_path / "batch.jsonl", [valid_row(asset_id="GPU-CLI-1")])
    store = tmp_path / "store"
    report_path = tmp_path / "report.json"

    code = main(
        [
            "ingest",
            str(source),
            "--provenance",
            "measured",
            "--store",
            str(store),
            "--report",
            str(report_path),
        ]
    )

    assert code == 0
    report = ScanBatchReport.model_validate_json(report_path.read_text(encoding="utf-8"))
    assert report.accepted_count == 1
    assert report.source_format.value == "jsonl_export"
    stored = JsonScanResultRepository(store).get("GPU-CLI-1")
    assert stored is not None
    assert stored.provenance.value == "measured"


def test_cli_rejects_bad_rows_but_still_succeeds(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    source = _write_jsonl(
        tmp_path / "batch.jsonl",
        [valid_row(asset_id="GPU-CLI-2"), {"model": "no-identity"}],
    )

    code = main(["ingest", str(source), "--provenance", "simulated"])

    assert code == 0  # one accepted row
    captured = capsys.readouterr()
    assert "accepted=1" in captured.out
    assert "code=missing_required_field" in captured.err


def test_cli_returns_nonzero_when_nothing_accepted(tmp_path: Path) -> None:
    source = _write_jsonl(tmp_path / "empty.jsonl", [{"model": "no-identity"}])

    code = main(["ingest", str(source), "--provenance", "simulated"])

    assert code == 1


def test_cli_rejects_unsupported_format(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    source = tmp_path / "batch.xlsx"
    source.write_text("not really", encoding="utf-8")

    code = main(["ingest", str(source), "--provenance", "simulated"])

    assert code == 2
    assert "unsupported inventory format" in capsys.readouterr().err


def test_cli_ingests_csv_export(tmp_path: Path) -> None:
    source = tmp_path / "batch.csv"
    row = stringify_row(valid_row(asset_id="GPU-CLI-3"))
    with source.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)

    code = main(["ingest", str(source), "--provenance", "measured"])

    assert code == 0
    rows, source_format = load_rows(source)
    assert source_format.value == "csv_export"
    assert rows
