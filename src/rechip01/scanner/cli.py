"""Command-line entry point for the scanner (spec §4: "CLI or module entry point").

Ingests one inventory file (CSV or JSONL) or simulator feed dump, stamps the
declared provenance, prints a batch summary, and optionally persists accepted
records to a file-backed scan repository and the full batch report to JSON.

Exit codes: 0 = at least one record accepted, 1 = nothing accepted,
2 = input error (unreadable path, unsupported format).
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from rechip01.repository import DuplicateScanError, JsonScanResultRepository, ScanResultRepository
from rechip01.scanner.file_io import load_rows
from rechip01.scanner.ingest import scan_batch
from rechip01.scanner.models import Provenance, RejectedUnit, ScanBatchReport


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rechip01-scanner",
        description="Validate, normalize, and provenance-stamp an inventory batch.",
    )
    subcommands = parser.add_subparsers(dest="command", required=True)

    ingest = subcommands.add_parser("ingest", help="Ingest one inventory file or feed dump.")
    ingest.add_argument("path", type=Path, help="Inventory file: .csv, .jsonl, or .ndjson")
    ingest.add_argument(
        "--provenance",
        required=True,
        choices=[member.value for member in Provenance],
        help="Origin of the batch, stamped on every accepted record",
    )
    ingest.add_argument(
        "--store",
        type=Path,
        default=None,
        help="Optional directory for the file-backed scan repository",
    )
    ingest.add_argument(
        "--report",
        type=Path,
        default=None,
        help="Optional path for the full batch report (JSON)",
    )
    return parser


def _persist(report: ScanBatchReport, repository: ScanResultRepository) -> int:
    """Save accepted records; duplicates across batches are counted, never silent."""
    saved = 0
    for unit in report.accepted:
        try:
            repository.save(unit.result)
        except DuplicateScanError:
            print(f"skipped (already stored): asset {unit.result.asset_id}", file=sys.stderr)
        else:
            saved += 1
    return saved


def _print_rejections(rejected: tuple[RejectedUnit, ...]) -> None:
    for unit in rejected:
        for reason in unit.reasons:
            field = reason.field if reason.field is not None else "-"
            print(
                f"row {unit.row_index}: code={reason.code.value} "
                f"field={field} detail={reason.detail}",
                file=sys.stderr,
            )


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point; see module docstring for exit codes."""
    args = _build_parser().parse_args(argv)

    try:
        rows, source_format = load_rows(args.path)
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    provenance = Provenance(args.provenance)
    report = scan_batch(rows, provenance, source_format=source_format)

    saved = None
    if args.store is not None:
        saved = _persist(report, JsonScanResultRepository(args.store))

    _print_rejections(report.rejected)

    summary = (
        f"total={report.total_rows} accepted={report.accepted_count} "
        f"rejected={report.rejected_count}"
    )
    if saved is not None:
        summary += f" persisted={saved}"
    print(summary)

    if args.report is not None:
        args.report.write_text(report.model_dump_json(indent=2), encoding="utf-8")

    return 0 if report.accepted_count > 0 else 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
