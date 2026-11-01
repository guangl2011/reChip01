"""Inventory export loaders: CSV and JSONL partner files and simulator dumps.

Loaders only read and dispatch — all validation happens in
:func:`rechip01.scanner.ingest.scan_batch`, so a simulator feed and a partner
export pass through the same interface (spec §4 parity invariant).
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

from rechip01.scanner.models import SourceFormat

_CSV_SUFFIXES = frozenset({".csv"})
_JSONL_SUFFIXES = frozenset({".jsonl", ".ndjson"})


def load_rows(path: Path) -> tuple[list[object], SourceFormat]:
    """Read an inventory file into raw rows plus its detected source format.

    ``.csv`` files are read as header-keyed records; ``.jsonl``/``.ndjson``
    files as one JSON object per line (blank lines skipped). Any other
    suffix raises ``ValueError`` — the scanner never guesses a format.
    """
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix in _CSV_SUFFIXES:
        with path.open(newline="", encoding="utf-8") as handle:
            rows: list[object] = list(csv.DictReader(handle))
        return rows, SourceFormat.CSV_EXPORT
    if suffix in _JSONL_SUFFIXES:
        rows = []
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{path}:{line_number}: invalid JSON ({exc.msg})") from exc
        return rows, SourceFormat.JSONL_EXPORT
    raise ValueError(
        f"unsupported inventory format {suffix!r} for {path} (expected .csv, .jsonl, or .ndjson)"
    )


__all__ = ["load_rows"]
