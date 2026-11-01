"""File-backed persistence behind repository interfaces (spec §3 stack decision).

Persistence stays behind repository interfaces so a real store can replace
the JSON files without touching callers. Two stores live here:

- :class:`JsonPassportRepository` — issued asset passports, keyed by
  (asset_id, run_id); append-only, saving twice is an error.
- :class:`JsonScanResultRepository` — scanner output, keyed by asset_id;
  append-only, one assessment record per asset.

Records are immutable, so both stores are append-only, and every write is
atomic (temporary file + rename) so reads never observe a partial record.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Protocol

from rechip01.schemas import AssetPassport, ScanResult


class DuplicatePassportError(RuntimeError):
    """A passport already exists for the same (asset_id, run_id) pair."""


class DuplicateScanError(RuntimeError):
    """A scan record already exists for the same asset_id."""


class PassportRepository(Protocol):
    """Storage boundary for issued asset passports."""

    def save(self, passport: AssetPassport) -> None:
        """Persist a passport exactly once for its (asset_id, run_id)."""
        ...

    def get(self, asset_id: str) -> AssetPassport | None:
        """Return the latest-issued passport for ``asset_id``, or None."""
        ...


def _record_slug(asset_id: str, run_id: str) -> str:
    """Return a filesystem-safe, fixed-width filename for a passport.

    Partner asset IDs and run IDs are arbitrary strings; hashing keeps them
    out of path construction and prevents traversal or collision-by-case.
    """
    digest = hashlib.sha256(f"{asset_id}|{run_id}".encode()).hexdigest()
    return f"passport-{digest}.json"


class ScanResultRepository(Protocol):
    """Storage boundary for scanner output (normalized scan records)."""

    def save(self, result: ScanResult) -> None:
        """Persist a scan record exactly once for its asset_id."""
        ...

    def get(self, asset_id: str) -> ScanResult | None:
        """Return the stored scan record for ``asset_id``, or None."""
        ...

    def list_all(self) -> list[ScanResult]:
        """Return every stored scan record in stable order."""
        ...


def _scan_slug(asset_id: str) -> str:
    """Return a filesystem-safe, fixed-width filename for a scan record."""
    digest = hashlib.sha256(asset_id.encode()).hexdigest()
    return f"scan-{digest}.json"


def _atomic_write_text(path: Path, text: str) -> None:
    """Write ``text`` to ``path`` atomically: readers never see a partial file."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


class JsonPassportRepository:
    """Append-only JSON file store, one file per issued passport."""

    def __init__(self, root: Path) -> None:
        self._root = Path(root)
        self._root.mkdir(parents=True, exist_ok=True)

    def save(self, passport: AssetPassport) -> None:
        target = self._root / _record_slug(passport.asset_id, passport.run_id)
        if target.exists():
            raise DuplicatePassportError(
                f"passport already issued for asset {passport.asset_id!r}, run {passport.run_id!r}"
            )
        _atomic_write_text(target, passport.model_dump_json())

    def get(self, asset_id: str) -> AssetPassport | None:
        latest: AssetPassport | None = None
        for path in sorted(self._root.glob("passport-*.json")):
            candidate = AssetPassport.model_validate_json(path.read_text(encoding="utf-8"))
            if candidate.asset_id != asset_id:
                continue
            if latest is None or candidate.issued_at > latest.issued_at:
                latest = candidate
        return latest


class JsonScanResultRepository:
    """Append-only JSON file store for scan records, one file per asset.

    Filenames are content-independent hashes of the asset ID (path-safe for
    arbitrary partner identifiers); writes are atomic, so a read either sees
    a complete record or none at all.
    """

    def __init__(self, root: Path) -> None:
        self._root = Path(root)
        self._root.mkdir(parents=True, exist_ok=True)

    def save(self, result: ScanResult) -> None:
        target = self._root / _scan_slug(result.asset_id)
        if target.exists():
            raise DuplicateScanError(f"scan record already stored for asset {result.asset_id!r}")
        _atomic_write_text(target, result.model_dump_json())

    def get(self, asset_id: str) -> ScanResult | None:
        target = self._root / _scan_slug(asset_id)
        if not target.exists():
            return None
        return ScanResult.model_validate_json(target.read_text(encoding="utf-8"))

    def list_all(self) -> list[ScanResult]:
        return [
            ScanResult.model_validate_json(path.read_text(encoding="utf-8"))
            for path in sorted(self._root.glob("scan-*.json"))
        ]


__all__ = [
    "DuplicatePassportError",
    "DuplicateScanError",
    "JsonPassportRepository",
    "JsonScanResultRepository",
    "PassportRepository",
    "ScanResultRepository",
]
