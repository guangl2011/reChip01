"""File-backed persistence for asset passports (spec §3 stack decision).

Persistence stays behind a repository interface so a real store can replace
the JSON files without touching callers. Passports are immutable records, so
the repository is append-only: saving a passport for an (asset, run) pair
twice is an error, and reads never return a partially written file's data.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Protocol

from rechip01.schemas import AssetPassport


class DuplicatePassportError(RuntimeError):
    """A passport already exists for the same (asset_id, run_id) pair."""


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
        target.write_text(passport.model_dump_json(), encoding="utf-8")

    def get(self, asset_id: str) -> AssetPassport | None:
        latest: AssetPassport | None = None
        for path in sorted(self._root.glob("passport-*.json")):
            candidate = AssetPassport.model_validate_json(path.read_text(encoding="utf-8"))
            if candidate.asset_id != asset_id:
                continue
            if latest is None or candidate.issued_at > latest.issued_at:
                latest = candidate
        return latest


__all__ = ["DuplicatePassportError", "JsonPassportRepository", "PassportRepository"]
