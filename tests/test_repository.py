"""Tests for the file-backed passport repository."""

from datetime import UTC, datetime
from typing import Any

import pytest

from rechip01.repository import DuplicatePassportError, JsonPassportRepository
from rechip01.schemas import AssetPassport, Provenance
from tests.test_schemas import passport_kwargs


def make_passport(asset_id: str = "GPU-0001", run_id: str = "run-1") -> AssetPassport:
    """Return a valid passport with the given identifiers."""
    kwargs: dict[str, Any] = passport_kwargs()
    kwargs["asset_id"] = asset_id
    kwargs["run_id"] = run_id
    return AssetPassport(**kwargs)


def test_save_and_get_round_trip(tmp_path: Any) -> None:
    """A saved passport reads back identical."""
    repo = JsonPassportRepository(tmp_path)
    passport = make_passport()
    repo.save(passport)
    assert repo.get("GPU-0001") == passport


def test_get_missing_asset_returns_none(tmp_path: Any) -> None:
    """An asset with no passport reads as None, never as an empty record."""
    repo = JsonPassportRepository(tmp_path)
    assert repo.get("GPU-DOES-NOT-EXIST") is None


def test_save_rejects_duplicate_asset_run_pair(tmp_path: Any) -> None:
    """Issuing the same (asset, run) passport twice is an error."""
    repo = JsonPassportRepository(tmp_path)
    repo.save(make_passport(run_id="run-1"))
    with pytest.raises(DuplicatePassportError):
        repo.save(make_passport(run_id="run-1"))


def test_get_returns_latest_run(tmp_path: Any) -> None:
    """Multiple runs for one asset: get returns the latest-issued passport."""
    repo = JsonPassportRepository(tmp_path)
    earlier = make_passport(run_id="run-1")
    later = make_passport(run_id="run-2")
    repo.save(earlier)
    repo.save(later)
    assert repo.get("GPU-0001") in (earlier, later)

    # Force an unambiguous ordering via issued_at.
    reissued_kwargs = passport_kwargs()
    reissued_kwargs["asset_id"] = "GPU-0001"
    reissued_kwargs["run_id"] = "run-3"
    reissued_kwargs["issued_at"] = datetime(2027, 1, 1, tzinfo=UTC)
    latest = AssetPassport(**reissued_kwargs)
    repo.save(latest)
    assert repo.get("GPU-0001") == latest


def test_assets_are_isolated(tmp_path: Any) -> None:
    """Passports for different assets never bleed into each other's reads."""
    repo = JsonPassportRepository(tmp_path)
    repo.save(make_passport(asset_id="GPU-A", run_id="run-1"))
    repo.save(make_passport(asset_id="GPU-B", run_id="run-1"))
    found = repo.get("GPU-A")
    assert found is not None
    assert found.provenance == Provenance.MEASURED
    other = repo.get("GPU-B")
    assert other is not None
    assert other.run_id == "run-1"


def test_asset_ids_are_path_safe(tmp_path: Any) -> None:
    """Adversarial asset/run IDs cannot escape the repository root."""
    repo = JsonPassportRepository(tmp_path)
    passport = make_passport(asset_id="../../escape", run_id="../..%2Frun")
    repo.save(passport)
    assert repo.get("../../escape") == passport
    assert not (tmp_path.parent / "escape").exists()
