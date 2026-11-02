"""Shared fixtures for the passport/API test suite.

The engine fixture is session-scoped (fitting is the expensive step; the
fitted engine is stateless at assess time); each test gets fresh file-backed
repositories via a function-scoped app so tests never see each other's
records.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from rechip01.engine.engine import ReliabilityEngine
from rechip01.passport import ServiceDeps, create_app
from rechip01.repository import JsonPassportRepository, JsonScanResultRepository
from rechip01.simulator import SimulatedFleet
from tests.test_disposition.helpers import make_economy
from tests.test_engine.helpers import make_fleet

TRAIN_SEED = 303
TRAIN_N = 300
ASSESS_ECONOMY = make_economy()


@pytest.fixture(scope="session")
def fleet() -> SimulatedFleet:
    """A simulated fleet whose scans the API ingests (never the training fleet)."""
    return make_fleet(seed=404, n_units=24)


@pytest.fixture(scope="session")
def engine(fleet: SimulatedFleet) -> ReliabilityEngine:
    """Engine trained on its own small simulated fleet (demo mode)."""
    return ReliabilityEngine().fit(make_fleet(seed=TRAIN_SEED, n_units=TRAIN_N))


@pytest.fixture
def app(engine: ReliabilityEngine, tmp_path: Path) -> FastAPI:
    """App over a session engine plus per-test repositories."""
    deps = ServiceDeps(
        engine=engine,
        scans=JsonScanResultRepository(tmp_path / "scans"),
        passports=JsonPassportRepository(tmp_path / "passports"),
        training_fingerprint=engine.training_fingerprint(),
    )
    return create_app(deps)


@pytest.fixture
def client(app: FastAPI) -> Iterator[TestClient]:
    with TestClient(app) as test_client:
        yield test_client


def ingest_fleet(
    client: TestClient,
    fleet: SimulatedFleet,
    *,
    n_units: int = 4,
    provenance: str = "simulated",
    start: int = 0,
) -> dict[str, Any]:
    """Ingest ``n_units`` fleet scans and return the batch report payload."""
    rows = [fleet.scans[start + offset].model_dump() for offset in range(n_units)]
    if provenance != "simulated":
        # Fleet scans are simulated records; re-labeling them here models a
        # genuinely measured export whose row provenance matches the
        # declaration — a mismatch would be a provenance_conflict by design.
        for row in rows:
            row["provenance"] = provenance
    response = client.post("/scan", json={"provenance": provenance, "rows": rows})
    assert response.status_code == 200, response.text
    report: dict[str, Any] = response.json()
    return report


def assess_request(
    *,
    horizon_days: int = 365,
    target_workloads: list[str] | None = None,
    asset_ids: list[str] | None = None,
    run_id: str | None = None,
) -> dict[str, Any]:
    """A valid /assess request body against the shared test economy."""
    payload: dict[str, Any] = {
        "horizon_days": horizon_days,
        "target_workloads": target_workloads if target_workloads is not None else ["inference"],
        "economy": ASSESS_ECONOMY.model_dump(mode="json"),
    }
    if asset_ids is not None:
        payload["asset_ids"] = asset_ids
    if run_id is not None:
        payload["run_id"] = run_id
    return payload
