"""Shared fixtures for the dashboard test suite.

Same shape as the API suite: session-scoped engine (fitting is the expensive
step), function-scoped app with fresh file-backed repositories so tests never
see each other's records.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from tests.test_disposition.helpers import make_economy
from tests.test_engine.helpers import make_fleet

from rechip01.engine.engine import ReliabilityEngine
from rechip01.passport import ServiceDeps, create_app
from rechip01.repository import JsonPassportRepository, JsonScanResultRepository
from rechip01.simulator import SimulatedFleet

TRAIN_SEED = 303


@pytest.fixture(scope="session")
def fleet() -> SimulatedFleet:
    """A simulated fleet whose scans the dashboard ingests (never the training fleet)."""
    return make_fleet(seed=404, n_units=24)


@pytest.fixture(scope="session")
def engine(fleet: SimulatedFleet) -> ReliabilityEngine:
    """Engine trained on its own simulated fleet (demo mode)."""
    return ReliabilityEngine().fit(make_fleet(seed=TRAIN_SEED, n_units=300))


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


def ingest_rows(
    client: TestClient, rows: list[dict[str, Any]], provenance: str = "simulated"
) -> dict[str, Any]:
    """Ingest raw scan rows via ``POST /scan`` and return the batch report."""
    response = client.post("/scan", json={"provenance": provenance, "rows": rows})
    assert response.status_code == 200, response.text
    report: dict[str, Any] = response.json()
    return report


def assess_all(client: TestClient, asset_ids: list[str] | None = None) -> dict[str, Any]:
    """Run one assessment over stored scans with the shared test economy."""
    payload: dict[str, Any] = {
        "horizon_days": 365,
        "target_workloads": ["inference"],
        "economy": make_economy().model_dump(mode="json"),
    }
    if asset_ids is not None:
        payload["asset_ids"] = asset_ids
    response = client.post("/assess", json=payload)
    assert response.status_code == 200, response.text
    result: dict[str, Any] = response.json()
    return result
