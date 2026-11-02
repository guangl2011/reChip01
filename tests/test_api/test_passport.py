"""GET /passport/{asset_id} tests: retrieval, immutability, replay semantics."""

from __future__ import annotations

from typing import Any

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from rechip01.schemas import AssetPassport, Provenance
from rechip01.simulator import SimulatedFleet
from tests.test_api.conftest import assess_request, ingest_fleet

RUN_ID = "run-fixed-001"


def _issue_passport(client: TestClient, fleet: SimulatedFleet) -> dict[str, Any]:
    """Ingest one scan and assess it under a fixed run id."""
    ingest_fleet(client, fleet, n_units=1)
    response = client.post("/assess", json=assess_request(run_id=RUN_ID))
    assert response.status_code == 200, response.text
    result: dict[str, Any] = response.json()
    return result


def test_passport_retrieval(client: TestClient, fleet: SimulatedFleet) -> None:
    """An issued passport carries provenance, uncertainty, versions, and an
    evidence trace with a data-lineage hash."""
    result = _issue_passport(client, fleet)
    asset_id = result["items"][0]["asset_id"]

    payload = client.get(f"/passport/{asset_id}").json()
    assert payload["asset_id"] == asset_id
    assert payload["run_id"] == result["run_id"]
    assert payload["provenance"] == "simulated"
    assert payload["model_version"]

    # Uncertainty bands are present and bracket the headline probability.
    low, high = payload["reliability"]["ci"]
    assert 0.0 <= low <= payload["reliability"]["p_fail"] <= high <= 1.0
    assert payload["reliability"]["horizon_days"] == 365
    assert payload["reliability"]["model_version"] == payload["model_version"]

    assert payload["disposition"]["recommendation"]
    assert len(payload["evidence_trace"]) >= 4
    assert any(line.startswith("lineage: sha256=") for line in payload["evidence_trace"])


def test_passport_unknown_asset_is_404(client: TestClient) -> None:
    """Retrieving a passport that was never issued is a machine-readable 404."""
    response = client.get("/passport/never-scanned")
    assert response.status_code == 404
    assert "never-scanned" in response.json()["detail"]


def test_passport_not_issued_before_assessment(client: TestClient, fleet: SimulatedFleet) -> None:
    """A stored but unassessed scan has no passport yet."""
    ingest_fleet(client, fleet, n_units=1)
    asset_id = client.get("/batch").json()["units"][0]["asset_id"]
    assert client.get(f"/passport/{asset_id}").status_code == 404


def test_passport_frozen_after_issuance(client: TestClient, fleet: SimulatedFleet) -> None:
    """Post-issuance mutation is rejected by the schema, not just by convention."""
    result = _issue_passport(client, fleet)
    asset_id = result["items"][0]["asset_id"]
    passport = AssetPassport.model_validate(client.get(f"/passport/{asset_id}").json())
    with pytest.raises(ValidationError):
        passport.provenance = Provenance.MEASURED
    with pytest.raises(ValidationError):
        passport.reliability.p_fail = 0.0


def test_passport_provenance_immutable_after_issuance(
    client: TestClient, fleet: SimulatedFleet
) -> None:
    """Re-reading a stored passport returns the issued provenance unchanged —
    nothing downstream rewrites the simulated/measured label."""
    result = _issue_passport(client, fleet)
    asset_id = result["items"][0]["asset_id"]
    first = client.get(f"/passport/{asset_id}").json()
    second = client.get(f"/passport/{asset_id}").json()
    assert first["provenance"] == second["provenance"] == "simulated"
    assert first["evidence_trace"] == second["evidence_trace"]


def test_same_run_id_replay_is_409(client: TestClient, fleet: SimulatedFleet) -> None:
    """Retrying the same assessment run conflicts — passports are issued
    exactly once per (asset_id, run_id); the first record stands."""
    _issue_passport(client, fleet)
    response = client.post("/assess", json=assess_request(run_id=RUN_ID))
    assert response.status_code == 409
    assert "already issued" in response.json()["detail"]


def test_second_run_updates_latest_passport(client: TestClient, fleet: SimulatedFleet) -> None:
    """A second run issues a new passport; retrieval returns the latest."""
    _issue_passport(client, fleet)
    ingest_fleet(client, fleet, n_units=0)  # no-op; keep store identical
    second_run = "run-fixed-002"
    response = client.post("/assess", json=assess_request(run_id=second_run))
    assert response.status_code == 200, response.text

    asset_id = response.json()["items"][0]["asset_id"]
    payload = client.get(f"/passport/{asset_id}").json()
    assert payload["run_id"] == second_run
