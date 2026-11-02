"""POST /assess integration tests: engine + disposition + passport issuance."""

from __future__ import annotations

from typing import Any

from fastapi.testclient import TestClient

from rechip01.simulator import SimulatedFleet
from tests.test_api.conftest import assess_request, ingest_fleet


def _assess(client: TestClient, **kwargs: Any) -> dict[str, Any]:
    response = client.post("/assess", json=assess_request(**kwargs))
    assert response.status_code == 200, response.text
    result: dict[str, Any] = response.json()
    return result


def test_assess_happy_path(client: TestClient, fleet: SimulatedFleet) -> None:
    """A batch of stored scans assesses, decides, and issues passports."""
    ingest_fleet(client, fleet, n_units=4)
    result = _assess(client)

    assert result["run_id"].startswith("run-")
    assert result["horizon_days"] == 365
    assert result["target_workloads"] == ["inference"]
    assert result["provenance_counts"] == {"simulated": 4}
    assert len(result["items"]) == 4

    for item in result["items"]:
        assert item["provenance"] == "simulated"
        low, high = item["ci"]
        assert 0.0 <= low <= item["p_fail"] <= high <= 1.0
        assert isinstance(item["ood_flag"], bool)
        assert item["recommendation"]
        assert item["regret_bound"] >= 0.0

    # Every assessed unit has an issued passport retrievable afterwards.
    passport = client.get(f"/passport/{result['items'][0]['asset_id']}")
    assert passport.status_code == 200


def test_assess_surfaces_ood_flag_and_review_path(
    client: TestClient, fleet: SimulatedFleet
) -> None:
    """An out-of-distribution unit is flagged and routed to manual review —
    never scored silently (spec §5 low-confidence state)."""
    from tests.test_engine.helpers import degraded_twin

    twin = degraded_twin(fleet.scans[0]).model_copy(update={"asset_id": "OOD-TWIN-001"})
    rows = [fleet.scans[1].model_dump(), twin.model_dump()]
    response = client.post("/scan", json={"provenance": "simulated", "rows": rows})
    assert response.status_code == 200

    result = _assess(client)
    by_asset = {item["asset_id"]: item for item in result["items"]}
    assert by_asset["OOD-TWIN-001"]["ood_flag"] is True
    assert by_asset["OOD-TWIN-001"]["recommendation"] == "manual_review"
    assert by_asset[fleet.scans[1].asset_id]["ood_flag"] is False


def test_assess_measured_provenance_flows_through(
    client: TestClient, fleet: SimulatedFleet
) -> None:
    """Measured-declared scans produce measured passports — never pooled."""
    ingest_fleet(client, fleet, n_units=2, provenance="measured")
    result = _assess(client)

    assert result["provenance_counts"] == {"measured": 2}
    asset_id = result["items"][0]["asset_id"]
    passport = client.get(f"/passport/{asset_id}").json()
    assert passport["provenance"] == "measured"
    assert passport["reliability"]["provenance"] == "measured"
    assert passport["disposition"]["provenance"] == "measured"


def test_assess_unknown_workload_is_422(client: TestClient, fleet: SimulatedFleet) -> None:
    """Unknown workload names are caller errors, not silent skips."""
    ingest_fleet(client, fleet, n_units=1)
    response = client.post("/assess", json=assess_request(target_workloads=["crypto-mining"]))
    assert response.status_code == 422
    assert "crypto-mining" in response.json()["detail"]


def test_assess_zero_horizon_is_422(client: TestClient, fleet: SimulatedFleet) -> None:
    """A non-positive horizon fails validation before any assessment runs."""
    ingest_fleet(client, fleet, n_units=1)
    response = client.post("/assess", json=assess_request(horizon_days=0))
    assert response.status_code == 422


def test_assess_economy_missing_price_is_422(client: TestClient, fleet: SimulatedFleet) -> None:
    """An economy without a price for every assessed workload is a 422."""
    ingest_fleet(client, fleet, n_units=1)
    body = assess_request()
    body["economy"]["resale_price_by_workload"].pop("inference")
    response = client.post("/assess", json=body)
    assert response.status_code == 422
    assert "inference" in response.json()["detail"]


def test_assess_unknown_asset_ids_is_422(client: TestClient, fleet: SimulatedFleet) -> None:
    """Requesting scans that were never stored is a 422, not an empty batch."""
    ingest_fleet(client, fleet, n_units=2)
    response = client.post("/assess", json=assess_request(asset_ids=["ghost-asset"]))
    assert response.status_code == 422
    assert "ghost-asset" in response.json()["detail"]


def test_assess_asset_id_subset(client: TestClient, fleet: SimulatedFleet) -> None:
    """asset_ids selects a subset of the stored batch."""
    ingest_fleet(client, fleet, n_units=3)
    selected = fleet.scans[0].asset_id
    result = _assess(client, asset_ids=[selected])
    assert [item["asset_id"] for item in result["items"]] == [selected]
    assert result["provenance_counts"] == {"simulated": 1}


def test_assess_empty_store_is_an_empty_batch(client: TestClient) -> None:
    """Assessing nothing stored is an explicit empty batch, not an error."""
    result = _assess(client)
    assert result["items"] == []
    assert result["provenance_counts"] == {}


def test_assess_generated_run_ids_differ(client: TestClient, fleet: SimulatedFleet) -> None:
    """Two runs get distinct run ids; each issues its own passports."""
    ingest_fleet(client, fleet, n_units=1)
    first = _assess(client)
    second = _assess(client)
    assert first["run_id"] != second["run_id"]
