"""POST /scan integration tests: ingest, quarantine, and persistence."""

from __future__ import annotations

from typing import Any

from fastapi.testclient import TestClient

from rechip01.simulator import SimulatedFleet
from tests.test_api.conftest import ingest_fleet
from tests.test_scanner.fixtures import valid_row


def test_ingest_persists_and_reports(client: TestClient, fleet: SimulatedFleet) -> None:
    """Valid rows are accepted with declared provenance and persist for /batch."""
    report = ingest_fleet(client, fleet, n_units=3)
    assert len(report["accepted"]) == 3
    assert len(report["rejected"]) == 0
    assert report["total_rows"] == 3
    assert report["provenance"] == "simulated"

    overview = client.get("/batch").json()
    assert overview["total_units"] == 3
    assert overview["provenance_counts"] == {"simulated": 3}
    assert all(unit["assessed"] is False for unit in overview["units"])


def test_invalid_rows_quarantined_with_reasons(client: TestClient) -> None:
    """Schema violations and non-object rows are quarantined, never dropped."""
    response = client.post(
        "/scan",
        json={"provenance": "simulated", "rows": [valid_row(), {"model": "no-asset-id"}, 42]},
    )
    assert response.status_code == 200
    report = response.json()
    assert len(report["accepted"]) == 1
    assert len(report["rejected"]) == 2
    codes = sorted(reason["code"] for unit in report["rejected"] for reason in unit["reasons"])
    assert set(codes) == {"missing_required_field", "parse_error"}


def test_duplicate_within_batch_quarantined(client: TestClient) -> None:
    """The same asset twice in one batch: the second row is quarantined."""
    row = valid_row()
    response = client.post("/scan", json={"provenance": "simulated", "rows": [row, row]})
    report = response.json()
    assert len(report["accepted"]) == 1
    reasons = report["rejected"][0]["reasons"]
    assert reasons[0]["code"] == "duplicate_asset_id"
    assert reasons[0]["field"] == "asset_id"


def test_duplicate_across_batches_never_overwrites(
    client: TestClient, fleet: SimulatedFleet
) -> None:
    """Re-ingesting a stored asset is an explicit per-row rejection; the
    earlier record is append-only and untouched."""
    first = ingest_fleet(client, fleet, n_units=1)
    assert len(first["accepted"]) == 1

    stored_before = client.get("/batch").json()["units"]
    second = ingest_fleet(client, fleet, n_units=1)
    assert len(second["accepted"]) == 0
    assert len(second["rejected"]) == 1
    assert second["rejected"][0]["reasons"][0]["code"] == "duplicate_asset_id"
    assert "already stored" in second["rejected"][0]["reasons"][0]["detail"]

    assert client.get("/batch").json()["units"] == stored_before


def test_provenance_conflict_quarantined(client: TestClient) -> None:
    """A row claiming measured provenance under a simulated declaration is
    quarantined — provenance conflicts are never silently resolved."""
    response = client.post(
        "/scan",
        json={"provenance": "simulated", "rows": [valid_row(provenance="measured")]},
    )
    assert response.status_code == 200
    report = response.json()
    assert len(report["accepted"]) == 0
    assert report["rejected"][0]["reasons"][0]["code"] == "provenance_conflict"


def test_measured_provenance_labeled_end_to_end(client: TestClient, fleet: SimulatedFleet) -> None:
    """A measured-declared batch stays measured on ingest and on /batch."""
    report = ingest_fleet(client, fleet, n_units=2, provenance="measured")
    assert report["provenance"] == "measured"
    overview = client.get("/batch").json()
    assert overview["provenance_counts"] == {"measured": 2}
    assert {unit["provenance"] for unit in overview["units"]} == {"measured"}


def test_missing_provenance_is_a_validation_error(client: TestClient) -> None:
    """A batch without declared provenance is a 4xx, never a silent default."""
    response = client.post("/scan", json={"rows": [valid_row()]})
    assert response.status_code == 422
    assert response.json()["detail"][0]["loc"] == ["body", "provenance"]


def test_unknown_provenance_value_is_a_validation_error(client: TestClient) -> None:
    """Unknown provenance labels are rejected at the schema boundary."""
    response = client.post("/scan", json={"provenance": "unverified", "rows": []})
    assert response.status_code == 422


def test_source_format_recorded_for_lineage(client: TestClient, fleet: SimulatedFleet) -> None:
    """The declared source format rides on the report for lineage."""
    rows: list[dict[str, Any]] = [scan.model_dump() for scan in fleet.scans[:1]]
    response = client.post(
        "/scan",
        json={"provenance": "simulated", "source_format": "jsonl_export", "rows": rows},
    )
    assert response.status_code == 200
    assert response.json()["source_format"] == "jsonl_export"
