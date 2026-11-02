"""GET /batch overview and CSV/JSON export tests."""

from __future__ import annotations

import csv
import io

from fastapi.testclient import TestClient

from rechip01.simulator import SimulatedFleet
from tests.test_api.conftest import assess_request, ingest_fleet


def test_batch_overview_joins_passport_uncertainty(
    client: TestClient, fleet: SimulatedFleet
) -> None:
    """After assessment, overview rows carry the passport's uncertainty and
    recommendation; before it, they are explicitly unassessed."""
    ingest_fleet(client, fleet, n_units=2)

    before = client.get("/batch").json()
    assert all(unit["assessed"] is False for unit in before["units"])
    assert all(unit["p_fail"] is None for unit in before["units"])

    response = client.post("/assess", json=assess_request())
    assert response.status_code == 200, response.text

    after = client.get("/batch").json()
    assert all(unit["assessed"] is True for unit in after["units"])
    for unit in after["units"]:
        low, high = unit["ci"]
        assert 0.0 <= low <= unit["p_fail"] <= high <= 1.0
        assert unit["ood_flag"] is not None
        assert unit["recommendation"]


def test_batch_overview_mixed_provenance_counts(client: TestClient, fleet: SimulatedFleet) -> None:
    """Simulated and measured units are counted separately, never pooled."""
    ingest_fleet(client, fleet, n_units=2, provenance="simulated")
    ingest_fleet(client, fleet, n_units=1, provenance="measured", start=5)

    overview = client.get("/batch").json()
    assert overview["total_units"] == 3
    assert overview["provenance_counts"] == {"simulated": 2, "measured": 1}


def test_empty_batch_overview(client: TestClient) -> None:
    """An empty store is an explicit empty overview, not an error."""
    overview = client.get("/batch").json()
    assert overview == {"total_units": 0, "provenance_counts": {}, "units": []}


def test_csv_export(client: TestClient, fleet: SimulatedFleet) -> None:
    """CSV export: one row per unit, provenance explicit, uncertainty where
    assessed and empty where not."""
    ingest_fleet(client, fleet, n_units=2)
    response = client.post("/assess", json=assess_request())
    assert response.status_code == 200, response.text

    export = client.get("/batch/export.csv")
    assert export.status_code == 200
    assert export.headers["content-type"].startswith("text/csv")

    rows = list(csv.DictReader(io.StringIO(export.text)))
    assert len(rows) == 2
    assert {row["provenance"] for row in rows} == {"simulated"}
    for row in rows:
        assert 0.0 <= float(row["p_fail"]) <= 1.0
        assert float(row["ci_low"]) <= float(row["p_fail"]) <= float(row["ci_high"])
        assert row["ood_flag"] in {"True", "False"}
        assert row["recommendation"]


def test_csv_export_before_assessment_has_empty_assessment_columns(
    client: TestClient, fleet: SimulatedFleet
) -> None:
    """Unassessed units export with empty assessment fields — no fake zeros."""
    ingest_fleet(client, fleet, n_units=1)
    export = client.get("/batch/export.csv")
    rows = list(csv.DictReader(io.StringIO(export.text)))
    assert rows[0]["p_fail"] == ""
    assert rows[0]["recommendation"] == ""
    assert rows[0]["provenance"] == "simulated"


def test_json_export_matches_overview(client: TestClient, fleet: SimulatedFleet) -> None:
    """The JSON export is exactly the /batch payload."""
    ingest_fleet(client, fleet, n_units=1)
    overview = client.get("/batch").json()
    export = client.get("/batch/export.json").json()
    assert export == overview
