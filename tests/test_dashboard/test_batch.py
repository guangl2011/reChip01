"""Batch overview page tests (spec art_vuIPvUhR, §5)."""

from __future__ import annotations

from fastapi.testclient import TestClient
from tests.test_dashboard.conftest import assess_all, ingest_rows
from tests.test_dashboard.helpers import make_review_passport

from rechip01.simulator import SimulatedFleet


def test_root_redirects_to_dashboard(client: TestClient) -> None:
    response = client.get("/", follow_redirects=False)
    assert response.status_code in {303, 307}
    assert response.headers["location"] == "/dashboard"


def test_empty_batch_renders_empty_state_with_provenance_legend(
    client: TestClient,
) -> None:
    """Empty state names the action to take; provenance labels are visible
    even with zero units (summary cards), never hidden on any screen."""
    response = client.get("/dashboard")
    assert response.status_code == 200
    assert "No units stored yet" in response.text
    assert "simulated" in response.text
    assert "measured" in response.text
    assert response.headers["content-type"].startswith("text/html")


def test_batch_renders_pending_units_with_provenance_badges(
    client: TestClient, fleet: SimulatedFleet
) -> None:
    rows = [fleet.scans[0].model_dump(), fleet.scans[1].model_dump()]
    ingest_rows(client, rows)

    html = client.get("/dashboard").text
    assert fleet.scans[0].asset_id in html
    assert fleet.scans[1].asset_id in html
    assert "pending" in html
    assert ">simulated<" in html
    assert "units stored" in html


def test_partial_unit_shows_missing_fields(client: TestClient, fleet: SimulatedFleet) -> None:
    row = fleet.scans[2].model_dump()
    del row["age_months"]
    ingest_rows(client, [row])

    html = client.get("/dashboard").text
    assert "partial" in html
    assert "missing: age_months" in html


def test_batch_after_assessment_renders_uncertainty_and_states(
    client: TestClient, fleet: SimulatedFleet
) -> None:
    ingest_rows(client, [scan.model_dump() for scan in fleet.scans[:2]])
    assess_all(client)

    html = client.get("/dashboard").text
    overview = client.get("/batch").json()
    for unit in overview["units"]:
        assert unit["asset_id"] in html
        if unit["assessed"]:
            # Percentages, recommendation, and the OOD answer rendered as
            # visible text — uncertainty and flags are never blank.
            assert "%" in html
            assert unit["recommendation"] in html
            assert unit["ood_flag"] is not None


def test_batch_renders_review_state_for_ood_units(
    client: TestClient, fleet: SimulatedFleet
) -> None:
    """An OOD unit's passport routes to manual_review; the overview shows the
    review state and the review-count card reflects it."""
    ingest_rows(client, [fleet.scans[0].model_dump()])
    deps = client.app.state.deps  # type: ignore[attr-defined]
    deps.passports.save(make_review_passport(fleet.scans[0]))

    html = client.get("/dashboard").text
    assert "review" in html
    assert "1" in html  # review-count card shows one review-routed unit
