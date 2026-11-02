"""Passport page tests (spec art_vuIPvUhR, §5)."""

from __future__ import annotations

from typing import Any

from fastapi.testclient import TestClient
from tests.test_dashboard.conftest import assess_all, ingest_rows

from rechip01.simulator import SimulatedFleet


def test_passport_page_renders_full_record(client: TestClient, fleet: SimulatedFleet) -> None:
    """The passport page renders the JSON API's own passport as its oracle:
    identity, provenance, uncertainty, disposition, traces, and record."""
    ingest_rows(client, [fleet.scans[0].model_dump()])
    assess_all(client)
    asset_id = fleet.scans[0].asset_id
    passport: dict[str, Any] = client.get(f"/passport/{asset_id}").json()

    response = client.get(f"/dashboard/passport/{asset_id}")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")

    html = response.text
    assert asset_id in html
    assert passport["run_id"] in html
    assert passport["model_version"] in html
    assert passport["provenance"] in html
    assert passport["disposition"]["recommendation"] in html
    assert "%" in html  # p_fail / CI / fitness rendered as visible percentages
    assert "Calibrated interval" in html
    assert "Evidence trace" in html
    for line in passport["evidence_trace"]:
        assert line in html
    for line in passport["disposition"]["assumption_trace"]:
        assert line in html


def test_unassessed_asset_renders_404_page(client: TestClient, fleet: SimulatedFleet) -> None:
    """A stored-but-unassessed unit has no passport yet — a rendered 404 page
    that says so, not a blank screen or a JSON error."""
    ingest_rows(client, [fleet.scans[0].model_dump()])

    response = client.get(f"/dashboard/passport/{fleet.scans[0].asset_id}")
    assert response.status_code == 404
    assert "No passport" in response.text
    assert "assessed" in response.text


def test_unknown_asset_renders_404_page(client: TestClient) -> None:
    response = client.get("/dashboard/passport/does-not-exist")
    assert response.status_code == 404
    assert "No passport" in response.text
    assert "does-not-exist" in response.text
