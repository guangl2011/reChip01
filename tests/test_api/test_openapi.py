"""OpenAPI schema checks: paths exist, honesty fields ride every response."""

from __future__ import annotations

from fastapi.testclient import TestClient


def test_expected_paths_present(client: TestClient) -> None:
    """The service exposes the spec §5 contract endpoints plus health."""
    schema = client.get("/openapi.json").json()
    assert set(schema["paths"]) == {
        "/scan",
        "/batch",
        "/batch/export.csv",
        "/batch/export.json",
        "/assess",
        "/passport/{asset_id}",
        "/health",
    }


def test_response_schemas_carry_provenance_and_uncertainty(client: TestClient) -> None:
    """Every assessment-bearing response schema carries provenance; passport
    responses carry the evidence trace, model version, and uncertainty."""
    schema = client.get("/openapi.json").json()
    components = schema["components"]["schemas"]

    # Mixed-provenance batches have no single provenance: BatchOverview
    # carries per-provenance counts instead.
    for model in ("AssessmentSummary", "AssetPassport"):
        assert "provenance" in components[model]["properties"], model
    assert "provenance_counts" in components["BatchOverview"]["properties"]

    passport = components["AssetPassport"]["properties"]
    assert "evidence_trace" in passport
    assert "model_version" in passport
    assert "issued_at" in passport

    summary = components["AssessmentSummary"]["properties"]
    assert "p_fail" in summary and "ci" in summary and "ood_flag" in summary

    scan_report = components["ScanBatchReport"]["properties"]
    assert "provenance" in scan_report and "rejected" in scan_report


def test_ingest_and_assess_requests_are_strictly_typed(client: TestClient) -> None:
    """Required request fields are marked required in the OpenAPI document."""
    schema = client.get("/openapi.json").json()
    ingest = schema["components"]["schemas"]["ScanIngestRequest"]
    assert set(ingest["required"]) == {"provenance", "rows"}
    assess = schema["components"]["schemas"]["AssessRequest"]
    assert set(assess["required"]) == {"horizon_days", "target_workloads", "economy"}


def test_health_serves(client: TestClient) -> None:
    """/health reports liveness and package version."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"
