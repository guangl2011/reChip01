"""Tests proving the package imports and the health check serves."""

import rechip01
from rechip01.health import HealthStatus, app, get_health


def test_package_imports_and_reports_version() -> None:
    """The package imports and exposes a valid semver string."""
    assert rechip01.__version__.count(".") == 2


def test_get_health_reports_ok() -> None:
    """The health builder reports ok with a version string."""
    status = get_health()
    assert status.status == "ok"
    assert status.version


def test_health_model_fields() -> None:
    """HealthStatus validates its fields."""
    status = HealthStatus(status="ok", version="0.1.0")
    assert status.status == "ok"
    assert status.version == "0.1.0"


def test_health_endpoint_serves() -> None:
    """The FastAPI app serves /health with a 200 payload."""
    from fastapi.testclient import TestClient

    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert payload["version"]
