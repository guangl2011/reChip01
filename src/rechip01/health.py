"""Runnable health check: proves the package installs, imports, and serves.

Kept deliberately small so CI can verify the whole toolchain (ruff + mypy +
pytest) against a real FastAPI application before any domain logic lands.
"""

from importlib.metadata import PackageNotFoundError, version

from fastapi import FastAPI
from pydantic import BaseModel

APP_NAME = "reChip01"


def package_version() -> str:
    """Return the installed distribution version, falling back for source runs."""
    try:
        return version("rechip01")
    except PackageNotFoundError:
        return "0.0.0+unknown"


class HealthStatus(BaseModel):
    """Liveness payload for the health endpoint."""

    status: str
    version: str


def get_health() -> HealthStatus:
    """Build the current health status."""
    return HealthStatus(status="ok", version=package_version())


app = FastAPI(title=APP_NAME, version=package_version())


@app.get("/health", response_model=HealthStatus)
def health() -> HealthStatus:
    """Report service liveness and package version."""
    return get_health()
