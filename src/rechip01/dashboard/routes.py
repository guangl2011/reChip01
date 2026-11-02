"""Server-rendered dashboard routes (spec art_vuIPvUhR, §4–5).

Two HTML pages over the same ``ServiceDeps`` as the JSON API — server-rendered
so the honesty invariants (provenance badges, uncertainty bands, review
routing) are enforced server-side and cannot be toggled off client-side:

- ``GET /dashboard`` — batch overview: summary counts plus one row per stored
  unit with its flow state, provenance badge, uncertainty, and recommendation.
- ``GET /dashboard/passport/{asset_id}`` — the full passport page: reliability
  estimate with uncertainty, disposition decision, evidence trace.

A passport that does not exist (unknown asset, or stored but not yet
assessed) is a rendered 404 page, not a JSON error — this is a browser
surface. The root path redirects to the batch overview.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates

from rechip01.dashboard.views import build_batch_view, build_passport_view
from rechip01.repository import PassportRepository, ScanResultRepository

_TEMPLATES_DIR = Jinja2Templates(directory=str(Path(__file__).parent / "templates"))


class DashboardDeps(Protocol):
    """What the dashboard needs; ``ServiceDeps`` satisfies this structurally."""

    @property
    def scans(self) -> ScanResultRepository: ...

    @property
    def passports(self) -> PassportRepository: ...


def build_dashboard_router(deps: DashboardDeps) -> APIRouter:
    """HTML routes over injected repositories (no DB, no extra state).

    Excluded from the OpenAPI schema: the OpenAPI document is the JSON API's
    machine contract (guarded as an exact set by the API tests); browser
    pages are not part of that contract.
    """
    router = APIRouter(include_in_schema=False)
    templates = _TEMPLATES_DIR

    @router.get("/", response_class=RedirectResponse)
    def root() -> RedirectResponse:
        return RedirectResponse(url="/dashboard", status_code=303)

    @router.get("/dashboard", response_class=HTMLResponse)
    def dashboard(request: Request) -> HTMLResponse:
        view = build_batch_view(deps.scans, deps.passports)
        return templates.TemplateResponse(
            request=request,
            name="batch.html",
            context={"view": view},
        )

    @router.get("/dashboard/passport/{asset_id}", response_class=HTMLResponse)
    def passport_page(request: Request, asset_id: str) -> HTMLResponse:
        passport = deps.passports.get(asset_id)
        if passport is None:
            return templates.TemplateResponse(
                request=request,
                name="passport_missing.html",
                context={"asset_id": asset_id},
                status_code=404,
            )
        return templates.TemplateResponse(
            request=request,
            name="passport.html",
            context={"view": build_passport_view(passport)},
        )

    return router


__all__ = ["DashboardDeps", "build_dashboard_router"]
