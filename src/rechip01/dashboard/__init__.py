"""Server-rendered dashboard over the passport service (spec art_vuIPvUhR, §4–5).

Batch overview and per-unit passport pages. The honesty invariants are
enforced server-side: provenance badges, uncertainty bands, and review
routing are visible text in the rendered HTML, never a client-side toggle.
"""

from rechip01.dashboard.routes import DashboardDeps, build_dashboard_router
from rechip01.dashboard.state import UnitState, unit_state
from rechip01.dashboard.views import build_batch_view, build_passport_view

__all__ = [
    "DashboardDeps",
    "UnitState",
    "build_batch_view",
    "build_dashboard_router",
    "build_passport_view",
    "unit_state",
]
