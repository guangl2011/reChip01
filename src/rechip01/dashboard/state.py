"""Dashboard state derivation and formatting (spec art_vuIPvUhR, §4–5).

Pure functions only — no I/O, no framework objects — so the state machine and
number formatting stay unit-testable without HTTP.

States mirror the spec's assessment flow for *stored* units:

- ``pending`` — stored scan, complete inputs, not yet assessed.
- ``partial`` — stored scan with missing optional fields (explicit
  missingness flags, never silent imputation).
- ``ready`` — assessed; the optimizer issued a calibrated recommendation.
- ``review`` — assessed; the optimizer routed the unit to manual review
  (OOD flag or uncertainty above threshold). No automatic recommendation.

Rows rejected at ingest are quarantined with machine-readable reasons and are
never persisted to the append-only scan store, so they cannot appear on the
dashboard — rejection is visible in the ``POST /scan`` batch report instead.
"""

from __future__ import annotations

from enum import StrEnum

from rechip01.disposition.optimizer import PATH_MANUAL_REVIEW
from rechip01.schemas import AssetPassport, ScanResult


class UnitState(StrEnum):
    """Assessment-flow state of one stored unit on the dashboard."""

    PENDING = "pending"
    PARTIAL = "partial"
    READY = "ready"
    REVIEW = "review"


def missing_scan_fields(scan: ScanResult) -> tuple[str, ...]:
    """Names of optional scan fields that are absent (explicit missingness)."""
    missing: list[str] = []
    if scan.age_months is None:
        missing.append("age_months")
    if scan.power_on_hours is None:
        missing.append("power_on_hours")
    return tuple(missing)


def passport_state(passport: AssetPassport) -> UnitState:
    """Flow state of an assessed unit: a passport always exists here."""
    if passport.disposition.recommendation == PATH_MANUAL_REVIEW:
        return UnitState.REVIEW
    return UnitState.READY


def unit_state(scan: ScanResult, passport: AssetPassport | None) -> UnitState:
    """Derive the flow state of one stored unit.

    The passport's disposition recommendation is authoritative for assessed
    units: the optimizer already routes OOD and over-threshold uncertainty to
    ``manual_review``, so the dashboard never re-derives that decision (and
    can never disagree with the passport).
    """
    if passport is not None:
        return passport_state(passport)
    if missing_scan_fields(scan):
        return UnitState.PARTIAL
    return UnitState.PENDING


def format_pct(value: float) -> str:
    """Probability as a percentage with one decimal, e.g. ``12.3%``."""
    return f"{value * 100:.1f}%"


def format_ci(ci: tuple[float, float]) -> str:
    """Calibrated interval as a percentage band, e.g. ``8.1–19.4%``."""
    low, high = ci
    return f"{format_pct(low)}–{format_pct(high)}"


def format_money(value: float) -> str:
    """Risk-adjusted recovery in USD, e.g. ``$1,234.56``."""
    return f"${value:,.2f}"


def format_bool_flag(flag: bool) -> str:
    """Render a boolean flag as visible text — flags are never blank."""
    return "yes" if flag else "no"


__all__ = [
    "UnitState",
    "format_bool_flag",
    "format_ci",
    "format_money",
    "format_pct",
    "missing_scan_fields",
    "unit_state",
]
