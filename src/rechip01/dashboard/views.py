"""Dashboard view models built from repositories (spec art_vuIPvUhR, §4–5).

Pure builders from ``(scans, passports)`` to frozen view models the templates
render verbatim. All honesty invariants are baked in here — provenance labels
and flags are pre-formatted visible text, never behind a toggle — so the
templates cannot accidentally hide them.
"""

from __future__ import annotations

from dataclasses import dataclass

from rechip01.dashboard.state import (
    UnitState,
    format_bool_flag,
    format_ci,
    format_money,
    format_pct,
    missing_scan_fields,
    passport_state,
    unit_state,
)
from rechip01.repository import PassportRepository, ScanResultRepository
from rechip01.schemas import AssetPassport


@dataclass(frozen=True)
class UnitRow:
    """One row on the batch overview table."""

    asset_id: str
    model: str
    provenance: str
    state: UnitState
    missing_fields: tuple[str, ...]
    p_fail: str | None
    ci: str | None
    ood_flag: str | None
    recommendation: str | None
    href: str


@dataclass(frozen=True)
class BatchView:
    """Batch overview: summary counts plus one row per stored unit."""

    total_units: int
    simulated_count: int
    measured_count: int
    review_count: int
    units: tuple[UnitRow, ...]


@dataclass(frozen=True)
class OptionRow:
    """One disposition path the optimizer evaluated."""

    path: str
    expected_net_recovery: str
    regret_bound: str
    recommended: bool


@dataclass(frozen=True)
class PassportView:
    """One asset passport, flattened for the passport page."""

    asset_id: str
    provenance: str
    state: UnitState
    issued_at: str
    run_id: str
    model_version: str
    p_fail: str
    ci: str
    horizon_days: int
    ood_flag: str
    workload_fitness: tuple[tuple[str, str], ...]
    recommendation: str
    expected_net_recovery: str
    regret_bound: str
    options: tuple[OptionRow, ...]
    assumption_trace: tuple[str, ...]
    evidence_trace: tuple[str, ...]


def build_batch_view(scans: ScanResultRepository, passports: PassportRepository) -> BatchView:
    """One row per stored scan, joined with its passport when issued."""
    rows: list[UnitRow] = []
    simulated = 0
    measured = 0
    review = 0
    for scan in scans.list_all():
        passport = passports.get(scan.asset_id)
        state = unit_state(scan, passport)
        if scan.provenance.value == "simulated":
            simulated += 1
        else:
            measured += 1
        if state is UnitState.REVIEW:
            review += 1
        reliability = passport.reliability if passport is not None else None
        disposition = passport.disposition if passport is not None else None
        rows.append(
            UnitRow(
                asset_id=scan.asset_id,
                model=scan.model,
                provenance=scan.provenance.value,
                state=state,
                missing_fields=missing_scan_fields(scan),
                p_fail=None if reliability is None else format_pct(reliability.p_fail),
                ci=None if reliability is None else format_ci(reliability.ci),
                ood_flag=None if reliability is None else format_bool_flag(reliability.ood_flag),
                recommendation=None if disposition is None else disposition.recommendation,
                href=f"/dashboard/passport/{scan.asset_id}",
            )
        )
    return BatchView(
        total_units=len(rows),
        simulated_count=simulated,
        measured_count=measured,
        review_count=review,
        units=tuple(rows),
    )


def build_passport_view(passport: AssetPassport) -> PassportView:
    """Flatten one issued passport into its page view model."""
    reliability = passport.reliability
    disposition = passport.disposition
    return PassportView(
        asset_id=passport.asset_id,
        provenance=passport.provenance.value,
        state=passport_state(passport),
        issued_at=passport.issued_at.strftime("%Y-%m-%d %H:%M:%S %Z"),
        run_id=passport.run_id,
        model_version=passport.model_version,
        p_fail=format_pct(reliability.p_fail),
        ci=format_ci(reliability.ci),
        horizon_days=reliability.horizon_days,
        ood_flag=format_bool_flag(reliability.ood_flag),
        workload_fitness=tuple(
            (workload, format_pct(fitness))
            for workload, fitness in sorted(reliability.workload_fitness.items())
        ),
        recommendation=disposition.recommendation,
        expected_net_recovery=format_money(disposition.expected_net_recovery),
        regret_bound=format_money(disposition.regret_bound),
        options=tuple(
            OptionRow(
                path=option.path,
                expected_net_recovery=format_money(option.expected_net_recovery),
                regret_bound=format_money(option.regret_bound),
                recommended=option.path == disposition.recommendation,
            )
            for option in disposition.options_evaluated
        ),
        assumption_trace=tuple(disposition.assumption_trace),
        evidence_trace=tuple(passport.evidence_trace),
    )


__all__ = [
    "BatchView",
    "OptionRow",
    "PassportView",
    "UnitRow",
    "build_batch_view",
    "build_passport_view",
]
