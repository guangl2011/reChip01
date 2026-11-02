"""FastAPI service for assessment and asset passports (spec art_vuIPvUhR, §4–5).

Endpoints over the file-backed repositories:

- ``POST /scan`` — ingest a batch (raw rows + declared provenance); accepted
  records persist to the scan store, quarantined rows return with
  machine-readable reasons.
- ``GET /batch`` — batch overview: every stored unit with its provenance and,
  once assessed, its passport's uncertainty and recommendation.
- ``POST /assess`` — one assessment run over stored scans: engine scores,
  disposition optimizes, passports issue. Retrying the same ``run_id`` is a
  409 — passports are immutable and issued exactly once per (asset, run).
- ``GET /passport/{asset_id}`` — the latest-issued passport for one asset.
- ``GET /dashboard`` — server-rendered batch overview page.
- ``GET /dashboard/passport/{asset_id}`` — server-rendered passport page.
- ``GET /health`` — liveness.

Honesty invariants (spec §5 flow states): invalid rows never proceed
silently (quarantined with reasons), OOD and low-confidence units surface
their flag and route to manual review rather than receiving a confident
recommendation, and provenance rides on every response — simulated and
measured data stay structurally distinct end to end.

The service never issues accredited third-party certification; every
passport is decision support with explicit uncertainty.
"""

from __future__ import annotations

import csv
import uuid
from dataclasses import dataclass
from io import StringIO
from pathlib import Path
from typing import Final

from fastapi import FastAPI, HTTPException, Response

from rechip01.dashboard.routes import build_dashboard_router
from rechip01.disposition.optimizer import optimize_disposition
from rechip01.engine.engine import ReliabilityEngine
from rechip01.health import APP_NAME, HealthStatus, get_health
from rechip01.passport.builder import build_passport
from rechip01.passport.models import (
    AssessmentSummary,
    AssessRequest,
    AssessResponse,
    BatchOverview,
    ScanIngestRequest,
    UnitOverview,
)
from rechip01.repository import (
    DuplicatePassportError,
    DuplicateScanError,
    JsonPassportRepository,
    JsonScanResultRepository,
    PassportRepository,
    ScanResultRepository,
)
from rechip01.scanner import (
    AcceptedUnit,
    Provenance,
    RejectedUnit,
    RejectionCode,
    RejectionReason,
    scan_batch,
)
from rechip01.scanner.models import ScanBatchReport
from rechip01.schemas import AssetPassport, DispositionDecision, ReliabilityEstimate, ScanResult
from rechip01.simulator.config import WORKLOAD_PROFILES, WorkloadProfile, default_fleet_config
from rechip01.simulator.generate import generate_fleet

_RUN_ID_PREFIX: Final[str] = "run-"
_RUN_ID_CHARS: Final[int] = 12


@dataclass(frozen=True)
class ServiceDeps:
    """Injected service state: fitted engine plus the two file-backed stores."""

    engine: ReliabilityEngine
    scans: ScanResultRepository
    passports: PassportRepository
    training_fingerprint: str | None = None


def default_deps(
    root: Path, *, training_seed: int = 11, n_training_units: int = 400
) -> ServiceDeps:
    """Build service dependencies in demo mode: engine trained on a seeded
    simulated fleet, JSON stores under ``root``.

    Demo training is simulated by construction — every unit assessed by this
    service is scored against that fleet, and provenance on responses always
    reflects the *scan's* origin, never the training fleet's.
    """
    fleet = generate_fleet(default_fleet_config(seed=training_seed, n_units=n_training_units))
    engine = ReliabilityEngine().fit(fleet)
    return ServiceDeps(
        engine=engine,
        scans=JsonScanResultRepository(Path(root) / "scans"),
        passports=JsonPassportRepository(Path(root) / "passports"),
        training_fingerprint=engine.training_fingerprint(),
    )


def create_app(deps: ServiceDeps, *, title: str = APP_NAME) -> FastAPI:
    """Build the FastAPI application over injected dependencies."""
    app = FastAPI(title=title)
    app.state.deps = deps
    app.include_router(build_dashboard_router(deps))

    # ------------------------------------------------------------------ scan
    @app.post("/scan", response_model=ScanBatchReport)
    def scan(request: ScanIngestRequest) -> ScanBatchReport:
        """Ingest one batch; accepted units persist, bad rows are quarantined."""
        report = scan_batch(request.rows, request.provenance, request.source_format)
        return _persist_report(report, deps.scans)

    # ----------------------------------------------------------------- batch
    @app.get("/batch", response_model=BatchOverview)
    def batch() -> BatchOverview:
        """Every stored unit with provenance and, where issued, its passport."""
        stored = deps.scans.list_all()
        units: list[UnitOverview] = []
        for record in stored:
            passport = deps.passports.get(record.asset_id)
            units.append(_unit_overview(record, passport))
        return BatchOverview(
            total_units=len(units),
            provenance_counts=dict(_provenance_counts([record.provenance for record in stored])),
            units=tuple(units),
        )

    # ---------------------------------------------------------------- assess
    @app.post("/assess", response_model=AssessResponse)
    def assess(request: AssessRequest) -> AssessResponse:
        """Assess stored scans and issue one passport per unit."""
        scans = _select_scans(deps, request.asset_ids)
        profiles = _resolve_workloads(request.target_workloads)
        run_id = request.run_id if request.run_id is not None else _new_run_id()
        try:
            estimates = deps.engine.assess(
                scans, horizon_days=request.horizon_days, target_workloads=profiles
            )
        except ValueError as error:
            # Engine contract violations (bad horizon, empty workload list) are
            # caller errors: surfaced as 422, never as a silent partial run.
            raise HTTPException(status_code=422, detail=str(error)) from error

        items: list[AssessmentSummary] = []
        for index, (scan, estimate) in enumerate(zip(scans, estimates, strict=True)):
            try:
                decision = optimize_disposition(estimate, request.economy, seed=index)
            except ValueError as error:
                raise HTTPException(status_code=422, detail=str(error)) from error
            try:
                deps.passports.save(
                    build_passport(
                        scan,
                        estimate,
                        decision,
                        run_id=run_id,
                        training_fingerprint=deps.training_fingerprint,
                    )
                )
            except DuplicatePassportError as error:
                raise HTTPException(status_code=409, detail=str(error)) from error
            items.append(_assessment_summary(estimate, decision))

        return AssessResponse(
            run_id=run_id,
            horizon_days=request.horizon_days,
            target_workloads=request.target_workloads,
            provenance_counts=dict(_provenance_counts([scan.provenance for scan in scans])),
            items=tuple(items),
        )

    # -------------------------------------------------------------- passport
    @app.get("/passport/{asset_id}", response_model=AssetPassport)
    def get_passport(asset_id: str) -> AssetPassport:
        """Latest-issued passport for one asset (spec §5)."""
        passport = deps.passports.get(asset_id)
        if passport is None:
            raise HTTPException(status_code=404, detail=f"no passport issued for {asset_id!r}")
        return passport

    # ---------------------------------------------------------------- export
    @app.get("/batch/export.csv")
    def export_batch_csv() -> Response:
        """Batch overview as CSV — same fields as ``GET /batch``, one row per
        unit, provenance explicit on every row."""
        overview = _batch_overview(deps)
        buffer = StringIO(newline="")
        writer = csv.writer(buffer, lineterminator="\n")
        writer.writerow(
            [
                "asset_id",
                "model",
                "provenance",
                "assessed",
                "p_fail",
                "ci_low",
                "ci_high",
                "ood_flag",
                "recommendation",
            ]
        )
        for unit in overview.units:
            writer.writerow(
                [
                    unit.asset_id,
                    unit.model,
                    unit.provenance.value,
                    unit.assessed,
                    "" if unit.p_fail is None else unit.p_fail,
                    "" if unit.ci is None else unit.ci[0],
                    "" if unit.ci is None else unit.ci[1],
                    "" if unit.ood_flag is None else unit.ood_flag,
                    "" if unit.recommendation is None else unit.recommendation,
                ]
            )
        return Response(
            content=buffer.getvalue(), media_type="text/csv", headers={"Cache-Control": "no-store"}
        )

    @app.get("/batch/export.json", response_model=BatchOverview)
    def export_batch_json() -> BatchOverview:
        """Batch overview as JSON (machine-readable export of ``GET /batch``)."""
        return _batch_overview(deps)

    # ---------------------------------------------------------------- health
    @app.get("/health", response_model=HealthStatus)
    def health() -> HealthStatus:
        """Service liveness."""
        return get_health()

    return app


# --------------------------------------------------------------------------
# Route helpers — pure where practical so they stay unit-testable
# --------------------------------------------------------------------------
def _persist_report(report: ScanBatchReport, scans: ScanResultRepository) -> ScanBatchReport:
    """Persist accepted records; a store collision quarantines that row.

    Re-ingesting a stored asset is an explicit per-row rejection with a
    machine-readable reason — the earlier record is never overwritten.
    """
    accepted: list[AcceptedUnit] = []
    rejected: list[RejectedUnit] = list(report.rejected)
    for unit in report.accepted:
        try:
            scans.save(unit.result)
        except DuplicateScanError:
            rejected.append(
                RejectedUnit(
                    row_index=unit.row_index,
                    asset_id=unit.result.asset_id,
                    reasons=(
                        RejectionReason(
                            code=RejectionCode.DUPLICATE_ASSET_ID,
                            field="asset_id",
                            detail=(
                                f"asset {unit.result.asset_id!r} is already stored; "
                                "stored records are append-only and never overwritten"
                            ),
                        ),
                    ),
                )
            )
        else:
            accepted.append(unit)
    return report.model_copy(update={"accepted": tuple(accepted), "rejected": tuple(rejected)})


def _select_scans(deps: ServiceDeps, asset_ids: tuple[str, ...] | None) -> list[ScanResult]:
    """Resolve the assessment batch; unknown requested IDs are a 422."""
    if asset_ids is None:
        return deps.scans.list_all()
    missing = sorted(set(asset_ids) - {scan.asset_id for scan in deps.scans.list_all()})
    if missing:
        raise HTTPException(status_code=422, detail=f"no stored scans for asset ids: {missing}")
    return [scan for scan in deps.scans.list_all() if scan.asset_id in set(asset_ids)]


def _resolve_workloads(names: tuple[str, ...]) -> list[WorkloadProfile]:
    """Resolve workload names against the built-in profiles; unknown → 422."""
    unknown = sorted(set(names) - set(WORKLOAD_PROFILES))
    if unknown:
        raise HTTPException(
            status_code=422,
            detail=f"unknown target workloads: {unknown}; known: {sorted(WORKLOAD_PROFILES)}",
        )
    return [WORKLOAD_PROFILES[name] for name in names]


def _provenance_counts(provenances: list[Provenance]) -> dict[Provenance, int]:
    counts: dict[Provenance, int] = {}
    for provenance in provenances:
        counts[provenance] = counts.get(provenance, 0) + 1
    return counts


def _batch_overview(deps: ServiceDeps) -> BatchOverview:
    """Join every stored scan with its latest passport (uncertainty when issued)."""
    stored = deps.scans.list_all()
    units = [_unit_overview(record, deps.passports.get(record.asset_id)) for record in stored]
    return BatchOverview(
        total_units=len(units),
        provenance_counts=dict(_provenance_counts([record.provenance for record in stored])),
        units=tuple(units),
    )


def _unit_overview(record: ScanResult, passport: AssetPassport | None) -> UnitOverview:
    """One overview row: scan identity plus passport uncertainty when issued."""
    if passport is None:
        return UnitOverview(
            asset_id=record.asset_id,
            model=record.model,
            provenance=record.provenance,
            assessed=False,
        )
    return UnitOverview(
        asset_id=record.asset_id,
        model=record.model,
        provenance=record.provenance,
        assessed=True,
        p_fail=passport.reliability.p_fail,
        ci=passport.reliability.ci,
        ood_flag=passport.reliability.ood_flag,
        recommendation=passport.disposition.recommendation,
    )


def _assessment_summary(
    estimate: ReliabilityEstimate, decision: DispositionDecision
) -> AssessmentSummary:
    return AssessmentSummary(
        asset_id=estimate.asset_id,
        provenance=estimate.provenance,
        p_fail=estimate.p_fail,
        ci=estimate.ci,
        ood_flag=estimate.ood_flag,
        recommendation=decision.recommendation,
        expected_net_recovery=decision.expected_net_recovery,
        regret_bound=decision.regret_bound,
    )


def _new_run_id() -> str:
    return f"{_RUN_ID_PREFIX}{uuid.uuid4().hex[:_RUN_ID_CHARS]}"


__all__ = ["ServiceDeps", "create_app", "default_deps"]
