"""Pure state-derivation tests for the dashboard (spec §5 flow states)."""

from __future__ import annotations

from tests.test_dashboard.helpers import make_review_passport, ready_passport
from tests.test_disposition.helpers import make_economy, make_estimate
from tests.test_engine.helpers import make_fleet

from rechip01.dashboard import UnitState, unit_state
from rechip01.disposition.optimizer import optimize_disposition
from rechip01.passport.builder import build_passport
from rechip01.schemas import AssetPassport, ScanResult

_ECONOMY = make_economy()


def test_unassessed_complete_scan_is_pending() -> None:
    scan = make_fleet(seed=1, n_units=1).scans[0]
    assert unit_state(scan, None) is UnitState.PENDING


def test_unassessed_scan_with_missing_optional_field_is_partial() -> None:
    scan = make_fleet(seed=1, n_units=1).scans[0].model_copy(update={"age_months": None})
    assert unit_state(scan, None) is UnitState.PARTIAL


def test_missingness_lists_every_absent_field() -> None:
    scan = (
        make_fleet(seed=1, n_units=1)
        .scans[0]
        .model_copy(update={"age_months": None, "power_on_hours": None})
    )
    assert unit_state(scan, None) is UnitState.PARTIAL


def _assessed_state(scan: ScanResult, passport: AssetPassport) -> UnitState:
    return unit_state(scan, passport)


def test_assessed_passport_without_review_is_ready() -> None:
    scan = make_fleet(seed=1, n_units=1).scans[0]
    estimate = make_estimate(asset_id=scan.asset_id, provenance=scan.provenance)
    decision = optimize_disposition(estimate, _ECONOMY, seed=0)
    assert decision.recommendation != "manual_review"
    passport = build_passport(scan, estimate, decision, run_id="run-state-ready")
    assert _assessed_state(scan, passport) is UnitState.READY


def test_ood_passport_routes_to_review() -> None:
    scan = make_fleet(seed=1, n_units=1).scans[0]
    passport = make_review_passport(scan)
    assert passport.disposition.recommendation == "manual_review"
    assert _assessed_state(scan, passport) is UnitState.REVIEW


def test_ready_passport_helper_builds_immutably() -> None:
    scan = make_fleet(seed=1, n_units=1).scans[0]
    passport = ready_passport(scan)
    assert passport.asset_id == scan.asset_id
    assert passport.provenance is scan.provenance
