"""Passport builders for dashboard tests.

``ready_passport`` produces a normally-recommended passport;
``make_review_passport`` produces one the optimizer routes to manual review
(OOD flag set — the deterministic review trigger in the disposition config).
"""

from __future__ import annotations

from tests.test_disposition.helpers import make_economy, make_estimate

from rechip01.disposition.optimizer import optimize_disposition
from rechip01.passport.builder import build_passport
from rechip01.schemas import AssetPassport, ScanResult


def ready_passport(scan: ScanResult, run_id: str = "run-ready") -> AssetPassport:
    """An in-distribution passport with a calibrated (non-review) decision."""
    estimate = make_estimate(asset_id=scan.asset_id, provenance=scan.provenance)
    decision = optimize_disposition(estimate, make_economy(), seed=0)
    return build_passport(scan, estimate, decision, run_id=run_id)


def make_review_passport(scan: ScanResult, run_id: str = "run-review") -> AssetPassport:
    """An OOD passport — the optimizer routes it to manual review by design."""
    estimate = make_estimate(
        asset_id=scan.asset_id,
        provenance=scan.provenance,
        ood_flag=True,
        ci=(0.25, 0.35),
        p_fail=0.30,
    )
    decision = optimize_disposition(estimate, make_economy(), seed=0)
    assert decision.recommendation == "manual_review", (
        f"review fixture relies on the optimizer's OOD routing; got {decision.recommendation!r}"
    )
    return build_passport(scan, estimate, decision, run_id=run_id)


__all__ = ["make_review_passport", "ready_passport"]
