"""Unit tests for the passport builder (spec art_vuIPvUhR, §4)."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import pytest

from rechip01.disposition import EconomyAssumptions, optimize_disposition
from rechip01.engine.engine import MODEL_VERSION
from rechip01.passport import build_passport, data_lineage_hash
from rechip01.scanner import scan_batch
from rechip01.schemas import (
    DispositionDecision,
    Provenance,
    ReliabilityEstimate,
    ScanResult,
)
from tests.test_disposition.helpers import make_economy
from tests.test_scanner.fixtures import valid_row


def make_scan(
    asset_id: str = "GPU-0001", provenance: Provenance = Provenance.SIMULATED
) -> ScanResult:
    """A schema-valid scan record via the scanner itself."""
    report = scan_batch([valid_row(asset_id=asset_id)], provenance)
    assert report.accepted_count == 1
    return report.accepted[0].result


def make_estimate(asset_id: str = "GPU-0001", **overrides: Any) -> ReliabilityEstimate:
    """A degenerate-CI estimate whose fitness keys match the default economy."""
    base: dict[str, Any] = dict(
        asset_id=asset_id,
        horizon_days=365,
        provenance=Provenance.SIMULATED,
        p_fail=0.10,
        ci=(0.10, 0.10),
        workload_fitness={"inference": 0.90},
        ood_flag=False,
        model_version=MODEL_VERSION,
    )
    base.update(overrides)
    return ReliabilityEstimate(**base)


def make_decision(
    estimate: ReliabilityEstimate, economy: EconomyAssumptions
) -> DispositionDecision:
    return optimize_disposition(estimate, economy, seed=0)


ISSUED_AT = datetime(2026, 11, 1, 12, 0, tzinfo=UTC)


def test_build_passport_shape_and_evidence() -> None:
    """The passport binds scan + estimate + decision with a full evidence trace."""
    scan = make_scan()
    estimate = make_estimate()
    decision = make_decision(estimate, make_economy())
    passport = build_passport(
        scan, estimate, decision, run_id="run-1", issued_at=ISSUED_AT, training_fingerprint="fp"
    )

    assert passport.asset_id == "GPU-0001"
    assert passport.run_id == "run-1"
    assert passport.provenance == Provenance.SIMULATED
    assert passport.issued_at == ISSUED_AT
    assert passport.model_version == estimate.model_version
    assert passport.reliability is estimate
    assert passport.disposition is decision

    assert passport.evidence_trace[0].startswith("lineage: sha256=")
    assert any("provenance=simulated" in line for line in passport.evidence_trace)
    assert any(f"model_version={MODEL_VERSION}" in line for line in passport.evidence_trace)
    assert any(line == "training: fp" for line in passport.evidence_trace)


def test_lineage_hash_deterministic_and_input_sensitive() -> None:
    """Equal inputs hash equal; any input change changes the hash."""
    scan, estimate = make_scan(), make_estimate()
    decision = make_decision(estimate, make_economy())
    first = data_lineage_hash(scan, estimate, decision)
    second = data_lineage_hash(scan, estimate, decision)
    assert first == second

    degraded = make_estimate(p_fail=0.42, ci=(0.42, 0.42))
    degraded_decision = make_decision(degraded, make_economy())
    assert data_lineage_hash(scan, degraded, degraded_decision) != first


def test_build_passport_rejects_mixed_assets() -> None:
    """Inputs referencing different assets never issue a passport."""
    scan = make_scan(asset_id="GPU-A")
    estimate = make_estimate(asset_id="GPU-B")
    decision = make_decision(estimate, make_economy())
    with pytest.raises(ValueError, match="asset_id"):
        build_passport(scan, estimate, decision, run_id="run-1")


def test_build_passport_rejects_version_mismatch() -> None:
    """Estimate and decision must agree on model version."""
    scan = make_scan()
    estimate = make_estimate()
    decision = make_decision(estimate, make_economy())
    tampered = decision.model_copy(update={"model_version": "other-v9"})
    with pytest.raises(ValueError, match="model_version"):
        build_passport(scan, estimate, tampered, run_id="run-1")


def test_passport_provenance_follows_scan() -> None:
    """A measured scan produces a measured passport — never relabeled."""
    scan = make_scan(provenance=Provenance.MEASURED)
    estimate = make_estimate(provenance=Provenance.MEASURED)
    decision = make_decision(estimate, make_economy())
    passport = build_passport(scan, estimate, decision, run_id="run-1", issued_at=ISSUED_AT)
    assert passport.provenance == Provenance.MEASURED
