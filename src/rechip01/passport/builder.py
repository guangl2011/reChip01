"""Passport builder: assembles immutable asset passports (spec art_vuIPvUhR, §4).

A passport is issued once per (asset_id, run_id) and never edited: it binds
one scan record, its reliability estimate, and its disposition decision into
a single frozen record with an explicit evidence trace and a data-lineage
hash over exactly those three inputs. Any change to any input yields a
different hash, so the trace is verifiable, not decorative.

Honesty invariants carried here: passport provenance is the scan's
provenance — simulated and measured passports are structurally distinct —
and the evidence trace cites model and valuation versions so every number on
a passport is traceable to the code that produced it.
"""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from typing import Final

from rechip01.disposition.optimizer import DISPOSITION_VALUATION_VERSION
from rechip01.engine.engine import MODEL_VERSION
from rechip01.schemas import AssetPassport, DispositionDecision, ReliabilityEstimate, ScanResult

_LINEAGE_PREFIX: Final[str] = "lineage: sha256="


def data_lineage_hash(
    scan: ScanResult, estimate: ReliabilityEstimate, decision: DispositionDecision
) -> str:
    """Return the SHA-256 lineage hash over the passport's three inputs.

    Inputs are serialized to their canonical JSON forms (sorted keys) so the
    hash is stable across processes and pydantic versions.
    """
    payload = json.dumps(
        {
            "scan": scan.model_dump_json(),
            "estimate": estimate.model_dump_json(),
            "disposition": decision.model_dump_json(),
        },
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def build_passport(
    scan: ScanResult,
    estimate: ReliabilityEstimate,
    decision: DispositionDecision,
    *,
    run_id: str,
    issued_at: datetime | None = None,
    training_fingerprint: str | None = None,
) -> AssetPassport:
    """Assemble one immutable passport from its three assessed inputs.

    All three inputs must refer to the same asset and the estimate/decision
    pair must agree on model version — a passport never mixes records across
    assets or model versions. ``issued_at`` defaults to now (UTC); pass an
    explicit timestamp for reproducible runs.
    """
    if not (scan.asset_id == estimate.asset_id == decision.asset_id):
        raise ValueError(
            "passport inputs must share one asset_id: "
            f"scan={scan.asset_id!r}, estimate={estimate.asset_id!r}, "
            f"decision={decision.asset_id!r}"
        )
    if estimate.model_version != decision.model_version:
        raise ValueError(
            "estimate and decision disagree on model_version: "
            f"{estimate.model_version!r} != {decision.model_version!r}"
        )

    digest = data_lineage_hash(scan, estimate, decision)
    evidence_trace = [
        f"{_LINEAGE_PREFIX}{digest}",
        (
            f"scan: asset_id={scan.asset_id} model={scan.model} "
            f"provenance={scan.provenance.value} "
            f"custody_events={len(scan.chain_of_custody)}"
        ),
        (
            f"reliability: model_version={MODEL_VERSION} p_fail={estimate.p_fail:.4f} "
            f"ci=[{estimate.ci[0]:.4f}, {estimate.ci[1]:.4f}] ood_flag={estimate.ood_flag}"
        ),
        (
            f"disposition: valuation_version={DISPOSITION_VALUATION_VERSION} "
            f"recommendation={decision.recommendation}"
        ),
    ]
    if training_fingerprint is not None:
        evidence_trace.append(f"training: {training_fingerprint}")

    return AssetPassport(
        asset_id=scan.asset_id,
        run_id=run_id,
        provenance=scan.provenance,
        issued_at=issued_at if issued_at is not None else datetime.now(UTC),
        reliability=estimate,
        disposition=decision,
        evidence_trace=evidence_trace,
        model_version=MODEL_VERSION,
    )


__all__ = ["data_lineage_hash", "build_passport"]
