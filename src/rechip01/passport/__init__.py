"""Passport: auditable asset passports and the API layer (spec art_vuIPvUhR, §4–5).

Public API:

- :func:`build_passport` — assemble one immutable passport from a scan, its
  reliability estimate, and its disposition decision, with an evidence trace
  and data-lineage hash.
- :func:`data_lineage_hash` — the SHA-256 over exactly those three inputs.
- :func:`create_app` / :func:`default_deps` — the FastAPI service exposing
  scan ingest, batch overview, assessment, and passport retrieval over the
  file-backed repositories.
- API contracts: ``ScanIngestRequest``, ``AssessRequest``, ``AssessResponse``,
  ``AssessmentSummary``, ``BatchOverview``, ``UnitOverview``.

Every passport and every API response carries provenance and — wherever an
assessment is reported — calibrated uncertainty. Simulated and measured data
are structurally distinct end to end. The service is decision support with
explicit uncertainty; it never issues accredited third-party certification.
"""

from rechip01.passport.app import ServiceDeps, create_app, default_deps
from rechip01.passport.builder import build_passport, data_lineage_hash
from rechip01.passport.models import (
    AssessmentSummary,
    AssessRequest,
    AssessResponse,
    BatchOverview,
    ScanIngestRequest,
    UnitOverview,
)

__all__ = [
    "AssessRequest",
    "AssessResponse",
    "AssessmentSummary",
    "BatchOverview",
    "ScanIngestRequest",
    "ServiceDeps",
    "UnitOverview",
    "build_passport",
    "create_app",
    "data_lineage_hash",
    "default_deps",
]
