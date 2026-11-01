"""reChip01 — simulation-first reliability assessment for retired AI servers and GPUs.

Modules (spec art_vuIPvUhR, section 4):

- ``schemas``    — shared schema contracts: ScanResult, ReliabilityEstimate,
  DispositionDecision, AssetPassport, Provenance
- ``repository`` — passport persistence behind a repository interface
- ``simulator``  — physics-grounded synthetic fleet generation
- ``scanner``    — inventory/telemetry ingest, validation, provenance stamping
- ``engine``     — uncertainty-aware, workload-conditioned reliability model
- ``disposition``— risk-adjusted net-recovery disposition optimizer
- ``passport``   — auditable asset passport + API layer

``schemas`` and ``repository`` hold the shared contracts (spec §4); the other
modules are scaffolding placeholders that gain their interfaces in dedicated
PRs.
"""

from rechip01.repository import (
    DuplicatePassportError,
    JsonPassportRepository,
    PassportRepository,
)
from rechip01.schemas import (
    AssetPassport,
    DispositionDecision,
    OptionEvaluation,
    Provenance,
    ReliabilityEstimate,
    ScanResult,
)

__version__ = "0.1.0"

__all__ = [
    "AssetPassport",
    "DispositionDecision",
    "DuplicatePassportError",
    "JsonPassportRepository",
    "OptionEvaluation",
    "PassportRepository",
    "Provenance",
    "ReliabilityEstimate",
    "ScanResult",
    "__version__",
]
