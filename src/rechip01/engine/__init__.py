"""Reliability engine: workload-conditioned, uncertainty-aware assessment
(spec art_vuIPvUhR, §4–5).

Public API:

- ``ReliabilityEngine`` — fit on a simulated (or, later, measured) fleet;
  assess scans for a horizon under declared target workloads. Emits
  ``ReliabilityEstimate`` with provenance, calibrated intervals, workload
  fitness, and OOD flags that widen uncertainty instead of hiding it.
- ``EngineConfig`` — calibrated-rails: CI level, OOD fence multiplier,
  interval widening factors, fit bounds.
- ``run_harness`` / ``render_pipeline_report`` — the reproducible
  train→assess→evaluate pipeline check against simulator ground truth
  (AUROC + bootstrap CI, Mann-Whitney p, ECE, MAE, interval coverage).
- ``MODEL_VERSION`` — stamped on every estimate.

All metrics are simulation-based pipeline checks; the engine makes no
real-world reliability claim (spec §8).
"""

from rechip01.engine.config import EngineConfig
from rechip01.engine.engine import ENGINE_FEATURE_NAMES, MODEL_VERSION, ReliabilityEngine
from rechip01.engine.features import FEATURE_NAMES, FeatureSpace, Standardizer
from rechip01.engine.harness import HarnessResult, run_harness
from rechip01.engine.metrics import (
    CalibrationBin,
    EngineEvaluation,
    bootstrap_auroc_ci,
    evaluate,
    expected_calibration_error,
    render_pipeline_report,
)
from rechip01.engine.model import WeibullPHModel
from rechip01.engine.ood import OODDetector, OODVerdict

__all__ = [
    "CalibrationBin",
    "ENGINE_FEATURE_NAMES",
    "EngineConfig",
    "EngineEvaluation",
    "FEATURE_NAMES",
    "FeatureSpace",
    "HarnessResult",
    "MODEL_VERSION",
    "OODDetector",
    "OODVerdict",
    "ReliabilityEngine",
    "Standardizer",
    "WeibullPHModel",
    "bootstrap_auroc_ci",
    "evaluate",
    "expected_calibration_error",
    "render_pipeline_report",
    "run_harness",
]
