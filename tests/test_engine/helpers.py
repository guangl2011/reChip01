"""Shared builders for engine tests: scan variants and synthetic models.

Pure builders only — fixtures live in conftest.py, tests in test modules.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from rechip01.engine.features import ENGINE_BENCH_KEYS
from rechip01.engine.model import WeibullPHModel
from rechip01.schemas import Provenance, ScanResult
from rechip01.simulator import SimulatedFleet, default_fleet_config, generate_fleet

TRAIN_SEED = 101
EVAL_SEED = 202
TRAIN_N = 600
EVAL_N = 2000


def make_fleet(seed: int, n_units: int) -> SimulatedFleet:
    return generate_fleet(default_fleet_config(seed=seed, n_units=n_units))


def with_missing_fields(scan: ScanResult, absent_benchmarks: int = 3) -> ScanResult:
    """Degraded copy missing up to three engine-consumed snapshot keys (partial mode)."""
    data = scan.model_dump()
    remaining = absent_benchmarks
    for key in ENGINE_BENCH_KEYS:
        if remaining <= 0:
            break
        data["bench_mark_snapshot"].pop(key, None)
        remaining -= 1
    return ScanResult.model_validate(data)


def with_measured_provenance(scan: ScanResult) -> ScanResult:
    """Relabel a scan as MEASURED (a partner export would look like this)."""
    return scan.model_copy(update={"provenance": Provenance.MEASURED})


def degraded_twin(scan: ScanResult, ecc_scale: float = 60.0) -> ScanResult:
    """Copy pushed far outside training support (unit-level OOD probe).

    Inflates every event-count feature the model consumes (thermal_history is
    UnitInterval-bounded, so counts and rates carry the degradation signal).
    """
    data = scan.model_dump()
    scale = int(ecc_scale)
    data["ecc_events"] = data["ecc_events"] * scale + scale
    data["xid_errors"] = data["xid_errors"] * scale + scale
    data["retired_pages"] = data["retired_pages"] * scale + scale
    data["max_sustained_power_pct"] = 1.0
    data["thermal_history"] = [1.0] * len(data["thermal_history"])
    data["bench_mark_snapshot"]["hbm_ecc_error_rate_per_kh"] = (
        data["bench_mark_snapshot"].get("hbm_ecc_error_rate_per_kh", 1.0) * ecc_scale
    )
    data["bench_mark_snapshot"]["throttle_events_per_30d"] = (
        data["bench_mark_snapshot"].get("throttle_events_per_30d", 1.0) * ecc_scale
    )
    return ScanResult.model_validate(data)


def synthetic_model(
    coef: NDArray[np.float64], log_shape: float, variance: float = 0.05
) -> WeibullPHModel:
    """A WeibullPHModel built directly from known parameters (no fit).

    Covariance is a scaled identity — finite, positive-definite, good enough
    for interval-arithmetic properties.
    """
    covariance = np.eye(coef.shape[0] + 1, dtype=np.float64) * variance
    return WeibullPHModel(
        coef=np.asarray(coef, dtype=float), log_shape=log_shape, covariance=covariance
    )


__all__ = [
    "EVAL_N",
    "EVAL_SEED",
    "TRAIN_N",
    "TRAIN_SEED",
    "degraded_twin",
    "make_fleet",
    "synthetic_model",
    "with_measured_provenance",
    "with_missing_fields",
]
