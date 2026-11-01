"""Feature engineering: ScanResult + workload profile → model matrix (spec §4).

Unit features are drawn from ``ScanResult`` fields only — the engine's side of
the leakage firewall the simulator enforces on its side: ground-truth label
fields (latent defect, realized failure time, generator probabilities) have no
corresponding feature name, which the tests assert structurally.

Missing values are never silently imputed: every imputation sets the row's
missing flag, and the engine widens that unit's uncertainty interval in
response (spec §5 partial mode). Count-like features are log1p-transformed so
heavy-tailed event rates do not dominate a linear-hazard model.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Final

import numpy as np
from numpy.typing import NDArray

from rechip01.schemas import ScanResult
from rechip01.simulator.config import WorkloadProfile

# DCGM-class snapshot keys the engine consumes. The simulator duplicates some
# ScanResult fields into the snapshot (xid_events_cumulative, max_gpu_temp_c,
# sm_clock_max_mhz); the ScanResult fields are canonical, so those keys are
# skipped rather than double-counted.
ENGINE_BENCH_KEYS: Final[tuple[str, ...]] = (
    "hbm_ecc_error_rate_per_kh",
    "pcie_correctable_errors_per_h",
    "throttle_events_per_30d",
)

UNIT_FEATURE_NAMES: Final[tuple[str, ...]] = (
    "age_months",
    "log_power_on_hours",
    "log_xid_errors",
    "log_retired_pages",
    "log_ecc_rate_per_1000h",
    "thermal_mean",
    "thermal_max",
    "thermal_range",
    "max_sustained_power_pct",
    "log_hbm_ecc_error_rate_per_kh",
    "log_pcie_correctable_errors_per_h",
    "log_throttle_events_per_30d",
)

WORKLOAD_FEATURE_NAMES: Final[tuple[str, ...]] = (
    "workload_duty_cycle",
    "workload_thermal_offset",
    "workload_power_offset",
    "workload_log_ecc_multiplier",
)

FEATURE_NAMES: Final[tuple[str, ...]] = UNIT_FEATURE_NAMES + WORKLOAD_FEATURE_NAMES

# The workload-feature block sits at the end of the matrix so the engine can
# swap in a target workload's values per scoring pass.
WORKLOAD_SLICE: Final[slice] = slice(len(UNIT_FEATURE_NAMES), len(FEATURE_NAMES))

_HOURS_PER_KH: Final[float] = 1000.0
_MIN_HOURS_KH: Final[float] = 1e-6  # floor for the rate denominator on tiny hours

# Named column positions derived from UNIT_FEATURE_NAMES — no magic indices.
_COL_AGE: Final[int] = UNIT_FEATURE_NAMES.index("age_months")
_COL_LOG_HOURS: Final[int] = UNIT_FEATURE_NAMES.index("log_power_on_hours")
_COL_LOG_XID: Final[int] = UNIT_FEATURE_NAMES.index("log_xid_errors")
_COL_LOG_PAGES: Final[int] = UNIT_FEATURE_NAMES.index("log_retired_pages")
_COL_LOG_ECC_RATE: Final[int] = UNIT_FEATURE_NAMES.index("log_ecc_rate_per_1000h")
_COL_THERMAL_MEAN: Final[int] = UNIT_FEATURE_NAMES.index("thermal_mean")
_COL_THERMAL_MAX: Final[int] = UNIT_FEATURE_NAMES.index("thermal_max")
_COL_THERMAL_RANGE: Final[int] = UNIT_FEATURE_NAMES.index("thermal_range")
_COL_MAX_POWER: Final[int] = UNIT_FEATURE_NAMES.index("max_sustained_power_pct")
_COL_BENCH_FIRST: Final[int] = UNIT_FEATURE_NAMES.index("log_hbm_ecc_error_rate_per_kh")


def workload_feature_row(profile: WorkloadProfile) -> NDArray[np.float64]:
    """Return one workload's four feature values (pure, order-stable)."""
    return np.asarray(
        [
            profile.duty_cycle,
            profile.thermal_offset,
            profile.power_offset,
            math.log(profile.ecc_multiplier),
        ],
        dtype=np.float64,
    )


@dataclass(frozen=True)
class UnitFeatures:
    """Extracted unit-feature matrix plus per-row missingness flags."""

    matrix: NDArray[np.float64]  # (n_units, len(UNIT_FEATURE_NAMES))
    missing: NDArray[np.bool_]  # (n_units,) True when any feature was imputed


def _raw_unit_matrix(scans: list[ScanResult]) -> tuple[NDArray[np.float64], NDArray[np.bool_]]:
    """Extract unit features, NaN wherever the source value is absent.

    One extraction path for fit and transform: fit computes per-column
    ``nan``-medians, transform fills NaNs with them. The returned flag marks
    rows where any feature was missing (the partial-mode widening trigger).
    """
    n = len(scans)
    n_unit = len(UNIT_FEATURE_NAMES)
    matrix = np.full((n, n_unit), np.nan, dtype=np.float64)

    for row, scan in enumerate(scans):
        for offset, key in enumerate(ENGINE_BENCH_KEYS):
            value = scan.bench_mark_snapshot.get(key)
            if value is not None:
                matrix[row, _COL_BENCH_FIRST + offset] = np.log1p(value)

        history = scan.thermal_history
        if history:
            matrix[row, _COL_THERMAL_MEAN] = sum(history) / len(history)
            matrix[row, _COL_THERMAL_MAX] = max(history)
            matrix[row, _COL_THERMAL_RANGE] = max(history) - min(history)

        if scan.power_on_hours is not None:
            hours_kh = scan.power_on_hours / _HOURS_PER_KH
            matrix[row, _COL_LOG_HOURS] = np.log1p(scan.power_on_hours)
            matrix[row, _COL_LOG_ECC_RATE] = np.log1p(
                scan.ecc_events / max(hours_kh, _MIN_HOURS_KH)
            )

        if scan.age_months is not None:
            matrix[row, _COL_AGE] = scan.age_months

        # Always-present schema fields. (Raw ECC counts are deliberately not
        # a feature: ecc_events = rate x hours is algebraically derivable from
        # log_power_on_hours + log_ecc_rate_per_1000h, and that collinearity
        # makes the Weibull MLE unidentifiable — see TESTING.md.)
        matrix[row, _COL_LOG_XID] = np.log1p(scan.xid_errors)
        matrix[row, _COL_LOG_PAGES] = np.log1p(scan.retired_pages)
        matrix[row, _COL_MAX_POWER] = scan.max_sustained_power_pct

    missing = np.asarray(np.isnan(matrix).any(axis=1), dtype=np.bool_)
    return matrix, missing


@dataclass(frozen=True)
class FeatureSpace:
    """Fitted unit-feature extraction.

    ``fit`` computes imputation medians from the training scans only;
    ``transform`` applies them to any scans (train, eval, or real partner
    exports). Imputed positions set the row's missing flag, so downstream
    uncertainty widening is driven by exactly the imputed rows.
    """

    medians: dict[str, float]

    @classmethod
    def fit(cls, scans: list[ScanResult]) -> FeatureSpace:
        """Learn per-feature imputation medians from training scans."""
        if not scans:
            raise ValueError("cannot fit a feature space on an empty scan list")
        matrix, _ = _raw_unit_matrix(scans)
        medians: dict[str, float] = {}
        for column, name in enumerate(UNIT_FEATURE_NAMES):
            column_values = matrix[:, column]
            if np.all(np.isnan(column_values)):
                raise ValueError(
                    f"training scans never provide {name!r}; cannot learn an imputation median"
                )
            medians[name] = float(np.nanmedian(column_values))
        return cls(medians=medians)

    def transform(self, scans: list[ScanResult]) -> UnitFeatures:
        """Extract features for arbitrary scans using training medians."""
        matrix, missing = _raw_unit_matrix(scans)
        filled = matrix.copy()
        for column, name in enumerate(UNIT_FEATURE_NAMES):
            nan_positions = np.isnan(filled[:, column])
            filled[nan_positions, column] = self.medians[name]
        return UnitFeatures(matrix=filled, missing=missing)


@dataclass(frozen=True)
class Standardizer:
    """Mean/std standardization fitted on the training matrix."""

    means: NDArray[np.float64]
    stds: NDArray[np.float64]

    @classmethod
    def fit(cls, matrix: NDArray[np.float64]) -> Standardizer:
        """Fit means and stds; constant features get unit std (no division by zero)."""
        means = matrix.mean(axis=0)
        stds = matrix.std(axis=0)
        stds = np.where(stds < 1e-12, 1.0, stds)
        return cls(means=means, stds=stds)

    def transform(self, matrix: NDArray[np.float64]) -> NDArray[np.float64]:
        """Standardize a matrix with the fitted statistics."""
        return (matrix - self.means) / self.stds


__all__ = [
    "ENGINE_BENCH_KEYS",
    "FEATURE_NAMES",
    "FeatureSpace",
    "Standardizer",
    "UNIT_FEATURE_NAMES",
    "UnitFeatures",
    "WORKLOAD_FEATURE_NAMES",
    "WORKLOAD_SLICE",
    "workload_feature_row",
]
