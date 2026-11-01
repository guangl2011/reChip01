"""Degradation physics for the fleet simulator (spec art_vuIPvUhR, §4).

Pure functions and named constants only: every draw function takes a numpy
``Generator`` explicitly and every hazard function takes its inputs as arrays,
so the whole module is side-effect free and unit-testable in isolation.

Model of a retired unit
-----------------------

Two layers of state:

* **Latent covariates** — true age, first-life duty cycle, operating
  temperature, and the injected latent-defect flag. These drive the failure
  hazard directly.
* **Observable evidence** — ECC/XID event counts, retired pages, thermal
  history, sustained power, and DCGM-class snapshot readings. These are noisy
  observations *generated from* the latents; they are what the scanner sees and
  what the reliability engine must learn to decode.

Failure times follow Weibull proportional hazards: the log-hazard is a linear
function of the latent covariates (plus a workload shift for second-life
operation), and the baseline Weibull scale shrinks by ``exp(-log_hazard)``.
Latent defects multiply the hazard — the early-failure signature the product
exists to price — and stress workloads shift it further.

Coefficient values are pipeline constants, not fitted parameters: they are
tuned only against the simulator's held-out ground truth, never against
training performance (TESTING.md calibration rule).
"""

from __future__ import annotations

import math
from typing import Final

import numpy as np
from numpy.typing import NDArray

from rechip01.simulator.config import Archetype, WorkloadProfile

# Reference operating point for normalized temperature: the archetypes center
# near it, so exponent sign conventions read "hotter than reference = riskier".
T_REF: Final[float] = 0.5

# Arrhenius-lite acceleration: event rates scale by exp(coef * (T - T_REF)).
# A +0.1 normalized-temperature rise multiplies event rates by ~1.5.
THERMAL_RATE_COEF: Final[float] = 4.0

# Log-hazard coefficients (per unit of the standardized covariate).
HAZARD_THERMAL_COEF: Final[float] = 2.0
HAZARD_DEFECT_COEF: Final[float] = 1.2
HAZARD_AGE_COEF: Final[float] = 0.12  # per year of age (age_months / 12)
HAZARD_DUTY_COEF: Final[float] = 0.5  # per unit duty cycle above DUTY_REF
HAZARD_POWER_COEF: Final[float] = 0.8  # per unit normalized power offset
HAZARD_ECC_STRESS_COEF: Final[float] = 0.3  # per log workload ecc_multiplier

# Latent defects make themselves observable: hotter spots, more correctable
# and uncorrectable errors. These boosts are the signal the engine must learn.
DEFECT_THERMAL_BOOST: Final[float] = 0.04  # normalized-temperature uplift
DEFECT_ECC_BOOST: Final[float] = 0.6  # relative ECC-rate uplift
DEFECT_XID_BOOST: Final[float] = 1.0  # relative XID-rate uplift

# Thermal aging: units drift hotter the longer they ran in first life.
AGE_THERMAL_DRIFT: Final[float] = 0.003  # per month of age

# A retired page follows from an uncorrectable event this often.
PAGE_RETIRE_PROB: Final[float] = 0.01

# First-life reference duty cycle: deviations from it enter the hazard.
DUTY_REF: Final[float] = 0.6

# Normalized-temperature percentiles recorded per unit: the ScanResult
# thermal_history grid, fixed for every unit so downstream feature
# engineering sees a consistent shape.
THERMAL_PERCENTILES: Final[tuple[float, ...]] = (0.10, 0.25, 0.50, 0.75, 0.90, 0.99)


def thermal_acceleration(op_temp: NDArray[np.float64]) -> NDArray[np.float64]:
    """Return the multiplicative thermal acceleration of event rates."""
    return np.exp(THERMAL_RATE_COEF * (op_temp - T_REF))


def workload_log_hazard_shift(profile: WorkloadProfile) -> float:
    """Return the log-hazard shift of operating ``profile`` vs reference use.

    Derived from the profile's physical offsets — thermal and power stress
    enter linearly, memory-subsystem stress as the log of the ECC multiplier —
    so a profile cannot declare a risk inconsistent with its own physics.
    """
    return (
        HAZARD_THERMAL_COEF * profile.thermal_offset
        + HAZARD_POWER_COEF * profile.power_offset
        + HAZARD_ECC_STRESS_COEF * math.log(profile.ecc_multiplier)
    )


def latent_log_hazard(
    has_defect: NDArray[np.bool_],
    op_temp: NDArray[np.float64],
    age_months: NDArray[np.float64],
    duty_cycle: NDArray[np.float64],
    failure_rate_multiplier: float,
) -> NDArray[np.float64]:
    """Return per-unit log hazard relative to reference covariates.

    Latent covariates only — realized event counts are observations of these
    latents, not causes, so ground-truth failure times stay independent of the
    realized counts given the latents (the no-leakage structure the engine
    relies on).
    """
    age_years = age_months / 12.0
    return (
        HAZARD_DEFECT_COEF * has_defect.astype(np.float64)
        + HAZARD_THERMAL_COEF * (op_temp - T_REF)
        + HAZARD_AGE_COEF * age_years
        + HAZARD_DUTY_COEF * (duty_cycle - DUTY_REF)
        + math.log(failure_rate_multiplier)
    )


def weibull_scale_days(
    archetype: Archetype, log_hazard: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Return per-unit Weibull scales: baseline shrinks as hazard grows."""
    return archetype.baseline_scale_days * np.exp(-log_hazard)


def survival_probability(
    horizon_days: int, weibull_shape: float, scale_days: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Return P(fail within horizon) under a Weibull(weibull_shape, scale)."""
    exponent = np.power(np.float64(horizon_days) / scale_days, weibull_shape)
    return np.asarray(1.0 - np.exp(-exponent), dtype=np.float64)


def draw_failure_times(
    rng: np.random.Generator, weibull_shape: float, scale_days: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Draw per-unit failure times (days from assessment) under Weibull(shape, scale)."""
    return scale_days * rng.weibull(weibull_shape, size=scale_days.shape)


__all__ = [
    "AGE_THERMAL_DRIFT",
    "DEFECT_ECC_BOOST",
    "DEFECT_THERMAL_BOOST",
    "DEFECT_XID_BOOST",
    "DUTY_REF",
    "HAZARD_AGE_COEF",
    "HAZARD_DEFECT_COEF",
    "HAZARD_DUTY_COEF",
    "HAZARD_ECC_STRESS_COEF",
    "HAZARD_POWER_COEF",
    "HAZARD_THERMAL_COEF",
    "PAGE_RETIRE_PROB",
    "THERMAL_PERCENTILES",
    "THERMAL_RATE_COEF",
    "T_REF",
    "draw_failure_times",
    "latent_log_hazard",
    "survival_probability",
    "thermal_acceleration",
    "weibull_scale_days",
    "workload_log_hazard_shift",
]
