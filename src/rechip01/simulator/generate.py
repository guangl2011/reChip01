"""Fleet generation: seeded synthetic batches with held-out labels (spec §4).

``generate_fleet`` is the only entry point a caller needs: one ``FleetConfig``
in, one ``SimulatedFleet`` out — a ``ScanResult`` stream the scanner could have
produced (``provenance=SIMULATED`` on every record) plus ground-truth labels
the engine never sees at inference time.

Generation order is fixed and the seed is the sole entropy source, so equal
configs produce bit-identical batches; the seed and the headline outcome
counts are logged on every run.

Label integrity is structural, not aspirational: the hazard is driven by
latent covariates only, observation draws never see the realized failure time,
and ``SimulatedFleet`` rejects batches whose scans and labels disagree.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Final

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field, model_validator
from scipy.special import ndtri

from rechip01.schemas import Provenance, ScanResult, UnitInterval
from rechip01.simulator.config import WORKLOAD_PROFILES, FleetConfig
from rechip01.simulator.physics import (
    AGE_THERMAL_DRIFT,
    DEFECT_ECC_BOOST,
    DEFECT_THERMAL_BOOST,
    DEFECT_XID_BOOST,
    DUTY_REF,
    PAGE_RETIRE_PROB,
    THERMAL_PERCENTILES,
    draw_failure_times,
    latent_log_hazard,
    survival_probability,
    thermal_acceleration,
    workload_log_hazard_shift,
)

_LOGGER = logging.getLogger(__name__)

# DCGM-class diagnostic readings every simulated unit reports. The allowlist
# is the leakage firewall: ground-truth label names can never enter the
# feature set because snapshot features are keyed by this tuple alone.
BENCH_MARK_KEYS: Final[tuple[str, ...]] = (
    "hbm_ecc_error_rate_per_kh",
    "pcie_correctable_errors_per_h",
    "xid_events_cumulative",
    "max_gpu_temp_c",
    "sm_clock_max_mhz",
    "throttle_events_per_30d",
)

# Snapshot temperatures are physical Celsius readings mapped from the
# normalized scale: T=0 -> ambient, T=1 -> ambient + span.
AMBIENT_TEMP_C: Final[float] = 25.0
TEMP_C_SPAN: Final[float] = 70.0

# Chain-of-custody markers for simulated units: the passport layer requires a
# custody trace, so simulated intake carries synthetic, clearly-labeled events.
CUSTODY_EVENTS: Final[tuple[str, ...]] = (
    "SIM:manufactured",
    "SIM:first-life deployment",
    "SIM:decommissioned",
    "SIM:rechip01 intake",
)

# Normalized-temperature z-scores for the recorded percentile grid, computed
# once: history rows are op_temp + spread * z, so recorded percentiles are
# genuinely monotone in percentile, like real telemetry.
_TEMPERATURE_Z: Final[NDArray[np.float64]] = ndtri(np.asarray(THERMAL_PERCENTILES))

# Bounds that keep normalized draws strictly inside the schema's unit interval.
_TEMP_LO, _TEMP_HI = 0.001, 0.999
_POWER_LO, _POWER_HI = 0.05, 0.99

_HOURS_PER_MONTH: Final[float] = 730.0  # 30.42 days at 24 h


class GroundTruthLabel(BaseModel):
    """Held-out simulator truth for one unit (spec §4).

    Joined to a ``ScanResult`` only by ``asset_id`` at evaluation time. The
    latent-defect flag and the failure event/time are labels by construction:
    no field of ``ScanResult`` carries them (enforced by the tests and by the
    ``BENCH_MARK_KEYS`` allowlist).
    """

    model_config = ConfigDict(frozen=True)

    asset_id: str = Field(min_length=1)
    provenance: Provenance
    assigned_workload: str = Field(min_length=1)
    horizon_days: int = Field(ge=1)
    has_latent_defect: bool
    failed_within_horizon: bool
    time_to_failure_days: float | None = Field(
        default=None, description="Days from assessment to failure; None when censored"
    )
    p_fail_assigned_workload: UnitInterval
    p_fail_by_workload: dict[str, UnitInterval] = Field(
        description="Generator-truth failure probability per mix workload"
    )


class SimulatedFleet(BaseModel):
    """One generated batch: ScanResult stream plus aligned ground truth."""

    model_config = ConfigDict(frozen=True)

    config: FleetConfig
    scans: list[ScanResult]
    labels: list[GroundTruthLabel]

    @model_validator(mode="after")
    def _scans_and_labels_align(self) -> SimulatedFleet:
        if len(self.scans) != len(self.labels):
            raise ValueError(
                f"scans/labels length mismatch: {len(self.scans)} vs {len(self.labels)}"
            )
        scan_ids = [scan.asset_id for scan in self.scans]
        label_ids = [label.asset_id for label in self.labels]
        if scan_ids != label_ids:
            raise ValueError("scans and labels must align by asset_id and order")
        mismatched = [
            scan.asset_id
            for scan, label in zip(self.scans, self.labels, strict=True)
            if scan.provenance != Provenance.SIMULATED or label.provenance != Provenance.SIMULATED
        ]
        if mismatched:
            raise ValueError(
                f"all simulated-fleet records must be provenance=SIMULATED: {mismatched[:5]}"
            )
        return self

    @property
    def n_units(self) -> int:
        """Number of units in the batch."""
        return len(self.scans)

    def label_by_asset(self) -> dict[str, GroundTruthLabel]:
        """Return the label lookup keyed by asset_id (evaluation joins)."""
        return {label.asset_id: label for label in self.labels}


@dataclass(frozen=True)
class _Latents:
    """Per-unit latent state: the truth observations are noisy views of."""

    archetype_index: NDArray[np.int64]
    workload_index: NDArray[np.int64]
    age_months: NDArray[np.float64]
    duty_cycle: NDArray[np.float64]
    power_on_hours: NDArray[np.float64]
    op_temp: NDArray[np.float64]
    thermal_spread: NDArray[np.float64]
    has_defect: NDArray[np.bool_]


@dataclass(frozen=True)
class _Observations:
    """Per-unit observable evidence drawn from the latent state."""

    ecc_events: NDArray[np.int64]
    xid_errors: NDArray[np.int64]
    retired_pages: NDArray[np.int64]
    thermal_history: NDArray[np.float64]
    max_sustained_power: NDArray[np.float64]
    bench: dict[str, NDArray[np.float64]]


def _draw_latents(rng: np.random.Generator, config: FleetConfig) -> _Latents:
    """Draw per-unit latent covariates, archetype group by archetype group."""
    archetype_shares = np.asarray([a.fleet_share for a in config.archetypes], dtype=np.float64)
    archetype_index = rng.choice(len(config.archetypes), size=config.n_units, p=archetype_shares)

    n = config.n_units
    age_months = np.zeros(n)
    duty_cycle = np.zeros(n)
    op_temp = np.zeros(n)
    thermal_spread = np.zeros(n)
    has_defect = np.zeros(n, dtype=bool)

    for group_index, archetype in enumerate(config.archetypes):
        members = np.flatnonzero(archetype_index == group_index)
        if members.size == 0:
            continue
        n_members = members.size
        defect_rate = min(1.0, archetype.latent_defect_rate * config.latent_defect_rate_multiplier)
        age = rng.uniform(archetype.age_months_min, archetype.age_months_max, n_members)
        duty = np.clip(rng.normal(archetype.duty_cycle_mean, 0.12, n_members), 0.15, 0.98)
        defect = rng.random(n_members) < defect_rate
        aged_temp = archetype.thermal_mean + AGE_THERMAL_DRIFT * age
        defect_temp = DEFECT_THERMAL_BOOST * defect
        jitter = rng.normal(0.0, 0.03, n_members)
        spread = np.clip(rng.normal(0.07, 0.02, n_members), 0.02, 0.15)

        age_months[members] = age
        duty_cycle[members] = duty
        has_defect[members] = defect
        op_temp[members] = np.clip(aged_temp + defect_temp + jitter, 0.02, 0.98)
        thermal_spread[members] = spread

    workload_shares = np.asarray([w.share for w in config.workload_mix], dtype=np.float64)
    workload_index = rng.choice(len(config.workload_mix), size=n, p=workload_shares)

    return _Latents(
        archetype_index=archetype_index,
        workload_index=workload_index,
        age_months=age_months,
        duty_cycle=duty_cycle,
        power_on_hours=age_months * _HOURS_PER_MONTH * duty_cycle,
        op_temp=op_temp,
        thermal_spread=thermal_spread,
        has_defect=has_defect,
    )


def _draw_observations(
    rng: np.random.Generator, config: FleetConfig, latents: _Latents
) -> _Observations:
    """Draw observable evidence: event processes, thermal history, snapshot."""
    n = config.n_units
    ecc_events = np.zeros(n, dtype=np.int64)
    xid_errors = np.zeros(n, dtype=np.int64)
    retired_pages = np.zeros(n, dtype=np.int64)
    thermal_history = np.zeros((n, len(THERMAL_PERCENTILES)))
    max_power = np.zeros(n)
    bench: dict[str, NDArray[np.float64]] = {key: np.zeros(n) for key in BENCH_MARK_KEYS}

    for group_index, archetype in enumerate(config.archetypes):
        members = np.flatnonzero(latents.archetype_index == group_index)
        if members.size == 0:
            continue
        n_members = members.size
        hours_kh = latents.power_on_hours[members] / 1000.0
        accel = thermal_acceleration(latents.op_temp[members])
        defect = latents.has_defect[members].astype(np.float64)

        ecc_lam = (
            archetype.ecc_rate_per_1000h
            * config.ecc_rate_multiplier
            * accel
            * (1.0 + DEFECT_ECC_BOOST * defect)
            * hours_kh
        )
        xid_lam = (
            archetype.xid_rate_per_1000h
            * config.xid_rate_multiplier
            * accel
            * (1.0 + DEFECT_XID_BOOST * defect)
            * hours_kh
        )
        ecc = rng.poisson(ecc_lam)
        xid = rng.poisson(xid_lam)
        retired = rng.binomial(ecc, PAGE_RETIRE_PROB)

        power_jitter = rng.normal(0.0, 0.03, n_members)
        duty_stress = 0.2 * (latents.duty_cycle[members] - DUTY_REF)
        pcie_noise = np.clip(rng.normal(0.4, 0.1, n_members), 0.0, None)
        clock_noise = rng.normal(0.0, 5.0, n_members)
        clock_by_temp = archetype.base_clock_mhz * (1.0 - 0.08 * latents.op_temp[members])
        throttle_lam = np.clip(60.0 * (latents.op_temp[members] - 0.3) + 20.0 * defect, 0.0, None)

        ecc_events[members] = ecc
        xid_errors[members] = xid
        retired_pages[members] = retired
        thermal_history[members] = np.clip(
            latents.op_temp[members, None] + latents.thermal_spread[members, None] * _TEMPERATURE_Z,
            _TEMP_LO,
            _TEMP_HI,
        )
        max_power[members] = np.clip(
            archetype.power_mean + duty_stress + 0.03 * defect + power_jitter,
            _POWER_LO,
            _POWER_HI,
        )
        bench["hbm_ecc_error_rate_per_kh"][members] = (
            ecc / hours_kh + 30.0 * defect + np.abs(rng.normal(0.0, 2.0, n_members))
        )
        bench["pcie_correctable_errors_per_h"][members] = 0.4 * accel + 0.3 * defect + pcie_noise
        bench["xid_events_cumulative"][members] = xid
        bench["max_gpu_temp_c"][members] = AMBIENT_TEMP_C + TEMP_C_SPAN * thermal_history[
            members
        ].max(axis=1)
        bench["sm_clock_max_mhz"][members] = clock_by_temp + clock_noise
        bench["throttle_events_per_30d"][members] = rng.poisson(throttle_lam)

    return _Observations(
        ecc_events=ecc_events,
        xid_errors=xid_errors,
        retired_pages=retired_pages,
        thermal_history=thermal_history,
        max_sustained_power=max_power,
        bench=bench,
    )


def _build_scans(config: FleetConfig, latents: _Latents, obs: _Observations) -> list[ScanResult]:
    """Assemble the ScanResult stream: schema-shaped, provenance-stamped."""
    scans: list[ScanResult] = []
    for unit in range(config.n_units):
        archetype = config.archetypes[latents.archetype_index[unit]]
        scans.append(
            ScanResult(
                asset_id=f"SIM-{config.seed}-{unit:06d}",
                model=archetype.model_name,
                provenance=Provenance.SIMULATED,
                age_months=float(latents.age_months[unit]),
                power_on_hours=float(latents.power_on_hours[unit]),
                ecc_events=int(obs.ecc_events[unit]),
                xid_errors=int(obs.xid_errors[unit]),
                retired_pages=int(obs.retired_pages[unit]),
                thermal_history=[float(v) for v in obs.thermal_history[unit]],
                max_sustained_power_pct=float(obs.max_sustained_power[unit]),
                bench_mark_snapshot={key: float(values[unit]) for key, values in obs.bench.items()},
                chain_of_custody=list(CUSTODY_EVENTS),
            )
        )
    return scans


def _build_labels(
    rng: np.random.Generator, config: FleetConfig, latents: _Latents
) -> list[GroundTruthLabel]:
    """Draw failure times and assemble held-out ground-truth labels.

    The hazard uses latent covariates only. Workload shifts are applied for
    every mix profile, so each label carries the generator's true
    workload-conditioned failure probability — the calibration reference for
    the reliability engine (spec §4, H1–H3 harness).
    """
    n = config.n_units
    horizon = config.horizon_days
    profiles = [WORKLOAD_PROFILES[share.name] for share in config.workload_mix]
    shifts = np.asarray([workload_log_hazard_shift(profile) for profile in profiles])

    base_hazard = latent_log_hazard(
        latents.has_defect,
        latents.op_temp,
        latents.age_months,
        latents.duty_cycle,
        config.failure_rate_multiplier,
    )
    p_by_workload = np.zeros((n, len(profiles)))
    failure_times = np.zeros(n)

    for group_index, archetype in enumerate(config.archetypes):
        members = np.flatnonzero(latents.archetype_index == group_index)
        if members.size == 0:
            continue
        workload_scales = archetype.baseline_scale_days * np.exp(
            -(base_hazard[members, None] + shifts[None, :])
        )
        p_by_workload[members] = survival_probability(
            horizon, archetype.weibull_shape, workload_scales
        )
        # The realized event is drawn under the *assigned* workload's hazard so
        # `failed_within_horizon` is consistent with `p_fail_assigned_workload`.
        assigned_scales = workload_scales[np.arange(members.size), latents.workload_index[members]]
        failure_times[members] = draw_failure_times(rng, archetype.weibull_shape, assigned_scales)

    assigned_p = p_by_workload[np.arange(n), latents.workload_index]
    failed = failure_times <= horizon

    labels: list[GroundTruthLabel] = []
    for unit in range(n):
        labels.append(
            GroundTruthLabel(
                asset_id=f"SIM-{config.seed}-{unit:06d}",
                provenance=Provenance.SIMULATED,
                assigned_workload=config.workload_mix[latents.workload_index[unit]].name,
                horizon_days=horizon,
                has_latent_defect=bool(latents.has_defect[unit]),
                failed_within_horizon=bool(failed[unit]),
                time_to_failure_days=float(failure_times[unit]) if failed[unit] else None,
                p_fail_assigned_workload=float(assigned_p[unit]),
                p_fail_by_workload={
                    profile.name: float(p_by_workload[unit, column])
                    for column, profile in enumerate(profiles)
                },
            )
        )
    return labels


def generate_fleet(config: FleetConfig) -> SimulatedFleet:
    """Generate one simulated fleet batch from ``config``.

    The seed is the sole entropy source and the draw order is fixed, so equal
    configs yield bit-identical batches. Seeds and headline counts are logged.
    """
    _LOGGER.info(
        "simulator: generating fleet — seed=%d n_units=%d archetypes=%s horizon_days=%d "
        "multipliers(defect=%.3f, ecc=%.3f, xid=%.3f, failure=%.3f)",
        config.seed,
        config.n_units,
        [archetype.model_name for archetype in config.archetypes],
        config.horizon_days,
        config.latent_defect_rate_multiplier,
        config.ecc_rate_multiplier,
        config.xid_rate_multiplier,
        config.failure_rate_multiplier,
    )
    rng = np.random.default_rng(config.seed)
    latents = _draw_latents(rng, config)
    observations = _draw_observations(rng, config, latents)
    labels = _build_labels(rng, config, latents)
    fleet = SimulatedFleet(
        config=config, scans=_build_scans(config, latents, observations), labels=labels
    )
    failed = sum(label.failed_within_horizon for label in fleet.labels)
    _LOGGER.info(
        "simulator: fleet complete — seed=%d units=%d failed_within_horizon=%d/%d",
        config.seed,
        fleet.n_units,
        failed,
        fleet.n_units,
    )
    return fleet


__all__ = [
    "BENCH_MARK_KEYS",
    "CUSTODY_EVENTS",
    "GroundTruthLabel",
    "SimulatedFleet",
    "generate_fleet",
]
