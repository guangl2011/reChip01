"""Monotonicity tests for the simulator's tunable rates (spec §4).

Each knob must move its observed outcome in the physically sensible direction,
with margins wide enough that the assertion is robust to stream re-rollings
across numpy releases: knobs sweep by 2–4x at scenario scale (4,000 units).
"""

import numpy as np

from rechip01.simulator import FleetConfig, SimulatedFleet, generate_fleet

# Knob sweeps: (field, low, high). All up-sweeps.
_ECC_SWEEP = ("ecc_rate_multiplier", 0.5, 2.0)
_XID_SWEEP = ("xid_rate_multiplier", 0.5, 2.0)
_DEFECT_SWEEP = ("latent_defect_rate_multiplier", 0.5, 3.0)
_FAILURE_SWEEP = ("failure_rate_multiplier", 0.5, 2.0)


def _generate_pair(
    config: FleetConfig, field: str, low: float, high: float
) -> tuple[SimulatedFleet, SimulatedFleet]:
    low_fleet = generate_fleet(config.model_copy(update={field: low}))
    high_fleet = generate_fleet(config.model_copy(update={field: high}))
    return low_fleet, high_fleet


def test_ecc_rate_multiplier_raises_observed_ecc(scenario_config: FleetConfig) -> None:
    field, low, high = _ECC_SWEEP
    low_fleet, high_fleet = _generate_pair(scenario_config, field, low, high)
    low_mean = np.mean([scan.ecc_events for scan in low_fleet.scans])
    high_mean = np.mean([scan.ecc_events for scan in high_fleet.scans])
    assert high_mean > 2.0 * low_mean


def test_xid_rate_multiplier_raises_observed_xid(scenario_config: FleetConfig) -> None:
    field, low, high = _XID_SWEEP
    low_fleet, high_fleet = _generate_pair(scenario_config, field, low, high)
    low_mean = np.mean([scan.xid_errors for scan in low_fleet.scans])
    high_mean = np.mean([scan.xid_errors for scan in high_fleet.scans])
    assert high_mean > 2.0 * low_mean


def test_defect_multiplier_raises_defect_prevalence_and_failures(
    scenario_config: FleetConfig,
) -> None:
    field, low, high = _DEFECT_SWEEP
    low_fleet, high_fleet = _generate_pair(scenario_config, field, low, high)
    low_defects = np.mean([label.has_latent_defect for label in low_fleet.labels])
    high_defects = np.mean([label.has_latent_defect for label in high_fleet.labels])
    low_failed = np.mean([label.failed_within_horizon for label in low_fleet.labels])
    high_failed = np.mean([label.failed_within_horizon for label in high_fleet.labels])
    assert high_defects > 2.0 * low_defects
    assert high_failed > low_failed


def test_failure_rate_multiplier_raises_failures(scenario_config: FleetConfig) -> None:
    field, low, high = _FAILURE_SWEEP
    low_fleet, high_fleet = _generate_pair(scenario_config, field, low, high)
    low_failed = np.mean([label.failed_within_horizon for label in low_fleet.labels])
    high_failed = np.mean([label.failed_within_horizon for label in high_fleet.labels])
    assert high_failed > 2.0 * low_failed
