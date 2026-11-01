"""Property tests for the fleet simulator (spec §4, TESTING.md simulator row).

The task's three required properties hold for *arbitrary* valid configs, not
just the curated scenario:

- determinism: a fixed seed reproduces a batch bit-for-bit;
- schema validity: every emitted scan validates against ``ScanResult`` with
  ``provenance=SIMULATED`` and allowlisted snapshot keys;
- label/feature leakage: ground-truth names never appear in the feature set,
  and label-only config changes (horizon, hazard scale) leave every feature
  value untouched — the behavioral form of the no-leakage guarantee.

Configs are kept small (n_units <= 40) to bound runtime; the scenario suites
cover fleet-scale behavior.
"""

from hypothesis import given, settings
from hypothesis import strategies as st

from rechip01.schemas import Provenance, ScanResult
from rechip01.simulator import (
    BENCH_MARK_KEYS,
    FleetConfig,
    GroundTruthLabel,
    default_fleet_config,
    generate_fleet,
)

# Ground-truth label fields that must never surface in the feature set.
LABEL_ONLY_FIELDS = frozenset(
    {
        "assigned_workload",
        "has_latent_defect",
        "failed_within_horizon",
        "time_to_failure_days",
        "p_fail_assigned_workload",
        "p_fail_by_workload",
    }
)

SEEDS = st.integers(min_value=0, max_value=2**31 - 1)
N_UNITS = st.integers(min_value=1, max_value=40)
HORIZONS = st.integers(min_value=1, max_value=2000)


def fleet_config(seed: int, n_units: int, horizon_days: int, **overrides: float) -> FleetConfig:
    """Build a built-in fleet config for property testing."""
    config = default_fleet_config(seed=seed, n_units=n_units, horizon_days=horizon_days)
    return config.model_copy(update=overrides) if overrides else config


@settings(max_examples=25, deadline=None)
@given(seed=SEEDS, n_units=N_UNITS, horizon_days=HORIZONS)
def test_property_same_seed_is_bit_reproducible(seed: int, n_units: int, horizon_days: int) -> None:
    config = fleet_config(seed, n_units, horizon_days)
    assert generate_fleet(config) == generate_fleet(config)


@settings(max_examples=25, deadline=None)
@given(seed=SEEDS, n_units=N_UNITS, horizon_days=HORIZONS)
def test_property_every_scan_is_valid_simulated_schema(
    seed: int, n_units: int, horizon_days: int
) -> None:
    fleet = generate_fleet(fleet_config(seed, n_units, horizon_days))
    for scan in fleet.scans:
        validated = ScanResult.model_validate(scan.model_dump())
        assert validated.provenance == Provenance.SIMULATED
        assert set(validated.bench_mark_snapshot) == set(BENCH_MARK_KEYS)
        assert all(0.0 <= v <= 1.0 for v in validated.thermal_history)
        assert 0.0 <= validated.max_sustained_power_pct <= 1.0


@settings(max_examples=25, deadline=None)
@given(seed=SEEDS, n_units=N_UNITS, horizon_days=HORIZONS)
def test_property_labels_align_with_scans(seed: int, n_units: int, horizon_days: int) -> None:
    fleet = generate_fleet(fleet_config(seed, n_units, horizon_days))
    assert [scan.asset_id for scan in fleet.scans] == [label.asset_id for label in fleet.labels]
    for label in fleet.labels:
        assert isinstance(label, GroundTruthLabel)
        assert label.failed_within_horizon == (label.time_to_failure_days is not None)
        assert label.p_fail_by_workload[label.assigned_workload] == label.p_fail_assigned_workload


@settings(max_examples=25, deadline=None)
@given(seed=SEEDS, n_units=N_UNITS, horizon_days=HORIZONS)
def test_property_ground_truth_never_appears_in_features(
    seed: int, n_units: int, horizon_days: int
) -> None:
    fleet = generate_fleet(fleet_config(seed, n_units, horizon_days))
    scan_fields = set(ScanResult.model_fields)
    assert not LABEL_ONLY_FIELDS & scan_fields
    assert set(BENCH_MARK_KEYS).isdisjoint(LABEL_ONLY_FIELDS)
    for scan in fleet.scans:
        assert not LABEL_ONLY_FIELDS & set(scan.bench_mark_snapshot)


@settings(max_examples=25, deadline=None)
@given(seed=SEEDS, n_units=N_UNITS, horizon_a=HORIZONS, horizon_b=HORIZONS)
def test_property_label_horizon_never_leaks_into_features(
    seed: int, n_units: int, horizon_a: int, horizon_b: int
) -> None:
    config_a = fleet_config(seed, n_units, horizon_a)
    config_b = fleet_config(seed, n_units, horizon_b)
    assert generate_fleet(config_a).scans == generate_fleet(config_b).scans


@settings(max_examples=25, deadline=None)
@given(seed=SEEDS, n_units=N_UNITS, horizon_days=HORIZONS)
def test_property_hazard_scale_never_leaks_into_features(
    seed: int, n_units: int, horizon_days: int
) -> None:
    config_a = fleet_config(seed, n_units, horizon_days)
    config_b = fleet_config(seed, n_units, horizon_days, failure_rate_multiplier=2.5)
    assert generate_fleet(config_a).scans == generate_fleet(config_b).scans
