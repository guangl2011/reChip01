"""Unit-feature extraction, imputation semantics, and workload rows (spec §5)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from rechip01.engine.features import (
    ENGINE_BENCH_KEYS,
    FEATURE_NAMES,
    UNIT_FEATURE_NAMES,
    FeatureSpace,
    Standardizer,
    workload_feature_row,
)
from rechip01.simulator import SimulatedFleet, WorkloadProfile, default_fleet_config, generate_fleet
from tests.test_engine.helpers import with_missing_fields

# Label-only ground-truth fields that must never surface as feature names.
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


def test_feature_names_exclude_label_fields() -> None:
    """Structural leakage audit: no ground-truth field is a feature name."""
    overlap = LABEL_ONLY_FIELDS & set(FEATURE_NAMES)
    assert not overlap, f"leakage: label fields in the feature set: {overlap}"


def test_workload_row_matches_profile_fields() -> None:
    profile = WorkloadProfile(
        name="w", duty_cycle=0.8, thermal_offset=0.05, power_offset=0.12, ecc_multiplier=1.5
    )
    row = workload_feature_row(profile)
    assert row.shape == (4,)
    assert row[0] == pytest.approx(0.8)
    assert row[1] == pytest.approx(0.05)
    assert row[2] == pytest.approx(0.12)
    assert row[3] == pytest.approx(math.log(1.5))


def test_complete_scans_have_no_missing_rows(
    scenario_fleets: tuple[SimulatedFleet, SimulatedFleet],
) -> None:
    train, _ = scenario_fleets
    space = FeatureSpace.fit(train.scans)
    features = space.transform(train.scans)
    assert features.matrix.shape == (train.n_units, len(UNIT_FEATURE_NAMES))
    assert not features.missing.any()
    assert np.isfinite(features.matrix).all()


def test_partial_scan_gets_training_medians(
    scenario_fleets: tuple[SimulatedFleet, SimulatedFleet],
) -> None:
    train, _ = scenario_fleets
    space = FeatureSpace.fit(train.scans)

    degraded_features = space.transform([with_missing_fields(train.scans[0])])
    complete_features = space.transform([train.scans[0]])
    assert degraded_features.missing[0]

    # Every imputed cell equals the training median — not zero, not a sentinel.
    degraded_row = degraded_features.matrix[0]
    complete_row = complete_features.matrix[0]
    for column, name in enumerate(UNIT_FEATURE_NAMES):
        if complete_row[column] != degraded_row[column]:
            assert degraded_row[column] == pytest.approx(space.medians[name])


def test_partial_scan_missing_flag_tracks_absent_keys(
    scenario_fleets: tuple[SimulatedFleet, SimulatedFleet],
) -> None:
    """Missingness surfaces as the missing flag (which drives widening), not a feature."""
    train, _ = scenario_fleets
    space = FeatureSpace.fit(train.scans)
    complete = space.transform([train.scans[0]])
    degraded = space.transform([with_missing_fields(train.scans[0])])
    assert not complete.missing[0]
    assert degraded.missing[0]
    # Exactly the three engine-consumed benchmark columns changed.
    changed = [
        name
        for column, name in enumerate(UNIT_FEATURE_NAMES)
        if complete.matrix[0, column] != degraded.matrix[0, column]
    ]
    assert set(changed) == {f"log_{key}" for key in ENGINE_BENCH_KEYS}


def test_fit_raises_when_a_column_is_never_observed() -> None:
    fleet = generate_fleet(default_fleet_config(seed=909, n_units=50))
    stripped = [with_missing_fields(scan) for scan in fleet.scans]
    with pytest.raises(ValueError, match="never provide"):
        FeatureSpace.fit(stripped)


def test_fit_rejects_empty() -> None:
    with pytest.raises(ValueError, match="empty"):
        FeatureSpace.fit([])


def test_standardizer_handles_constant_features() -> None:
    matrix = np.array([[1.0, 2.0], [1.0, 4.0], [1.0, 6.0]])
    standardizer = Standardizer.fit(matrix)
    transformed = standardizer.transform(np.array([[1.0, 2.0], [1.0, 6.0]]))
    assert standardizer.stds[0] == pytest.approx(1.0)  # constant column -> unit std
    assert transformed[:, 0].tolist() == [0.0, 0.0]  # constant column maps to zero
    assert abs(transformed[1, 1]) > 0  # varying column still standardizes
