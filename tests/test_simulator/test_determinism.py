"""Determinism and reproducibility tests for the fleet simulator (spec §4).

The spec's reproducibility bar: seeded runs are bit-reproducible. These tests
pin the whole batch — every field of every record — under a fixed seed, and
pin the causal structure that makes the labels trustworthy: label-only knobs
(horizon, failure-rate multiplier) must never perturb the feature stream.
"""

import pytest

from rechip01.simulator import FleetConfig, default_fleet_config, generate_fleet


def test_same_seed_reproduces_the_batch_bit_for_bit(
    scenario_config: FleetConfig,
) -> None:
    first = generate_fleet(scenario_config)
    second = generate_fleet(scenario_config)
    assert first.scans == second.scans
    assert first.labels == second.labels
    assert first.model_dump() == second.model_dump()


def test_different_seeds_produce_different_batches(
    scenario_config: FleetConfig,
) -> None:
    other = generate_fleet(
        default_fleet_config(seed=scenario_config.seed + 1, n_units=scenario_config.n_units)
    )
    assert other.config.seed != scenario_config.seed
    assert other.scans != generate_fleet(scenario_config).scans


def test_horizon_change_leaves_scans_untouched(scenario_config: FleetConfig) -> None:
    """The labeling horizon is not a feature: features must not react to it."""
    short = generate_fleet(scenario_config.model_copy(update={"horizon_days": 30}))
    long_horizon = generate_fleet(scenario_config.model_copy(update={"horizon_days": 400}))
    baseline = generate_fleet(scenario_config)
    assert short.scans == long_horizon.scans == baseline.scans
    assert short.labels != long_horizon.labels


def test_failure_rate_multiplier_leaves_scans_untouched(
    scenario_config: FleetConfig,
) -> None:
    baseline = generate_fleet(scenario_config)
    faster = generate_fleet(scenario_config.model_copy(update={"failure_rate_multiplier": 2.0}))
    assert baseline.scans == faster.scans
    assert baseline.labels != faster.labels


def test_workload_mix_shares_must_sum_to_one(scenario_config: FleetConfig) -> None:
    with pytest.raises(ValueError, match="shares must sum to 1"):
        FleetConfig(
            seed=1,
            n_units=10,
            horizon_days=365,
            archetypes=scenario_config.archetypes,
            workload_mix=(*scenario_config.workload_mix, scenario_config.workload_mix[0]),
        )


def test_archetype_shares_must_sum_to_one(scenario_config: FleetConfig) -> None:
    with pytest.raises(ValueError, match="shares must sum to 1"):
        FleetConfig(
            seed=1,
            n_units=10,
            horizon_days=365,
            archetypes=scenario_config.archetypes + (scenario_config.archetypes[0],),
            workload_mix=scenario_config.workload_mix,
        )


def test_zero_failure_rate_multiplier_is_rejected(scenario_config: FleetConfig) -> None:
    with pytest.raises(ValueError, match="greater than 0"):
        FleetConfig(
            seed=1,
            n_units=10,
            horizon_days=365,
            archetypes=scenario_config.archetypes,
            workload_mix=scenario_config.workload_mix,
            failure_rate_multiplier=0.0,
        )


def test_unordered_age_window_is_rejected() -> None:
    archetype = (
        default_fleet_config(seed=1)
        .archetypes[0]
        .model_copy(update={"age_months_min": 60.0, "age_months_max": 24.0})
    )
    with pytest.raises(ValueError, match="age_months_min must not exceed age_months_max"):
        FleetConfig(
            seed=1,
            n_units=10,
            horizon_days=365,
            archetypes=(archetype,),
            workload_mix=default_fleet_config(seed=1).workload_mix,
        )
