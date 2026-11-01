"""Scenario tests for fleet generation (spec §4, TESTING.md simulator row).

Covers schema validation on sampled outputs, ground-truth label integrity,
and the workload-simulator behaviors the spec promises: risk ordering across
workloads, defect separation, and calibration of the analytic label
probabilities against the realized event draws.
"""

import logging

import numpy as np
import pytest

from rechip01.schemas import Provenance, ScanResult
from rechip01.simulator import (
    BENCH_MARK_KEYS,
    FleetConfig,
    GroundTruthLabel,
    SimulatedFleet,
    WorkloadShare,
    default_fleet_config,
    generate_fleet,
)


def test_every_scan_validates_against_the_schema(scenario_config: FleetConfig) -> None:
    fleet = generate_fleet(scenario_config)
    for scan in fleet.scans:
        validated = ScanResult.model_validate(scan.model_dump())
        assert validated.provenance == Provenance.SIMULATED
        assert validated.chain_of_custody


def test_every_snapshot_uses_only_allowlisted_keys(scenario_config: FleetConfig) -> None:
    fleet = generate_fleet(scenario_config)
    for scan in fleet.scans:
        assert set(scan.bench_mark_snapshot) == set(BENCH_MARK_KEYS)
        assert all(np.isfinite(value) for value in scan.bench_mark_snapshot.values())


def test_thermal_history_is_monotone_percentiles_in_unit_interval(
    scenario_config: FleetConfig,
) -> None:
    fleet = generate_fleet(scenario_config)
    for scan in fleet.scans:
        history = scan.thermal_history
        assert all(0.0 <= value <= 1.0 for value in history)
        assert history == sorted(history)


def test_fleet_mix_has_both_models_and_all_three_workloads(
    scenario_config: FleetConfig,
) -> None:
    fleet = generate_fleet(scenario_config)
    models = {scan.model for scan in fleet.scans}
    workloads = {label.assigned_workload for label in fleet.labels}
    assert {archetype.model_name for archetype in scenario_config.archetypes} == models
    assert {share.name for share in scenario_config.workload_mix} == workloads


def test_scans_and_labels_are_aligned_by_asset_id(scenario_config: FleetConfig) -> None:
    fleet = generate_fleet(scenario_config)
    assert fleet.n_units == scenario_config.n_units
    assert [scan.asset_id for scan in fleet.scans] == [label.asset_id for label in fleet.labels]
    lookup = fleet.label_by_asset()
    assert set(lookup) == {scan.asset_id for scan in fleet.scans}


def test_labels_are_internally_consistent(scenario_config: FleetConfig) -> None:
    fleet = generate_fleet(scenario_config)
    for label in fleet.labels:
        assert label.failed_within_horizon == (label.time_to_failure_days is not None)
        if label.time_to_failure_days is not None:
            assert label.time_to_failure_days <= label.horizon_days
        assert label.p_fail_by_workload[label.assigned_workload] == label.p_fail_assigned_workload
        assert 0.0 <= label.p_fail_assigned_workload <= 1.0


def test_asset_ids_encode_seed_and_are_unique(scenario_config: FleetConfig) -> None:
    fleet = generate_fleet(scenario_config)
    ids = [scan.asset_id for scan in fleet.scans]
    assert len(set(ids)) == len(ids)
    assert all(asset_id.startswith(f"SIM-{scenario_config.seed}-") for asset_id in ids)


def test_seed_and_outcomes_are_logged(
    scenario_config: FleetConfig, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(logging.INFO, logger="rechip01.simulator.generate"):
        generate_fleet(scenario_config)
    joined = "\n".join(record.getMessage() for record in caplog.records)
    assert f"seed={scenario_config.seed}" in joined


def test_workload_stress_orders_failure_probabilities(
    scenario_config: FleetConfig,
) -> None:
    """Both stress profiles must dominate inference risk for every unit."""
    fleet = generate_fleet(scenario_config)
    for label in fleet.labels:
        p_inference = label.p_fail_by_workload["inference"]
        p_finetune = label.p_fail_by_workload["finetune"]
        p_memory_stress = label.p_fail_by_workload["memory-stress"]
        assert p_finetune >= p_inference
        assert p_memory_stress >= p_inference


def test_latent_defects_raise_failure_risk(scenario_config: FleetConfig) -> None:
    fleet = generate_fleet(scenario_config)
    defects = [label for label in fleet.labels if label.has_latent_defect]
    clean = [label for label in fleet.labels if not label.has_latent_defect]
    assert defects, "scenario seed must inject latent defects"
    p_fail_defects = np.mean([label.failed_within_horizon for label in defects])
    p_fail_clean = np.mean([label.failed_within_horizon for label in clean])
    assert p_fail_defects > 2.0 * p_fail_clean


def test_label_probabilities_are_calibrated_against_realized_events(
    scenario_config: FleetConfig,
) -> None:
    """Realized failure fraction must track the generator's assigned p."""
    fleet = generate_fleet(scenario_config)
    realized = np.mean([label.failed_within_horizon for label in fleet.labels])
    expected = np.mean([label.p_fail_assigned_workload for label in fleet.labels])
    assert abs(realized - expected) < 0.03


def test_misaligned_batches_are_rejected(scenario_config: FleetConfig) -> None:
    fleet = generate_fleet(scenario_config)
    with pytest.raises(ValueError, match="length mismatch"):
        SimulatedFleet(config=fleet.config, scans=fleet.scans, labels=fleet.labels[:-1])
    with pytest.raises(ValueError, match="align by asset_id"):
        SimulatedFleet(
            config=fleet.config,
            scans=fleet.scans,
            labels=[fleet.labels[1], *fleet.labels[1:]],
        )


def test_measured_records_are_rejected_in_simulated_fleets(
    scenario_config: FleetConfig,
) -> None:
    fleet = generate_fleet(scenario_config)
    measured = fleet.scans[0].model_copy(update={"provenance": Provenance.MEASURED})
    with pytest.raises(ValueError, match="provenance=SIMULATED"):
        SimulatedFleet(config=fleet.config, scans=[measured, *fleet.scans[1:]], labels=fleet.labels)


def test_unknown_workload_names_are_rejected(scenario_config: FleetConfig) -> None:
    with pytest.raises(ValueError, match="unknown workload profiles"):
        FleetConfig(
            seed=1,
            n_units=10,
            horizon_days=365,
            archetypes=scenario_config.archetypes,
            workload_mix=(
                *scenario_config.workload_mix[:-1],
                WorkloadShare(name="crypto-mining", share=scenario_config.workload_mix[-1].share),
            ),
        )


def test_single_unit_fleet_is_valid() -> None:
    fleet = generate_fleet(default_fleet_config(seed=0, n_units=1))
    assert fleet.n_units == 1
    assert isinstance(fleet.scans[0], ScanResult)
    assert isinstance(fleet.labels[0], GroundTruthLabel)
