"""Engine facade tests: fit/assess end-to-end behavior on scenario fleets (spec §4–5)."""

from __future__ import annotations

import pytest

from rechip01.engine import EngineConfig, ReliabilityEngine
from rechip01.schemas import Provenance, ReliabilityEstimate
from rechip01.simulator import (
    WORKLOAD_PROFILES,
    SimulatedFleet,
    WorkloadProfile,
    default_fleet_config,
    generate_fleet,
)
from tests.test_engine.helpers import (
    degraded_twin,
    with_measured_provenance,
    with_missing_fields,
)

MIX = [WORKLOAD_PROFILES[name] for name in ("inference", "finetune", "memory-stress")]


def test_assess_emits_valid_estimates_in_order(
    fitted_engine: ReliabilityEngine,
    scenario_fleets: tuple[SimulatedFleet, SimulatedFleet],
) -> None:
    _, eval_fleet = scenario_fleets
    estimates = fitted_engine.assess(eval_fleet.scans, horizon_days=365, target_workloads=MIX)

    assert len(estimates) == eval_fleet.n_units
    for scan, estimate in zip(eval_fleet.scans, estimates, strict=True):
        assert isinstance(estimate, ReliabilityEstimate)
        assert estimate.asset_id == scan.asset_id
        assert estimate.horizon_days == 365
        assert 0.0 <= estimate.p_fail <= 1.0
        assert estimate.ci[0] <= estimate.p_fail <= estimate.ci[1]
        assert set(estimate.workload_fitness) == {profile.name for profile in MIX}
        assert estimate.model_version


def test_provenance_propagates_from_scans(
    fitted_engine: ReliabilityEngine,
    scenario_fleets: tuple[SimulatedFleet, SimulatedFleet],
) -> None:
    _, eval_fleet = scenario_fleets
    measured = [with_measured_provenance(scan) for scan in eval_fleet.scans[:5]]
    estimates = fitted_engine.assess(measured, horizon_days=365, target_workloads=MIX[:1])
    assert all(estimate.provenance == Provenance.MEASURED for estimate in estimates)


def test_fitness_matches_inverse_primary_p_fail(
    fitted_engine: ReliabilityEngine,
    scenario_fleets: tuple[SimulatedFleet, SimulatedFleet],
) -> None:
    _, eval_fleet = scenario_fleets
    estimates = fitted_engine.assess(eval_fleet.scans[:20], horizon_days=365, target_workloads=MIX)
    for estimate in estimates:
        assert estimate.workload_fitness["inference"] == pytest.approx(1.0 - estimate.p_fail)


def test_workload_conditioning_orders_risk(
    fitted_engine: ReliabilityEngine,
    scenario_fleets: tuple[SimulatedFleet, SimulatedFleet],
) -> None:
    """Memory-stress carries a larger generator hazard shift than inference."""
    _, eval_fleet = scenario_fleets
    estimates = fitted_engine.assess(eval_fleet.scans, horizon_days=365, target_workloads=MIX)
    mean_inference = sum(e.workload_fitness["inference"] for e in estimates) / len(estimates)
    mean_stress = sum(e.workload_fitness["memory-stress"] for e in estimates) / len(estimates)
    assert mean_stress < mean_inference  # lower fitness = higher modeled risk


def test_missing_fields_widen_uncertainty(
    fitted_engine: ReliabilityEngine,
    scenario_fleets: tuple[SimulatedFleet, SimulatedFleet],
) -> None:
    """Widening multiplies the hazard SE: same point, strictly wider interval."""
    train, eval_fleet = scenario_fleets
    scan = eval_fleet.scans[0]
    degraded = with_missing_fields(scan)
    assert fitted_engine.missing_flags([scan]) == [False]
    assert fitted_engine.missing_flags([degraded]) == [True]

    baseline = fitted_engine.assess([scan], horizon_days=365, target_workloads=MIX[:1])[0]
    partial = fitted_engine.assess([degraded], horizon_days=365, target_workloads=MIX[:1])[0]
    # Imputation visibly moves the point estimate (median substitution is not
    # identity); the widening check itself must hold at a fixed point, since
    # interval width in probability space also depends on where the point sits.
    assert partial.p_fail != baseline.p_fail

    tight = (
        ReliabilityEngine(EngineConfig(missing_widening=1.0, ood_widening=1.0))
        .fit(train)
        .assess([degraded], horizon_days=365, target_workloads=MIX[:1])[0]
    )
    assert partial.p_fail == pytest.approx(tight.p_fail)
    assert partial.ci[0] <= tight.ci[0]
    assert partial.ci[1] >= tight.ci[1]


def test_ood_unit_flagged_and_widened(
    fitted_engine: ReliabilityEngine,
    scenario_fleets: tuple[SimulatedFleet, SimulatedFleet],
) -> None:
    """An out-of-fence unit keeps its estimate but is flagged and widened."""
    _, eval_fleet = scenario_fleets
    scan = eval_fleet.scans[0]
    extreme = degraded_twin(scan)
    baseline = fitted_engine.assess([scan], horizon_days=365, target_workloads=MIX[:1])[0]
    flagged = fitted_engine.assess([extreme], horizon_days=365, target_workloads=MIX[:1])[0]

    assert baseline.ood_flag is False
    assert flagged.ood_flag is True
    assert (flagged.ci[1] - flagged.ci[0]) > (baseline.ci[1] - baseline.ci[0])


def test_ood_workload_flags_every_unit(
    fitted_engine: ReliabilityEngine,
    scenario_fleets: tuple[SimulatedFleet, SimulatedFleet],
) -> None:
    """A workload outside training support flags the whole estimate (never silent)."""
    _, eval_fleet = scenario_fleets
    exotic = WorkloadProfile(
        name="exotic-finetune",
        duty_cycle=0.95,
        thermal_offset=0.35,
        power_offset=0.50,
        ecc_multiplier=3.0,
    )
    in_support = fitted_engine.assess(
        eval_fleet.scans[:10], horizon_days=365, target_workloads=MIX[:1]
    )
    out_of_support = fitted_engine.assess(
        eval_fleet.scans[:10], horizon_days=365, target_workloads=[MIX[0], exotic]
    )

    assert all(estimate.ood_flag for estimate in out_of_support)
    assert not any(estimate.ood_flag for estimate in in_support)
    for in_estimate, out_estimate in zip(in_support, out_of_support, strict=True):
        assert (out_estimate.ci[1] - out_estimate.ci[0]) > (in_estimate.ci[1] - in_estimate.ci[0])


def test_assess_rejects_unfitted_engine() -> None:
    fleet = generate_fleet(default_fleet_config(seed=555, n_units=10))
    with pytest.raises(RuntimeError, match="fitted"):
        ReliabilityEngine().assess(fleet.scans, horizon_days=365, target_workloads=MIX[:1])


def test_assess_rejects_bad_arguments(
    fitted_engine: ReliabilityEngine,
    scenario_fleets: tuple[SimulatedFleet, SimulatedFleet],
) -> None:
    _, eval_fleet = scenario_fleets
    with pytest.raises(ValueError, match="target workload"):
        fitted_engine.assess(eval_fleet.scans[:3], horizon_days=365, target_workloads=[])
    with pytest.raises(ValueError, match="horizon_days"):
        fitted_engine.assess(eval_fleet.scans[:3], horizon_days=0, target_workloads=MIX[:1])


def test_engine_determinism_with_fixed_seed() -> None:
    """Same seed -> same fleet, same fit, bit-identical estimates."""
    train_a = generate_fleet(default_fleet_config(seed=3131, n_units=150))
    train_b = generate_fleet(default_fleet_config(seed=3131, n_units=150))
    assert train_a.scans == train_b.scans

    engine_a = ReliabilityEngine(EngineConfig()).fit(train_a)
    engine_b = ReliabilityEngine(EngineConfig()).fit(train_b)
    assert engine_a.training_fingerprint() == engine_b.training_fingerprint()

    eval_fleet = generate_fleet(default_fleet_config(seed=3232, n_units=40))
    estimates_a = engine_a.assess(eval_fleet.scans, horizon_days=365, target_workloads=MIX)
    estimates_b = engine_b.assess(eval_fleet.scans, horizon_days=365, target_workloads=MIX)
    assert [e.model_dump_json() for e in estimates_a] == [e.model_dump_json() for e in estimates_b]


def test_assess_empty_scan_list_returns_empty(fitted_engine: ReliabilityEngine) -> None:
    assert fitted_engine.assess([], horizon_days=365, target_workloads=MIX[:1]) == []
