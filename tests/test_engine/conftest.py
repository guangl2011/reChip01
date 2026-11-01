"""Fixtures for the engine test suite (spec §4–5, §8)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from rechip01.engine import EngineConfig, ReliabilityEngine
from rechip01.simulator import WORKLOAD_PROFILES, SimulatedFleet, WorkloadProfile
from tests.test_engine.helpers import EVAL_N, EVAL_SEED, TRAIN_N, TRAIN_SEED, make_fleet

if TYPE_CHECKING:
    from rechip01.engine.harness import HarnessResult

MIX = [WORKLOAD_PROFILES[name] for name in ("inference", "finetune", "memory-stress")]


@pytest.fixture
def scenario_fleets() -> tuple[SimulatedFleet, SimulatedFleet]:
    """Deterministic train/eval pair; eval is large for stable metrics."""
    return make_fleet(TRAIN_SEED, TRAIN_N), make_fleet(EVAL_SEED, EVAL_N)


@pytest.fixture
def fitted_engine(scenario_fleets: tuple[SimulatedFleet, SimulatedFleet]) -> ReliabilityEngine:
    train_fleet, _ = scenario_fleets
    return ReliabilityEngine(EngineConfig()).fit(train_fleet)


@pytest.fixture(scope="session")
def harness_result() -> HarnessResult:
    """One harness run shared by the go/no-go rail tests."""
    from rechip01.engine.harness import run_harness

    return run_harness()


@pytest.fixture
def exotic_workload() -> WorkloadProfile:
    """A workload strictly outside the simulator's named profiles."""
    return WorkloadProfile(
        name="exotic-finetune",
        duty_cycle=0.95,
        thermal_offset=0.35,
        power_offset=0.50,
        ecc_multiplier=3.0,
    )
