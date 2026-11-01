"""Reproducible evaluation harness: train → assess → evaluate (spec §8).

One function builds the default harness fleets (fixed seeds, defect injection
on the evaluation batch), fits the engine, scores every held-out unit under
its assigned workload as the primary target, and returns the evaluation. The
``python -m rechip01.engine`` CLI writes the same evaluation to a Markdown
report, which CI publishes as an artifact.

The default fleet sizes are pipeline knobs, not tuned hyperparameters: they
are the smallest batches that make the harness rails statistically stable in
CI (a few seconds of runtime).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Final

from rechip01.engine.config import EngineConfig
from rechip01.engine.engine import ReliabilityEngine
from rechip01.engine.metrics import EngineEvaluation, evaluate
from rechip01.schemas import ReliabilityEstimate
from rechip01.simulator.config import (
    BUILTIN_ARCHETYPES,
    BUILTIN_WORKLOAD_MIX,
    WORKLOAD_PROFILES,
    FleetConfig,
)
from rechip01.simulator.generate import SimulatedFleet, generate_fleet

_LOGGER = logging.getLogger(__name__)

# Harness fleet definitions. The evaluation batch carries injected latent
# defects (2x the base rate) so the early-failure signal the product prices
# is well-represented in the held-out ground truth.
TRAIN_SEED: Final[int] = 101
EVAL_SEED: Final[int] = 202
TRAIN_N_UNITS: Final[int] = 2500
EVAL_N_UNITS: Final[int] = 2000
HARNESS_HORIZON_DAYS: Final[int] = 365
EVAL_DEFECT_MULTIPLIER: Final[float] = 2.0


def harness_train_config() -> FleetConfig:
    """Training-fleet configuration (seed :data:`TRAIN_SEED`)."""
    return FleetConfig(
        seed=TRAIN_SEED,
        n_units=TRAIN_N_UNITS,
        horizon_days=HARNESS_HORIZON_DAYS,
        archetypes=BUILTIN_ARCHETYPES,
        workload_mix=BUILTIN_WORKLOAD_MIX,
    )


def harness_eval_config() -> FleetConfig:
    """Evaluation-fleet configuration (seed :data:`EVAL_SEED`, defects injected)."""
    return FleetConfig(
        seed=EVAL_SEED,
        n_units=EVAL_N_UNITS,
        horizon_days=HARNESS_HORIZON_DAYS,
        archetypes=BUILTIN_ARCHETYPES,
        workload_mix=BUILTIN_WORKLOAD_MIX,
        latent_defect_rate_multiplier=EVAL_DEFECT_MULTIPLIER,
    )


@dataclass(frozen=True)
class HarnessResult:
    """Everything the harness produced: fleets, estimates, and the evaluation."""

    train_fleet: SimulatedFleet
    eval_fleet: SimulatedFleet
    estimates: list[ReliabilityEstimate]
    evaluation: EngineEvaluation
    engine: ReliabilityEngine


def run_harness(config: EngineConfig | None = None) -> HarnessResult:
    """Run the full harness with the default fleet configs and engine config."""
    engine_config = config if config is not None else EngineConfig()
    train_fleet = generate_fleet(harness_train_config())
    eval_fleet = generate_fleet(harness_eval_config())

    engine = ReliabilityEngine(engine_config)
    engine.fit(train_fleet)

    _LOGGER.info("harness: scoring %d held-out units", eval_fleet.n_units)
    estimates = assess_by_assigned_workload(engine, eval_fleet)

    evaluation = evaluate(
        estimates,
        eval_fleet.label_by_asset(),
        ci_level=engine_config.ci_level,
        training_fingerprint=engine.training_fingerprint(),
    )
    return HarnessResult(
        train_fleet=train_fleet,
        eval_fleet=eval_fleet,
        estimates=estimates,
        evaluation=evaluation,
        engine=engine,
    )


def assess_by_assigned_workload(
    engine: ReliabilityEngine, fleet: SimulatedFleet
) -> list[ReliabilityEstimate]:
    """Assess each unit under its assigned workload as the primary target.

    Units are grouped by assigned workload so each group is one ``assess``
    call (every group also scores the full workload mix for fitness columns);
    the result is reassembled in fleet order.
    """
    mix_profiles = [WORKLOAD_PROFILES[share.name] for share in fleet.config.workload_mix]
    groups: dict[str, list[int]] = {}
    for index, label in enumerate(fleet.labels):
        groups.setdefault(label.assigned_workload, []).append(index)

    estimates_by_workload: dict[str, list[ReliabilityEstimate]] = {}
    for workload_name, indices in groups.items():
        primary = WORKLOAD_PROFILES[workload_name]
        targets = [primary] + [p for p in mix_profiles if p.name != primary.name]
        scans = [fleet.scans[i] for i in indices]
        estimates_by_workload[workload_name] = engine.assess(
            scans, horizon_days=fleet.config.horizon_days, target_workloads=targets
        )

    by_index: dict[int, ReliabilityEstimate] = {}
    for workload_name, indices in groups.items():
        for index, estimate in zip(indices, estimates_by_workload[workload_name], strict=True):
            by_index[index] = estimate
    return [by_index[i] for i in range(fleet.n_units)]


__all__ = [
    "EVAL_DEFECT_MULTIPLIER",
    "EVAL_N_UNITS",
    "EVAL_SEED",
    "HARNESS_HORIZON_DAYS",
    "HarnessResult",
    "TRAIN_N_UNITS",
    "TRAIN_SEED",
    "assess_by_assigned_workload",
    "harness_eval_config",
    "harness_train_config",
    "run_harness",
]
