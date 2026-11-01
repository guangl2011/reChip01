"""Physics-grounded synthetic fleet generation (spec art_vuIPvUhR, §4).

Public API:

- ``FleetConfig`` and companions — declared, frozen simulation inputs with
  built-in A100/H100-class archetypes and the three second-life workload
  profiles (inference, fine-tuning, memory-intensive stress).
- ``generate_fleet`` — one config in, one ``SimulatedFleet`` out: a
  ``ScanResult`` stream with ``provenance=SIMULATED`` plus held-out
  ``GroundTruthLabel`` failure labels and time-to-event, joined by ``asset_id``
  only at evaluation time.

Every run is seeded and the draw order is fixed: equal configs are
bit-identical. Ground truth is latent-covariate-driven, so label leakage into
the feature set is structurally excluded and verified by tests.
"""

from rechip01.simulator.config import (
    A100_SXM4_80GB,
    BUILTIN_ARCHETYPES,
    BUILTIN_WORKLOAD_MIX,
    BUILTIN_WORKLOAD_PROFILES,
    FINE_TUNE_PROFILE,
    H100_SXM4_80GB,
    INFERENCE_PROFILE,
    MEMORY_STRESS_PROFILE,
    WORKLOAD_PROFILES,
    Archetype,
    FleetConfig,
    WorkloadProfile,
    WorkloadShare,
    default_fleet_config,
)
from rechip01.simulator.generate import (
    BENCH_MARK_KEYS,
    CUSTODY_EVENTS,
    GroundTruthLabel,
    SimulatedFleet,
    generate_fleet,
)
from rechip01.simulator.physics import (
    THERMAL_PERCENTILES,
    latent_log_hazard,
    survival_probability,
    thermal_acceleration,
    workload_log_hazard_shift,
)

__all__ = [
    "A100_SXM4_80GB",
    "Archetype",
    "BENCH_MARK_KEYS",
    "BUILTIN_ARCHETYPES",
    "BUILTIN_WORKLOAD_MIX",
    "BUILTIN_WORKLOAD_PROFILES",
    "CUSTODY_EVENTS",
    "FleetConfig",
    "FINE_TUNE_PROFILE",
    "GroundTruthLabel",
    "H100_SXM4_80GB",
    "INFERENCE_PROFILE",
    "MEMORY_STRESS_PROFILE",
    "SimulatedFleet",
    "THERMAL_PERCENTILES",
    "WorkloadProfile",
    "WorkloadShare",
    "WORKLOAD_PROFILES",
    "default_fleet_config",
    "generate_fleet",
    "latent_log_hazard",
    "survival_probability",
    "thermal_acceleration",
    "workload_log_hazard_shift",
]
