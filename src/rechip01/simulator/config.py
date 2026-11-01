"""Simulator configuration contracts (spec art_vuIPvUhR, §4).

A simulated batch is fully described by its ``FleetConfig`` plus the seed it
carries, so any two runs with equal configs are bit-identical — that property
is enforced by tests, not convention. Built-in archetypes cover A100/H100-class
GPUs and the built-in workload profiles cover the three second-life regimes the
spec names (inference, fine-tuning, memory-intensive stress).

Every knob a partner-facing demo needs is here: fleet mix, age windows,
event-rate baselines, and the tunable defect/failure-rate multipliers the
evaluation harness sweeps. Hazard-model coefficients live in
``rechip01.simulator.physics`` — they are model physics, not configuration.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, model_validator

from rechip01.schemas import FiniteFloat, NonNegativeFloat, UnitInterval


class _FrozenContract(BaseModel):
    """Base for simulator contracts: immutable after construction."""

    model_config = ConfigDict(frozen=True)


class WorkloadProfile(_FrozenContract):
    """Second-life operating regime a retired unit is deployed into.

    The offsets describe how the regime differs from the unit's first-life
    operating point; they feed the hazard model through
    ``physics.workload_log_hazard_shift`` — one source of truth, no dead knobs.
    """

    name: str = Field(min_length=1)
    duty_cycle: UnitInterval = Field(description="Fraction of each 24h the unit is active")
    thermal_offset: FiniteFloat = Field(
        description="Normalized-temperature delta vs first-life operating point"
    )
    power_offset: FiniteFloat = Field(
        description="Normalized sustained-power delta vs first-life operating point"
    )
    ecc_multiplier: NonNegativeFloat = Field(
        gt=0, description="Memory-subsystem stress factor on correctable-error rates"
    )


class WorkloadShare(_FrozenContract):
    """One workload's weight in the second-life deployment mix."""

    name: str = Field(min_length=1)
    share: UnitInterval


class Archetype(_FrozenContract):
    """Hardware class and first-life baseline for one GPU model.

    Rate baselines are per 1,000 power-on hours; the Weibull parameters shape
    the baseline failure-time distribution at reference covariates (see
    ``physics`` for how covariates scale it).
    """

    model_name: str = Field(min_length=1, description='GPU model, e.g. "A100-SXM4-80GB"')
    fleet_share: UnitInterval = Field(description="Weight in the fleet mix; shares must sum to 1")
    age_months_min: NonNegativeFloat
    age_months_max: NonNegativeFloat = Field(description="Deployment-window bounds; min <= max")
    duty_cycle_mean: UnitInterval = Field(description="Center of first-life utilization")
    thermal_mean: UnitInterval = Field(description="Normalized first-life operating temperature")
    power_mean: UnitInterval = Field(description="Normalized first-life sustained power")
    ecc_rate_per_1000h: NonNegativeFloat
    xid_rate_per_1000h: NonNegativeFloat
    latent_defect_rate: UnitInterval = Field(description="Per-unit probability of a latent defect")
    weibull_shape: float = Field(gt=0)
    baseline_scale_days: float = Field(gt=0, description="Weibull scale at reference covariates")
    base_clock_mhz: float = Field(gt=0, description="Boost clock used by the throttle snapshot")

    @model_validator(mode="after")
    def _age_window_is_ordered(self) -> Archetype:
        if self.age_months_min > self.age_months_max:
            raise ValueError(
                f"age_months_min must not exceed age_months_max "
                f"({self.age_months_min} > {self.age_months_max})"
            )
        return self


class FleetConfig(_FrozenContract):
    """Full description of one simulated fleet batch (spec §4, §5 step 1)."""

    seed: int = Field(ge=0, description="Sole entropy source; logged with every run")
    n_units: int = Field(ge=1)
    horizon_days: int = Field(ge=1, description="Second-life observation window for labels")
    archetypes: tuple[Archetype, ...] = Field(min_length=1)
    workload_mix: tuple[WorkloadShare, ...] = Field(min_length=1)
    latent_defect_rate_multiplier: NonNegativeFloat = 1.0
    ecc_rate_multiplier: NonNegativeFloat = 1.0
    xid_rate_multiplier: NonNegativeFloat = 1.0
    failure_rate_multiplier: NonNegativeFloat = Field(
        default=1.0,
        gt=0,
        description="Hazard scale; 0 is meaningless for the Weibull form and rejected",
    )

    @model_validator(mode="after")
    def _mixes_are_valid(self) -> FleetConfig:
        self._shares_sum_to_one(
            [(archetype.model_name, archetype.fleet_share) for archetype in self.archetypes]
        )
        self._shares_sum_to_one([(share.name, share.share) for share in self.workload_mix])
        archetype_names = [archetype.model_name for archetype in self.archetypes]
        if len(set(archetype_names)) != len(archetype_names):
            raise ValueError(f"archetype model names must be unique: {archetype_names}")
        workload_names = [share.name for share in self.workload_mix]
        if len(set(workload_names)) != len(workload_names):
            raise ValueError(f"workload mix names must be unique: {workload_names}")
        unknown = sorted(set(workload_names) - set(WORKLOAD_PROFILES))
        if unknown:
            raise ValueError(f"unknown workload profiles in mix: {unknown}")
        return self

    @staticmethod
    def _shares_sum_to_one(named_shares: list[tuple[str, float]]) -> None:
        total = sum(share for _, share in named_shares)
        if abs(total - 1.0) > 1e-6:
            raise ValueError(
                f"shares must sum to 1 (within 1e-6), got {total} for "
                f"{[name for name, _ in named_shares]}"
            )


A100_SXM4_80GB = Archetype(
    model_name="A100-SXM4-80GB",
    fleet_share=0.5,
    age_months_min=24.0,
    age_months_max=66.0,
    duty_cycle_mean=0.62,
    thermal_mean=0.55,
    power_mean=0.72,
    ecc_rate_per_1000h=40.0,
    xid_rate_per_1000h=0.12,
    latent_defect_rate=0.06,
    weibull_shape=1.5,
    baseline_scale_days=2800.0,
    base_clock_mhz=1410.0,
)

H100_SXM4_80GB = Archetype(
    model_name="H100-SXM4-80GB",
    fleet_share=0.5,
    age_months_min=12.0,
    age_months_max=42.0,
    duty_cycle_mean=0.70,
    thermal_mean=0.60,
    power_mean=0.80,
    ecc_rate_per_1000h=30.0,
    xid_rate_per_1000h=0.15,
    latent_defect_rate=0.045,
    weibull_shape=1.4,
    baseline_scale_days=3200.0,
    base_clock_mhz=1980.0,
)

INFERENCE_PROFILE = WorkloadProfile(
    name="inference",
    duty_cycle=0.75,
    thermal_offset=0.02,
    power_offset=0.0,
    ecc_multiplier=1.0,
)

FINE_TUNE_PROFILE = WorkloadProfile(
    name="finetune",
    duty_cycle=0.90,
    thermal_offset=0.08,
    power_offset=0.10,
    ecc_multiplier=1.2,
)

MEMORY_STRESS_PROFILE = WorkloadProfile(
    name="memory-stress",
    duty_cycle=0.85,
    thermal_offset=0.04,
    power_offset=0.05,
    ecc_multiplier=1.8,
)

BUILTIN_ARCHETYPES: tuple[Archetype, ...] = (A100_SXM4_80GB, H100_SXM4_80GB)
BUILTIN_WORKLOAD_PROFILES: tuple[WorkloadProfile, ...] = (
    INFERENCE_PROFILE,
    FINE_TUNE_PROFILE,
    MEMORY_STRESS_PROFILE,
)

WORKLOAD_PROFILES: dict[str, WorkloadProfile] = {
    profile.name: profile for profile in BUILTIN_WORKLOAD_PROFILES
}

BUILTIN_WORKLOAD_MIX: tuple[WorkloadShare, ...] = (
    WorkloadShare(name=INFERENCE_PROFILE.name, share=0.5),
    WorkloadShare(name=FINE_TUNE_PROFILE.name, share=0.25),
    WorkloadShare(name=MEMORY_STRESS_PROFILE.name, share=0.25),
)


def default_fleet_config(seed: int, n_units: int = 500, horizon_days: int = 365) -> FleetConfig:
    """Return the built-in fleet config: A100/H100 mix, three-workload mix."""
    return FleetConfig(
        seed=seed,
        n_units=n_units,
        horizon_days=horizon_days,
        archetypes=BUILTIN_ARCHETYPES,
        workload_mix=BUILTIN_WORKLOAD_MIX,
    )


__all__ = [
    "A100_SXM4_80GB",
    "Archetype",
    "BUILTIN_ARCHETYPES",
    "BUILTIN_WORKLOAD_MIX",
    "BUILTIN_WORKLOAD_PROFILES",
    "FleetConfig",
    "FINE_TUNE_PROFILE",
    "H100_SXM4_80GB",
    "INFERENCE_PROFILE",
    "MEMORY_STRESS_PROFILE",
    "WorkloadProfile",
    "WorkloadShare",
    "WORKLOAD_PROFILES",
    "default_fleet_config",
]
