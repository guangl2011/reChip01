"""Reliability engine facade: fit on simulated fleets, assess real or simulated
scans (spec art_vuIPvUhR, §4–5).

The engine is workload-conditioned: a unit's failure probability is scored
under a *target* workload's operating profile, learned from each training
unit's assigned second-life workload. Every emitted
:class:`~rechip01.schemas.ReliabilityEstimate` carries the input scan's
provenance, the model version, a calibrated interval that brackets ``p_fail``,
and an OOD flag for units outside training support — OOD units are scored only
together with that flag, never silently.

Honesty invariants (spec §8): the engine trains on simulator data in the MVP;
every metric computed from it is a simulation-based pipeline check. The engine
itself makes no real-world reliability claim.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from typing import Final

import numpy as np
from numpy.typing import NDArray

from rechip01.engine.config import EngineConfig
from rechip01.engine.features import (
    FEATURE_NAMES,
    UNIT_FEATURE_NAMES,
    WORKLOAD_FEATURE_NAMES,
    WORKLOAD_SLICE,
    FeatureSpace,
    Standardizer,
    workload_feature_row,
)
from rechip01.engine.model import WeibullPHModel
from rechip01.engine.ood import OODDetector
from rechip01.schemas import Provenance, ReliabilityEstimate, ScanResult
from rechip01.simulator.config import WORKLOAD_PROFILES, WorkloadProfile
from rechip01.simulator.generate import GroundTruthLabel, SimulatedFleet

_LOGGER = logging.getLogger(__name__)

# Version stamped on every estimate and passport; bump on behavioral changes
# to the feature set, the likelihood, or the interval construction.
MODEL_VERSION: Final[str] = "rechip01-reliability-0.1.0+weibull-ph"


class ReliabilityEngine:
    """Uncertainty-aware, workload-conditioned reliability model (spec §4).

    Fit on a :class:`~rechip01.simulator.generate.SimulatedFleet` (scans plus
    held-out labels), then score arbitrary scans against declared target
    workloads. In the MVP the training data is simulated; the same interface
    accepts measured training labels when Phase I produces them.
    """

    def __init__(self, config: EngineConfig | None = None) -> None:
        self.config = config if config is not None else EngineConfig()
        self._feature_space: FeatureSpace | None = None
        self._standardizer: Standardizer | None = None
        self._model: WeibullPHModel | None = None
        self._unit_ood: OODDetector | None = None
        self._workload_ood: OODDetector | None = None
        self._training_fingerprint: str | None = None

    # ------------------------------------------------------------------ fit

    def fit(
        self,
        fleet: SimulatedFleet,
        *,
        workload_profiles: Mapping[str, WorkloadProfile] = WORKLOAD_PROFILES,
    ) -> ReliabilityEngine:
        """Fit features, model, and OOD detectors on one simulated fleet.

        Failures contribute their observed log failure times; surviving units
        are right-censored at their label horizon.
        """
        feature_space = FeatureSpace.fit(fleet.scans)
        unit = feature_space.transform(fleet.scans)

        workload_rows = np.asarray(
            [self._training_workload_row(label, workload_profiles) for label in fleet.labels],
            dtype=np.float64,
        )
        raw = np.hstack([unit.matrix, workload_rows])
        standardizer = Standardizer.fit(raw)
        design = _to_design(standardizer.transform(raw))

        event = np.asarray([label.failed_within_horizon for label in fleet.labels], dtype=np.bool_)
        observed: list[float] = []
        for label in fleet.labels:
            time = label.time_to_failure_days if label.failed_within_horizon else label.horizon_days
            if time is None:
                raise ValueError(f"label for {label.asset_id!r} has no observed or censoring time")
            observed.append(float(np.log(time)))
        log_time = np.asarray(observed, dtype=np.float64)
        model = WeibullPHModel.fit(design, log_time, event, ridge_lambda=self.config.ridge_lambda)

        unit_ood = OODDetector.fit(
            unit.matrix, UNIT_FEATURE_NAMES, self.config.ood_fence_multiplier
        )
        workload_ood = OODDetector.fit(
            raw[:, WORKLOAD_SLICE], WORKLOAD_FEATURE_NAMES, self.config.ood_fence_multiplier
        )

        self._feature_space = feature_space
        self._standardizer = standardizer
        self._model = model
        self._unit_ood = unit_ood
        self._workload_ood = workload_ood
        self._training_fingerprint = (
            f"seed={fleet.config.seed};n={fleet.n_units};"
            f"horizon_days={fleet.config.horizon_days};events={int(event.sum())}"
        )

        _LOGGER.info(
            "engine: fitted — %s shape=%.3f version=%s",
            self._training_fingerprint,
            model.shape,
            MODEL_VERSION,
        )
        return self

    @staticmethod
    def _training_workload_row(
        label: GroundTruthLabel, workload_profiles: Mapping[str, WorkloadProfile]
    ) -> NDArray[np.float64]:
        """Resolve a training label's assigned workload to its feature row."""
        profile = workload_profiles.get(label.assigned_workload)
        if profile is None:
            raise ValueError(
                f"training label references unknown workload "
                f"{label.assigned_workload!r}; pass its WorkloadProfile via "
                f"workload_profiles"
            )
        return workload_feature_row(profile)

    # --------------------------------------------------------------- assess

    def assess(
        self,
        scans: Sequence[ScanResult],
        *,
        horizon_days: int,
        target_workloads: Sequence[WorkloadProfile],
    ) -> list[ReliabilityEstimate]:
        """Score scans for one horizon under declared target workloads.

        ``p_fail`` and the calibrated interval refer to ``target_workloads[0]``
        — the primary deployment intent; ``workload_fitness`` covers every
        requested workload as modeled survival probability (1 − P(fail)). An
        estimate is OOD-flagged when the unit's features or *any* requested
        workload fall outside training support — flagged units are scored only
        together with the flag, never silently.
        """
        model, standardizer, feature_space, unit_ood, workload_ood = self._fitted_state()
        if not target_workloads:
            raise ValueError("assess requires at least one target workload")
        if horizon_days < 1:
            raise ValueError(f"horizon_days must be >= 1, got {horizon_days}")
        if not scans:
            return []

        unit = feature_space.transform(list(scans))
        unit_ood_flags = unit_ood.flag_matrix(unit.matrix)

        workload_rows = np.asarray(
            [workload_feature_row(profile) for profile in target_workloads],
            dtype=np.float64,
        )
        workload_ood_verdicts = [workload_ood.flag(row) for row in workload_rows]
        any_workload_ood = any(verdict.ood for verdict in workload_ood_verdicts)

        # Per-workload design matrices: (n_workloads, n_units, 1 + n_features).
        designs = [
            _to_design(
                standardizer.transform(
                    np.hstack([unit.matrix, np.tile(workload_row, (len(scans), 1))])
                )
            )
            for workload_row in workload_rows
        ]

        estimates: list[ReliabilityEstimate] = []
        for row_index, scan in enumerate(scans):
            ood_flag = unit_ood_flags[row_index].ood or any_workload_ood
            widening = 1.0
            if unit.missing[row_index]:
                widening *= self.config.missing_widening
            if ood_flag:
                widening *= self.config.ood_widening

            # The primary workload carries the headline probability; its
            # fitness entry is kept exactly consistent (fitness = 1 - p_fail).
            low, p_fail, high = model.interval(
                designs[0][row_index], horizon_days, self.config.ci_level, widening
            )
            fitness = {
                profile.name: 1.0 - float(model.p_fail(designs[column][row_index], horizon_days)[0])
                for column, profile in enumerate(target_workloads)
            }
            fitness[target_workloads[0].name] = 1.0 - p_fail

            estimates.append(
                ReliabilityEstimate(
                    asset_id=scan.asset_id,
                    horizon_days=horizon_days,
                    provenance=Provenance(scan.provenance),
                    p_fail=p_fail,
                    ci=(low, high),
                    workload_fitness=fitness,
                    ood_flag=ood_flag,
                    model_version=MODEL_VERSION,
                )
            )

        flagged = sum(estimate.ood_flag for estimate in estimates)
        _LOGGER.info(
            "engine: assessed %d units at horizon=%dd — ood_flagged=%d version=%s",
            len(estimates),
            horizon_days,
            flagged,
            MODEL_VERSION,
        )
        return estimates

    def missing_flags(self, scans: Sequence[ScanResult]) -> list[bool]:
        """Per-scan partial-mode flags (rows where any feature was imputed)."""
        feature_space = self._feature_space
        if feature_space is None:
            raise RuntimeError("ReliabilityEngine must be fitted before missing_flags")
        if not scans:
            return []
        unit = feature_space.transform(list(scans))
        return [bool(flag) for flag in unit.missing]

    def training_fingerprint(self) -> str:
        """Lineage string of the training run (seed, n, horizon, event count)."""
        if self._training_fingerprint is None:
            raise RuntimeError("engine has no training fingerprint before fit")
        return self._training_fingerprint

    def _fitted_state(
        self,
    ) -> tuple[WeibullPHModel, Standardizer, FeatureSpace, OODDetector, OODDetector]:
        """Return fitted components or raise — no silent unfitted scoring."""
        if (
            self._model is None
            or self._standardizer is None
            or self._feature_space is None
            or self._unit_ood is None
            or self._workload_ood is None
        ):
            raise RuntimeError("ReliabilityEngine must be fitted before assess")
        return (
            self._model,
            self._standardizer,
            self._feature_space,
            self._unit_ood,
            self._workload_ood,
        )


def _to_design(standardized: NDArray[np.float64]) -> NDArray[np.float64]:
    """Prepend the intercept column to a standardized feature block."""
    intercept = np.ones((standardized.shape[0], 1), dtype=np.float64)
    return np.hstack([intercept, standardized])


# Feature names the engine consumes — re-exported for leakage-audit tests.
ENGINE_FEATURE_NAMES: Final[tuple[str, ...]] = FEATURE_NAMES


__all__ = [
    "ENGINE_FEATURE_NAMES",
    "MODEL_VERSION",
    "ReliabilityEngine",
]
