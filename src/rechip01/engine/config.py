"""Engine configuration contracts (spec art_vuIPvUhR, §4).

Calibration thresholds and OOD fences are configuration, tuned only against
the simulator's held-out ground truth — never against training performance
(TESTING.md calibration rule). The defaults below are deliberately
conservative: they widen reported uncertainty and flag more units for review
rather than fewer.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from rechip01.schemas import UnitInterval


class _FrozenContract(BaseModel):
    """Base for engine contracts: immutable after construction."""

    model_config = ConfigDict(frozen=True)


class EngineConfig(_FrozenContract):
    """Tunables of the reliability engine.

    ``ci_level`` sets the calibrated-interval coverage. The OOD fence is a
    Tukey-style interval on robustly standardized features: a unit is
    out-of-distribution when any feature leaves ``[q1 - m*IQR, q3 + m*IQR]``
    of the training distribution. ``missing_widening`` and ``ood_widening``
    inflate the standard error of the hazard for partial-mode and OOD units
    respectively — uncertainty widens instead of silently narrowing.
    """

    ci_level: UnitInterval = Field(
        default=0.90,
        description="Calibrated-interval coverage (a 0.90 level yields a 90% interval)",
    )
    ood_fence_multiplier: float = Field(
        default=4.0,
        gt=0,
        description="IQR multiplier for the out-of-distribution fence; lower flags more",
    )
    missing_widening: float = Field(
        default=1.5,
        ge=1.0,
        description="Standard-error multiplier applied when unit features are missing",
    )
    ood_widening: float = Field(
        default=2.0,
        ge=1.0,
        description="Standard-error multiplier applied to OOD-flagged units",
    )
    ridge_lambda: float = Field(
        default=0.01,
        gt=0,
        description=(
            "L2 penalty on slope coefficients. The training workload block takes "
            "the three built-in profiles, so its columns are exactly collinear "
            "with the intercept; the ridge makes the fit identifiable without "
            "measurably shrinking well-estimated directions."
        ),
    )


__all__ = ["EngineConfig"]
