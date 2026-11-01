"""Out-of-distribution detection: flag units outside training support (spec §4).

The detector is deliberately interpretable: a per-feature Tukey fence on the
training distribution — ``[q1 - m*IQR, q3 + m*IQR]`` — evaluated on every
feature the model consumes, workload features included. A unit is OOD when any
feature leaves its fence; the flagged feature names ride on the result so the
review path (and the passport, later) can say *why* a unit was flagged.

Fence multipliers are ``EngineConfig.ood_fence_multiplier``; the default is
conservative (flags more units rather than fewer). A feature that is constant
in training gets a zero-width fence: any deviation from the constant is OOD,
which is the honest reading of "outside training support".
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class OODVerdict:
    """One unit's OOD outcome: the flag plus the features that triggered it."""

    ood: bool
    triggered_features: tuple[str, ...]


@dataclass(frozen=True)
class OODDetector:
    """Tukey-fence OOD detector fitted on the training feature matrix."""

    feature_names: tuple[str, ...]
    lower: NDArray[np.float64]  # per-feature fence lower bound
    upper: NDArray[np.float64]  # per-feature fence upper bound

    @classmethod
    def fit(
        cls, matrix: NDArray[np.float64], feature_names: tuple[str, ...], multiplier: float
    ) -> OODDetector:
        """Fit per-feature fences from training rows.

        ``multiplier`` is the IQR multiplier ``m`` in the fence definition.
        """
        if matrix.shape[1] != len(feature_names):
            raise ValueError(
                f"feature matrix width {matrix.shape[1]} does not match "
                f"{len(feature_names)} feature names"
            )
        q1 = np.percentile(matrix, 25, axis=0)
        q3 = np.percentile(matrix, 75, axis=0)
        iqr = q3 - q1
        return cls(
            feature_names=feature_names,
            lower=q1 - multiplier * iqr,
            upper=q3 + multiplier * iqr,
        )

    def flag(self, row: NDArray[np.float64]) -> OODVerdict:
        """Flag one feature row: OOD iff any feature leaves its fence."""
        if row.shape[0] != len(self.feature_names):
            raise ValueError(
                f"feature row width {row.shape[0]} does not match "
                f"{len(self.feature_names)} feature names"
            )
        outside = (row < self.lower) | (row > self.upper)
        triggered = tuple(self.feature_names[column] for column in np.flatnonzero(outside))
        return OODVerdict(ood=bool(triggered), triggered_features=triggered)

    def flag_matrix(self, matrix: NDArray[np.float64]) -> list[OODVerdict]:
        """Flag every row of a feature matrix, order-preserving."""
        return [self.flag(matrix[row]) for row in range(matrix.shape[0])]


__all__ = ["OODDetector", "OODVerdict"]
