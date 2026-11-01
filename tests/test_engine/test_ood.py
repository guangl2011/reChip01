"""OOD detector tests: Tukey fences flag outliers with named features (spec §5)."""

from __future__ import annotations

import numpy as np
import pytest

from rechip01.engine.ood import OODDetector

NAMES = ("temp", "ecc_rate")


@pytest.fixture
def detector() -> OODDetector:
    """Fences fitted on a clean 2-feature training cloud."""
    rng = np.random.default_rng(11)
    matrix = rng.normal(size=(500, 2))
    return OODDetector.fit(matrix, NAMES, multiplier=4.0)


def test_inliers_pass(detector: OODDetector) -> None:
    verdict = detector.flag(np.asarray([0.0, 0.0]))
    assert not verdict.ood
    assert verdict.triggered_features == ()


def test_extreme_row_flags_with_named_features(detector: OODDetector) -> None:
    verdict = detector.flag(np.asarray([10.0, 0.0]))
    assert verdict.ood
    assert verdict.triggered_features == ("temp",)

    both = detector.flag(np.asarray([10.0, -10.0]))
    assert both.ood
    assert set(both.triggered_features) == {"temp", "ecc_rate"}


def test_flag_matrix_preserves_order(detector: OODDetector) -> None:
    matrix = np.asarray([[0.0, 0.0], [50.0, 0.0], [0.0, 0.0]])
    verdicts = detector.flag_matrix(matrix)
    assert [v.ood for v in verdicts] == [False, True, False]


def test_fit_rejects_width_mismatch() -> None:
    matrix = np.zeros((10, 3))
    with pytest.raises(ValueError, match="does not match"):
        OODDetector.fit(matrix, NAMES, multiplier=4.0)


def test_flag_rejects_width_mismatch(detector: OODDetector) -> None:
    with pytest.raises(ValueError, match="does not match"):
        detector.flag(np.zeros(5))


def test_zero_iqr_feature_flags_any_deviating_value() -> None:
    """A constant training feature fences at the constant itself: any deviation flags."""
    matrix = np.ones((20, 1))
    detector = OODDetector.fit(matrix, ("const",), multiplier=4.0)
    assert not detector.flag(np.asarray([1.0])).ood
    assert detector.flag(np.asarray([1.001])).ood
