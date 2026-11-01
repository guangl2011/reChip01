"""Weibull proportional-hazards model tests: likelihood, fit, intervals (spec §4)."""

from __future__ import annotations

import numpy as np
import pytest
from numpy.typing import NDArray

from rechip01.engine.model import nll_and_gradient, weibull_cloglog
from tests.test_engine.helpers import synthetic_model

# Fixed synthetic-data parameters (deterministic, wide margins).
TRUE_BETA = np.array([-0.3, 0.8, -0.4])
TRUE_LOG_SHAPE = np.log(1.7)
N_SYNTHETIC = 4000
SYNTH_SEED = 4242


def _synthetic_survival_data(
    n: int = N_SYNTHETIC,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.bool_]]:
    """Draw uncensored Weibull PH data from known parameters (fixed seed)."""
    rng = np.random.default_rng(SYNTH_SEED)
    design = np.hstack([np.ones((n, 1)), rng.normal(size=(n, 2))])
    linear = design @ TRUE_BETA
    scale = np.exp(-linear)
    times = scale * rng.weibull(np.exp(TRUE_LOG_SHAPE), size=n)
    return design, np.log(times), np.ones(n, dtype=np.bool_)


def test_gradient_matches_finite_difference() -> None:
    """Analytic gradient agrees with central differences of the NLL."""
    design, log_time, event = _synthetic_survival_data(n=200)
    rng = np.random.default_rng(7)
    theta = rng.normal(scale=0.3, size=design.shape[1] + 1)

    _, analytic = nll_and_gradient(theta, design, log_time, event)
    analytic = np.asarray(analytic)

    numeric = np.zeros_like(theta)
    for column in range(theta.shape[0]):
        step = 1e-5 * max(abs(theta[column]), 1.0)
        up, down = theta.copy(), theta.copy()
        up[column] += step
        down[column] -= step
        nll_up, _ = nll_and_gradient(up, design, log_time, event)
        nll_down, _ = nll_and_gradient(down, design, log_time, event)
        numeric[column] = (nll_up - nll_down) / (2.0 * step)

    relative = np.abs(analytic - numeric) / (np.abs(numeric) + 1e-8)
    assert relative.max() < 1e-5, f"gradient mismatch: analytic={analytic} numeric={numeric}"


def test_fit_recovers_parameters_uncensored() -> None:
    """MLE recovers the generating coefficients and shape on uncensored data."""
    from rechip01.engine.model import WeibullPHModel

    design, log_time, event = _synthetic_survival_data()
    model = WeibullPHModel.fit(design, log_time, event)
    assert np.allclose(model.coef, TRUE_BETA, atol=0.12)
    assert model.shape == pytest.approx(np.exp(TRUE_LOG_SHAPE), rel=0.10)


def test_censored_fit_recovers_shape() -> None:
    """Right-censoring at a horizon keeps the fit consistent (shape within 15%)."""
    from rechip01.engine.model import WeibullPHModel

    design, log_time, _ = _synthetic_survival_data()
    times = np.exp(log_time)
    horizon = float(np.percentile(times, 70))  # ~30% censored
    event = times <= horizon
    assert event.mean() > 0.5  # sanity: enough events to fit
    log_time_censored = np.log(np.minimum(times, horizon))

    model = WeibullPHModel.fit(design, log_time_censored, event)
    assert model.shape == pytest.approx(np.exp(TRUE_LOG_SHAPE), rel=0.15)


def test_fit_raises_with_zero_events() -> None:
    from rechip01.engine.model import WeibullPHModel

    design = np.ones((10, 1))
    log_time = np.log(np.linspace(1.0, 5.0, 10))
    event = np.zeros(10, dtype=np.bool_)
    with pytest.raises(ValueError, match="no observed failures"):
        WeibullPHModel.fit(design, log_time, event)


def test_p_fail_monotone_in_linear_predictor_and_horizon() -> None:
    # Small shape and a short horizon keep p_fail inside (0, 1) — no saturation,
    # so strict ordering is actually testable.
    model = synthetic_model(np.array([0.0, 1.0]), log_shape=-0.5)
    rows = np.asarray([[1.0, v] for v in np.linspace(-2.0, 2.0, 9)])
    probabilities = [float(model.p_fail(rows[i], 1)[0]) for i in range(rows.shape[0])]
    # Pairwise comparison: intentionally uneven lengths, so strict is False.
    assert all(lo < hi for lo, hi in zip(probabilities, probabilities[1:], strict=False))

    at_one_day = float(model.p_fail(np.asarray([[1.0, 0.5]]), 1)[0])
    at_thirty_days = float(model.p_fail(np.asarray([[1.0, 0.5]]), 30)[0])
    assert at_thirty_days > at_one_day


def test_weibull_cloglog_matches_closed_form() -> None:
    """The cloglog transform equals 1 - exp(-exp(x b) t^k) exactly."""
    linear = np.asarray([-1.5, 0.0, 2.0])
    log_time = np.log(400.0)
    shape = 1.3
    expected = 1.0 - np.exp(-np.exp(linear) * np.exp(log_time) ** shape)
    np.testing.assert_allclose(weibull_cloglog(linear, log_time, shape), expected)


def test_interval_brackets_point_and_widens() -> None:
    # Coefficients place the row mid-curve (p_fail ~ 0.75) so endpoints are
    # distinct — saturated models clamp every endpoint to 1.0.
    model = synthetic_model(np.array([-6.0, 0.6, -0.3]), log_shape=float(np.log(1.3)))
    row = np.asarray([1.0, 0.7, -0.2])

    low_base, p_base, high_base = model.interval(row, 365, ci_level=0.90, se_widening=1.0)
    assert low_base < p_base < high_base

    low_wide, p_wide, high_wide = model.interval(row, 365, ci_level=0.90, se_widening=2.0)
    assert p_wide == pytest.approx(p_base)
    assert low_wide < low_base and high_wide > high_base  # widening only stretches

    low_99, _, high_99 = model.interval(row, 365, ci_level=0.99, se_widening=1.0)
    assert low_99 <= low_base and high_99 >= high_base  # higher level -> wider


def test_interval_rejects_invalid_arguments() -> None:
    model = synthetic_model(np.array([0.0, 1.0]), log_shape=0.5)
    row = np.asarray([1.0, 0.2])
    with pytest.raises(ValueError, match="ci_level"):
        model.interval(row, 365, ci_level=1.2, se_widening=1.0)
    with pytest.raises(ValueError, match="se_widening"):
        model.interval(row, 365, ci_level=0.9, se_widening=0.5)
