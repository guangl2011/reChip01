"""Reusable Hypothesis strategies for engine tests."""

from __future__ import annotations

from typing import Any

import numpy as np
from hypothesis import strategies as st
from hypothesis.strategies import SearchStrategy, composite


@composite
def survival_sets(
    draw: Any, min_n: int = 150, max_n: int = 400, min_events: int = 40
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """(design, log_time, event, true_linear) draws from a random Weibull PH.

    Design includes the intercept column. At least one slope coefficient has
    |beta| >= 0.3 so the model has signal to find; right-censoring is a fixed
    per-draw horizon chosen so the event count always clears min_events.
    """
    n = draw(st.integers(min_n, max_n))
    n_covariates = draw(st.integers(1, 2))
    seed = draw(st.integers(0, 2**31 - 1))
    shape = draw(st.floats(0.8, 3.0))
    censor_frac = draw(st.floats(0.0, 0.30))

    rng = np.random.default_rng(seed)
    slopes = []
    for _ in range(n_covariates):
        beta = draw(st.floats(-1.5, 1.5))
        if abs(beta) < 0.3:
            beta = 0.3 if beta >= 0 else -0.3
        slopes.append(beta)
    beta = np.asarray([draw(st.floats(-0.5, 0.5)), *slopes])

    design = np.hstack([np.ones((n, 1)), rng.normal(size=(n, n_covariates))])
    linear = design @ beta
    scale = np.exp(-linear)
    times = scale * (-np.log(rng.uniform(size=n))) ** (1.0 / shape)

    horizon = float(np.quantile(times, 1.0 - censor_frac)) if censor_frac > 0 else np.inf
    event = times <= horizon
    log_time = np.log(np.minimum(times, horizon))
    return design, log_time, event, linear


def small_floats() -> SearchStrategy[float]:
    return st.floats(-3.0, 3.0, allow_nan=False, allow_infinity=False)
