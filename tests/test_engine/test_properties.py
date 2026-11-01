"""Hypothesis property tests for the reliability engine (spec §8).

Properties pinned here:
1. Calibration: on clean synthetic draws the fitted model's risk ordering
   correlates with the generator's true risk ordering.
2. Uncertainty ordering: intervals bracket the point, higher CI levels are
   supersets, widening stretches without moving the point.
3. The OOD flag never scores silently: a flagged estimate widens (never
   shrinks) uncertainty at a fixed point, an unflagged estimate scores
   unwidened, and either way the estimate stays valid and bracketed.
   Tukey fences are data-dependent — log1p compresses count features, so a
   mildly scaled probe can legitimately stay in-distribution — so the flag
   premise is asserted conditionally, never assumed.
4. Determinism: a fixed seed reproduces the fleet and assessment exactly.

Tests 2–4 share one fitted engine (module scope): the properties concern the
assessment contract, not refit stability, and a 20-parameter survival fit on
a 25-unit fleet is statistically fragile. Refit determinism on an adequately
sized fleet is pinned separately in test_assess.py.
"""

from __future__ import annotations

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st

from rechip01.engine import EngineConfig, ReliabilityEngine
from rechip01.engine.model import WeibullPHModel
from rechip01.schemas import Provenance, ReliabilityEstimate
from rechip01.simulator import (
    WORKLOAD_PROFILES,
    default_fleet_config,
    generate_fleet,
)
from tests.test_engine.helpers import (
    degraded_twin,
    synthetic_model,
    with_measured_provenance,
    with_missing_fields,
)
from tests.test_engine.strategies import survival_sets

WORKLOADS = [WORKLOAD_PROFILES[name] for name in ("inference", "finetune", "memory-stress")]
PRIMARY = WORKLOADS[:1]

# Shared fixtures: one adequately sized fleet, two engines (default widening;
# widening disabled). Fits are deterministic, so both engines share identical
# coefficients and differ only in interval widening.
_SHARED_FLEET = generate_fleet(default_fleet_config(seed=4242, n_units=150))
_ENGINE = ReliabilityEngine(EngineConfig()).fit(_SHARED_FLEET)
_ENGINE_NO_WIDENING = ReliabilityEngine(EngineConfig(missing_widening=1.0, ood_widening=1.0)).fit(
    _SHARED_FLEET
)


# 1. Calibration on clean synthetic draws -------------------------


@given(survival_sets())
@settings(max_examples=10, deadline=None)
def test_mle_risk_ordering_tracks_generator(
    data: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
) -> None:
    design, log_time, event, true_linear = data
    model = WeibullPHModel.fit(design, log_time, event)
    fitted_linear = design @ model.coef
    correlation = float(np.corrcoef(fitted_linear, true_linear)[0, 1])
    assert correlation > 0.5, f"fitted risk ordering lost the generator's (r={correlation:.3f})"
    assert 0.2 <= model.shape <= 6.0  # shape stays in a sane range


# 2. Uncertainty ordering -----------------------------------------


@given(st.floats(-1.5, 1.5, allow_nan=False), st.floats(1.0, 2.5, allow_nan=False))
@settings(max_examples=25, deadline=None)
def test_interval_brackets_and_orders(beta_shift: float, se_scale: float) -> None:
    model = synthetic_model(np.array([0.0, 1.0]), log_shape=0.4)
    row = np.asarray([1.0, beta_shift])

    low90, point, high90 = model.interval(row, 365, ci_level=0.90, se_widening=1.0)
    low99, _, high99 = model.interval(row, 365, ci_level=0.99, se_widening=1.0)
    low_wide, point_wide, high_wide = model.interval(row, 365, ci_level=0.90, se_widening=se_scale)

    assert low90 <= point <= high90
    assert low99 <= low90 and high99 >= high90  # higher level is a superset
    assert point_wide == point  # widening never moves the point
    assert low_wide <= point <= high_wide


# 3. Flagged units never score silently ---------------------------


@given(
    st.floats(min_value=6.0, max_value=80.0, allow_nan=False),
    st.integers(min_value=0, max_value=149),
)
@settings(max_examples=15, deadline=None)
def test_flagged_units_carry_valid_estimates(extreme_ecc: float, unit_index: int) -> None:
    """Degraded twins score validly, and any flag decision is explicit.

    Tukey fences are data-dependent: log1p compresses the count features, so
    a mildly scaled twin can legitimately stay in-distribution (the default
    scale=60 twin is pinned to flag in test_assess.py). The engine contract
    is the conditional — flagged units score only with a widened interval,
    unflagged units score with the unwidened interval, and either way the
    estimate is valid and bracketed.
    """
    extreme = degraded_twin(_SHARED_FLEET.scans[unit_index], ecc_scale=extreme_ecc)
    estimate = _ENGINE.assess([extreme], horizon_days=365, target_workloads=PRIMARY)[0]
    unwidened = _ENGINE_NO_WIDENING.assess([extreme], horizon_days=365, target_workloads=PRIMARY)[0]

    assert isinstance(estimate, ReliabilityEstimate)
    assert 0.0 <= estimate.p_fail <= 1.0
    assert estimate.ci[0] <= estimate.p_fail <= estimate.ci[1]
    assert estimate.p_fail == unwidened.p_fail  # widening never moves the point
    if estimate.ood_flag:
        assert estimate.ci[0] <= unwidened.ci[0] and estimate.ci[1] >= unwidened.ci[1]
    else:
        assert estimate.ci == unwidened.ci  # unflagged: no silent widening


@given(st.integers(min_value=1, max_value=3), st.integers(min_value=0, max_value=149))
@settings(max_examples=10, deadline=None)
def test_missing_inputs_widen_never_shrink(absent: int, unit_index: int) -> None:
    """Widening multiplies the hazard SE: same point, strictly wider interval.

    The p-space width itself can legitimately shrink when imputation moves the
    point into a flatter region of the cloglog curve — the invariant that must
    never break is the standard-error inflation at a fixed point.
    """
    scan = _SHARED_FLEET.scans[unit_index]
    degraded = with_missing_fields(scan, absent_benchmarks=absent)
    clean, partial = _ENGINE.assess([scan, degraded], horizon_days=365, target_workloads=PRIMARY)
    tight = _ENGINE_NO_WIDENING.assess([degraded], horizon_days=365, target_workloads=PRIMARY)[0]

    assert _ENGINE.missing_flags([degraded]) == [True]
    assert partial.p_fail == tight.p_fail  # widening never moves the point
    assert partial.ci[0] <= tight.ci[0]
    assert partial.ci[1] >= tight.ci[1]
    assert 0.0 <= clean.p_fail <= 1.0


# 4. Provenance propagation ---------------------------------------


@given(st.integers(min_value=0, max_value=149), st.booleans())
@settings(max_examples=10, deadline=None)
def test_provenance_rides_through_assessment(unit_index: int, measured: bool) -> None:
    scan = _SHARED_FLEET.scans[unit_index]
    if measured:
        scan = with_measured_provenance(scan)
    estimate = _ENGINE.assess([scan], horizon_days=365, target_workloads=PRIMARY)[0]
    assert estimate.provenance == (Provenance.MEASURED if measured else Provenance.SIMULATED)


# 5. Determinism ---------------------------------------------------


@given(st.integers(min_value=1, max_value=10_000))
@settings(max_examples=5, deadline=None)
def test_fleet_and_estimates_reproduce_from_seed(seed: int) -> None:
    """Same seed -> identical fleet; repeated assessment -> identical estimates."""
    fleet_a = generate_fleet(default_fleet_config(seed=seed, n_units=30))
    fleet_b = generate_fleet(default_fleet_config(seed=seed, n_units=30))
    assert fleet_a.scans == fleet_b.scans

    estimates_a = _ENGINE.assess(fleet_a.scans[:10], horizon_days=180, target_workloads=PRIMARY)
    estimates_b = _ENGINE.assess(fleet_a.scans[:10], horizon_days=180, target_workloads=PRIMARY)
    assert [e.model_dump_json() for e in estimates_a] == [e.model_dump_json() for e in estimates_b]
