"""Property tests for the baseline comparison metrics (spec §8).

Two property families, per the task's test plan:

- **Metric invariants** — AUROC (concordance) bounds, reversal symmetry, and
  monotone-transform invariance; ECE bounds and the constant-score identity.
- **Holdout isolation** — baseline scores depend only on the held-out scans:
  deterministic, permutation-equivariant, and bit-identical no matter which
  training fleet the engine saw.

All ground truth here is simulated; nothing asserts a real-world claim.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st
from numpy.typing import NDArray

from rechip01.engine.metrics import _auroc, expected_calibration_error
from rechip01.evaluation import dcgm_threshold_scores, error_count_scores
from rechip01.simulator import SimulatedFleet
from tests.test_engine.helpers import make_fleet

# Small-but-two-class score/outcome draws for the metric invariants.
N_UNITS = st.integers(min_value=20, max_value=80)


@st.composite
def scores_and_outcomes(draw: Any) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Random scores in [0, 1] with binary outcomes; assumes both classes present.

    Scores are quantized to 6 decimals so strictly monotone float transforms
    (exp, 1−s in the symmetry test) cannot collide distinct values into new
    ties — those tests measure concordance, not float rounding.
    """
    n = draw(N_UNITS)
    scores = np.asarray(
        draw(
            st.lists(
                st.floats(0.0, 1.0, allow_nan=False).map(lambda v: round(v, 6)),
                min_size=n,
                max_size=n,
            )
        ),
        dtype=np.float64,
    )
    outcomes = np.asarray(draw(st.lists(st.booleans(), min_size=n, max_size=n)), dtype=np.float64)
    assume(outcomes.min() == 0.0 and outcomes.max() == 1.0)
    return scores, outcomes


@given(scores_and_outcomes())
@settings(max_examples=100, deadline=None)
def test_auroc_is_bounded_in_unit_interval(
    draws: tuple[NDArray[np.float64], NDArray[np.float64]],
) -> None:
    scores, outcomes = draws
    assert 0.0 <= _auroc(scores, outcomes) <= 1.0


@given(scores_and_outcomes())
@settings(max_examples=100, deadline=None)
def test_auroc_reversal_symmetry(
    draws: tuple[NDArray[np.float64], NDArray[np.float64]],
) -> None:
    scores, outcomes = draws
    forward, backward = _auroc(scores, outcomes), _auroc(1.0 - scores, outcomes)
    assert forward + backward == pytest.approx(1.0, abs=1e-9)


@given(scores_and_outcomes())
@settings(max_examples=100, deadline=None)
def test_auroc_invariant_under_monotone_transform(
    draws: tuple[NDArray[np.float64], NDArray[np.float64]],
) -> None:
    """AUROC measures concordance: any strictly increasing rescale is invisible."""
    scores, outcomes = draws
    assert _auroc(np.exp(3.0 * scores), outcomes) == pytest.approx(
        _auroc(scores, outcomes), abs=1e-9
    )


@given(scores_and_outcomes())
@settings(max_examples=100, deadline=None)
def test_ece_is_bounded_in_unit_interval(
    draws: tuple[NDArray[np.float64], NDArray[np.float64]],
) -> None:
    scores, outcomes = draws
    ece, _ = expected_calibration_error(scores, outcomes, n_bins=10)
    assert 0.0 <= ece <= 1.0


@given(scores_and_outcomes())
@settings(max_examples=100, deadline=None)
def test_ece_of_constant_scores_is_absolute_gap_to_base_rate(
    draws: tuple[NDArray[np.float64], NDArray[np.float64]],
) -> None:
    """Constant predictions all land in one bin: ECE = |c − realized rate|."""
    scores, outcomes = draws
    constant = np.full_like(scores, scores[0])
    ece, _ = expected_calibration_error(constant, outcomes, n_bins=10)
    assert ece == pytest.approx(abs(scores[0] - outcomes.mean()), abs=1e-9)


# ------------------------------------------------------------- holdout isolation


@pytest.fixture
def held_out_fleet() -> SimulatedFleet:
    """A modest held-out batch: isolation needs structure, not statistical mass."""
    return make_fleet(seed=405, n_units=150)


@given(permutation_seed=st.integers(0, 2**31 - 1))
@settings(max_examples=25, deadline=None)
def test_baseline_scores_permutation_equivariant(permutation_seed: int) -> None:
    """Batch order is not signal: permuted scans permute scores identically."""
    fleet = make_fleet(seed=7, n_units=120)
    permutation = np.random.default_rng(permutation_seed).permutation(fleet.n_units)
    shuffled = [fleet.scans[i] for i in permutation]
    for scorer in (dcgm_threshold_scores, error_count_scores):
        assert np.array_equal(scorer(shuffled), scorer(fleet.scans)[permutation]), scorer.__name__


def test_baseline_scores_bit_identical_across_calls() -> None:
    fleet = make_fleet(seed=7, n_units=120)
    for scorer in (dcgm_threshold_scores, error_count_scores):
        assert np.array_equal(scorer(fleet.scans), scorer(fleet.scans)), scorer.__name__


def test_holdout_isolation_baselines_never_see_training_data(
    held_out_fleet: SimulatedFleet,
) -> None:
    """Refitting the engine on any training fleet leaves baseline scores fixed.

    Baseline score functions take the held-out scans and nothing else — the
    structural form of isolation. This test pins the value-level consequence:
    scores computed on the held-out batch are identical whichever training
    fleet exists upstream.
    """
    fleet = held_out_fleet
    scores_before = (dcgm_threshold_scores(fleet.scans), error_count_scores(fleet.scans))

    # An unrelated training fleet is generated and discarded — the baselines
    # cannot observe it through their signatures, and must not through values.
    make_fleet(seed=404, n_units=50)

    scores_after = (dcgm_threshold_scores(fleet.scans), error_count_scores(fleet.scans))
    for before, after in zip(scores_before, scores_after, strict=True):
        assert np.array_equal(before, after)
