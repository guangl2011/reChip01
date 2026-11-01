"""Evaluation-metric tests on hand-computed toy cases (spec §8)."""

from __future__ import annotations

import numpy as np
import pytest

from rechip01.engine.engine import MODEL_VERSION
from rechip01.engine.metrics import (
    _auroc,
    bootstrap_auroc_ci,
    evaluate,
    expected_calibration_error,
    render_pipeline_report,
)
from rechip01.schemas import Provenance, ReliabilityEstimate
from rechip01.simulator import GroundTruthLabel


def _label(asset_id: str, failed: bool, truth: float, horizon: int = 365) -> GroundTruthLabel:
    return GroundTruthLabel(
        asset_id=asset_id,
        provenance=Provenance.SIMULATED,
        assigned_workload="inference",
        horizon_days=horizon,
        has_latent_defect=False,
        failed_within_horizon=failed,
        time_to_failure_days=100.0 if failed else None,
        p_fail_assigned_workload=truth,
        p_fail_by_workload={"inference": truth},
    )


def _estimate(asset_id: str, p: float) -> ReliabilityEstimate:
    low = max(0.0, p - 0.1)
    high = min(1.0, p + 0.1)
    return ReliabilityEstimate(
        asset_id=asset_id,
        horizon_days=365,
        provenance=Provenance.SIMULATED,
        p_fail=p,
        ci=(low, high),
        workload_fitness={"inference": 1.0 - p},
        ood_flag=False,
        model_version="test-0.0.0",
    )


def test_auroc_matches_hand_computation() -> None:
    scores = np.asarray([0.1, 0.4, 0.35, 0.8])
    outcomes = np.asarray([0.0, 0.0, 1.0, 1.0])
    # Pairs: (0.35>0.1) win, (0.35>0.4) loss, (0.8>0.1) win, (0.8>0.4) win -> 3/4.
    assert _auroc(scores, outcomes) == pytest.approx(0.75)


def test_auroc_perfect_separation() -> None:
    scores = np.asarray([0.1, 0.2, 0.8, 0.9])
    outcomes = np.asarray([0.0, 0.0, 1.0, 1.0])
    assert _auroc(scores, outcomes) == pytest.approx(1.0)


def test_bootstrap_ci_deterministic_and_contains_point() -> None:
    rng = np.random.default_rng(99)
    scores = np.concatenate([rng.normal(0.0, 1.0, 200), rng.normal(2.0, 1.0, 200)])
    outcomes = np.concatenate([np.zeros(200), np.ones(200)])

    low_a, high_a = bootstrap_auroc_ci(scores, outcomes, n_bootstrap=200, seed=17)
    low_b, high_b = bootstrap_auroc_ci(scores, outcomes, n_bootstrap=200, seed=17)
    assert (low_a, high_a) == (low_b, high_b)  # seeded runs are identical
    assert 0.0 <= low_a <= high_a <= 1.0
    assert low_a <= _auroc(scores, outcomes) <= high_a  # point estimate inside the CI


def test_ece_zero_when_predictions_match_empirical() -> None:
    predicted = np.full(1000, 0.3)
    outcomes = np.concatenate([np.ones(300), np.zeros(700)])
    ece, bins = expected_calibration_error(predicted, outcomes, n_bins=10)
    assert ece == pytest.approx(0.0, abs=1e-12)
    populated = [b for b in bins if b.count > 0]
    assert len(populated) == 1
    assert populated[0].empirical_failure_rate == pytest.approx(0.3)


def test_ece_counts_all_rows_including_p_one() -> None:
    predicted = np.asarray([0.05, 0.5, 0.95, 1.0])
    outcomes = np.asarray([0.0, 0.0, 1.0, 1.0])
    _, bins = expected_calibration_error(predicted, outcomes, n_bins=10)
    assert sum(b.count for b in bins) == 4
    # 0.95 and 1.0 both land in the final bin via the min() clamp.
    last = bins[-1]
    assert last.count == 2 and last.lower == pytest.approx(0.9)


def test_ece_known_miscalibration() -> None:
    predicted = np.full(100, 0.9)
    outcomes = np.concatenate([np.ones(30), np.zeros(70)])
    ece, _ = expected_calibration_error(predicted, outcomes, n_bins=10)
    assert ece == pytest.approx(0.6)


def test_ece_rejects_bad_bin_count() -> None:
    with pytest.raises(ValueError, match="n_bins"):
        expected_calibration_error(np.asarray([0.5]), np.asarray([1.0]), n_bins=0)


def test_evaluate_rejects_unlabeled_estimates() -> None:
    estimates = [_estimate("A", 0.3), _estimate("B", 0.6)]
    labels = {"A": _label("A", failed=True, truth=0.4)}
    with pytest.raises(ValueError, match="ground-truth labels"):
        evaluate(estimates, labels)


def test_evaluate_rejects_empty() -> None:
    with pytest.raises(ValueError, match="at least one estimate"):
        evaluate([], {})


def test_evaluate_and_report_on_toy_data() -> None:
    """Toy case with known separation: metrics computed and report honest."""
    scores = [0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95]
    outcomes = [False, False, False, False, True, False, True, True, True, True]
    estimates = [_estimate(f"U{i:02d}", p) for i, p in enumerate(scores)]
    labels = {
        f"U{i:02d}": _label(f"U{i:02d}", failed=o, truth=p)
        for i, (p, o) in enumerate(zip(scores, outcomes, strict=True))
    }

    evaluation = evaluate(estimates, labels, n_bootstrap=100, bootstrap_seed=3)

    assert evaluation.n_units == 10
    assert evaluation.failure_count == 5
    assert evaluation.auroc > 0.5
    assert evaluation.auroc_ci_low <= evaluation.auroc <= evaluation.auroc_ci_high
    assert 0.0 <= evaluation.ece <= 1.0
    assert evaluation.mae_vs_generator_truth == pytest.approx(
        np.mean([abs(e.p_fail - labels[e.asset_id].p_fail_assigned_workload) for e in estimates])
    )
    assert evaluation.ood_flagged == 0
    assert evaluation.model_version == MODEL_VERSION  # stamped by the engine, not the estimate

    report = render_pipeline_report(evaluation)
    assert "SIMULATION-BASED PIPELINE CHECK" in report
    assert "Not a real-world reliability result" in report
    assert "not an H1/H2/H3 outcome" in report
    assert "AUROC" in report
    assert "Expected calibration error" in report
    assert "Calibration table" in report
    assert evaluation.model_version in report
