"""Baseline comparison harness tests: unit, integration, CLI, report (spec §8).

The go/no-go rails for the engine itself live in ``test_evaluation.py``; this
suite covers the H1 scaffold — the comparison against the DCGM-threshold-style
grading rule and the error-count heuristic on identical held-out simulated
data. Every assertion here is a pipeline health check on simulated ground
truth, never a real-world reliability claim or an H1/H2/H3 outcome.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import numpy as np
import pytest

from rechip01.engine.engine import MODEL_VERSION, ReliabilityEngine
from rechip01.engine.harness import harness_eval_config, harness_train_config
from rechip01.engine.metrics import expected_calibration_error
from rechip01.evaluation import (
    DCGM_METHOD_NAME,
    ENGINE_METHOD_NAME,
    ERROR_COUNT_METHOD_NAME,
    BaselineComparison,
    MethodEvaluation,
    _evaluate_method,
    compare_baselines,
    dcgm_threshold_scores,
    error_count_scores,
    main,
    render_comparison_report,
)
from rechip01.schemas import Provenance, ScanResult
from rechip01.simulator import generate_fleet
from rechip01.simulator.generate import GroundTruthLabel
from tests.test_engine.helpers import EVAL_SEED, make_fleet

if TYPE_CHECKING:
    from pathlib import Path

ALL_METHOD_NAMES = {ENGINE_METHOD_NAME, DCGM_METHOD_NAME, ERROR_COUNT_METHOD_NAME}
WORKLOAD_NAMES = {"inference", "finetune", "memory-stress"}


@pytest.fixture(scope="session")
def comparison() -> BaselineComparison:
    """One full default-harness comparison shared by the integration tests."""
    train_fleet = generate_fleet(harness_train_config())
    eval_fleet = generate_fleet(harness_eval_config())
    engine = ReliabilityEngine().fit(train_fleet)
    return compare_baselines(eval_fleet, engine)


def method_by_name(comparison: BaselineComparison, name: str) -> MethodEvaluation:
    """Return one method's evaluation row by name."""
    return next(method for method in comparison.methods if method.name == name)


# ---------------------------------------------------------------- integration


def test_comparison_runs_end_to_end(comparison: BaselineComparison) -> None:
    assert {method.name for method in comparison.methods} == ALL_METHOD_NAMES
    assert comparison.n_units == 2000
    assert comparison.eval_seed == EVAL_SEED
    assert comparison.horizon_days == 365
    assert 0 < comparison.failure_count < comparison.n_units
    assert comparison.train_fingerprint.startswith("seed=101;")


def test_all_metrics_within_unit_interval(comparison: BaselineComparison) -> None:
    for method in comparison.methods:
        assert 0.0 <= method.auroc <= 1.0, method.name
        assert 0.0 <= method.auroc_ci_low <= method.auroc_ci_high <= 1.0, method.name
        assert 0.0 <= method.auroc_above_half_p <= 1.0, method.name
        assert 0.0 <= method.ece <= 1.0, method.name
        for workload, workload_ece in method.ece_by_workload.items():
            assert 0.0 <= workload_ece <= 1.0, (method.name, workload)
        assert len(method.calibration_bins) > 0, method.name


def test_all_methods_score_the_identical_batch(comparison: BaselineComparison) -> None:
    counts = {(method.n_units, method.failure_count) for method in comparison.methods}
    assert len(counts) == 1, "methods were not evaluated on the same held-out batch"


def test_engine_row_metadata(comparison: BaselineComparison) -> None:
    engine_row = method_by_name(comparison, ENGINE_METHOD_NAME)
    assert engine_row.model_version == MODEL_VERSION
    assert set(engine_row.ece_by_workload) == WORKLOAD_NAMES
    for name in (DCGM_METHOD_NAME, ERROR_COUNT_METHOD_NAME):
        row = method_by_name(comparison, name)
        assert row.model_version is None, name
        assert set(row.ece_by_workload) == WORKLOAD_NAMES, name


def test_holdout_isolation_baselines_untouched_by_training_fleet(
    comparison: BaselineComparison,
) -> None:
    """Baseline rows are identical when the engine's training fleet changes.

    Baseline scores depend on the held-out scans alone, so refitting the engine
    on a different training fleet must leave every baseline metric bit-identical
    (same scores, same bootstrap seed and size) while the engine row moves.
    """
    other_train_fleet = make_fleet(seed=303, n_units=400)
    other_engine = ReliabilityEngine().fit(other_train_fleet)
    eval_fleet = generate_fleet(harness_eval_config())
    rerun = compare_baselines(eval_fleet, other_engine)

    assert rerun.train_fingerprint != comparison.train_fingerprint
    for name in (DCGM_METHOD_NAME, ERROR_COUNT_METHOD_NAME):
        before, after = method_by_name(comparison, name), method_by_name(rerun, name)
        assert before.auroc == after.auroc, name
        assert before.ece == after.ece, name
        assert before.auroc_ci_low == after.auroc_ci_low, name
        assert before.auroc_ci_high == after.auroc_ci_high, name
        assert before.ece_by_workload == after.ece_by_workload, name


# --------------------------------------------------- metric computation units


def _label(
    asset_id: str, failed: bool, workload: str = "inference", p_fail: float = 0.4
) -> GroundTruthLabel:
    return GroundTruthLabel(
        asset_id=asset_id,
        provenance=Provenance.SIMULATED,
        assigned_workload=workload,
        horizon_days=365,
        has_latent_defect=failed,
        failed_within_horizon=failed,
        time_to_failure_days=100.0 if failed else None,
        p_fail_assigned_workload=p_fail,
        p_fail_by_workload={workload: p_fail},
    )


def test_perfect_separation_scores_auroc_one() -> None:
    scores = np.asarray([0.1, 0.2, 0.8, 0.9])
    labels = [_label("a", False), _label("b", False), _label("c", True), _label("d", True)]
    evaluation = _evaluate_method(
        "probe", "probe", scores, labels, bootstrap_seed=0, n_bootstrap=50, n_calibration_bins=10
    )
    assert evaluation.auroc == 1.0
    # Exact Mann-Whitney p with a 2-vs-2 split bottoms out at 1/6, so the
    # one-sided p cannot reach 0.05 at this size — it still sits at the floor.
    assert evaluation.auroc_above_half_p == pytest.approx(1.0 / 6.0)
    assert evaluation.n_units == 4
    assert evaluation.failure_count == 2


def test_misaligned_scores_and_labels_raise() -> None:
    scores = np.asarray([0.1, 0.2, 0.3])
    labels = [_label("a", False), _label("b", True)]
    with pytest.raises(ValueError, match="misaligned"):
        _evaluate_method(
            "probe",
            "probe",
            scores,
            labels,
            bootstrap_seed=0,
            n_bootstrap=50,
            n_calibration_bins=10,
        )


def test_single_outcome_class_rejects() -> None:
    scores = np.asarray([0.1, 0.2, 0.3])
    labels = [_label("a", False), _label("b", False), _label("c", False)]
    with pytest.raises(ValueError, match="both outcomes"):
        _evaluate_method(
            "probe",
            "probe",
            scores,
            labels,
            bootstrap_seed=0,
            n_bootstrap=50,
            n_calibration_bins=10,
        )


def test_per_workload_ece_matches_subset_computation() -> None:
    scores = np.asarray([0.05, 0.15, 0.35, 0.55, 0.75, 0.95])
    labels = [
        _label("a", False, workload="inference", p_fail=0.1),
        _label("b", True, workload="inference", p_fail=0.2),
        _label("c", False, workload="inference", p_fail=0.3),
        _label("d", False, workload="finetune", p_fail=0.4),
        _label("e", True, workload="finetune", p_fail=0.6),
        _label("f", True, workload="finetune", p_fail=0.9),
    ]
    evaluation = _evaluate_method(
        "probe", "probe", scores, labels, bootstrap_seed=0, n_bootstrap=50, n_calibration_bins=10
    )
    assert set(evaluation.ece_by_workload) == {"inference", "finetune"}

    inference = slice(0, 3)
    finetune = slice(3, 6)
    for workload, members in (("inference", inference), ("finetune", finetune)):
        expected, _ = expected_calibration_error(
            scores[members],
            np.asarray([label.failed_within_horizon for label in labels[members]], dtype=float),
            10,
        )
        assert evaluation.ece_by_workload[workload] == pytest.approx(expected)


# ----------------------------------------------------------------- report/CLI


def test_report_renders_honest_pre_registered_content(comparison: BaselineComparison) -> None:
    report = render_comparison_report(comparison)
    assert "SIMULATION-BASED PIPELINE CHECK" in report
    assert "not an H1/H2/H3 outcome" in report
    assert "Pre-registered comparison protocol" in report
    for name in ALL_METHOD_NAMES:
        assert name in report
        row = method_by_name(comparison, name)
        assert f"{row.auroc:.3f}" in report
        assert f"{row.ece:.4f}" in report
    assert "Per-workload expected calibration error" in report
    assert "Calibration table" in report


def test_cli_writes_report_and_metrics(tmp_path: Path) -> None:
    exit_code = main(["--out", str(tmp_path)])
    assert exit_code == 0

    report_path = tmp_path / "baseline-comparison.md"
    metrics_path = tmp_path / "baseline-metrics.json"
    assert report_path.exists()
    assert metrics_path.exists()

    report = report_path.read_text(encoding="utf-8")
    for name in ALL_METHOD_NAMES:
        assert name in report
    assert "SIMULATION-BASED PIPELINE CHECK" in report

    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert payload["honesty_banner"].startswith("SIMULATION-BASED PIPELINE CHECK")
    method_names = {method["name"] for method in payload["methods"]}
    assert method_names == ALL_METHOD_NAMES
    for method in payload["methods"]:
        assert 0.0 <= method["auroc"] <= 1.0
        assert method["n_units"] == 2000
        assert len(method["ece_by_workload"]) == len(WORKLOAD_NAMES)


# ------------------------------------------------------- baseline score units


def _scan_with_events(template: ScanResult, asset_id: str, **overrides: object) -> ScanResult:
    """One fleet scan re-stamped with controlled event counters (unit fixture)."""
    data = template.model_dump()
    data["asset_id"] = asset_id
    data.update(overrides)
    return ScanResult.model_validate(data)


def test_dcgm_threshold_scores_rank_by_violation_weight() -> None:
    """Hand-computed fixture: clean < soft alarm < hard errors < everything."""
    template = make_fleet(seed=11, n_units=1).scans[0]
    quiet_snapshot = {
        "max_gpu_temp_c": 60.0,
        "hbm_ecc_error_rate_per_kh": 1.0,
        "throttle_events_per_30d": 0.0,
    }
    hot_snapshot = {**quiet_snapshot, "max_gpu_temp_c": 95.0}
    loud_snapshot = {
        "max_gpu_temp_c": 95.0,
        "hbm_ecc_error_rate_per_kh": 50.0,
        "throttle_events_per_30d": 9.0,
    }
    scans = [
        _scan_with_events(
            template, "clean", xid_errors=0, retired_pages=0, bench_mark_snapshot=quiet_snapshot
        ),
        _scan_with_events(
            template, "soft", xid_errors=0, retired_pages=0, bench_mark_snapshot=hot_snapshot
        ),
        _scan_with_events(
            template, "hard", xid_errors=3, retired_pages=2, bench_mark_snapshot=quiet_snapshot
        ),
        _scan_with_events(
            template, "worst", xid_errors=30, retired_pages=20, bench_mark_snapshot=loud_snapshot
        ),
    ]

    scores = dcgm_threshold_scores(scans)
    # Weights: thermal/ecc-rate/throttle = 1, xid/retired-pages = 2; total 7.
    assert scores[0] == pytest.approx(0.0)  # no violations
    assert scores[1] == pytest.approx(1.0 / 7.0)  # thermal alarm only
    assert scores[2] == pytest.approx(4.0 / 7.0)  # xid + retired pages
    assert scores[3] == pytest.approx(1.0)  # every channel violating
    assert np.all(np.diff(scores) > 0)


def test_error_count_scores_monotone_in_events() -> None:
    template = make_fleet(seed=11, n_units=1).scans[0]
    # Counts stay below the squash's float-saturation knee (~2560 events) so
    # strict increase survives float64: above it, scores collapse onto 1.0.
    scans = [
        _scan_with_events(template, f"u{events}", ecc_events=events, xid_errors=0, retired_pages=0)
        for events in (0, 1, 5, 50, 200, 1000)
    ]
    scores = error_count_scores(scans)
    assert np.all(np.diff(scores) > 0)
    assert np.all((scores >= 0.0) & (scores < 1.0))  # 0 events scores exactly 0


def test_baseline_scores_deterministic_and_permutation_equivariant() -> None:
    fleet = make_fleet(seed=7, n_units=120)
    permutation = np.random.default_rng(5).permutation(fleet.n_units)
    for scorer in (dcgm_threshold_scores, error_count_scores):
        first = scorer(fleet.scans)
        second = scorer(fleet.scans)
        assert np.array_equal(first, second), scorer.__name__
        shuffled_scans = [fleet.scans[i] for i in permutation]
        assert np.array_equal(scorer(shuffled_scans), first[permutation]), scorer.__name__
