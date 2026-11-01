"""Baseline comparison harness — the H1 scaffold (spec art_vuIPvUhR §5, §8).

One command compares the reliability engine against two deliberately simple
baselines on **identical held-out simulated data**:

- ``reliability-engine`` — the workload-conditioned Weibull PH engine,
  fitted on the training fleet and scored under each unit's assigned
  workload (the same protocol as :mod:`rechip01.engine.harness`).
- ``dcgm-threshold-grading`` — a DCGM-threshold-style grading rule: a
  weighted count of alarm-style telemetry violations, in the spirit of
  pass/fail burn-in grading. The point is a plausible, *untuned* grading
  baseline, not a competitor model.
- ``error-count-heuristic`` — a simple error-count heuristic: total
  logged error events, monotonically squashed to (0, 1].

Every method is scored on the same held-out batch with the same metrics:
AUROC (the concordance probability) with a seeded bootstrap confidence
interval and the one-sided Mann-Whitney p-value for AUROC > 0.5, plus
expected calibration error overall and **per assigned workload** (the H2
scaffold: workload-conditioned calibration is the quantity H2 will test
once the Phase I dataset exists).

Honesty invariants: all three methods are evaluated against *simulated*
ground truth; the emitted report is a simulation-based pipeline check —
never a real-world reliability result and never an H1/H2/H3 outcome. The
comparison protocol (method set, thresholds, seeds, metric definitions,
held-out batch) is fixed in this module's code *before* results are
computed — the pre-registration the report restates. Baseline thresholds
are plausible defaults, never tuned against outcomes.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Final

import numpy as np
from numpy.typing import NDArray
from scipy.stats import mannwhitneyu
from sklearn.metrics import roc_auc_score

from rechip01.engine.engine import MODEL_VERSION, ReliabilityEngine
from rechip01.engine.harness import (
    assess_by_assigned_workload,
    harness_eval_config,
    harness_train_config,
)
from rechip01.engine.metrics import CalibrationBin, bootstrap_auroc_ci, expected_calibration_error
from rechip01.schemas import ScanResult
from rechip01.simulator.generate import GroundTruthLabel, SimulatedFleet, generate_fleet

_LOGGER = logging.getLogger(__name__)

_HONESTY_BANNER: Final[str] = (
    "SIMULATION-BASED PIPELINE CHECK — computed against simulated ground truth "
    "(spec art_vuIPvUhR §5, §8). Not a real-world reliability result and not an "
    "H1/H2/H3 outcome."
)

ENGINE_METHOD_NAME: Final[str] = "reliability-engine"
DCGM_METHOD_NAME: Final[str] = "dcgm-threshold-grading"
ERROR_COUNT_METHOD_NAME: Final[str] = "error-count-heuristic"

# Baseline score normalization: any strictly decreasing transform of the raw
# count preserves the ranking AUROC measures; the squash only keeps scores in
# (0, 1] so they can sit next to probability-like scores in the same report.
_ERROR_COUNT_SCALE: Final[float] = 64.0

# DCGM-class alarm-style thresholds. Deliberately plausible, never tuned
# against outcomes: a grading rule a test engineer might write on day one.
_TEMP_C_ALARM_C: Final[float] = 90.0
_ECC_RATE_ALARM_PER_KH: Final[float] = 10.0
_THROTTLE_ALARM_PER_30D: Final[float] = 5.0

# Violation weights: hard error signals (XID events, retired pages) count
# double against soft telemetry alarms. Weights are fixed pre-registration.
_WEIGHT_THERMAL: Final[float] = 1.0
_WEIGHT_ECC_RATE: Final[float] = 1.0
_WEIGHT_THROTTLE: Final[float] = 1.0
_WEIGHT_XID: Final[float] = 2.0
_WEIGHT_RETIRED_PAGES: Final[float] = 2.0
_TOTAL_WEIGHT: Final[float] = (
    _WEIGHT_THERMAL + _WEIGHT_ECC_RATE + _WEIGHT_THROTTLE + _WEIGHT_XID + _WEIGHT_RETIRED_PAGES
)


# --------------------------------------------------------------------------
# Baselines: pure functions of the held-out scans only. No label, no engine,
# no training data ever enters a baseline score — holdout isolation is
# structural (enforced by the tests).
# --------------------------------------------------------------------------
def dcgm_threshold_scores(scans: Sequence[ScanResult]) -> NDArray[np.float64]:
    """Score units by weighted alarm-style threshold violations (grading rule).

    Missing snapshot readings are treated as non-violating (grade on what the
    export shows), consistent with threshold grading on partial telemetry.
    """
    scores = np.empty(len(scans), dtype=np.float64)
    for row, scan in enumerate(scans):
        snapshot = scan.bench_mark_snapshot
        violations = 0.0
        temperature = snapshot.get("max_gpu_temp_c")
        if temperature is not None and temperature > _TEMP_C_ALARM_C:
            violations += _WEIGHT_THERMAL
        ecc_rate = snapshot.get("hbm_ecc_error_rate_per_kh")
        if ecc_rate is not None and ecc_rate > _ECC_RATE_ALARM_PER_KH:
            violations += _WEIGHT_ECC_RATE
        throttles = snapshot.get("throttle_events_per_30d")
        if throttles is not None and throttles > _THROTTLE_ALARM_PER_30D:
            violations += _WEIGHT_THROTTLE
        if scan.xid_errors > 0:
            violations += _WEIGHT_XID
        if scan.retired_pages > 0:
            violations += _WEIGHT_RETIRED_PAGES
        scores[row] = violations / _TOTAL_WEIGHT
    return scores


def error_count_scores(scans: Sequence[ScanResult]) -> NDArray[np.float64]:
    """Score units by total logged error events, squashed monotonically to (0, 1]."""
    raw = np.asarray(
        [scan.ecc_events + scan.xid_errors + scan.retired_pages for scan in scans],
        dtype=np.float64,
    )
    return 1.0 - np.exp(-raw / _ERROR_COUNT_SCALE)


# --------------------------------------------------------------------------
# Method evaluation and comparison
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class MethodEvaluation:
    """One method's metrics on the shared held-out batch (simulated truth)."""

    name: str
    description: str
    n_units: int
    failure_count: int
    auroc: float
    auroc_ci_low: float
    auroc_ci_high: float
    auroc_above_half_p: float
    ece: float
    ece_by_workload: dict[str, float]  # H2 scaffold: workload-conditioned calibration
    calibration_bins: tuple[CalibrationBin, ...]
    model_version: str | None  # None for rule-based baselines


@dataclass(frozen=True)
class BaselineComparison:
    """The full comparison: every method on identical held-out simulated data."""

    methods: tuple[MethodEvaluation, ...]
    n_units: int
    failure_count: int
    horizon_days: int
    eval_seed: int
    train_fingerprint: str
    ci_level: float
    bootstrap_seed: int
    n_bootstrap: int
    n_calibration_bins: int


def _evaluate_method(
    name: str,
    description: str,
    scores: NDArray[np.float64],
    labels: Sequence[GroundTruthLabel],
    *,
    bootstrap_seed: int,
    n_bootstrap: int,
    n_calibration_bins: int,
    model_version: str | None = None,
) -> MethodEvaluation:
    """Compute the shared metric set for one method's scores on held-out labels."""
    if len(scores) != len(labels) or len(labels) == 0:
        raise ValueError(f"scores/labels misaligned for {name!r}: {len(scores)} vs {len(labels)}")
    outcomes = np.asarray([label.failed_within_horizon for label in labels], dtype=np.float64)
    n_failures = int(outcomes.sum())
    if n_failures == 0 or n_failures == len(outcomes):
        raise ValueError(
            f"{name!r}: held-out batch must contain both outcomes to compute AUROC; "
            f"got {n_failures}/{len(outcomes)} failures"
        )

    auroc = float(roc_auc_score(outcomes, scores))
    ci_low, ci_high = bootstrap_auroc_ci(
        scores, outcomes, n_bootstrap=n_bootstrap, seed=bootstrap_seed
    )
    failed_scores = scores[outcomes == 1.0]
    survived_scores = scores[outcomes == 0.0]
    p_value = float(mannwhitneyu(failed_scores, survived_scores, alternative="greater").pvalue)
    ece, bins = expected_calibration_error(scores, outcomes, n_calibration_bins)

    ece_by_workload: dict[str, float] = {}
    workload_names = sorted({label.assigned_workload for label in labels})
    for workload in workload_names:
        members = np.asarray(
            [label.assigned_workload == workload for label in labels], dtype=np.bool_
        )
        workload_ece, _ = expected_calibration_error(
            scores[members], outcomes[members], n_calibration_bins
        )
        ece_by_workload[workload] = workload_ece

    return MethodEvaluation(
        name=name,
        description=description,
        n_units=len(labels),
        failure_count=n_failures,
        auroc=auroc,
        auroc_ci_low=ci_low,
        auroc_ci_high=ci_high,
        auroc_above_half_p=p_value,
        ece=ece,
        ece_by_workload=ece_by_workload,
        calibration_bins=bins,
        model_version=model_version,
    )


def compare_baselines(
    eval_fleet: SimulatedFleet,
    engine: ReliabilityEngine,
    *,
    bootstrap_seed: int = 0,
    n_bootstrap: int = 500,
    n_calibration_bins: int = 10,
) -> BaselineComparison:
    """Compare the fitted engine and both baselines on identical held-out data.

    ``engine`` must already be fitted on a training fleet disjoint from
    ``eval_fleet`` — this function reads no training data at all. Baseline
    scores are computed from ``eval_fleet.scans`` alone; the held-out labels
    are joined only to compute metrics, never fed back to any method.
    """
    if eval_fleet.n_units == 0:
        raise ValueError("eval_fleet must contain at least one unit")

    estimates = assess_by_assigned_workload(engine, eval_fleet)
    engine_scores = np.asarray([estimate.p_fail for estimate in estimates], dtype=np.float64)

    labels = eval_fleet.labels
    methods = [
        _evaluate_method(
            ENGINE_METHOD_NAME,
            "Workload-conditioned Weibull PH engine, scored under each unit's assigned workload",
            engine_scores,
            labels,
            model_version=MODEL_VERSION,
            bootstrap_seed=bootstrap_seed,
            n_bootstrap=n_bootstrap,
            n_calibration_bins=n_calibration_bins,
        ),
        _evaluate_method(
            DCGM_METHOD_NAME,
            "DCGM-threshold-style grading: weighted alarm-style telemetry "
            "violations (pass/fail burn-in spirit; untuned)",
            dcgm_threshold_scores(eval_fleet.scans),
            labels,
            bootstrap_seed=bootstrap_seed,
            n_bootstrap=n_bootstrap,
            n_calibration_bins=n_calibration_bins,
        ),
        _evaluate_method(
            ERROR_COUNT_METHOD_NAME,
            "Simple error-count heuristic: total logged error events",
            error_count_scores(eval_fleet.scans),
            labels,
            bootstrap_seed=bootstrap_seed,
            n_bootstrap=n_bootstrap,
            n_calibration_bins=n_calibration_bins,
        ),
    ]

    comparison = BaselineComparison(
        methods=tuple(methods),
        n_units=eval_fleet.n_units,
        failure_count=methods[0].failure_count,
        horizon_days=eval_fleet.config.horizon_days,
        eval_seed=eval_fleet.config.seed,
        train_fingerprint=engine.training_fingerprint(),
        ci_level=engine.config.ci_level,
        bootstrap_seed=bootstrap_seed,
        n_bootstrap=n_bootstrap,
        n_calibration_bins=n_calibration_bins,
    )
    for method in comparison.methods:
        _LOGGER.info(
            "comparison: %-24s AUROC=%.3f (CI %.3f–%.3f) ECE=%.4f — %s",
            method.name,
            method.auroc,
            method.auroc_ci_low,
            method.auroc_ci_high,
            method.ece,
            _HONESTY_BANNER,
        )
    return comparison


# --------------------------------------------------------------------------
# Pre-registered report
# --------------------------------------------------------------------------
def _format_calibration_table(bins: tuple[CalibrationBin, ...]) -> list[str]:
    """Render one method's calibration table (same layout as the engine report)."""
    lines = [
        "| p_fail bin | mean predicted | empirical failure rate | count |",
        "|---|---|---|---|",
    ]
    for bin_row in bins:
        predicted = "—" if math.isnan(bin_row.predicted_mean) else f"{bin_row.predicted_mean:.3f}"
        empirical = (
            "—"
            if bin_row.empirical_failure_rate is None
            else f"{bin_row.empirical_failure_rate:.3f}"
        )
        lines.append(
            f"| [{bin_row.lower:.1f}, {bin_row.upper:.1f}) | {predicted} | {empirical} "
            f"| {bin_row.count} |"
        )
    return lines


def render_comparison_report(comparison: BaselineComparison) -> str:
    """Render the pre-registered baseline comparison as a Markdown report."""
    workload_names: list[str] = []
    for method in comparison.methods:
        for workload in method.ece_by_workload:
            if workload not in workload_names:
                workload_names.append(workload)

    lines = [
        "# Baseline comparison report (H1 scaffold)",
        "",
        f"> **{_HONESTY_BANNER}**",
        "",
        "## Pre-registered comparison protocol",
        "",
        "Fixed before running — restated here so the report cannot quietly",
        "retrofit its own rules:",
        "",
        "- **Methods:** the fitted reliability engine, a DCGM-threshold-style",
        "  grading rule, and a simple error-count heuristic. Baseline",
        "  thresholds and weights are plausible defaults, never tuned against",
        "  outcomes.",
        f"- **Held-out batch:** one simulated evaluation fleet "
        f"(seed={comparison.eval_seed}, n={comparison.n_units}, "
        f"horizon={comparison.horizon_days} days); all methods score the "
        "identical batch.",
        f"- **Metrics:** AUROC (concordance) with a seeded percentile bootstrap "
        f"CI ({comparison.n_bootstrap} resamples, seed={comparison.bootstrap_seed}); "
        "one-sided Mann-Whitney p for AUROC > 0.5; expected calibration error "
        f"({comparison.n_calibration_bins} equal-width bins) overall and per "
        "assigned workload.",
        "- **Reading:** the engine's interval level is "
        f"{comparison.ci_level:.0%}; training lineage "
        f"`{comparison.train_fingerprint}`.",
        "",
        "## Method comparison",
        "",
        "| Method | AUROC (concordance) | 95% CI | p (AUROC > 0.5) | ECE |",
        "|---|---|---|---|---|",
    ]
    for method in comparison.methods:
        lines.append(
            f"| {method.name} | {method.auroc:.3f} "
            f"| [{method.auroc_ci_low:.3f}, {method.auroc_ci_high:.3f}] "
            f"| {method.auroc_above_half_p:.3g} | {method.ece:.4f} |"
        )

    lines += [
        "",
        "## Per-workload expected calibration error (H2 scaffold)",
        "",
        "Workload-conditioned calibration is the quantity the Phase I H2",
        "comparison will test on real data; these columns are its simulated",
        "scaffold. Lower is better; bins per workload follow the protocol.",
        "",
    ]
    header = "| Method | " + " | ".join(workload_names) + " |"
    divider = "|---" * (len(workload_names) + 1) + "|"
    lines += [header, divider]
    for method in comparison.methods:
        cells = " | ".join(
            f"{method.ece_by_workload.get(workload, float('nan')):.4f}"
            for workload in workload_names
        )
        lines.append(f"| {method.name} | {cells} |")

    for method in comparison.methods:
        lines += [
            "",
            f"## Calibration table — {method.name}",
            "",
            f"{method.description}. Model version: "
            f"`{method.model_version if method.model_version else 'n/a (rule-based)'}`.",
            "",
            *_format_calibration_table(method.calibration_bins),
        ]

    lines += [
        "",
        "All numbers above are computed against simulated ground truth and are",
        "pipeline health checks only — they must never be quoted as real-world",
        "reliability results or as H1/H2/H3 outcomes (TESTING.md).",
        "",
    ]
    return "\n".join(lines)


def _json_sanitize(value: object) -> object:
    """Recursively convert NaN floats to None (strict-JSON-safe output)."""
    if isinstance(value, float):
        return None if math.isnan(value) else value
    if isinstance(value, dict):
        return {key: _json_sanitize(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_sanitize(item) for item in value]
    return value


def _jsonable(comparison: BaselineComparison) -> dict[str, object]:
    """JSON-safe view of the comparison (NaN bin entries become null)."""
    payload: dict[str, object] = asdict(comparison)
    payload["honesty_banner"] = _HONESTY_BANNER
    sanitized = _json_sanitize(payload)
    assert isinstance(sanitized, dict)  # narrowing for the strict typecheck
    return sanitized


# --------------------------------------------------------------------------
# CLI — the one command (spec §8: "the comparison runs as one command")
# --------------------------------------------------------------------------
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for the baseline comparison entry point."""
    parser = argparse.ArgumentParser(
        prog="rechip01.evaluation",
        description=(
            "Compare the reliability engine against DCGM-threshold-style and "
            "error-count baselines on held-out simulated ground truth "
            "(simulation-based pipeline check, H1 scaffold)."
        ),
    )
    parser.add_argument(
        "--out", type=Path, required=True, help="Output directory (created if missing)."
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run the default comparison and write baseline-comparison.md + metrics JSON."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
    args = parse_args(argv)

    train_fleet = generate_fleet(harness_train_config())
    eval_fleet = generate_fleet(harness_eval_config())
    engine = ReliabilityEngine().fit(train_fleet)
    comparison = compare_baselines(eval_fleet, engine)

    args.out.mkdir(parents=True, exist_ok=True)
    report_path = args.out / "baseline-comparison.md"
    metrics_path = args.out / "baseline-metrics.json"
    report_path.write_text(render_comparison_report(comparison), encoding="utf-8")
    metrics_path.write_text(
        json.dumps(_jsonable(comparison), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    engine_row = next(m for m in comparison.methods if m.name == ENGINE_METHOD_NAME)
    _LOGGER.info(
        "baseline comparison: engine AUROC=%.3f vs dcgm=%.3f error-count=%.3f — wrote %s, %s",
        engine_row.auroc,
        comparison.methods[1].auroc,
        comparison.methods[2].auroc,
        report_path,
        metrics_path,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "DCGM_METHOD_NAME",
    "ENGINE_METHOD_NAME",
    "ERROR_COUNT_METHOD_NAME",
    "MethodEvaluation",
    "BaselineComparison",
    "compare_baselines",
    "dcgm_threshold_scores",
    "error_count_scores",
    "main",
    "parse_args",
    "render_comparison_report",
]
