"""Evaluation metrics for the reliability engine (spec art_vuIPvUhR, §8).

Everything here is a **simulation-based pipeline check**: scores are compared
against the simulator's held-out ground truth (realized failures and generator
probabilities). No number produced by this module is a real-world reliability
claim, and every report says so at the top.

Discrimination — can the engine rank units? AUROC of ``p_fail`` against the
realized failure outcome, with a seeded bootstrap confidence interval and the
one-sided Mann-Whitney p-value for AUROC > 0.5 (AUROC *is* the normalized
Mann-Whitney U statistic).

Calibration — do the numbers mean what they say? Expected calibration error
over equal-width probability bins, the per-bin table, mean absolute error
against the generator's true workload-conditioned failure probability, and
interval coverage of that generator truth (the fraction of units whose
calibrated interval contains it).
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Final

import numpy as np
from numpy.typing import NDArray
from scipy.stats import mannwhitneyu
from sklearn.metrics import roc_auc_score

from rechip01.engine.engine import MODEL_VERSION
from rechip01.schemas import ReliabilityEstimate
from rechip01.simulator.generate import GroundTruthLabel

_LOGGER = logging.getLogger(__name__)

_HONESTY_BANNER: Final[str] = (
    "SIMULATION-BASED PIPELINE CHECK — computed against simulated ground truth "
    "(spec art_vuIPvUhR §8). Not a real-world reliability result and not an "
    "H1/H2/H3 outcome."
)


@dataclass(frozen=True)
class CalibrationBin:
    """One equal-width probability bin of the calibration table."""

    lower: float
    upper: float
    predicted_mean: float
    empirical_failure_rate: float | None  # None when the bin is empty
    count: int


@dataclass(frozen=True)
class EngineEvaluation:
    """Engine quality against held-out simulated ground truth (spec §8)."""

    n_units: int
    failure_count: int
    auroc: float
    auroc_ci_low: float
    auroc_ci_high: float
    auroc_above_half_p: float  # one-sided Mann-Whitney p for AUROC > 0.5
    ece: float
    calibration_bins: tuple[CalibrationBin, ...]
    mae_vs_generator_truth: float
    interval_coverage: float  # generator-truth p inside the calibrated interval
    ci_level: float
    ood_flagged: int
    model_version: str
    training_fingerprint: str


def _auroc(scores: NDArray[np.float64], outcomes: NDArray[np.float64]) -> float:
    """AUROC of scores against binary outcomes (float-wrapped for strict typing)."""
    return float(roc_auc_score(outcomes, scores))


def bootstrap_auroc_ci(
    scores: NDArray[np.float64],
    outcomes: NDArray[np.float64],
    *,
    n_bootstrap: int,
    seed: int,
    level: float = 0.95,
) -> tuple[float, float]:
    """Percentile bootstrap CI for AUROC (seeded, deterministic).

    Resamples that land on a single outcome class have undefined AUROC and are
    skipped; the CI is the level-quantile range over valid resamples.
    """
    if not 0.0 < level < 1.0:
        raise ValueError(f"level must be in (0, 1), got {level}")
    rng = np.random.default_rng(seed)
    n = scores.shape[0]
    values: list[float] = []
    attempts = 0
    max_attempts = n_bootstrap * 10
    while len(values) < n_bootstrap and attempts < max_attempts:
        attempts += 1
        sample = rng.integers(0, n, size=n)
        sample_outcomes = outcomes[sample]
        if sample_outcomes.min() == sample_outcomes.max():
            continue
        values.append(_auroc(scores[sample], sample_outcomes))
    if len(values) < n_bootstrap // 2:
        raise RuntimeError(
            f"bootstrap AUROC failed: only {len(values)} valid resamples in {attempts} attempts"
        )
    alpha = (1.0 - level) / 2.0
    low, high = np.percentile(values, [100.0 * alpha, 100.0 * (1.0 - alpha)])
    return float(low), float(high)


def expected_calibration_error(
    predicted: NDArray[np.float64],
    outcomes: NDArray[np.float64],
    n_bins: int,
) -> tuple[float, tuple[CalibrationBin, ...]]:
    """ECE over equal-width bins, plus the full calibration table."""
    if n_bins < 1:
        raise ValueError(f"n_bins must be >= 1, got {n_bins}")
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.minimum((predicted * n_bins).astype(int), n_bins - 1)
    ece = 0.0
    bins: list[CalibrationBin] = []
    for bin_id in range(n_bins):
        members = bin_ids == bin_id
        count = int(members.sum())
        lower, upper = float(edges[bin_id]), float(edges[bin_id + 1])
        if count == 0:
            bins.append(
                CalibrationBin(
                    lower=lower,
                    upper=upper,
                    predicted_mean=float("nan"),
                    empirical_failure_rate=None,
                    count=0,
                )
            )
            continue
        predicted_mean = float(predicted[members].mean())
        empirical = float(outcomes[members].mean())
        ece += (count / predicted.shape[0]) * abs(predicted_mean - empirical)
        bins.append(
            CalibrationBin(
                lower=lower,
                upper=upper,
                predicted_mean=predicted_mean,
                empirical_failure_rate=empirical,
                count=count,
            )
        )
    return float(ece), tuple(bins)


def evaluate(
    estimates: Sequence[ReliabilityEstimate],
    labels: Mapping[str, GroundTruthLabel],
    *,
    ci_level: float = 0.90,
    training_fingerprint: str = "unspecified",
    bootstrap_seed: int = 0,
    n_bootstrap: int = 500,
    n_calibration_bins: int = 10,
) -> EngineEvaluation:
    """Evaluate estimates against held-out simulator ground truth.

    Precondition: every estimate was scored with its label's assigned workload
    as the primary target workload, so ``p_fail`` is comparable to
    ``label.p_fail_assigned_workload`` and ``failed_within_horizon``. Every
    estimate must have a label; unmatched asset IDs raise. ``ci_level`` and
    ``training_fingerprint`` come from the caller's engine configuration so
    the report states exactly which interval level was measured.
    """
    if not estimates:
        raise ValueError("evaluate requires at least one estimate")
    missing = [e.asset_id for e in estimates if e.asset_id not in labels]
    if missing:
        raise ValueError(f"estimates without ground-truth labels: {missing[:5]}")

    ordered = [labels[e.asset_id] for e in estimates]
    scores = np.asarray([e.p_fail for e in estimates], dtype=np.float64)
    outcomes = np.asarray([label.failed_within_horizon for label in ordered], dtype=np.float64)
    generator_truth = np.asarray(
        [label.p_fail_assigned_workload for label in ordered], dtype=np.float64
    )

    auroc = _auroc(scores, outcomes)
    ci_low, ci_high = bootstrap_auroc_ci(
        scores, outcomes, n_bootstrap=n_bootstrap, seed=bootstrap_seed
    )
    failed_scores = scores[outcomes == 1.0]
    survived_scores = scores[outcomes == 0.0]
    p_value = float(mannwhitneyu(failed_scores, survived_scores, alternative="greater").pvalue)
    ece, bins = expected_calibration_error(scores, outcomes, n_calibration_bins)
    mae = float(np.abs(scores - generator_truth).mean())

    inside = np.asarray(
        [
            float(e.ci[0]) <= truth <= float(e.ci[1])
            for e, truth in zip(estimates, generator_truth, strict=True)
        ]
    )
    coverage = float(inside.mean())

    ood_flagged = sum(1 for e in estimates if e.ood_flag)
    failure_count = int(outcomes.sum())
    _LOGGER.info(
        "metrics: AUROC=%.3f (95%% CI %.3f–%.3f) ECE=%.4f MAE=%.4f — %s",
        auroc,
        ci_low,
        ci_high,
        ece,
        mae,
        _HONESTY_BANNER,
    )
    return EngineEvaluation(
        n_units=len(estimates),
        failure_count=failure_count,
        auroc=auroc,
        auroc_ci_low=ci_low,
        auroc_ci_high=ci_high,
        auroc_above_half_p=p_value,
        ece=ece,
        calibration_bins=bins,
        mae_vs_generator_truth=mae,
        interval_coverage=coverage,
        ci_level=ci_level,
        ood_flagged=ood_flagged,
        model_version=MODEL_VERSION,
        training_fingerprint=training_fingerprint,
    )


def render_pipeline_report(evaluation: EngineEvaluation) -> str:
    """Render the evaluation as a Markdown report with the honesty banner."""
    lines = [
        "# Engine calibration report",
        "",
        f"> **{_HONESTY_BANNER}**",
        "",
        "## Headline metrics",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| Units scored | {evaluation.n_units} |",
        f"| Failures within horizon | {evaluation.failure_count} "
        f"({evaluation.failure_count / evaluation.n_units:.1%}) |",
        f"| AUROC | {evaluation.auroc:.3f} (95% CI {evaluation.auroc_ci_low:.3f}–"
        f"{evaluation.auroc_ci_high:.3f}) |",
        f"| Mann-Whitney p (AUROC > 0.5) | {evaluation.auroc_above_half_p:.3g} |",
        f"| Expected calibration error ({len(evaluation.calibration_bins)} bins) | "
        f"{evaluation.ece:.4f} |",
        f"| MAE vs generator-truth P(fail) | {evaluation.mae_vs_generator_truth:.4f} |",
        f"| Interval coverage of generator truth ({evaluation.ci_level:.0%} intervals) | "
        f"{evaluation.interval_coverage:.1%} |",
        f"| OOD-flagged units | {evaluation.ood_flagged} |",
        f"| Model version | `{evaluation.model_version}` |",
        f"| Training lineage | `{evaluation.training_fingerprint}` |",
        "",
        "## Calibration table",
        "",
        "| p_fail bin | mean predicted | empirical failure rate | count |",
        "|---|---|---|---|",
    ]
    for bin_row in evaluation.calibration_bins:
        predicted = "—" if np.isnan(bin_row.predicted_mean) else f"{bin_row.predicted_mean:.3f}"
        empirical = (
            "—"
            if bin_row.empirical_failure_rate is None
            else f"{bin_row.empirical_failure_rate:.3f}"
        )
        lines.append(
            f"| [{bin_row.lower:.1f}, {bin_row.upper:.1f}) | {predicted} | {empirical} "
            f"| {bin_row.count} |"
        )
    lines.append("")
    return "\n".join(lines)


__all__ = [
    "CalibrationBin",
    "EngineEvaluation",
    "bootstrap_auroc_ci",
    "evaluate",
    "expected_calibration_error",
    "render_pipeline_report",
]
