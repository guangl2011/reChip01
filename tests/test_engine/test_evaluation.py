"""The harness go/no-go gate: engine quality rails on simulator ground truth.

These are the spec §8 acceptance criteria, as statistical rails with wide
margins (set conservatively, tuned only against held-out ground truth):

- discrimination: AUROC's 95% bootstrap CI excludes 0.5 (a meaningful margin),
  and the point estimate clears the floor;
- calibration: expected calibration error within the ceiling, interval
  coverage of generator truth above the floor;
- honesty: the report carries the simulation-only banner, and the OOD flag
  rate on in-distribution held-out data stays small.

All assertions are pipeline health checks on simulated data — never real-world
reliability claims and never H1/H2/H3 outcomes (TESTING.md).
"""

from __future__ import annotations

from rechip01.engine.harness import HarnessResult
from rechip01.engine.metrics import render_pipeline_report

# Pipeline rails, not aspirations: the spec §8 bar is AUROC > 0.5 "by a
# statistically meaningful margin" — enforced statistically by the CI
# exclusion below. The 0.65 floor adds a wide-margin point check on the
# observed held-out value (the generator's latent-defect channel is only
# weakly visible in telemetry by design, which caps achievable AUROC well
# below 1.0). Rails are tuned only against held-out ground truth.
HARNESS_AUROC_FLOOR = 0.65
HARNESS_ECE_CEILING = 0.10
HARNESS_COVERAGE_FLOOR = 0.80
HARNESS_MAX_OOD_RATE = 0.05


def test_auroc_clears_floor_with_margin(harness_result: HarnessResult) -> None:
    evaluation = harness_result.evaluation
    assert evaluation.auroc_ci_low > 0.5, (
        f"AUROC CI [{evaluation.auroc_ci_low:.3f}, {evaluation.auroc_ci_high:.3f}] "
        "includes 0.5 — discrimination is not statistically meaningful"
    )
    assert evaluation.auroc >= HARNESS_AUROC_FLOOR


def test_calibration_and_coverage_within_rails(harness_result: HarnessResult) -> None:
    evaluation = harness_result.evaluation
    assert evaluation.ece <= HARNESS_ECE_CEILING
    assert evaluation.interval_coverage >= HARNESS_COVERAGE_FLOOR


def test_in_distribution_ood_rate_stays_small(harness_result: HarnessResult) -> None:
    evaluation = harness_result.evaluation
    rate = evaluation.ood_flagged / evaluation.n_units
    assert rate < HARNESS_MAX_OOD_RATE


def test_harness_counts_match_config(harness_result: HarnessResult) -> None:
    evaluation = harness_result.evaluation
    assert evaluation.n_units == 2000
    assert evaluation.failure_count > 0
    assert evaluation.training_fingerprint.startswith("seed=101;")
    assert evaluation.model_version


def test_report_is_honest_and_renders(harness_result: HarnessResult) -> None:
    report = render_pipeline_report(harness_result.evaluation)
    assert "SIMULATION-BASED PIPELINE CHECK" in report
    assert "not an H1/H2/H3 outcome" in report
    assert f"{harness_result.evaluation.auroc:.3f}" in report
    assert "| p_fail bin |" in report
