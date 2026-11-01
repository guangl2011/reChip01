"""H3-shaped disposition comparison — simulation-based (spec art_vuIPvUhR §4, §8).

Compares disposition policies on the **identical held-out simulated fleet**
under one pre-registered synthetic economy:

- ``pass-fail-resale`` — the simple policy the market falls back to: units
  whose modeled failure risk passes a fixed gate are resold into their
  assigned workload; everything else is recycled. No uncertainty, no
  sensitivity, no intermediate paths — risk is absorbed silently.
- ``risk-adjusted-optimizer`` — :func:`optimize_disposition` run on the
  engine's estimates (manual-review routing included).
- ``risk-adjusted-oracle`` — the same optimizer fed the simulator's true
  per-unit failure probabilities (perfect-information ceiling).

Realization rules: reuse outcomes follow the simulator's ground-truth labels
(a resold unit fails exactly when its label says it failed within the
horizon); repair residual risk, harvesting, and recycling realize at the
economy's midpoint values — the synthetic economy's parameters are the
economy of record by construction. Reuse actions are restricted to each
unit's assigned second-life workload, the deployment ground truth the
simulator provides labels for.

Every output is a simulation-based pipeline check — never a real-world
result and never an H3 outcome. The protocol (policy set, threshold,
economy, seeds, fleet) is fixed in code before results are computed.
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Final

import numpy as np
from numpy.typing import NDArray

from rechip01.disposition.config import DispositionConfig, EconomyAssumptions, PriceRange
from rechip01.disposition.optimizer import (
    PATH_HARVEST,
    PATH_MANUAL_REVIEW,
    PATH_RECYCLE,
    PATH_REPAIR,
    PATH_REUSE,
    optimize_disposition,
)
from rechip01.schemas import DispositionDecision, Provenance, ReliabilityEstimate
from rechip01.simulator.generate import GroundTruthLabel, SimulatedFleet

_LOGGER = logging.getLogger(__name__)

_HONESTY_BANNER: Final[str] = (
    "SIMULATION-BASED PIPELINE CHECK — decisions scored against simulated ground "
    "truth under a synthetic economy (spec art_vuIPvUhR §4, §8). Not a real-world "
    "result and not an H1/H2/H3 outcome."
)

PASS_FAIL_POLICY_NAME: Final[str] = "pass-fail-resale"
OPTIMIZER_POLICY_NAME: Final[str] = "risk-adjusted-optimizer"
ORACLE_POLICY_NAME: Final[str] = "risk-adjusted-oracle"
ORACLE_MODEL_VERSION: Final[str] = "simulator-ground-truth-oracle"

# Pre-registered comparison protocol (fixed before results are computed).
PASS_FAIL_P_FAIL_THRESHOLD: Final[float] = 0.5
_BOOTSTRAP_SEED: Final[int] = 7
_N_BOOTSTRAP: Final[int] = 500
_OPTIMIZER_SEED: Final[int] = 0

# Synthetic economy of record — assumptions of the simulation, never market
# data. Prices are what a working unit fetches when deployed into each
# second-life regime; ranges are deliberately wide to make the comparison
# sensitive to how each policy handles uncertainty.
_ECONOMY_RESALE_PRICE: Final[dict[str, tuple[float, float]]] = {
    "inference": (1500.0, 1900.0),
    "finetune": (1250.0, 1650.0),
    "memory-stress": (1000.0, 1400.0),
}
_ECONOMY_TEST_COST: Final[float] = 40.0
_ECONOMY_REPAIR_COST: Final[float] = 150.0
_ECONOMY_REPAIR_RESIDUAL: Final[float] = 0.10
_ECONOMY_DOWNTIME_LIABILITY: Final[float] = 900.0
_ECONOMY_HARVEST_RECOVERY: Final[tuple[float, float]] = (250.0, 400.0)
_ECONOMY_RECYCLING_VALUE: Final[tuple[float, float]] = (60.0, 90.0)


def comparison_economy() -> EconomyAssumptions:
    """The pre-registered synthetic economy used by the CLI comparison."""
    return EconomyAssumptions(
        resale_price_by_workload={
            name: PriceRange(low=low, high=high)
            for name, (low, high) in _ECONOMY_RESALE_PRICE.items()
        },
        test_cost=_ECONOMY_TEST_COST,
        repair_cost=_ECONOMY_REPAIR_COST,
        repair_residual_p_fail=_ECONOMY_REPAIR_RESIDUAL,
        downtime_liability=_ECONOMY_DOWNTIME_LIABILITY,
        harvest_recovery=PriceRange(
            low=_ECONOMY_HARVEST_RECOVERY[0], high=_ECONOMY_HARVEST_RECOVERY[1]
        ),
        recycling_value=PriceRange(
            low=_ECONOMY_RECYCLING_VALUE[0], high=_ECONOMY_RECYCLING_VALUE[1]
        ),
    )


# --------------------------------------------------------------------------
# Policy action selection
# --------------------------------------------------------------------------
def pass_fail_actions(
    estimates: list[ReliabilityEstimate],
    labels: list[GroundTruthLabel],
    *,
    threshold: float = PASS_FAIL_P_FAIL_THRESHOLD,
) -> list[str]:
    """Pass/fail resale policy: gate on modeled risk, else recycle (no hedging)."""
    if len(estimates) != len(labels):
        raise ValueError(f"estimates/labels misaligned: {len(estimates)} vs {len(labels)}")
    actions: list[str] = []
    for estimate, label in zip(estimates, labels, strict=True):
        if estimate.asset_id != label.asset_id:
            raise ValueError(
                f"estimates/labels misaligned at {estimate.asset_id!r} vs {label.asset_id!r}"
            )
        if estimate.p_fail <= threshold:
            actions.append(f"{PATH_REUSE}_{label.assigned_workload}")
        else:
            actions.append(PATH_RECYCLE)
    return actions


def _assigned_estimate(
    estimate: ReliabilityEstimate, label: GroundTruthLabel
) -> ReliabilityEstimate:
    """Restrict an estimate's reuse options to the assigned workload (truth alignment)."""
    if label.assigned_workload not in estimate.workload_fitness:
        raise ValueError(
            f"estimate for {estimate.asset_id!r} lacks fitness for assigned workload "
            f"{label.assigned_workload!r}"
        )
    return ReliabilityEstimate(
        asset_id=estimate.asset_id,
        horizon_days=estimate.horizon_days,
        provenance=estimate.provenance,
        p_fail=estimate.p_fail,
        ci=estimate.ci,
        workload_fitness={
            label.assigned_workload: estimate.workload_fitness[label.assigned_workload]
        },
        ood_flag=estimate.ood_flag,
        model_version=estimate.model_version,
    )


def optimizer_actions(
    estimates: list[ReliabilityEstimate],
    labels: list[GroundTruthLabel],
    economy: EconomyAssumptions,
    *,
    config: DispositionConfig | None = None,
    seed: int = _OPTIMIZER_SEED,
) -> list[DispositionDecision]:
    """Run the optimizer per unit; decisions align with the fleet order."""
    if len(estimates) != len(labels):
        raise ValueError(f"estimates/labels misaligned: {len(estimates)} vs {len(labels)}")
    decisions: list[DispositionDecision] = []
    for index, (estimate, label) in enumerate(zip(estimates, labels, strict=True)):
        if estimate.asset_id != label.asset_id:
            raise ValueError(
                f"estimates/labels misaligned at {estimate.asset_id!r} vs {label.asset_id!r}"
            )
        decisions.append(
            optimize_disposition(
                _assigned_estimate(estimate, label),
                economy,
                config=config,
                seed=seed + index,
            )
        )
    return decisions


# --------------------------------------------------------------------------
# Realization under ground truth
# --------------------------------------------------------------------------
def realize_action(path: str, label: GroundTruthLabel, economy: EconomyAssumptions) -> float:
    """Realized net recovery of one executed action for one unit.

    Reuse failure follows the simulator's ground-truth label; repair residual
    risk, harvest, and recycling realize at the economy's midpoints (the
    synthetic economy is the economy of record by construction).
    """
    if path.startswith(f"{PATH_REUSE}_"):
        workload = path[len(PATH_REUSE) + 1 :]
        price_range = economy.resale_price_by_workload.get(workload)
        if price_range is None:
            raise ValueError(f"economy lacks resale price for workload {workload!r}")
        if label.failed_within_horizon:
            return -economy.test_cost - economy.downtime_liability
        return -economy.test_cost + price_range.midpoint()
    if path == PATH_REPAIR:
        best_price = max(
            price_range.midpoint() for price_range in economy.resale_price_by_workload.values()
        )
        residual = economy.repair_residual_p_fail
        return (
            -economy.test_cost
            - economy.repair_cost
            + (1.0 - residual) * best_price
            - residual * economy.downtime_liability
        )
    if path == PATH_HARVEST:
        return -economy.test_cost + economy.harvest_recovery.midpoint()
    if path == PATH_RECYCLE:
        return economy.recycling_value.midpoint()
    raise ValueError(f"unknown disposition path {path!r}")


def _realize_decision(
    decision: DispositionDecision,
    label: GroundTruthLabel,
    economy: EconomyAssumptions,
    review_cost: float,
) -> float:
    """Realize one optimizer decision; manual review resolves to its best option."""
    if decision.recommendation != PATH_MANUAL_REVIEW:
        return realize_action(decision.recommendation, label, economy)
    economic = [
        option for option in decision.options_evaluated if option.path != PATH_MANUAL_REVIEW
    ]
    best = max(economic, key=lambda option: option.expected_net_recovery)
    return realize_action(best.path, label, economy) - review_cost


# --------------------------------------------------------------------------
# Comparison
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class PolicyOutcome:
    """One policy's realized results on the shared held-out fleet."""

    name: str
    description: str
    n_units: int
    total_realized: float
    mean_per_unit: float
    action_counts: dict[str, int]
    review_count: int


@dataclass(frozen=True)
class PolicyDifference:
    """Paired difference vs the pass/fail baseline with a bootstrap CI."""

    name: str
    mean_diff_per_unit: float
    ci_low: float
    ci_high: float


@dataclass(frozen=True)
class DispositionComparison:
    """The full comparison: every policy on identical held-out simulated data."""

    policies: tuple[PolicyOutcome, ...]
    differences: tuple[PolicyDifference, ...]
    n_units: int
    horizon_days: int
    eval_seed: int
    economy: EconomyAssumptions
    pass_fail_threshold: float
    optimizer_seed: int
    bootstrap_seed: int
    n_bootstrap: int


def _bootstrap_mean_diff(
    treatment: NDArray[np.float64],
    control: NDArray[np.float64],
    *,
    n_bootstrap: int,
    seed: int,
) -> tuple[float, float, float]:
    """Paired per-unit mean difference with a seeded percentile bootstrap CI."""
    if treatment.shape != control.shape or treatment.size == 0:
        raise ValueError(f"realized arrays misaligned: {treatment.shape} vs {control.shape}")
    diffs = treatment - control
    rng = np.random.default_rng(seed)
    n = diffs.size
    means = np.empty(n_bootstrap, dtype=np.float64)
    for iteration in range(n_bootstrap):
        resample = diffs[rng.integers(0, n, n)]
        means[iteration] = resample.mean()
    return (
        float(diffs.mean()),
        float(np.quantile(means, 0.025)),
        float(np.quantile(means, 0.975)),
    )


def _outcome(
    name: str, description: str, realized: NDArray[np.float64], actions: list[str]
) -> PolicyOutcome:
    counts: dict[str, int] = {}
    for action in actions:
        counts[action] = counts.get(action, 0) + 1
    return PolicyOutcome(
        name=name,
        description=description,
        n_units=realized.size,
        total_realized=float(realized.sum()),
        mean_per_unit=float(realized.mean()),
        action_counts=counts,
        review_count=counts.get(PATH_MANUAL_REVIEW, 0),
    )


def compare_disposition_policies(
    estimates: list[ReliabilityEstimate],
    fleet: SimulatedFleet,
    economy: EconomyAssumptions,
    *,
    config: DispositionConfig | None = None,
    optimizer_seed: int = _OPTIMIZER_SEED,
    bootstrap_seed: int = _BOOTSTRAP_SEED,
    n_bootstrap: int = _N_BOOTSTRAP,
) -> DispositionComparison:
    """Compare the three policies on the identical held-out simulated fleet.

    ``estimates`` must align with ``fleet`` (one estimate per scan, same
    order) — the protocol of :func:`rechip01.engine.harness.assess_by_assigned_workload`.
    """
    labels = fleet.labels
    if len(estimates) != fleet.n_units:
        raise ValueError(f"estimates/fleet misaligned: {len(estimates)} vs {fleet.n_units}")
    review_cost = (config if config is not None else DispositionConfig()).review_cost

    pass_fail = pass_fail_actions(estimates, labels)

    model_decisions = optimizer_actions(
        estimates, labels, economy, config=config, seed=optimizer_seed
    )
    oracle_estimates = [
        ReliabilityEstimate(
            asset_id=label.asset_id,
            horizon_days=label.horizon_days,
            provenance=Provenance.SIMULATED,
            p_fail=label.p_fail_assigned_workload,
            ci=(label.p_fail_assigned_workload, label.p_fail_assigned_workload),
            workload_fitness={label.assigned_workload: 1.0 - label.p_fail_assigned_workload},
            ood_flag=False,
            model_version=ORACLE_MODEL_VERSION,
        )
        for label in labels
    ]
    oracle_decisions = optimizer_actions(
        oracle_estimates, labels, economy, config=config, seed=optimizer_seed
    )

    pass_fail_realized = np.asarray(
        [
            realize_action(action, label, economy)
            for action, label in zip(pass_fail, labels, strict=True)
        ],
        dtype=np.float64,
    )
    model_realized = np.asarray(
        [
            _realize_decision(decision, label, economy, review_cost)
            for decision, label in zip(model_decisions, labels, strict=True)
        ],
        dtype=np.float64,
    )
    oracle_realized = np.asarray(
        [
            _realize_decision(decision, label, economy, review_cost)
            for decision, label in zip(oracle_decisions, labels, strict=True)
        ],
        dtype=np.float64,
    )

    outcomes = [
        _outcome(
            PASS_FAIL_POLICY_NAME,
            (
                "Simple pass/fail resale: units with modeled failure risk at or below a "
                f"fixed gate ({PASS_FAIL_P_FAIL_THRESHOLD:.2f}) resell into their assigned "
                "workload; the rest recycle. No uncertainty propagation, no intermediate paths."
            ),
            pass_fail_realized,
            pass_fail,
        ),
        _outcome(
            OPTIMIZER_POLICY_NAME,
            (
                "Risk-adjusted optimizer on engine estimates: uncertainty-propagated ENR "
                "argmax across reuse/repair/harvest/recycle with manual-review routing."
            ),
            model_realized,
            [decision.recommendation for decision in model_decisions],
        ),
        _outcome(
            ORACLE_POLICY_NAME,
            (
                "The same optimizer fed the simulator's true per-unit failure probabilities "
                "(perfect-information ceiling on the decision layer)."
            ),
            oracle_realized,
            [decision.recommendation for decision in oracle_decisions],
        ),
    ]

    differences = []
    for outcome, realized in (
        (OPTIMIZER_POLICY_NAME, model_realized),
        (ORACLE_POLICY_NAME, oracle_realized),
    ):
        mean_diff, ci_low, ci_high = _bootstrap_mean_diff(
            realized,
            pass_fail_realized,
            n_bootstrap=n_bootstrap,
            seed=bootstrap_seed,
        )
        differences.append(
            PolicyDifference(
                name=outcome,
                mean_diff_per_unit=mean_diff,
                ci_low=ci_low,
                ci_high=ci_high,
            )
        )

    comparison = DispositionComparison(
        policies=tuple(outcomes),
        differences=tuple(differences),
        n_units=fleet.n_units,
        horizon_days=fleet.config.horizon_days,
        eval_seed=fleet.config.seed,
        economy=economy,
        pass_fail_threshold=PASS_FAIL_P_FAIL_THRESHOLD,
        optimizer_seed=optimizer_seed,
        bootstrap_seed=bootstrap_seed,
        n_bootstrap=n_bootstrap,
    )
    for policy in comparison.policies:
        _LOGGER.info(
            "comparison: %-26s realized/unit=%9.2f — %s",
            policy.name,
            policy.mean_per_unit,
            _HONESTY_BANNER,
        )
    return comparison


# --------------------------------------------------------------------------
# Pre-registered report
# --------------------------------------------------------------------------
def render_comparison_report(comparison: DispositionComparison) -> str:
    """Render the pre-registered disposition comparison as a Markdown report."""
    lines = [
        "# Disposition comparison report (H3 scaffold)",
        "",
        f"> **{_HONESTY_BANNER}**",
        "",
        "## Pre-registered comparison protocol",
        "",
        "Fixed before running — restated here so the report cannot quietly",
        "retrofit its own rules:",
        "",
        "- **Policies:** simple pass/fail resale (fixed risk gate, recycle the",
        "  rest), the risk-adjusted optimizer on engine estimates, and the",
        "  same optimizer on the simulator's true per-unit failure",
        "  probabilities (oracle ceiling).",
        f"- **Held-out batch:** one simulated evaluation fleet (seed={comparison.eval_seed}, "
        f"n={comparison.n_units}, horizon={comparison.horizon_days} days); all policies",
        "  act on the identical batch, reuse restricted to each unit's assigned",
        "  second-life workload (the deployment ground truth the simulator",
        "  labels provide).",
        "- **Economy:** synthetic assumptions of record — resale price ranges",
        "  per workload, test/repair costs, downtime liability, harvest",
        "  recovery, recycling value (see JSON artifact); no figure is market",
        "  data.",
        "- **Metrics:** realized net recovery per unit; paired per-unit",
        "difference vs the pass/fail baseline with a seeded percentile",
        f"bootstrap CI ({comparison.n_bootstrap} resamples, seed={comparison.bootstrap_seed});",
        "  optimizer Monte Carlo seed=" + str(comparison.optimizer_seed) + ".",
        "- **Reading:** realized failure of resold units follows the",
        "  simulator's ground-truth labels; repair residual risk and",
        "  harvest/recycle realize at economy midpoints.",
        "",
        "## Policy comparison",
        "",
        "| Policy | Total realized (USD) | Mean per unit (USD) | Manual reviews | Actions |",
        "|---|---|---|---|---|",
    ]
    for policy in comparison.policies:
        actions = ", ".join(
            f"{path}×{count}" for path, count in sorted(policy.action_counts.items())
        )
        lines.append(
            f"| {policy.name} | {policy.total_realized:.0f} | {policy.mean_per_unit:.2f} "
            f"| {policy.review_count} | {actions} |"
        )

    lines += [
        "",
        "## Paired difference vs pass/fail baseline (per unit, USD)",
        "",
        "| Policy | Mean difference | 95% CI |",
        "|---|---|---|",
    ]
    for difference in comparison.differences:
        lines.append(
            f"| {difference.name} | {difference.mean_diff_per_unit:+.2f} "
            f"| [{difference.ci_low:+.2f}, {difference.ci_high:+.2f}] |"
        )

    lines += [
        "",
        "All numbers above are computed against simulated ground truth under a",
        "synthetic economy and are pipeline health checks only — they must never",
        "be quoted as real-world results or as an H3 outcome (TESTING.md).",
        "",
    ]
    return "\n".join(lines)


def _jsonable(comparison: DispositionComparison) -> dict[str, object]:
    """JSON-safe view of the comparison (pydantic economy becomes a dict)."""
    payload: dict[str, object] = asdict(comparison)
    payload["economy"] = comparison.economy.model_dump(mode="json")
    payload["honesty_banner"] = _HONESTY_BANNER
    return payload


# --------------------------------------------------------------------------
# CLI — one command for the H3-scaffold comparison
# --------------------------------------------------------------------------
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for the disposition comparison entry point."""
    parser = argparse.ArgumentParser(
        prog="rechip01.disposition",
        description=(
            "Compare the risk-adjusted disposition optimizer against a simple "
            "pass/fail resale policy on held-out simulated ground truth under a "
            "synthetic economy (simulation-based pipeline check, H3 scaffold)."
        ),
    )
    parser.add_argument(
        "--out", type=Path, required=True, help="Output directory (created if missing)."
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run the default comparison and write disposition-comparison.md + metrics JSON."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
    args = parse_args(argv)

    from rechip01.engine.harness import run_harness

    result = run_harness()
    comparison = compare_disposition_policies(
        result.estimates, result.eval_fleet, comparison_economy()
    )

    args.out.mkdir(parents=True, exist_ok=True)
    report_path = args.out / "disposition-comparison.md"
    metrics_path = args.out / "disposition-metrics.json"
    report_path.write_text(render_comparison_report(comparison), encoding="utf-8")
    metrics_path.write_text(
        json.dumps(_jsonable(comparison), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    optimizer_row = next(p for p in comparison.policies if p.name == OPTIMIZER_POLICY_NAME)
    baseline_row = next(p for p in comparison.policies if p.name == PASS_FAIL_POLICY_NAME)
    _LOGGER.info(
        "disposition comparison: optimizer=%.2f/unit vs pass-fail=%.2f/unit — wrote %s, %s",
        optimizer_row.mean_per_unit,
        baseline_row.mean_per_unit,
        report_path,
        metrics_path,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "OPTIMIZER_POLICY_NAME",
    "ORACLE_MODEL_VERSION",
    "ORACLE_POLICY_NAME",
    "PASS_FAIL_P_FAIL_THRESHOLD",
    "PASS_FAIL_POLICY_NAME",
    "PolicyDifference",
    "PolicyOutcome",
    "DispositionComparison",
    "compare_disposition_policies",
    "comparison_economy",
    "main",
    "parse_args",
    "pass_fail_actions",
    "realize_action",
    "render_comparison_report",
]
