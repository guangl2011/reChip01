"""Risk-adjusted disposition optimizer (spec art_vuIPvUhR, §4).

Consumes one :class:`~rechip01.schemas.ReliabilityEstimate` plus an explicit
:class:`EconomyAssumptions` and compares the disposition paths — reuse into
each workload, repair, component harvesting, certified recycling — on
risk-adjusted expected net recovery (ENR). Uncertainty is propagated by
Monte Carlo: failure probabilities are sampled from the estimate's calibrated
interval (triangular, mode at the point estimate), prices and recovery values
uniformly within their declared ranges. The argmax path becomes the
recommendation; when model uncertainty breaches a review threshold the
optimizer routes to ``manual_review`` instead of recommending (spec §5,
"Low-confidence / OOD" flow state).

Honesty invariants carried here: every decision cites the exact price/cost
assumptions it consumed in ``assumption_trace``; regret bounds are
non-negative by construction; simulated and measured provenance propagate
from the estimate untouched. All ENR figures are decision-support arithmetic
on declared assumptions — never market data, never a certification claim.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final, Literal

import numpy as np
from numpy.typing import NDArray

from rechip01.disposition.config import DispositionConfig, EconomyAssumptions, PriceRange
from rechip01.schemas import DispositionDecision, OptionEvaluation, ReliabilityEstimate

PATH_REUSE: Final[str] = "reuse"  # emitted as f"{PATH_REUSE}_{workload}"
PATH_REPAIR: Final[str] = "repair"
PATH_HARVEST: Final[str] = "harvest"
PATH_RECYCLE: Final[str] = "recycle"
PATH_MANUAL_REVIEW: Final[str] = "manual_review"

# Stamped into every assumption_trace: bump on behavioral changes to the
# valuation formulas or the propagation model.
DISPOSITION_VALUATION_VERSION: Final[str] = "rechip01-disposition-0.1.0"

_ENR_QUANTILES: Final[tuple[float, float]] = (0.05, 0.95)

# Per-quantity seed salts: each sampled quantity draws from its own stream so
# a degenerate interval (zero-width CI, fixed price) cannot shift another
# quantity's random stream — determinism and cross-scenario comparability.
_SALT_PRICE: Final[int] = 1
_SALT_FAILURE: Final[int] = 2
_SALT_HARVEST: Final[int] = 3
_SALT_RECYCLE: Final[int] = 4

PriceScenario = Literal["low", "mid", "high"]


# --------------------------------------------------------------------------
# Uncertainty propagation model
# --------------------------------------------------------------------------
def workload_failure_interval(
    estimate: ReliabilityEstimate, workload: str
) -> tuple[float, float, float]:
    """Return ``(low, mode, high)`` for one workload's failure probability.

    The estimate's calibrated interval is derived for the primary target
    workload; ``workload_fitness`` entries are point values. Propagation
    model: the headline interval's *width* transfers (model uncertainty is
    workload-common to first order), centered on the workload's own failure
    probability, clipped to ``[0, 1]``. For the primary workload — whose
    fitness is exactly ``1 - p_fail`` — this reproduces the headline interval.
    """
    fitness = estimate.workload_fitness[workload]
    p_fail_workload = 1.0 - fitness
    ci_low, ci_high = estimate.ci
    half_width = (ci_high - ci_low) / 2.0
    return (
        max(0.0, p_fail_workload - half_width),
        p_fail_workload,
        min(1.0, p_fail_workload + half_width),
    )


def _sample_failure_draws(
    rng: np.random.Generator, interval: tuple[float, float, float], n_samples: int
) -> NDArray[np.float64]:
    """Triangular draws within the failure interval (constant when degenerate)."""
    low, mode, high = interval
    low = min(low, mode)
    high = max(high, mode)
    if high - low <= 1e-12:
        return np.full(n_samples, mode, dtype=np.float64)
    return np.asarray(rng.triangular(low, mode, high, n_samples), dtype=np.float64)


def _sample_range(
    rng: np.random.Generator, price_range: PriceRange, n_samples: int
) -> NDArray[np.float64]:
    """Uniform draws within one price/recovery range (constant when fixed)."""
    if price_range.high - price_range.low <= 1e-12:
        return np.full(n_samples, price_range.low, dtype=np.float64)
    return rng.uniform(price_range.low, price_range.high, n_samples)


# --------------------------------------------------------------------------
# Option valuations — pure, vectorized over draws
# --------------------------------------------------------------------------
def value_reuse(
    failure_draws: NDArray[np.float64],
    price_draws: NDArray[np.float64],
    *,
    test_cost: float,
    downtime_liability: float,
) -> NDArray[np.float64]:
    """ENR of reselling into one workload, per draw.

    Survive the horizon → resale proceeds; fail → seller-borne downtime
    liability and no proceeds; the functional test always runs first.
    """
    return -test_cost + (1.0 - failure_draws) * price_draws - failure_draws * downtime_liability


def value_repair(
    price_draws_by_workload: NDArray[np.float64],
    *,
    test_cost: float,
    repair_cost: float,
    residual_p_fail: float,
    downtime_liability: float,
) -> NDArray[np.float64]:
    """ENR of repairing and reselling, per draw.

    A repaired unit is workload-flexible: it sells into the best-priced
    workload per draw. Failure risk afterwards is the economy's residual
    parameter — repair reduces but does not eliminate risk.
    """
    best_price = np.asarray(price_draws_by_workload.max(axis=1), dtype=np.float64)
    return (
        -test_cost
        - repair_cost
        + (1.0 - residual_p_fail) * best_price
        - residual_p_fail * downtime_liability
    )


def value_harvest(
    harvest_draws: NDArray[np.float64],
    *,
    test_cost: float,
) -> NDArray[np.float64]:
    """ENR of component harvesting, per draw (no horizon risk, triage test applies)."""
    return -test_cost + harvest_draws


def value_recycle(recycle_draws: NDArray[np.float64]) -> NDArray[np.float64]:
    """ENR of certified recycling, per draw (no test, no downstream liability)."""
    return recycle_draws


# --------------------------------------------------------------------------
# Aggregation
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class OptionValuation:
    """Monte Carlo aggregate for one disposition path."""

    path: str
    mean: float
    quantile_low: float
    quantile_high: float

    def to_evaluation(self, regret_bound: float) -> OptionEvaluation:
        return OptionEvaluation(
            path=self.path,
            expected_net_recovery=self.mean,
            regret_bound=max(0.0, regret_bound),
        )


def _aggregate(path: str, draws: NDArray[np.float64]) -> OptionValuation:
    low_q, high_q = _ENR_QUANTILES
    return OptionValuation(
        path=path,
        mean=float(draws.mean()),
        quantile_low=float(np.quantile(draws, low_q)),
        quantile_high=float(np.quantile(draws, high_q)),
    )


def point_option_values(
    estimate: ReliabilityEstimate,
    economy: EconomyAssumptions,
    *,
    price_scenario: PriceScenario = "mid",
    downtime_multiplier: float = 1.0,
) -> dict[str, float]:
    """Deterministic ENR per economic option at point probabilities.

    Uses each workload's point failure probability (``1 - fitness``) and the
    scenario price (range low / midpoint / high). This is the sensitivity
    engine — and the brute-force reference the Monte Carlo converges to when
    every interval is degenerate. ``downtime_multiplier`` stresses the
    failure-cost assumption (spec §4: sensitivity on price and failure-cost).
    """
    workloads = sorted(estimate.workload_fitness)
    _require_prices_cover(estimate, economy)

    def scenario_price(price_range: PriceRange) -> float:
        return {"low": price_range.low, "mid": price_range.midpoint(), "high": price_range.high}[
            price_scenario
        ]

    liability = economy.downtime_liability * downtime_multiplier
    values: dict[str, float] = {}
    for workload in workloads:
        p_fail_workload = 1.0 - estimate.workload_fitness[workload]
        values[f"{PATH_REUSE}_{workload}"] = float(
            value_reuse(
                np.asarray([p_fail_workload]),
                np.asarray([scenario_price(economy.resale_price_by_workload[workload])]),
                test_cost=economy.test_cost,
                downtime_liability=liability,
            )[0]
        )
    values[PATH_REPAIR] = float(
        value_repair(
            np.asarray([[scenario_price(economy.resale_price_by_workload[w]) for w in workloads]]),
            test_cost=economy.test_cost,
            repair_cost=economy.repair_cost,
            residual_p_fail=economy.repair_residual_p_fail,
            downtime_liability=liability,
        )[0]
    )
    values[PATH_HARVEST] = float(
        value_harvest(
            np.asarray([scenario_price(economy.harvest_recovery)]), test_cost=economy.test_cost
        )[0]
    )
    values[PATH_RECYCLE] = float(
        value_recycle(np.asarray([scenario_price(economy.recycling_value)]))[0]
    )
    return values


def _require_prices_cover(estimate: ReliabilityEstimate, economy: EconomyAssumptions) -> None:
    """Every workload in the estimate needs a resale price — no silent skip."""
    if not estimate.workload_fitness:
        raise ValueError(f"estimate for {estimate.asset_id!r} carries no workload fitness")
    missing = sorted(set(estimate.workload_fitness) - set(economy.resale_price_by_workload))
    if missing:
        raise ValueError(f"economy lacks resale prices for workloads: {missing}")


# --------------------------------------------------------------------------
# Optimizer
# --------------------------------------------------------------------------
def optimize_disposition(
    estimate: ReliabilityEstimate,
    economy: EconomyAssumptions,
    *,
    config: DispositionConfig | None = None,
    seed: int = 0,
) -> DispositionDecision:
    """Compare disposition paths for one unit and emit the decision (spec §4).

    Recommendation is the argmax risk-adjusted ENR among economic options —
    unless model uncertainty breaches a review threshold (OOD flag, or
    headline CI width above ``max_ci_width``), in which case the decision
    routes to ``manual_review`` and no automatic recommendation is made. The
    full valuation table, assumption citations, sensitivity lines, and trigger
    reasons ride on ``assumption_trace``.
    """
    cfg = config if config is not None else DispositionConfig()
    _require_prices_cover(estimate, economy)

    n_samples = cfg.n_samples
    price_rng = np.random.default_rng([seed, _SALT_PRICE])
    failure_rng = np.random.default_rng([seed, _SALT_FAILURE])
    harvest_rng = np.random.default_rng([seed, _SALT_HARVEST])
    recycle_rng = np.random.default_rng([seed, _SALT_RECYCLE])

    workloads = sorted(estimate.workload_fitness)
    price_draws_by_workload = np.column_stack(
        [
            _sample_range(price_rng, economy.resale_price_by_workload[w], n_samples)
            for w in workloads
        ]
    )
    harvest_draws = _sample_range(harvest_rng, economy.harvest_recovery, n_samples)
    recycle_draws = _sample_range(recycle_rng, economy.recycling_value, n_samples)

    valuations: list[OptionValuation] = []
    for column, workload in enumerate(workloads):
        interval = workload_failure_interval(estimate, workload)
        failure_draws = _sample_failure_draws(failure_rng, interval, n_samples)
        reuse_draws = value_reuse(
            failure_draws,
            price_draws_by_workload[:, column],
            test_cost=economy.test_cost,
            downtime_liability=economy.downtime_liability,
        )
        valuations.append(_aggregate(f"{PATH_REUSE}_{workload}", reuse_draws))

    valuations.append(
        _aggregate(
            PATH_REPAIR,
            value_repair(
                price_draws_by_workload,
                test_cost=economy.test_cost,
                repair_cost=economy.repair_cost,
                residual_p_fail=economy.repair_residual_p_fail,
                downtime_liability=economy.downtime_liability,
            ),
        )
    )
    valuations.append(
        _aggregate(
            PATH_HARVEST,
            value_harvest(harvest_draws, test_cost=economy.test_cost),
        )
    )
    valuations.append(_aggregate(PATH_RECYCLE, value_recycle(recycle_draws)))

    ranked = sorted(valuations, key=lambda v: v.mean, reverse=True)
    best, runner_up = ranked[0], ranked[1]
    flip_margin = best.mean - runner_up.mean

    ci_low, ci_high = estimate.ci
    ci_width = ci_high - ci_low
    triggers: list[str] = []
    if estimate.ood_flag:
        triggers.append(
            f"{PATH_MANUAL_REVIEW}: ood_flag set — unit outside training support; "
            "never scored silently"
        )
    if ci_width > cfg.max_ci_width:
        triggers.append(
            f"{PATH_MANUAL_REVIEW}: calibrated interval width {ci_width:.4f} exceeds "
            f"threshold {cfg.max_ci_width:.4f}"
        )

    if triggers:
        review_value = OptionValuation(
            path=PATH_MANUAL_REVIEW,
            mean=best.mean - cfg.review_cost,
            quantile_low=best.quantile_low - cfg.review_cost,
            quantile_high=best.quantile_high - cfg.review_cost,
        )
        valuations.append(review_value)
        recommendation, decision_enr = PATH_MANUAL_REVIEW, review_value.mean
    else:
        recommendation, decision_enr = best.path, best.mean

    max_mean = best.mean
    options = [valuation.to_evaluation(max_mean - valuation.mean) for valuation in valuations]
    trace = _assumption_trace(
        estimate=estimate,
        economy=economy,
        cfg=cfg,
        seed=seed,
        ci_width=ci_width,
        valuations=valuations,
        mid_price_pick=best.path,
        flip_margin=flip_margin,
        triggers=triggers,
    )

    return DispositionDecision(
        asset_id=estimate.asset_id,
        provenance=estimate.provenance,
        recommendation=recommendation,
        expected_net_recovery=decision_enr,
        regret_bound=flip_margin,
        options_evaluated=options,
        assumption_trace=trace,
        model_version=estimate.model_version,
    )


def _assumption_trace(
    *,
    estimate: ReliabilityEstimate,
    economy: EconomyAssumptions,
    cfg: DispositionConfig,
    seed: int,
    ci_width: float,
    valuations: list[OptionValuation],
    mid_price_pick: str,
    flip_margin: float,
    triggers: list[str],
) -> list[str]:
    """Cite every assumption the decision consumed, plus sensitivity outcomes."""
    workloads = sorted(estimate.workload_fitness)
    trace = [
        f"valuation: {DISPOSITION_VALUATION_VERSION} monte-carlo draws={cfg.n_samples} seed={seed}",
        (
            f"reliability: p_fail={estimate.p_fail:.4f} ci=[{estimate.ci[0]:.4f}, "
            f"{estimate.ci[1]:.4f}] width={ci_width:.4f} ood={estimate.ood_flag} "
            f"horizon={estimate.horizon_days}d model_version={estimate.model_version}"
        ),
        (
            "propagation: per-workload failure interval = headline CI width centered on "
            "1 - fitness[w] (clipped to [0, 1]); prices and recovery values uniform within ranges"
        ),
    ]
    trace += [
        f"assumption: resale_price_by_workload[{workload}] = "
        f"[{economy.resale_price_by_workload[workload].low:.2f}, "
        f"{economy.resale_price_by_workload[workload].high:.2f}] USD"
        for workload in workloads
    ]
    trace += [
        f"assumption: test_cost = {economy.test_cost:.2f} USD (applies to reuse, repair, harvest)",
        (
            f"assumption: repair_cost = {economy.repair_cost:.2f} USD; "
            f"repair_residual_p_fail = {economy.repair_residual_p_fail:.3f}"
        ),
        f"assumption: downtime_liability = {economy.downtime_liability:.2f} USD on failure",
        (
            f"assumption: harvest_recovery = [{economy.harvest_recovery.low:.2f}, "
            f"{economy.harvest_recovery.high:.2f}] USD"
        ),
        (
            f"assumption: recycling_value = [{economy.recycling_value.low:.2f}, "
            f"{economy.recycling_value.high:.2f}] USD"
        ),
        f"assumption: review_cost = {cfg.review_cost:.2f} USD",
    ]

    def sensitivity_line(label: str, values: dict[str, float]) -> str:
        pick = max(values, key=lambda path: values[path])
        verdict = "holds" if pick == mid_price_pick else "changes"
        return (
            f"sensitivity: at {label} the recommendation is {pick} "
            f"({verdict} vs mid-price pick {mid_price_pick})"
        )

    trace += [
        sensitivity_line(
            "price-low prices", point_option_values(estimate, economy, price_scenario="low")
        ),
        sensitivity_line(
            "price-high prices", point_option_values(estimate, economy, price_scenario="high")
        ),
        sensitivity_line(
            "2x downtime_liability", point_option_values(estimate, economy, downtime_multiplier=2.0)
        ),
    ]
    trace += [
        (
            f"valuation detail: {valuation.path} ENR mean={valuation.mean:.2f} "
            f"[{valuation.quantile_low:.2f}, {valuation.quantile_high:.2f}] (5-95%)"
        )
        for valuation in valuations
    ]
    trace.append(
        f"regret bound: the recommendation flips if the top-two ENR gap ({flip_margin:.2f} USD) "
        "is exceeded by model error"
    )
    trace += triggers
    return trace


__all__ = [
    "DISPOSITION_VALUATION_VERSION",
    "PATH_HARVEST",
    "PATH_MANUAL_REVIEW",
    "PATH_REPAIR",
    "PATH_RECYCLE",
    "PATH_REUSE",
    "OptionValuation",
    "PriceScenario",
    "optimize_disposition",
    "point_option_values",
    "value_harvest",
    "value_repair",
    "value_recycle",
    "value_reuse",
    "workload_failure_interval",
]
