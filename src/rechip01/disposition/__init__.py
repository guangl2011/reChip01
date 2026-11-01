"""Disposition: risk-adjusted net-recovery optimization (spec art_vuIPvUhR, §4).

Public API:

- ``optimize_disposition`` — compare reuse/repair/harvest/recycle for one
  unit on uncertainty-propagated expected net recovery; emit a
  ``DispositionDecision`` with regret bound, sensitivity lines, and the full
  assumption trace. Routes to ``manual_review`` on OOD or over-wide intervals
  instead of recommending.
- ``point_option_values`` — deterministic point valuations (the sensitivity
  engine and the brute-force reference).
- ``value_reuse`` / ``value_repair`` / ``value_harvest`` / ``value_recycle`` —
  pure per-option valuations, vectorized over Monte Carlo draws.
- ``EconomyAssumptions`` / ``PriceRange`` / ``DispositionConfig`` — the
  declared economy and review thresholds.
- ``compare_disposition_policies`` / ``render_comparison_report`` — the
  H3-shaped offline comparison against a simple pass/fail resale policy
  (simulation-based pipeline check).
- ``comparison_economy`` — the pre-registered synthetic economy of the CLI
  comparison.

All valuations are decision-support arithmetic on declared assumptions —
never market data, never a certification claim.
"""

from rechip01.disposition.comparison import (
    OPTIMIZER_POLICY_NAME,
    ORACLE_POLICY_NAME,
    PASS_FAIL_POLICY_NAME,
    DispositionComparison,
    PolicyDifference,
    PolicyOutcome,
    compare_disposition_policies,
    comparison_economy,
    render_comparison_report,
)
from rechip01.disposition.config import DispositionConfig, EconomyAssumptions, PriceRange
from rechip01.disposition.optimizer import (
    DISPOSITION_VALUATION_VERSION,
    PATH_HARVEST,
    PATH_MANUAL_REVIEW,
    PATH_RECYCLE,
    PATH_REPAIR,
    PATH_REUSE,
    OptionValuation,
    optimize_disposition,
    point_option_values,
    value_harvest,
    value_recycle,
    value_repair,
    value_reuse,
    workload_failure_interval,
)

__all__ = [
    "DISPOSITION_VALUATION_VERSION",
    "OPTIMIZER_POLICY_NAME",
    "ORACLE_POLICY_NAME",
    "PASS_FAIL_POLICY_NAME",
    "PATH_HARVEST",
    "PATH_MANUAL_REVIEW",
    "PATH_REPAIR",
    "PATH_RECYCLE",
    "PATH_REUSE",
    "DispositionComparison",
    "DispositionConfig",
    "EconomyAssumptions",
    "OptionValuation",
    "PolicyDifference",
    "PolicyOutcome",
    "PriceRange",
    "compare_disposition_policies",
    "comparison_economy",
    "optimize_disposition",
    "point_option_values",
    "render_comparison_report",
    "value_harvest",
    "value_repair",
    "value_recycle",
    "value_reuse",
    "workload_failure_interval",
]
