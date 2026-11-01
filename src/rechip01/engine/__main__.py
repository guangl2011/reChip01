"""Reliability-engine harness CLI (spec art_vuIPvUhR, §8).

Fits the engine on the default simulated training fleet, scores the held-out
evaluation fleet, and writes the simulation-based pipeline-check report plus
per-unit estimates:

    python -m rechip01.engine --out=/tmp/rechip-engine

Outputs: ``calibration-report.md`` (the CI artifact), ``estimates.jsonl``,
and ``config.json``. Every number in the report is labeled as a simulated
ground-truth check — never a real-world reliability claim.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from rechip01.engine.harness import run_harness
from rechip01.engine.metrics import render_pipeline_report

_LOGGER = logging.getLogger(__name__)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for the engine harness entry point."""
    parser = argparse.ArgumentParser(
        prog="rechip01.engine",
        description=(
            "Train, assess, and evaluate the reliability engine on simulator "
            "ground truth (simulation-based pipeline check)."
        ),
    )
    parser.add_argument(
        "--out", type=Path, required=True, help="Output directory (created if missing)."
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run the harness and write calibration-report.md, estimates.jsonl, config.json."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
    args = parse_args(argv)

    result = run_harness()

    args.out.mkdir(parents=True, exist_ok=True)
    report_path = args.out / "calibration-report.md"
    estimates_path = args.out / "estimates.jsonl"
    config_path = args.out / "config.json"

    report_path.write_text(render_pipeline_report(result.evaluation), encoding="utf-8")
    estimates_path.write_text(
        "".join(f"{estimate.model_dump_json()}\n" for estimate in result.estimates),
        encoding="utf-8",
    )
    config_path.write_text(result.engine.config.model_dump_json(indent=2) + "\n", encoding="utf-8")

    _LOGGER.info(
        "engine harness: AUROC=%.3f ECE=%.4f — wrote report=%s estimates=%s config=%s",
        result.evaluation.auroc,
        result.evaluation.ece,
        report_path,
        estimates_path,
        config_path,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
