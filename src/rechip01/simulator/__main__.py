"""Demonstration-batch CLI for the fleet simulator (spec §7 demo mode).

Generates a seeded synthetic fleet and writes it to disk as JSONL — the
fastest way to show partners the output quality before they share real
inventory. Simulated provenance is enforced by the generator.

    python -m rechip01.simulator --seed=1234 --n-units=4000 --out=/tmp/rechip-fleet
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from rechip01.simulator import default_fleet_config, generate_fleet

_LOGGER = logging.getLogger(__name__)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for the simulator demo entry point."""
    parser = argparse.ArgumentParser(
        prog="rechip01.simulator",
        description="Generate a seeded demonstration fleet (simulated provenance).",
    )
    parser.add_argument("--seed", type=int, default=0, help="Generator seed (logged).")
    parser.add_argument("--n-units", type=int, default=100, help="Fleet size.")
    parser.add_argument("--horizon-days", type=int, default=365, help="Labeling horizon.")
    parser.add_argument(
        "--out", type=Path, required=True, help="Output directory (created if missing)."
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Generate the fleet and write scans.jsonl, labels.jsonl, and config.json."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
    args = parse_args(argv)

    config = default_fleet_config(
        seed=args.seed, n_units=args.n_units, horizon_days=args.horizon_days
    )
    fleet = generate_fleet(config)

    args.out.mkdir(parents=True, exist_ok=True)
    scans_path = args.out / "scans.jsonl"
    labels_path = args.out / "labels.jsonl"
    config_path = args.out / "config.json"
    scans_path.write_text(
        "".join(f"{scan.model_dump_json()}\n" for scan in fleet.scans), encoding="utf-8"
    )
    labels_path.write_text(
        "".join(f"{label.model_dump_json()}\n" for label in fleet.labels), encoding="utf-8"
    )
    config_path.write_text(f"{config.model_dump_json(indent=2)}\n", encoding="utf-8")

    _LOGGER.info(
        "wrote %d units to %s (scans=%s labels=%s config=%s)",
        fleet.n_units,
        args.out,
        scans_path,
        labels_path,
        config_path,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
