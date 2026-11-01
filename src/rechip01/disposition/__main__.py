"""Disposition comparison CLI (spec art_vuIPvUhR, §8).

Runs the H3-shaped offline comparison: the risk-adjusted optimizer (on engine
estimates, plus an oracle arm fed the simulator's true per-unit failure
probabilities) against a simple pass/fail resale policy on the identical
held-out simulated fleet under the pre-registered synthetic economy:

    python -m rechip01.disposition --out=/tmp/rechip-disposition

Outputs: ``disposition-comparison.md`` and ``disposition-metrics.json``.
Every number is a simulation-based pipeline check — never a real-world
result and never an H3 outcome.
"""

from __future__ import annotations

import sys

from rechip01.disposition.comparison import main

if __name__ == "__main__":
    sys.exit(main())
