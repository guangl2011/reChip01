"""Passport service CLI: serve the assessment API (spec art_vuIPvUhR, §4–5).

Demo mode by construction: the engine is trained on a seeded simulated
fleet and stores are file-backed under ``--data-root``. Every response
keeps its provenance labeling — simulated and measured stay distinct end
to end.

    python -m rechip01.passport --host=127.0.0.1 --port=8000 --data-root=./rechip01-data

Requires the server extra: ``pip install "rechip01[server]"``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from rechip01.passport.app import create_app, default_deps


def main(argv: list[str] | None = None) -> int:
    """Parse arguments and serve the API; returns a process exit code."""
    parser = argparse.ArgumentParser(
        prog="rechip01.passport",
        description=(
            "Serve the reChip01 assessment API (demo mode: engine trained on "
            "a seeded simulated fleet; decision support, not certification)"
        ),
    )
    parser.add_argument("--host", default="127.0.0.1", help="Bind address (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8000, help="Bind port (default: 8000)")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("./rechip01-data"),
        help="File-backed store root for scans and passports (default: ./rechip01-data)",
    )
    parser.add_argument(
        "--training-seed",
        type=int,
        default=11,
        help="Simulated training-fleet seed (default: 11)",
    )
    parser.add_argument(
        "--training-units",
        type=int,
        default=400,
        help="Simulated training-fleet size (default: 400)",
    )
    args = parser.parse_args(argv)

    try:
        import uvicorn
    except ImportError:  # pragma: no cover - exercised only without the extra
        print(
            'uvicorn is not installed; install the server extra: pip install "rechip01[server]"',
            file=sys.stderr,
        )
        return 1

    deps = default_deps(
        args.data_root, training_seed=args.training_seed, n_training_units=args.training_units
    )
    uvicorn.run(create_app(deps), host=args.host, port=args.port)
    return 0


if __name__ == "__main__":
    sys.exit(main())
