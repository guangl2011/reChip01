"""Module entry point: ``python -m rechip01.scanner``."""

from __future__ import annotations

import sys

from rechip01.scanner.cli import main

if __name__ == "__main__":
    sys.exit(main())
