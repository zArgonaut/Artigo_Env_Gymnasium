"""Entry point: ``python -m tools.calibration``."""

from __future__ import annotations

import sys

from tools.calibration.cli import main

if __name__ == "__main__":
    sys.exit(main())
