"""``python -m quip_solver_dwave`` entry point."""
from __future__ import annotations

import sys

from quip_solver_dwave.cli import main

if __name__ == "__main__":
    sys.exit(main())
