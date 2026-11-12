"""Importing the package must not need quip-solver-core.

The research scripts (``round2_cpu.py portfolio-identity`` and others) import
``quip_solver_dwave`` submodules inside pinned reference environments that do
not carry the SDK wheel. Only the modules that speak the wire import it.
"""

from __future__ import annotations

import subprocess
import sys


def test_the_package_root_imports_without_the_sdk():
    code = (
        "import sys\n"
        "import quip_solver_dwave\n"
        "assert 'quip_solver_core' not in sys.modules, 'the root imported the SDK'\n"
    )
    subprocess.run([sys.executable, "-c", code], check=True)
