# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025 QUIP Protocol Contributors

"""PyInstaller entry point for the frozen ``quip-dwave-qa`` executable.

Deliberately thin: it mirrors ``quip_solver_dwave/__main__.py`` so the frozen
binary and ``python -m quip_solver_dwave`` take the same path into the CLI.
PyInstaller needs a real script file to analyse, which is why this exists at
all rather than the spec pointing at the module.

No ``multiprocessing.freeze_support()`` here: neither the miner nor Ocean
starts a process, so in onefile mode there is nothing to guard against
re-executing the bundle.
"""

import sys

from quip_solver_dwave.cli import main

if __name__ == "__main__":
    sys.exit(main())
