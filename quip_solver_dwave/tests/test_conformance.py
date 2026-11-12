"""End-to-end conformance: miner vs the quip-solver-drive binary.

Spawns ``quip-solver-drive`` (the scripted-coordinator CLI from the published
``quip-solver-conformance`` crate) against a wrapper that launches
``python -m quip_solver_dwave --mock``.

The driver binary resolves in order: the ``QUIP_SOLVER_DRIVE`` environment
variable, then ``quip-solver-drive`` on ``PATH``. Without either the test
skips — the driver is a Rust binary and this repo carries no Rust toolchain.
CI runs it in the ``conformance`` job, which cargo-installs the crate.
"""
from __future__ import annotations

import os
import shutil
import stat
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest


def _drive_bin() -> Path:
    explicit = os.environ.get("QUIP_SOLVER_DRIVE")
    if explicit:
        p = Path(explicit)
        if not p.is_file():
            pytest.fail(f"QUIP_SOLVER_DRIVE points at no file: {explicit}")
        return p
    found = shutil.which("quip-solver-drive")
    if found:
        return Path(found)
    pytest.skip(
        "quip-solver-drive not found: set QUIP_SOLVER_DRIVE or put it on "
        "PATH (cargo install --locked quip-solver-conformance)"
    )


def _write_miner_wrapper(tmpdir: Path) -> Path:
    """Shell script quip-solver-drive can exec as a solver binary.

    Uses the interpreter running this test, so the wrapper sees the same
    installed ``quip_solver_dwave`` package — no repository-layout
    assumptions.
    """
    wrapper = tmpdir / "quip-dwave-qa"
    script = f"""#!/usr/bin/env bash
set -euo pipefail
export QUIP_DWAVE_MOCK=1
exec "{sys.executable}" -m quip_solver_dwave --mock "$@"
"""
    wrapper.write_text(script)
    wrapper.chmod(wrapper.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
    return wrapper


def _drive(mode: list[str], *, backend: str, timeout: int) -> str:
    """Run quip-solver-drive in ``mode`` against the mock miner; return its report.

    The driver prints a per-case report and exits 0 iff the composite verdict
    passes.
    """
    drive = _drive_bin()
    with tempfile.TemporaryDirectory(prefix="quip-dwave-conf-") as td:
        tdp = Path(td)
        miner = _write_miner_wrapper(tdp)
        env = os.environ.copy()
        env["QUIP_SESSION_TOKEN"] = "test-token"
        env["QUIP_DWAVE_MOCK"] = "1"
        env["QUIP_DWAVE_MOCK_BACKEND"] = backend
        sock = tdp / "conf.sock"
        proc = subprocess.run(
            [str(drive), *mode, str(miner), f"unix://{sock}"],
            capture_output=True,
            text=True,
            timeout=timeout,
            env=env,
        )
        out = (proc.stdout or "") + (proc.stderr or "")
        assert proc.returncode == 0 and "[FAIL]" not in out, (
            f"quip-solver-drive returned {proc.returncode}\n{out}"
        )
        return out


def test_conformance_against_quip_solver_drive():
    out = _drive([], backend="exact", timeout=120)
    # The miner lists ISING_GENERATE, so the salt-lease walk must have run.
    assert "[pass] salt lease" in out


def test_family_gate_against_quip_solver_drive():
    # The --family gate drives every advertised generated revision through
    # FAM-47 admission, a full lease, a stale lease and an expired one. Its
    # instances have 80 nodes, past the exact mock, so the SA mock answers.
    out = _drive(["--family"], backend="sa", timeout=120)
    assert "[pass] supported-revision quip/twisted-ice-dimer@1" in out
