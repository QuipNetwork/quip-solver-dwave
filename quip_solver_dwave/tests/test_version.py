"""Keep the reported version in step with the packaged one.

``--version`` prints ``quip_solver_dwave.__version__``, but the release process
bumps ``pyproject.toml``. The two drifted apart for eight release candidates:
every binary from 0.3.1 through 0.3.2-rc8 reported ``0.3.0``.

Reading the metadata at runtime instead would couple ``--version`` to whether
PyInstaller collected the distribution info into the bundle. A test costs the
frozen binary nothing and fails the build the moment the two disagree.
"""

from __future__ import annotations

import tomllib
from pathlib import Path

import pytest

from quip_solver_dwave import __version__

PYPROJECT = Path(__file__).resolve().parents[2] / "pyproject.toml"


def packaged_version() -> str:
    return tomllib.loads(PYPROJECT.read_text())["project"]["version"]


@pytest.mark.skipif(not PYPROJECT.is_file(), reason="running outside the source tree")
def test_reported_version_matches_pyproject():
    assert __version__ == packaged_version(), (
        f"__version__ is {__version__!r} but pyproject.toml says "
        f"{packaged_version()!r}. Bump quip_solver_dwave/__init__.py too, or "
        f"`--version` will misreport the running binary."
    )
