"""The deprecated DWAVE_API_KEY name still resolves to a usable token."""

import os

import pytest

pytest.importorskip("numpy")  # ocean imports numpy at module load

from quip_solver_dwave import EXIT_CLEAN  # noqa: E402
from quip_solver_dwave.cli import main  # noqa: E402
from quip_solver_dwave.ocean import (  # noqa: E402
    adopt_legacy_token_env,
    credentials_present,
)


def test_legacy_name_is_promoted(monkeypatch):
    monkeypatch.setenv("DWAVE_API_KEY", "legacy-token")
    monkeypatch.delenv("DWAVE_API_TOKEN", raising=False)
    assert adopt_legacy_token_env() is True
    assert os.environ["DWAVE_API_TOKEN"] == "legacy-token"
    assert credentials_present() is True


def test_canonical_name_wins(monkeypatch):
    monkeypatch.setenv("DWAVE_API_KEY", "legacy-token")
    monkeypatch.setenv("DWAVE_API_TOKEN", "real-token")
    assert adopt_legacy_token_env() is False
    assert os.environ["DWAVE_API_TOKEN"] == "real-token"


def test_no_legacy_value_is_a_noop(monkeypatch):
    monkeypatch.delenv("DWAVE_API_KEY", raising=False)
    monkeypatch.delenv("DWAVE_API_TOKEN", raising=False)
    assert adopt_legacy_token_env() is False
    assert "DWAVE_API_TOKEN" not in os.environ


def test_blank_legacy_value_is_a_noop(monkeypatch):
    monkeypatch.setenv("DWAVE_API_KEY", "   ")
    monkeypatch.delenv("DWAVE_API_TOKEN", raising=False)
    assert adopt_legacy_token_env() is False
    assert "DWAVE_API_TOKEN" not in os.environ


def test_check_accepts_the_legacy_name(monkeypatch, capsys):
    """--check on a v0.2 node: the token is there, under the old name."""
    monkeypatch.setenv("DWAVE_API_KEY", "legacy-token")
    monkeypatch.delenv("DWAVE_API_TOKEN", raising=False)
    monkeypatch.delenv("QUIP_DWAVE_MOCK", raising=False)
    monkeypatch.setattr("quip_solver_dwave.cli.ocean_importable", lambda: True)
    assert main(["--check"]) == EXIT_CLEAN
    assert "credentials present" in capsys.readouterr().out
