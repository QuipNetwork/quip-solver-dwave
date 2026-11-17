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


@pytest.fixture(autouse=True)
def no_token(monkeypatch):
    # adopt_legacy_token_env writes os.environ itself. monkeypatch restores
    # only what it recorded, and a delenv of an absent name records nothing,
    # so the promoted token outlived the test and reached every later one.
    # The setenv records the absent state, and the delenv starts the test
    # without a token.
    monkeypatch.setenv("DWAVE_API_TOKEN", "")
    monkeypatch.delenv("DWAVE_API_TOKEN")


def test_legacy_name_is_promoted(monkeypatch):
    monkeypatch.setenv("DWAVE_API_KEY", "legacy-token")
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
    assert adopt_legacy_token_env() is False
    assert "DWAVE_API_TOKEN" not in os.environ


def test_blank_legacy_value_is_a_noop(monkeypatch):
    monkeypatch.setenv("DWAVE_API_KEY", "   ")
    assert adopt_legacy_token_env() is False
    assert "DWAVE_API_TOKEN" not in os.environ


def test_check_accepts_the_legacy_name(monkeypatch, capsys):
    """--check on a v0.2 node: the token is there, under the old name."""
    monkeypatch.setenv("DWAVE_API_KEY", "legacy-token")
    monkeypatch.delenv("QUIP_DWAVE_MOCK", raising=False)
    monkeypatch.setattr("quip_solver_dwave.cli.ocean_importable", lambda: True)
    assert main(["--check"]) == EXIT_CLEAN
    assert "credentials present" in capsys.readouterr().out
