"""CLI surface tests."""

import json
import signal

import pytest

from quip_solver_dwave import (
    EXIT_CLEAN,
    EXIT_CONFIG_INVALID,
    EXIT_ENV_INCOMPATIBLE,
)
from quip_solver_dwave.cli import install_sigterm_handler, main, run_check


def test_version(capsys):
    assert main(["--version"]) == EXIT_CLEAN
    out = capsys.readouterr().out
    assert "quip-dwave-qa" in out
    assert "protocol 2" in out


def test_capabilities(capsys):
    # The same JSON object, key for key, that a Rust solver built on
    # quip-solver-core 0.0.11 prints (session.rs capabilities_json_from): the
    # identity enums as their stable lowercase names, the enum lists as proto
    # names, and the unset optional fields and the empty family lists omitted.
    assert main(["--capabilities"]) == EXIT_CLEAN
    out = capsys.readouterr().out
    assert json.loads(out) == {
        "supportedKinds": ["ISING_SAMPLE"],
        "maxNodes": 10_000,
        "maxEdges": 100_000,
        "features": ["quantum-anneal", "native-topology", "initial-spins"],
        "protocolVersion": 2,
        "streamWidth": 1,
        "encodings": [
            "COEFFICIENT_ENCODING_I32",
            "COEFFICIENT_ENCODING_I16",
            "COEFFICIENT_ENCODING_I8",
            "COEFFICIENT_ENCODING_F16",
            "COEFFICIENT_ENCODING_F32",
            "COEFFICIENT_ENCODING_F64",
        ],
        "generators": [],
        "backend": "dwave-qpu",
        "algorithm": "quantum-anneal",
        "solverId": "quip-solver-dwave",
    }
    # One compact line, as serde_json writes it.
    assert out.count("\n") == 1 and ": " not in out


def test_capabilities_json_carries_the_hash_as_standard_base64():
    from quip_solver_dwave.session_loop import capabilities_json, capabilities_message

    # 0xfb 0xff exercises both characters where standard and URL-safe base64
    # differ ('+' and '/').
    view = json.loads(capabilities_json(capabilities_message(b"\xfb\xff" * 16)))
    assert view["nativeTopologyHash"] == "+//7//v/" * 5 + "+/8="
    assert list(view)[6] == "nativeTopologyHash"  # between streamWidth and encodings


def test_missing_coordinator():
    assert main([]) == EXIT_CONFIG_INVALID


def test_check_mock_ok(monkeypatch):
    monkeypatch.setenv("QUIP_DWAVE_MOCK", "1")
    assert run_check(force_mock=True) == EXIT_CLEAN


def test_check_no_creds_fails(monkeypatch):
    monkeypatch.delenv("DWAVE_API_KEY", raising=False)
    monkeypatch.delenv("DWAVE_API_TOKEN", raising=False)
    monkeypatch.delenv("QUIP_DWAVE_MOCK", raising=False)
    # Home without a dwave.conf token: force credentials_present False
    monkeypatch.setattr(
        "quip_solver_dwave.cli.credentials_present", lambda: False
    )
    monkeypatch.setattr(
        "quip_solver_dwave.cli.ocean_importable", lambda: True
    )
    assert run_check(force_mock=False) == EXIT_ENV_INCOMPATIBLE


class _FakeSampler:
    def __init__(self):
        self.closed = False

    def close(self):
        self.closed = True


def test_sigterm_handler_closes_sampler_and_exits_clean():
    """SIGTERM must close the sampler and request a clean exit code, matching
    the v0.2 ``_cleanup_handler`` behavior (quip-w5p.8b)."""
    original = signal.getsignal(signal.SIGTERM)
    sampler = _FakeSampler()
    try:
        install_sigterm_handler(sampler)
        handler = signal.getsignal(signal.SIGTERM)
        assert handler is not original
        assert callable(handler)
        with pytest.raises(SystemExit) as exc_info:
            handler(signal.SIGTERM, None)
        assert exc_info.value.code == EXIT_CLEAN
        assert sampler.closed
    finally:
        signal.signal(signal.SIGTERM, original)


def test_sigterm_handler_is_idempotent():
    """A second delivery (or re-entrant signal) must not raise/close twice."""
    original = signal.getsignal(signal.SIGTERM)
    sampler = _FakeSampler()
    try:
        install_sigterm_handler(sampler)
        handler = signal.getsignal(signal.SIGTERM)
        assert callable(handler)
        with pytest.raises(SystemExit):
            handler(signal.SIGTERM, None)
        # Second delivery: already triggered, must return quietly (no raise).
        handler(signal.SIGTERM, None)
    finally:
        signal.signal(signal.SIGTERM, original)


def test_profile_prints_the_grid(tmp_path, capsys):
    from quip_solver_dwave.history import HistoryStore

    path = str(tmp_path / "usage.db")
    HistoryStore(path).close()
    assert main(["--profile", "--usage-db", path]) == EXIT_CLEAN
    out = capsys.readouterr().out
    assert "QPU throughput by day of month" in out
    assert "d28 " in out


def test_profile_without_a_database_fails_clearly(tmp_path, capsys):
    missing = str(tmp_path / "nope.db")
    assert main(["--profile", "--usage-db", missing]) == EXIT_CONFIG_INVALID
    assert "--usage-db" in capsys.readouterr().err
