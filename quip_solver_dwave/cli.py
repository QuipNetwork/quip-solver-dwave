"""CLI for ``quip-dwave-qa`` / ``python -m quip_solver_dwave``."""

from __future__ import annotations

import argparse
import json
import logging
import os
import signal
import sys
import threading
import time

from google.protobuf.json_format import MessageToDict

from quip_solver_dwave import (
    EXIT_CLEAN,
    EXIT_CONFIG_INVALID,
    EXIT_ENV_INCOMPATIBLE,
    EXIT_INTERNAL_FATAL,
    __version__,
)
from quip_solver_dwave.budget import DEFAULT_USAGE_DB
from quip_solver_dwave.capture import (
    DEFAULT_ALLOWED_H_MILLI,
    DEFAULT_ALLOWED_J_MILLI,
    capture_spec,
    compare_specs,
    format_comparison,
    load_spec,
    write_spec,
)
from quip_solver_dwave.history import HistoryStore
from quip_solver_dwave.ocean import (
    OceanSampler,
    SupportsClose,
    adopt_legacy_token_env,
    credentials_present,
    mock_mode_enabled,
    ocean_importable,
)
from quip_solver_dwave.report import render_profile
from quip_solver_dwave.session_loop import capabilities_message, run_session_sync


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="quip-dwave-qa",
        description="QuIP D-Wave quantum-annealing miner (v0.3 protocol)",
    )
    p.add_argument("--quip-coordinator", default=None, help="unix:// path or host")
    p.add_argument("--miner-id", default="qpu-0")
    p.add_argument(
        "--capabilities",
        action="store_true",
        help="print capability JSON and exit",
    )
    p.add_argument(
        "--check",
        action="store_true",
        help="verify Ocean import + D-Wave credentials, then exit",
    )
    p.add_argument(
        "--version",
        action="store_true",
        help="print version and exit",
    )
    p.add_argument(
        "--log-level",
        default="info",
        help="logging level (default: info)",
    )
    p.add_argument(
        "--mock",
        action="store_true",
        help="force offline mock sampler (also set by QUIP_DWAVE_MOCK=1)",
    )
    p.add_argument(
        "--dump-topology",
        metavar="PATH",
        default=None,
        help="write this solver's working graph as a coordinator topology "
        "spec and exit (feed it to `quip-coordinator seed-chain --topology`)",
    )
    p.add_argument(
        "--solver",
        default=None,
        help="solver to capture from (default: whatever DWAVE_API_SOLVER or "
        "the SDK config selects)",
    )
    p.add_argument(
        "--compare",
        metavar="PATH",
        default=None,
        help="with --dump-topology: also diff the capture against the spec "
        "currently in force",
    )
    p.add_argument(
        "--allowed-h-milli",
        default=",".join(str(v) for v in DEFAULT_ALLOWED_H_MILLI),
        help="comma-separated allowed h values for the captured spec "
        "(network policy, not a chip property)",
    )
    p.add_argument(
        "--allowed-j-milli",
        default=",".join(str(v) for v in DEFAULT_ALLOWED_J_MILLI),
        help="comma-separated allowed J values for the captured spec",
    )
    p.add_argument(
        "--attempts-dir",
        metavar="PATH",
        default=None,
        help="coordinator attempts directory to seed round history from "
        "(default: the 'attempts' directory beside the usage database)",
    )
    p.add_argument(
        "--profile",
        action="store_true",
        help="print the day-of-month by hour-of-day QPU history and round outcomes, then exit",
    )
    p.add_argument(
        "--usage-db",
        metavar="PATH",
        default=DEFAULT_USAGE_DB,
        help="usage database to read for --profile "
        f"(default: {DEFAULT_USAGE_DB})",
    )
    return p


def _milli_list(raw: str, flag: str) -> list[int]:
    try:
        values = [int(part) for part in raw.split(",") if part.strip()]
    except ValueError as exc:
        raise SystemExit(f"{flag}: expected comma-separated integers, got {raw!r}") from exc
    if not values:
        raise SystemExit(f"{flag}: needs at least one value")
    return values


def run_dump_topology(args) -> int:
    """``--dump-topology``: capture the live working graph, optionally diffed.

    Connects eagerly, unlike session mode: the point of the run is to read the
    chip.
    """
    if not ocean_importable():
        print("FAIL: dwave-ocean-sdk / dimod not importable", file=sys.stderr)
        return EXIT_ENV_INCOMPATIBLE
    sampler = OceanSampler(solver_name=args.solver, mock=False)
    try:
        sampler.ensure_connected()
    except Exception as exc:  # noqa: BLE001 - operator-facing message, not a trace
        print(f"FAIL: could not reach the solver: {exc}", file=sys.stderr)
        return EXIT_ENV_INCOMPATIBLE
    try:
        spec = capture_spec(
            sampler.live_nodes,
            sampler.live_edges,
            allowed_h_milli=_milli_list(args.allowed_h_milli, "--allowed-h-milli"),
            allowed_j_milli=_milli_list(args.allowed_j_milli, "--allowed-j-milli"),
        )
        write_spec(spec, args.dump_topology)
        print(
            f"captured {len(spec['nodes'])} nodes and {len(spec['edges'])} couplers "
            f"to {args.dump_topology}"
        )
        print(
            f"allowed_h_milli={spec['allowed_h_milli']} "
            f"allowed_j_milli={spec['allowed_j_milli']}"
        )
        if args.compare:
            print()
            print(format_comparison(compare_specs(load_spec(args.compare), spec)))
        print()
        print(
            "The on-chain topology hash is computed by the coordinator; register "
            "with: quip-coordinator seed-chain --topology "
            f"{args.dump_topology}"
        )
    finally:
        sampler.close()
    return EXIT_CLEAN


def print_capabilities() -> None:
    """SPEC section 8: print the protobuf JSON mapping of ``Capabilities``.

    Field names are lowerCamelCase and the output stays identical to the
    in-session ``GetCapabilities`` reply — the two are one message.
    """
    print(
        json.dumps(
            MessageToDict(
                capabilities_message(),
                always_print_fields_with_no_presence=True,
                preserving_proto_field_name=False,
            )
        )
    )


def run_check(*, force_mock: bool = False) -> int:
    """``--check``: Ocean importable + (creds present OR mock mode)."""
    if not ocean_importable():
        print("FAIL: dwave-ocean-sdk / dimod not importable", file=sys.stderr)
        return EXIT_ENV_INCOMPATIBLE
    if force_mock or mock_mode_enabled():
        print("OK: ocean importable; mock mode (no live QPU required)")
        return EXIT_CLEAN
    if not credentials_present():
        print(
            "FAIL: no D-Wave credentials "
            "(set DWAVE_API_TOKEN or configure ~/.config/dwave/dwave.conf)",
            file=sys.stderr,
        )
        return EXIT_ENV_INCOMPATIBLE
    print("OK: ocean importable; credentials present")
    return EXIT_CLEAN


def run_profile(args: argparse.Namespace) -> int:
    """``--profile``: print the day-of-month by hour-of-day history and exit. No QPU, no token."""
    if not os.path.exists(args.usage_db):
        print(
            f"FAIL: no usage database at {args.usage_db} (set --usage-db)",
            file=sys.stderr,
        )
        return EXIT_CONFIG_INVALID
    store = HistoryStore(args.usage_db)
    try:
        print(render_profile(store, time.time()))
    finally:
        store.close()
    return EXIT_CLEAN


def install_sigterm_handler(sampler: SupportsClose) -> None:
    """Register a SIGTERM handler that closes ``sampler`` before exiting.

    A default-disposition SIGTERM (``kill``, orchestrator shutdown) tears
    down the interpreter without unwinding the stack, so it skips
    ``session_loop.run_session``'s normal-exit ``finally`` block and leaks
    the D-Wave cloud client/session until its own idle timeout. Mirrors
    ``QPU/dwave_miner.py``'s ``_cleanup_handler``: close the sampler, then
    raise ``SystemExit`` so ``main``'s existing ``except SystemExit`` path
    (and, if the signal lands while a session is running, ``run_session``'s
    ``finally``) still gets a chance to unwind normally.

    ``signal.signal`` only accepts a handler on the main thread; installing
    from any other thread would raise, so this is a no-op there.
    """
    if threading.current_thread() is not threading.main_thread():
        return

    triggered = threading.Event()

    def handler(signum, frame):  # noqa: ARG001 - required signal handler signature
        if triggered.is_set():
            return
        triggered.set()
        log = logging.getLogger(__name__)
        log.info("SIGTERM received; closing D-Wave sampler")
        try:
            sampler.close()
        except Exception:  # noqa: BLE001 - best-effort cleanup before exit
            log.exception("sampler.close() failed during SIGTERM shutdown")
        raise SystemExit(EXIT_CLEAN)

    signal.signal(signal.SIGTERM, handler)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    if args.version:
        print(f"quip-dwave-qa {__version__} protocol 1")
        return EXIT_CLEAN
    if args.capabilities:
        print_capabilities()
        return EXIT_CLEAN
    if args.profile:
        return run_profile(args)
    # Before anything reads credentials: a v0.2 node delivers the token under
    # the pre-Ocean name, and both --check and the session path resolve it
    # through the SDK.
    adopt_legacy_token_env()

    if args.dump_topology:
        return run_dump_topology(args)

    if args.check:
        return run_check(force_mock=args.mock)

    if not args.quip_coordinator:
        print("--quip-coordinator required for session mode", file=sys.stderr)
        return EXIT_CONFIG_INVALID

    use_mock = args.mock or mock_mode_enabled()
    try:
        # Real-QPU construction does not connect; the connection is deferred
        # until the coordinator sends Configure (or --check forces it). Mock
        # mode is ready immediately.
        sampler = OceanSampler(mock=use_mock)
    except Exception as exc:  # noqa: BLE001
        logging.getLogger(__name__).exception("sampler init failed: %s", exc)
        return EXIT_ENV_INCOMPATIBLE

    install_sigterm_handler(sampler)

    try:
        return run_session_sync(
            args.quip_coordinator,
            args.miner_id,
            sampler,
            attempts_dir=args.attempts_dir,
        )
    except SystemExit as e:
        code = e.code if isinstance(e.code, int) else EXIT_INTERNAL_FATAL
        return code
    except Exception:
        logging.getLogger(__name__).exception("fatal")
        return EXIT_INTERNAL_FATAL


if __name__ == "__main__":
    sys.exit(main())
