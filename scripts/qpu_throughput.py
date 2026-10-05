#!/usr/bin/env python3
"""QPU throughput with several jobs in flight, on one audited set of models.

``round2_capture.py`` submits one job and waits for it, so its wall time per
call (about 1.5 s) hides how much of a call is the QPU and how much is the
round trip. This script runs the same native problems at a fixed set of
``(reads, concurrency)`` arms and measures the end-to-end models per second
when that many calls overlap.

``manifest``  Reads a job spec (one cell, its nonces, one requested scale,
              one anneal time, the arms, an order seed, and the same source
              roots as a Round 2 job spec) and writes every job of every arm.
              Each arm is one pass over all models in one seeded order that
              every arm shares. ``round2_capture.py approve`` records a cap
              against the manifest's sha256.
``capture``   Runs the arms one after another. Inside an arm, at most
              ``concurrency`` jobs are in flight; each job runs
              ``OceanSampler.sample`` on its own worker thread, which blocks
              until SAPI's future resolves, so the in-flight count is the
              number of busy workers. Each job reserves its estimate in the
              spend journal before its submit and charges D-Wave's reported
              access time after its answer, both under one lock, using
              ``round2_capture``'s journal, failed-capture file, and mode
              marker.
``analyze``   Reports makespan, models per second, wall time per job, access
              time, and the access-limited rate per arm, and compares each
              model's best energy against a reference capture.

``--mock`` uses ``round2_capture``'s offline sampler behind a fixed latency
and a single mock QPU that serves one job at a time, so overlap is visible
with no network. A mock run needs no approval and writes no spend journal.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import importlib.util
import json
import random
import statistics
import sys
import threading
import time
import uuid
from collections import deque
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from quip_miner_dwave import regime_io, regimes, replay, round2_io
from quip_miner_dwave.ocean import OceanSampler


def _load_round2_capture():
    path = Path(__file__).resolve().parent / "round2_capture.py"
    spec = importlib.util.spec_from_file_location("round2_capture", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["round2_capture"] = module
    spec.loader.exec_module(module)
    return module


capture_tools = _load_round2_capture()

LABEL = "quip-qpu-throughput"
ANALYSIS_FILE = "analysis.json"

# The mock's timing. A call waits MOCK_LATENCY_S, which overlaps across calls
# the way a network round trip and SAPI's queue do, then holds the one mock
# QPU for its access time. The access model is a line through the measured
# cubic-dimer-pm1 access times at 400 us: about 0.07 s at 64 reads and 0.6 s
# at 1024 reads.
MOCK_LATENCY_S = 1.5
MOCK_PROGRAMMING_US = 35_000
MOCK_PER_READ_US = 552


# ---------------------------------------------------------------- manifest


def arm_name(reads: int, concurrency: int) -> str:
    return f"reads-{reads}-conc-{concurrency}"


def throughput_capture_key(model_capture_key: str, concurrency: int) -> str:
    """A job's identity: ``build_job``'s key, which already holds the reads, plus the concurrency."""
    return round2_io.run_id({"model_capture_key": model_capture_key, "concurrency": concurrency})


def _context(sources: dict[str, Any]) -> Any:
    """``round2_capture.Context`` from a job spec's or a manifest's source paths."""
    properties_path = Path(sources["properties_path"])
    spec_path = sources.get("spec_path")
    embeddings_dir = sources.get("embeddings_dir")
    return capture_tools.Context(
        bundles_root=Path(sources["bundles_root"]),
        range_audit_dir=Path(sources["range_audit_dir"]),
        spec=replay.load_spec(spec_path) if spec_path else None,
        embeddings_dir=Path(embeddings_dir) if embeddings_dir else None,
        properties=json.loads(properties_path.read_text(encoding="utf-8")),
        properties_hash=hashlib.sha256(properties_path.read_bytes()).hexdigest(),
    )


def build_manifest(job_spec: dict[str, Any]) -> dict[str, Any]:
    ctx = _context(job_spec)
    cell = str(job_spec["cell"])
    requested_scale = float(job_spec["requested_scale"])
    anneal_us = int(job_spec["anneal_us"])
    margin = float(job_spec["reservation_margin"])
    arms = [{"reads": int(arm["reads"]), "concurrency": int(arm["concurrency"])} for arm in job_spec["arms"]]
    if any(arm["concurrency"] < 1 or arm["reads"] < 1 for arm in arms):
        raise ValueError("every arm needs at least 1 read and a concurrency of at least 1")

    order_seed = int(job_spec["order_seed"])
    model_order = sorted(str(nonce) for nonce in job_spec["nonces"])
    random.Random(order_seed).shuffle(model_order)

    # build_job reads the bundle, the embedding, and the range audit; its
    # result depends on the reads but not on the concurrency.
    built: dict[tuple[str, int], dict[str, Any]] = {}
    jobs = []
    for arm in arms:
        name = arm_name(arm["reads"], arm["concurrency"])
        for nonce in model_order:
            key = (nonce, arm["reads"])
            if key not in built:
                built[key] = capture_tools.build_job(
                    cell, nonce, requested_scale, anneal_us, arm["reads"], margin, ctx
                )
            base = built[key]
            jobs.append({
                **base,
                "arm": name,
                "concurrency": arm["concurrency"],
                "model_capture_key": base["capture_key"],
                "capture_key": throughput_capture_key(base["capture_key"], arm["concurrency"]),
            })

    keys = [job["capture_key"] for job in jobs]
    if len(set(keys)) != len(keys):
        raise ValueError("two jobs in this job spec produced the same capture key; refusing to build a manifest")
    job_order_hash = hashlib.sha256(json.dumps(keys).encode()).hexdigest()

    arm_totals = []
    for arm in arms:
        name = arm_name(arm["reads"], arm["concurrency"])
        arm_jobs = [job for job in jobs if job["arm"] == name]
        arm_totals.append({
            **arm,
            "arm": name,
            "jobs": len(arm_jobs),
            "estimate_us": sum(job["estimate_us"] for job in arm_jobs),
            "reservation_us": sum(job["reservation_us"] for job in arm_jobs),
        })

    return {
        "created_at": datetime.now(UTC).isoformat(),
        "cell": cell,
        "requested_scale": requested_scale,
        "anneal_us": anneal_us,
        "reservation_margin": margin,
        "arms": arm_totals,
        "sources": {
            "bundles_root": str(job_spec["bundles_root"]),
            "range_audit_dir": str(job_spec["range_audit_dir"]),
            "properties_path": str(job_spec["properties_path"]),
            "properties_hash": ctx.properties_hash,
            "spec_path": job_spec.get("spec_path"),
            "embeddings_dir": job_spec.get("embeddings_dir"),
        },
        "model_order_seed": order_seed,
        "model_order": model_order,
        "jobs": jobs,
        "job_order": keys,
        "job_order_hash": job_order_hash,
        "total_estimate_us": sum(job["estimate_us"] for job in jobs),
        "total_reservation_us": sum(job["reservation_us"] for job in jobs),
    }


def cmd_manifest(args: argparse.Namespace) -> int:
    job_spec = json.loads(Path(args.job_spec).read_text(encoding="utf-8"))
    manifest = build_manifest(job_spec)
    regime_io.atomic_write_json(args.out, manifest)
    manifest_hash = hashlib.sha256(Path(args.out).read_bytes()).hexdigest()
    for arm in manifest["arms"]:
        print(
            f"{arm['arm']}: {arm['jobs']} jobs, {arm['estimate_us'] / 1e6:.1f} s estimated, "
            f"{arm['reservation_us'] / 1e6:.1f} s reserved"
        )
    print(f"wrote {args.out} ({len(manifest['jobs'])} jobs)")
    print(
        f"total: {manifest['total_estimate_us'] / 1e6:.1f} s estimated, "
        f"{manifest['total_reservation_us'] / 1e6:.1f} s reserved"
    )
    print(f"manifest hash: {manifest_hash}")
    return 0


# ----------------------------------------------------------------- sampler


# The answer metadata of the job on this thread. OceanSampler.sample reads
# the answer on the calling thread and returns no timing dict or problem id,
# so the override below parks them here for the same thread to pick up.
_ANSWER = threading.local()


def answer_metadata(raw: Any) -> tuple[dict[str, Any], str | None]:
    """The timing dict and the problem id of a cloud ``Future`` or a ``dimod.SampleSet``."""
    if getattr(raw, "record", None) is not None:
        info = getattr(raw, "info", None) or {}
        timing = info.get("timing") or {}
        problem_id = info.get("problem_id")
    else:
        timing = raw.timing or {}
        problem_id = raw.id
    return {str(k): float(v) for k, v in timing.items()}, (None if problem_id is None else str(problem_id))


class RecordingOceanSampler(OceanSampler):
    """``OceanSampler`` that also keeps each answer's timing dict and problem id."""

    @staticmethod
    def _decode_and_view(future_or_ss: Any):
        view = OceanSampler._decode_and_view(future_or_ss)
        _ANSWER.timing, _ANSWER.problem_id = answer_metadata(future_or_ss)
        return view


class LatencySampler:
    """The mock QPU: a fixed round trip that overlaps, and an access time served one job at a time."""

    def __init__(self, inner: Any, latency_s: float = MOCK_LATENCY_S,
                 programming_us: int = MOCK_PROGRAMMING_US, per_read_us: int = MOCK_PER_READ_US) -> None:
        self._inner = inner
        self._latency_s = latency_s
        self._programming_us = programming_us
        self._per_read_us = per_read_us
        self._qpu = threading.Lock()
        self.properties: dict = {}

    def sample_ising(self, h, j, **kwargs):
        import dimod

        reads = int(kwargs.get("num_reads") or 1)
        sampling_us = self._per_read_us * reads
        access_us = self._programming_us + sampling_us
        time.sleep(self._latency_s)
        with self._qpu:
            time.sleep(access_us / 1e6)
        sampleset = self._inner.sample_ising(h, j, **kwargs)
        timing = {
            "qpu_programming_time": float(self._programming_us),
            "qpu_sampling_time": float(sampling_us),
            "qpu_access_time": float(access_us),
        }
        return dimod.SampleSet(sampleset.record, sampleset.variables, {"timing": timing}, sampleset.vartype)


def connect(mock: bool, workers: int) -> Any:
    """The sampler of a run. Only the real one reaches the QPU."""
    if mock:
        return RecordingOceanSampler(
            sampler=LatencySampler(capture_tools.OfflineSampler()), mock=True, submit_workers=workers
        )
    return RecordingOceanSampler(mock=False, submit_workers=workers)


# ------------------------------------------------------------------ capture


def output_path(round_dir: Path, job: dict[str, Any]) -> Path:
    """``round2_capture``'s layout under one directory per arm, so arms never share a file."""
    return capture_tools._output_path(round_dir / job["arm"], job)


def check_resumable(
    jobs: list[dict[str, Any]], round_dir: Path, failed: dict[str, Any], open_keys: set, charged_keys: set
) -> list[dict[str, Any]]:
    """The jobs still to capture, or a refusal for any job whose outcome is uncertain.

    The same three refusals as ``round2_capture.run_capture``, checked for
    every job before the first reservation rather than one job at a time.
    """
    pending = []
    for job in jobs:
        key = job["capture_key"]
        label = f"{job['arm']} {job['cell']}/{job['nonce']}: capture key {key}"
        if key in failed:
            raise SystemExit(
                f"{label} previously failed ({failed[key]['reason']}); this runner never retries an uncertain "
                f"submission automatically. Resolve it and remove it from "
                f"{round_dir / capture_tools.FAILED_FILE} before running again"
            )
        if key in open_keys:
            raise SystemExit(
                f"{label} has an open reservation in {round_dir / regime_io.SPEND_JOURNAL} with no matching "
                "charge, from a run that did not finish. This runner never retries an uncertain submission "
                "automatically. Confirm with D-Wave whether it ran, then append a charge line or a "
                "failed-capture record before running again"
            )
        out_path = output_path(round_dir, job)
        if out_path.exists() and capture_tools._output_matches(out_path, key):
            continue
        if key in charged_keys:
            raise SystemExit(
                f"{label} has a charge in {round_dir / regime_io.SPEND_JOURNAL} with no output file to show "
                "for it. The QPU already ran and was already paid for; recover the reads by hand if possible, "
                f"or accept the loss and record capture key {key} as failed before running again"
            )
        pending.append(job)
    return pending


def validate(manifest: dict[str, Any], ctx: Any) -> dict[str, dict[str, Any]]:
    """Every model's physical problem, hash-checked against the manifest, keyed by nonce.

    ``round2_capture.validate_and_load`` checks one job per model; every
    other job of that model must name the same audited problem.
    """
    first: dict[str, dict[str, Any]] = {}
    for job in manifest["jobs"]:
        first.setdefault(job["nonce"], job)
    identity = ("cell", "model_hash", "embedding_hash", "requested_scale", "coefficient_hash", "anneal_us")
    for job in manifest["jobs"]:
        reference = first[job["nonce"]]
        if any(job[field] != reference[field] for field in identity):
            raise SystemExit(f"{job['arm']} {job['nonce']}: the jobs of one model name different problems")
    view = {"sources": manifest["sources"], "jobs": list(first.values())}
    loaded = capture_tools.validate_and_load(view, ctx)
    return {nonce: loaded[job["capture_key"]] for nonce, job in first.items()}


@dataclass
class ArmRun:
    """What every job of one arm shares."""

    round_dir: Path
    sampler: Any
    loaded: dict[str, dict[str, Any]]
    mock: bool
    label: str
    run_id: str
    capture_setup_s: float
    journal_lock: threading.Lock


def _record_failed(run: ArmRun, key: str, exc: BaseException) -> None:
    if not run.mock:
        with run.journal_lock:
            capture_tools.record_failed(run.round_dir, key, f"{type(exc).__name__}: {exc}")


def _charge(run: ArmRun, job: dict[str, Any], charged_us: int) -> None:
    """Close the job's reservation with D-Wave's charge, following ``round2_capture``'s order.

    With no usable access time, the key is marked failed first and then
    charged at its reservation, so a kill between the two writes leaves the
    key open or failed, never closed and retryable.
    """
    key = job["capture_key"]
    with run.journal_lock:
        if charged_us <= 0:
            capture_tools.record_failed(run.round_dir, key, "unknown charge: the solver reported no access time")
            regime_io.append_spend(
                run.round_dir, regime_io.CHARGE, job["cell"], key, job["anneal_us"], int(job["reservation_us"])
            )
            raise SystemExit(
                f"{job['arm']} {job['cell']}/{job['nonce']}: the solver reported no QPU access time for capture "
                f"key {key}; charged the reservation amount and marked it failed"
            )
        regime_io.append_spend(run.round_dir, regime_io.CHARGE, job["cell"], key, job["anneal_us"], charged_us)


def run_job(run: ArmRun, job: dict[str, Any], job_started_ns: int) -> int:
    """Submit, charge, decode, and save one job on a worker thread; return the charged access time."""
    key = job["capture_key"]
    data = run.loaded[job["nonce"]]
    _ANSWER.timing, _ANSWER.problem_id = None, None
    submitted_at_utc = datetime.now(UTC).isoformat()
    submitted_ns = time.monotonic_ns()
    try:
        result = run.sampler.sample(
            data["phys_nodes"], data["h_scaled"], data["phys_edges"], data["j_scaled"],
            num_reads=job["reads"], anneal_time_us=job["anneal_us"],
            nonce_seed=bytes.fromhex(job["nonce"]), label=run.label, auto_scale=False,
        )
        resolved_ns = time.monotonic_ns()
    except BaseException as exc:
        _record_failed(run, key, exc)
        raise
    timing = _ANSWER.timing or {}
    problem_id = _ANSWER.problem_id
    # D-Wave bills qpu_access_time. The wrapper's figure is programming plus
    # sampling; the larger of the two is the safe charge.
    access_us = max(int(result.device_access_time_us), int(timing.get("qpu_access_time", 0)))
    if not run.mock:
        _charge(run, job, access_us)
    try:
        model = capture_tools._model_for(job, data)
        spins, broken = regimes.decode_reads(model, result.spins, result.variables)
        if len(spins) != job["reads"]:
            raise SystemExit(
                f"{job['arm']} {job['cell']}/{job['nonce']} holds {len(spins)} rows, and the job asked for "
                f"{job['reads']} reads. The answer format of the solver has changed, so the run stops"
            )
        energies = regimes.score(model, spins)
        scored_usable_ns = time.monotonic_ns()
        timing_points = (job_started_ns, submitted_ns, resolved_ns, scored_usable_ns)
        if timing_points != tuple(sorted(timing_points)):
            raise ValueError("invalid monotonic timestamp order for captured answer")
        _save(run, job, data, result, spins, broken, energies, access_us, timing, problem_id,
              submitted_at_utc, job_started_ns, submitted_ns, resolved_ns, scored_usable_ns)
    except BaseException as exc:
        _record_failed(run, key, exc)
        raise
    return access_us


def _save(run: ArmRun, job, data, result, spins, broken, energies, access_us, timing, problem_id,
          submitted_at_utc, job_started_ns, submitted_ns, resolved_ns, scored_usable_ns) -> None:
    """Every field ``round2_capture`` saves, then this script's throughput fields."""
    sapi_ms = result.extra.get("sapi_ms")
    timing_known = result.submitted_on_s is not None and result.solved_on_s is not None
    out_path = output_path(run.round_dir, job)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    regime_io.atomic_savez(
        out_path,
        spins=spins,
        raw_spins=np.asarray(result.spins),
        raw_variables=np.asarray(result.variables, dtype=np.int64),
        energies=energies,
        access_us=np.int64(access_us),
        device_access_s=np.float64(access_us / 1e6),
        capture_setup_s=np.float64(run.capture_setup_s),
        job_setup_s=np.float64((submitted_ns - job_started_ns) / 1e9),
        sampler_call_s=np.float64((resolved_ns - submitted_ns) / 1e9),
        decode_score_s=np.float64((scored_usable_ns - resolved_ns) / 1e9),
        end_to_end_s=np.float64((scored_usable_ns - job_started_ns) / 1e9),
        job_started_monotonic_s=np.float64(job_started_ns / 1e9),
        submitted_monotonic_s=np.float64(submitted_ns / 1e9),
        scored_usable_monotonic_s=np.float64(scored_usable_ns / 1e9),
        submitted_at_utc=submitted_at_utc,
        model_setup_cache_status="cached",
        cold_model_setup_s=np.float64(np.nan),
        cold_model_setup_unavailable_reason=(
            "physical model and embedding were prepared before capture; "
            "cold cache construction was not measured in this process"
        ),
        anneal_us=np.int64(job["anneal_us"]),
        mock=np.bool_(run.mock),
        requested_scale=np.float64(job["requested_scale"]),
        submitted_scale=np.float64(job["submitted_scale"]),
        auto_scale=np.bool_(False),
        coefficient_hash=job["coefficient_hash"],
        request_bytes_hash=capture_tools.request_identity_hash(
            data["phys_nodes"], data["h_scaled"], data["phys_edges"], data["j_scaled"]
        ),
        ceiling=np.float64(job["ceiling"]),
        limiting_constraint=json.dumps(job["limiting_constraint"]),
        chain_strength=np.float64(job["chain_strength"] or 0.0),
        chain_strength_ratio_to_logical=np.float64(
            job["chain_strength_ratio_to_logical"] if job["chain_strength_ratio_to_logical"] is not None else np.nan
        ),
        broken_chain_share=np.float64(broken),
        returned_scaling_present=np.bool_(False),
        returned_scaling_reason=(
            "auto_scale=False; the OceanSampler wrapper does not expose any returned scaling "
            "metadata for this submission"
        ),
        submitted_on_s=np.float64(result.submitted_on_s) if result.submitted_on_s is not None else np.nan,
        solved_on_s=np.float64(result.solved_on_s) if result.solved_on_s is not None else np.nan,
        sapi_ms=np.float64(sapi_ms) if sapi_ms is not None else np.nan,
        timing_unavailable_reason=(
            "" if timing_known else
            "mock or non-cloud sampler; SAPI's submitted_on/solved_on/sapi_ms need a real cloud Future"
        ),
        capture_key=job["capture_key"],
        model_hash=job["model_hash"],
        # Throughput fields.
        arm=job["arm"],
        reads=np.int64(job["reads"]),
        concurrency=np.int64(job["concurrency"]),
        model_capture_key=job["model_capture_key"],
        run_id=run.run_id,
        resolved_monotonic_s=np.float64(resolved_ns / 1e9),
        wall_s=np.float64((resolved_ns - submitted_ns) / 1e9),
        qpu_access_time_us=np.int64(access_us),
        timing_json=json.dumps(timing, sort_keys=True),
        problem_id=problem_id or "",
        best_energy=np.float64(np.min(energies)),
    )


def capture_arm(
    run: ArmRun, jobs: list[dict[str, Any]], concurrency: int, cap_seconds: float | None
) -> tuple[int, int]:
    """Run one arm's pending jobs with at most ``concurrency`` in flight.

    Returns ``(status, captured)``: status 0, or 3 when the next reservation
    would pass the cap.

    A job is in flight from its reservation until its worker returns. After
    any failure no new job is submitted, and every job already in flight is
    still waited for, so its charge reaches the journal before this raises.
    """
    queue = deque(jobs)
    in_flight: dict[Future, dict[str, Any]] = {}
    status = 0
    captured = 0
    error: BaseException | None = None
    with ThreadPoolExecutor(max_workers=concurrency, thread_name_prefix="qpu-throughput") as pool:
        while True:
            while error is None and status == 0 and queue and len(in_flight) < concurrency:
                job = queue[0]
                job_started_ns = time.monotonic_ns()
                if not run.mock:
                    assert cap_seconds is not None
                    with run.journal_lock:
                        spent, _ = regime_io.read_spend(run.round_dir)
                        if (spent + job["reservation_us"]) / 1e6 > cap_seconds:
                            print(
                                f"stopping before {job['arm']} {job['nonce']}: the round has {spent / 1e6:.2f} s "
                                "charged plus reserved, and this job's reservation would pass the approved cap "
                                f"of {cap_seconds:.1f} s",
                                file=sys.stderr,
                            )
                            status = 3
                            break
                        regime_io.append_spend(
                            run.round_dir, regime_io.SUBMIT, job["cell"], job["capture_key"], job["anneal_us"],
                            int(job["reservation_us"]),
                        )
                queue.popleft()
                in_flight[pool.submit(run_job, run, job, job_started_ns)] = job
            if not in_flight:
                break
            try:
                done, _ = wait(in_flight, return_when=FIRST_COMPLETED)
            except BaseException as exc:
                # An interrupt stops new submissions and drains the rest; a
                # second one leaves their reservations open in the journal.
                if error is not None:
                    raise
                error = exc
                continue
            for future in done:
                in_flight.pop(future)
                exc = future.exception()
                if exc is None:
                    captured += 1
                elif error is None:
                    error = exc
    if error is not None:
        raise error
    return status, captured


def run_capture(args: argparse.Namespace) -> int:
    manifest_path = Path(args.manifest)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    # A mock run spends nothing and journals nothing, so it needs no approval.
    cap_seconds = None if args.mock else capture_tools.check_approval(manifest_path, Path(args.approval))

    round_dir = Path(args.round_dir)
    round_dir.mkdir(parents=True, exist_ok=True)
    capture_tools.check_mode_marker(round_dir, args.mock)

    ctx = _context(manifest["sources"])
    loaded = validate(manifest, ctx)

    if not args.mock and not args.yes:
        print(f"{len(manifest['jobs'])} jobs manifested; pass --yes to spend against the approved cap")
        return 2

    failed = capture_tools.load_failed(round_dir)
    open_keys, charged_keys = capture_tools.journal_capture_status(round_dir)
    pending = check_resumable(manifest["jobs"], round_dir, failed, open_keys, charged_keys)

    capture_setup_started_ns = time.monotonic_ns()
    workers = max(arm["concurrency"] for arm in manifest["arms"])
    sampler = connect(args.mock, workers)
    captured = 0
    try:
        sampler.ensure_connected()
        # One job per model is enough: every job of a model is the same problem.
        first_jobs: dict[str, dict[str, Any]] = {}
        for job in manifest["jobs"]:
            first_jobs.setdefault(job["nonce"], job)
        view = {"jobs": list(first_jobs.values())}
        by_key = {job["capture_key"]: loaded[job["nonce"]] for job in view["jobs"]}
        capture_tools.preflight(view, by_key, sampler.live_nodes, sampler.live_edges, args.mock)
        if manifest["model_order"]:
            first = loaded[manifest["model_order"][0]]
            sampler.set_session_topology(
                [int(q) for q in first["phys_nodes"]], [(int(u), int(v)) for u, v in first["phys_edges"]]
            )
        run = ArmRun(
            round_dir=round_dir, sampler=sampler, loaded=loaded, mock=args.mock, label=args.label,
            run_id=uuid.uuid4().hex, capture_setup_s=(time.monotonic_ns() - capture_setup_started_ns) / 1e9,
            journal_lock=threading.Lock(),
        )
        for arm in manifest["arms"]:
            arm_jobs = [job for job in pending if job["arm"] == arm["arm"]]
            print(f"{arm['arm']}: {len(arm_jobs)} of {arm['jobs']} jobs to capture", flush=True)
            started = time.monotonic()
            status, arm_captured = capture_arm(run, arm_jobs, arm["concurrency"], cap_seconds)
            captured += arm_captured
            print(f"{arm['arm']}: done in {time.monotonic() - started:.1f} s", flush=True)
            if status:
                return status
    finally:
        sampler.close()
    spent, _ = regime_io.read_spend(round_dir)
    print(f"captured {captured} of {len(pending)} pending jobs; the round has {spent / 1e6:.2f} s charged")
    return 0


def cmd_capture(args: argparse.Namespace) -> int:
    if not args.mock and args.approval is None:
        raise SystemExit("--approval is required unless --mock is given")
    round_dir = Path(args.round_dir)
    round_dir.mkdir(parents=True, exist_ok=True)
    # One capture owns the round through reservation and charging, as in
    # round2_capture.cmd_capture.
    with open(round_dir / ".capture.lock", "a", encoding="utf-8") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise SystemExit("another capture owns this round; wait for it to finish") from exc
        return run_capture(args)


# ----------------------------------------------------------------- analysis


def arm_stats(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Throughput of one arm from its per-job records.

    Each record has ``run_id``, ``submitted_s``, ``resolved_s``, ``wall_s``,
    and ``access_us``. Monotonic times compare only inside one process, so a
    resumed arm's makespan is the sum of each run's first-submit to
    last-resolve span.
    """
    n = len(records)
    spans: dict[str, tuple[float, float]] = {}
    for record in records:
        lo, hi = spans.get(record["run_id"], (record["submitted_s"], record["resolved_s"]))
        spans[record["run_id"]] = (min(lo, record["submitted_s"]), max(hi, record["resolved_s"]))
    makespan_s = sum(hi - lo for lo, hi in spans.values())
    walls = [float(record["wall_s"]) for record in records]
    access_s = [float(record["access_us"]) / 1e6 for record in records]
    total_access_s = sum(access_s)
    achieved = n / makespan_s
    access_limited = n / total_access_s
    return {
        "jobs": n,
        "runs": len(spans),
        "makespan_s": makespan_s,
        "models_per_s": achieved,
        "wall_median_s": statistics.median(walls),
        "wall_p90_s": float(np.percentile(walls, 90)),
        "access_total_s": total_access_s,
        "access_median_s": statistics.median(access_s),
        "access_limited_models_per_s": access_limited,
        "access_limited_over_achieved": access_limited / achieved,
        "achieved_over_access_limited": achieved / access_limited,
    }


def quality_vs_reference(best: dict[str, float], reference: dict[str, float]) -> dict[str, Any]:
    """Per-model best energy against a reference capture's best, over the models both hold."""
    shared = sorted(set(best) & set(reference))
    if not shared:
        return {"models": 0}
    diffs = [best[nonce] - reference[nonce] for nonce in shared]
    return {
        "models": len(shared),
        "median_diff": statistics.median(diffs),
        "mean_diff": statistics.fmean(diffs),
        "share_equal": sum(abs(d) <= 1e-9 for d in diffs) / len(shared),
        "share_at_or_below": sum(d <= 1e-9 for d in diffs) / len(shared),
    }


def load_records(round_dir: Path, jobs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    records = []
    for job in jobs:
        path = output_path(round_dir, job)
        if not path.exists():
            continue
        with np.load(path) as saved:
            if str(saved["capture_key"]) != job["capture_key"]:
                continue
            records.append({
                "nonce": job["nonce"],
                "run_id": str(saved["run_id"]),
                "submitted_s": float(saved["submitted_monotonic_s"]),
                "resolved_s": float(saved["resolved_monotonic_s"]),
                "wall_s": float(saved["wall_s"]),
                "access_us": int(saved["qpu_access_time_us"]),
                "best_energy": float(saved["best_energy"]),
            })
    return records


def reference_best(reference_round: Path, jobs: list[dict[str, Any]]) -> dict[str, float]:
    """Each model's best energy in a ``round2_capture`` round, matched on nonce and model hash."""
    best = {}
    for job in jobs:
        path = capture_tools._output_path(reference_round, job)
        if not path.exists():
            continue
        with np.load(path) as saved:
            if str(saved["model_hash"]) == job["model_hash"]:
                best[job["nonce"]] = float(np.min(saved["energies"]))
    return best


def analyze(manifest: dict[str, Any], round_dir: Path, reference_round: Path | None) -> dict[str, Any]:
    arms = []
    for arm in manifest["arms"]:
        jobs = [job for job in manifest["jobs"] if job["arm"] == arm["arm"]]
        records = load_records(round_dir, jobs)
        entry: dict[str, Any] = {"arm": arm["arm"], "reads": arm["reads"], "concurrency": arm["concurrency"]}
        if records:
            entry.update(arm_stats(records))
            if reference_round is not None:
                best = {record["nonce"]: record["best_energy"] for record in records}
                entry["vs_reference"] = quality_vs_reference(best, reference_best(reference_round, jobs))
        else:
            entry["jobs"] = 0
        arms.append(entry)
    return {"round_dir": str(round_dir), "reference_round": str(reference_round) if reference_round else None,
            "arms": arms}


def format_table(analysis: dict[str, Any]) -> str:
    header = (
        f"{'arm':<20} {'jobs':>4} {'makespan_s':>10} {'models/s':>8} {'wall_med':>8} {'wall_p90':>8} "
        f"{'acc_sum_s':>9} {'acc_med_s':>9} {'acc_lim/s':>9} {'lim/ach':>7} {'ach/lim':>7} "
        f"{'dE_med':>8} {'<=ref':>5}"
    )
    lines = [header]
    for arm in analysis["arms"]:
        if not arm["jobs"]:
            lines.append(f"{arm['arm']:<20} {0:>4}")
            continue
        ref = arm.get("vs_reference") or {}
        de = f"{ref['median_diff']:>8.2f}" if ref.get("models") else f"{'-':>8}"
        share = f"{ref['share_at_or_below']:>5.2f}" if ref.get("models") else f"{'-':>5}"
        lines.append(
            f"{arm['arm']:<20} {arm['jobs']:>4} {arm['makespan_s']:>10.2f} {arm['models_per_s']:>8.3f} "
            f"{arm['wall_median_s']:>8.3f} {arm['wall_p90_s']:>8.3f} {arm['access_total_s']:>9.2f} "
            f"{arm['access_median_s']:>9.3f} {arm['access_limited_models_per_s']:>9.3f} "
            f"{arm['access_limited_over_achieved']:>7.2f} {arm['achieved_over_access_limited']:>7.2f} {de} {share}"
        )
    return "\n".join(lines)


def cmd_analyze(args: argparse.Namespace) -> int:
    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    round_dir = Path(args.round_dir)
    analysis = analyze(manifest, round_dir, Path(args.reference_round) if args.reference_round else None)
    regime_io.atomic_write_json(round_dir / ANALYSIS_FILE, analysis)
    print(format_table(analysis))
    print(f"wrote {round_dir / ANALYSIS_FILE}")
    return 0


# --------------------------------------------------------------------- CLI


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)

    manifest = sub.add_parser("manifest", help="Build the immutable job manifest from a job-spec JSON.")
    manifest.add_argument("--job-spec", required=True)
    manifest.add_argument("--out", required=True)
    manifest.set_defaults(func=cmd_manifest)

    capture = sub.add_parser("capture", help="Run every arm, with each arm's jobs overlapped.")
    capture.add_argument("--manifest", required=True)
    capture.add_argument("--approval", help="round2_capture.py approve's file; required unless --mock")
    capture.add_argument("--round-dir", required=True)
    capture.add_argument("--label", default=LABEL)
    capture.add_argument("--yes", action="store_true", help="confirm the QPU spend")
    capture.add_argument("--mock", action="store_true", help="use the offline mock sampler and spend nothing")
    capture.set_defaults(func=cmd_capture)

    analyze_cmd = sub.add_parser("analyze", help="Report throughput per arm and quality against a reference.")
    analyze_cmd.add_argument("--manifest", required=True)
    analyze_cmd.add_argument("--round-dir", required=True)
    analyze_cmd.add_argument("--reference-round", help="a round2_capture round holding the same models")
    analyze_cmd.set_defaults(func=cmd_analyze)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
