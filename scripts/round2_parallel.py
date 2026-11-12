#!/usr/bin/env python3
"""Resume a frozen CPU campaign with parallel, quality-only subprocesses."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import fcntl
import json
import os
from pathlib import Path
from queue import Empty, Queue
import sys
from threading import Event
import time
from typing import Any, Callable, TypeVar

from quip_solver_dwave import regime_io, round2_io
from quip_solver_dwave import round2_runner as runner

T = TypeVar("T")


def dispatch(jobs: list[T], cpus: list[int], run: Callable[[T, int], None]) -> None:
    if not cpus or len(set(cpus)) != len(cpus):
        raise ValueError("workers require distinct CPUs")
    pending: Queue[T] = Queue()
    for job in jobs:
        pending.put(job)
    stopped = Event()

    def worker(cpu: int) -> None:
        while not stopped.is_set():
            try:
                job = pending.get_nowait()
            except Empty:
                return
            try:
                run(job, cpu)
            except BaseException:
                stopped.set()
                raise

    with ThreadPoolExecutor(max_workers=len(cpus)) as pool:
        futures = [pool.submit(worker, cpu) for cpu in cpus]
        for future in futures:
            future.result()


def quality_record(record: dict[str, Any], *, cpu: int | None, workers: int | None,
                   elapsed: float | None, position: int) -> dict[str, Any]:
    record = dict(record)
    record.update(
        child_end_to_end_s=record.get("end_to_end_s"),
        operational_elapsed_s=elapsed, end_to_end_s=None, caller_end_to_end_s=None,
        sequence_position=position,
        timing_evidence={"available": False, "reason": "parallel quality-only execution"},
        host={"cpu": cpu, "affinity": [] if cpu is None else [cpu], "parallel_workers": workers,
              "contamination": {"contaminated": True, "repeat_required": True,
                                "automatic_repeat": False,
                                "reasons": [{"kind": "parallel-quality-only"}]}},
    )
    return record


def ensure_no_existing_runner(root: Path) -> None:
    # Detached run-one children can outlive the sequential parent.
    for path in Path('/proc').glob('[0-9]*/cmdline'):
        try:
            args = path.read_bytes().split(b'\0')
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        if not any(arg.endswith(b'/round2_cpu.py') for arg in args):
            continue
        if b'--out' in args:
            out = Path(os.fsdecode(args[args.index(b'--out') + 1]))
            if out == root or root in out.parents:
                raise RuntimeError(f"existing campaign writer PID {path.parent.name}")


def campaign(root: Path, cpus: list[int], timeout: float, dry_run: bool) -> None:
    if len(set(cpus)) != len(cpus) or not cpus or not set(cpus) <= os.sched_getaffinity(0):
        raise ValueError("CPUs must be distinct and within allowed affinity")
    if not dry_run:
        ensure_no_existing_runner(root)
    manifest = json.loads((root / 'manifest.json').read_text())
    identity = runner.solver_identity()
    if identity != manifest['solver_identity']:
        raise ValueError('frozen solver identity changed')
    policy = runner.ComparisonPolicy.from_dict(manifest['comparison_policy'])
    bundles = Path(manifest['bundles_root'])
    tasks = []
    keys = set()
    saved = 0
    for position, row in enumerate(manifest['jobs']):
        job = runner.CpuJob.from_dict(row)
        model, arrays = round2_io.read_bundle(bundles / job.cell / job.nonce)
        if model['hash'] != job.model_hash:
            raise ValueError('frozen model changed')
        beta = runner.logical_beta_range(arrays['h'], arrays['edges'], arrays['j'])
        key = runner.run_key(job, identity, beta, policy)
        if key in keys:
            raise ValueError('duplicate manifest job')
        keys.add(key)
        run_dir = root / 'runs' / key
        if (run_dir / 'dispatch-failure.json').exists():
            raise RuntimeError(f'unresolved dispatch failure: {run_dir}')
        existing = runner.completed_result(run_dir, key)
        if existing is not None:
            if existing['status'] == 'completed' and 'caller_end_to_end_s' not in existing and not dry_run:
                # The child finished, but its parent never finalized provenance.
                # Keep quality and explicitly withhold unknown timing and affinity.
                existing = quality_record(existing, cpu=None, workers=None,
                                          elapsed=None, position=position)
                existing['timing_evidence']['reason'] = 'parent finalization was interrupted'
                existing['host']['contamination']['reasons'] = [{'kind': 'interrupted-parent'}]
                existing['comparison'] = runner.comparison_observation(existing, policy)
                regime_io.atomic_write_json(run_dir / 'result.json', existing)
            saved += 1
        else:
            tasks.append((position, job, beta, key))
    print(json.dumps({'saved': saved, 'pending': len(tasks), 'workers': len(cpus)}), flush=True)
    if dry_run:
        return
    execution = {
        'cpus': cpus, 'workers': len(cpus), 'quality_only': True,
        'manifest_sha256': runner.sha256_file(root / 'manifest.json'),
        'dispatcher_sha256': runner.sha256_file(Path(__file__)),
        'run_one_sha256': runner.sha256_file(Path(__file__).with_name('round2_cpu.py')),
        'saved_at_start': saved, 'pending_at_start': len(tasks),
        'started_unix_s': time.time(),
    }
    regime_io.atomic_write_json(root / 'parallel-execution.json', execution)

    def run(task: tuple[int, runner.CpuJob, tuple[float, float], str], cpu: int) -> None:
        position, job, beta, key = task
        started = time.monotonic()
        run_dir = root / 'runs' / key
        run_dir.mkdir(parents=True, exist_ok=True)
        payload = root / 'dispatch' / f'{key}.json'
        regime_io.atomic_write_json(payload, {
            'job': job.to_dict(), 'solver_identity': identity,
            'logical_beta_range': list(beta), 'comparison_policy': policy.to_dict(),
        })
        # Numeric libraries must not create additional workers behind the affinity pin.
        env = dict(os.environ, OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
        outcome = runner.run_with_deadline(
            ['taskset', '-c', str(cpu), sys.executable,
             str(Path(__file__).with_name('round2_cpu.py')), 'run-one',
             '--job', str(payload), '--bundles-root', str(bundles), '--out', str(run_dir)],
            timeout_s=timeout, cwd=Path(__file__).resolve().parents[1], env=env,
        )
        if outcome['status'] != 'completed':
            # Leave any child output intact and stop dispatch, rather than treating
            # resource failures as completed scientific observations on resume.
            regime_io.atomic_write_json(run_dir / 'dispatch-failure.json', outcome)
            raise RuntimeError(f"job {key}: {outcome['status']}; inspect {run_dir}")
        record = runner.completed_result(run_dir, key)
        if record is None:
            raise RuntimeError(f'child returned no result: {key}')
        record = quality_record(record, cpu=cpu, workers=len(cpus),
                                elapsed=time.monotonic() - started, position=position)
        record['comparison'] = runner.comparison_observation(record, policy)
        regime_io.atomic_write_json(run_dir / 'result.json', record)
        print(json.dumps({'position': position, 'cpu': cpu, 'status': record['status']}), flush=True)

    dispatch(tasks, cpus, run)
    print('All frozen campaign jobs have saved results', flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--cpus', type=int, nargs='+', required=True)
    parser.add_argument('--timeout-s', type=float, default=3600)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    with (args.out / '.parallel.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        campaign(args.out.resolve(), args.cpus, args.timeout_s, args.dry_run)


if __name__ == '__main__':
    main()
