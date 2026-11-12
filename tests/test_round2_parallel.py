"""The parallel dispatcher preserves quality evidence and exclusive job ownership."""
import importlib.util
import threading
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location(
    "round2_parallel", Path(__file__).parents[1] / "scripts/round2_parallel.py"
)
assert SPEC and SPEC.loader
parallel = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(parallel)


def test_workers_claim_each_job_once_and_never_share_a_cpu():
    active = set()
    seen = []
    lock = threading.Lock()
    barrier = threading.Barrier(4)

    def run(job, cpu):
        with lock:
            assert cpu not in active
            active.add(cpu)
        if job < 4:
            barrier.wait(timeout=3)
        with lock:
            seen.append(job)
            active.remove(cpu)

    parallel.dispatch(list(range(40)), [0, 1, 2, 3], run)
    assert sorted(seen) == list(range(40))


def test_worker_failure_propagates_and_stops_new_dispatch():
    def fail(job, cpu):
        raise ValueError("broken artifact")

    with pytest.raises(ValueError, match="broken artifact"):
        parallel.dispatch(list(range(40)), [0], fail)


def test_parallel_timing_is_not_eligible_for_runtime_comparisons():
    record = {"status": "completed", "end_to_end_s": 3, "best_energy": -5}
    result = parallel.quality_record(record, cpu=4, workers=28, elapsed=4, position=7)
    assert result['end_to_end_s'] is None
    assert result['caller_end_to_end_s'] is None
    assert result['operational_elapsed_s'] == 4
    assert result['child_end_to_end_s'] == 3
    assert result['host']['affinity'] == [4]
    assert result['host']['contamination']['contaminated']
    assert result['timing_evidence']['available'] is False
    assert result['best_energy'] == -5


def test_duplicate_cpus_are_rejected():
    with pytest.raises(ValueError, match="distinct"):
        parallel.dispatch([1], [0, 0], lambda job, cpu: None)


def test_campaign_preserves_saved_results_and_finalizes_new_quality_records(tmp_path, monkeypatch):
    import json
    import os
    import numpy as np
    from quip_solver_dwave import round2_io, round2_runner as runner

    bundles = tmp_path / 'bundles'
    model_hash = round2_io.write_bundle(
        bundles / 'native-pm1' / 'a', {'cell': 'native-pm1', 'nonce': 'a', 'offset': 0},
        {'h': np.zeros(2), 'edges': np.array([[0, 1]]), 'j': np.array([-1.0])},
    )
    jobs = runner.build_quality_jobs(
        {'rows': [{'cell': 'native-pm1', 'nonce': 'a', 'model_hash': model_hash}]},
        ['native-pm1'], repetitions=1, sweeps=(512,),
    )
    identity = {'binary': 'frozen'}
    policy = runner.ComparisonPolicy.calibration()
    monkeypatch.setattr(runner, 'solver_identity', lambda: identity)
    monkeypatch.setattr(runner, 'logical_beta_range', lambda *args: (0.1, 1.0))
    root = tmp_path / 'campaign'
    root.mkdir()
    (root / 'dispatch').mkdir()
    (root / 'manifest.json').write_text(json.dumps({
        'solver_identity': identity, 'comparison_policy': policy.to_dict(),
        'bundles_root': str(bundles), 'jobs': [j.to_dict() for j in jobs],
    }))
    key = runner.run_key(jobs[0], identity, (0.1, 1.0), policy)
    runner.write_result(root / 'runs' / key, {'run_key': key, 'status': 'unsupported'}, None, None)
    saved = (root / 'runs' / key / 'result.json').read_bytes()
    calls = []

    def child(command, **kwargs):
        payload = json.loads(Path(command[command.index('--job') + 1]).read_text())
        out = Path(command[command.index('--out') + 1])
        calls.append(out.name)
        runner.write_result(out, {
            'run_key': out.name, **payload['job'], 'status': 'completed',
            'end_to_end_s': 0.5, 'best_energy': -1,
        }, np.ones((64, 2), dtype=np.int8), -np.ones(64))
        return {'status': 'completed'}

    monkeypatch.setattr(runner, 'run_with_deadline', child)
    cpus = sorted(os.sched_getaffinity(0))[:2]
    parallel.campaign(root, cpus, 60, False)
    assert len(calls) == len(jobs) - 1
    assert (root / 'runs' / key / 'result.json').read_bytes() == saved
    for k in calls:
        record = runner.completed_result(root / 'runs' / k, k)
        assert record is not None
        assert record['end_to_end_s'] is None
        assert record['host']['contamination']['contaminated']
    parallel.campaign(root, cpus, 60, False)
    assert len(calls) == len(jobs) - 1

    interrupted_path = root / 'runs' / calls[0] / 'result.json'
    interrupted = json.loads(interrupted_path.read_text())
    interrupted.pop('caller_end_to_end_s')
    interrupted['end_to_end_s'] = 0.5
    interrupted_path.write_text(json.dumps(interrupted))
    parallel.campaign(root, cpus, 60, False)
    repaired = json.loads(interrupted_path.read_text())
    assert repaired['end_to_end_s'] is None
    assert repaired['timing_evidence']['available'] is False
    assert len(calls) == len(jobs) - 1

    # A prior failed dispatch must remain visible even if child output exists.
    (root / 'runs' / key / 'dispatch-failure.json').write_text('{}')
    with pytest.raises(RuntimeError, match='unresolved dispatch failure'):
        parallel.campaign(root, cpus, 60, False)
