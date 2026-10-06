"""Tests of the win estimate: the clears of each arm, the run time of several launches and cards, and the shares."""

import importlib.util
import json
import warnings
from datetime import UTC, datetime
from pathlib import Path

import pytest

from quip_miner_dwave import regime_io, wins

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "cascade_wins.py"


def load():
    spec = importlib.util.spec_from_file_location("cascade_wins", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


T0 = 1_790_000_000.0  # seconds since the epoch at the origin of the first journal
GATE = -14_709_553  # milli energy; the median gate of ROUNDS


def start(t=0.0, origin=T0):
    """A ``start`` event. Each one in a journal carries the origin of the journal, and ``t`` counts from it."""
    return {"type": "start", "t": t, "started_at": datetime.fromtimestamp(origin, UTC).isoformat()}


def cleared_by_gpu(nonce, found, cleared):
    """The survivor and the ``gpu-256`` event of a model whose cold leg the kernel stopped as a hit below the gate."""
    return [{"type": "survivor", "t": found, "nonce": nonce, "arm": "gpu-256", "energy": -14310},
            {"type": "deep", "t": cleared, "arm": "gpu-256", "nonce": nonce, "status": "hit", "best": -14712}]


def stats(t, salts):
    """What the miner sends: ``salts`` counts from the launch of the miner."""
    return {"type": "stats", "t": t, "salts": salts, "survivors": 7}


EVENTS = [
    start(),
    {"type": "survivor", "t": 10.0, "nonce": "aa", "arm": "gpu-256", "energy": -14310},
    {"type": "survivor", "t": 20.0, "nonce": "bb", "arm": "qpu-cold", "energy": -14320},
    {"type": "qpu", "t": 22.0, "nonce": "bb", "arm": "qpu-cold", "kept": True, "access_us": 50_000},
    {"type": "survivor", "t": 25.0, "nonce": "ff", "arm": "qpu-seeded", "energy": -14325},
    {"type": "qpu", "t": 27.0, "nonce": "ff", "arm": "qpu-seeded", "kept": True, "access_us": 50_000},
    {"type": "survivor", "t": 30.0, "nonce": "cc", "arm": "gpu-256", "energy": -14330},
    {"type": "survivor", "t": 40.0, "nonce": "dd", "arm": "gpu-256", "energy": -14340},
    # The gpu-256 program holds the 256-sweep gated leg and then the cold leg. One event reports both legs.
    # dd was dropped by the gated leg, so no cold leg ran. A best below the threshold is not a clear.
    {"type": "deep", "t": 41.0, "arm": "gpu-256", "nonce": "dd", "status": "dropped", "best": -14720},
    {"type": "survivor", "t": 50.0, "nonce": "ee", "arm": "qpu-cold", "energy": -14350},
    {"type": "qpu", "t": 52.0, "nonce": "ee", "arm": "qpu-cold", "kept": False, "access_us": 50_000},
    # gg belongs to qpu-cold, and the sidecar skipped the QPU for it. It runs on gpu-256, so it counts there.
    {"type": "survivor", "t": 60.0, "nonce": "gg", "arm": "qpu-cold", "energy": -14360},
    {"type": "qpu_skip", "t": 61.0, "nonce": "gg", "arm": "qpu-cold", "reason": "pace"},
    # aa passed the gate, and the kernel stopped its cold leg as a hit below the threshold: the clear of the gpu arm.
    {"type": "deep", "t": 130.0, "arm": "gpu-256", "nonce": "aa", "status": "hit", "best": -14712},
    # bb and ff clear: the QPU gate kept them, and the kernel stopped their legs as hits.
    {"type": "deep", "t": 140.0, "arm": "cold", "nonce": "bb", "status": "hit", "best": -14710},
    {"type": "deep", "t": 150.0, "arm": "qpu-seeded", "nonce": "ff", "status": "hit", "best": -14714},
    # cc passed the gate and ended at the threshold: a proof must be strictly below it.
    {"type": "deep", "t": 160.0, "arm": "gpu-256", "nonce": "cc", "status": "passed", "best": -14709},
    # A cold leg of a survivor that the QPU gate dropped clears no arm: the gpu arm counts its own program only.
    {"type": "deep", "t": 170.0, "arm": "cold", "nonce": "ee", "status": "passed", "best": -14712},
    # gg runs on gpu-256 and ends dropped.
    {"type": "deep", "t": 180.0, "arm": "gpu-256", "nonce": "gg", "status": "dropped", "best": -14400},
    stats(999.0, 5_000_000),
    {"type": "end", "t": 1000.0},
]
ROUNDS = {"block_seconds": 6.0, "rounds": [{"blocks": 50, "gate_milli": -14_709_553},
                                           {"blocks": 100, "gate_milli": -14_709_553}]}


def write_journal(directory, events, spend_us=()):
    """``events.jsonl`` and, for each call in ``spend_us``, a ``submit`` and a ``charge`` line in ``spend.jsonl``."""
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "events.jsonl"
    path.write_text("".join(json.dumps(e) + "\n" for e in events))
    for index, micros in enumerate(spend_us):
        regime_io.append_spend(directory, regime_io.SUBMIT, "cell", f"job{index}", 80, 59_861)
        regime_io.append_spend(directory, regime_io.CHARGE, "cell", f"job{index}", 80, micros)
    return str(path)


I64_MAX = 2**63 - 1  # what the coordinator writes as ``best_energy_milli`` when no row cleared its own gate


def attempt(ts_ms, raw_best_energy_milli, best_energy_milli=I64_MAX):
    """A row of ``attempts.jsonl`` as the coordinator writes it, with the fields that matter here."""
    return {"ts_ms": ts_ms, "qblock_id": 41, "job_id": "aa", "best_energy_milli": best_energy_milli,
            "raw_best_energy_milli": raw_best_energy_milli, "threshold_milli": -14_700_000, "submitted": False}


def legacy_attempt(ts_ms, best_energy_milli):
    """A row from before ``raw_best_energy_milli`` existed: the coordinator wrote 13 fields, and this is one of them."""
    return {"ts_ms": ts_ms, "qblock_id": 19, "generation": 1, "miner_id": "cuda-1", "job_id": "aa", "is_pow": True,
            "order_id": "", "best_energy_milli": best_energy_milli, "diversity_milli": 0, "n_valid": 0,
            "accepted": False, "submitted": False, "device_access_time_us": 4_804_992}


def write_attempts(directory, rows):
    """``attempts.jsonl`` with a line for each row. A row that is a string goes in as it is."""
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "attempts.jsonl"
    path.write_text("".join((row if isinstance(row, str) else json.dumps(row)) + "\n" for row in rows) + "\n")
    return str(path)


def run(module, tmp_path, journals, *extra):
    """Run ``main`` over the journals, and return the summary that it wrote."""
    (tmp_path / "rounds.json").write_text(json.dumps(ROUNDS))
    out = tmp_path / "wins.json"
    args = [arg for journal in journals for arg in ("--events", journal)]
    assert module.main([*args, "--rounds", str(tmp_path / "rounds.json"), "--out", str(out), *extra]) == 0
    return json.loads(out.read_text())


def test_each_arm_counts_only_the_clears_its_own_gate_allowed():
    module = load()
    gate = GATE
    assert module.arm_clears(EVENTS, "gpu", gate) == [(10.0, 130.0)]
    # The gpu arm clears at the gate and not at the energy of its own clear: -14,712 is not below -14,712,000.
    assert module.arm_clears(EVENTS, "gpu", -14_712_000) == []
    assert module.arm_clears(EVENTS, "qpu-cold", gate) == [(20.0, 140.0)]
    assert module.arm_clears(EVENTS, "qpu-seeded", gate) == [(25.0, 150.0)]
    # -14,710 is not below a threshold of -14,710,000.
    assert module.arm_clears(EVENTS, "qpu-cold", -14_710_000) == []


def one_leg(arm, deep_arm, status, best):
    """A survivor found at 10 s and its one deep leg, which ends at 100 s. The QPU gate kept the model of a QPU arm."""
    events = [{"type": "survivor", "t": 10.0, "nonce": "aa", "energy": -14310}]
    if arm != "gpu":
        events.append({"type": "qpu", "t": 12.0, "nonce": "aa", "kept": True, "access_us": 50_000})
    return [*events, {"type": "deep", "t": 100.0, "arm": deep_arm, "nonce": "aa", "status": status, "best": best}]


@pytest.mark.parametrize(("arm", "deep_arm", "status", "best", "clears"), [
    pytest.param("gpu", "gpu-256", "hit", -14712, True, id="gpu: the kernel stopped the leg as a hit below the gate"),
    pytest.param("gpu", "gpu-256", "passed", -14712, True, id="gpu: the leg ran to its end below the gate"),
    pytest.param("gpu", "gpu-256", "hit", -14705, False,
                 id="gpu: a hit above the gate stopped early at a looser live target, so its best is not final"),
    pytest.param("gpu", "gpu-256", "passed", -14709, False, id="gpu: at the threshold is not below it"),
    pytest.param("gpu", "gpu-256", "dropped", -14720, False, id="gpu: a dropped leg never clears"),
    pytest.param("qpu-cold", "cold", "hit", -14712, True, id="qpu-cold: a hit below the gate"),
    pytest.param("qpu-cold", "cold", "dropped", -14720, False, id="qpu-cold: an aborted leg ends dropped"),
    pytest.param("qpu-seeded", "qpu-seeded", "hit", -14712, True, id="qpu-seeded: a hit below the gate"),
    pytest.param("qpu-seeded", "qpu-seeded", "dropped", -14720, False, id="qpu-seeded: an aborted leg ends dropped"),
])
def test_a_leg_clears_unless_it_was_dropped_and_its_best_is_below_the_gate(arm, deep_arm, status, best, clears):
    module = load()
    assert module.arm_clears(one_leg(arm, deep_arm, status, best), arm, GATE) == ([(10.0, 100.0)] if clears else [])


def test_a_qpu_survivor_that_the_sidecar_skipped_and_ran_on_the_gpu_counts_for_the_gpu_arm():
    module = load()
    events = [{"type": "survivor", "t": 10.0, "nonce": "aa", "arm": "qpu-cold", "energy": -14310},
              {"type": "qpu_skip", "t": 11.0, "nonce": "aa", "arm": "qpu-cold", "reason": "pace"},
              {"type": "deep", "t": 90.0, "arm": "gpu-256", "nonce": "aa", "status": "hit", "best": -14712}]
    assert module.arm_clears(events, "gpu", GATE) == [(10.0, 90.0)]
    assert module.arm_clears(events, "qpu-cold", GATE) == []


def test_summary_reports_rates_shares_and_qpu_time(tmp_path, capsys):
    module = load()
    # Four calls reached the QPU and were charged. The journal has a `qpu` event for three: the fourth call failed
    # after its charge.
    summary = run(module, tmp_path, [write_journal(tmp_path, EVENTS, spend_us=[50_000] * 4)])
    assert summary["gate_milli"] == -14_709_553 and summary["run_seconds"] == 1000.0
    assert summary["qpu_access_s"] == pytest.approx(0.2) and summary["qpu_event_access_s"] == pytest.approx(0.15)
    assert summary["salts"] == 5_000_000
    # The flows behind the clears: the survivors that each arm took, and the QPU survivors that ran on the GPU.
    assert summary["survivors"] == {"gpu-256": 3, "qpu-cold": 3, "qpu-seeded": 1} and summary["qpu_skips"] == 1
    assert "survivors gpu-256 3, qpu-cold 3, qpu-seeded 1; QPU skips 1" in capsys.readouterr().out
    gpu = summary["arms"]["gpu"]
    # The gpu arm ran its own three survivors and the survivor that qpu-cold skipped: four deep legs and one clear.
    assert gpu["clears"] == 1 and gpu["deep_legs"] == 4 and gpu["median_find_to_clear_s"] == 120.0
    assert gpu["clears_per_hour"] == pytest.approx(3.6)
    # The one clear takes 120 s. A 300 s round can win it and a 600 s round can win it from more offsets.
    assert 0 < gpu["replay_win_share"] < 0.5
    assert summary["arms"]["qpu-seeded"]["deep_legs"] == 1
    # The field clears once per 450 s on average, and one clear per 1,000 s is 31% of that total.
    assert gpu["equilibrium_share"] == pytest.approx(0.001 / (0.001 + 1 / 450))


def test_summarize_returns_the_whole_summary_with_the_qpu_access_it_is_given(tmp_path):
    module = load()
    events, launches = module.read_journal(write_journal(tmp_path, EVENTS))
    summary = module.summarize(events, launches, ROUNDS, GATE, 0.2)
    assert summary["qpu_access_s"] == 0.2 and summary["qpu_event_access_s"] == pytest.approx(0.15)
    assert summary["run_seconds"] == 1000.0 and summary["arms"]["gpu"]["clears"] == 1


def test_a_journal_reads_on_the_clock_of_its_first_start(tmp_path):
    module = load()
    path = write_journal(tmp_path, [start(0.0, T0 + 300), {"type": "survivor", "t": 10.0}, {"type": "end", "t": 50.0}])
    events, launches = module.read_journal(path)
    assert [e["t"] for e in events] == [T0 + 300, T0 + 310, T0 + 350]
    assert [(launch.start, launch.end) for launch in launches] == [(T0 + 300, T0 + 350)]
    with pytest.raises(ValueError, match="no start event"):
        module.read_journal(write_journal(tmp_path / "bare", [{"type": "survivor", "t": 10.0}]))


@pytest.mark.parametrize(("events", "spans"), [
    pytest.param([start(0.0), {"type": "survivor", "t": 10.0}, {"type": "survivor", "t": 40.0}], [(0.0, 40.0)],
                 id="no end: to the last event"),
    pytest.param([start(0.0), {"type": "error", "t": 100.0, "message": "RuntimeError: the solver is offline"}],
                 [(0.0, 100.0)], id="an error ends it"),
    pytest.param([start(0.0), {"type": "error", "t": 500.0, "message": "OutDirBusy: another sidecar holds the lock"},
                  {"type": "end", "t": 1000.0}], [(0.0, 1000.0)],
                 id="a second sidecar that gives up on the lock of the out-dir does not end the launch that holds it"),
    pytest.param([start(0.0),
                  {"type": "qpu_error", "t": 10.0, "stage": "sample", "exception": "RuntimeError",
                   "message": "the solver is offline"},
                  {"type": "lane_check", "t": 20.0, "id": "7:1", "lanes": 128},
                  {"type": "lane_order_mismatch", "t": 30.0, "id": "7:2", "lanes": 128, "mismatched": 128},
                  {"type": "qpu_recovered", "t": 40.0, "retry_s": 30.0}, {"type": "end", "t": 100.0}],
                 [(0.0, 100.0)], id="a failure of the QPU side, a lane check, and a recovery do not end it"),
    pytest.param([start(0.0), {"type": "end", "t": 100.0}, start(500.0), {"type": "end", "t": 600.0}],
                 [(0.0, 100.0), (500.0, 600.0)], id="the gap after an end is out"),
    pytest.param([start(0.0), {"type": "survivor", "t": 50.0}, start(200.0), {"type": "end", "t": 300.0}],
                 [(0.0, 50.0), (200.0, 300.0)], id="a restart closes a launch that has no end at its last event"),
])
def test_a_launch_runs_from_its_start_to_its_end_its_error_or_its_last_event(tmp_path, events, spans):
    module = load()
    _, launches = module.read_journal(write_journal(tmp_path, events))
    assert [(launch.start - T0, launch.end - T0) for launch in launches] == spans


def test_run_seconds_is_the_union_of_the_launches_of_every_journal(tmp_path):
    module = load()
    card0 = write_journal(tmp_path / "card0", [start(0.0), {"type": "end", "t": 600.0},
                                               start(1000.0), {"type": "end", "t": 1400.0}])
    card1 = write_journal(tmp_path / "card1", [start(0.0, T0 + 300), {"type": "end", "t": 500.0}])
    # Card 0 runs 0-600 s and 1,000-1,400 s, and card 1 runs 300-800 s: the union is 0-800 s and 1,000-1,400 s.
    assert run(module, tmp_path, [card0, card1])["run_seconds"] == 1200.0


def test_the_clears_of_two_journals_merge_on_the_absolute_clock(tmp_path):
    module = load()
    card0 = write_journal(tmp_path / "card0", [start(), *cleared_by_gpu("aa", 10.0, 130.0),
                                               {"type": "end", "t": 1000.0}])
    card1 = write_journal(tmp_path / "card1", [start(0.0, T0 + 100), *cleared_by_gpu("bb", 20.0, 140.0),
                                               {"type": "end", "t": 1000.0}])
    summary = run(module, tmp_path, [card0, card1])
    gpu = summary["arms"]["gpu"]
    assert summary["run_seconds"] == 1100.0 and gpu["clears"] == 2 and gpu["median_find_to_clear_s"] == 120.0
    # Card 1 starts 100 s after card 0, so its clear is found at 120 s and cleared at 240 s of the run.
    assert gpu["replay_win_share"] == pytest.approx(
        wins.replay_win_share([(10.0, 130.0), (120.0, 240.0)], 1100.0, [300.0, 600.0]))


def test_the_replay_runs_on_the_clock_of_the_run_and_skips_the_gaps(tmp_path):
    module = load()
    whole = write_journal(tmp_path / "whole", [start(), *cleared_by_gpu("aa", 510.0, 630.0),
                                               {"type": "end", "t": 1000.0}])
    # The same 1,000 s of run in two launches, with a gap of 500 s between them.
    split = write_journal(tmp_path / "split", [start(), {"type": "end", "t": 500.0}, start(1000.0),
                                               *cleared_by_gpu("aa", 1010.0, 1130.0), {"type": "end", "t": 1500.0}])
    steady, gapped = run(module, tmp_path, [whole]), run(module, tmp_path, [split])
    assert steady["run_seconds"] == gapped["run_seconds"] == 1000.0
    assert gapped["arms"]["gpu"]["median_find_to_clear_s"] == 120.0
    # A round of 300 s wins from the 19 starts that hold the clear, and a round of 600 s from 38 of 41.
    assert steady["arms"]["gpu"]["replay_win_share"] == pytest.approx((19 + 38) / (71 + 41))
    assert gapped["arms"]["gpu"]["replay_win_share"] == steady["arms"]["gpu"]["replay_win_share"]


def test_salts_are_the_last_count_of_each_launch_summed_over_the_journals(tmp_path):
    module = load()
    card0 = write_journal(tmp_path / "card0", [
        start(), stats(60.0, 100), stats(120.0, 300), {"type": "end", "t": 130.0},
        start(500.0), stats(560.0, 50), stats(620.0, 70), {"type": "end", "t": 630.0},
    ])
    card1 = write_journal(tmp_path / "card1", [start(), stats(60.0, 1000)])
    # The count restarts with each launch, so each launch adds its last count: 300 + 70 + 1,000.
    assert run(module, tmp_path, [card0, card1])["salts"] == 1370


def test_a_card_with_no_spend_journal_counts_no_qpu_access_and_the_script_says_so(tmp_path, capsys):
    module = load()
    spent = write_journal(tmp_path / "card0", [start(), {"type": "end", "t": 10.0}], spend_us=[50_000, 30_000])
    idle = write_journal(tmp_path / "card1", [start(), {"type": "end", "t": 10.0}])
    assert run(module, tmp_path, [spent, idle])["qpu_access_s"] == pytest.approx(0.08)
    err = capsys.readouterr().err  # the note names the journal by its resolved path
    assert "spend.jsonl" in err and str(Path(idle).resolve()) in err and str(Path(spent).resolve()) not in err
    assert run(module, tmp_path, [idle])["qpu_access_s"] == 0.0


def test_attempts_count_the_coordinator_rows_and_those_below_the_gate(tmp_path, capsys):
    module = load()
    journal = write_journal(tmp_path / "card0", [start(), {"type": "end", "t": 10.0}])
    # The gate is -14,709,553 milli. The true best of a row is `raw_best_energy_milli`, because `best_energy_milli`
    # holds the sentinel when the row missed the gate of the coordinator at the time.
    first = write_attempts(tmp_path / "data" / "41", [
        attempt(1_790_000_100_000, -14_712_000),
        attempt(1_790_000_200_000, -14_709_553),  # at the gate: a proof must be strictly below it
        attempt(1_790_000_300_000, I64_MAX),  # the coordinator got no solution
    ])
    second = write_attempts(tmp_path / "data" / "42", [attempt(1_790_000_050_250, -14_800_000, -14_800_000)])
    glob = run(module, tmp_path, [journal], "--attempts", str(tmp_path / "data" / "*" / "attempts.jsonl"))
    assert glob["attempts"] == 4 and glob["attempts_below_gate"] == 2 and glob["attempts_unreadable"] == 0
    assert glob["first_attempt_at"] == "2026-09-21T14:14:10.250000+00:00"
    assert glob["last_attempt_at"] == "2026-09-21T14:18:20+00:00"
    assert "4 attempts" in capsys.readouterr().out
    # The shell may expand the glob into several paths, and the option may repeat.
    assert run(module, tmp_path, [journal], "--attempts", first, second) == glob
    assert run(module, tmp_path, [journal], "--attempts", first, "--attempts", second) == glob
    assert "attempts" not in run(module, tmp_path, [journal])
    empty = run(module, tmp_path, [journal], "--attempts", write_attempts(tmp_path / "empty", []))
    assert empty["attempts"] == 0 and empty["attempts_unreadable"] == 0
    assert empty["first_attempt_at"] is None and empty["last_attempt_at"] is None


def test_attempts_read_a_legacy_row_and_count_the_lines_that_hold_no_row(tmp_path):
    module = load()
    journal = write_journal(tmp_path / "card0", [start(), {"type": "end", "t": 10.0}])
    row = json.dumps(attempt(1_790_000_100_000, -14_712_000))
    path = write_attempts(tmp_path / "data" / "19", [
        row,  # a current row below the gate
        legacy_attempt(1_790_000_200_000, -14_712_000),  # a legacy row holds only best_energy_milli
        legacy_attempt(1_790_000_300_000, I64_MAX),  # its sentinel never passes a gate
        row + row,  # two rows on one line, which `json.loads` refuses with "Extra data"
        row[:60],  # a write that was cut short
        "7",  # JSON that is not a row
        json.dumps({"ts_ms": 1_790_000_400_000}),  # a row with no energy
        json.dumps(legacy_attempt("soon", -14_712_000)),  # a time that is not a number
        json.dumps(legacy_attempt(1_790_000_450_000, None)),  # an energy that is not a number
    ])
    # One line that holds no row must not end the run before the summary is written: `run` reads the file `--out`.
    summary = run(module, tmp_path, [journal], "--attempts", path)
    assert summary["attempts"] == 3 and summary["attempts_below_gate"] == 2 and summary["attempts_unreadable"] == 6
    # The times of the lines that were skipped do not count.
    assert summary["first_attempt_at"] == "2026-09-21T14:15:00+00:00"
    assert summary["last_attempt_at"] == "2026-09-21T14:18:20+00:00"


def test_an_attempts_pattern_that_matches_no_file_is_a_usage_error(tmp_path, capsys):
    module = load()
    journal = write_journal(tmp_path / "card0", [start(), {"type": "end", "t": 10.0}])
    with pytest.raises(SystemExit) as stopped:
        run(module, tmp_path, [journal], "--attempts", str(tmp_path / "data" / "*" / "attempts.jsonl"))
    assert stopped.value.code == 2 and "matches no file" in capsys.readouterr().err


def test_the_poisson_share_is_zero_without_a_round_or_a_clear_rate_and_does_not_warn():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert wins.poisson_win_share([], 0.01, 10.0) == 0.0
        assert wins.poisson_win_share([], 0.0, 0.0) == 0.0
        assert wins.poisson_win_share([300.0, 600.0], 0.0, 120.0) == 0.0


def test_a_journal_given_twice_counts_once_even_when_the_paths_are_spelled_differently(tmp_path):
    module = load()
    journal = write_journal(tmp_path / "card0", EVENTS, spend_us=[50_000, 50_000, 50_000])
    once = run(module, tmp_path, [journal])
    again = str(tmp_path / "card0" / ".." / "card0" / "events.jsonl")  # the same file through the parent directory
    twice = run(module, tmp_path, [journal, journal, again])
    # Counted again, the journal would add its salts and its spend twice, and the union of its launches would hide it.
    assert twice == once and twice["salts"] == 5_000_000


def test_an_attempts_file_that_two_spellings_name_counts_once(tmp_path):
    module = load()
    journal = write_journal(tmp_path / "card0", [start(), {"type": "end", "t": 10.0}])
    attempts = write_attempts(tmp_path / "data" / "41", [attempt(1_790_000_100_000, -14_712_000)])
    again = str(tmp_path / "data" / ".." / "data" / "41" / "attempts.jsonl")
    assert run(module, tmp_path, [journal], "--attempts", attempts, again)["attempts"] == 1
