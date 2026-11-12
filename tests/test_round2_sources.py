"""The provenance check, credential filter, and file inventory of round2_sources.py.

The provenance check runs before any sampler integration, per Task 1 of
docs/superpowers/plans/2026-09-22-regime-search-round2.md: current defaults
never stand in for a historical record, and a missing field is explicit
rather than silently substituted.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import pathlib
import subprocess

import pytest

SCRIPT_PATH = pathlib.Path(__file__).resolve().parents[1] / "scripts" / "round2_sources.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("round2_sources", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


round2_sources = _load_module()

REQUIRED_FIELDS = (
    "market_snapshot_sha256",
    "market_as_of",
    "basket",
    "sliders",
    "deployment_commit",
    "runtime_config",
    "effective_beta",
    "reads",
    "sa_sweeps",
    "qpu_anneal_us",
    "qpu_solver",
    "tie_rules",
)


def _complete_record() -> dict:
    return {
        "market_snapshot_sha256": "a" * 64,
        "market_as_of": "2026-09-20",
        "basket": ["A0", "A1"],
        "sliders": {"hold_count": 5},
        "deployment_commit": "9136be4c4b57abdab7f0c46120c341ba94fae014",
        "runtime_config": {"beta": 0.4},
        "effective_beta": 0.4,
        "reads": 64,
        "sa_sweeps": 500,
        "qpu_anneal_us": [80, 400],
        "qpu_solver": "Advantage2_system1",
        "tie_rules": "strict",
    }


# ------------------------------------------------------------- historical_gaps
def test_current_defaults_do_not_prove_historical_configuration():
    gaps = round2_sources.historical_gaps({"runtime_config": {"beta": 0.4}})
    assert "market_snapshot_sha256" in gaps
    assert "effective_beta" in gaps
    assert "qpu_anneal_us" in gaps


def test_historical_gaps_empty_when_every_field_present():
    assert round2_sources.historical_gaps(_complete_record()) == []


def test_historical_gaps_treats_none_as_missing():
    record = _complete_record()
    record["market_as_of"] = None
    gaps = round2_sources.historical_gaps(record)
    assert gaps == ["market_as_of"]


def test_historical_gaps_reports_every_required_key_on_empty_record():
    assert round2_sources.historical_gaps({}) == list(REQUIRED_FIELDS)


# ------------------------------------------------------------- credential filter
@pytest.mark.parametrize(
    "name",
    [
        ".env",
        ".env.local",
        "api_secret.json",
        "SECRETS.yaml",
        "access_token.txt",
        "oauth-token",
        "keyring.db",
        ".pass-store/entry.gpg",
    ],
)
def test_is_credential_like_flags_secret_shaped_names(name):
    assert round2_sources.is_credential_like(pathlib.Path("/x/y") / name)


@pytest.mark.parametrize("name", ["manifest.json", "ladder.csv", "run-info.json", "qpu.csv"])
def test_is_credential_like_leaves_ordinary_files_alone(name):
    assert not round2_sources.is_credential_like(pathlib.Path("/x/y") / name)


# ------------------------------------------------------------- hashing / inventory
def test_sha256_file_matches_hashlib(tmp_path):
    target = tmp_path / "data.bin"
    target.write_bytes(b"round 2 provenance fixture")
    expected = hashlib.sha256(b"round 2 provenance fixture").hexdigest()
    assert round2_sources.sha256_file(target) == expected


def test_inventory_paths_records_size_hash_and_mtime_and_skips_credentials(tmp_path):
    (tmp_path / "manifest.json").write_text("{}", encoding="utf-8")
    (tmp_path / ".env").write_text("SECRET=1", encoding="utf-8")
    sub = tmp_path / "nested"
    sub.mkdir()
    (sub / "ladder.csv").write_text("a,b\n1,2\n", encoding="utf-8")

    entries = round2_sources.inventory_paths(tmp_path)
    names = {entry["path"].name for entry in entries}

    assert names == {"manifest.json", "ladder.csv"}
    for entry in entries:
        assert entry["size_bytes"] == entry["path"].stat().st_size
        assert entry["sha256"] == round2_sources.sha256_file(entry["path"])
        assert entry["mtime"].endswith("+00:00") or entry["mtime"].endswith("Z")


def test_inventory_paths_on_missing_root_returns_empty(tmp_path):
    assert round2_sources.inventory_paths(tmp_path / "does-not-exist") == []


# ------------------------------------------------------------- source record assembly
def test_build_source_record_keeps_known_fields_and_computes_gaps(tmp_path):
    (tmp_path / "qpo_real.json").write_text("[]", encoding="utf-8")

    record = round2_sources.build_source_record(
        name="portfolio-gutcheck",
        provenance="synthetic-test",
        root=tmp_path,
        known_fields={"reads": 64, "sa_sweeps": 500},
    )

    assert record["name"] == "portfolio-gutcheck"
    assert record["provenance"] == "synthetic-test"
    assert record["reads"] == 64
    assert record["sa_sweeps"] == 500
    assert record["market_snapshot_sha256"] is None
    assert "market_snapshot_sha256" in record["gaps"]
    assert "reads" not in record["gaps"]
    assert len(record["inventory"]) == 1
    assert record["inventory"][0]["path"].name == "qpo_real.json"


def test_build_source_record_rejects_unknown_provenance(tmp_path):
    with pytest.raises(ValueError):
        round2_sources.build_source_record(
            name="x", provenance="current", root=tmp_path, known_fields={}
        )


# --------------------------------------------------- excluded-directory pruning
def test_inventory_paths_prunes_excluded_dirs_before_opening_any_file(tmp_path, monkeypatch):
    git_dir = tmp_path / ".git"
    git_dir.mkdir()
    (git_dir / "config").write_text("must never be opened", encoding="utf-8")
    (tmp_path / "README.md").write_text("hello", encoding="utf-8")

    real_sha256_file = round2_sources.sha256_file

    def _guarded(path):
        if ".git" in pathlib.Path(path).parts:
            raise AssertionError(f"sha256_file must not be called on {path}")
        return real_sha256_file(path)

    monkeypatch.setattr(round2_sources, "sha256_file", _guarded)

    entries = round2_sources.inventory_paths(tmp_path, exclude_dirs=frozenset({".git"}))

    assert {entry["path"].name for entry in entries} == {"README.md"}


def test_build_source_record_threads_exclude_dirs_into_inventory(tmp_path):
    git_dir = tmp_path / ".git"
    git_dir.mkdir()
    (git_dir / "config").write_text("x", encoding="utf-8")
    (tmp_path / "README.md").write_text("hello", encoding="utf-8")

    record = round2_sources.build_source_record(
        name="p",
        provenance="prospective",
        root=tmp_path,
        known_fields={},
        exclude_dirs=frozenset({".git"}),
    )

    assert {entry["path"].name for entry in record["inventory"]} == {"README.md"}


# --------------------------------------------------- real archive field extraction
def test_extract_gutcheck_fields_leaves_reads_and_sa_sweeps_as_gaps_for_multiple_budgets(tmp_path):
    # Shaped like portfolio-gutcheck/qpo_real.json: several budgets in one run.
    data = [
        {
            "name": "real28-K5",
            "rows": [
                {"method": "greedy from zeros", "budget": "1 start"},
                {"method": "neal select + repair", "budget": "500 reads x 500 sweeps"},
                {"method": "kernel select + repair", "budget": "64 reads x 500 sweeps"},
                {"method": "kernel select + repair", "budget": "64 reads x 5000 sweeps"},
            ],
        }
    ]
    (tmp_path / "qpo_real.json").write_text(json.dumps(data), encoding="utf-8")

    fields = round2_sources.extract_gutcheck_fields(tmp_path)

    assert "reads" not in fields
    assert "sa_sweeps" not in fields
    assert fields["observed_candidates"]["reads"] == [64, 500]
    assert fields["observed_candidates"]["sa_sweeps"] == [500, 5000]

    record = round2_sources.build_source_record(
        name="portfolio-gutcheck", provenance="synthetic-test", root=tmp_path, known_fields=fields
    )
    assert "reads" in record["gaps"]
    assert "sa_sweeps" in record["gaps"]


def test_extract_gutcheck_fields_fills_a_single_agreed_budget(tmp_path):
    data = [{"name": "x", "rows": [{"budget": "64 reads x 500 sweeps"}, {"budget": "64 reads x 500 sweeps"}]}]
    (tmp_path / "qpo_real.json").write_text(json.dumps(data), encoding="utf-8")

    fields = round2_sources.extract_gutcheck_fields(tmp_path)

    assert fields["reads"] == 64
    assert fields["sa_sweeps"] == 500


def test_extract_round1_manifest_fields_leaves_qpu_solver_as_gap(tmp_path):
    # Shaped like round1/*/manifest.json: reads and anneal_us, never a solver field.
    for cell in ("native-125", "clique-portfolio"):
        cell_dir = tmp_path / cell
        cell_dir.mkdir()
        (cell_dir / "manifest.json").write_text(
            json.dumps({"reads": 64, "anneal_us": [80, 400], "mock": False}), encoding="utf-8"
        )

    fields = round2_sources.extract_round1_manifest_fields(tmp_path)

    assert fields["reads"] == 64
    assert fields["qpu_anneal_us"] == [80, 400]
    assert "qpu_solver" not in fields

    record = round2_sources.build_source_record(
        name="round1-regime-manifests", provenance="historical", root=tmp_path, known_fields=fields
    )
    assert "qpu_solver" in record["gaps"]
    assert "reads" not in record["gaps"]
    assert "qpu_anneal_us" not in record["gaps"]


def test_extract_round1_manifest_fields_leaves_reads_as_gap_when_manifests_disagree(tmp_path):
    (tmp_path / "a").mkdir()
    (tmp_path / "a" / "manifest.json").write_text(
        json.dumps({"reads": 64, "anneal_us": [80, 400]}), encoding="utf-8"
    )
    (tmp_path / "b").mkdir()
    (tmp_path / "b" / "manifest.json").write_text(
        json.dumps({"reads": 32, "anneal_us": [80, 400]}), encoding="utf-8"
    )

    fields = round2_sources.extract_round1_manifest_fields(tmp_path)

    assert "reads" not in fields
    assert fields["qpu_anneal_us"] == [80, 400]


# --------------------------------------------------- environment probe failure handling
def test_run_reports_failure_explicitly_for_a_missing_executable():
    result = round2_sources._run(["definitely-not-a-real-command-xyz"])

    assert result["ok"] is False
    assert result["stdout"] is None
    assert result["error"]


def test_git_facts_on_a_non_repo_directory_records_failure_not_a_clean_tree(tmp_path):
    facts = round2_sources._git_facts(tmp_path)

    assert facts["dirty"] is None
    assert facts["head"] is None
    assert facts["probe_errors"]


def test_git_facts_on_a_clean_repo_has_no_probe_errors(tmp_path):
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.email", "a@example.com"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.name", "a"], cwd=tmp_path, check=True)
    subprocess.run(
        ["git", "commit", "--allow-empty", "-q", "-m", "init"], cwd=tmp_path, check=True
    )

    facts = round2_sources._git_facts(tmp_path)

    assert facts["dirty"] is False
    assert facts["probe_errors"] == []
    assert facts["head"]


# --------------------------------------------------- environment lockfile hashing
def test_collect_environment_hashes_every_repositorys_lockfile(tmp_path, monkeypatch):
    d_repo, c_repo, p_repo = tmp_path / "d", tmp_path / "c", tmp_path / "p"
    for repo in (d_repo, c_repo, p_repo):
        repo.mkdir()
    (d_repo / "uv.lock").write_bytes(b"d-lock")
    (p_repo / "uv.lock").write_bytes(b"p-lock")
    (c_repo / "Cargo.lock").write_bytes(b"c-lock")
    (c_repo / "py").mkdir()
    (c_repo / "py" / "Cargo.lock").write_bytes(b"c-py-lock")

    empty_git_facts = {
        "path": None,
        "head": None,
        "branch": None,
        "dirty": None,
        "dirty_files": [],
        "probe_errors": [],
    }
    monkeypatch.setattr(round2_sources, "_git_facts", lambda repo: dict(empty_git_facts, path=str(repo)))
    monkeypatch.setattr(
        round2_sources,
        "_python_facts",
        lambda python, imports: {"interpreter": str(python), "version": None, "modules": {}},
    )

    env = round2_sources.collect_environment(
        d_repo=d_repo,
        c_repo=c_repo,
        p_repo=p_repo,
        plan_d_commit="d",
        plan_c_commit="c",
        plan_p_commit="p",
    )

    assert env["repositories"]["D"]["lockfile_sha256"] == hashlib.sha256(b"d-lock").hexdigest()
    assert env["repositories"]["P"]["lockfile_sha256"] == hashlib.sha256(b"p-lock").hexdigest()
    assert env["repositories"]["C"]["lockfile_sha256"] == hashlib.sha256(b"c-lock").hexdigest()
    assert env["repositories"]["C"]["py_lockfile_sha256"] == hashlib.sha256(b"c-py-lock").hexdigest()
