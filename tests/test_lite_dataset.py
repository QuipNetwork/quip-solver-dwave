import importlib.util
import pathlib

SCRIPT_PATH = pathlib.Path(__file__).resolve().parents[1] / "scripts" / "lite_dataset.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("lite_dataset", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


lite_dataset = _load_module()


def test_already_done_skips_a_row_truncated_before_sweeps(tmp_path):
    path = tmp_path / "lite.csv"
    path.write_text(
        "nonce,sweeps,lite_best_milli,lite_wall_ms\n"
        "aaaa,128,-100,1.2\n"
        "bbbb"
    )
    assert lite_dataset.already_done(str(path)) == {("aaaa", 128)}


def test_already_done_skips_a_row_truncated_mid_field(tmp_path):
    path = tmp_path / "lite.csv"
    path.write_text(
        "nonce,sweeps,lite_best_milli,lite_wall_ms\n"
        "aaaa,128,-100,1.2\n"
        "bbbb,256,-5"
    )
    assert lite_dataset.already_done(str(path)) == {("aaaa", 128), ("bbbb", 256)}


def test_ensure_trailing_newline_adds_newline_to_truncated_tail(tmp_path):
    path = tmp_path / "lite.csv"
    path.write_bytes(b"nonce,sweeps,lite_best_milli,lite_wall_ms\naaaa,128,-100,1.2\nbbbb")
    lite_dataset.ensure_trailing_newline(str(path))
    assert path.read_bytes().endswith(b"\n")


def test_ensure_trailing_newline_is_a_no_op_when_already_clean(tmp_path):
    path = tmp_path / "lite.csv"
    original = b"nonce,sweeps,lite_best_milli,lite_wall_ms\naaaa,128,-100,1.2\n"
    path.write_bytes(original)
    lite_dataset.ensure_trailing_newline(str(path))
    assert path.read_bytes() == original


def test_ensure_trailing_newline_handles_missing_file(tmp_path):
    path = tmp_path / "missing.csv"
    lite_dataset.ensure_trailing_newline(str(path))
    assert not path.exists()
