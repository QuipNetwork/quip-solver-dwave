import importlib.util
import json
import struct
from pathlib import Path

import pytest

from quip_miner_dwave.tests.test_chain import RAW_8400

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "chain_fetch.py"
TOPOLOGY = "cbec1eb4e9dcc7ae343b386902d4dbba19b694a66cebf95bc06bd9260c3270e7"


def load():
    spec = importlib.util.spec_from_file_location("chain_fetch", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def node(records):
    """A fake node: the latest id, a difficulty, and the given raw qblocks by id."""

    def post(url, batch):
        answers = []
        for call in batch:
            method, argument = call["params"]
            if method == "QuantumPowApi_latest_qblock_id":
                result = "0x01" + struct.pack("<Q", max(records)).hex()
            elif method == "QuantumPowApi_current_difficulty":
                result = "0x01000000ea5c1fffffffffff00000000"
            else:
                result = records[struct.unpack("<Q", bytes.fromhex(argument[2:]))[0]]
            answers.append({"id": call["id"], "result": result})
        return answers

    return post


def shifted(raw, block):
    """The real record with another block number."""
    body = bytearray.fromhex(raw[2:])
    struct.pack_into("<I", body, 89, block)
    return "0x" + body.hex()


def test_fetch_writes_qblocks_and_rounds(tmp_path):
    records = {8400: RAW_8400, 8401: shifted(RAW_8400, 402_152)}
    code = load().main(["--count", "2", "--topology-hash", TOPOLOGY, "--out-dir", str(tmp_path)], post=node(records))
    assert code == 0
    lines = [json.loads(line) for line in (tmp_path / "qblocks.jsonl").read_text().splitlines()]
    assert [line["qblock_id"] for line in lines] == [8400, 8401]
    assert lines[0]["energy_milli"] == -14_728_000
    rounds = json.loads((tmp_path / "rounds.json").read_text())
    assert rounds["latest_qblock_id"] == 8401
    assert rounds["current_difficulty"]["gate_milli"] == -14_721_814
    assert rounds["rounds"] == [
        {"qblock_id": 8401, "start_block": 402_073, "blocks": 79, "gate_milli": -14_714_334,
         "energy_milli": -14_728_000}
    ]


def test_fetch_refuses_a_qblock_of_another_topology(tmp_path):
    with pytest.raises(SystemExit, match="qblocks \\[8400\\] use another topology"):
        load().main(["--count", "1", "--topology-hash", "00" * 32, "--out-dir", str(tmp_path)],
                    post=node({8400: RAW_8400}))
    assert not (tmp_path / "qblocks.jsonl").exists()
