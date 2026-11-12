import struct

import pytest

from quip_solver_dwave import chain

# A real answer of QuantumPowApi_qblock_by_id for qblock 8400, read on 2026-10-05.
RAW_8400 = (
    "0x01449bdfcdf08621dea2ce11fc50f001475a362afc997f4c18008606441f5b7112"
    "5c023c22c50804c0302777420600000000000000000000000000000000000000"
    "c0441fffffffffff0010a5d4e80000000000000000000000"
    "9922060001000000227a1fffffffffff00000000"
    "292593e6113eee6bb80646a916803833e445db6a6a5b976e75a2ea615c01c1b3"
    "cbec1eb4e9dcc7ae343b386902d4dbba19b694a66cebf95bc06bd9260c3270e7"
    "93991e0000000000"
    "ade07330390be65cb3f208664a3a04392441cb8dfcacfb915b5f290eeedc7a23"
)


def test_decode_qblock_reads_every_field_of_a_real_record():
    q = chain.decode_qblock(8400, RAW_8400)
    assert q is not None
    assert q.miner == "449bdfcdf08621dea2ce11fc50f001475a362afc997f4c18008606441f5b7112"
    assert q.salt == "5c023c22c50804c0302777420600000000000000000000000000000000000000"
    assert q.energy_milli == -14_728_000
    assert q.block == 402_073
    assert q.min_solutions == 1
    assert q.gate_milli == -14_714_334
    assert q.min_diversity_milli == 0
    assert q.last_proof_block_hash == "292593e6113eee6bb80646a916803833e445db6a6a5b976e75a2ea615c01c1b3"
    assert q.topology_hash == "cbec1eb4e9dcc7ae343b386902d4dbba19b694a66cebf95bc06bd9260c3270e7"
    assert q.device_access_us == 2_005_395
    # Reversed: the draw takes this byte order.
    assert q.nonce == "237adcee0e295f5b91fbacfc8dcb412439043a4a6608f2b35ce60b393073e0ad"


def test_decode_qblock_returns_none_for_a_missing_qblock():
    assert chain.decode_qblock(10**9, "0x00") is None


def test_decode_qblock_refuses_a_short_record():
    with pytest.raises(chain.ChainError, match="expected 213 bytes"):
        chain.decode_qblock(8400, RAW_8400[:-2])


def test_decode_latest_id_and_difficulty_read_real_answers():
    assert chain.decode_latest_id("0x01d224000000000000") == 9426
    assert chain.decode_difficulty("0x01000000ea5c1fffffffffff00000000") == {
        "min_solutions": 1, "gate_milli": -14_721_814, "min_diversity_milli": 0,
    }


def fake_post(answers):
    def post(url, batch):
        return [{"id": call["id"], "result": answers[call["id"]]} for call in batch if call["id"] in answers]
    return post


def test_fetch_qblocks_sends_the_id_as_a_little_endian_u64_and_keeps_order():
    seen = []

    def post(url, batch):
        seen.extend(call["params"] for call in batch)
        return [{"id": call["id"], "result": RAW_8400} for call in reversed(batch)]

    found = chain.fetch_qblocks(8400, 8402, url="http://node", post=post)
    assert [q.qblock_id for q in found] == [8400, 8401, 8402]
    assert seen[0] == ["QuantumPowApi_qblock_by_id", "0x" + struct.pack("<Q", 8400).hex()]


def test_fetch_qblocks_stops_on_a_missing_answer():
    with pytest.raises(chain.ChainError, match="qblock 8401: no result"):
        chain.fetch_qblocks(8400, 8401, url="http://node", post=fake_post({8400: RAW_8400}))


def test_rounds_use_the_previous_qblock_as_the_round_start():
    a = chain.decode_qblock(8400, RAW_8400)
    assert a is not None
    b = chain.QBlock(**{**a.__dict__, "qblock_id": 8401, "block": a.block + 79, "gate_milli": -14_709_553})
    assert chain.rounds([a, b]) == [
        {"qblock_id": 8401, "start_block": 402_073, "blocks": 79, "gate_milli": -14_709_553,
         "energy_milli": -14_728_000}
    ]


def test_rounds_refuse_a_gap_in_the_ids():
    a = chain.decode_qblock(8400, RAW_8400)
    assert a is not None
    c = chain.QBlock(**{**a.__dict__, "qblock_id": 8402})
    with pytest.raises(chain.ChainError, match="not consecutive"):
        chain.rounds([a, c])
