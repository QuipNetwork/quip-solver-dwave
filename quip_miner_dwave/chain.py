"""Read qblocks from the Aglais chain through the ``QuantumPowApi`` runtime API.

A qblock record is the SCALE encoding of ``Option<QBlockWithNonce>``: one tag
byte, then a fixed layout of 212 bytes. The round of qblock ``i`` starts at the
block of qblock ``i - 1``, because the pallet stores no start time.
"""

from __future__ import annotations

import json
import struct
import urllib.request
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence

DEFAULT_RPC = "https://bootnode-1.aglais.quip.network:20049/rpc"
QBLOCK_BYTES = 213
BLOCK_SECONDS = 6.02
BATCH = 100

Post = Callable[[str, List[Dict[str, Any]]], List[Dict[str, Any]]]


class ChainError(RuntimeError):
    """The node answered with something this module cannot use."""


@dataclass(frozen=True)
class QBlock:
    qblock_id: int
    miner: str
    salt: str
    energy_milli: int
    block: int
    min_solutions: int
    gate_milli: int
    min_diversity_milli: int
    last_proof_block_hash: str
    topology_hash: str
    device_access_us: int
    nonce: str


def decode_qblock(qblock_id: int, raw_hex: str) -> Optional[QBlock]:
    """Decode one ``QuantumPowApi_qblock_by_id`` result. ``None`` means the chain has no such qblock."""
    raw = bytes.fromhex(raw_hex.removeprefix("0x"))
    if raw in (b"", b"\x00"):
        return None
    if len(raw) != QBLOCK_BYTES or raw[0] != 1:
        raise ChainError(f"qblock {qblock_id}: expected {QBLOCK_BYTES} bytes with tag 1, got {len(raw)} bytes")
    (energy,) = struct.unpack_from("<q", raw, 65)
    block, min_solutions = struct.unpack_from("<II", raw, 89)
    (gate,) = struct.unpack_from("<q", raw, 97)
    (min_diversity,) = struct.unpack_from("<I", raw, 105)
    (access,) = struct.unpack_from("<Q", raw, 173)
    return QBlock(
        qblock_id=qblock_id,
        miner=raw[1:33].hex(),
        salt=raw[33:65].hex(),
        energy_milli=energy,
        block=block,
        min_solutions=min_solutions,
        gate_milli=gate,
        min_diversity_milli=min_diversity,
        last_proof_block_hash=raw[109:141].hex(),
        topology_hash=raw[141:173].hex(),
        device_access_us=access,
        # The API returns the nonce little-endian. The draw takes the reversed bytes.
        nonce=raw[181:213][::-1].hex(),
    )


def decode_latest_id(raw_hex: str) -> int:
    raw = bytes.fromhex(raw_hex.removeprefix("0x"))
    if len(raw) != 9 or raw[0] != 1:
        raise ChainError(f"latest qblock id: expected 9 bytes with tag 1, got {raw.hex()}")
    return struct.unpack_from("<Q", raw, 1)[0]


def decode_difficulty(raw_hex: str) -> Dict[str, int]:
    raw = bytes.fromhex(raw_hex.removeprefix("0x"))
    if len(raw) != 16:
        raise ChainError(f"difficulty: expected 16 bytes, got {len(raw)}")
    min_solutions, gate, min_diversity = struct.unpack("<IqI", raw)
    return {"min_solutions": min_solutions, "gate_milli": gate, "min_diversity_milli": min_diversity}


def http_post(url: str, batch: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    request = urllib.request.Request(
        url, data=json.dumps(batch).encode(), headers={"content-type": "application/json"}
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        answer = json.loads(response.read())
    return answer if isinstance(answer, list) else [answer]


def state_call(method: str, argument_hex: str, request_id: int) -> Dict[str, Any]:
    return {"jsonrpc": "2.0", "id": request_id, "method": "state_call", "params": [method, argument_hex]}


def fetch_qblocks(first: int, last: int, url: str = DEFAULT_RPC, post: Post = http_post) -> List[QBlock]:
    """Qblocks ``first..=last`` in id order. A missing id or an RPC error stops the fetch."""
    found: List[QBlock] = []
    ids = list(range(first, last + 1))
    for start in range(0, len(ids), BATCH):
        chunk = ids[start : start + BATCH]
        batch = [state_call("QuantumPowApi_qblock_by_id", "0x" + struct.pack("<Q", i).hex(), i) for i in chunk]
        answers = {int(answer["id"]): answer for answer in post(url, batch)}
        for qblock_id in chunk:
            answer = answers.get(qblock_id)
            if answer is None or "result" not in answer:
                raise ChainError(f"qblock {qblock_id}: no result ({answer})")
            record = decode_qblock(qblock_id, answer["result"])
            if record is None:
                raise ChainError(f"qblock {qblock_id} does not exist")
            found.append(record)
    return found


def rounds(qblocks: Sequence[QBlock]) -> List[Dict[str, int]]:
    """One row per qblock after the first: the round that the qblock ended.

    ``blocks`` is the round length. The ids must be consecutive, because the
    start of a round is the block of the previous qblock.
    """
    rows = []
    for before, after in zip(qblocks, qblocks[1:]):
        if after.qblock_id != before.qblock_id + 1:
            raise ChainError(f"qblock ids {before.qblock_id} and {after.qblock_id} are not consecutive")
        rows.append(
            {
                "qblock_id": after.qblock_id,
                "start_block": before.block,
                "blocks": after.block - before.block,
                "gate_milli": after.gate_milli,
                "energy_milli": after.energy_milli,
            }
        )
    return rows
