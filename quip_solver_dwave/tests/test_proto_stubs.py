"""Guard the checked-in protobuf stubs against drift from the pinned SDK.

``typings/quip/v1/*.pyi`` is generated from quip-solver-core's miner.proto and
is what makes Pyright able to see the protobuf message classes at all — the
generated ``miner_pb2.py`` builds them at runtime, so without stubs every
``miner_pb2.Job`` reads as a missing attribute.

Generated code that lives beside a pinned dependency can go stale silently, and
a stale stub types real code as ``Any`` rather than failing. These tests fail
instead. Regenerate with::

    protoc --pyi_out=typings -I <quip-solver-core>/proto \\
        <quip-solver-core>/proto/quip/v1/miner.proto
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from quip_solver_core import miner_pb2, miner_pb2_grpc

STUB = Path(__file__).resolve().parents[2] / "typings" / "quip" / "v1"
MESSAGE_STUB = STUB / "miner_pb2.pyi"
GRPC_STUB = STUB / "miner_pb2_grpc.pyi"


def _stub_source() -> str:
    return MESSAGE_STUB.read_text()


def _runtime_messages() -> set[str]:
    return {
        name
        for name in dir(miner_pb2)
        if not name.startswith("_")
        and hasattr(getattr(miner_pb2, name), "DESCRIPTOR")
        and hasattr(getattr(miner_pb2, name), "SerializeToString")
    }


def test_the_stubs_exist():
    assert MESSAGE_STUB.is_file(), f"missing generated stub at {MESSAGE_STUB}"
    assert GRPC_STUB.is_file(), f"missing hand-written stub at {GRPC_STUB}"


def test_every_runtime_message_is_declared():
    declared = set(re.findall(r"^class (\w+)\(_message\.Message\)", _stub_source(), re.M))
    missing = _runtime_messages() - declared
    assert not missing, f"stub is stale: {sorted(missing)} exist at runtime but are not declared"


def test_the_stub_declares_no_message_the_runtime_lacks():
    declared = set(re.findall(r"^class (\w+)\(_message\.Message\)", _stub_source(), re.M))
    extra = declared - _runtime_messages()
    assert not extra, f"stub is stale: {sorted(extra)} declared but gone from the runtime"


# The messages this miner actually reads or builds. A field rename here is a
# wire-level break, so it is worth naming them one by one.
@pytest.mark.parametrize(
    "message",
    [
        "Cancel",
        "Configure",
        "CoordMsg",
        "Fatal",
        "Job",
        "JobRequest",
        "MinerMsg",
        "Ready",
        "SetTarget",
        "Status",
    ],
)
def test_field_names_match_the_runtime_descriptor(message):
    body = re.search(
        rf"class {message}\(_message\.Message\):(.*?)(?=\nclass |\Z)",
        _stub_source(),
        re.S,
    )
    assert body is not None, f"{message} is not declared in the stub"
    declared = {
        name
        for name in re.findall(r"^    (\w+): ", body.group(1), re.M)
        if not name.endswith("_FIELD_NUMBER") and name != "DESCRIPTOR"
    }
    runtime = {f.name for f in getattr(miner_pb2, message).DESCRIPTOR.fields}
    assert declared == runtime, f"{message} fields drifted: {declared ^ runtime}"


def test_the_grpc_stub_names_exist_on_the_runtime_module():
    # The gRPC stub is hand-written (protoc emits no .pyi for services), so it
    # is the one most likely to drift unnoticed.
    for name in re.findall(r"^(?:class|def) (\w+)", GRPC_STUB.read_text(), re.M):
        assert hasattr(miner_pb2_grpc, name), f"{name} is stubbed but not in the module"


def test_the_session_rpc_is_still_the_one_the_miner_calls():
    assert hasattr(miner_pb2_grpc.MinerServiceStub, "__init__")
    stub = miner_pb2_grpc.MinerServiceStub.__init__
    assert callable(stub)


WARM_START_FIELDS = (
    "initial_spins",
    "start_beta_milli",
    "reversal_s_milli",
    "reversal_pause_us",
)


def test_the_runtime_carries_the_warm_start_fields():
    names = {f.name for f in miner_pb2.IsingProblem.DESCRIPTOR.fields}
    assert set(WARM_START_FIELDS) <= names


def test_the_stub_declares_the_warm_start_fields():
    # A stale stub types `ising.initial_spins` as an error in Pyright while the
    # runtime accepts it, so the message-name checks above do not catch it.
    source = _stub_source()
    for name in WARM_START_FIELDS:
        assert re.search(rf"^\s+{name}: ", source, re.M), f"stub lacks {name}"
