"""quip-solver-dwave: D-Wave quantum-annealing miner for the v0.3 protocol.

Binary entry point: ``quip-dwave-qa`` (``python -m quip_solver_dwave``).
"""

from quip_solver_core import miner_pb2 as _miner_pb2

# Kept in step with pyproject.toml by tests/test_version.py. `--version`
# prints this string, so drift here misreports the running binary.
__version__ = "0.3.5"

# sysexits-style exit codes (mirrored by quip_protocol::session::ExitCode)
EXIT_CLEAN = 0
EXIT_CONFIG_INVALID = 64
EXIT_ENV_INCOMPATIBLE = 69
EXIT_INTERNAL_FATAL = 70
EXIT_TOKEN_REJECTED = 77

# Wire identity (Capabilities.backend / .algorithm / .solver_id). The enums'
# stable names are "dwave-qpu" and "quantum-anneal" in
# quip_solver_core.session.BACKEND_NAMES / ALGORITHM_NAMES.
BACKEND = _miner_pb2.BACKEND_DWAVE_QPU
ALGORITHM = _miner_pb2.ALGORITHM_QUANTUM_ANNEAL
SOLVER_ID = "quip-solver-dwave"
# Capability advertisement, shared between --capabilities (cli.py) and the
# live Hello/Capabilities session traffic (session_loop.py) so the two never
# drift apart.
MAX_NODES = 10_000
MAX_EDGES = 100_000
FEATURES = ("quantum-anneal", "native-topology")
