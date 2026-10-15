"""quip-solver-dwave: D-Wave quantum-annealing miner for the v0.3 protocol.

Binary entry point: ``quip-dwave-qa`` (``python -m quip_solver_dwave``).
"""

# Kept in step with pyproject.toml by tests/test_version.py. `--version`
# prints this string, so drift here misreports the running binary.
__version__ = "0.3.5"

# sysexits-style exit codes (mirrored by quip_protocol::session::ExitCode)
EXIT_CLEAN = 0
EXIT_CONFIG_INVALID = 64
EXIT_ENV_INCOMPATIBLE = 69
EXIT_INTERNAL_FATAL = 70
EXIT_TOKEN_REJECTED = 77

BACKEND = "dwave-qpu"
ALGORITHM = "quantum-anneal"
# Capability advertisement, shared between --capabilities (cli.py) and the
# live Hello/Capabilities session traffic (session_loop.py) so the two never
# drift apart.
MAX_NODES = 10_000
MAX_EDGES = 100_000
# "initial-spins" is SPEC section 3's promise that a job's start state is
# used: this backend reverse-anneals from it (see quip_solver_dwave.warm).
FEATURES = ("quantum-anneal", "native-topology", "initial-spins")
