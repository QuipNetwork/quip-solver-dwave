from google.protobuf.internal import containers as _containers
from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from collections.abc import Iterable as _Iterable, Mapping as _Mapping
from typing import ClassVar as _ClassVar, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class Backend(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    BACKEND_UNSPECIFIED: _ClassVar[Backend]
    BACKEND_CPU: _ClassVar[Backend]
    BACKEND_CUDA: _ClassVar[Backend]
    BACKEND_METAL: _ClassVar[Backend]
    BACKEND_ANE: _ClassVar[Backend]
    BACKEND_DWAVE_QPU: _ClassVar[Backend]
    BACKEND_EXEC: _ClassVar[Backend]
    BACKEND_MOCK: _ClassVar[Backend]

class Algorithm(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    ALGORITHM_UNSPECIFIED: _ClassVar[Algorithm]
    ALGORITHM_SA: _ClassVar[Algorithm]
    ALGORITHM_GIBBS: _ClassVar[Algorithm]
    ALGORITHM_QUANTUM_ANNEAL: _ClassVar[Algorithm]
    ALGORITHM_FSA: _ClassVar[Algorithm]
    ALGORITHM_MSA: _ClassVar[Algorithm]
    ALGORITHM_FLATIRON: _ClassVar[Algorithm]
    ALGORITHM_MPS: _ClassVar[Algorithm]
    ALGORITHM_MFA: _ClassVar[Algorithm]
    ALGORITHM_SB: _ClassVar[Algorithm]
    ALGORITHM_BSB: _ClassVar[Algorithm]
    ALGORITHM_GBSB: _ClassVar[Algorithm]
    ALGORITHM_GDSB: _ClassVar[Algorithm]
    ALGORITHM_GGDSB: _ClassVar[Algorithm]
    ALGORITHM_HBSB: _ClassVar[Algorithm]
    ALGORITHM_HDSB: _ClassVar[Algorithm]
    ALGORITHM_SBQA: _ClassVar[Algorithm]
    ALGORITHM_TEDSB: _ClassVar[Algorithm]
    ALGORITHM_EXTERNAL: _ClassVar[Algorithm]

class CoefficientEncoding(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    COEFFICIENT_ENCODING_UNSPECIFIED: _ClassVar[CoefficientEncoding]
    COEFFICIENT_ENCODING_I32: _ClassVar[CoefficientEncoding]
    COEFFICIENT_ENCODING_I16: _ClassVar[CoefficientEncoding]
    COEFFICIENT_ENCODING_I8: _ClassVar[CoefficientEncoding]
    COEFFICIENT_ENCODING_F16: _ClassVar[CoefficientEncoding]
    COEFFICIENT_ENCODING_F32: _ClassVar[CoefficientEncoding]
    COEFFICIENT_ENCODING_F64: _ClassVar[CoefficientEncoding]

class GeneratorAlgorithm(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    GENERATOR_ALGORITHM_UNSPECIFIED: _ClassVar[GeneratorAlgorithm]
    GENERATOR_ALGORITHM_BLAKE3_CHACHA8_V1: _ClassVar[GeneratorAlgorithm]

class JobKind(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    JOB_KIND_UNSPECIFIED: _ClassVar[JobKind]
    ISING_SAMPLE: _ClassVar[JobKind]
    GATE_CIRCUIT: _ClassVar[JobKind]
    ISING_GENERATE: _ClassVar[JobKind]

class RejectReason(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    REJECT_REASON_UNSPECIFIED: _ClassVar[RejectReason]
    UNSUPPORTED_KIND: _ClassVar[RejectReason]
    TOO_LARGE: _ClassVar[RejectReason]
    EXPIRED: _ClassVar[RejectReason]
    OVERLOADED: _ClassVar[RejectReason]
    SHUTTING_DOWN: _ClassVar[RejectReason]
    MALFORMED: _ClassVar[RejectReason]
    TOPOLOGY_MISMATCH: _ClassVar[RejectReason]
    TOPOLOGY_MISSING: _ClassVar[RejectReason]
    TARGET_MISSING: _ClassVar[RejectReason]
BACKEND_UNSPECIFIED: Backend
BACKEND_CPU: Backend
BACKEND_CUDA: Backend
BACKEND_METAL: Backend
BACKEND_ANE: Backend
BACKEND_DWAVE_QPU: Backend
BACKEND_EXEC: Backend
BACKEND_MOCK: Backend
ALGORITHM_UNSPECIFIED: Algorithm
ALGORITHM_SA: Algorithm
ALGORITHM_GIBBS: Algorithm
ALGORITHM_QUANTUM_ANNEAL: Algorithm
ALGORITHM_FSA: Algorithm
ALGORITHM_MSA: Algorithm
ALGORITHM_FLATIRON: Algorithm
ALGORITHM_MPS: Algorithm
ALGORITHM_MFA: Algorithm
ALGORITHM_SB: Algorithm
ALGORITHM_BSB: Algorithm
ALGORITHM_GBSB: Algorithm
ALGORITHM_GDSB: Algorithm
ALGORITHM_GGDSB: Algorithm
ALGORITHM_HBSB: Algorithm
ALGORITHM_HDSB: Algorithm
ALGORITHM_SBQA: Algorithm
ALGORITHM_TEDSB: Algorithm
ALGORITHM_EXTERNAL: Algorithm
COEFFICIENT_ENCODING_UNSPECIFIED: CoefficientEncoding
COEFFICIENT_ENCODING_I32: CoefficientEncoding
COEFFICIENT_ENCODING_I16: CoefficientEncoding
COEFFICIENT_ENCODING_I8: CoefficientEncoding
COEFFICIENT_ENCODING_F16: CoefficientEncoding
COEFFICIENT_ENCODING_F32: CoefficientEncoding
COEFFICIENT_ENCODING_F64: CoefficientEncoding
GENERATOR_ALGORITHM_UNSPECIFIED: GeneratorAlgorithm
GENERATOR_ALGORITHM_BLAKE3_CHACHA8_V1: GeneratorAlgorithm
JOB_KIND_UNSPECIFIED: JobKind
ISING_SAMPLE: JobKind
GATE_CIRCUIT: JobKind
ISING_GENERATE: JobKind
REJECT_REASON_UNSPECIFIED: RejectReason
UNSUPPORTED_KIND: RejectReason
TOO_LARGE: RejectReason
EXPIRED: RejectReason
OVERLOADED: RejectReason
SHUTTING_DOWN: RejectReason
MALFORMED: RejectReason
TOPOLOGY_MISMATCH: RejectReason
TOPOLOGY_MISSING: RejectReason
TARGET_MISSING: RejectReason

class MinerMsg(_message.Message):
    __slots__ = ("hello", "ready", "job_request", "result", "reject", "status", "fatal", "capabilities", "lease_done")
    HELLO_FIELD_NUMBER: _ClassVar[int]
    READY_FIELD_NUMBER: _ClassVar[int]
    JOB_REQUEST_FIELD_NUMBER: _ClassVar[int]
    RESULT_FIELD_NUMBER: _ClassVar[int]
    REJECT_FIELD_NUMBER: _ClassVar[int]
    STATUS_FIELD_NUMBER: _ClassVar[int]
    FATAL_FIELD_NUMBER: _ClassVar[int]
    CAPABILITIES_FIELD_NUMBER: _ClassVar[int]
    LEASE_DONE_FIELD_NUMBER: _ClassVar[int]
    hello: Hello
    ready: Ready
    job_request: JobRequest
    result: Result
    reject: Reject
    status: Status
    fatal: Fatal
    capabilities: Capabilities
    lease_done: LeaseDone
    def __init__(self, hello: _Optional[_Union[Hello, _Mapping]] = ..., ready: _Optional[_Union[Ready, _Mapping]] = ..., job_request: _Optional[_Union[JobRequest, _Mapping]] = ..., result: _Optional[_Union[Result, _Mapping]] = ..., reject: _Optional[_Union[Reject, _Mapping]] = ..., status: _Optional[_Union[Status, _Mapping]] = ..., fatal: _Optional[_Union[Fatal, _Mapping]] = ..., capabilities: _Optional[_Union[Capabilities, _Mapping]] = ..., lease_done: _Optional[_Union[LeaseDone, _Mapping]] = ...) -> None: ...

class CoordMsg(_message.Message):
    __slots__ = ("welcome", "configure", "topology", "job", "cancel", "ping", "shutdown", "set_target", "get_capabilities")
    WELCOME_FIELD_NUMBER: _ClassVar[int]
    CONFIGURE_FIELD_NUMBER: _ClassVar[int]
    TOPOLOGY_FIELD_NUMBER: _ClassVar[int]
    JOB_FIELD_NUMBER: _ClassVar[int]
    CANCEL_FIELD_NUMBER: _ClassVar[int]
    PING_FIELD_NUMBER: _ClassVar[int]
    SHUTDOWN_FIELD_NUMBER: _ClassVar[int]
    SET_TARGET_FIELD_NUMBER: _ClassVar[int]
    GET_CAPABILITIES_FIELD_NUMBER: _ClassVar[int]
    welcome: Welcome
    configure: Configure
    topology: Topology
    job: Job
    cancel: Cancel
    ping: Ping
    shutdown: Shutdown
    set_target: SetTarget
    get_capabilities: GetCapabilities
    def __init__(self, welcome: _Optional[_Union[Welcome, _Mapping]] = ..., configure: _Optional[_Union[Configure, _Mapping]] = ..., topology: _Optional[_Union[Topology, _Mapping]] = ..., job: _Optional[_Union[Job, _Mapping]] = ..., cancel: _Optional[_Union[Cancel, _Mapping]] = ..., ping: _Optional[_Union[Ping, _Mapping]] = ..., shutdown: _Optional[_Union[Shutdown, _Mapping]] = ..., set_target: _Optional[_Union[SetTarget, _Mapping]] = ..., get_capabilities: _Optional[_Union[GetCapabilities, _Mapping]] = ...) -> None: ...

class Hello(_message.Message):
    __slots__ = ("miner_id", "session_token", "capabilities")
    MINER_ID_FIELD_NUMBER: _ClassVar[int]
    SESSION_TOKEN_FIELD_NUMBER: _ClassVar[int]
    CAPABILITIES_FIELD_NUMBER: _ClassVar[int]
    miner_id: str
    session_token: str
    capabilities: Capabilities
    def __init__(self, miner_id: _Optional[str] = ..., session_token: _Optional[str] = ..., capabilities: _Optional[_Union[Capabilities, _Mapping]] = ...) -> None: ...

class Welcome(_message.Message):
    __slots__ = ("protocol_version",)
    PROTOCOL_VERSION_FIELD_NUMBER: _ClassVar[int]
    protocol_version: int
    def __init__(self, protocol_version: _Optional[int] = ...) -> None: ...

class Configure(_message.Message):
    __slots__ = ("queue_depth", "idle_timeout_s", "heartbeat_s", "reconnect_window_s", "backend_toml")
    QUEUE_DEPTH_FIELD_NUMBER: _ClassVar[int]
    IDLE_TIMEOUT_S_FIELD_NUMBER: _ClassVar[int]
    HEARTBEAT_S_FIELD_NUMBER: _ClassVar[int]
    RECONNECT_WINDOW_S_FIELD_NUMBER: _ClassVar[int]
    BACKEND_TOML_FIELD_NUMBER: _ClassVar[int]
    queue_depth: int
    idle_timeout_s: int
    heartbeat_s: int
    reconnect_window_s: int
    backend_toml: str
    def __init__(self, queue_depth: _Optional[int] = ..., idle_timeout_s: _Optional[int] = ..., heartbeat_s: _Optional[int] = ..., reconnect_window_s: _Optional[int] = ..., backend_toml: _Optional[str] = ...) -> None: ...

class Ready(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class Topology(_message.Message):
    __slots__ = ("hash", "nodes", "edges", "allowed_h_milli", "allowed_j_milli")
    HASH_FIELD_NUMBER: _ClassVar[int]
    NODES_FIELD_NUMBER: _ClassVar[int]
    EDGES_FIELD_NUMBER: _ClassVar[int]
    ALLOWED_H_MILLI_FIELD_NUMBER: _ClassVar[int]
    ALLOWED_J_MILLI_FIELD_NUMBER: _ClassVar[int]
    hash: bytes
    nodes: _containers.RepeatedScalarFieldContainer[int]
    edges: EdgeList
    allowed_h_milli: _containers.RepeatedScalarFieldContainer[int]
    allowed_j_milli: _containers.RepeatedScalarFieldContainer[int]
    def __init__(self, hash: _Optional[bytes] = ..., nodes: _Optional[_Iterable[int]] = ..., edges: _Optional[_Union[EdgeList, _Mapping]] = ..., allowed_h_milli: _Optional[_Iterable[int]] = ..., allowed_j_milli: _Optional[_Iterable[int]] = ...) -> None: ...

class SetTarget(_message.Message):
    __slots__ = ("max_energy_milli", "min_solutions", "min_diversity_milli", "num_reads", "num_sweeps", "anneal_time_us", "max_proof_solutions")
    MAX_ENERGY_MILLI_FIELD_NUMBER: _ClassVar[int]
    MIN_SOLUTIONS_FIELD_NUMBER: _ClassVar[int]
    MIN_DIVERSITY_MILLI_FIELD_NUMBER: _ClassVar[int]
    NUM_READS_FIELD_NUMBER: _ClassVar[int]
    NUM_SWEEPS_FIELD_NUMBER: _ClassVar[int]
    ANNEAL_TIME_US_FIELD_NUMBER: _ClassVar[int]
    MAX_PROOF_SOLUTIONS_FIELD_NUMBER: _ClassVar[int]
    max_energy_milli: int
    min_solutions: int
    min_diversity_milli: int
    num_reads: int
    num_sweeps: int
    anneal_time_us: int
    max_proof_solutions: int
    def __init__(self, max_energy_milli: _Optional[int] = ..., min_solutions: _Optional[int] = ..., min_diversity_milli: _Optional[int] = ..., num_reads: _Optional[int] = ..., num_sweeps: _Optional[int] = ..., anneal_time_us: _Optional[int] = ..., max_proof_solutions: _Optional[int] = ...) -> None: ...

class EdgeList(_message.Message):
    __slots__ = ("u", "v")
    U_FIELD_NUMBER: _ClassVar[int]
    V_FIELD_NUMBER: _ClassVar[int]
    u: _containers.RepeatedScalarFieldContainer[int]
    v: _containers.RepeatedScalarFieldContainer[int]
    def __init__(self, u: _Optional[_Iterable[int]] = ..., v: _Optional[_Iterable[int]] = ...) -> None: ...

class IsingProblem(_message.Message):
    __slots__ = ("topology_hash", "edges", "encoding", "scale", "h", "j", "num_reads", "num_sweeps", "anneal_time_us", "initial_spins", "start_beta_milli", "reversal_s_milli", "reversal_pause_us")
    TOPOLOGY_HASH_FIELD_NUMBER: _ClassVar[int]
    EDGES_FIELD_NUMBER: _ClassVar[int]
    ENCODING_FIELD_NUMBER: _ClassVar[int]
    SCALE_FIELD_NUMBER: _ClassVar[int]
    H_FIELD_NUMBER: _ClassVar[int]
    J_FIELD_NUMBER: _ClassVar[int]
    NUM_READS_FIELD_NUMBER: _ClassVar[int]
    NUM_SWEEPS_FIELD_NUMBER: _ClassVar[int]
    ANNEAL_TIME_US_FIELD_NUMBER: _ClassVar[int]
    INITIAL_SPINS_FIELD_NUMBER: _ClassVar[int]
    START_BETA_MILLI_FIELD_NUMBER: _ClassVar[int]
    REVERSAL_S_MILLI_FIELD_NUMBER: _ClassVar[int]
    REVERSAL_PAUSE_US_FIELD_NUMBER: _ClassVar[int]
    topology_hash: bytes
    edges: EdgeList
    encoding: CoefficientEncoding
    scale: int
    h: bytes
    j: bytes
    num_reads: int
    num_sweeps: int
    anneal_time_us: int
    initial_spins: _containers.RepeatedScalarFieldContainer[bytes]
    start_beta_milli: int
    reversal_s_milli: int
    reversal_pause_us: int
    def __init__(self, topology_hash: _Optional[bytes] = ..., edges: _Optional[_Union[EdgeList, _Mapping]] = ..., encoding: _Optional[_Union[CoefficientEncoding, str]] = ..., scale: _Optional[int] = ..., h: _Optional[bytes] = ..., j: _Optional[bytes] = ..., num_reads: _Optional[int] = ..., num_sweeps: _Optional[int] = ..., anneal_time_us: _Optional[int] = ..., initial_spins: _Optional[_Iterable[bytes]] = ..., start_beta_milli: _Optional[int] = ..., reversal_s_milli: _Optional[int] = ..., reversal_pause_us: _Optional[int] = ...) -> None: ...

class Provenance(_message.Message):
    __slots__ = ("is_pow", "order_id")
    IS_POW_FIELD_NUMBER: _ClassVar[int]
    ORDER_ID_FIELD_NUMBER: _ClassVar[int]
    is_pow: bool
    order_id: bytes
    def __init__(self, is_pow: _Optional[bool] = ..., order_id: _Optional[bytes] = ...) -> None: ...

class Job(_message.Message):
    __slots__ = ("job_id", "kind", "generation", "deadline_ms", "ising", "provenance", "generator")
    JOB_ID_FIELD_NUMBER: _ClassVar[int]
    KIND_FIELD_NUMBER: _ClassVar[int]
    GENERATION_FIELD_NUMBER: _ClassVar[int]
    DEADLINE_MS_FIELD_NUMBER: _ClassVar[int]
    ISING_FIELD_NUMBER: _ClassVar[int]
    PROVENANCE_FIELD_NUMBER: _ClassVar[int]
    GENERATOR_FIELD_NUMBER: _ClassVar[int]
    job_id: bytes
    kind: JobKind
    generation: int
    deadline_ms: int
    ising: IsingProblem
    provenance: Provenance
    generator: IsingProblemGenerator
    def __init__(self, job_id: _Optional[bytes] = ..., kind: _Optional[_Union[JobKind, str]] = ..., generation: _Optional[int] = ..., deadline_ms: _Optional[int] = ..., ising: _Optional[_Union[IsingProblem, _Mapping]] = ..., provenance: _Optional[_Union[Provenance, _Mapping]] = ..., generator: _Optional[_Union[IsingProblemGenerator, _Mapping]] = ...) -> None: ...

class JobRequest(_message.Message):
    __slots__ = ("credits",)
    CREDITS_FIELD_NUMBER: _ClassVar[int]
    credits: int
    def __init__(self, credits: _Optional[int] = ...) -> None: ...

class Cancel(_message.Message):
    __slots__ = ("max_generation",)
    MAX_GENERATION_FIELD_NUMBER: _ClassVar[int]
    max_generation: int
    def __init__(self, max_generation: _Optional[int] = ...) -> None: ...

class Ping(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class Shutdown(_message.Message):
    __slots__ = ("grace_ms",)
    GRACE_MS_FIELD_NUMBER: _ClassVar[int]
    grace_ms: int
    def __init__(self, grace_ms: _Optional[int] = ...) -> None: ...

class Solution(_message.Message):
    __slots__ = ("energy_milli", "spins")
    ENERGY_MILLI_FIELD_NUMBER: _ClassVar[int]
    SPINS_FIELD_NUMBER: _ClassVar[int]
    energy_milli: int
    spins: bytes
    def __init__(self, energy_milli: _Optional[int] = ..., spins: _Optional[bytes] = ...) -> None: ...

class SamplerMeta(_message.Message):
    __slots__ = ("reads", "sweeps", "device_access_time_us", "qpu_access_us", "extra")
    class ExtraEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: str
        def __init__(self, key: _Optional[str] = ..., value: _Optional[str] = ...) -> None: ...
    READS_FIELD_NUMBER: _ClassVar[int]
    SWEEPS_FIELD_NUMBER: _ClassVar[int]
    DEVICE_ACCESS_TIME_US_FIELD_NUMBER: _ClassVar[int]
    QPU_ACCESS_US_FIELD_NUMBER: _ClassVar[int]
    EXTRA_FIELD_NUMBER: _ClassVar[int]
    reads: int
    sweeps: int
    device_access_time_us: int
    qpu_access_us: int
    extra: _containers.ScalarMap[str, str]
    def __init__(self, reads: _Optional[int] = ..., sweeps: _Optional[int] = ..., device_access_time_us: _Optional[int] = ..., qpu_access_us: _Optional[int] = ..., extra: _Optional[_Mapping[str, str]] = ...) -> None: ...

class Result(_message.Message):
    __slots__ = ("job_id", "solutions", "meta", "salt", "nonce")
    JOB_ID_FIELD_NUMBER: _ClassVar[int]
    SOLUTIONS_FIELD_NUMBER: _ClassVar[int]
    META_FIELD_NUMBER: _ClassVar[int]
    SALT_FIELD_NUMBER: _ClassVar[int]
    NONCE_FIELD_NUMBER: _ClassVar[int]
    job_id: bytes
    solutions: _containers.RepeatedCompositeFieldContainer[Solution]
    meta: SamplerMeta
    salt: bytes
    nonce: bytes
    def __init__(self, job_id: _Optional[bytes] = ..., solutions: _Optional[_Iterable[_Union[Solution, _Mapping]]] = ..., meta: _Optional[_Union[SamplerMeta, _Mapping]] = ..., salt: _Optional[bytes] = ..., nonce: _Optional[bytes] = ...) -> None: ...

class Reject(_message.Message):
    __slots__ = ("job_id", "reason")
    JOB_ID_FIELD_NUMBER: _ClassVar[int]
    REASON_FIELD_NUMBER: _ClassVar[int]
    job_id: bytes
    reason: RejectReason
    def __init__(self, job_id: _Optional[bytes] = ..., reason: _Optional[_Union[RejectReason, str]] = ...) -> None: ...

class Status(_message.Message):
    __slots__ = ("miner_id", "utilization", "jobs_done", "abandoned_generation", "sampler_stats")
    class SamplerStatsEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: str
        def __init__(self, key: _Optional[str] = ..., value: _Optional[str] = ...) -> None: ...
    MINER_ID_FIELD_NUMBER: _ClassVar[int]
    UTILIZATION_FIELD_NUMBER: _ClassVar[int]
    JOBS_DONE_FIELD_NUMBER: _ClassVar[int]
    ABANDONED_GENERATION_FIELD_NUMBER: _ClassVar[int]
    SAMPLER_STATS_FIELD_NUMBER: _ClassVar[int]
    miner_id: str
    utilization: float
    jobs_done: int
    abandoned_generation: int
    sampler_stats: _containers.ScalarMap[str, str]
    def __init__(self, miner_id: _Optional[str] = ..., utilization: _Optional[float] = ..., jobs_done: _Optional[int] = ..., abandoned_generation: _Optional[int] = ..., sampler_stats: _Optional[_Mapping[str, str]] = ...) -> None: ...

class Fatal(_message.Message):
    __slots__ = ("exit_code", "reason", "restart_required")
    EXIT_CODE_FIELD_NUMBER: _ClassVar[int]
    REASON_FIELD_NUMBER: _ClassVar[int]
    RESTART_REQUIRED_FIELD_NUMBER: _ClassVar[int]
    exit_code: int
    reason: str
    restart_required: bool
    def __init__(self, exit_code: _Optional[int] = ..., reason: _Optional[str] = ..., restart_required: _Optional[bool] = ...) -> None: ...

class GetCapabilities(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class Capabilities(_message.Message):
    __slots__ = ("supported_kinds", "max_nodes", "max_edges", "features", "protocol_version", "stream_width", "native_topology_hash", "encodings", "generators", "backend", "algorithm", "max_degree", "max_local_field_milli", "exact_resolution_milli")
    SUPPORTED_KINDS_FIELD_NUMBER: _ClassVar[int]
    MAX_NODES_FIELD_NUMBER: _ClassVar[int]
    MAX_EDGES_FIELD_NUMBER: _ClassVar[int]
    FEATURES_FIELD_NUMBER: _ClassVar[int]
    PROTOCOL_VERSION_FIELD_NUMBER: _ClassVar[int]
    STREAM_WIDTH_FIELD_NUMBER: _ClassVar[int]
    NATIVE_TOPOLOGY_HASH_FIELD_NUMBER: _ClassVar[int]
    ENCODINGS_FIELD_NUMBER: _ClassVar[int]
    GENERATORS_FIELD_NUMBER: _ClassVar[int]
    BACKEND_FIELD_NUMBER: _ClassVar[int]
    ALGORITHM_FIELD_NUMBER: _ClassVar[int]
    MAX_DEGREE_FIELD_NUMBER: _ClassVar[int]
    MAX_LOCAL_FIELD_MILLI_FIELD_NUMBER: _ClassVar[int]
    EXACT_RESOLUTION_MILLI_FIELD_NUMBER: _ClassVar[int]
    supported_kinds: _containers.RepeatedScalarFieldContainer[JobKind]
    max_nodes: int
    max_edges: int
    features: _containers.RepeatedScalarFieldContainer[str]
    protocol_version: int
    stream_width: int
    native_topology_hash: bytes
    encodings: _containers.RepeatedScalarFieldContainer[CoefficientEncoding]
    generators: _containers.RepeatedScalarFieldContainer[GeneratorAlgorithm]
    backend: Backend
    algorithm: Algorithm
    max_degree: int
    max_local_field_milli: int
    exact_resolution_milli: int
    def __init__(self, supported_kinds: _Optional[_Iterable[_Union[JobKind, str]]] = ..., max_nodes: _Optional[int] = ..., max_edges: _Optional[int] = ..., features: _Optional[_Iterable[str]] = ..., protocol_version: _Optional[int] = ..., stream_width: _Optional[int] = ..., native_topology_hash: _Optional[bytes] = ..., encodings: _Optional[_Iterable[_Union[CoefficientEncoding, str]]] = ..., generators: _Optional[_Iterable[_Union[GeneratorAlgorithm, str]]] = ..., backend: _Optional[_Union[Backend, str]] = ..., algorithm: _Optional[_Union[Algorithm, str]] = ..., max_degree: _Optional[int] = ..., max_local_field_milli: _Optional[int] = ..., exact_resolution_milli: _Optional[int] = ...) -> None: ...

class IsingProblemGenerator(_message.Message):
    __slots__ = ("algorithm", "topology_hash", "last_proof_block_hash", "miner_account", "base_salt", "salt_start", "salt_count")
    ALGORITHM_FIELD_NUMBER: _ClassVar[int]
    TOPOLOGY_HASH_FIELD_NUMBER: _ClassVar[int]
    LAST_PROOF_BLOCK_HASH_FIELD_NUMBER: _ClassVar[int]
    MINER_ACCOUNT_FIELD_NUMBER: _ClassVar[int]
    BASE_SALT_FIELD_NUMBER: _ClassVar[int]
    SALT_START_FIELD_NUMBER: _ClassVar[int]
    SALT_COUNT_FIELD_NUMBER: _ClassVar[int]
    algorithm: GeneratorAlgorithm
    topology_hash: bytes
    last_proof_block_hash: bytes
    miner_account: bytes
    base_salt: bytes
    salt_start: int
    salt_count: int
    def __init__(self, algorithm: _Optional[_Union[GeneratorAlgorithm, str]] = ..., topology_hash: _Optional[bytes] = ..., last_proof_block_hash: _Optional[bytes] = ..., miner_account: _Optional[bytes] = ..., base_salt: _Optional[bytes] = ..., salt_start: _Optional[int] = ..., salt_count: _Optional[int] = ...) -> None: ...

class LeaseDone(_message.Message):
    __slots__ = ("job_id", "salts_done", "best_energy_milli")
    JOB_ID_FIELD_NUMBER: _ClassVar[int]
    SALTS_DONE_FIELD_NUMBER: _ClassVar[int]
    BEST_ENERGY_MILLI_FIELD_NUMBER: _ClassVar[int]
    job_id: bytes
    salts_done: int
    best_energy_milli: int
    def __init__(self, job_id: _Optional[bytes] = ..., salts_done: _Optional[int] = ..., best_energy_milli: _Optional[int] = ...) -> None: ...
