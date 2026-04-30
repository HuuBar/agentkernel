"""AgentKernel — Core type definitions and data models."""
from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Any, Callable, Coroutine, Dict, List, Optional, Protocol, Set, TypeVar, Union
from datetime import datetime
from uuid import UUID, uuid4


# ---------------------------------------------------------------------------
# Identity & Addressing
# ---------------------------------------------------------------------------

class AgentId:
    """Globally unique agent identifier with hierarchical naming.
    
    Format: "realm::namespace::local_name#instance"
    Examples: "prod::research::planner#7a3f", "dev::coding::reviewer"
    """
    def __init__(self, local: str, namespace: str = "default", realm: str = "local", instance: Optional[str] = None) -> None:
        self.local = local
        self.namespace = namespace
        self.realm = realm
        self.instance = instance or uuid4().hex[:6]
    
    @property
    def full(self) -> str:
        return f"{self.realm}::{self.namespace}::{self.local}#{self.instance}"
    
    def __str__(self) -> str:
        return self.full
    
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, AgentId):
            return NotImplemented
        return self.full == other.full
    
    def __hash__(self) -> int:
        return hash(self.full)


# ---------------------------------------------------------------------------
# Messages
# ---------------------------------------------------------------------------

class MessageKind(enum.Enum):
    TASK = "task"           # Request to perform work
    RESULT = "result"       # Response to a task
    EVENT = "event"         # Async notification
    SYSTEM = "system"       # Kernel-level control (supervision)
    META = "meta"           # Metacognitive introspection


@dataclass(frozen=True)
class Message:
    """Immutable message passed between actors."""
    id: UUID = field(default_factory=uuid4)
    kind: MessageKind = MessageKind.TASK
    sender: Optional[AgentId] = None
    recipient: AgentId = field(default_factory=lambda: AgentId("kernel"))
    payload: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)
    correlation_id: Optional[UUID] = None  # Links result to original task
    contract_id: Optional[UUID] = None       # Links to governing contract


# ---------------------------------------------------------------------------
# Agent Contracts (Resource Governance)
# ---------------------------------------------------------------------------

class ContractMode(enum.Enum):
    URGENT = "urgent"       # Minimize time, accept higher cost
    BALANCED = "balanced"   # Optimize quality-cost-time
    ECONOMICAL = "economical"  # Minimize cost, accept longer runtime


@dataclass
class ResourceBudget:
    """Multi-dimensional resource constraints."""
    max_tokens: int = 100_000
    max_api_calls: int = 50
    max_iterations: int = 10
    max_web_searches: int = 10
    max_compute_seconds: float = 300.0
    max_cost_usd: float = 5.0


@dataclass
class AgentContract:
    """Formal contract governing an agent's execution.
    
    C = (I, O, S, R, T, Φ, Ψ) per Agent Contracts paper (arXiv:2601.08815)
    """
    id: UUID = field(default_factory=uuid4)
    # I: Input specification
    input_schema: Dict[str, Any] = field(default_factory=dict)
    # O: Output specification  
    output_schema: Dict[str, Any] = field(default_factory=dict)
    quality_threshold: float = 0.7  # Q_min
    # S: Skills (tools) available
    allowed_skills: Set[str] = field(default_factory=set)
    # R: Resource constraints
    budget: ResourceBudget = field(default_factory=ResourceBudget)
    # T: Temporal constraints
    deadline: Optional[datetime] = None
    max_duration_seconds: float = 300.0
    # Φ: Success criteria
    success_predicate: Optional[Callable[[Any], bool]] = None
    # Ψ: Termination conditions
    mode: ContractMode = ContractMode.BALANCED
    # Conservation tracking
    consumed_tokens: int = 0
    consumed_api_calls: int = 0
    consumed_iterations: int = 0
    consumed_cost_usd: float = 0.0
    start_time: Optional[datetime] = None
    status: ContractStatus = field(default_factory=lambda: ContractStatus.PENDING)


class ContractStatus(enum.Enum):
    PENDING = "pending"
    ACTIVE = "active"
    VIOLATED = "violated"   # Budget exceeded
    EXPIRED = "expired"     # Deadline missed
    SATISFIED = "satisfied" # Success criteria met
    TERMINATED = "terminated"


# ---------------------------------------------------------------------------
# Actor State
# ---------------------------------------------------------------------------

class ActorStatus(enum.Enum):
    SPAWNING = "spawning"
    IDLE = "idle"
    PROCESSING = "processing"
    SUSPENDED = "suspended"
    CRASHED = "crashed"
    TERMINATED = "terminated"


@dataclass
class ActorState:
    """Serializable state snapshot of an actor."""
    agent_id: AgentId
    status: ActorStatus = ActorStatus.SPAWNING
    mailbox_size: int = 0
    processed_messages: int = 0
    current_contract: Optional[UUID] = None
    memory_refs: List[str] = field(default_factory=list)  # Keys into event store
    capability_fingerprint: str = ""  # Hash of skills + config
    last_checkpoint_at: Optional[datetime] = None


# ---------------------------------------------------------------------------
# Sandbox
# ---------------------------------------------------------------------------

class IsolationLevel(enum.Enum):
    """Threat-adaptive isolation gradient."""
    NONE = "none"               # Trusted code, no isolation
    PROCESS = "process"         # Separate process
    CONTAINER = "container"     # Docker/containerd
    GVISOR = "gvisor"           # Userspace kernel
    MICROVM = "microvm"         # Firecracker/Kata
    TEE = "tee"                 # Confidential computing


@dataclass
class SandboxPolicy:
    """Policy-driven sandbox selection."""
    default_isolation: IsolationLevel = IsolationLevel.CONTAINER
    network_access: bool = False
    filesystem_access: str = "readonly"  # none/readonly/readwrite
    max_cpu_cores: int = 1
    max_memory_mb: int = 512
    allowed_executables: Set[str] = field(default_factory=set)
    observation_filter: Optional[str] = None  # Regex/classifier for PII filtering


# ---------------------------------------------------------------------------
# Metacognition
# ---------------------------------------------------------------------------

class MetaLevel(enum.Enum):
    """Metacognitive monitoring levels."""
    COGNITIVE = 1      # Level 1: Task execution
    MONITOR = 2        # Level 2a: Observe cognitive layer
    GENERATE = 3       # Level 2b: Generate alternative strategies
    VERIFY = 4         # Level 2c: Validate output quality
    REVISE = 5         # Level 2d: Trigger self-correction


@dataclass
class MetacognitiveRecord:
    """Record of a metacognitive episode."""
    agent_id: AgentId
    level: MetaLevel
    trigger: str  # What triggered introspection
    observation: Dict[str, Any] = field(default_factory=dict)
    decision: str = ""  # What action was decided
    confidence: float = 0.0  # 0-1
    timestamp: datetime = field(default_factory=datetime.utcnow)


# ---------------------------------------------------------------------------
# Events (for Event Sourcing)
# ---------------------------------------------------------------------------

class EventType(enum.Enum):
    ACTOR_SPAWNED = "actor_spawned"
    MESSAGE_RECEIVED = "message_received"
    MESSAGE_PROCESSED = "message_processed"
    CONTRACT_CREATED = "contract_created"
    CONTRACT_UPDATED = "contract_updated"
    CONTRACT_VIOLATED = "contract_violated"
    SANDBOX_ASSIGNED = "sandbox_assigned"
    SANDBOX_EXECUTED = "sandbox_executed"
    META_EPISODE = "meta_episode"
    ACTOR_CRASHED = "actor_crashed"
    ACTOR_RECOVERED = "actor_recovered"
    CHECKPOINT = "checkpoint"


@dataclass(frozen=True)
class DomainEvent:
    """Immutable domain event for event sourcing."""
    event_id: UUID = field(default_factory=uuid4)
    event_type: EventType = EventType.MESSAGE_RECEIVED
    actor_id: AgentId = field(default_factory=lambda: AgentId("kernel"))
    payload: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)
    vector_clock: Dict[str, int] = field(default_factory=dict)  # For distributed causality


# ---------------------------------------------------------------------------
# Protocols
# ---------------------------------------------------------------------------

class ToolCapability(Protocol):
    """MCP-compatible tool interface."""
    name: str
    description: str
    input_schema: Dict[str, Any]
    
    async def invoke(self, params: Dict[str, Any]) -> Dict[str, Any]: ...


class AgentCapability(Protocol):
    """A2A-compatible agent interface."""
    agent_card: Dict[str, Any]
    
    async def handle_task(self, task: Dict[str, Any]) -> Dict[str, Any]: ...
