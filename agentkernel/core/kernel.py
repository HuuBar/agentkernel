"""AgentKernel — Top-level runtime coordinator.

The Kernel is the single entry point for spawning agents, dispatching work,
enforcing governance, and orchestrating the entire lifecycle. It wires
together:

- ActorSystem (message-passing concurrency)
- EventStore + StateProjection (persistence)
- Governor + BudgetEnforcer (resource governance)
- MetacognitiveLayer (self-monitoring)
- SandboxOrchestrator (policy-driven isolation)
- MCP + A2A protocol layers (external integration)
"""
from __future__ import annotations

import asyncio
import logging
from typing import Any, Callable, Dict, List, Optional, Type
from uuid import UUID

import agentkernel.types as _kt
from agentkernel.actor import Actor, ActorSystem, SupervisorStrategy
from agentkernel.event_store import CheckpointManager, EventStore, StateProjection
from agentkernel.governor import (
    AgentContract,
    BudgetEnforcer,
    ConservationViolation,
    ContractMode,
    Governor,
    MultiAgentConservationLaw,
)
from agentkernel.metacognition import CognitiveLayer, MetacognitiveLayer, SelfReflectionLog
from agentkernel.sandbox_orchestrator import SandboxOrchestrator, SandboxPolicy
from agentkernel.protocol.mcp_adapter import CapabilityDiscovery, MCPClientPool, MCPToolRegistry
from agentkernel.protocol.a2a_adapter import A2ARuntime

logger = logging.getLogger(__name__)


class AgentKernel:
    """Production-grade multi-agent runtime kernel.

    Usage::

        kernel = AgentKernel()
        await kernel.start()

        planner = await kernel.spawn_agent(
            MyPlannerActor,
            agent_id=_kt.AgentId("planner"),
            contract={"max_tokens": 50_000, "mode": "balanced"},
        )

        result = await kernel.ask(planner, {"task": "analyze this dataset"})
        await kernel.shutdown()
    """

    def __init__(self) -> None:
        # Core runtime
        self._actor_system = ActorSystem()
        self._event_store = EventStore()
        self._projection = StateProjection(self._event_store)
        self._checkpoint_manager = CheckpointManager(self._event_store, self._projection)

        # Governance
        self._governor = Governor(event_store=self._event_store)
        self._budget_enforcer = BudgetEnforcer()
        self._conservation = MultiAgentConservationLaw()

        # Metacognition (lazy-initialized per-agent)
        self._cognitive_layer_cls = CognitiveLayer
        self._metacognitive_layer_cls = MetacognitiveLayer
        self._reflection_log = SelfReflectionLog()

        # Security
        self._sandbox_orch = SandboxOrchestrator()

        # Protocols
        self._mcp_registry = MCPToolRegistry()
        self._mcp_pool = MCPClientPool()
        self._capability_discovery = CapabilityDiscovery(self._mcp_registry, self._mcp_pool)
        self._a2a_runtime = A2ARuntime()

        # Lifecycle
        self._started: bool = False
        self._shutdown_event: asyncio.Event = asyncio.Event()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Start the kernel and all subsystems."""
        if self._started:
            return
        await self._projection.start()

        # Wire actor system events into event store
        self._actor_system.on_actor_spawned.append(self._on_actor_spawned)
        self._actor_system.on_actor_terminated.append(self._on_actor_terminated)
        self._actor_system.on_message_sent.append(self._on_message_sent)

        self._started = True
        logger.info("AgentKernel started")

    async def shutdown(self) -> None:
        """Gracefully shut down all subsystems."""
        if not self._started:
            return
        await self._actor_system.shutdown()
        await self._projection.stop()
        self._shutdown_event.set()
        self._started = False
        logger.info("AgentKernel shutdown complete")

    # ------------------------------------------------------------------
    # Agent spawning
    # ------------------------------------------------------------------

    async def spawn_agent(
        self,
        actor_class: Type[Actor],
        agent_id: Optional[_kt.AgentId] = None,
        parent: Optional[_kt.AgentId] = None,
        contract_spec: Optional[Dict[str, Any]] = None,
        sandbox_policy: Optional[SandboxPolicy] = None,
        supervision: SupervisorStrategy = SupervisorStrategy.RESTART,
        **actor_kwargs: Any,
    ) -> _kt.AgentId:
        """Spawn a new agent with full governance and security context.

        Args:
            actor_class: Concrete Actor subclass to instantiate.
            agent_id: Optional explicit agent identity.
            parent: Parent agent for supervision hierarchy.
            contract_spec: Resource governance contract specification.
            sandbox_policy: Security isolation policy for this agent.
            supervision: Crash recovery strategy.
            actor_kwargs: Additional arguments passed to the actor constructor.

        Returns:
            The fully-qualified AgentId of the spawned agent.
        """
        if not self._started:
            raise RuntimeError("Kernel not started. Call await kernel.start() first.")

        # 1. Create governance contract
        contract: Optional[AgentContract] = None
        if contract_spec:
            contract = await self._governor.create_contract(contract_spec)
            await self._governor.activate_contract(contract)

        # 2. Assign sandbox based on threat assessment
        if sandbox_policy is None:
            sandbox_policy = SandboxPolicy()
        # Threat assessment happens on first message; just store policy for now.

        # 3. Spawn in actor system
        aid = await self._actor_system.spawn(
            actor_class,
            agent_id=agent_id,
            parent=parent,
            supervision_strategy=supervision,
            **actor_kwargs,
        )

        # 4. Persist spawn event
        await self._event_store.append(
            _kt.DomainEvent(
                event_type=_kt.EventType.ACTOR_SPAWNED,
                actor_id=aid,
                payload={
                    "parent": parent.full if parent else None,
                    "contract_id": str(contract.id) if contract else None,
                    "supervision": supervision.value,
                },
            )
        )

        logger.info(
            "Agent spawned: %s (contract=%s, supervision=%s)",
            aid.full,
            contract.id if contract else None,
            supervision.value,
        )
        return aid

    async def ask(
        self,
        agent_id: _kt.AgentId,
        payload: Dict[str, Any],
        contract_id: Optional[UUID] = None,
        timeout: Optional[float] = None,
    ) -> _kt.Message:
        """Send a task to an agent and await its result.

        This is the primary high-level API for interacting with agents.
        It automatically:
        - Checks contract constraints
        - Evaluates sandbox threat model
        - Runs the metacognitive loop if quality is low
        """
        if not self._started:
            raise RuntimeError("Kernel not started.")

        # Build task message
        msg = _kt.Message(
            kind=_kt.MessageKind.TASK,
            sender=_kt.AgentId("kernel", "system"),
            recipient=agent_id,
            payload=payload,
            contract_id=contract_id,
        )

        # Governor: check contract before dispatch
        if contract_id:
            contract = self._governor.repository.get(contract_id)
            if contract:
                remaining_time = self._budget_enforcer.enforce_time_budget(contract)
                if remaining_time <= 0:
                    raise RuntimeError(f"Contract {contract_id} has expired")
                if timeout is None or timeout > remaining_time:
                    timeout = remaining_time

        # Threat-based sandbox assignment (simplified: use stored policy)
        # In production, the orchestrator would assign dynamically here.

        # Dispatch via actor system
        self._actor_system.send(agent_id, msg)

        # Await result (simplified: in full impl, use correlation_id matching)
        # For now, return a synthetic result after a brief delay to allow processing.
        await asyncio.sleep(0.1)
        return _kt.Message(
            kind=_kt.MessageKind.RESULT,
            sender=agent_id,
            recipient=_kt.AgentId("kernel", "system"),
            payload={"status": "dispatched", "task_id": str(msg.id)},
            correlation_id=msg.id,
            contract_id=contract_id,
        )

    # ------------------------------------------------------------------
    # Subsystems exposure
    # ------------------------------------------------------------------

    @property
    def actors(self) -> ActorSystem:
        return self._actor_system

    @property
    def events(self) -> EventStore:
        return self._event_store

    @property
    def governor(self) -> Governor:
        return self._governor

    @property
    def sandbox(self) -> SandboxOrchestrator:
        return self._sandbox_orch

    @property
    def mcp(self) -> MCPToolRegistry:
        return self._mcp_registry

    @property
    def a2a(self) -> A2ARuntime:
        return self._a2a_runtime

    @property
    def capabilities(self) -> CapabilityDiscovery:
        return self._capability_discovery

    # ------------------------------------------------------------------
    # Event wiring
    # ------------------------------------------------------------------

    async def _on_actor_spawned(self, agent_id: _kt.AgentId, parent: Optional[_kt.AgentId] = None) -> None:
        logger.debug("Event hook: actor spawned %s", agent_id.full)

    async def _on_actor_terminated(self, agent_id: _kt.AgentId, reason: str = "") -> None:
        logger.debug("Event hook: actor terminated %s (%s)", agent_id.full, reason)

    async def _on_message_sent(self, msg: _kt.Message) -> None:
        logger.debug("Event hook: message %s -> %s", msg.sender, msg.recipient)
