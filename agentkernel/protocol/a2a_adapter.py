"""AgentKernel — A2A (Agent-to-Agent Protocol) native adapter.

A2A is treated as a first-class citizen for peer-to-peer agent collaboration.
This module implements Agent Card hosting, task delegation, and artifact
exchange with A2A semantics mapped onto Actor messages.
"""
from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional
from uuid import UUID, uuid4

import agentkernel.types as _kt

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Agent Card
# ---------------------------------------------------------------------------

@dataclass
class AgentCard:
    """A2A Agent Card describing an agent's capabilities."""
    agent_id: _kt.AgentId
    name: str
    description: str
    version: str = "1.0"
    capabilities: List[str] = field(default_factory=list)
    skills: List[str] = field(default_factory=list)
    endpoint: Optional[str] = None  # URL for remote A2A communication
    authentication: Dict[str, Any] = field(default_factory=dict)
    input_modalities: List[str] = field(default_factory=lambda: ["text"])
    output_modalities: List[str] = field(default_factory=lambda: ["text"])

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "version": self.version,
            "capabilities": self.capabilities,
            "skills": self.skills,
            "endpoint": self.endpoint,
            "authentication": self.authentication,
            "inputModalities": self.input_modalities,
            "outputModalities": self.output_modalities,
            "agentId": str(self.agent_id),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> AgentCard:
        return cls(
            agent_id=_kt.AgentId(data.get("agentId", "unknown")),
            name=data["name"],
            description=data["description"],
            version=data.get("version", "1.0"),
            capabilities=data.get("capabilities", []),
            skills=data.get("skills", []),
            endpoint=data.get("endpoint"),
            authentication=data.get("authentication", {}),
            input_modalities=data.get("inputModalities", ["text"]),
            output_modalities=data.get("outputModalities", ["text"]),
        )


# ---------------------------------------------------------------------------
# A2A Task Lifecycle
# ---------------------------------------------------------------------------

class A2ATaskState:
    """A2A task states per Google A2A spec."""
    SUBMITTED = "submitted"
    WORKING = "working"
    INPUT_REQUIRED = "input_required"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELED = "canceled"


@dataclass
class A2ATask:
    """Internal representation of an A2A task."""
    task_id: UUID = field(default_factory=uuid4)
    state: str = A2ATaskState.SUBMITTED
    creator: Optional[_kt.AgentId] = None
    handler: Optional[_kt.AgentId] = None
    message: _kt.Message = field(default_factory=_kt.Message)
    artifacts: List[Dict[str, Any]] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: str(_kt.datetime.utcnow()))
    updated_at: str = field(default_factory=lambda: str(_kt.datetime.utcnow()))


# ---------------------------------------------------------------------------
# A2A Runtime
# ---------------------------------------------------------------------------

class A2ARuntime:
    """A2A runtime integrated with AgentKernel ActorSystem.

    Each A2A-capable agent hosts an Agent Card and can:
    - receive task delegations from other agents
    - send tasks to remote agents (via endpoint or in-memory)
    - manage task lifecycle state machine
    """

    def __init__(self) -> None:
        self._cards: Dict[str, AgentCard] = {}  # agent_id.full -> card
        self._tasks: Dict[UUID, A2ATask] = {}
        self._handlers: Dict[str, Callable[[A2ATask], Any]] = {}  # skill -> handler
        self._lock: asyncio.Lock = asyncio.Lock()

    async def register_card(self, card: AgentCard) -> None:
        async with self._lock:
            self._cards[card.agent_id.full] = card
        logger.info("A2A Agent Card registered: %s", card.name)

    async def get_card(self, agent_id: _kt.AgentId) -> Optional[AgentCard]:
        async with self._lock:
            return self._cards.get(agent_id.full)

    async def create_task(
        self,
        creator: _kt.AgentId,
        handler: _kt.AgentId,
        message: _kt.Message,
    ) -> A2ATask:
        """Create a new A2A task from an Actor message."""
        task = A2ATask(
            creator=creator,
            handler=handler,
            message=message,
            state=A2ATaskState.SUBMITTED,
        )
        async with self._lock:
            self._tasks[task.task_id] = task
        logger.info("A2A task %s created: %s -> %s", task.task_id, creator.full, handler.full)
        return task

    async def update_task_state(self, task_id: UUID, new_state: str, artifacts: Optional[List[Dict]] = None) -> A2ATask:
        async with self._lock:
            task = self._tasks.get(task_id)
            if task is None:
                raise ValueError(f"Task {task_id} not found")
            task.state = new_state
            task.updated_at = str(_kt.datetime.utcnow())
            if artifacts:
                task.artifacts.extend(artifacts)
        logger.info("A2A task %s state -> %s", task_id, new_state)
        return task

    async def delegate_to_agent(
        self,
        caller: _kt.AgentId,
        target: _kt.AgentId,
        payload: Dict[str, Any],
        contract_id: Optional[UUID] = None,
    ) -> _kt.Message:
        """Delegate work to another agent via A2A semantics.

        Returns a RESULT message when the task completes (async).
        """
        # Create A2A task record
        msg = _kt.Message(
            kind=_kt.MessageKind.TASK,
            sender=caller,
            recipient=target,
            payload=payload,
            contract_id=contract_id,
        )
        task = await self.create_task(caller, target, msg)

        # In a full implementation, this would serialize to A2A JSON-RPC
        # and send over HTTP to the target endpoint. Here we map it to
        # an in-memory Actor message.
        logger.debug("A2A delegation mapped to Actor message: %s", task.task_id)

        # Return a placeholder result message
        return _kt.Message(
            kind=_kt.MessageKind.RESULT,
            sender=target,
            recipient=caller,
            payload={"status": "accepted", "task_id": str(task.task_id)},
            correlation_id=msg.id,
            contract_id=contract_id,
        )

    def register_skill_handler(self, skill: str, handler: Callable[[A2ATask], Any]) -> None:
        """Register a handler for a specific skill/capability."""
        self._handlers[skill] = handler

    async def route_task(self, task: A2ATask) -> None:
        """Route an incoming A2A task to the appropriate handler."""
        card = await self.get_card(task.handler)
        if card is None:
            await self.update_task_state(task.task_id, A2ATaskState.FAILED)
            return

        for skill in card.skills:
            handler = self._handlers.get(skill)
            if handler:
                try:
                    result = handler(task)
                    if asyncio.iscoroutine(result):
                        result = await result
                    await self.update_task_state(
                        task.task_id,
                        A2ATaskState.COMPLETED,
                        artifacts=[{"type": "text", "text": str(result)}],
                    )
                    return
                except Exception as exc:
                    logger.exception("A2A task handler failed for skill %s", skill)
                    await self.update_task_state(task.task_id, A2ATaskState.FAILED)
                    return

        await self.update_task_state(task.task_id, A2ATaskState.FAILED)
