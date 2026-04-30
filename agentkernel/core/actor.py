"""AgentKernel — Actor Model runtime kernel.

This module implements a complete Actor Model runtime using asyncio:

- **Mailbox**: Asynchronous FIFO message queue per actor.
- **Actor**: Base class with lifecycle, mailbox, and supervised execution.
- **ActorSystem**: Runtime manager that spawns actors, routes messages,
  and enforces supervision strategies.

All public APIs are fully typed and use async/await for concurrency.
"""

from __future__ import annotations

import asyncio
import enum
import logging
import time
import traceback
from typing import Any, Callable, Dict, List, Optional, Set, Type

from .types import (
    AgentId,
    ActorState,
    ActorStatus,
    Message,
    MessageKind,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Supervisor Strategy
# ---------------------------------------------------------------------------


class SupervisorStrategy(enum.Enum):
    """Strategy for handling child-actor failures.

    - ``RESTART``:  Attempt to restart the crashed actor (with rate limiting).
    - ``TERMINATE``: Shut down the crashed actor permanently.
    - ``ESCALATE``:  Notify the parent actor via a SYSTEM message, then terminate.
    """

    RESTART = "restart"
    TERMINATE = "terminate"
    ESCALATE = "escalate"


# ---------------------------------------------------------------------------
# Mailbox
# ---------------------------------------------------------------------------


class Mailbox:
    """Asynchronous message queue for a single actor.

    Each actor owns exactly one mailbox.  Messages are delivered in FIFO
    order.  The mailbox supports optional timeouts on dequeue operations
    and can be bounded (``max_size > 0``) to apply back-pressure.
    """

    __slots__ = ("_queue",)

    def __init__(self, max_size: int = 0) -> None:
        """Create a new mailbox.

        Args:
            max_size: Maximum number of messages the mailbox can hold.
                ``0`` means unbounded.  When full, ``put`` blocks until
                space is available.
        """
        self._queue: asyncio.Queue[Message] = asyncio.Queue(maxsize=max_size)

    async def put(self, msg: Message) -> None:
        """Enqueue a message.

        Blocks if the mailbox is full (when ``max_size > 0``).

        Args:
            msg: The message to enqueue.
        """
        await self._queue.put(msg)

    async def get(self, timeout: Optional[float] = None) -> Message:
        """Dequeue a message.

        Args:
            timeout: If provided, the maximum seconds to wait for a message.
                Raises ``asyncio.TimeoutError`` if no message arrives in time.

        Returns:
            The next message in FIFO order.
        """
        if timeout is not None:
            return await asyncio.wait_for(self._queue.get(), timeout=timeout)
        return await self._queue.get()

    def size(self) -> int:
        """Return the current number of messages in the mailbox."""
        return self._queue.qsize()

    def is_empty(self) -> bool:
        """Return ``True`` if the mailbox contains no messages."""
        return self._queue.empty()


# ---------------------------------------------------------------------------
# Actor
# ---------------------------------------------------------------------------


class Actor:
    """Base class for all actors in the AgentKernel runtime.

    Each actor is an independent unit of computation with:

    - A globally unique :class:`AgentId`.
    - A :class:`Mailbox` for receiving messages.
    - An :class:`ActorState` tracking its lifecycle and statistics.
    - A reference to the :class:`ActorSystem` so it can send messages to peers.

    Subclasses **must** override :meth:`receive` to define message-handling
    behaviour.  The actor runs inside its own ``asyncio.Task`` created by the
    :class:`ActorSystem`.

    **Supervision**

    When a child actor crashes, the :class:`ActorSystem` applies the
    ``supervision_strategy`` registered at spawn time.  If the strategy is
    ``ESCALATE``, the parent actor receives a ``SYSTEM`` message describing
    the failure.
    """

    __slots__ = (
        "agent_id",
        "_actor_system",
        "_parent",
        "_mailbox",
        "_state",
        "_running",
        "_shutdown_requested",
        "_restart_count",
        "_max_restarts",
        "_restart_window",
        "_restart_timestamps",
    )

    def __init__(
        self,
        agent_id: AgentId,
        actor_system: ActorSystem,
        parent: Optional[AgentId] = None,
        max_mailbox_size: int = 0,
    ) -> None:
        """Initialise an actor.

        Args:
            agent_id: The unique identifier for this actor.
            actor_system: The runtime system that manages this actor.
            parent: Optional parent actor in the supervision hierarchy.
            max_mailbox_size: Bound for the mailbox (``0`` = unbounded).
        """
        self.agent_id: AgentId = agent_id
        self._actor_system: ActorSystem = actor_system
        self._parent: Optional[AgentId] = parent
        self._mailbox: Mailbox = Mailbox(max_size=max_mailbox_size)
        self._state: ActorState = ActorState(agent_id=agent_id)
        self._running: bool = False
        self._shutdown_requested: bool = False
        self._restart_count: int = 0
        self._max_restarts: int = 10
        self._restart_window: float = 60.0
        self._restart_timestamps: List[float] = []

    @property
    def state(self) -> ActorState:
        """Read-only access to the actor's current state snapshot."""
        self._state.mailbox_size = self._mailbox.size()
        return self._state

    @property
    def mailbox(self) -> Mailbox:
        """Access the actor's mailbox (used mainly for testing / inspection)."""
        return self._mailbox

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def receive(self, msg: Message) -> Optional[Message]:
        """Handle an incoming message.

        **Must be overridden by concrete subclasses.**

        Args:
            msg: The message delivered from the mailbox.

        Returns:
            An optional response :class:`Message`.  If a response is returned
            and the original message had a ``sender``, the system will route the
            response back to that sender with ``kind=RESULT`` and the original
            ``message.id`` as ``correlation_id``.
        """
        raise NotImplementedError(
            f"Actor {self.agent_id} must implement receive()"
        )

    def send(self, target: AgentId, msg: Message) -> None:
        """Send a message to another actor via the :class:`ActorSystem`.

        The ``sender`` field of the message is automatically set to this
        actor's ``agent_id``.

        Args:
            target: The recipient actor.
            msg: The message payload.  The ``sender`` and ``recipient``
                fields are rewritten by this call.
        """
        updated = Message(
            id=msg.id,
            kind=msg.kind,
            sender=self.agent_id,
            recipient=target,
            payload=msg.payload,
            timestamp=msg.timestamp,
            correlation_id=msg.correlation_id,
            contract_id=msg.contract_id,
        )
        self._actor_system.send(target, updated)

    async def stop(self) -> None:
        """Request graceful shutdown of this actor.

        Sets the internal ``_running`` flag to ``False``.  The actor will
        finish processing its current message (if any) and then exit its
        event loop.
        """
        self._running = False
        self._shutdown_requested = True

    # ------------------------------------------------------------------
    # Internal machinery
    # ------------------------------------------------------------------

    async def _run(self) -> None:
        """Main event loop — invoked by the :class:`ActorSystem` task wrapper.

        The loop repeatedly:

        1. Waits for a message from the mailbox (with a short timeout so that
           ``_running`` is checked periodically).
        2. Updates state to ``PROCESSING``.
        3. Delegates to :meth:`receive`.
        4. If ``receive`` returns a message, routes it back to the original
           sender as a ``RESULT``.
        5. Returns state to ``IDLE``.

        Any exception raised by :meth:`receive` bubbles up to the task
        wrapper, which applies the supervision strategy.
        """
        self._running = True
        self._state.status = ActorStatus.IDLE

        while self._running:
            try:
                # Short timeout allows periodic checks of _running while idle.
                msg = await self._mailbox.get(timeout=0.5)
            except asyncio.TimeoutError:
                continue

            self._state.status = ActorStatus.PROCESSING
            self._state.mailbox_size = self._mailbox.size()
            self._state.processed_messages += 1

            try:
                result = await self.receive(msg)

                # Auto-route a non-None result back to the original sender.
                if result is not None and msg.sender is not None:
                    result_msg = Message(
                        kind=MessageKind.RESULT,
                        sender=self.agent_id,
                        recipient=msg.sender,
                        payload=(
                            result.payload
                            if isinstance(result, Message)
                            else {"result": result}
                        ),
                        correlation_id=msg.id,
                    )
                    self._actor_system.send(msg.sender, result_msg)

            except asyncio.CancelledError:
                self._state.status = ActorStatus.TERMINATED
                raise
            except Exception:
                logger.exception(
                    "Actor %s crashed processing message %s",
                    self.agent_id,
                    msg.id,
                )
                self._state.status = ActorStatus.CRASHED
                raise  # Re-raise so the system wrapper can apply supervision.
            finally:
                if self._running and self._state.status != ActorStatus.CRASHED:
                    self._state.status = ActorStatus.IDLE
                    self._state.mailbox_size = self._mailbox.size()

        # Loop exited because _running became False.
        self._state.status = ActorStatus.TERMINATED

    def _should_restart(self) -> bool:
        """Check whether a restart is permitted under rate-limiting rules.

        Returns:
            ``True`` if the number of restarts within the sliding window is
            below ``_max_restarts``.
        """
        now = time.monotonic()
        cutoff = now - self._restart_window
        self._restart_timestamps = [t for t in self._restart_timestamps if t > cutoff]
        return len(self._restart_timestamps) < self._max_restarts


# ---------------------------------------------------------------------------
# ActorSystem
# ---------------------------------------------------------------------------


class ActorSystem:
    """Runtime container that manages the lifecycle of all actors.

    Responsibilities
    ----------------
    - **Spawning**: Instantiate and start actor tasks.
    - **Routing**: Deliver messages to the correct actor mailboxes.
    - **Supervision**: Apply ``SupervisorStrategy`` when an actor crashes.
    - **Hierarchy**: Maintain parent–child relationships for escalation.
    - **Observation**: Fire lifecycle hooks (``on_actor_spawned``,
      ``on_actor_terminated``, ``on_message_sent``) so that external
      systems (e.g. an Event Store) can subscribe.

    Thread-safety
    -------------
    An ``asyncio.Lock`` protects cross-await critical sections.  Care is
    taken never to ``await`` a running actor task while holding the lock,
    which prevents deadlocks between :meth:`terminate` / :meth:`shutdown`
    and the internal supervision cleanup path.
    """

    __slots__ = (
        "_actors",
        "_tasks",
        "_parent_map",
        "_children_map",
        "_supervision_policies",
        "_lock",
        "_shutdown",
        "_shutdown_event",
        "on_actor_spawned",
        "on_actor_terminated",
        "on_message_sent",
    )

    def __init__(self) -> None:
        """Create a new, empty actor system."""
        self._actors: Dict[AgentId, Actor] = {}
        self._tasks: Dict[AgentId, asyncio.Task[None]] = {}
        self._parent_map: Dict[AgentId, AgentId] = {}          # child -> parent
        self._children_map: Dict[AgentId, Set[AgentId]] = {}   # parent -> {children}
        self._supervision_policies: Dict[AgentId, SupervisorStrategy] = {}
        self._lock = asyncio.Lock()
        self._shutdown = False
        self._shutdown_event = asyncio.Event()

        # Event hooks — lists of callables that observers can append to.
        self.on_actor_spawned: List[Callable[[AgentId, Optional[AgentId]], None]] = []
        self.on_actor_terminated: List[Callable[[AgentId], None]] = []
        self.on_message_sent: List[Callable[[Message], None]] = []

    # ------------------------------------------------------------------
    # Actor lifecycle
    # ------------------------------------------------------------------

    async def spawn(
        self,
        actor_class: Type[Actor],
        agent_id: Optional[AgentId] = None,
        parent: Optional[AgentId] = None,
        supervision_strategy: SupervisorStrategy = SupervisorStrategy.RESTART,
        **kwargs: Any,
    ) -> AgentId:
        """Create and start a new actor.

        Args:
            actor_class: Concrete subclass of :class:`Actor` to instantiate.
            agent_id: Explicit identifier.  If ``None``, a default is
                generated (``local="actor"``).
            parent: Optional parent in the supervision tree.
            supervision_strategy: Strategy applied when this actor crashes.
            **kwargs: Extra keyword arguments forwarded to the actor
                constructor.

        Returns:
            The :class:`AgentId` of the newly spawned actor.

        Raises:
            RuntimeError: If the system is shutting down.
            ValueError: If an actor with the given ``agent_id`` already exists.
        """
        if self._shutdown:
            raise RuntimeError("ActorSystem is shutting down — cannot spawn new actors")

        if agent_id is None:
            agent_id = AgentId("actor")

        async with self._lock:
            if agent_id in self._actors:
                raise ValueError(f"Actor with id {agent_id} already exists")

            actor = actor_class(
                agent_id=agent_id,
                actor_system=self,
                parent=parent,
                **kwargs,
            )
            self._actors[agent_id] = actor
            if parent is not None:
                self._parent_map[agent_id] = parent
                self._children_map.setdefault(parent, set()).add(agent_id)
            self._supervision_policies[agent_id] = supervision_strategy

            task = asyncio.create_task(self._run_actor(actor))
            self._tasks[agent_id] = task

        # Fire hooks *outside* the lock so observers cannot deadlock us.
        for hook in self.on_actor_spawned:
            try:
                hook(agent_id, parent)
            except Exception:
                logger.exception("on_actor_spawned hook failed for %s", agent_id)

        return agent_id

    async def terminate(self, actor_id: AgentId) -> None:
        """Gracefully terminate an actor and all its descendants.

        Steps:

        1. Signal the actor to stop (sets ``_running = False``).
        2. Cancel its task and await cleanup.
        3. Remove the actor (and children) from all internal registries.
        4. Fire ``on_actor_terminated`` hooks.

        Args:
            actor_id: The actor to terminate.
        """
        # Phase 1 — grab references under the lock but do NOT await here.
        async with self._lock:
            actor = self._actors.get(actor_id)
            task = self._tasks.get(actor_id)

        if actor is None:
            return

        # Phase 2 — signal stop and wait for the task to finish *outside* the
        # lock so the actor's own _run_actor wrapper can acquire the lock
        # for its final cleanup if necessary.
        await actor.stop()

        if task is not None and not task.done():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

        # Phase 3 — idempotent registry cleanup.
        await self._cleanup_actor(actor_id)

    # ------------------------------------------------------------------
    # Messaging
    # ------------------------------------------------------------------

    def send(self, target: AgentId, msg: Message) -> None:
        """Send a message to an actor's mailbox.

        Delivery is best-effort and asynchronous.  If the target actor does
        not exist, a warning is logged and the message is dropped.

        Args:
            target: The recipient :class:`AgentId`.
            msg: The message to deliver.
        """
        if self._shutdown:
            return

        actor = self._actors.get(target)
        if actor is None:
            logger.warning("Message sent to non-existent actor %s", target)
            return

        # Schedule the put so that a full mailbox does not block the caller.
        asyncio.create_task(actor._mailbox.put(msg))

        # Notify observers.
        for hook in self.on_message_sent:
            try:
                hook(msg)
            except Exception:
                logger.exception("on_message_sent hook failed")

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def get_state(self, actor_id: AgentId) -> ActorState:
        """Return a snapshot of an actor's current state.

        Args:
            actor_id: The actor to inspect.

        Returns:
            A copy of the actor's :class:`ActorState`.

        Raises:
            KeyError: If no actor with the given ID exists.
        """
        actor = self._actors.get(actor_id)
        if actor is None:
            raise KeyError(f"Actor {actor_id} not found")
        return actor.state

    def list_actors(self) -> List[AgentId]:
        """List all currently active actor IDs."""
        return list(self._actors.keys())

    def get_children(self, parent_id: AgentId) -> Set[AgentId]:
        """Return the set of children for a given parent actor."""
        return self._children_map.get(parent_id, set()).copy()

    def get_parent(self, child_id: AgentId) -> Optional[AgentId]:
        """Return the parent of a given child actor, if any."""
        return self._parent_map.get(child_id)

    def get_supervision_strategy(self, actor_id: AgentId) -> SupervisorStrategy:
        """Return the supervision strategy registered for an actor."""
        return self._supervision_policies.get(
            actor_id, SupervisorStrategy.RESTART
        )

    # ------------------------------------------------------------------
    # Shutdown
    # ------------------------------------------------------------------

    async def shutdown(self, timeout: float = 30.0) -> None:
        """Gracefully shut down the entire actor system.

        Steps:

        1. Sets the shutdown flag so no new actors can be spawned.
        2. Signals every actor to stop.
        3. Cancels all running tasks.
        4. Waits up to *timeout* seconds for tasks to finish.
        5. Clears all internal state.

        Args:
            timeout: Maximum seconds to wait for graceful termination.
        """
        self._shutdown = True
        self._shutdown_event.set()

        # Take a snapshot under the lock, then release it before awaiting.
        async with self._lock:
            actors_snapshot = list(self._actors.values())
            pending = [t for t in self._tasks.values() if not t.done()]

        # Signal every actor to stop.
        for actor in actors_snapshot:
            await actor.stop()

        # Cancel all outstanding tasks.
        for task in pending:
            task.cancel()

        if pending:
            try:
                await asyncio.wait_for(
                    asyncio.gather(*pending, return_exceptions=True),
                    timeout=timeout,
                )
            except asyncio.TimeoutError:
                logger.warning(
                    "Shutdown timed out after %ss; forcing cancellation", timeout
                )
                for task in pending:
                    if not task.done():
                        task.cancel()

        # Final registry clearance.
        async with self._lock:
            self._actors.clear()
            self._tasks.clear()
            self._parent_map.clear()
            self._children_map.clear()
            self._supervision_policies.clear()

    # ------------------------------------------------------------------
    # Internal — supervision wrapper
    # ------------------------------------------------------------------

    async def _run_actor(self, actor: Actor) -> None:
        """Wrap an actor's :meth:`_run` loop with supervision logic.

        If the actor raises an exception, this wrapper inspects the
        registered supervision strategy and acts accordingly:

        - ``RESTART``:  Reset the actor state and re-enter ``_run``
          (subject to rate limiting).
        - ``TERMINATE``: Remove the actor from the system.
        - ``ESCALATE``: Send a ``SYSTEM`` message to the parent, then
          remove the actor.
        """
        actor_id = actor.agent_id

        while True:
            try:
                await actor._run()
                break  # Normal exit (actor stopped gracefully).
            except asyncio.CancelledError:
                logger.debug("Actor %s cancelled", actor_id)
                break
            except Exception as exc:
                logger.exception("Actor %s crashed: %s", actor_id, exc)

                strategy = self._supervision_policies.get(
                    actor_id, SupervisorStrategy.RESTART
                )

                if strategy == SupervisorStrategy.RESTART:
                    if actor._should_restart():
                        actor._restart_timestamps.append(time.monotonic())
                        actor._restart_count += 1
                        actor._state.status = ActorStatus.SPAWNING
                        logger.info(
                            "Restarting actor %s (restart #%d)",
                            actor_id,
                            actor._restart_count,
                        )
                        # Allow the next loop iteration to re-enter _run().
                        actor._running = True
                        await asyncio.sleep(0.1)  # Brief backoff
                        actor._state.status = ActorStatus.IDLE
                        continue
                    else:
                        logger.warning(
                            "Actor %s exceeded maximum restarts; terminating", actor_id
                        )
                        await self._cleanup_actor(actor_id)
                        break

                elif strategy == SupervisorStrategy.TERMINATE:
                    await self._cleanup_actor(actor_id)
                    break

                elif strategy == SupervisorStrategy.ESCALATE:
                    parent_id = self._parent_map.get(actor_id)
                    if parent_id is not None:
                        err_msg = Message(
                            kind=MessageKind.SYSTEM,
                            sender=actor_id,
                            recipient=parent_id,
                            payload={
                                "event": "child_crashed",
                                "child_id": actor_id.full,
                                "error": str(exc),
                                "traceback": traceback.format_exc(),
                            },
                        )
                        self.send(parent_id, err_msg)
                    await self._cleanup_actor(actor_id)
                    break

        # Ensure cleanup happens even if the loop broke unexpectedly.
        if actor_id in self._actors:
            await self._cleanup_actor(actor_id)

    async def _cleanup_actor(self, actor_id: AgentId) -> None:
        """Idempotent cleanup of an actor after crash or stop.

        Removes the actor (and any remaining children) from all internal
        registries and fires ``on_actor_terminated`` hooks.  Children tasks
        are cancelled but not awaited here — their own wrappers will exit
        independently.
        """
        async with self._lock:
            actor = self._actors.get(actor_id)
            if actor is None:
                # Already cleaned up by a prior call (e.g. from _run_actor).
                return

            actor._running = False

            # Sever parent link.
            parent = self._parent_map.pop(actor_id, None)
            if parent is not None:
                self._children_map.get(parent, set()).discard(actor_id)

            # Cancel children and wipe their registry entries.
            children = self._children_map.pop(actor_id, set()).copy()
            for child_id in children:
                child_task = self._tasks.get(child_id)
                if child_task is not None and not child_task.done():
                    child_task.cancel()
                self._parent_map.pop(child_id, None)
                self._actors.pop(child_id, None)
                self._tasks.pop(child_id, None)
                self._supervision_policies.pop(child_id, None)

            # Wipe the actor itself.
            self._actors.pop(actor_id, None)
            self._tasks.pop(actor_id, None)
            self._supervision_policies.pop(actor_id, None)

        # Fire hooks outside the lock.
        for hook in self.on_actor_terminated:
            try:
                hook(actor_id)
            except Exception:
                logger.exception("on_actor_terminated hook failed for %s", actor_id)
