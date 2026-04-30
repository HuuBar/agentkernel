"""AgentKernel — Event Sourcing + CQRS persistence layer.

This module implements the write-model (EventStore) and read-model
(StateProjection) for the AgentKernel actor system. It follows event-sourcing
principles: state is never updated in-place; every mutation is captured as an
append-only immutable ``DomainEvent``. Read-models are rebuilt by projecting the
event stream, enabling time-travel debugging and deterministic replay.

All public methods are ``async`` and thread-safe via ``asyncio.Lock``. The
implementation is in-memory but the interface is designed so that a
PostgreSQL/MongoDB-backed store can be swapped in without changing callers.

Example::

    store = EventStore()
    projection = StateProjection(store)
    await projection.start()

    event = DomainEvent(
        event_type=EventType.ACTOR_SPAWNED,
        actor_id=AgentId("planner"),
        payload={"capabilities": ["search", "code"]},
    )
    await store.append(event)

    state = await projection.get_actor_state(event.actor_id)
    assert state is not None
    assert state.status == ActorStatus.SPAWNING
"""
from __future__ import annotations

import asyncio
import logging
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Coroutine, Dict, List, Optional, Set, Tuple, Union
from uuid import UUID, uuid4

import agentkernel.types as _kernel_types

# Re-export types used in this module for local convenience.
ActorId = _kernel_types.AgentId
ActorState = _kernel_types.ActorState
ActorStatus = _kernel_types.ActorStatus
ContractStatus = _kernel_types.ContractStatus
DomainEvent = _kernel_types.DomainEvent
EventType = _kernel_types.EventType

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Vector-clock helpers
# ---------------------------------------------------------------------------

def _is_vector_clock_after(
    clock_a: Dict[str, int], clock_b: Dict[str, int]
) -> bool:
    """Causality comparison for vector clocks.

    Returns ``True`` iff *clock_a* is strictly after *clock_b* in the
    partial order (i.e. ``clock_a >= clock_b`` component-wise and
    ``clock_a != clock_b``).

    Args:
        clock_a: Candidate later clock.
        clock_b: Reference clock.

    Returns:
        Whether *clock_a* causally follows *clock_b*.
    """
    if not clock_b:
        # Nothing to compare against — any clock is "after" the empty clock.
        return True
    all_ge = all(clock_a.get(k, 0) >= v for k, v in clock_b.items())
    any_gt = any(clock_a.get(k, 0) > v for k, v in clock_b.items())
    return all_ge and any_gt


def _copy_vector_clock(clock: Dict[str, int]) -> Dict[str, int]:
    """Deep-copy a vector clock dictionary."""
    return {k: v for k, v in clock.items()}


# ---------------------------------------------------------------------------
# Callback normalisation
# ---------------------------------------------------------------------------

def _ensure_async_callback(
    callback: Callable[[DomainEvent], Any],
) -> Callable[[DomainEvent], Coroutine[Any, Any, None]]:
    """Wrap a sync callback so it can be ``await``-ed uniformly.

    If *callback* is already a coroutine function it is returned unchanged.
    """
    if asyncio.iscoroutinefunction(callback):
        # asyncio.iscoroutinefunction guarantees the callable returns a coroutine.
        return callback  # type: ignore[return-value]

    async def _wrapper(event: DomainEvent) -> None:
        callback(event)

    return _wrapper


# ---------------------------------------------------------------------------
# EventStore — Write model
# ---------------------------------------------------------------------------

class EventStore:
    """Immutable append-only event log with publisher capabilities.

    This is the **write-model** in the CQRS split. Every state mutation in the
    system is recorded as an ordered, immutable ``DomainEvent``. The store
    maintains a monotonic vector clock per actor so that distributed replays
    preserve causality.

    Thread-safety: ``append`` is serialised by an ``asyncio.Lock``. Reads are
    lock-free because the backing list is only ever appended to (and
    occasionally rebuilt during pruning). Indexes are kept consistent under the
    same lock.

    Attributes:
        _events: Global ordered list of all persisted events.
        _actor_index: Mapping ``actor_id.full -> [indices into _events]`` for
            fast per-actor queries.
        _vector_clocks: Per-actor current vector-clock horizon.
        _append_lock: Serialises appends and pruning.
        _subscribers: Registered callbacks keyed by subscription id.
    """

    def __init__(self) -> None:
        self._events: List[DomainEvent] = []
        self._actor_index: Dict[str, List[int]] = {}
        self._vector_clocks: Dict[str, Dict[str, int]] = {}
        self._append_lock: asyncio.Lock = asyncio.Lock()
        self._subscribers: Dict[
            str,
            Tuple[
                Optional[List[EventType]],
                Callable[[DomainEvent], Coroutine[Any, Any, None]],
            ],
        ] = {}
        self._sub_lock: asyncio.Lock = asyncio.Lock()

    # -- Public read API ---------------------------------------------------

    async def get_events(
        self,
        actor_id: AgentId,
        after_clock: Optional[Dict[str, int]] = None,
    ) -> List[DomainEvent]:
        """Return events for *actor_id*, optionally filtered by vector clock.

        Args:
            actor_id: The actor whose events are requested.
            after_clock: If provided, only events whose vector clock is
                strictly after this reference clock are returned. This enables
                incremental replay and gap detection.

        Returns:
            A **new** list containing the matching events in chronological
            order.
        """
        key = actor_id.full
        indices = self._actor_index.get(key, [])

        if after_clock is None:
            return [self._events[i] for i in indices]

        result: List[DomainEvent] = []
        for i in indices:
            event = self._events[i]
            if _is_vector_clock_after(event.vector_clock, after_clock):
                result.append(event)
        return result

    async def get_all_events(
        self,
        event_types: Optional[List[EventType]] = None,
    ) -> List[DomainEvent]:
        """Return all events, optionally filtered by type.

        Args:
            event_types: If provided, only events whose ``event_type`` is in
                this list are returned.

        Returns:
            A **new** list of matching events in chronological order.
        """
        if event_types is None:
            return list(self._events)

        allowed: Set[EventType] = set(event_types)
        return [e for e in self._events if e.event_type in allowed]

    def get_vector_clock(self, actor_id: AgentId) -> Dict[str, int]:
        """Return the current vector clock horizon for *actor_id*.

        This is useful when creating checkpoints or resuming a distributed
        replay from a known frontier.

        Args:
            actor_id: Actor whose clock is requested.

        Returns:
            A deep-copied dictionary ``{actor_full_id: clock_value, ...}``.
        """
        return _copy_vector_clock(self._vector_clocks.get(actor_id.full, {}))

    # -- Write API ---------------------------------------------------------

    async def append(self, event: DomainEvent) -> DomainEvent:
        """Atomically append an immutable event to the log.

        The event's vector clock is automatically incremented for the emitting
        actor so that causality is preserved without caller intervention.

        Args:
            event: The domain event to persist. Its ``vector_clock`` field may
                be empty; in that case the store derives the next logical clock
                from its internal ledger.

        Returns:
            The **actual** persisted event (a new ``DomainEvent`` instance with
            the updated vector clock and deep-copied payload).

        Raises:
            Exception: Subscriber notification failures are logged but never
                propagated, guaranteeing that the append itself is durable.
        """
        actor_key = event.actor_id.full

        async with self._append_lock:
            # Derive next vector clock for this actor.
            new_clock = _copy_vector_clock(
                self._vector_clocks.get(actor_key, {})
            )
            new_clock[actor_key] = new_clock.get(actor_key, 0) + 1

            # Rebuild an immutable event with the incremented clock.
            persisted = DomainEvent(
                event_id=event.event_id,
                event_type=event.event_type,
                actor_id=event.actor_id,
                payload=deepcopy(event.payload),
                timestamp=event.timestamp,
                vector_clock=_copy_vector_clock(new_clock),
            )

            self._events.append(persisted)
            idx = len(self._events) - 1
            self._actor_index.setdefault(actor_key, []).append(idx)
            self._vector_clocks[actor_key] = _copy_vector_clock(new_clock)

        # Notify outside the lock so slow subscribers cannot stall writers.
        try:
            await self._notify_subscribers(persisted)
        except Exception:
            logger.exception(
                "Subscriber notification failed for event %s",
                persisted.event_id,
            )

        return persisted

    async def prune_events(
        self,
        actor_id: AgentId,
        before_clock: Dict[str, int],
    ) -> int:
        """Remove events for *actor_id* that are at or before *before_clock*.

        This is the log-compaction primitive used by ``CheckpointManager``.
        Events that are causally after *before_clock* are retained so that
        replay from the checkpoint horizon is still possible.

        Args:
            actor_id: Actor whose events should be compacted.
            before_clock: Vector-clock frontier. Events ``<=`` this clock are
                removed.

        Returns:
            Number of events removed.
        """
        actor_key = actor_id.full

        async with self._append_lock:
            indices = self._actor_index.get(actor_key, [])
            if not indices:
                return 0

            # Determine which indices to keep.
            kept_indices: List[int] = []
            for i in indices:
                ev = self._events[i]
                if _is_vector_clock_after(ev.vector_clock, before_clock):
                    kept_indices.append(i)

            removed_count = len(indices) - len(kept_indices)
            if removed_count == 0:
                return 0

            removed_set = set(indices) - set(kept_indices)

            # Rebuild global structures compactly.
            new_events: List[DomainEvent] = []
            new_index: Dict[str, List[int]] = {}

            for i, ev in enumerate(self._events):
                if i in removed_set:
                    continue
                new_idx = len(new_events)
                new_events.append(ev)
                key = ev.actor_id.full
                new_index.setdefault(key, []).append(new_idx)

            self._events = new_events
            self._actor_index = new_index
            # Update the actor's vector clock to the maximum retained event clock
            # so that subsequent appends continue from the correct logical time.
            max_retained_clock = _copy_vector_clock(before_clock)
            for i in kept_indices:
                ev = new_events[i]
                for key, val in ev.vector_clock.items():
                    max_retained_clock[key] = max(max_retained_clock.get(key, 0), val)
            self._vector_clocks[actor_key] = max_retained_clock

            return removed_count

    # -- Publisher / Subscriber pattern --------------------------------------

    async def subscribe(
        self,
        event_types: Optional[List[EventType]],
        callback: Callable[[DomainEvent], Any],
    ) -> str:
        """Register an asynchronous callback for a subset of event types.

        Args:
            event_types: List of event types to listen for. ``None`` means
                **all** event types.
            callback: Invoked for every matching event. May be a sync or async
                callable.

        Returns:
            A unique subscription id that can later be passed to
            :meth:`unsubscribe`.
        """
        sub_id = uuid4().hex
        wrapped = _ensure_async_callback(callback)
        async with self._sub_lock:
            self._subscribers[sub_id] = (event_types, wrapped)
        return sub_id

    async def unsubscribe(self, sub_id: str) -> None:
        """Remove a subscription.

        Args:
            sub_id: The identifier returned by :meth:`subscribe`.

        Raises:
            KeyError: If *sub_id* is not known.
        """
        async with self._sub_lock:
            if sub_id not in self._subscribers:
                raise KeyError(f"Subscription {sub_id} not found")
            del self._subscribers[sub_id]

    async def _notify_subscribers(self, event: DomainEvent) -> None:
        """Asynchronously dispatch *event* to every matching subscriber.

        Exceptions in individual callbacks are isolated so that one bad
        subscriber cannot break delivery to the rest.
        """
        async with self._sub_lock:
            # Copy to avoid mutation during iteration.
            subs = list(self._subscribers.items())

        tasks: List[asyncio.Task[None]] = []
        for sub_id, (filter_types, callback) in subs:
            if filter_types is not None and event.event_type not in filter_types:
                continue
            tasks.append(
                asyncio.create_task(
                    self._safe_notify(sub_id, callback, event),
                    name=f"evt-notify-{sub_id}-{event.event_id.hex[:8]}",
                )
            )

        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    async def _safe_notify(
        self,
        sub_id: str,
        callback: Callable[[DomainEvent], Coroutine[Any, Any, None]],
        event: DomainEvent,
    ) -> None:
        """Invoke a single subscriber, swallowing and logging errors."""
        try:
            await callback(event)
        except Exception:
            logger.exception(
                "Subscriber %s failed processing event %s",
                sub_id,
                event.event_id,
            )


# ---------------------------------------------------------------------------
# StateProjection — CQRS read model
# ---------------------------------------------------------------------------

class StateProjection:
    """Incremental read-model built by projecting the event stream.

    ``StateProjection`` subscribes to an ``EventStore`` and continuously
    updates in-memory snapshots of actor and contract state. Because the
    snapshots are derived purely from events, they can be rebuilt from scratch
    at any time (time-travel debugging) and are guaranteed to be deterministic.

    The internal caches are protected by an ``asyncio.Lock`` so that concurrent
    reads and incremental updates are safe.

    Attributes:
        _event_store: The store this projection listens to.
        _projections: Cached actor states keyed by ``actor_id.full``.
        _contract_states: Cached contract metadata keyed by contract UUID.
        _lock: Protects all internal mutable state.
        _sub_id: Active subscription handle (or ``None`` if not started).
    """

    def __init__(self, event_store: EventStore) -> None:
        self._event_store = event_store
        self._projections: Dict[str, ActorState] = {}
        self._contract_states: Dict[UUID, Dict[str, Any]] = {}
        self._lock: asyncio.Lock = asyncio.Lock()
        self._sub_id: Optional[str] = None

    async def start(self) -> None:
        """Begin consuming events from the attached store.

        Idempotent: multiple calls are a no-op if already started.
        """
        async with self._lock:
            if self._sub_id is not None:
                return
        # Subscribe outside the projection lock to avoid deadlocks with the
        # store's internal subscriber lock.
        self._sub_id = await self._event_store.subscribe(
            None,  # all event types
            self._on_event,
        )
        logger.info("StateProjection started (sub_id=%s)", self._sub_id)

    async def stop(self) -> None:
        """Stop consuming events. Idempotent no-op if not started."""
        async with self._lock:
            sid = self._sub_id
            self._sub_id = None
        if sid is not None:
            try:
                await self._event_store.unsubscribe(sid)
            except KeyError:
                pass  # Already unsubscribed by external code.
            logger.info("StateProjection stopped (sub_id=%s)", sid)

    # -- Read API ----------------------------------------------------------

    async def get_actor_state(self, actor_id: ActorId) -> Optional[ActorState]:
        """Return the projected current state for *actor_id*.

        Args:
            actor_id: Actor to look up.

        Returns:
            A deep-copied ``ActorState`` or ``None`` if the actor has never
            been observed.
        """
        async with self._lock:
            state = self._projections.get(actor_id.full)
            if state is None:
                return None
            return deepcopy(state)

    async def get_contract_state(
        self, contract_id: UUID
    ) -> Optional[Dict[str, Any]]:
        """Return the projected current state for *contract_id*.

        Args:
            contract_id: Contract to look up.

        Returns:
            A deep-copied dictionary or ``None`` if the contract is unknown.
        """
        async with self._lock:
            cs = self._contract_states.get(contract_id)
            if cs is None:
                return None
            return deepcopy(cs)

    # -- Rebuild / time-travel ----------------------------------------------

    async def rebuild_from_events(self, events: List[DomainEvent]) -> None:
        """Destructively rebuild all projections from a complete event stream.

        This is the primitive for **time-travel debugging**: by replaying a
        filtered or historical slice of events you can inspect the system state
        at any logical point in time.

        Args:
            events: Ordered list of events to project. Usually obtained via
                ``EventStore.get_all_events()`` or ``get_events()``.
        """
        async with self._lock:
            self._projections.clear()
            self._contract_states.clear()
            for event in events:
                self._apply_event(event)
        logger.info("Rebuilt projections from %d events", len(events))

    # -- Internal event application -----------------------------------------

    async def _on_event(self, event: DomainEvent) -> None:
        """Callback invoked by the EventStore for every new event."""
        async with self._lock:
            self._apply_event(event)

    def _apply_event(self, event: DomainEvent) -> None:
        """Synchronously update projection caches for a single event.

        Must only be called while holding ``self._lock``.
        """
        actor_key = event.actor_id.full
        et = event.event_type
        payload = event.payload

        # ------------------------------------------------------------------
        # Lifecycle events
        # ------------------------------------------------------------------
        if et == EventType.ACTOR_SPAWNED:
            if actor_key not in self._projections:
                self._projections[actor_key] = ActorState(
                    agent_id=event.actor_id,
                    status=ActorStatus.SPAWNING,
                )
            else:
                self._projections[actor_key].status = ActorStatus.SPAWNING

        elif et == EventType.ACTOR_CRASHED:
            state = self._projections.setdefault(
                actor_key, ActorState(agent_id=event.actor_id)
            )
            state.status = ActorStatus.CRASHED

        elif et == EventType.ACTOR_RECOVERED:
            state = self._projections.setdefault(
                actor_key, ActorState(agent_id=event.actor_id)
            )
            state.status = ActorStatus.IDLE

        # ------------------------------------------------------------------
        # Message flow
        # ------------------------------------------------------------------
        elif et == EventType.MESSAGE_RECEIVED:
            state = self._projections.setdefault(
                actor_key, ActorState(agent_id=event.actor_id)
            )
            state.mailbox_size += 1

        elif et == EventType.MESSAGE_PROCESSED:
            state = self._projections.setdefault(
                actor_key, ActorState(agent_id=event.actor_id)
            )
            state.processed_messages += 1
            if state.mailbox_size > 0:
                state.mailbox_size -= 1

        # ------------------------------------------------------------------
        # Contract governance
        # ------------------------------------------------------------------
        elif et == EventType.CONTRACT_CREATED:
            contract_id = payload.get("contract_id")
            if contract_id is not None:
                # Normalise to UUID if passed as string.
                if isinstance(contract_id, str):
                    contract_id = UUID(contract_id)
                state = self._projections.setdefault(
                    actor_key, ActorState(agent_id=event.actor_id)
                )
                state.current_contract = contract_id
                self._contract_states[contract_id] = deepcopy(payload)

        elif et == EventType.CONTRACT_UPDATED:
            contract_id = payload.get("contract_id")
            if isinstance(contract_id, str):
                contract_id = UUID(contract_id)
            if contract_id is not None:
                if contract_id in self._contract_states:
                    self._contract_states[contract_id].update(deepcopy(payload))
                else:
                    self._contract_states[contract_id] = deepcopy(payload)
                state = self._projections.setdefault(
                    actor_key, ActorState(agent_id=event.actor_id)
                )
                state.current_contract = contract_id

        elif et == EventType.CONTRACT_VIOLATED:
            contract_id = payload.get("contract_id")
            if isinstance(contract_id, str):
                contract_id = UUID(contract_id)
            if contract_id is not None and contract_id in self._contract_states:
                self._contract_states[contract_id]["status"] = ContractStatus.VIOLATED

        # ------------------------------------------------------------------
        # Sandbox & metacognition
        # ------------------------------------------------------------------
        elif et == EventType.SANDBOX_ASSIGNED:
            # Sandbox policy can be stored on the actor if needed later.
            state = self._projections.setdefault(
                actor_key, ActorState(agent_id=event.actor_id)
            )
            policy = payload.get("policy")
            if policy:
                state.memory_refs.append(f"sandbox:{policy}")

        elif et == EventType.SANDBOX_EXECUTED:
            # Currently a no-op for the projection; metrics may be added later.
            pass

        elif et == EventType.META_EPISODE:
            state = self._projections.setdefault(
                actor_key, ActorState(agent_id=event.actor_id)
            )
            memory_ref = payload.get("memory_ref")
            if memory_ref:
                state.memory_refs.append(str(memory_ref))

        elif et == EventType.CHECKPOINT:
            state = self._projections.setdefault(
                actor_key, ActorState(agent_id=event.actor_id)
            )
            state.last_checkpoint_at = event.timestamp

        else:
            logger.warning("Unhandled event type in projection: %s", et.value)


# ---------------------------------------------------------------------------
# CheckpointManager — Snapshotting & log compaction
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Checkpoint:
    """Immutable snapshot of an actor's state at a logical point in time."""

    checkpoint_id: UUID
    actor_id: ActorId
    state: ActorState
    vector_clock: Dict[str, int]
    created_at: datetime


class CheckpointManager:
    """Creates, stores and restores actor checkpoints.

    A checkpoint is a named snapshot of ``ActorState`` together with the
    vector-clock frontier that produced it. Checkpoints enable two powerful
    operations:

    1. **Fast recovery** — an actor can be restored from its latest checkpoint
       without replaying the entire event log.
    2. **Log compaction** — events that are already reflected in a checkpoint
       can be pruned, keeping the event store bounded in size.

    Attributes:
        _event_store: Store that owns the event log.
        _projection: Live read-model used to capture current state.
        _checkpoints: In-memory registry of checkpoints.
        _lock: Serialises checkpoint creation and restoration.
    """

    def __init__(
        self, event_store: EventStore, projection: StateProjection
    ) -> None:
        self._event_store = event_store
        self._projection = projection
        self._checkpoints: Dict[UUID, Checkpoint] = {}
        self._lock: asyncio.Lock = asyncio.Lock()

    async def create_checkpoint(self, actor_id: ActorId) -> UUID:
        """Capture the current projected state of *actor_id* as a checkpoint.

        A ``CHECKPOINT`` domain event is first appended to the store so that the
        vector-clock horizon includes the checkpoint itself, then the updated
        clock is captured in the snapshot record.

        Args:
            actor_id: Actor to snapshot.

        Returns:
            The unique identifier of the newly created checkpoint.

        Raises:
            ValueError: If the actor has no projection (i.e. no events for it
                have ever been observed).
        """
        # 1. Emit CHECKPOINT event first so the vector clock advances.
        await self._event_store.append(
            DomainEvent(
                event_type=EventType.CHECKPOINT,
                actor_id=actor_id,
                payload={},
                timestamp=datetime.utcnow(),
            )
        )

        async with self._lock:
            state = await self._projection.get_actor_state(actor_id)
            if state is None:
                raise ValueError(
                    f"No projection found for actor {actor_id.full}; "
                    "cannot checkpoint an unseen actor."
                )

            # 2. Capture clock *after* the CHECKPOINT event is persisted.
            clock = self._event_store.get_vector_clock(actor_id)
            cp_id = uuid4()
            checkpoint = Checkpoint(
                checkpoint_id=cp_id,
                actor_id=actor_id,
                state=deepcopy(state),
                vector_clock=_copy_vector_clock(clock),
                created_at=datetime.utcnow(),
            )
            self._checkpoints[cp_id] = checkpoint

        logger.info(
            "Checkpoint %s created for actor %s at clock %s",
            cp_id,
            actor_id.full,
            clock,
        )
        return cp_id

    async def restore_from_checkpoint(self, checkpoint_id: UUID) -> ActorState:
        """Restore an actor state from a previously created checkpoint.

        Args:
            checkpoint_id: The checkpoint to restore.

        Returns:
            A deep-copied ``ActorState`` snapshot.

        Raises:
            ValueError: If the checkpoint does not exist.
        """
        async with self._lock:
            cp = self._checkpoints.get(checkpoint_id)
            if cp is None:
                raise ValueError(f"Checkpoint {checkpoint_id} not found")
            return deepcopy(cp.state)

    async def prune_events_before(self, checkpoint_id: UUID) -> int:
        """Compact the event log by removing events already captured in a checkpoint.

        Only events for the checkpoint's actor that are causally at or before
        the checkpoint's vector-clock frontier are removed. Events after the
        frontier are retained so that incremental replay remains possible.

        Args:
            checkpoint_id: Checkpoint whose horizon should be used for pruning.

        Returns:
            Number of events removed from the store.

        Raises:
            ValueError: If the checkpoint does not exist.
        """
        async with self._lock:
            cp = self._checkpoints.get(checkpoint_id)
            if cp is None:
                raise ValueError(f"Checkpoint {checkpoint_id} not found")

        removed = await self._event_store.prune_events(
            cp.actor_id, cp.vector_clock
        )
        logger.info(
            "Pruned %d events before checkpoint %s",
            removed,
            checkpoint_id,
        )
        return removed
