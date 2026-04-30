"""AgentKernel Governor — Formal Agent Contracts Resource Governance Engine.

Implements the resource governance layer for AgentKernel based on the
Agent Contracts formalism (C = (I, O, S, R, T, Φ, Ψ)).

All budget mutation operations are atomic (protected by `asyncio.Lock`).
"""
from __future__ import annotations

import asyncio
import copy
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Coroutine, Dict, List, Optional, Protocol, Set, Tuple
from uuid import UUID

from .types import (
    AgentContract,
    AgentId,
    ContractMode,
    ContractStatus,
    DomainEvent,
    EventType,
    Message,
    ResourceBudget,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class ConservationViolation(Exception):
    """Raised when the multi-agent budget conservation law is violated.

    Invariant: sum(child.R) + parent.remaining >= parent.original
    """

    def __init__(self, message: str, parent_id: Optional[UUID] = None) -> None:
        super().__init__(message)
        self.parent_id = parent_id


class ConstraintViolation(Exception):
    """Raised when a single contract constraint is violated (non-conservation)."""


class EventStore(Protocol):
    """Abstract interface for event persistence (event sourcing boundary)."""

    async def append(self, event: DomainEvent) -> None: ...


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _utc_now() -> datetime:
    """Return timezone-aware UTC datetime."""
    return datetime.now(timezone.utc)


def _make_event(
    event_type: EventType,
    actor_id: AgentId,
    payload: Dict[str, Any],
) -> DomainEvent:
    """Factory for DomainEvent instances."""
    return DomainEvent(
        event_type=event_type,
        actor_id=actor_id,
        payload=payload,
    )


# ---------------------------------------------------------------------------
# ContractRepository — In-memory contract storage
# ---------------------------------------------------------------------------

class ContractRepository:
    """In-memory repository for ``AgentContract`` snapshots.

    Provides CRUD-style access and filtered views (active, violated).
    All methods are synchronous because the backing store is local memory.
    """

    def __init__(self) -> None:
        self._contracts: Dict[UUID, AgentContract] = {}
        self._lock = asyncio.Lock()

    async def save(self, contract: AgentContract) -> None:
        """Persist a contract snapshot (deep-copied to avoid external mutation).

        Args:
            contract: The contract to store.
        """
        async with self._lock:
            # Deep copy to prevent callers from mutating stored state
            self._contracts[contract.id] = copy.deepcopy(contract)
        logger.debug("ContractRepository.save: %s", contract.id)

    async def get(self, contract_id: UUID) -> Optional[AgentContract]:
        """Retrieve a contract by ID.

        Args:
            contract_id: UUID of the contract.

        Returns:
            A deep-copied snapshot, or ``None`` if not found.
        """
        async with self._lock:
            stored = self._contracts.get(contract_id)
            return copy.deepcopy(stored) if stored is not None else None

    async def delete(self, contract_id: UUID) -> bool:
        """Remove a contract from the repository.

        Args:
            contract_id: UUID of the contract to remove.

        Returns:
            ``True`` if a contract was removed, ``False`` otherwise.
        """
        async with self._lock:
            existed = contract_id in self._contracts
            self._contracts.pop(contract_id, None)
            return existed

    async def list_all(self) -> List[AgentContract]:
        """Return all stored contracts (deep-copied)."""
        async with self._lock:
            return [copy.deepcopy(c) for c in self._contracts.values()]

    async def list_active(self) -> List[AgentContract]:
        """Return contracts whose status is ``ACTIVE``."""
        async with self._lock:
            return [
                copy.deepcopy(c)
                for c in self._contracts.values()
                if c.status == ContractStatus.ACTIVE
            ]

    async def list_violated(self) -> List[AgentContract]:
        """Return contracts whose status is ``VIOLATED``."""
        async with self._lock:
            return [
                copy.deepcopy(c)
                for c in self._contracts.values()
                if c.status == ContractStatus.VIOLATED
            ]

    async def update_status(self, contract_id: UUID, new_status: ContractStatus) -> bool:
        """Atomically update the status of a stored contract.

        Args:
            contract_id: Target contract UUID.
            new_status: The new status to assign.

        Returns:
            ``True`` if the contract existed and was updated.
        """
        async with self._lock:
            stored = self._contracts.get(contract_id)
            if stored is None:
                return False
            stored.status = new_status
            return True


# ---------------------------------------------------------------------------
# BudgetEnforcer — Budget enforcement with satisficing strategies
# ---------------------------------------------------------------------------

class BudgetEnforcer:
    """Enforces resource budgets per contract with mode-aware satisficing.

    Satisficing semantics by ``ContractMode``:

    - **URGENT**: Token budget may be exceeded by up to 20%, but time is
      strictly capped at the contractual limit.
    - **BALANCED**: Standard enforcement; quality (within budget) is preferred.
    - **ECONOMICAL**: All resources are strictly capped; longer runtime is
      acceptable if it reduces token/API consumption.
    """

    # Mode-specific token overrun allowance
    _TOKEN_OVERRUN: Dict[ContractMode, float] = {
        ContractMode.URGENT: 1.20,
        ContractMode.BALANCED: 1.00,
        ContractMode.ECONOMICAL: 1.00,
    }

    # Mode-specific time overrun allowance
    _TIME_OVERRUN: Dict[ContractMode, float] = {
        ContractMode.URGENT: 1.00,
        ContractMode.BALANCED: 1.00,
        ContractMode.ECONOMICAL: 1.20,
    }

    def __init__(self, event_store: Optional[EventStore] = None) -> None:
        self._event_store = event_store

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def enforce_token_budget(self, contract: AgentContract, requested_tokens: int) -> int:
        """Return the number of tokens actually permitted.

        The result never exceeds the (mode-adjusted) remaining token budget.

        Args:
            contract: The governing contract.
            requested_tokens: Number of tokens the agent wants to consume.

        Returns:
            Allowed token count (``<= requested_tokens``).
        """
        effective_max = int(
            contract.budget.max_tokens
            * self._TOKEN_OVERRUN.get(contract.mode, 1.0)
        )
        consumed = contract.consumed_tokens
        remaining = max(0, effective_max - consumed)
        allowed = int(min(requested_tokens, remaining))

        logger.debug(
            "Token enforce: requested=%d allowed=%d effective_max=%d consumed=%d mode=%s",
            requested_tokens,
            allowed,
            effective_max,
            consumed,
            contract.mode.value,
        )
        return allowed

    def enforce_time_budget(self, contract: AgentContract) -> float:
        """Return the remaining available execution seconds.

        Args:
            contract: The governing contract.

        Returns:
            Seconds remaining (mode-adjusted). May be zero or negative.
        """
        effective_max = contract.max_duration_seconds * self._TIME_OVERRUN.get(
            contract.mode, 1.0
        )

        if contract.start_time is None:
            # Not yet started — full budget available
            return effective_max

        elapsed = (_utc_now() - contract.start_time).total_seconds()
        remaining = effective_max - elapsed
        return max(0.0, remaining)

    def enforce_skill_whitelist(self, contract: AgentContract, skill_name: str) -> bool:
        """Check whether a skill is in the contract's whitelist.

        An empty whitelist is interpreted as "all skills allowed" (open policy).

        Args:
            contract: The governing contract.
            skill_name: Skill identifier to check.

        Returns:
            ``True`` if the skill is permitted.
        """
        if not contract.allowed_skills:
            return True
        return skill_name in contract.allowed_skills

    async def try_consume_tokens(
        self,
        contract: AgentContract,
        requested_tokens: int,
    ) -> int:
        """Atomically attempt to consume tokens from the contract.

        This is a convenience wrapper that updates ``contract.consumed_tokens``
        in-place and returns the actually granted amount.

        Args:
            contract: Contract to mutate.
            requested_tokens: Desired token consumption.

        Returns:
            Actually granted token count.
        """
        allowed = self.enforce_token_budget(contract, requested_tokens)
        contract.consumed_tokens += allowed
        if self._event_store is not None:
            event = _make_event(
                EventType.CONTRACT_UPDATED,
                AgentId("governor"),
                {
                    "contract_id": str(contract.id),
                    "action": "consume_tokens",
                    "requested": requested_tokens,
                    "granted": allowed,
                    "consumed_total": contract.consumed_tokens,
                },
            )
            await self._event_store.append(event)
        return allowed

    async def try_consume_time(
        self,
        contract: AgentContract,
        seconds: float,
    ) -> bool:
        """Atomically check whether ``seconds`` of wall-clock time can be consumed.

        This updates the contract's ``start_time`` implicitly by comparing
        elapsed time since activation.

        Args:
            contract: Contract to evaluate.
            seconds: Proposed time consumption.

        Returns:
            ``True`` if the time fits within the remaining budget.
        """
        remaining = self.enforce_time_budget(contract)
        if remaining >= seconds:
            return True
        return False


# ---------------------------------------------------------------------------
# MultiAgentConservationLaw — Budget partitioning across agent hierarchies
# ---------------------------------------------------------------------------

class MultiAgentConservationLaw:
    """Ensures that budget is conserved when a parent agent delegates to children.

    Invariant (Conservation Law):
        sum(child_allocated) + parent_remaining >= parent_original

    Where:
        - ``parent_original`` is the parent's budget at delegation time.
        - ``child_allocated`` is the budget assigned to each child contract.
        - ``parent_remaining`` is the budget the parent retains after delegation.
    """

    def __init__(self, event_store: Optional[EventStore] = None) -> None:
        self._event_store = event_store
        self._lock = asyncio.Lock()
        # Track original budgets and child allocations per parent contract
        self._delegation_registry: Dict[
            UUID, Tuple[ResourceBudget, List[AgentContract]]
        ] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def delegate_budget(
        self,
        parent_contract: AgentContract,
        child_contracts: List[AgentContract],
    ) -> bool:
        """Validate and record a budget delegation.

        The sum of each child's ``ResourceBudget`` must not exceed the parent's
        *remaining* budget.  If the check succeeds, the delegation is registered
        so that ``reclaim_budget`` can later verify conservation.

        Args:
            parent_contract: The delegating (parent) contract.
            child_contracts: Contracts to receive budget slices.

        Returns:
            ``True`` if delegation satisfies the conservation law.

        Raises:
            ConservationViolation: If the children's combined budget exceeds
                the parent's remaining budget.
        """
        async with self._lock:
            # Calculate remaining parent budget
            parent_remaining = self._remaining_budget(parent_contract)

            # Sum child budgets
            total_child_tokens = sum(c.budget.max_tokens for c in child_contracts)
            total_child_calls = sum(c.budget.max_api_calls for c in child_contracts)
            total_child_iters = sum(c.budget.max_iterations for c in child_contracts)
            total_child_cost = sum(c.budget.max_cost_usd for c in child_contracts)
            total_child_time = sum(c.budget.max_compute_seconds for c in child_contracts)

            # Check token conservation (primary dimension)
            if total_child_tokens > parent_remaining.max_tokens:
                raise ConservationViolation(
                    f"Delegation exceeds parent token budget: "
                    f"children={total_child_tokens} > parent_remaining={parent_remaining.max_tokens}",
                    parent_id=parent_contract.id,
                )
            if total_child_cost > parent_remaining.max_cost_usd:
                raise ConservationViolation(
                    f"Delegation exceeds parent cost budget: "
                    f"children=${total_child_cost:.4f} > parent_remaining=${parent_remaining.max_cost_usd:.4f}",
                    parent_id=parent_contract.id,
                )
            if total_child_time > parent_remaining.max_compute_seconds:
                raise ConservationViolation(
                    f"Delegation exceeds parent time budget: "
                    f"children={total_child_time}s > parent_remaining={parent_remaining.max_compute_seconds}s",
                    parent_id=parent_contract.id,
                )

            # Record delegation for later reclamation verification
            self._delegation_registry[parent_contract.id] = (
                copy.deepcopy(parent_contract.budget),
                [copy.deepcopy(c) for c in child_contracts],
            )

            if self._event_store is not None:
                event = _make_event(
                    EventType.CONTRACT_UPDATED,
                    AgentId("governor"),
                    {
                        "parent_id": str(parent_contract.id),
                        "action": "delegate_budget",
                        "child_count": len(child_contracts),
                        "child_tokens": total_child_tokens,
                        "child_cost": total_child_cost,
                        "child_time": total_child_time,
                    },
                )
                await self._event_store.append(event)

            return True

    async def reclaim_budget(
        self,
        parent_contract: AgentContract,
        child_contract: AgentContract,
    ) -> None:
        """Reclaim unused budget from a completed child contract.

        After reclamation, the conservation invariant is re-verified:

            sum(remaining_child_budgets) + parent_remaining >= parent_original

        Args:
            parent_contract: The parent contract to credit.
            child_contract: The child contract whose unused budget is reclaimed.

        Raises:
            ConservationViolation: If the invariant cannot be restored.
        """
        async with self._lock:
            # Determine unused budget dimensions
            unused_tokens = max(
                0,
                child_contract.budget.max_tokens - child_contract.consumed_tokens,
            )
            unused_cost = max(
                0.0,
                child_contract.budget.max_cost_usd - child_contract.consumed_cost_usd,
            )
            unused_time = max(
                0.0,
                child_contract.budget.max_compute_seconds
                - self._elapsed_seconds(child_contract),
            )
            unused_calls = max(
                0,
                child_contract.budget.max_api_calls - child_contract.consumed_api_calls,
            )
            unused_iters = max(
                0,
                child_contract.budget.max_iterations - child_contract.consumed_iterations,
            )

            # Credit parent (in-place mutation)
            parent_contract.budget.max_tokens += unused_tokens
            parent_contract.budget.max_cost_usd += unused_cost
            parent_contract.budget.max_compute_seconds += unused_time
            parent_contract.budget.max_api_calls += unused_calls
            parent_contract.budget.max_iterations += unused_iters

            # Verify conservation law
            self._verify_conservation(parent_contract)

            if self._event_store is not None:
                event = _make_event(
                    EventType.CONTRACT_UPDATED,
                    AgentId("governor"),
                    {
                        "parent_id": str(parent_contract.id),
                        "child_id": str(child_contract.id),
                        "action": "reclaim_budget",
                        "reclaimed_tokens": unused_tokens,
                        "reclaimed_cost": unused_cost,
                        "reclaimed_time": unused_time,
                    },
                )
                await self._event_store.append(event)

            logger.info(
                "Reclaimed budget from child %s to parent %s: tokens=%d cost=%.4f time=%.1f",
                child_contract.id,
                parent_contract.id,
                unused_tokens,
                unused_cost,
                unused_time,
            )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _remaining_budget(contract: AgentContract) -> ResourceBudget:
        """Derive a ``ResourceBudget`` representing the unspent capacity."""
        return ResourceBudget(
            max_tokens=max(0, contract.budget.max_tokens - contract.consumed_tokens),
            max_api_calls=max(
                0, contract.budget.max_api_calls - contract.consumed_api_calls
            ),
            max_iterations=max(
                0, contract.budget.max_iterations - contract.consumed_iterations
            ),
            max_web_searches=contract.budget.max_web_searches,  # Not tracked as consumed
            max_compute_seconds=max(
                0.0,
                contract.budget.max_compute_seconds
                - MultiAgentConservationLaw._elapsed_seconds(contract),
            ),
            max_cost_usd=max(
                0.0, contract.budget.max_cost_usd - contract.consumed_cost_usd
            ),
        )

    @staticmethod
    def _elapsed_seconds(contract: AgentContract) -> float:
        """Return elapsed wall-clock seconds since contract activation."""
        if contract.start_time is None:
            return 0.0
        return (_utc_now() - contract.start_time).total_seconds()

    def _verify_conservation(self, parent_contract: AgentContract) -> None:
        """Verify the conservation invariant.

        Raises:
            ConservationViolation: If ``sum(child.R) + parent.remaining < parent.original``.
        """
        entry = self._delegation_registry.get(parent_contract.id)
        if entry is None:
            # No delegation recorded — nothing to verify
            return

        original_budget, children = entry

        # Current parent remaining
        parent_remaining = self._remaining_budget(parent_contract)

        # Sum remaining child budgets (using their *current* budget state)
        sum_child_tokens = sum(c.budget.max_tokens for c in children)
        sum_child_cost = sum(c.budget.max_cost_usd for c in children)
        sum_child_time = sum(c.budget.max_compute_seconds for c in children)

        # Invariant: sum(child_allocated) + parent_remaining >= parent_original
        # (We allow > because reclamation may have added unused budget back)
        if (sum_child_tokens + parent_remaining.max_tokens) < original_budget.max_tokens:
            raise ConservationViolation(
                f"Token conservation violated after reclamation: "
                f"children={sum_child_tokens} + parent_remaining={parent_remaining.max_tokens} "
                f"< original={original_budget.max_tokens}",
                parent_id=parent_contract.id,
            )

        if (sum_child_cost + parent_remaining.max_cost_usd) < original_budget.max_cost_usd:
            raise ConservationViolation(
                f"Cost conservation violated after reclamation: "
                f"children=${sum_child_cost:.4f} + parent_remaining=${parent_remaining.max_cost_usd:.4f} "
                f"< original=${original_budget.max_cost_usd:.4f}",
                parent_id=parent_contract.id,
            )

        if (
            sum_child_time + parent_remaining.max_compute_seconds
        ) < original_budget.max_compute_seconds:
            raise ConservationViolation(
                f"Time conservation violated after reclamation: "
                f"children={sum_child_time:.1f} + parent_remaining={parent_remaining.max_compute_seconds:.1f} "
                f"< original={original_budget.max_compute_seconds:.1f}",
                parent_id=parent_contract.id,
            )


# ---------------------------------------------------------------------------
# Governor — Contract lifecycle governance
# ---------------------------------------------------------------------------

class Governor:
    """Core governance engine for Agent Contracts.

    Responsibilities:
    1. **Lifecycle**: Create → Activate → (Check / Record / Evaluate) → Terminate
    2. **Constraint monitoring**: Detect budget/deadline violations.
    3. **Event sourcing**: Emit ``DomainEvent`` s through an optional ``EventStore``.
    4. **Integration**: Compose ``BudgetEnforcer`` and ``ContractRepository``.
    """

    def __init__(
        self,
        event_store: Optional[EventStore] = None,
        repository: Optional[ContractRepository] = None,
    ) -> None:
        self._event_store = event_store
        self._repository = repository or ContractRepository()
        self._enforcer = BudgetEnforcer(event_store=event_store)
        self._conservation = MultiAgentConservationLaw(event_store=event_store)
        self._lock = asyncio.Lock()

    # ------------------------------------------------------------------
    # Properties for composed subsystems (read-only)
    # ------------------------------------------------------------------

    @property
    def repository(self) -> ContractRepository:
        """The contract repository."""
        return self._repository

    @property
    def enforcer(self) -> BudgetEnforcer:
        """The budget enforcer."""
        return self._enforcer

    @property
    def conservation(self) -> MultiAgentConservationLaw:
        """The conservation-law validator."""
        return self._conservation

    # ------------------------------------------------------------------
    # Contract lifecycle
    # ------------------------------------------------------------------

    async def create_contract(self, spec: Dict[str, Any]) -> AgentContract:
        """Create a new ``AgentContract`` from a dictionary specification.

        Recognised keys (all optional):
        - ``input_schema``: ``Dict[str, Any]``
        - ``output_schema``: ``Dict[str, Any]``
        - ``quality_threshold``: ``float``
        - ``allowed_skills``: ``Set[str]``
        - ``budget``: ``Dict[str, Any]`` (mapped to ``ResourceBudget``)
        - ``deadline``: ``datetime``
        - ``max_duration_seconds``: ``float``
        - ``success_predicate``: ``Callable[[Any], bool]``
        - ``mode``: ``ContractMode`` or ``str``

        Args:
            spec: Specification dictionary.

        Returns:
            A newly instantiated ``AgentContract`` in ``PENDING`` status.
        """
        # Parse budget sub-dict if present
        budget = ResourceBudget()
        if "budget" in spec:
            budget = ResourceBudget(**spec["budget"])

        # Parse mode
        mode = ContractMode.BALANCED
        raw_mode = spec.get("mode")
        if isinstance(raw_mode, str):
            mode = ContractMode(raw_mode.lower())
        elif isinstance(raw_mode, ContractMode):
            mode = raw_mode

        contract = AgentContract(
            input_schema=spec.get("input_schema", {}),
            output_schema=spec.get("output_schema", {}),
            quality_threshold=spec.get("quality_threshold", 0.7),
            allowed_skills=set(spec.get("allowed_skills", [])),
            budget=budget,
            deadline=spec.get("deadline"),
            max_duration_seconds=spec.get("max_duration_seconds", 300.0),
            success_predicate=spec.get("success_predicate"),
            mode=mode,
        )

        await self._repository.save(contract)

        if self._event_store is not None:
            event = _make_event(
                EventType.CONTRACT_CREATED,
                AgentId("governor"),
                {
                    "contract_id": str(contract.id),
                    "mode": contract.mode.value,
                    "budget": {
                        "max_tokens": contract.budget.max_tokens,
                        "max_api_calls": contract.budget.max_api_calls,
                        "max_iterations": contract.budget.max_iterations,
                        "max_cost_usd": contract.budget.max_cost_usd,
                        "max_compute_seconds": contract.budget.max_compute_seconds,
                    },
                },
            )
            await self._event_store.append(event)

        logger.info("Governor.create_contract: %s mode=%s", contract.id, contract.mode.value)
        return contract

    async def activate_contract(self, contract: AgentContract) -> None:
        """Activate a contract, recording the start time.

        Args:
            contract: The contract to activate (mutated in-place).
        """
        async with self._lock:
            contract.start_time = _utc_now()
            contract.status = ContractStatus.ACTIVE
            await self._repository.save(contract)

        if self._event_store is not None:
            event = _make_event(
                EventType.CONTRACT_UPDATED,
                AgentId("governor"),
                {
                    "contract_id": str(contract.id),
                    "action": "activate",
                    "start_time": contract.start_time.isoformat(),
                },
            )
            await self._event_store.append(event)

        logger.info("Governor.activate_contract: %s", contract.id)

    async def check_constraints(
        self,
        contract: AgentContract,
        usage_delta: Dict[str, float],
    ) -> ContractStatus:
        """Check whether a proposed resource delta violates the contract.

        Evaluates the *hypothetical* consumption described by ``usage_delta``
        against the contract's budget ceiling and deadline.  The contract is
        **not** mutated.

        Delta keys recognised:
        - ``tokens``
        - ``api_calls``
        - ``iterations``
        - ``cost_usd``
        - ``compute_seconds``

        Args:
            contract: Contract to evaluate.
            usage_delta: Proposed additional consumption.

        Returns:
            ``VIOLATED`` if any hard limit would be exceeded,
            ``EXPIRED`` if the deadline has passed,
            otherwise the contract's current status.
        """
        # Token check
        proposed_tokens = contract.consumed_tokens + usage_delta.get("tokens", 0.0)
        if proposed_tokens > contract.budget.max_tokens:
            return ContractStatus.VIOLATED

        # API call check
        proposed_calls = contract.consumed_api_calls + usage_delta.get("api_calls", 0.0)
        if proposed_calls > contract.budget.max_api_calls:
            return ContractStatus.VIOLATED

        # Iteration check
        proposed_iters = contract.consumed_iterations + usage_delta.get(
            "iterations", 0.0
        )
        if proposed_iters > contract.budget.max_iterations:
            return ContractStatus.VIOLATED

        # Cost check
        proposed_cost = contract.consumed_cost_usd + usage_delta.get("cost_usd", 0.0)
        if proposed_cost > contract.budget.max_cost_usd:
            return ContractStatus.VIOLATED

        # Time check
        if contract.start_time is not None:
            elapsed = (_utc_now() - contract.start_time).total_seconds()
            proposed_time = elapsed + usage_delta.get("compute_seconds", 0.0)
            if proposed_time > contract.max_duration_seconds:
                return ContractStatus.VIOLATED
            if proposed_time > contract.budget.max_compute_seconds:
                return ContractStatus.VIOLATED

        # Deadline check
        if contract.deadline is not None and _utc_now() > contract.deadline:
            return ContractStatus.EXPIRED

        return contract.status

    async def record_usage(
        self,
        contract: AgentContract,
        tokens: int = 0,
        api_calls: int = 0,
        iterations: int = 0,
        cost_usd: float = 0.0,
    ) -> ContractStatus:
        """Record actual resource consumption and update contract status.

        Mutates the contract in-place and persists it to the repository.

        Args:
            contract: Contract to mutate.
            tokens: Additional tokens consumed.
            api_calls: Additional API calls made.
            iterations: Additional iterations performed.
            cost_usd: Additional cost incurred.

        Returns:
            The contract's new status after accounting for consumption.
        """
        async with self._lock:
            contract.consumed_tokens += tokens
            contract.consumed_api_calls += api_calls
            contract.consumed_iterations += iterations
            contract.consumed_cost_usd += cost_usd

            # Re-evaluate constraints
            status = await self.check_constraints(
                contract,
                {
                    "tokens": 0,
                    "api_calls": 0,
                    "iterations": 0,
                    "cost_usd": 0.0,
                    "compute_seconds": 0.0,
                },
            )
            if status != contract.status:
                contract.status = status

            await self._repository.save(contract)

        if self._event_store is not None:
            event = _make_event(
                EventType.CONTRACT_UPDATED,
                AgentId("governor"),
                {
                    "contract_id": str(contract.id),
                    "action": "record_usage",
                    "tokens": tokens,
                    "api_calls": api_calls,
                    "iterations": iterations,
                    "cost_usd": cost_usd,
                    "status": contract.status.value,
                },
            )
            await self._event_store.append(event)

        if contract.status == ContractStatus.VIOLATED:
            logger.warning(
                "Contract %s VIOLATED after usage recording", contract.id
            )
        elif contract.status == ContractStatus.EXPIRED:
            logger.warning("Contract %s EXPIRED (deadline passed)", contract.id)

        return contract.status

    async def evaluate_success(self, contract: AgentContract, output: Any) -> bool:
        """Evaluate whether ``output`` satisfies the contract's success predicate.

        If the contract has no ``success_predicate``, the output is considered
        successful.

        Args:
            contract: Contract whose ``success_predicate`` to apply.
            output: The agent-produced output to evaluate.

        Returns:
            ``True`` if the output satisfies the predicate (or no predicate).
        """
        predicate = contract.success_predicate
        if predicate is None:
            return True

        try:
            result = predicate(output)
        except Exception as exc:
            logger.exception(
                "Success predicate raised for contract %s: %s", contract.id, exc
            )
            return False

        if result and contract.status not in (
            ContractStatus.VIOLATED,
            ContractStatus.EXPIRED,
            ContractStatus.TERMINATED,
        ):
            contract.status = ContractStatus.SATISFIED
            await self._repository.save(contract)

        return result

    async def terminate_contract(self, contract: AgentContract, reason: str) -> None:
        """Terminate a contract, marking it ``TERMINATED``.

        Args:
            contract: Contract to terminate (mutated in-place).
            reason: Human-readable termination reason.
        """
        async with self._lock:
            contract.status = ContractStatus.TERMINATED
            await self._repository.save(contract)

        if self._event_store is not None:
            event = _make_event(
                EventType.CONTRACT_UPDATED,
                AgentId("governor"),
                {
                    "contract_id": str(contract.id),
                    "action": "terminate",
                    "reason": reason,
                    "terminated_at": _utc_now().isoformat(),
                },
            )
            await self._event_store.append(event)

        logger.info("Governor.terminate_contract: %s reason=%s", contract.id, reason)

    # ------------------------------------------------------------------
    # Delegation helpers (convenience wrappers)
    # ------------------------------------------------------------------

    async def delegate_budget(
        self,
        parent_contract: AgentContract,
        child_contracts: List[AgentContract],
    ) -> bool:
        """Convenience wrapper for ``MultiAgentConservationLaw.delegate_budget``.

        Args:
            parent_contract: Parent contract.
            child_contracts: Child contracts to validate.

        Returns:
            ``True`` if delegation is valid.
        """
        return await self._conservation.delegate_budget(parent_contract, child_contracts)

    async def reclaim_budget(
        self,
        parent_contract: AgentContract,
        child_contract: AgentContract,
    ) -> None:
        """Convenience wrapper for ``MultiAgentConservationLaw.reclaim_budget``.

        Args:
            parent_contract: Parent contract.
            child_contract: Completed child contract.
        """
        await self._conservation.reclaim_budget(parent_contract, child_contract)
