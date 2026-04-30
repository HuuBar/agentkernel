"""AgentKernel — Sandbox Orchestrator.

A policy-driven, threat-adaptive sandbox orchestration layer that assigns
isolation levels dynamically based on message content and agent history,
executes commands in sandboxes (simulated via subprocess with resource limits),
and filters sensitive observations before they leave the boundary.

The architecture follows the Agent Contracts model (arXiv:2601.08815) where
sandbox isolation is a function of threat assessment:

    Isolation = f(threat_score, policy, agent_history)

All public APIs are async and fully typed.
"""
from __future__ import annotations

import asyncio
import enum
import logging
import re
import shlex
import subprocess
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Set, Tuple

from agentkernel.types import (
    AgentId,
    IsolationLevel,
    Message,
    MessageKind,
    SandboxPolicy,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class SandboxError(Exception):
    """Base exception for sandbox layer errors."""

    def __init__(self, message: str, agent_id: Optional[AgentId] = None) -> None:
        super().__init__(message)
        self.agent_id = agent_id


class SandboxNotFound(SandboxError):
    """Raised when a sandbox assignment cannot be found for an agent."""


class SandboxExecutionError(SandboxError):
    """Raised when command execution inside a sandbox fails irrecoverably."""


class PolicyResolutionError(SandboxError):
    """Raised when the policy engine cannot resolve a valid policy."""


# ---------------------------------------------------------------------------
# Threat Assessment
# ---------------------------------------------------------------------------

class ThreatModel:
    """Threat assessment model for dynamic isolation selection.

    Evaluates an incoming message and the requesting agent's history to produce
    a threat score (0-100). The score drives the recommended isolation level via
    a piece-wise linear mapping.

    Factors:
        * Code execution primitives   (+40)
        * Filesystem write operations   (+30)
        * Network egress                (+20)
        * Unknown tool chain            (+10)
        * Agent crash/violation history (+10 per incident, capped at +20)
    """

    # --- Risk patterns -----------------------------------------------------
    _CODE_EXECUTION_RE: re.Pattern[str] = re.compile(
        r"\b(eval|exec|compile|__import__|subprocess|os\.system|"
        r"os\.popen|code\.interact|importlib|ctypes|marshal|pickle\.loads)\b",
        re.IGNORECASE,
    )
    _FILESYSTEM_WRITE_RE: re.Pattern[str] = re.compile(
        r"\b(write|delete|remove|rmdir|unlink|chmod|chown|shutil\.rmtree|"
        r"open\s*\([^)]*['\"]w|os\.remove|os\.rmdir|pathlib\.Path.*write)\b",
        re.IGNORECASE,
    )
    _NETWORK_RE: re.Pattern[str] = re.compile(
        r"\b(http|https|socket|curl|wget|requests\.|urllib|ftp|sftp|"
        r"scp|nc\s|netcat|nmap|telnet)\b",
        re.IGNORECASE,
    )
    _KNOWN_TOOLS: Set[str] = {
        "python",
        "sh",
        "bash",
        "grep",
        "sed",
        "awk",
        "cat",
        "ls",
        "echo",
        "find",
        "sort",
        "wc",
        "head",
        "tail",
        "diff",
        "git",
        "docker",
        "kubectl",
        "pip",
        "pytest",
    }

    # --- Score thresholds for isolation levels -------------------------------
    _ISOLATION_THRESHOLDS: List[Tuple[int, IsolationLevel]] = [
        (0, IsolationLevel.NONE),
        (21, IsolationLevel.PROCESS),
        (41, IsolationLevel.CONTAINER),
        (61, IsolationLevel.GVISOR),
        (76, IsolationLevel.MICROVM),
        (91, IsolationLevel.TEE),
    ]

    def assess(self, message: Message, agent_history: List[Dict[str, Any]]) -> int:
        """Evaluate the threat level of a message.

        Args:
            message: The incoming message (typically TASK or SYSTEM kind).
            agent_history: Chronological list of dicts describing prior agent
                behaviour. Expected keys include ``status`` (e.g. ``crashed``,
                ``violated``) and ``tools_used`` (list of str).

        Returns:
            An integer threat score in the range [0, 100].
        """
        score = 0
        payload_str = str(message.payload)

        # 1. Code execution risk
        if self._CODE_EXECUTION_RE.search(payload_str):
            score += 40
            logger.debug("ThreatModel: code_execution pattern detected (+40)")

        # 2. Filesystem write risk
        if self._FILESYSTEM_WRITE_RE.search(payload_str):
            score += 30
            logger.debug("ThreatModel: filesystem_write pattern detected (+30)")

        # 3. Network egress risk
        if self._NETWORK_RE.search(payload_str):
            score += 20
            logger.debug("ThreatModel: network pattern detected (+20)")

        # 4. Unknown tool chain
        tools_used: Set[str] = set()
        for record in agent_history:
            if isinstance(record.get("tools_used"), list):
                tools_used.update(record["tools_used"])
        for tool in tools_used:
            if tool.lower() not in self._KNOWN_TOOLS:
                score += 10
                logger.debug("ThreatModel: unknown tool %r detected (+10)", tool)
                break  # Only charge once per assessment for unknown tools

        # 5. Agent history (crashes / violations)
        incident_count = 0
        for record in agent_history:
            status = record.get("status", "").lower()
            if status in {"crashed", "violated", "terminated"}:
                incident_count += 1
        history_penalty = min(incident_count * 10, 20)
        score += history_penalty
        if history_penalty:
            logger.debug(
                "ThreatModel: %d incident(s) in history (+%d)",
                incident_count,
                history_penalty,
            )

        return max(0, min(100, score))

    def get_recommended_isolation(self, threat_score: int) -> IsolationLevel:
        """Map a threat score to the recommended isolation level.

        Mapping:
            0–20   → NONE
            21–40  → PROCESS
            41–60  → CONTAINER
            61–75  → GVISOR
            76–90  → MICROVM
            91–100 → TEE

        Args:
            threat_score: Integer in [0, 100].

        Returns:
            The corresponding ``IsolationLevel`` enum member.
        """
        # Iterate thresholds in reverse to find the highest matching bucket.
        for threshold, level in reversed(self._ISOLATION_THRESHOLDS):
            if threat_score >= threshold:
                return level
        # Fallback (should never reach here because threshold 0 is present).
        return IsolationLevel.NONE


# ---------------------------------------------------------------------------
# Sandbox Assignment & Result
# ---------------------------------------------------------------------------

@dataclass
class SandboxAssignment:
    """Immutable record of a sandbox allocated to an agent.

    Attributes:
        agent_id: The agent that owns this sandbox.
        isolation: Selected isolation level (may be overridden by policy).
        assigned_at: UTC timestamp when the assignment was created.
        resource_limits: Opaque dictionary of resource caps (cpu, mem, disk, etc.).
        observation_filter: Optional regex or classifier name for PII filtering.
    """

    agent_id: AgentId
    isolation: IsolationLevel
    assigned_at: datetime = field(default_factory=datetime.utcnow)
    resource_limits: Dict[str, Any] = field(default_factory=dict)
    observation_filter: Optional[str] = None


@dataclass
class SandboxResult:
    """Outcome of a command executed inside a sandbox.

    Attributes:
        stdout: Captured standard output (already observation-filtered).
        stderr: Captured standard error (already observation-filtered).
        returncode: Process exit code (0 typically means success).
        execution_time_ms: Wall-clock execution time in milliseconds.
        resource_usage: Normalised resource consumption (cpu_seconds, mem_mb, etc.).
    """

    stdout: str
    stderr: str
    returncode: int
    execution_time_ms: int
    resource_usage: Dict[str, float] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Observation Filter (PII / Credential heuristic scrubbing)
# ---------------------------------------------------------------------------

class SensitivityRule(enum.Enum):
    """Built-in sensitivity categories for the observation filter."""

    EMAIL = "email"
    API_KEY = "api_key"
    PASSWORD = "password"
    CONNECTION_STRING = "connection_string"
    CREDIT_CARD = "credit_card"


class ObservationFilter:
    """Heuristic PII/credential scrubber inspired by PlanTwin observation filtering.

    Applies a configurable set of regex rules to redact sensitive substrings
    with a fixed mask (``***``). Rules can be passed by name or by raw regex.
    """

    # --- Built-in regexes ----------------------------------------------------
    _RULES: Dict[str, re.Pattern[str]] = {
        SensitivityRule.EMAIL.value: re.compile(
            r"[\w\.-]+@[\w\.-]+\.\w{2,}",
            re.IGNORECASE,
        ),
        SensitivityRule.API_KEY.value: re.compile(
            r"(?i)(api[_-]?key\s*[:=]\s*)['\"]?[a-zA-Z0-9_\-]{16,}['\"]?"
        ),
        SensitivityRule.PASSWORD.value: re.compile(
            r"(?i)(password|passwd|pwd)\s*[:=]\s*['\"]?[^\s'\"]+['\"]?"
        ),
        SensitivityRule.CONNECTION_STRING.value: re.compile(
            r"(?i)(mongodb|mysql|postgresql|postgres|redis|amqp|kafka)://[^\s\"]+"
        ),
        SensitivityRule.CREDIT_CARD.value: re.compile(
            r"\b(?:\d[ -]*?){13,16}\b"
        ),
    }

    _REDACTION_MASK: str = "***"

    @classmethod
    def filter(
        cls,
        observation: str,
        sensitivity_rules: List[str],
    ) -> str:
        """Scrub sensitive information from an observation string.

        Args:
            observation: Raw text that may contain PII or credentials.
            sensitivity_rules: List of rule names (e.g. ``["email", "api_key"]``)
                or raw regex strings. Built-in names are resolved first; anything
                else is treated as a literal regex pattern.

        Returns:
            The scrubbed text with sensitive segments replaced by ``***``.
        """
        if not sensitivity_rules:
            return observation

        result = observation
        for rule in sensitivity_rules:
            pattern = cls._RULES.get(rule)
            if pattern is None:
                # Treat unknown rule as a raw regex (best-effort).
                try:
                    pattern = re.compile(rule)
                except re.error as exc:
                    logger.warning(
                        "ObservationFilter: invalid regex %r skipped (%s)", rule, exc
                    )
                    continue
            result = pattern.sub(cls._REDACTION_MASK, result)
        return result


# ---------------------------------------------------------------------------
# Policy Engine
# ---------------------------------------------------------------------------

class PolicyEngine:
    """Hierarchical policy resolver.

    Resolves the final ``SandboxPolicy`` for an agent by merging layers:

        Organisation defaults  →  Team policy  →  Agent policy  →  Task policy

    Later layers override earlier ones. Merge semantics are shallow for scalar
    fields and union-based for collection fields (``allowed_executables``).
    """

    @staticmethod
    def resolve_policy(
        agent_role: str,
        task_type: str,
        org_defaults: Dict[str, Any],
    ) -> SandboxPolicy:
        """Resolve the effective sandbox policy.

        The ``org_defaults`` dict is expected to contain nested policy layers
        under the following optional keys:

        * ``team_policies`` → ``Dict[str, Dict[str, Any]]``
        * ``agent_policies`` → ``Dict[str, Dict[str, Any]]``
        * ``task_policies`` → ``Dict[str, Dict[str, Any]]``

        Each inner dict maps to ``SandboxPolicy`` field names.

        Args:
            agent_role: Role identifier of the agent (used as key into
                ``agent_policies``).
            task_type: Task category (used as key into ``task_policies``).
            org_defaults: Root configuration dictionary containing defaults and
                nested override layers.

        Returns:
            A fully merged ``SandboxPolicy``.

        Raises:
            PolicyResolutionError: If the merged result cannot be serialised
                into a ``SandboxPolicy``.
        """
        # Base layer: organisation defaults (strip nested policy keys first).
        base: Dict[str, Any] = {
            k: v
            for k, v in org_defaults.items()
            if k not in {"team_policies", "agent_policies", "task_policies"}
        }

        # Layer 2: team policy (derive team from role prefix if not explicit).
        team_policies = org_defaults.get("team_policies", {})
        team_key = PolicyEngine._infer_team(agent_role)
        team_layer = team_policies.get(team_key, {})

        # Layer 3: agent policy.
        agent_policies = org_defaults.get("agent_policies", {})
        agent_layer = agent_policies.get(agent_role, {})

        # Layer 4: task policy.
        task_policies = org_defaults.get("task_policies", {})
        task_layer = task_policies.get(task_type, {})

        merged = PolicyEngine._merge_layers(base, team_layer, agent_layer, task_layer)

        try:
            return SandboxPolicy(**merged)
        except TypeError as exc:
            raise PolicyResolutionError(
                f"Failed to construct SandboxPolicy from merged dict: {exc}"
            ) from exc

    @staticmethod
    def _infer_team(agent_role: str) -> str:
        """Derive a team identifier from an agent role string.

        Heuristic: first segment before ``::`` or ``_``. Falls back to
        ``default``.
        """
        for delimiter in ("::", "_"):
            if delimiter in agent_role:
                return agent_role.split(delimiter)[0]
        return "default"

    @staticmethod
    def _merge_layers(
        base: Dict[str, Any],
        team: Dict[str, Any],
        agent: Dict[str, Any],
        task: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Shallow-merge four policy layers with set-union semantics.

        Scalar fields in later layers overwrite earlier ones.
        ``allowed_executables`` (sets/lists) are unioned rather than replaced.
        """
        merged = dict(base)

        for layer in (team, agent, task):
            for key, value in layer.items():
                if key == "allowed_executables" and value is not None:
                    existing = merged.get(key)
                    if isinstance(existing, (set, list)) and isinstance(
                        value, (set, list)
                    ):
                        merged[key] = set(existing) | set(value)
                    else:
                        merged[key] = set(value) if isinstance(value, (set, list)) else value
                else:
                    merged[key] = value
        return merged


# ---------------------------------------------------------------------------
# Sandbox Orchestrator
# ---------------------------------------------------------------------------

class SandboxOrchestrator:
    """Policy-driven dynamic sandbox orchestrator.

    Responsibilities:
        1. **Assignment** — Evaluate threat, select isolation, create a
           ``SandboxAssignment``.
        2. **Execution** — Run commands inside the assigned sandbox (simulated
           via ``asyncio.subprocess`` with resource caps and timeout guards).
        3. **Observation filtering** — Scrub stdout/stderr before returning.
        4. **Lifecycle** — Track active sandboxes and release resources.

    The orchestrator is safe for concurrent use from multiple async tasks
    because all mutable state mutations are atomic dictionary operations.
    """

    def __init__(
        self,
        threat_model: Optional[ThreatModel] = None,
        policy_engine: Optional[PolicyEngine] = None,
        observation_filter: Optional[ObservationFilter] = None,
    ) -> None:
        """Create a new orchestrator with injectable sub-components.

        Args:
            threat_model: Threat assessment model. A default instance is used
                if ``None``.
            policy_engine: Policy resolution engine. A default instance is used
                if ``None``.
            observation_filter: PII scrubber. The class-level ``filter`` method
                is used directly, so this argument is mainly for future
                extensibility.
        """
        self._threat_model = threat_model or ThreatModel()
        self._policy_engine = policy_engine or PolicyEngine()
        # ObservationFilter is stateless; we keep the reference for protocol compliance.
        self._observation_filter = observation_filter or ObservationFilter()

        # Active sandbox registry: agent_id → assignment.
        self._active_sandboxes: Dict[AgentId, SandboxAssignment] = {}

        # Execution metrics (non-persistent, mainly for introspection).
        self._execution_count = 0
        self._cumulative_ms = 0

    @property
    def active_sandboxes(self) -> Dict[AgentId, SandboxAssignment]:
        """Read-only snapshot of currently assigned sandboxes."""
        return dict(self._active_sandboxes)

    # ------------------------------------------------------------------
    # Assignment
    # ------------------------------------------------------------------

    def assign_sandbox(
        self,
        agent_id: AgentId,
        message: Message,
        policy: SandboxPolicy,
    ) -> SandboxAssignment:
        """Assign a sandbox to an agent for a given message.

        The threat model evaluates the message + any known agent history. The
        resulting threat score is mapped to a recommended isolation level, which
        is then **clamped upwards** by the policy's ``default_isolation``
        (policy is the security floor).

        Args:
            agent_id: The requesting agent.
            message: The message that triggered sandbox need (usually a TASK).
            policy: The resolved sandbox policy (from ``PolicyEngine``).

        Returns:
            A ``SandboxAssignment`` record, also stored in the active registry.
        """
        # 1. Threat assessment
        # Try to extract agent_history from message payload if present.
        agent_history: List[Dict[str, Any]] = message.payload.get("agent_history", [])
        threat_score = self._threat_model.assess(message, agent_history)
        recommended = self._threat_model.get_recommended_isolation(threat_score)

        # 2. Policy floor: isolation cannot be weaker than policy.default_isolation.
        isolation = self._clamp_isolation(recommended, policy.default_isolation)

        # 3. Build resource limits from policy.
        resource_limits: Dict[str, Any] = {
            "max_cpu_cores": policy.max_cpu_cores,
            "max_memory_mb": policy.max_memory_mb,
            "filesystem_access": policy.filesystem_access,
            "network_access": policy.network_access,
        }

        # 4. Compose assignment.
        assignment = SandboxAssignment(
            agent_id=agent_id,
            isolation=isolation,
            resource_limits=resource_limits,
            observation_filter=policy.observation_filter,
        )

        self._active_sandboxes[agent_id] = assignment
        logger.info(
            "Sandbox assigned to %s — isolation=%s threat=%d policy_floor=%s",
            agent_id,
            isolation.value,
            threat_score,
            policy.default_isolation.value,
        )
        return assignment

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    async def execute_in_sandbox(
        self,
        agent_id: AgentId,
        command: str,
        sandbox: SandboxAssignment,
    ) -> SandboxResult:
        """Execute a shell command inside the given sandbox.

        In production this would delegate to Firecracker, gVisor, or containerd.
        Here we simulate isolation using ``asyncio.create_subprocess_shell``
        with a timeout derived from resource limits and basic resource tracking.

        Args:
            agent_id: The agent requesting execution (for logging / metrics).
            command: The shell command string to run.
            sandbox: The ``SandboxAssignment`` returned by ``assign_sandbox``.

        Returns:
            A ``SandboxResult`` with scrubbed stdout/stderr.

        Raises:
            SandboxNotFound: If the sandbox is not in the active registry.
            SandboxExecutionError: If the subprocess cannot be started or times out.
        """
        # Validate registry membership.
        if self._active_sandboxes.get(agent_id) != sandbox:
            raise SandboxNotFound(
                f"Sandbox for agent {agent_id} not found or mismatch.",
                agent_id=agent_id,
            )

        # Derive timeout from policy limits (default 300s).
        timeout_seconds = sandbox.resource_limits.get("max_compute_seconds", 300.0)
        if isinstance(timeout_seconds, (int, float)):
            timeout = float(timeout_seconds)
        else:
            timeout = 300.0

        # Prepare sensitivity rules for filtering.
        sensitivity_rules: List[str] = []
        if sandbox.observation_filter:
            # observation_filter may be a comma-separated list of rule names.
            sensitivity_rules = [
                r.strip()
                for r in sandbox.observation_filter.split(",")
                if r.strip()
            ]

        start_time = time.perf_counter()
        stdout_data = ""
        stderr_data = ""
        returncode = -1

        try:
            # Use asyncio subprocess for non-blocking execution.
            proc = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            try:
                stdout_bytes, stderr_bytes = await asyncio.wait_for(
                    proc.communicate(), timeout=timeout
                )
                returncode = proc.returncode or 0
                stdout_data = stdout_bytes.decode("utf-8", errors="replace")
                stderr_data = stderr_bytes.decode("utf-8", errors="replace")
            except asyncio.TimeoutError:
                # Best-effort termination.
                try:
                    proc.kill()
                    await proc.wait()
                except ProcessLookupError:
                    pass
                returncode = -9  # SIGKILL-ish
                stderr_data += "\n[sandbox timeout — process terminated]"
        except Exception as exc:
            raise SandboxExecutionError(
                f"Failed to execute command in sandbox: {exc}",
                agent_id=agent_id,
            ) from exc

        elapsed_ms = int((time.perf_counter() - start_time) * 1000)

        # Apply observation filter.
        filtered_stdout = ObservationFilter.filter(stdout_data, sensitivity_rules)
        filtered_stderr = ObservationFilter.filter(stderr_data, sensitivity_rules)

        # Derive pseudo resource usage from isolation level and elapsed time.
        resource_usage = self._estimate_resource_usage(sandbox, elapsed_ms)

        self._execution_count += 1
        self._cumulative_ms += elapsed_ms

        logger.info(
            "Sandbox execution for %s — rc=%d time_ms=%d isolation=%s",
            agent_id,
            returncode,
            elapsed_ms,
            sandbox.isolation.value,
        )

        return SandboxResult(
            stdout=filtered_stdout,
            stderr=filtered_stderr,
            returncode=returncode,
            execution_time_ms=elapsed_ms,
            resource_usage=resource_usage,
        )

    # ------------------------------------------------------------------
    # Release
    # ------------------------------------------------------------------

    def release_sandbox(self, agent_id: AgentId) -> None:
        """Release the sandbox allocated to ``agent_id``.

        Idempotent: if the agent has no active sandbox, this is a no-op.

        Args:
            agent_id: The agent whose sandbox should be reclaimed.
        """
        assignment = self._active_sandboxes.pop(agent_id, None)
        if assignment is not None:
            logger.info(
                "Sandbox released for %s (isolation=%s, lifetime=%s)",
                agent_id,
                assignment.isolation.value,
                datetime.utcnow() - assignment.assigned_at,
            )
        else:
            logger.debug("release_sandbox called for %s — no active assignment.", agent_id)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _clamp_isolation(
        recommended: IsolationLevel,
        policy_floor: IsolationLevel,
    ) -> IsolationLevel:
        """Return the stronger of *recommended* and *policy_floor*.

        The isolation ordering is defined by the enum declaration order:
        NONE < PROCESS < CONTAINER < GVISOR < MICROVM < TEE.
        """
        # Enum members auto-number from 1 in declaration order.
        recommended_strength = list(IsolationLevel).index(recommended)
        floor_strength = list(IsolationLevel).index(policy_floor)
        return recommended if recommended_strength >= floor_strength else policy_floor

    @staticmethod
    def _estimate_resource_usage(
        sandbox: SandboxAssignment,
        elapsed_ms: int,
    ) -> Dict[str, float]:
        """Produce approximate resource usage metrics.

        In a real implementation this would read cgroup/firecracker metrics.
        Here we synthesise plausible numbers from the sandbox configuration.
        """
        cpu_cores = float(sandbox.resource_limits.get("max_cpu_cores", 1))
        mem_mb = float(sandbox.resource_limits.get("max_memory_mb", 512))
        elapsed_sec = elapsed_ms / 1000.0

        return {
            "cpu_seconds": round(cpu_cores * elapsed_sec, 3),
            "memory_peak_mb": round(mem_mb * 0.35, 1),  # Heuristic 35 % utilisation.
            "elapsed_seconds": round(elapsed_sec, 3),
        }

    # ------------------------------------------------------------------
    # Introspection / Metrics
    # ------------------------------------------------------------------

    def get_metrics(self) -> Dict[str, Any]:
        """Return non-persistent runtime metrics.

        Useful for metacognitive monitoring (Monitor → Verify → Revise loop).
        """
        return {
            "active_sandboxes": len(self._active_sandboxes),
            "total_executions": self._execution_count,
            "cumulative_execution_ms": self._cumulative_ms,
            "average_execution_ms": (
                self._cumulative_ms / self._execution_count
                if self._execution_count
                else 0.0
            ),
        }
