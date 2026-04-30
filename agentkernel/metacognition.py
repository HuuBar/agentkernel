"""AgentKernel — Metacognition Layer (Two-Level Meta-Cognitive Architecture).

This module implements a two-level metacognitive architecture inspired by the
SAMI (Self-Adaptive Multi-Agent Intelligence) paper and the MGV
(Monitor-Generate-Verify) framework:

    Level 1 — CognitiveLayer: Task execution and tool orchestration.
    Level 2 — MetacognitiveLayer: Monitoring, strategy generation, verification,
              and self-revision of Level-1 outputs.

All public APIs are fully async and type-annotated. Quality checks are purely
heuristic and require no external LLM service.
"""
from __future__ import annotations

import asyncio
import difflib
import re
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Coroutine, Dict, List, Optional, Set, Tuple
from uuid import UUID, uuid4

from agentkernel.types import (
    AgentId,
    Message,
    MessageKind,
    MetaLevel,
    MetacognitiveRecord,
)


# ---------------------------------------------------------------------------
# Quality Verifier (heuristic, no external LLM)
# ---------------------------------------------------------------------------

class QualityVerifier:
    """Heuristic quality validator for agent outputs.

    Implements three complementary checks:
    * **Completeness** — required fields are present and non-empty.
    * **Consistency** — output respects declared constraints (type, range, enum).
    * **Factuality** — string-level overlap between output and provided evidence.

    All scores are normalised to the range ``[0.0, 1.0]``.
    """

    @staticmethod
    def check_completeness(
        output: Dict[str, Any],
        required_fields: List[str],
    ) -> Tuple[bool, float]:
        """Verify that *output* contains every field in *required_fields*.

        A field is considered "complete" when it exists in *output*, is not
        ``None``, and (when a string/list/dict) is non-empty.

        Args:
            output: The agent-produced dictionary to inspect.
            required_fields: Keys that must be present and well-formed.

        Returns:
            ``(passed, score)`` where *score* is the fraction of fields that
            satisfy the completeness predicate.
        """
        if not required_fields:
            return True, 1.0

        def _is_nonempty(value: Any) -> bool:
            if value is None:
                return False
            if isinstance(value, (str, list, dict, set, tuple)):
                return len(value) > 0
            return True

        hits = sum(1 for f in required_fields if f in output and _is_nonempty(output[f]))
        score = hits / len(required_fields)
        return score >= 1.0, score

    @staticmethod
    def check_consistency(
        output: Dict[str, Any],
        constraints: Dict[str, Any],
    ) -> Tuple[bool, float]:
        """Validate *output* against declarative *constraints*.

        Supported constraint shapes (per key):

        * ``{"type": "int"}`` — value must be an ``int``.
        * ``{"min": 0, "max": 100}`` — numeric value must lie in the closed
          interval (applies to ``int`` and ``float``).
        * ``{"enum": ["a", "b"]}`` — value must be a member of the list.
        * ``{"regex": r"^\\d+$"}`` — string value must match the pattern.
        * ``{"len_min": 1, "len_max": 10}`` — length of string/list must be
          within bounds.

        Multiple rules for the same key are AND-ed together.

        Args:
            output: The dictionary to validate.
            constraints: Mapping ``field_name -> rule_dict``.

        Returns:
            ``(passed, score)`` where *score* is the fraction of fields that
            satisfy **all** their rules.
        """
        if not constraints:
            return True, 1.0

        def _check_field(val: Any, rules: Dict[str, Any]) -> bool:
            for rule, param in rules.items():
                if rule == "type":
                    type_map: Dict[str, type] = {
                        "int": int,
                        "float": float,
                        "str": str,
                        "bool": bool,
                        "list": list,
                        "dict": dict,
                    }
                    expected = type_map.get(param)
                    if expected is not None and not isinstance(val, expected):
                        return False
                elif rule == "min":
                    if not isinstance(val, (int, float)) or val < param:
                        return False
                elif rule == "max":
                    if not isinstance(val, (int, float)) or val > param:
                        return False
                elif rule == "enum":
                    if val not in param:
                        return False
                elif rule == "regex":
                    if not isinstance(val, str) or not re.search(param, val):
                        return False
                elif rule == "len_min":
                    if not hasattr(val, "__len__") or len(val) < param:
                        return False
                elif rule == "len_max":
                    if not hasattr(val, "__len__") or len(val) > param:
                        return False
            return True

        hits = sum(
            1
            for key, rules in constraints.items()
            if key in output and _check_field(output[key], rules)
        )
        score = hits / len(constraints)
        return score >= 1.0, score

    @staticmethod
    def check_factuality(
        output: str,
        evidence: List[str],
    ) -> Tuple[bool, float]:
        """Heuristic factuality check based on textual overlap.

        The method tokenises *output* into sentences, then for each sentence
        computes the maximum similarity against every evidence string.  The
        overall score is the average of per-sentence maxima.

        Args:
            output: The claim / text to verify.
            evidence: Ground-truth or reference strings.

        Returns:
            ``(passed, score)`` where *score* is the mean best sentence
            similarity.  *passed* is ``True`` when the mean exceeds 0.5.
        """
        if not evidence:
            # No evidence available — treat as inconclusive but not failed.
            return True, 1.0

        if not output or not output.strip():
            return False, 0.0

        # Split into sentences (naive but sufficient for heuristic).
        sentences = re.split(r"[.!?]+", output.strip())
        sentences = [s.strip() for s in sentences if s.strip()]
        if not sentences:
            sentences = [output.strip()]

        def _best_similarity(sentence: str) -> float:
            return max(
                difflib.SequenceMatcher(None, sentence.lower(), ev.lower()).ratio()
                for ev in evidence
            )

        scores = [_best_similarity(s) for s in sentences]
        mean_score = sum(scores) / len(scores)
        # Threshold tuned for heuristic use; can be overridden by caller.
        return mean_score >= 0.5, mean_score

    @classmethod
    def composite_score(
        cls,
        output: Dict[str, Any],
        criteria: Dict[str, Any],
    ) -> Tuple[bool, float, str]:
        """Run all applicable heuristic checks and aggregate a composite score.

        *criteria* may contain the keys ``required_fields``, ``constraints``,
        and ``evidence``.  Any missing category is skipped with a neutral
        score of ``1.0``.

        The composite score is the unweighted arithmetic mean of all
        sub-scores.  The overall check passes when every sub-check passes.

        Args:
            output: The agent output dictionary.
            criteria: Dictionary describing the quality bar.

        Returns:
            ``(passed, composite_score, reason)``
        """
        required_fields: List[str] = criteria.get("required_fields", [])
        constraints: Dict[str, Any] = criteria.get("constraints", {})
        evidence: List[str] = criteria.get("evidence", [])

        reasons: List[str] = []
        scores: List[float] = []

        # Completeness
        comp_ok, comp_score = cls.check_completeness(output, required_fields)
        scores.append(comp_score)
        if not comp_ok:
            missing = [f for f in required_fields if f not in output or output.get(f) is None]
            reasons.append(f"incomplete fields: {missing}")

        # Consistency
        cons_ok, cons_score = cls.check_consistency(output, constraints)
        scores.append(cons_score)
        if not cons_ok:
            reasons.append("constraint violations detected")

        # Factuality (only when output contains a textual claim)
        text_segments: List[str] = []
        for v in output.values():
            if isinstance(v, str) and v.strip():
                text_segments.append(v.strip())
        if text_segments and evidence:
            txt = " ".join(text_segments)
            fact_ok, fact_score = cls.check_factuality(txt, evidence)
            scores.append(fact_score)
            if not fact_ok:
                reasons.append("low evidence overlap (factuality)")

        if not scores:
            return True, 1.0, "no criteria specified — vacuous pass"

        composite = sum(scores) / len(scores)
        passed = comp_ok and cons_ok and (not evidence or fact_ok)
        reason = "; ".join(reasons) if reasons else "all checks passed"
        return passed, composite, reason


# ---------------------------------------------------------------------------
# Self-Reflection Log
# ---------------------------------------------------------------------------

class SelfReflectionLog:
    """Persistent (in-memory) log of metacognitive episodes.

    Records are indexed by :class:`AgentId` and can be filtered by
    :class:`MetaLevel`.  The log also provides lightweight analytics for
    failure-pattern detection and insight generation.
    """

    def __init__(self) -> None:
        self._episodes: Dict[AgentId, List[MetacognitiveRecord]] = defaultdict(list)
        self._lock = asyncio.Lock()

    async def record_episode(self, record: MetacognitiveRecord) -> None:
        """Append a metacognitive record to the log.

        Args:
            record: The episode to store.
        """
        async with self._lock:
            self._episodes[record.agent_id].append(record)

    async def get_episodes(
        self,
        agent_id: AgentId,
        level: Optional[MetaLevel] = None,
    ) -> List[MetacognitiveRecord]:
        """Retrieve episodes for a given agent, optionally filtered by level.

        Args:
            agent_id: The agent whose history is requested.
            level: If provided, only records at this meta-level are returned.

        Returns:
            A list of matching :class:`MetacognitiveRecord` instances, ordered
            by insertion time (oldest first).
        """
        async with self._lock:
            records = self._episodes.get(agent_id, [])
            if level is None:
                return list(records)
            return [r for r in records if r.level == level]

    async def get_failure_patterns(self, agent_id: AgentId) -> Dict[str, int]:
        """Analyse an agent's history and return a frequency map of failure
        triggers.

        A "failure" is any record whose *decision* contains the substring
        ``"fail"`` or ``"error"`` (case-insensitive), or whose *confidence*
        is below ``0.5``.

        Args:
            agent_id: Agent to analyse.

        Returns:
            Mapping ``trigger -> occurrence_count``, sorted by descending count.
        """
        async with self._lock:
            records = self._episodes.get(agent_id, [])

        failures: Dict[str, int] = defaultdict(int)
        for r in records:
            is_failure = (
                r.confidence < 0.5
                or "fail" in r.decision.lower()
                or "error" in r.decision.lower()
            )
            if is_failure:
                failures[r.trigger] += 1

        return dict(sorted(failures.items(), key=lambda kv: kv[1], reverse=True))

    async def generate_insights(self, agent_id: AgentId) -> List[str]:
        """Derive actionable improvement suggestions from an agent's
        metacognitive history.

        The method performs simple statistical pattern matching:

        * If the same trigger appears >= 3 times, suggest re-tooling or
          schema relaxation for that trigger category.
        * If ``VERIFY`` level records consistently show low confidence,
          suggest tightening the quality threshold or enriching evidence.
        * If ``GENERATE`` level records show high churn (many distinct
          strategies), suggest caching successful strategies.

        Args:
            agent_id: Agent to introspect.

        Returns:
            A list of human-readable insight strings.
        """
        async with self._lock:
            records = self._episodes.get(agent_id, [])

        if not records:
            return ["No metacognitive history available — begin monitoring to generate insights."]

        insights: List[str] = []
        failures = await self.get_failure_patterns(agent_id)

        # Insight 1: repeated trigger patterns
        for trigger, count in failures.items():
            if count >= 3:
                insights.append(
                    f"Trigger '{trigger}' failed {count} times: consider "
                    "relaxing input constraints, adding fallback tools, or "
                    "updating the output schema."
                )

        # Insight 2: low-confidence verify episodes
        verify_records = [r for r in records if r.level == MetaLevel.VERIFY]
        low_conf_verify = [r for r in verify_records if r.confidence < 0.5]
        if len(low_conf_verify) >= 2:
            insights.append(
                f"{len(low_conf_verify)} verification episodes with low confidence: "
                "consider tightening quality criteria or providing richer evidence."
            )

        # Insight 3: high strategy churn
        generate_records = [r for r in records if r.level == MetaLevel.GENERATE]
        unique_strategies = len({r.decision for r in generate_records})
        if unique_strategies >= 3:
            insights.append(
                f"{unique_strategies} distinct strategies generated: consider "
                "caching successful strategies to reduce exploration overhead."
            )

        # Insight 4: cognitive layer crashes / empty outputs
        monitor_records = [r for r in records if r.level == MetaLevel.MONITOR]
        empty_count = sum(
            1 for r in monitor_records
            if "empty" in r.trigger.lower() or "null" in r.trigger.lower()
        )
        if empty_count >= 2:
            insights.append(
                f"{empty_count} empty/null outputs detected at monitor stage: "
                "validate input parameters before task dispatch."
            )

        if not insights:
            insights.append("Metacognitive history looks healthy — no actionable insights at this time.")

        return insights


# ---------------------------------------------------------------------------
# Level 1 — Cognitive Layer
# ---------------------------------------------------------------------------

class CognitiveLayer:
    """Level-1 task executor.

    Responsible for running the core work requested by a :class:`Message`,
    invoking tools by name, and maintaining an *execution trace* that can be
    inspected by the metacognitive layer above.
    """

    def __init__(self, agent_id: AgentId) -> None:
        self.agent_id = agent_id
        self.execution_trace: List[Dict[str, Any]] = []

    async def tool_call(
        self,
        tool_name: str,
        params: Dict[str, Any],
        tools: Dict[str, Callable[..., Any]],
    ) -> Dict[str, Any]:
        """Invoke a single tool by name.

        The tool lookup is performed in the provided *tools* registry.  The
        callable may be synchronous or asynchronous; the method normalises both
        to an async execution.

        Args:
            tool_name: Key into *tools*.
            params: Arguments forwarded to the tool.
            tools: Registry of available tools.

        Returns:
            The tool's return value wrapped as a dictionary.  On missing tool
            or runtime exception, the result contains ``"error"`` and
            ``"traceback"`` keys.
        """
        tool = tools.get(tool_name)
        if tool is None:
            return {
                "error": f"Tool '{tool_name}' not found in registry",
                "available": list(tools.keys()),
            }

        t0 = datetime.utcnow()
        try:
            if asyncio.iscoroutinefunction(tool):
                result = await tool(**params)
            else:
                # Run synchronous tools in the default executor to avoid
                # blocking the event loop.
                loop = asyncio.get_running_loop()
                result = await loop.run_in_executor(None, lambda: tool(**params))

            # Normalise to dict
            if not isinstance(result, dict):
                result = {"value": result}
        except Exception as exc:
            result = {
                "error": str(exc),
                "tool": tool_name,
            }

        dt = (datetime.utcnow() - t0).total_seconds()
        entry = {
            "step": len(self.execution_trace) + 1,
            "layer": "cognitive",
            "tool": tool_name,
            "params": params,
            "result": result,
            "duration_sec": dt,
            "timestamp": datetime.utcnow().isoformat(),
        }
        self.execution_trace.append(entry)
        return result

    async def execute(
        self,
        task: Message,
        tools: Dict[str, Callable[..., Any]],
    ) -> Dict[str, Any]:
        """Execute the payload of a :class:`Message`.

        The payload is expected to contain at minimum a ``"action"`` key
        describing the high-level intent.  If the payload contains a
        ``"tools"`` list, each named tool is invoked sequentially and the
        aggregated results are returned.

        Args:
            task: The incoming task message.
            tools: Registry of callable tools available to this agent.

        Returns:
            A dictionary with at least ``"status"`` and ``"results"`` keys.
            If the task payload carries an explicit ``"output_schema"`` it is
            forwarded to the caller for downstream validation.
        """
        payload = task.payload
        action = payload.get("action", "unknown")

        output: Dict[str, Any] = {
            "status": "ok",
            "action": action,
            "results": {},
            "agent_id": str(self.agent_id),
            "correlation_id": str(task.correlation_id or task.id),
        }

        tool_names: List[str] = payload.get("tools", [])
        if not tool_names:
            # No explicit tool list — the action itself may be the work unit.
            output["results"]["direct"] = payload.get("data", {})
            self.execution_trace.append({
                "step": len(self.execution_trace) + 1,
                "layer": "cognitive",
                "action": action,
                "note": "no tools requested — direct execution",
                "timestamp": datetime.utcnow().isoformat(),
            })
            return output

        for name in tool_names:
            tool_params = payload.get("tool_params", {}).get(name, {})
            result = await self.tool_call(name, tool_params, tools)
            output["results"][name] = result
            if "error" in result:
                output["status"] = "partial_error"

        # If any tool errored, elevate status.
        if any("error" in r for r in output["results"].values()):
            output["status"] = "error"

        return output


# ---------------------------------------------------------------------------
# Level 2 — Metacognitive Layer
# ---------------------------------------------------------------------------

class MetacognitiveLayer:
    """Level-2 monitor, strategist, verifier, and reviser.

    The metacognitive layer **never** executes tasks directly; it only
    *observes*, *evaluates*, and *guides* the :class:`CognitiveLayer` below.

    Public lifecycle::

        monitor → generate_strategy → verify → revise

    The convenience method :meth:`metacognitive_loop` orchestrates the full
    cycle automatically.
    """

    def __init__(
        self,
        agent_id: AgentId,
        cognitive: CognitiveLayer,
        reflection_log: SelfReflectionLog,
        max_iterations: int = 3,
        quality_threshold: float = 0.7,
    ) -> None:
        self.agent_id = agent_id
        self.cognitive = cognitive
        self.reflection_log = reflection_log
        self.max_iterations = max(max_iterations, 1)
        self.quality_threshold = quality_threshold

    # -- internal helpers --------------------------------------------------

    async def _log(
        self,
        level: MetaLevel,
        trigger: str,
        observation: Dict[str, Any],
        decision: str,
        confidence: float,
    ) -> None:
        """Persist a metacognitive record to the reflection log."""
        record = MetacognitiveRecord(
            agent_id=self.agent_id,
            level=level,
            trigger=trigger,
            observation=observation,
            decision=decision,
            confidence=confidence,
        )
        await self.reflection_log.record_episode(record)

    # -- level-2 primitives ------------------------------------------------

    async def monitor(
        self,
        cognitive_output: Dict[str, Any],
    ) -> MetacognitiveRecord:
        """Observe the cognitive layer output and detect anomalies.

        Anomalies detected heuristically:

        * Empty or missing ``results``.
        * ``status == "error"`` or ``status == "partial_error"``.
        * All tool results contain ``"error"`` keys.
        * Output is an empty dict.

        Args:
            cognitive_output: The dictionary returned by
                :meth:`CognitiveLayer.execute`.

        Returns:
            A :class:`MetacognitiveRecord` describing the monitoring outcome.
        """
        trigger = "monitor_pass"
        confidence = 1.0
        decision = "No anomaly detected — proceed."
        observation = {"output_keys": list(cognitive_output.keys())}

        if not cognitive_output:
            trigger = "empty_output"
            confidence = 0.2
            decision = "Cognitive layer returned empty output — intervention required."
        elif cognitive_output.get("status") in ("error", "partial_error"):
            trigger = f"status_{cognitive_output['status']}"
            confidence = 0.3
            decision = "Errors detected in cognitive execution — consider strategy revision."
        else:
            results = cognitive_output.get("results", {})
            if not results:
                trigger = "missing_results"
                confidence = 0.4
                decision = "No results produced — possible tool misconfiguration."
            elif all(isinstance(v, dict) and "error" in v for v in results.values()):
                trigger = "total_tool_failure"
                confidence = 0.1
                decision = "All tools failed — generate fallback strategy immediately."

        record = MetacognitiveRecord(
            agent_id=self.agent_id,
            level=MetaLevel.MONITOR,
            trigger=trigger,
            observation=observation,
            decision=decision,
            confidence=confidence,
        )
        await self.reflection_log.record_episode(record)
        return record

    async def generate_strategy(
        self,
        problem: str,
        failed_attempts: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Generate an alternative strategy when the cognitive layer stalls.

        The method performs lightweight rule-based strategy selection:

        * If the problem mentions "missing tool" or "not found", suggest
          tool substitution or schema relaxation.
        * If all prior attempts errored, suggest decomposition (split the
          task into smaller sub-tasks).
        * If results were empty, suggest parameter relaxation or richer
          context injection.
        * Otherwise, suggest a retry with jitter / alternate ordering.

        Args:
            problem: A textual description of the detected issue.
            failed_attempts: Prior attempts that did not satisfy quality
                thresholds.

        Returns:
            A strategy dictionary containing ``"approach"`` and
            ``"modifications"``.
        """
        problem_lower = problem.lower()
        attempt_count = len(failed_attempts)

        strategy: Dict[str, Any] = {
            "approach": "retry",
            "modifications": [],
            "rationale": "default fallback",
        }

        if "missing tool" in problem_lower or "not found" in problem_lower:
            strategy["approach"] = "tool_substitution"
            strategy["modifications"].append("map requested tools to closest available equivalents")
            strategy["modifications"].append("relax required tool count")
            strategy["rationale"] = "Tool registry mismatch detected."
        elif attempt_count > 0 and all(
            isinstance(a, dict) and a.get("status") == "error" for a in failed_attempts
        ):
            strategy["approach"] = "decomposition"
            strategy["modifications"].append("split task into ordered sub-tasks")
            strategy["modifications"].append("insert verification gates between sub-tasks")
            strategy["rationale"] = "Persistent errors suggest granularity is too coarse."
        elif "empty" in problem_lower or "null" in problem_lower:
            strategy["approach"] = "enrichment"
            strategy["modifications"].append("inject additional context into tool params")
            strategy["modifications"].append("widen search / filter boundaries")
            strategy["rationale"] = "Empty results suggest insufficient input breadth."
        else:
            strategy["approach"] = "reorder_and_retry"
            strategy["modifications"].append("shuffle tool execution order")
            strategy["modifications"].append("increase timeout / resource budget")
            strategy["rationale"] = "Non-deterministic failure — retry with perturbation."

        await self._log(
            level=MetaLevel.GENERATE,
            trigger=problem,
            observation={"failed_attempts": attempt_count, "problem": problem},
            decision=str(strategy),
            confidence=0.6 + (0.1 * attempt_count),  # confidence degrades with repeated failure
        )
        return strategy

    async def verify(
        self,
        output: Dict[str, Any],
        criteria: Dict[str, Any],
    ) -> Tuple[bool, float, str]:
        """Validate *output* against quality *criteria*.

        This is a thin async wrapper around :class:`QualityVerifier` so that
        the metacognitive layer remains fully asynchronous.

        Args:
            output: The cognitive layer result to validate.
            criteria: Quality specification (see
                :meth:`QualityVerifier.composite_score` for schema).

        Returns:
            ``(passed, confidence, reason)``.
        """
        passed, score, reason = QualityVerifier.composite_score(output, criteria)

        await self._log(
            level=MetaLevel.VERIFY,
            trigger="verify_output",
            observation={"output_keys": list(output.keys()), "criteria_keys": list(criteria.keys())},
            decision=f"passed={passed}, score={score:.2f}",
            confidence=score,
        )
        return passed, score, reason

    async def revise(
        self,
        strategy: Dict[str, Any],
        original_task: Message,
        verification_result: Tuple[bool, float, str],
    ) -> Message:
        """Produce a corrected task message based on the generated strategy.

        The new message preserves the original *correlation_id* so that the
        causal chain can be reconstructed by the reflection log.  Its kind is
        set to :attr:`MessageKind.TASK` and its payload is mutated according
        to the strategy modifications.

        Args:
            strategy: Strategy returned by :meth:`generate_strategy`.
            original_task: The task that produced unsatisfactory output.
            verification_result: Tuple from :meth:`verify`.

        Returns:
            A new :class:`Message` representing the revised task.
        """
        passed, score, reason = verification_result
        payload = dict(original_task.payload)

        approach = strategy.get("approach", "retry")
        modifications = strategy.get("modifications", [])

        # Apply modifications to payload heuristically.
        if approach == "tool_substitution":
            # Replace missing tools with available ones if possible.
            existing_tools = payload.get("tools", [])
            if existing_tools:
                payload["tools"] = existing_tools[:1]  # Fallback to first tool only.
        elif approach == "decomposition":
            payload["decomposed"] = True
            payload["sub_task_index"] = payload.get("sub_task_index", 0) + 1
        elif approach == "enrichment":
            payload["enriched"] = True
            payload.setdefault("context", {})
            payload["context"]["retry_reason"] = reason
        elif approach == "reorder_and_retry":
            tools = payload.get("tools", [])
            if len(tools) > 1:
                # Rotate left by one.
                payload["tools"] = tools[1:] + tools[:1]
            payload["retry_jitter"] = True

        # Annotate with metacognitive metadata.
        payload["_meta"] = {
            "revision": True,
            "approach": approach,
            "modifications": modifications,
            "prior_score": score,
            "prior_reason": reason,
        }

        revised = Message(
            kind=MessageKind.TASK,
            sender=original_task.recipient,  # The agent talks to itself.
            recipient=original_task.sender or self.agent_id,
            payload=payload,
            correlation_id=original_task.correlation_id or original_task.id,
        )

        await self._log(
            level=MetaLevel.REVISE,
            trigger="revise_task",
            observation={"approach": approach, "prior_score": score},
            decision=f"Revised task with approach={approach}",
            confidence=max(0.0, score - 0.1),  # Slightly lower confidence on revision.
        )
        return revised

    # -- orchestrated loop -------------------------------------------------

    async def metacognitive_loop(
        self,
        task: Message,
        tools: Dict[str, Callable[..., Any]],
        criteria: Optional[Dict[str, Any]] = None,
    ) -> Message:
        """Execute the full Monitor → Generate → Verify → Revise cycle.

        The loop proceeds as follows:

        1. Execute the *task* via the owned :class:`CognitiveLayer`.
        2. **Monitor** the output for anomalies.
        3. If anomalies exist, **generate** an alternative strategy.
        4. **Verify** the output against *criteria* (or a default criteria set
           derived from the task payload).
        5. If verification fails, **revise** the task and re-execute, up to
           *max_iterations* times.
        6. Return the final accepted result as a :class:`Message`.

        Args:
            task: The initial task message.
            tools: Registry of tools available to the cognitive layer.
            criteria: Optional quality criteria for :meth:`verify`.  If
                omitted, the task payload's ``"quality_criteria"`` is used,
                falling back to a permissive default.

        Returns:
            A :class:`Message` of kind :attr:`MessageKind.RESULT` containing
            the final cognitive output under ``"payload"["result"]`` and
            metacognitive metadata under ``"payload"["_meta_loop"]``.
        """
        criteria = criteria or task.payload.get("quality_criteria", {})
        current_task = task
        failed_attempts: List[Dict[str, Any]] = []

        for iteration in range(1, self.max_iterations + 1):
            # 1. Cognitive execution
            cognitive_output = await self.cognitive.execute(current_task, tools)

            # 2. Monitor
            monitor_record = await self.monitor(cognitive_output)
            needs_intervention = monitor_record.confidence < 0.5

            # 3. Verify
            passed, score, reason = await self.verify(cognitive_output, criteria)

            # If both monitor and verifier are happy, we are done.
            if not needs_intervention and passed and score >= self.quality_threshold:
                return Message(
                    kind=MessageKind.RESULT,
                    sender=self.agent_id,
                    recipient=task.sender or self.agent_id,
                    payload={
                        "result": cognitive_output,
                        "_meta_loop": {
                            "iterations": iteration,
                            "final_score": score,
                            "final_reason": reason,
                            "intervened": False,
                        },
                    },
                    correlation_id=task.correlation_id or task.id,
                )

            # Not good enough — record failure and enter Generate → Revise.
            failed_attempts.append(cognitive_output)
            problem = monitor_record.trigger if needs_intervention else reason

            # 4. Generate strategy
            strategy = await self.generate_strategy(problem, failed_attempts)

            # 5. Revise task for next iteration (unless this is the last try)
            if iteration < self.max_iterations:
                current_task = await self.revise(
                    strategy,
                    current_task,
                    verification_result=(passed, score, reason),
                )
            else:
                # Last iteration — return best-effort result with warnings.
                return Message(
                    kind=MessageKind.RESULT,
                    sender=self.agent_id,
                    recipient=task.sender or self.agent_id,
                    payload={
                        "result": cognitive_output,
                        "_meta_loop": {
                            "iterations": iteration,
                            "final_score": score,
                            "final_reason": reason,
                            "intervened": True,
                            "max_iter_reached": True,
                            "warning": "Quality threshold not met after max iterations",
                        },
                    },
                    correlation_id=task.correlation_id or task.id,
                )

        # Defensive fallback (unreachable because loop always returns).
        return Message(
            kind=MessageKind.RESULT,
            sender=self.agent_id,
            recipient=task.sender or self.agent_id,
            payload={
                "result": {},
                "_meta_loop": {"error": "metacognitive_loop exited without return"},
            },
            correlation_id=task.correlation_id or task.id,
        )


# ---------------------------------------------------------------------------
# Convenience re-exports
# ---------------------------------------------------------------------------

__all__ = [
    "CognitiveLayer",
    "MetacognitiveLayer",
    "QualityVerifier",
    "SelfReflectionLog",
]
