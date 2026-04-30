"""AgentKernel v2 — Constraints as Effects (CaaE)

核心范式：每个 Agent 操作声明「效果签名」（Effect），统一效果处理器
在执行前自动拦截检查（Declare → Check → Execute → Record）。

五种效果原语覆盖所有约束维度：
  CONSUME_RESOURCE    — 资源边界（token/cost/time/api）
  REQUIRE_CAPABILITY  — 安全边界（权限/沙盒/网络）
  PRODUCE_OUTPUT      — 质量边界（schema/置信度）
  DELEGATE_TASK       — 委托边界（预算传播/守恒律）
  CHECK_INVARIANT     — 不变量边界（安全/一致性/PII）

不是六个独立模块，是一个效果系统，五种效果。
"""
from __future__ import annotations

import asyncio
import enum
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Protocol, Set, Tuple
from uuid import UUID, uuid4


# ═══════════════════════════════════════════════════════════════
# 1. Effect — 效果签名（统一约束原语）
# ═══════════════════════════════════════════════════════════════

class EffectKind(enum.Enum):
    """五种效果原语 = 所有约束维度的完备覆盖。"""
    CONSUME_RESOURCE = "consume"
    REQUIRE_CAPABILITY = "require"
    PRODUCE_OUTPUT = "produce"
    DELEGATE_TASK = "delegate"
    CHECK_INVARIANT = "invariant"


@dataclass(frozen=True)
class Effect:
    """不可变的效果签名。每个 Agent 操作在执行前声明自己要产生的效果。

    Example:
        Effect(
            kind=EffectKind.CONSUME_RESOURCE,
            operation="llm_call",
            estimated_cost={"tokens": 1000, "usd": 0.02, "seconds": 2.0},
        )
    """
    kind: EffectKind
    operation: str
    estimated_cost: Dict[str, float] = field(default_factory=dict)
    required_permissions: Set[str] = field(default_factory=set)
    constraints: Dict[str, Any] = field(default_factory=dict)
    description: str = ""

    # 效果代数：两个效果可以合并（同类型同操作的资源消耗累加）
    def __add__(self, other: Effect) -> Effect:
        if self.kind != other.kind or self.operation != other.operation:
            raise ValueError("Effects must have same kind and operation to combine")
        merged_cost = {k: self.estimated_cost.get(k, 0) + other.estimated_cost.get(k, 0)
                        for k in set(self.estimated_cost) | set(other.estimated_cost)}
        return Effect(
            kind=self.kind,
            operation=self.operation,
            estimated_cost=merged_cost,
            required_permissions=self.required_permissions | other.required_permissions,
            constraints={**self.constraints, **other.constraints},
        )


# ═══════════════════════════════════════════════════════════════
# 2. ExecutionContext — 约束状态容器（可传递、可分割、可回收）
# ═══════════════════════════════════════════════════════════════

@dataclass
class ResourceBudget:
    """多维资源配额。支持原子扣除和余额查询。"""
    tokens: float = 10_000
    usd: float = 5.0
    seconds: float = 300.0
    api_calls: float = 50

    def can_afford(self, cost: Dict[str, float]) -> bool:
        return all(
            getattr(self, k, 0) >= v
            for k, v in cost.items()
        )

    def consume(self, cost: Dict[str, float]) -> ResourceBudget:
        if not self.can_afford(cost):
            raise InsufficientBudget(f"Cannot afford {cost} from {self}")
        return ResourceBudget(
            tokens=self.tokens - cost.get("tokens", 0),
            usd=self.usd - cost.get("usd", 0),
            seconds=self.seconds - cost.get("seconds", 0),
            api_calls=self.api_calls - cost.get("api_calls", 0),
        )

    def __repr__(self) -> str:
        return f"Budget(t={self.tokens:.0f}, ${self.usd:.2f}, s={self.seconds:.0f})"


@dataclass
class CapabilitySet:
    """能力集合：Agent 拥有的权限和沙盒级别。"""
    sandbox_level: str = "none"  # none / process / container / gvisor / microvm / tee
    network: bool = False
    filesystem: str = "none"  # none / readonly / readwrite
    allowed_operations: Set[str] = field(default_factory=set)

    def allows(self, permissions: Set[str]) -> bool:
        return permissions.issubset(self.allowed_operations)


@dataclass
class QualityGate:
    """质量门控：输出必须满足的标准。"""
    output_schema: Optional[Dict[str, Any]] = None
    min_confidence: float = 0.0
    required_fields: List[str] = field(default_factory=list)


@dataclass
class ExecutionContext:
    """Agent 执行时的约束状态容器。

    关键设计：
    - 不可变追加：每次操作产生新的 Context（函数式风格）
    - 可分割：委托时 split() 原子划分预算
    - 可回收：子 Agent 完成后 reclaim() 返还剩余
    - 可传递：在 Agent 间传递时约束自动传播
    """
    run_id: UUID = field(default_factory=uuid4)
    budget: ResourceBudget = field(default_factory=ResourceBudget)
    capabilities: CapabilitySet = field(default_factory=CapabilitySet)
    quality_gate: QualityGate = field(default_factory=QualityGate)
    audit_trail: AuditTrail = field(default_factory=lambda: AuditTrail())
    parent_id: Optional[UUID] = None
    child_contexts: Dict[UUID, ExecutionContext] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)

    # ─── 资源操作 ───

    def consume(self, cost: Dict[str, float]) -> ExecutionContext:
        """消费资源，返回新 Context（不可变）。"""
        new_budget = self.budget.consume(cost)
        return self._replace(budget=new_budget)

    def split(self, allocation: Dict[str, float]) -> Tuple[ExecutionContext, ExecutionContext]:
        """委托时原子划分预算。

        Returns:
            (parent_context_after, child_context) — 父上下文扣除后，子上下文获得分配
        """
        if not self.budget.can_afford(allocation):
            raise ConservationViolation(
                f"Cannot delegate {allocation} from budget {self.budget}"
            )
        child = ExecutionContext(
            run_id=uuid4(),
            budget=ResourceBudget(
                tokens=allocation.get("tokens", 0),
                usd=allocation.get("usd", 0),
                seconds=allocation.get("seconds", 0),
                api_calls=allocation.get("api_calls", 0),
            ),
            capabilities=self.capabilities,  # 继承能力
            quality_gate=self.quality_gate,    # 继承质量标准
            parent_id=self.run_id,
        )
        self.child_contexts[child.run_id] = child
        parent_after = self.consume(allocation)
        return parent_after, child

    def reclaim(self, child_or_id: Union[UUID, 'ExecutionContext']) -> ExecutionContext:
        """子 Agent 完成后，回收未使用预算。

        Args:
            child_or_id: 子 Agent 的 run_id (UUID) 或子 Agent 的 ExecutionContext。
                         推荐传入 ExecutionContext，确保回收的是实际剩余预算。
        """
        if isinstance(child_or_id, ExecutionContext):
            child = child_or_id
        else:
            child = self.child_contexts.pop(child_or_id, None)
            if child is None:
                return self
        # 返还子 Agent 剩余预算
        return self.consume({
            "tokens": -child.budget.tokens,
            "usd": -child.budget.usd,
            "seconds": -child.budget.seconds,
            "api_calls": -child.budget.api_calls,
        })

    # ─── 内部 ───

    def _replace(self, **kwargs) -> ExecutionContext:
        data = {
            "run_id": self.run_id,
            "budget": self.budget,
            "capabilities": self.capabilities,
            "quality_gate": self.quality_gate,
            "audit_trail": self.audit_trail,
            "parent_id": self.parent_id,
            "child_contexts": self.child_contexts,
            "created_at": self.created_at,
        }
        data.update(kwargs)
        return ExecutionContext(**data)


# ═══════════════════════════════════════════════════════════════
# 3. AuditTrail — 自动审计记录
# ═══════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class AuditEvent:
    """不可变的审计事件。"""
    event_id: UUID = field(default_factory=uuid4)
    run_id: UUID = field(default_factory=uuid4)
    phase: str = ""  # "declare" / "check_pass" / "check_fail" / "execute" / "record"
    effects: Tuple[Effect, ...] = ()
    budget_before: ResourceBudget = field(default_factory=ResourceBudget)
    budget_after: ResourceBudget = field(default_factory=ResourceBudget)
    timestamp: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AuditTrail:
    """审计轨迹：完整记录 Agent 执行的所有约束相关事件。"""
    events: List[AuditEvent] = field(default_factory=list)
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    async def append(self, event: AuditEvent) -> None:
        async with self._lock:
            self.events.append(event)

    def get_events(self, phase: Optional[str] = None) -> List[AuditEvent]:
        if phase is None:
            return list(self.events)
        return [e for e in self.events if e.phase == phase]

    def total_consumed(self) -> Dict[str, float]:
        """计算总资源消耗。"""
        total = {"tokens": 0.0, "usd": 0.0, "seconds": 0.0, "api_calls": 0.0}
        for e in self.events:
            for k, v in e.budget_before.__dict__.items():
                after_v = getattr(e.budget_after, k, 0)
                total[k] = total.get(k, 0) + max(0, v - after_v)
        return total


# ═══════════════════════════════════════════════════════════════
# 4. EffectHandler — 统一效果处理器（Declare → Check → Record）
# ═══════════════════════════════════════════════════════════════

class InsufficientBudget(Exception):
    pass


class ConservationViolation(Exception):
    pass


class CapabilityDenied(Exception):
    pass


class QualityNotMet(Exception):
    pass


class InvariantViolated(Exception):
    pass


@dataclass
class EffectHandler:
    """统一效果处理器。

    核心循环：Declare → Check → Execute → Record
    Agent 代码只需：1) 声明 effects；2) 调用 handle()；3) 执行业务逻辑。
    """
    validators: Dict[str, Callable[[Any, Dict[str, Any]], bool]] = field(default_factory=dict)

    async def handle(
        self,
        effects: List[Effect],
        context: ExecutionContext,
        operation_name: str = "",
    ) -> ExecutionContext:
        """处理一组效果，返回更新后的 ExecutionContext。

        自动执行：
        1. 记录 Declare 事件
        2. 按 kind 分发检查
        3. 记录 Check 事件（通过或失败）
        4. 消费资源 / 验证权限
        5. 返回新 Context
        """
        # Phase 1: Declare
        await context.audit_trail.append(AuditEvent(
            run_id=context.run_id,
            phase="declare",
            effects=tuple(effects),
            budget_before=context.budget,
            budget_after=context.budget,
            metadata={"operation": operation_name},
        ))

        # Phase 2: Check（按效果类型分发）
        new_context = context
        for effect in effects:
            new_context = await self._check_effect(effect, new_context)

        # Phase 3: Record（所有效果处理完成后的状态）
        await new_context.audit_trail.append(AuditEvent(
            run_id=new_context.run_id,
            phase="check_pass",
            effects=tuple(effects),
            budget_before=context.budget,
            budget_after=new_context.budget,
            metadata={"operation": operation_name},
        ))

        return new_context

    async def _check_effect(self, effect: Effect, context: ExecutionContext) -> ExecutionContext:
        """单个效果的约束检查。"""
        match effect.kind:

            case EffectKind.CONSUME_RESOURCE:
                # 资源边界：预算是否足够？
                if not context.budget.can_afford(effect.estimated_cost):
                    raise InsufficientBudget(
                        f"Effect {effect.operation}: cannot afford {effect.estimated_cost} "
                        f"from budget {context.budget}"
                    )
                # 扣除预估消耗（实际消耗在 execute 后修正）
                return context.consume(effect.estimated_cost)

            case EffectKind.REQUIRE_CAPABILITY:
                # 安全边界：权限是否满足？
                if not context.capabilities.allows(effect.required_permissions):
                    raise CapabilityDenied(
                        f"Effect {effect.operation}: requires {effect.required_permissions}, "
                        f"but capabilities = {context.capabilities.allowed_operations}"
                    )
                return context

            case EffectKind.PRODUCE_OUTPUT:
                # 质量边界：输出验证器注册
                # 实际验证在操作完成后由 verify_output() 执行
                return context

            case EffectKind.DELEGATE_TASK:
                # 委托边界：守恒律检查在 split() 内部完成
                return context

            case EffectKind.CHECK_INVARIANT:
                # 不变量边界：立即检查
                invariant_fn = effect.constraints.get("checker")
                if invariant_fn and not invariant_fn(context):
                    raise InvariantViolated(f"Effect {effect.operation}: invariant check failed")
                return context

            case _:
                return context

    async def verify_output(
        self,
        output: Any,
        effects: List[Effect],
        context: ExecutionContext,
    ) -> ExecutionContext:
        """操作完成后验证 PRODUCE_OUTPUT 效果。"""
        for effect in effects:
            if effect.kind != EffectKind.PRODUCE_OUTPUT:
                continue

            # Schema 验证
            schema = effect.constraints.get("schema")
            if schema and not self._validate_schema(output, schema):
                raise QualityNotMet(f"Output does not match schema {schema}")

            # 置信度验证
            min_conf = effect.constraints.get("min_confidence", 0.0)
            actual_conf = getattr(output, "confidence", 1.0) if hasattr(output, "confidence") else 1.0
            if actual_conf < min_conf:
                raise QualityNotMet(f"Confidence {actual_conf} < required {min_conf}")

            # 必需字段验证
            required = effect.constraints.get("required_fields", [])
            if isinstance(output, dict):
                missing = [f for f in required if f not in output or output[f] is None]
                if missing:
                    raise QualityNotMet(f"Missing required fields: {missing}")

        # 记录执行后状态
        await context.audit_trail.append(AuditEvent(
            run_id=context.run_id,
            phase="execute",
            effects=tuple(effects),
            budget_before=context.budget,
            budget_after=context.budget,
            metadata={"output_type": type(output).__name__},
        ))

        return context

    def _validate_schema(self, output: Any, schema: Dict[str, Any]) -> bool:
        """简化的 schema 验证。"""
        if not isinstance(output, dict):
            return False
        for key, expected_type in schema.items():
            if key not in output:
                return False
            if expected_type == "list" and not isinstance(output[key], list):
                return False
            if expected_type == "str" and not isinstance(output[key], str):
                return False
            if expected_type == "dict" and not isinstance(output[key], dict):
                return False
        return True


# ═══════════════════════════════════════════════════════════════
# 5. Agent 基类 — 展示 CaaE 的最小侵入度
# ═══════════════════════════════════════════════════════════════

class Agent:
    """CaaE 范式的 Agent 基类。

    侵入度：子类只需在 run() 中：
        1. 声明 effects（一行）
        2. 调用 handler.handle()（一行）
        3. 执行业务逻辑（不变）
    """

    def __init__(self, name: str, handler: EffectHandler) -> None:
        self.name = name
        self.handler = handler

    async def run(self, input_data: Any, context: ExecutionContext) -> Tuple[Any, ExecutionContext]:
        """子类必须实现。返回 (output, new_context)。"""
        raise NotImplementedError

    def declare_effects(
        self,
        operation: str,
        cost: Optional[Dict[str, float]] = None,
        permissions: Optional[Set[str]] = None,
        output_constraints: Optional[Dict[str, Any]] = None,
    ) -> List[Effect]:
        """辅助方法：快速声明常见效果组合。"""
        effects = []
        if cost:
            effects.append(Effect(
                kind=EffectKind.CONSUME_RESOURCE,
                operation=operation,
                estimated_cost=cost,
            ))
        if permissions:
            effects.append(Effect(
                kind=EffectKind.REQUIRE_CAPABILITY,
                operation=operation,
                required_permissions=permissions,
            ))
        if output_constraints:
            effects.append(Effect(
                kind=EffectKind.PRODUCE_OUTPUT,
                operation=operation,
                constraints=output_constraints,
            ))
        return effects


# ═══════════════════════════════════════════════════════════════
# 6. 辅助：效果推导器（自动推导常见操作的效果）
# ═══════════════════════════════════════════════════════════════

class EffectDeriver:
    """自动推导常见 Agent 操作的效果签名。

    减少开发者负担：调用 gpt-4 → 自动推导 token 消耗、权限要求。
    """

    # 已知操作的预估消耗表
    _estimates: Dict[str, Dict[str, float]] = {
        "llm_call:gpt-4": {"tokens": 1000, "usd": 0.03, "seconds": 2.0, "api_calls": 1},
        "llm_call:gpt-3.5": {"tokens": 1000, "usd": 0.002, "seconds": 1.0, "api_calls": 1},
        "web_search": {"tokens": 500, "usd": 0.01, "seconds": 3.0, "api_calls": 1},
        "code_execute": {"tokens": 200, "usd": 0.005, "seconds": 5.0, "api_calls": 0},
        "file_read": {"tokens": 100, "usd": 0.001, "seconds": 0.5, "api_calls": 0},
        "file_write": {"tokens": 100, "usd": 0.001, "seconds": 0.5, "api_calls": 0},
    }

    _permissions: Dict[str, Set[str]] = {
        "llm_call": {"network", "llm_api"},
        "web_search": {"network"},
        "code_execute": {"sandbox", "compute"},
        "file_read": {"filesystem"},
        "file_write": {"filesystem", "write"},
    }

    @classmethod
    def derive(cls, operation: str, model: Optional[str] = None) -> List[Effect]:
        """自动推导操作的效果签名。"""
        key = f"{operation}:{model}" if model else operation
        cost = cls._estimates.get(key, cls._estimates.get(operation, {}))
        perms = cls._permissions.get(operation, set())

        effects = []
        if cost:
            effects.append(Effect(
                kind=EffectKind.CONSUME_RESOURCE,
                operation=operation,
                estimated_cost=cost,
            ))
        if perms:
            effects.append(Effect(
                kind=EffectKind.REQUIRE_CAPABILITY,
                operation=operation,
                required_permissions=perms,
            ))
        return effects
