"""6层约束面（分层独立设计）

借鉴 Photoshop 图层思想，每层约束独立工作：
  Layer 1: IdentityLayer   — 身份约束（节点级）
  Layer 2: ResourceLayer   — 资源约束（路径级 + Token 流动）
  Layer 3: SecurityLayer   — 安全约束（边级）
  Layer 4: QualityLayer    — 质量约束（路径级衰减）
  Layer 5: ConcurrencyLayer — 并发约束（层级级）
  Layer 6: TopologyLayer   — 拓扑约束（全局级）
"""
from __future__ import annotations

import math
import time
from abc import ABC, abstractmethod
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Optional, Tuple

from .token import BudgetToken, InsufficientBudget


@dataclass
class ConstraintNode:
    """属性图节点：带有约束属性的调用图节点。

    借鉴 Neo4j 的 Labeled Property Graph 模型和 OpenTelemetry Span：
      - 每个节点有标签（op_type）和属性字典
      - parent_id / children_ids 构成 Span 树
    """

    # 标识
    node_id: str
    op_type: str  # "llm_call" | "tool_invoke" | "agent_delegate" | "search" | "sandbox"
    actor_id: str = "system"

    # 资源属性（BudgetToken 容器）
    received_tokens: List[BudgetToken] = field(default_factory=list)
    consumed: Dict[str, float] = field(default_factory=dict)

    # 质量属性
    quality_score: float = 1.0

    # 树结构（Span 树风格）
    parent_id: Optional[str] = None
    children_ids: List[str] = field(default_factory=list)
    depth: int = 0

    # 状态机
    status: str = "pending"  # pending → running → [completed|failed|pruned]

    # 安全属性
    security_level: str = "default"
    allowed_child_ops: List[str] = field(default_factory=list)

    # 生命周期时间戳
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    completed_at: Optional[float] = None

    # 元数据
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def remaining(self) -> Dict[str, float]:
        """计算剩余预算（所有 Token 的维度之和 - 已消耗）。"""
        total = defaultdict(float)
        for token in self.received_tokens:
            for k, v in token.dimensions.items():
                total[k] += v
        for k, v in self.consumed.items():
            total[k] -= v
        return {k: max(0, v) for k, v in total.items()}

    @property
    def subtree_cost(self, graph: CallGraph = None) -> Dict[str, float]:
        """递归计算子树总消耗（需要外部图引用）。"""
        return self.consumed.copy()

    def to_span_format(self) -> Dict[str, Any]:
        """导出为 OpenTelemetry Span 格式。"""
        return {
            "spanId": self.node_id,
            "parentSpanId": self.parent_id,
            "opType": self.op_type,
            "status": self.status,
            "startTime": self.started_at,
            "endTime": self.completed_at,
            "attributes": {
                "consumed": self.consumed,
                "remaining": self.remaining,
                "quality_score": self.quality_score,
                "actor_id": self.actor_id,
                "depth": self.depth,
            }
        }


@dataclass
class ConstraintEdge:
    """属性图边：带有约束属性的调用依赖边。"""
    source_id: str
    target_id: str
    trigger_reason: str = "explicit_call"
    resource_flow: Dict[str, float] = field(default_factory=dict)
    security_clearance: str = "default"
    created_at: float = field(default_factory=time.time)


class ConstraintLayer(ABC):
    """约束层抽象基类。

    每个约束层独立工作，就像 Photoshop 的图层。
    """

    name: str = "base"

    @abstractmethod
    def check_node(self, node: ConstraintNode) -> Tuple[bool, Optional[str]]:
        """检查节点是否满足约束。返回 (是否通过, 原因)。"""
        ...

    def check_edge(self, edge: ConstraintEdge) -> Tuple[bool, Optional[str]]:
        """检查边是否满足约束。默认通过。"""
        return True, None

    def check_path(self, path: List[ConstraintNode]) -> Tuple[bool, Optional[str]]:
        """检查路径是否满足约束。默认通过。"""
        return True, None

    def check_level(self, nodes_at_level: List[ConstraintNode]) -> Tuple[bool, Optional[str]]:
        """检查某层是否满足约束。默认通过。"""
        return True, None

    def check_global(self, graph: CallGraph) -> Tuple[bool, Optional[str]]:
        """检查整棵树是否满足约束。默认通过。"""
        return True, None


# ─── Layer 1: 身份约束（节点级）───────────────────────────────────

class IdentityLayer(ConstraintLayer):
    """身份约束层：节点级准入控制。

    借鉴 AWS IAM 和 CHERI 能力模型：
      - 每个 actor 有允许的操作类型列表
      - 节点创建时检查 actor 是否有权限
    """

    name = "identity"

    def __init__(self, acl: Dict[str, List[str]] = None):
        self.acl = acl or {}

    def check_node(self, node: ConstraintNode) -> Tuple[bool, Optional[str]]:
        allowed = self.acl.get(node.actor_id, ["*"])
        if "*" in allowed or node.op_type in allowed:
            return True, None
        return False, (
            f"Identity: actor '{node.actor_id}' cannot create "
            f"op_type '{node.op_type}'"
        )


# ─── Layer 2: 资源约束（路径级 + Token 流动）───────────────────────

class ResourceLayer(ConstraintLayer):
    """资源约束层：路径级预算控制 + Petri Token 流动。"""

    name = "resource"

    def __init__(self, total_budget: Dict[str, float] = None):
        self.total_budget = total_budget or {"tokens": 10000, "usd": 5.0}
        self.path_costs: Dict[str, Dict[str, float]] = {}

    def check_node(self, node: ConstraintNode) -> Tuple[bool, Optional[str]]:
        if not node.received_tokens:
            return True, None
        remaining = node.remaining
        if not remaining or all(v <= 0 for v in remaining.values()):
            return False, f"Resource: node '{node.node_id}' has no budget tokens"
        return True, None

    def check_path(self, path: List[ConstraintNode]) -> Tuple[bool, Optional[str]]:
        total_consumed = defaultdict(float)
        for node in path:
            for k, v in node.consumed.items():
                total_consumed[k] += v

        for k, limit in self.total_budget.items():
            if total_consumed.get(k, 0) > limit:
                return False, (
                    f"Resource: path consumed {total_consumed[k]:.2f} {k}, "
                    f"exceeds budget {limit:.2f}"
                )
        return True, None

    def propagate_token(
        self, 
        parent: ConstraintNode, 
        child_specs: List[Tuple[str, Dict[str, float]]]
    ) -> List[BudgetToken]:
        """将父节点的 Token 分配给子节点。"""
        if not parent.received_tokens:
            return []

        parent_dims = defaultdict(float)
        for token in parent.received_tokens:
            for k, v in token.dimensions.items():
                parent_dims[k] += v

        for k, v in parent.consumed.items():
            parent_dims[k] -= v

        total_weights = defaultdict(float)
        for _, weights in child_specs:
            for k, w in weights.items():
                total_weights[k] += w

        parent_token = BudgetToken(
            dimensions=dict(parent_dims),
            path=parent.received_tokens[0].path if parent.received_tokens else [parent.node_id],
            source_node=parent.node_id,
        )

        child_ratios = []
        for child_name, weights in child_specs:
            avg_ratio = sum(
                weights.get(k, 0) / max(total_weights[k], 1e-6) 
                for k in total_weights
            ) / len(total_weights) if total_weights else 1.0 / len(child_specs)
            child_ratios.append((child_name, avg_ratio))

        total_ratio = sum(r for _, r in child_ratios)
        if total_ratio > 0:
            child_ratios = [(n, r / total_ratio * 0.95) for n, r in child_ratios]

        return parent_token.split(child_ratios)


# ─── Layer 3: 安全约束（边级）─────────────────────────────────────

class SecurityLayer(ConstraintLayer):
    """安全约束层：边级白名单控制。"""

    name = "security"

    def __init__(self, edge_whitelist: Dict[str, List[str]] = None):
        self.edge_whitelist = edge_whitelist or {
            "llm_call": ["llm_call", "tool_invoke", "search", "agent_delegate"],
            "tool_invoke": ["tool_invoke", "llm_call"],
            "search": ["llm_call", "search"],
            "agent_delegate": ["llm_call", "tool_invoke", "search", "agent_delegate"],
            "sandbox": ["sandbox", "llm_call", "tool_invoke"],
            "root": ["llm_call", "agent_delegate", "search"],
        }

    def check_edge(self, edge: ConstraintEdge, parent_op: str = None, child_op: str = None) -> Tuple[bool, Optional[str]]:
        allowed = self.edge_whitelist.get(parent_op, [])
        if "*" in allowed or child_op in allowed:
            return True, None
        return False, (
            f"Security: op '{parent_op}' cannot trigger op '{child_op}'"
        )

    def check_node(self, node: ConstraintNode) -> Tuple[bool, Optional[str]]:
        if node.allowed_child_ops:
            parent_allowed = self.edge_whitelist.get(node.op_type, [])
            for op in node.allowed_child_ops:
                if op not in parent_allowed and "*" not in parent_allowed:
                    return False, (
                        f"Security: node '{node.node_id}' op '{node.op_type}' "
                        f"cannot have child op '{op}'"
                    )
        return True, None


# ─── Layer 4: 质量约束（路径级衰减）────────────────────────────────

class QualityLayer(ConstraintLayer):
    """质量约束层：路径级质量衰减控制。

    质量沿路径指数衰减：quality_at_depth_d = initial_quality * decay_rate ^ d
    """

    name = "quality"

    def __init__(self, decay_rate: float = 0.9, min_quality: float = 0.3):
        self.decay_rate = decay_rate
        self.min_quality = min_quality

    def compute_quality(self, depth: int) -> float:
        """计算给定深度处的质量分数。"""
        return max(self.min_quality, 1.0 * (self.decay_rate ** depth))

    def check_path(self, path: List[ConstraintNode]) -> Tuple[bool, Optional[str]]:
        if not path:
            return True, None
        depth = path[-1].depth
        quality = self.compute_quality(depth)
        if quality <= self.min_quality:
            return False, (
                f"Quality: depth {depth} quality {quality:.3f} "
                f"below minimum {self.min_quality}"
            )
        return True, None

    def check_node(self, node: ConstraintNode) -> Tuple[bool, Optional[str]]:
        node.quality_score = self.compute_quality(node.depth)
        return True, None


# ─── Layer 5: 并发约束（层级级）────────────────────────────────────

class ConcurrencyLayer(ConstraintLayer):
    """并发约束层：层级级宽度控制（图着色思想）。"""

    name = "concurrency"

    def __init__(self, max_width: int = 5):
        self.max_width = max_width

    def check_node(self, node: ConstraintNode) -> Tuple[bool, Optional[str]]:
        return True, None

    def check_level(self, nodes_at_level: List[ConstraintNode]) -> Tuple[bool, Optional[str]]:
        running = [n for n in nodes_at_level if n.status in ("pending", "running")]
        if len(running) > self.max_width:
            return False, (
                f"Concurrency: level has {len(running)} active nodes, "
                f"max allowed {self.max_width}"
            )
        return True, None


# ─── Layer 6: 拓扑约束（全局级）────────────────────────────────────

class TopologyLayer(ConstraintLayer):
    """拓扑约束层：全局结构控制（分支过程）。

    核心：Galton-Watson 分支过程
      - m < 1: 消耗收敛，允许正常执行
      - m >= 1: 消耗发散，强制截断
    """

    name = "topology"

    def __init__(self, max_depth: int = 5, max_branching: float = 3.0):
        self.max_depth = max_depth
        self.max_branching = max_branching
        self.observed_branching: Deque[float] = deque(maxlen=50)

    def observe_branching(self, n_children: int) -> None:
        """观测一次分支事件。"""
        self.observed_branching.append(float(n_children))

    @property
    def mean_branching(self) -> float:
        if not self.observed_branching:
            return 0.0
        return sum(self.observed_branching) / len(self.observed_branching)

    def check_node(self, node: ConstraintNode) -> Tuple[bool, Optional[str]]:
        if node.depth > self.max_depth:
            return False, (
                f"Topology: depth {node.depth} exceeds max {self.max_depth}"
            )
        return True, None

    def check_global(self, graph: CallGraph) -> Tuple[bool, Optional[str]]:
        m = self.mean_branching
        if m >= 1.0:
            return False, (
                f"Topology: mean branching {m:.2f} >= 1.0, "
                f"expected consumption diverges (infinite)"
            )
        return True, None

    def recommend_action(self) -> str:
        """根据拓扑状态推荐行动。"""
        m = self.mean_branching
        if m >= 1.0:
            return "TRUNCATE"
        elif m >= 0.8:
            return "TIGHTEN"
        elif m >= 0.5:
            return "MONITOR"
        else:
            return "NORMAL"
