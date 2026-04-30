"""TopologyBudget — 拓扑感知的预算分配

核心思想：
  不是"给 Agent X tokens"，而是"给调用树根节点一个预算池，
  根据分支过程模型递归分配给子节点"。

分配公式（递归预算传播）：
  B(node) = B(parent) · w(node) / Σw(siblings)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

from .branching import BranchingModel


# Forward reference for CallNode used in the original bacg.py
@dataclass
class CallNode:
    """调用图节点（v1 兼容）"""
    node_id: str
    op_type: str = "llm_call"
    estimated_cost: Dict[str, float] = field(default_factory=dict)
    actual_cost: Optional[Dict[str, float]] = None
    children: list = field(default_factory=list)
    parent: Optional[CallNode] = None
    depth: int = 0
    branch_factor_observed: float = 0.0

    def add_child(self, child: CallNode) -> None:
        child.parent = self
        child.depth = self.depth + 1
        self.children.append(child)

    @property
    def total_actual_cost(self) -> Dict[str, float]:
        from collections import defaultdict
        total = defaultdict(float)
        if self.actual_cost:
            for k, v in self.actual_cost.items():
                total[k] += v
        for child in self.children:
            child_total = child.total_actual_cost
            for k, v in child_total.items():
                total[k] += v
        return dict(total)

    @property
    def subtree_size(self) -> int:
        return 1 + sum(c.subtree_size for c in self.children)

    @property
    def max_depth(self) -> int:
        if not self.children:
            return self.depth
        return max(c.max_depth for c in self.children)


@dataclass
class CallGraphV1:
    """调用图 v1（兼容 bacg.py）"""
    root: CallNode
    all_nodes: Dict[str, CallNode] = field(default_factory=dict)

    def __post_init__(self):
        self._index_nodes(self.root)

    def _index_nodes(self, node: CallNode) -> None:
        self.all_nodes[node.node_id] = node
        for child in node.children:
            self._index_nodes(child)

    def get_branching_stats(self) -> Dict[str, float]:
        branch_counts = [len(n.children) for n in self.all_nodes.values()]
        if not branch_counts:
            return {"mean": 0, "max": 0, "variance": 0}
        mean = sum(branch_counts) / len(branch_counts)
        variance = sum((c - mean) ** 2 for c in branch_counts) / len(branch_counts)
        return {"mean": mean, "max": max(branch_counts), "variance": variance}

    def get_cost_distribution(self) -> Dict[str, list]:
        from collections import defaultdict
        costs = defaultdict(list)
        for node in self.all_nodes.values():
            if node.actual_cost:
                for k, v in node.actual_cost.items():
                    costs[f"{node.op_type}:{k}"].append(v)
        return dict(costs)


@dataclass
class TopologyBudget:
    """拓扑感知的预算分配器。"""

    total_budget: Dict[str, float] = field(default_factory=lambda: {"tokens": 10000, "usd": 10})
    allocation_strategy: str = "branch_aware"  # uniform / value / branch_aware
    reserve_ratio: float = 0.1

    def allocate_to_graph(self, root: CallNode, branching: BranchingModel) -> None:
        """将预算分配给整个调用图。"""
        root_budget = {
            k: v * (1 - self.reserve_ratio)
            for k, v in self.total_budget.items()
        }
        root.estimated_cost = root_budget
        self._allocate_recursive(root, branching)

    def _allocate_recursive(self, node: CallNode, branching: BranchingModel) -> None:
        """递归分配预算给子节点。"""
        if not node.children:
            return

        weights = []
        for child in node.children:
            w = self._compute_weight(child, branching)
            weights.append(w)

        total_weight = sum(weights) or 1.0

        parent_available = {}
        for k, v in node.estimated_cost.items():
            consumed = node.actual_cost.get(k, 0) if node.actual_cost else 0
            parent_available[k] = max(0, v - consumed)

        for child, w in zip(node.children, weights):
            ratio = w / total_weight
            child.estimated_cost = {
                k: v * ratio
                for k, v in parent_available.items()
            }
            self._allocate_recursive(child, branching)

    def _compute_weight(self, node: CallNode, branching: BranchingModel) -> float:
        """计算节点的分配权重。"""
        if self.allocation_strategy == "uniform":
            return 1.0

        elif self.allocation_strategy == "value":
            value_map = {
                "llm_call": 1.0,
                "tool_invoke": 0.8,
                "agent_delegate": 1.2,
                "search": 0.6,
            }
            return value_map.get(node.op_type, 1.0)

        elif self.allocation_strategy == "branch_aware":
            m = branching.mean_branching
            if m < 1.0:
                expected_subtree = 1.0 / (1.0 - m)
            else:
                expected_subtree = 10.0
            return expected_subtree

        return 1.0

    def reallocate_after_discovery(
        self,
        discovered_node: CallNode,
        actual_cost: Dict[str, float],
        branching: BranchingModel,
    ) -> None:
        """运行时重新分配：发现新节点后调整预算。"""
        branching.observe(len(discovered_node.children), actual_cost)

        if branching.mean_branching > 1.2:
            self._tighten_allocation(discovered_node)

    def _tighten_allocation(self, node: CallNode) -> None:
        """收紧节点的预算分配。"""
        for child in node.children:
            child.estimated_cost = {
                k: v * 0.8
                for k, v in child.estimated_cost.items()
            }
