"""CallGraph — 统一图容器（支持图变换）

核心操作不是简单 add/remove，而是**图变换**（Graph Transformation）：
  - SURVEY:  空图 → 骨架树
  - CALIBRATE: 骨架树 → 预算流树
  - EXPAND:  预算流树 + 新发现 → 扩展树
  - CONTRACT: 扩展树 + 超支信号 → 修剪树
  - REBALANCE: 修剪树 → 重新分配 Token
"""
from __future__ import annotations

import random
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from .token import BudgetToken
from .constraints import (
    ConcurrencyLayer,
    ConstraintEdge,
    ConstraintLayer,
    ConstraintNode,
    ResourceLayer,
    SecurityLayer,
    TopologyLayer,
)


class CallGraph:
    """BACG 调用图：属性有向树 + 6 层约束面。"""

    def __init__(self, root_id: str = "root"):
        self.root_id = root_id
        self.nodes: Dict[str, ConstraintNode] = {}
        self.edges: Dict[str, ConstraintEdge] = {}
        self.layers: List[ConstraintLayer] = []

        self._create_root()

    def _create_root(self) -> None:
        root = ConstraintNode(
            node_id=self.root_id,
            op_type="root",
            actor_id="system",
            depth=0,
            security_level="system",
        )
        self.nodes[self.root_id] = root

    def add_layer(self, layer: ConstraintLayer) -> None:
        """添加约束层（Photoshop 图层风格）。"""
        self.layers.append(layer)

    # ─── 图变换操作 ───────────────────────────────────────────────

    def apply_survey(self, branching_model) -> None:
        """变换1: SURVEY — 根据分支模型生成预估骨架树。"""
        root = self.nodes[self.root_id]
        self._build_skeleton(root, branching_model, depth=0)

    def _build_skeleton(self, parent: ConstraintNode, branching_model, depth: int) -> None:
        """递归构建预估骨架。"""
        if depth >= 3:
            return

        m = getattr(branching_model, 'mean_branching', 0.5)
        n_children = min(int(m) + 1, 3)

        for i in range(n_children):
            child_id = f"{parent.node_id}_c{i}"
            op_types = ["llm_call", "tool_invoke", "search"]
            op = random.choice(op_types) if hasattr(random, 'choice') else "llm_call"

            child = ConstraintNode(
                node_id=child_id,
                op_type=op,
                parent_id=parent.node_id,
                depth=depth + 1,
            )

            if self._check_node_through_layers(child):
                self.nodes[child_id] = child
                parent.children_ids.append(child_id)

                edge = ConstraintEdge(
                    source_id=parent.node_id,
                    target_id=child_id,
                    trigger_reason="survey_estimate",
                )
                self.edges[f"{parent.node_id}->{child_id}"] = edge

                self._build_skeleton(child, branching_model, depth + 1)

    def apply_calibrate(self, total_budget: Dict[str, float]) -> None:
        """变换2: CALIBRATE — 将总预算作为 Token 注入根节点，然后递归传播。"""
        root = self.nodes[self.root_id]

        root_token = BudgetToken(
            dimensions=total_budget.copy(),
            path=[self.root_id],
            source_node=self.root_id,
        )
        root.received_tokens = [root_token]

        self._propagate_budget_recursive(root)

    def _propagate_budget_recursive(self, parent: ConstraintNode) -> None:
        """递归传播预算 Token。"""
        if not parent.children_ids:
            return

        resource_layer = None
        for layer in self.layers:
            if isinstance(layer, ResourceLayer):
                resource_layer = layer
                break

        if not resource_layer or not parent.received_tokens:
            return

        child_specs = []
        for child_id in parent.children_ids:
            child = self.nodes.get(child_id)
            if child:
                weights = {
                    "tokens": {"llm_call": 1.0, "tool_invoke": 0.6, "search": 0.3, "agent_delegate": 1.2}.get(child.op_type, 0.5),
                    "usd": {"llm_call": 1.0, "tool_invoke": 0.6, "search": 0.3, "agent_delegate": 1.2}.get(child.op_type, 0.5),
                }
                child_specs.append((child_id, weights))

        child_tokens = resource_layer.propagate_token(parent, child_specs)

        for i, child_id in enumerate(parent.children_ids):
            if i < len(child_tokens):
                child = self.nodes[child_id]
                child.received_tokens = [child_tokens[i]]

                edge_key = f"{parent.node_id}->{child_id}"
                if edge_key in self.edges:
                    self.edges[edge_key].resource_flow = child_tokens[i].dimensions

        for child_id in parent.children_ids:
            if child_id in self.nodes:
                self._propagate_budget_recursive(self.nodes[child_id])

    def apply_expand(self, parent_id: str, child_specs: List[Dict[str, Any]]) -> Optional[List[str]]:
        """变换3: EXPAND — 运行时动态扩展图。"""
        if parent_id not in self.nodes:
            return None

        parent = self.nodes[parent_id]
        created_ids = []

        security_layer = None
        topology_layer = None
        for layer in self.layers:
            if isinstance(layer, SecurityLayer):
                security_layer = layer
            if isinstance(layer, TopologyLayer):
                topology_layer = layer

        for i, spec in enumerate(child_specs):
            child_id = f"{parent_id}_d{i}_{int(time.time() * 1000) % 10000}"

            child = ConstraintNode(
                node_id=child_id,
                op_type=spec.get("op_type", "llm_call"),
                actor_id=spec.get("actor_id", parent.actor_id),
                parent_id=parent_id,
                depth=parent.depth + 1,
                security_level=spec.get("security_level", parent.security_level),
            )

            if security_layer:
                passed, reason = security_layer.check_edge(
                    ConstraintEdge(parent_id, child_id),
                    parent_op=parent.op_type,
                    child_op=child.op_type,
                )
                if not passed:
                    print(f"  [Security] Blocked: {reason}")
                    continue

            if not self._check_node_through_layers(child):
                continue

            if topology_layer:
                n_siblings = len(parent.children_ids) + 1
                topology_layer.observe_branching(n_siblings)
                passed, reason = topology_layer.check_node(child)
                if not passed:
                    print(f"  [Topology] Blocked: {reason}")
                    continue

            self.nodes[child_id] = child
            parent.children_ids.append(child_id)

            edge = ConstraintEdge(
                source_id=parent_id,
                target_id=child_id,
                trigger_reason=spec.get("trigger_reason", "runtime_discovery"),
            )
            self.edges[f"{parent_id}->{child_id}"] = edge

            created_ids.append(child_id)

        if created_ids:
            self._propagate_budget_recursive(parent)

        return created_ids

    def apply_contract(self, node_id: str) -> None:
        """变换4: CONTRACT — 截断子树。"""
        if node_id not in self.nodes:
            return

        node = self.nodes[node_id]

        for child_id in list(node.children_ids):
            self.apply_contract(child_id)
            child = self.nodes.get(child_id)
            if child:
                child.status = "pruned"

        node.children_ids = []

    def apply_rebalance(self) -> None:
        """变换5: REBALANCE — 重新分配预算 Token。"""
        if self.root_id in self.nodes:
            self._propagate_budget_recursive(self.nodes[self.root_id])

    # ─── 约束检查 ─────────────────────────────────────────────────

    def _check_node_through_layers(self, node: ConstraintNode) -> bool:
        """通过所有约束层检查节点。"""
        for layer in self.layers:
            passed, reason = layer.check_node(node)
            if not passed:
                print(f"  [{layer.name}] Node rejected: {reason}")
                return False
        return True

    def check_all(self) -> Tuple[bool, List[str]]:
        """全局约束检查。"""
        violations = []

        for layer in self.layers:
            passed, reason = layer.check_global(self)
            if not passed:
                violations.append(f"[{layer.name}] {reason}")

            if isinstance(layer, ConcurrencyLayer):
                max_depth = max((n.depth for n in self.nodes.values()), default=0)
                for d in range(max_depth + 1):
                    nodes_at_d = [n for n in self.nodes.values() if n.depth == d]
                    passed, reason = layer.check_level(nodes_at_d)
                    if not passed:
                        violations.append(f"[{layer.name}] Level {d}: {reason}")

        return len(violations) == 0, violations

    # ─── 查询 ─────────────────────────────────────────────────────

    def get_path(self, node_id: str) -> List[ConstraintNode]:
        """获取从根到指定节点的路径。"""
        path = []
        current = self.nodes.get(node_id)
        while current:
            path.append(current)
            current = self.nodes.get(current.parent_id) if current.parent_id else None
        return list(reversed(path))

    def get_level(self, depth: int) -> List[ConstraintNode]:
        """获取某层的所有节点。"""
        return [n for n in self.nodes.values() if n.depth == depth]

    def branching_stats(self) -> Dict[str, float]:
        """计算分支统计。"""
        counts = [len(n.children_ids) for n in self.nodes.values()]
        if not counts:
            return {"mean": 0, "max": 0, "total_nodes": 0}
        return {
            "mean": sum(counts) / len(counts),
            "max": max(counts),
            "total_nodes": len(self.nodes),
            "total_edges": len(self.edges),
        }

    def total_consumed(self) -> Dict[str, float]:
        """计算全图总消耗。"""
        total = defaultdict(float)
        for node in self.nodes.values():
            for k, v in node.consumed.items():
                total[k] += v
        return dict(total)

    def to_otel_spans(self) -> List[Dict[str, Any]]:
        """导出为 OpenTelemetry Span 格式。"""
        return [n.to_span_format() for n in self.nodes.values()]

    def __repr__(self) -> str:
        stats = self.branching_stats()
        consumed = self.total_consumed()
        return (
            f"CallGraph(nodes={stats['total_nodes']}, "
            f"edges={stats['total_edges']}, "
            f"branching={stats['mean']:.2f}, "
            f"consumed={consumed})"
        )
