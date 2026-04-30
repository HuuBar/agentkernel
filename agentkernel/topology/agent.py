"""BACG Agent — 拓扑感知的 Agent

BACG v1 Agent: 使用 CallNode/CallGraphV1（简单树结构）
BACG v2 Agent: 使用 CallGraph（属性有向树 + 6层约束面）

执行循环：Survey → Calibrate → Exploit
"""
from __future__ import annotations

import asyncio
import random
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from .budget import CallNode, CallGraphV1, TopologyBudget, BranchingModel
from .token import BudgetToken
from .constraints import (
    ConcurrencyLayer,
    ConstraintNode,
    IdentityLayer,
    QualityLayer,
    ResourceLayer,
    SecurityLayer,
    TopologyLayer,
)
from .graph import CallGraph


# ─── BACG Agent v1 ───────────────────────────────────────────────

@dataclass
class BACGAgent:
    """Budget-Aware Call Graph Agent (v1)。

    执行循环：Survey → Calibrate → Exploit
    """

    name: str
    total_budget: Dict[str, float]
    branching_model: BranchingModel = field(default_factory=BranchingModel)
    topology_budget: Optional[TopologyBudget] = None
    current_graph: Optional[CallGraphV1] = None

    def __post_init__(self):
        self.topology_budget = TopologyBudget(total_budget=self.total_budget)

    async def execute(self, task: str) -> Tuple[Any, CallGraphV1, Dict[str, Any]]:
        """执行任务，在拓扑感知下执行。"""
        print(f"\n[BACG] Survey: 勘测 '{task}' 的调用拓扑")

        root = CallNode(node_id="root", op_type="llm_call")
        self._build_estimate_graph(root, task)
        self.current_graph = CallGraphV1(root=root)

        stats = self.current_graph.get_branching_stats()
        print(f"  预估分支因子: mean={stats['mean']:.2f}, max={stats['max']}")

        print(f"[BACG] Calibrate: 根据拓扑分配预算")

        expected = self.branching_model.expected_total_cost()
        if expected is None:
            print(f"  ⚠️ 分支因子 m={self.branching_model.mean_branching:.2f} >= 1.0，期望消耗无穷大！")
            print(f"  → 强制限制深度为 3")
            max_depth = 3
        else:
            print(f"  期望总消耗: {expected}")
            max_depth = self.branching_model.recommend_depth_limit(self.total_budget)

        self.topology_budget.allocate_to_graph(root, self.branching_model)

        print(f"[BACG] Exploit: 在拓扑约束下执行 (max_depth={max_depth})")

        result = await self._execute_recursive(root, max_depth)

        total_cost = root.total_actual_cost
        summary = {
            "task": task,
            "total_cost": total_cost,
            "nodes": root.subtree_size,
            "max_depth": root.max_depth,
            "branching_mean": stats["mean"],
            "budget_remaining": {
                k: self.total_budget.get(k, 0) - total_cost.get(k, 0)
                for k in set(self.total_budget) | set(total_cost)
            },
        }

        return result, self.current_graph, summary

    def _build_estimate_graph(self, node: CallNode, task: str, depth: int = 0) -> None:
        """根据历史模型构建预估调用图。"""
        if depth >= 3:
            return

        m = self.branching_model.mean_branching
        n_children = min(int(m) + 1, 5)

        for i in range(n_children):
            op_types = ["llm_call", "tool_invoke", "search"]
            weights = [0.5, 0.3, 0.2]
            op = random.choices(op_types, weights=weights)[0]

            child = CallNode(
                node_id=f"{node.node_id}_child_{i}",
                op_type=op,
                estimated_cost=self.branching_model.mean_cost.copy(),
            )
            node.add_child(child)
            self._build_estimate_graph(child, task, depth + 1)

    async def _execute_recursive(self, node: CallNode, max_depth: int) -> Dict[str, Any]:
        """递归执行调用图节点。"""
        if node.depth >= max_depth:
            return {"status": "depth_limited", "node": node.node_id}

        if node.parent and node.parent.branch_factor_observed > 3.0:
            return {"status": "branch_limited", "node": node.node_id}

        await asyncio.sleep(0.01)

        actual = {}
        for k, v in self.branching_model.mean_cost.items():
            noise = random.gauss(0, v * 0.2)
            actual[k] = max(0, v + noise)
        node.actual_cost = actual

        n_children = len(node.children)
        node.branch_factor_observed = float(n_children)
        self.branching_model.observe(n_children, actual)

        if self.branching_model.mean_branching > 1.5 and n_children > 0:
            print(f"    ⚠️ 节点 {node.node_id}: 分支因子 {n_children} > 预期 {self.branching_model.mean_branching:.2f}")

        child_results = []
        for child in node.children:
            child_result = await self._execute_recursive(child, max_depth)
            child_results.append(child_result)

        return {
            "status": "completed",
            "node": node.node_id,
            "cost": actual,
            "children": child_results,
        }


# ─── BACG Agent v2 ───────────────────────────────────────────────

@dataclass
class BACGAgentV2:
    """BACG Agent v2：使用生产级 CallGraph。

    执行循环：Survey → Calibrate → [Expand → Check → (Contract|Rebalance)]*

    关键改进：
      1. CallGraph 是核心数据结构
      2. 6 层约束面独立工作
      3. 图变换而非简单加节点
      4. BudgetToken 在图上流动
    """

    name: str
    total_budget: Dict[str, float]
    graph: Optional[CallGraph] = None

    def __post_init__(self):
        self.graph = CallGraph(root_id=f"{self.name}_root")

        self.graph.add_layer(IdentityLayer(
            acl={"system": ["*"], "user": ["llm_call", "search"]}
        ))
        self.graph.add_layer(ResourceLayer(total_budget=self.total_budget))
        self.graph.add_layer(SecurityLayer())
        self.graph.add_layer(QualityLayer(decay_rate=0.9, min_quality=0.3))
        self.graph.add_layer(ConcurrencyLayer(max_width=5))
        self.graph.add_layer(TopologyLayer(max_depth=5, max_branching=3.0))

    async def execute(self, task: str) -> Dict[str, Any]:
        """执行循环：Survey → Calibrate → Exploit。"""
        print(f"\n{'='*60}")
        print(f"[BACG v2] Agent '{self.name}' executing: '{task}'")
        print(f"{'='*60}")

        # Step 1: SURVEY
        print("\n[1] SURVEY: 构建预估骨架图")
        branching_model = type('obj', (object,), {'mean_branching': 0.5})()
        self.graph.apply_survey(branching_model)
        stats = self.graph.branching_stats()
        print(f"    骨架: {stats['total_nodes']} 节点, "
              f"分支均值={stats['mean']:.2f}")

        # Step 2: CALIBRATE
        print(f"\n[2] CALIBRATE: 注入预算 Token")
        self.graph.apply_calibrate(self.total_budget)
        root = self.graph.nodes[self.graph.root_id]
        print(f"    根 Token: {root.received_tokens[0] if root.received_tokens else 'None'}")

        # Step 3: EXPLOIT
        print(f"\n[3] EXPLOIT: 执行并动态扩展")

        for node_id, node in list(self.graph.nodes.items()):
            if node_id == self.graph.root_id:
                continue

            if node.status == "pending":
                node.status = "running"
                node.started_at = time.time()

                if node.received_tokens:
                    dims = node.received_tokens[0].dimensions
                    node.consumed = {
                        k: v * random.uniform(0.1, 0.3)
                        for k, v in dims.items()
                    }

                await asyncio.sleep(0.001)
                node.status = "completed"
                node.completed_at = time.time()

        # Step 4: EXPAND
        print(f"\n[4] EXPAND: 模拟运行时扩展")
        expandable = [
            nid for nid, n in self.graph.nodes.items()
            if n.op_type == "llm_call" and n.status == "completed" and n.depth < 2
        ]
        if expandable:
            parent_id = expandable[0]
            new_children = self.graph.apply_expand(parent_id, [
                {"op_type": "search", "trigger_reason": "llm_decided_to_search"},
                {"op_type": "tool_invoke", "trigger_reason": "llm_decided_tool"},
            ])
            if new_children:
                print(f"    节点 '{parent_id}' 扩展了 {len(new_children)} 个子节点")
                for cid in new_children:
                    child = self.graph.nodes[cid]
                    print(f"      {cid}: op={child.op_type}, "
                          f"tokens={child.remaining.get('tokens', 0):.0f}")

        # Step 5: 全局检查
        print(f"\n[5] GLOBAL CHECK")
        passed, violations = self.graph.check_all()
        if violations:
            for v in violations:
                print(f"    ⚠️ {v}")
        else:
            print(f"    ✅ 所有约束满足")

        final_stats = self.graph.branching_stats()
        total_consumed = self.graph.total_consumed()

        print(f"\n{'='*60}")
        print(f"执行完成:")
        print(f"  总节点: {final_stats['total_nodes']}")
        print(f"  总边: {final_stats['total_edges']}")
        print(f"  分支均值: {final_stats['mean']:.2f}")
        print(f"  总消耗: {total_consumed}")
        print(f"  剩余: " + ", ".join(
            f"{k}={self.total_budget.get(k,0) - total_consumed.get(k,0):.2f}"
            for k in set(self.total_budget) | set(total_consumed)
        ))
        print(f"{'='*60}")

        return {
            "agent": self.name,
            "task": task,
            "nodes": final_stats['total_nodes'],
            "edges": final_stats['total_edges'],
            "branching_mean": final_stats['mean'],
            "total_consumed": total_consumed,
            "remaining": {
                k: self.total_budget.get(k, 0) - total_consumed.get(k, 0)
                for k in set(self.total_budget) | set(total_consumed)
            },
        }
