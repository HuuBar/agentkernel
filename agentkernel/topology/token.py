"""BudgetToken — 流动的资源令牌（CHERI Capability Monotonicity）

借鉴 Petri 网的 token 机制和 CHERI 的能力单调性：
  - Token 在传播中总量只能减少（不能凭空产生资源）
  - Token 记录完整路径（OpenTelemetry Span 风格）
  - Token 可以被回收（reclaim）如果子节点失败

CHERI Monotonicity: capability.derivation can only REDUCE rights.
Budget Monotonicity: token.split() can only REDUCE dimensions.
"""
from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Tuple


@dataclass
class BudgetToken:
    """预算令牌：在调用树上流动的多维资源单位。"""

    dimensions: Dict[str, float] = field(default_factory=dict)
    path: List[str] = field(default_factory=list)
    source_node: str = "root"
    created_at: float = field(default_factory=time.time)
    overhead_rate: float = 0.05  # 5% 传播开销（管理成本）

    @property
    def total_value(self) -> float:
        """Token 的总价值（标量摘要）。"""
        return sum(self.dimensions.values())

    def can_afford(self, cost: Dict[str, float]) -> bool:
        """检查 Token 是否能支付给定成本。"""
        return all(
            self.dimensions.get(k, 0) >= v 
            for k, v in cost.items()
        )

    def consume(self, cost: Dict[str, float]) -> Dict[str, float]:
        """消耗资源，返回剩余。"""
        remaining = {}
        for k, v in cost.items():
            available = self.dimensions.get(k, 0)
            if available < v:
                raise InsufficientBudget(
                    f"维度 '{k}': 需要 {v:.2f}, 只有 {available:.2f}, "
                    f"路径: {' -> '.join(self.path)}"
                )
            remaining[k] = available - v
        self.dimensions = remaining
        return remaining

    def split(self, child_specs: List[Tuple[str, float]]) -> List[BudgetToken]:
        """将 Token 分裂给多个子节点。

        Args:
            child_specs: [(child_name, ratio), ...], ratios 之和 <= 1.0

        Returns:
            子 Token 列表，每个子 Token 的维度 = 父维度 * ratio * (1 - overhead)

        单调性保证：sum(子 Token.dimensions) <= 父 Token.dimensions
        """
        total_ratio = sum(r for _, r in child_specs)
        if total_ratio > 1.0:
            raise ValueError(f"分裂比例之和 {total_ratio} > 1.0，违反单调性")

        children = []
        for child_name, ratio in child_specs:
            child_dims = {
                k: max(0.0, v * ratio * (1 - self.overhead_rate))
                for k, v in self.dimensions.items()
            }
            children.append(BudgetToken(
                dimensions=child_dims,
                path=self.path + [child_name],
                source_node=child_name,
                overhead_rate=self.overhead_rate,
            ))

        # 验证单调性
        total_children = defaultdict(float)
        for child in children:
            for k, v in child.dimensions.items():
                total_children[k] += v

        for k, v in self.dimensions.items():
            assert total_children[k] <= v * 1.001, (
                f"单调性违反: {k} 子总和 {total_children[k]:.4f} > 父 {v:.4f}"
            )

        return children

    def reclaim(self, child_token: BudgetToken) -> None:
        """回收子节点的剩余 Token（子节点失败时）。"""
        for k, v in child_token.dimensions.items():
            self.dimensions[k] = self.dimensions.get(k, 0) + v

    def __repr__(self) -> str:
        dims_str = ", ".join(f"{k}={v:.2f}" for k, v in self.dimensions.items())
        path_str = " -> ".join(self.path) if self.path else "root"
        return f"Token[{dims_str} @ {path_str}]"


class InsufficientBudget(Exception):
    """预算不足异常。"""
    pass
