"""BranchingModel — 分支过程模型（Galton-Watson 过程）

核心参数：
  m = E[Z]：平均分支因子
  σ² = Var(Z)：分支因子方差
  E[C]：每次调用的平均消耗

关键公式：
  期望总消耗（m < 1）：E[Total] = E[C] / (1 - m)
  方差（m < 1）：Var(Total) = σ²·E[C]² / (1 - m)³
  灭绝概率（m ≤ 1）：q = 1（必然灭绝）
  灭绝概率（m > 1）：q 是方程 q = G(q) 的最小根
"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional


@dataclass
class BranchingModel:
    """分支过程模型：建模 Agent 调用的级联行为。"""

    mean_branching: float = 1.0       # m = E[Z]
    variance_branching: float = 0.5   # σ² = Var(Z)
    mean_cost: Dict[str, float] = field(default_factory=lambda: {"tokens": 1000, "usd": 0.03})
    variance_cost: Dict[str, float] = field(default_factory=lambda: {"tokens": 10000, "usd": 0.001})

    # 历史观测
    observed_branching: Deque[float] = field(default_factory=lambda: deque(maxlen=50))
    observed_costs: Dict[str, Deque[float]] = field(default_factory=dict)

    def observe(self, n_children: int, cost: Dict[str, float]) -> None:
        """观测一次调用的分支因子和消耗。"""
        self.observed_branching.append(float(n_children))
        for k, v in cost.items():
            if k not in self.observed_costs:
                self.observed_costs[k] = deque(maxlen=50)
            self.observed_costs[k].append(v)
        self._update_estimates()

    def _update_estimates(self) -> None:
        """根据观测更新模型参数。"""
        if len(self.observed_branching) > 0:
            self.mean_branching = sum(self.observed_branching) / len(self.observed_branching)
            if len(self.observed_branching) > 1:
                mean = self.mean_branching
                self.variance_branching = sum((b - mean) ** 2 for b in self.observed_branching) / len(self.observed_branching)

        for k, values in self.observed_costs.items():
            if len(values) > 0:
                self.mean_cost[k] = sum(values) / len(values)

    def expected_total_cost(self) -> Optional[Dict[str, float]]:
        """计算期望总消耗。

        Returns:
            None 如果 m >= 1（期望无穷大）
            消耗字典 如果 m < 1
        """
        if self.mean_branching >= 1.0:
            return None

        factor = 1.0 / (1.0 - self.mean_branching)
        return {
            k: v * factor
            for k, v in self.mean_cost.items()
        }

    def survival_probability(self, max_depth: int) -> float:
        """计算调用树在 max_depth 层内不灭绝的概率。"""
        if self.mean_branching <= 1.0:
            return 1.0
        return min(1.0, (self.mean_branching ** (-max_depth)))

    def extinction_probability(self) -> float:
        """计算灭绝概率。"""
        if self.mean_branching <= 1.0:
            return 1.0
        q = 0.5
        for _ in range(10):
            if len(self.observed_branching) > 0:
                g = sum(b ** q for b in self.observed_branching) / len(self.observed_branching)
            else:
                g = q ** self.mean_branching
            q = g
        return q

    def recommend_depth_limit(self, total_budget: Dict[str, float]) -> int:
        """根据总预算推荐最大深度限制。"""
        m = self.mean_branching
        if m >= 1.0:
            return 3

        token_budget = total_budget.get("tokens", 10000)
        token_mean = self.mean_cost.get("tokens", 1000)

        ratio = token_budget / token_mean * (1 - m)
        if ratio <= 1:
            return 1

        depth = int(math.log(ratio) / math.log(1 / m))
        return max(1, min(depth, 10))
