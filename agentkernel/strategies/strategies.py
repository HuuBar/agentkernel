"""AgentKernel — Allocation Strategy System

CaaE 的核心不是"声明效果"，而是"如何分配"。
本模块提供多种分配策略、策略选择框架、历史学习机制。
"""
from __future__ import annotations

import json
import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple
from datetime import datetime
from uuid import UUID, uuid4

from agentkernel.caae import ResourceBudget, ExecutionContext


# ═══════════════════════════════════════════════════════════════
# Task Descriptor — 任务描述符（策略决策的输入）
# ═══════════════════════════════════════════════════════════════

@dataclass
class TaskDescriptor:
    """任务的复杂度描述，供分配策略使用。"""
    task_type: str                    # "data_analysis", "code_generation", "research"
    estimated_steps: int = 1           # 预估步骤数
    requires_tools: List[str] = field(default_factory=list)   # 需要的工具
    requires_llm: bool = True         # 是否需要 LLM
    expected_output_size: str = "medium"  # small/medium/large
    priority: int = 1                   # 1-5，越高越重要
    deadline_seconds: Optional[float] = None  # 时间约束
    
    def complexity_score(self) -> float:
        """计算任务复杂度分数（0-1），供策略使用。"""
        score = 0.0
        score += min(self.estimated_steps / 20, 1.0) * 0.3  # 步骤复杂度
        score += min(len(self.requires_tools) / 5, 1.0) * 0.2  # 工具复杂度
        score += {"small": 0.1, "medium": 0.3, "large": 0.6}.get(self.expected_output_size, 0.3)
        score += (self.priority - 1) / 4 * 0.1  # 优先级微调
        return min(score, 1.0)


# ═══════════════════════════════════════════════════════════════
# Strategy Base — 策略基类
# ═══════════════════════════════════════════════════════════════

@dataclass
class AllocationStrategy(ABC):
    """分配策略基类。
    
    核心接口：输入（父预算、任务描述、子能力）→ 输出（分配方案）。
    所有策略遵循同一契约，可插拔替换。
    """
    name: str = "base"
    
    @abstractmethod
    def allocate(
        self,
        parent_budget: ResourceBudget,
        task: TaskDescriptor,
        sibling_count: int = 1,           # 同级 Worker 数量
        sibling_index: int = 0,           # 当前 Worker 序号
        history: Optional[TaskHistory] = None,
    ) -> Dict[str, float]:
        """返回分配给子 Agent 的预算字典。"""
        pass
    
    def allocate_all(
        self,
        parent_budget: ResourceBudget,
        tasks: List[TaskDescriptor],
        history: Optional[TaskHistory] = None,
    ) -> List[Dict[str, float]]:
        """为多个子 Agent 批量分配预算。"""
        allocations = []
        for i, task in enumerate(tasks):
            alloc = self.allocate(
                parent_budget=parent_budget,
                task=task,
                sibling_count=len(tasks),
                sibling_index=i,
                history=history,
            )
            allocations.append(alloc)
        return allocations


# ═══════════════════════════════════════════════════════════════
# Strategy 1: Conservative Equal Split — 保守均分（默认策略）
# ═══════════════════════════════════════════════════════════════

@dataclass
class ConservativeEqualSplit(AllocationStrategy):
    """保守均分策略。
    
    核心思想：平均分配，但每个子 Agent 只获得计算值的 80%，
    预留 20% 作为 Manager 的应急 buffer。
    
    适用场景：任务同质、风险未知、需要保守执行。
    """
    name: str = "conservative_equal"
    buffer_ratio: float = 0.2  # 预留 20%
    
    def allocate(
        self,
        parent_budget: ResourceBudget,
        task: TaskDescriptor,
        sibling_count: int = 1,
        sibling_index: int = 0,
        history: Optional[TaskHistory] = None,
    ) -> Dict[str, float]:
        # 可用预算 = 总预算 - buffer
        available = {
            "tokens": parent_budget.tokens * (1 - self.buffer_ratio),
            "usd": parent_budget.usd * (1 - self.buffer_ratio),
            "seconds": parent_budget.seconds * (1 - self.buffer_ratio),
            "api_calls": parent_budget.api_calls * (1 - self.buffer_ratio),
        }
        # 均分
        per_child = {
            k: v / sibling_count for k, v in available.items()
        }
        return per_child


# ═══════════════════════════════════════════════════════════════
# Strategy 2: Historical Allocation — 历史学习型
# ═══════════════════════════════════════════════════════════════

@dataclass
class HistoricalAllocation(AllocationStrategy):
    """基于历史数据的学习型策略。
    
    核心思想：查询该类型任务的历史执行记录，
    取 P75 或 P95 分位数作为分配基准。
    
    适用场景：有历史数据、任务类型可分类、愿意用数据驱动决策。
    """
    name: str = "historical"
    percentile: float = 0.75  # 取 75 分位数（比均值更保守）
    safety_multiplier: float = 1.2  # 额外 20% 安全余量
    
    def allocate(
        self,
        parent_budget: ResourceBudget,
        task: TaskDescriptor,
        sibling_count: int = 1,
        sibling_index: int = 0,
        history: Optional[TaskHistory] = None,
    ) -> Dict[str, float]:
        if history is None or not history.has_data(task.task_type):
            # 无历史数据，回退到保守均分
            return ConservativeEqualSplit().allocate(
                parent_budget, task, sibling_count, sibling_index, history
            )
        
        # 查询历史统计
        stats = history.get_stats(task.task_type)
        
        # 基于 P75 + 安全余量分配
        base = {
            "tokens": stats.token_p75 * self.safety_multiplier,
            "usd": stats.usd_p75 * self.safety_multiplier,
            "seconds": stats.time_p75 * self.safety_multiplier,
            "api_calls": stats.api_calls_p75 * self.safety_multiplier,
        }
        
        # 如果历史基准超过可用预算，裁剪到预算范围内
        max_alloc = {
            "tokens": parent_budget.tokens / sibling_count,
            "usd": parent_budget.usd / sibling_count,
            "seconds": parent_budget.seconds / sibling_count,
            "api_calls": parent_budget.api_calls / sibling_count,
        }
        
        return {
            k: min(base.get(k, 0), max_alloc.get(k, 0)) 
            for k in ["tokens", "usd", "seconds", "api_calls"]
        }


# ═══════════════════════════════════════════════════════════════
# Strategy 3: Adaptive Complexity Split — 任务复杂度自适应
# ═══════════════════════════════════════════════════════════════

@dataclass
class AdaptiveComplexitySplit(AllocationStrategy):
    """根据任务复杂度动态调整分配比例。
    
    核心思想：不是均分，而是按复杂度加权分配。
    复杂任务获得更多预算，简单任务获得更少。
    
    适用场景：任务异质、复杂度差异大、需要差异化分配。
    """
    name: str = "adaptive_complexity"
    min_ratio: float = 0.5  # 最低分配比例（防止简单任务被饿死）
    
    def allocate_all(
        self,
        parent_budget: ResourceBudget,
        tasks: List[TaskDescriptor],
        history: Optional[TaskHistory] = None,
    ) -> List[Dict[str, float]]:
        # 计算每个任务的复杂度分数
        scores = [t.complexity_score() for t in tasks]
        total_score = sum(scores) or 1.0
        
        allocations = []
        for i, (task, score) in enumerate(zip(tasks, scores)):
            # 加权比例，但有最低保障
            ratio = max(score / total_score, self.min_ratio / len(tasks))
            
            alloc = {
                "tokens": parent_budget.tokens * ratio,
                "usd": parent_budget.usd * ratio,
                "seconds": parent_budget.seconds * ratio,
                "api_calls": parent_budget.api_calls * ratio,
            }
            allocations.append(alloc)
        
        return allocations
    
    def allocate(
        self,
        parent_budget: ResourceBudget,
        task: TaskDescriptor,
        sibling_count: int = 1,
        sibling_index: int = 0,
        history: Optional[TaskHistory] = None,
    ) -> Dict[str, float]:
        # 单任务调用时回退到均分
        return ConservativeEqualSplit().allocate(
            parent_budget, task, sibling_count, sibling_index, history
        )


# ═══════════════════════════════════════════════════════════════
# Strategy 4: Greedy Reservation — 贪婪预留
# ═══════════════════════════════════════════════════════════════

@dataclass
class GreedyReservation(AllocationStrategy):
    """贪婪预留策略。
    
    核心思想：Manager 自己预留大部分预算，只给 Worker 最小必要预算。
    适合 Manager 需要大量资源做汇总/验证的场景。
    
    适用场景：Manager 工作量大、Worker 只是辅助、不信任 Worker。
    """
    name: str = "greedy"
    worker_ratio: float = 0.3  # Worker 只获得 30%
    
    def allocate(
        self,
        parent_budget: ResourceBudget,
        task: TaskDescriptor,
        sibling_count: int = 1,
        sibling_index: int = 0,
        history: Optional[TaskHistory] = None,
    ) -> Dict[str, float]:
        # Worker 获得总预算的 30%，再均分
        worker_pool = {
            "tokens": parent_budget.tokens * self.worker_ratio,
            "usd": parent_budget.usd * self.worker_ratio,
            "seconds": parent_budget.seconds * self.worker_ratio,
            "api_calls": parent_budget.api_calls * self.worker_ratio,
        }
        return {
            k: v / sibling_count for k, v in worker_pool.items()
        }


# ═══════════════════════════════════════════════════════════════
# Strategy 5: Dynamic Reallocation — 动态重分配
# ═══════════════════════════════════════════════════════════════

@dataclass
class DynamicReallocation(AllocationStrategy):
    """动态重分配策略。
    
    核心思想：第一轮分配保守，观察前序 Worker 的实际消耗后，
    动态调整后序 Worker 的分配。
    
    适用场景：任务串行执行、前序任务能提供后序参考。
    """
    name: str = "dynamic"
    base_strategy: AllocationStrategy = field(default_factory=ConservativeEqualSplit)
    adjustment_factor: float = 1.0  # 根据前序实际消耗的修正系数
    
    def allocate(
        self,
        parent_budget: ResourceBudget,
        task: TaskDescriptor,
        sibling_count: int = 1,
        sibling_index: int = 0,
        history: Optional[TaskHistory] = None,
    ) -> Dict[str, float]:
        base = self.base_strategy.allocate(
            parent_budget, task, sibling_count, sibling_index, history
        )
        
        if history is None or sibling_index == 0:
            return base
        
        # 查询前序 Worker 的实际消耗 vs 预估消耗
        prev_actual = history.get_last_actual(task.task_type)
        if prev_actual is None:
            return base
        
        # 修正：如果前序实际消耗远低于预估，增加后序分配
        # 如果前序实际消耗接近预估，保持保守
        ratio = prev_actual["usd"] / (base.get("usd", 1) or 1)
        
        if ratio < 0.5:
            # 前序消耗远低于预估，后序可以更激进
            factor = 1.5
        elif ratio > 0.9:
            # 前序接近预估，保持保守
            factor = 0.8
        else:
            factor = 1.0
        
        return {
            k: v * factor for k, v in base.items()
        }


# ═══════════════════════════════════════════════════════════════
# Task History — 历史记录系统（学习型策略的数据源）
# ═══════════════════════════════════════════════════════════════

@dataclass
class TaskExecutionRecord:
    """单次任务执行记录。"""
    run_id: UUID
    task_type: str
    strategy_used: str
    allocated_budget: Dict[str, float]
    actual_consumed: Dict[str, float]
    success: bool
    duration_seconds: float
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class TaskTypeStats:
    """某类任务的历史统计。"""
    task_type: str
    count: int = 0
    token_mean: float = 0.0
    token_p75: float = 0.0
    token_p95: float = 0.0
    usd_mean: float = 0.0
    usd_p75: float = 0.0
    usd_p95: float = 0.0
    time_mean: float = 0.0
    time_p75: float = 0.0
    api_calls_mean: float = 0.0
    api_calls_p75: float = 0.0
    success_rate: float = 0.0


@dataclass
class TaskHistory:
    """任务历史记录系统。"""
    records: List[TaskExecutionRecord] = field(default_factory=list)
    _stats_cache: Dict[str, TaskTypeStats] = field(default_factory=dict)
    
    def record(
        self,
        task_type: str,
        strategy: str,
        allocated: Dict[str, float],
        actual: Dict[str, float],
        success: bool,
        duration: float,
    ) -> None:
        self.records.append(TaskExecutionRecord(
            run_id=uuid4(),
            task_type=task_type,
            strategy_used=strategy,
            allocated_budget=allocated,
            actual_consumed=actual,
            success=success,
            duration_seconds=duration,
        ))
        # 清除缓存，下次查询时重新计算
        self._stats_cache.pop(task_type, None)
    
    def has_data(self, task_type: str) -> bool:
        return any(r.task_type == task_type for r in self.records)
    
    def get_stats(self, task_type: str) -> TaskTypeStats:
        """计算某类任务的统计信息。"""
        if task_type in self._stats_cache:
            return self._stats_cache[task_type]
        
        relevant = [r for r in self.records if r.task_type == task_type]
        if not relevant:
            return TaskTypeStats(task_type=task_type)
        
        def percentile(values: List[float], p: float) -> float:
            s = sorted(values)
            idx = int(len(s) * p)
            return s[min(idx, len(s)-1)]
        
        tokens = [r.actual_consumed.get("tokens", 0) for r in relevant]
        usds = [r.actual_consumed.get("usd", 0) for r in relevant]
        times = [r.actual_consumed.get("seconds", 0) for r in relevant]
        apis = [r.actual_consumed.get("api_calls", 0) for r in relevant]
        
        stats = TaskTypeStats(
            task_type=task_type,
            count=len(relevant),
            token_mean=sum(tokens) / len(tokens),
            token_p75=percentile(tokens, 0.75),
            token_p95=percentile(tokens, 0.95),
            usd_mean=sum(usds) / len(usds),
            usd_p75=percentile(usds, 0.75),
            usd_p95=percentile(usds, 0.95),
            time_mean=sum(times) / len(times),
            time_p75=percentile(times, 0.75),
            api_calls_mean=sum(apis) / len(apis),
            api_calls_p75=percentile(apis, 0.75),
            success_rate=sum(1 for r in relevant if r.success) / len(relevant),
        )
        self._stats_cache[task_type] = stats
        return stats
    
    def get_last_actual(self, task_type: str) -> Optional[Dict[str, float]]:
        """获取最近一次的实际消耗。"""
        relevant = [r for r in self.records if r.task_type == task_type]
        if not relevant:
            return None
        return relevant[-1].actual_consumed
    
    def summary(self) -> str:
        """历史摘要。"""
        lines = [f"TaskHistory: {len(self.records)} records"]
        types = set(r.task_type for r in self.records)
        for t in sorted(types):
            stats = self.get_stats(t)
            lines.append(
                f"  {t}: n={stats.count}, "
                f"usd_mean=${stats.usd_mean:.3f}, "
                f"usd_p75=${stats.usd_p75:.3f}, "
                f"success={stats.success_rate:.0%}"
            )
        return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════
# Strategy Selector — 策略选择框架
# ═══════════════════════════════════════════════════════════════

@dataclass
class StrategySelector:
    """策略选择器。
    
    核心问题：给定场景，应该用什么策略？
    基于任务特征、历史数据、风险偏好自动选择。
    """
    
    # 策略注册表
    strategies: Dict[str, AllocationStrategy] = field(default_factory=dict)
    
    def register(self, name: str, strategy: AllocationStrategy) -> None:
        self.strategies[name] = strategy
    
    def select(
        self,
        tasks: List[TaskDescriptor],
        history: Optional[TaskHistory] = None,
        risk_tolerance: str = "medium",  # low/medium/high
    ) -> AllocationStrategy:
        """根据场景特征选择最优策略。"""
        
        # 规则 1: 有历史数据 + 低风险偏好 → 历史学习型
        if history and any(history.has_data(t.task_type) for t in tasks):
            if risk_tolerance == "low" and "historical" in self.strategies:
                return self.strategies["historical"]
        
        # 规则 2: 任务异质 + 复杂度差异大 → 自适应复杂度
        complexity_variance = self._complexity_variance(tasks)
        if complexity_variance > 0.3 and "adaptive" in self.strategies:
            return self.strategies["adaptive"]
        
        # 规则 3: 高风险偏好 + 需要快速完成 → 贪婪预留
        if risk_tolerance == "high" and "greedy" in self.strategies:
            return self.strategies["greedy"]
        
        # 规则 4: 串行任务 + 有历史 → 动态重分配
        if len(tasks) > 1 and history and "dynamic" in self.strategies:
            # 检查是否有串行特征
            return self.strategies["dynamic"]
        
        # 默认: 保守均分（最安全）
        return self.strategies.get("conservative", ConservativeEqualSplit())
    
    def _complexity_variance(self, tasks: List[TaskDescriptor]) -> float:
        """计算任务复杂度的方差。"""
        if len(tasks) <= 1:
            return 0.0
        scores = [t.complexity_score() for t in tasks]
        mean = sum(scores) / len(scores)
        variance = sum((s - mean) ** 2 for s in scores) / len(scores)
        return math.sqrt(variance)
