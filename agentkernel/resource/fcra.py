"""AgentKernel — Feedback-Controlled Resource Allocation (FCRA)

核心范式：不是静态分配，而是运行时根据实际消耗动态调整的反馈回路。

借鉴了以下成熟方案：
- Linux CGroup cpu.cfs_burst_us — 信用累积
- Token Bucket — 突发允许
- SJF指数平滑 — 消耗预测
- PID控制器 — 反馈调节
- Circuit Breaker — 异常熔断
- 追踪止损 — 动态上限

解决的核心问题：
  "平时只用1%额度，但一张图片调研突然用掉70%，事先无法预知，事后无法补偿。"

五个反馈回路层：
  1. Credit Accumulation — 信用累积（平时攒信用，突发时可用）
  2. Exponential Smoothing — 消耗预测（基于历史预测下次消耗，偏差>3倍标记异常）
  3. PID Controller — 反馈调节（消耗过快时自动减少后续分配，过慢时增加）
  4. Circuit Breaker — 异常熔断（消耗速率>历史平均3倍时自动限制）
  5. Trailing Stop Loss — 追踪止损（根据任务进展动态调整剩余预算上限）
"""
from __future__ import annotations

import asyncio
import math
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Deque, Dict, List, Optional, Tuple
from collections import deque

from agentkernel.caae import (
    ResourceBudget, ExecutionContext, Effect, EffectKind,
    EffectHandler, InsufficientBudget,
)


# ═══════════════════════════════════════════════════════════════
# 1. Credit Accumulation — 信用累积层（借鉴 Linux CGroup burst）
# ═══════════════════════════════════════════════════════════════

@dataclass
class BurstCredit:
    """突发信用池。
    
    核心思想（来自 Linux cpu.cfs_burst_us）：
    - 平时低消耗时积累未使用的配额作为"信用"
    - 突发请求时可以从信用池透支
    - 信用有上限，防止无限累积
    
    类比：手机流量本月用不完，可以累积到下月，但有上限。
    """
    accumulated: float = 0.0          # 当前累积的信用
    max_limit: float = 5000.0         # 信用上限（如 5× 基础配额）
    accumulation_rate: float = 0.8    # 未使用配额的累积比例（80%存起来）
    decay_rate: float = 0.01          # 每天衰减 1%（防止旧信用无限期有效）
    
    def add_unused(self, allocated: float, actual: float) -> float:
        """将未使用的配额转化为信用。
        
        Args:
            allocated: 分配的配额
            actual: 实际使用的配额
            
        Returns:
            新增的信用量
        """
        unused = max(0, allocated - actual)
        credit = unused * self.accumulation_rate
        self.accumulated = min(self.accumulated + credit, self.max_limit)
        return credit
    
    def can_borrow(self, amount: float) -> bool:
        """检查是否可以借用指定量的信用。"""
        return self.accumulated >= amount
    
    def borrow(self, amount: float) -> float:
        """借用信用。返回实际借到的量（可能不足）。"""
        borrowed = min(amount, self.accumulated)
        self.accumulated -= borrowed
        return borrowed
    
    def decay(self, days: float = 1.0) -> None:
        """信用衰减（旧信用失效）。"""
        self.accumulated *= (1 - self.decay_rate) ** days


# ═══════════════════════════════════════════════════════════════
# 2. Exponential Smoothing — 指数平滑预测层（借鉴 SJF 调度）
# ═══════════════════════════════════════════════════════════════

@dataclass
class ConsumptionPredictor:
    """基于指数平滑的消耗预测器。
    
    核心思想（来自 SJF 调度算法的指数平均）：
      T_{n+1} = α · t_n + (1-α) · T_n
    
    - T_{n+1}: 下一次预测的消耗
    - t_n: 上一次实际消耗
    - T_n: 上一次预测消耗
    - α: 平滑因子（0.5 = 同等重视历史和新数据）
    
    异常检测：如果 actual / predicted > threshold（如 3.0），
    标记为"突发消耗异常"。
    """
    alpha: float = 0.5                # 平滑因子
    threshold: float = 3.0            # 异常阈值（3倍偏差）
    
    # 历史记录
    predicted_history: Deque[float] = field(default_factory=lambda: deque([1000.0], maxlen=100))
    actual_history: Deque[float] = field(default_factory=lambda: deque(maxlen=100))
    
    def predict(self) -> float:
        """预测下一次的消耗量。"""
        if len(self.actual_history) == 0:
            return self.predicted_history[-1]
        last_predicted = self.predicted_history[-1]
        last_actual = self.actual_history[-1]
        prediction = self.alpha * last_actual + (1 - self.alpha) * last_predicted
        self.predicted_history.append(prediction)
        return prediction
    
    def record_actual(self, actual: float) -> Tuple[str, float]:
        """记录实际消耗，返回（状态, 偏差率）。
        
        Returns:
            ("normal", ratio) — 正常消耗
            ("warning", ratio) — 轻微异常（2-3倍）
            ("burst", ratio) — 突发异常（>3倍）
        """
        self.actual_history.append(actual)
        
        if len(self.predicted_history) == 0:
            return ("normal", 1.0)
        
        predicted = self.predicted_history[-1]
        if predicted == 0:
            return ("normal", 1.0)
        
        ratio = actual / predicted
        
        if ratio > self.threshold:
            return ("burst", ratio)
        elif ratio > self.threshold * 0.67:  # 2.0
            return ("warning", ratio)
        else:
            return ("normal", ratio)
    
    def is_anomaly(self, actual: float) -> bool:
        """检查是否为异常消耗。"""
        status, _ = self.record_actual(actual)
        return status == "burst"


# ═══════════════════════════════════════════════════════════════
# 3. PID Controller — 反馈调节层（借鉴控制理论）
# ═══════════════════════════════════════════════════════════════

@dataclass
class PIDController:
    """PID 控制器，用于动态调整资源分配。
    
    核心思想：根据目标消耗速率与实际速率的误差，
    按比例(P)、积分(I)、微分(D)三个维度调整输出。
    
    公式：
      u(t) = Kp·e(t) + Ki·∫e(t)dt + Kd·de(t)/dt
    
    其中 e(t) = target_rate - actual_rate
    
    Agent 场景映射：
    - 目标速率：每步希望消耗 1000 tokens
    - 实际速率：上步消耗了 3000 tokens（过快！）
    - 误差：1000 - 3000 = -2000
    - PID 输出：负值 → 减少后续分配
    """
    kp: float = 0.5      # 比例增益
    ki: float = 0.1      # 积分增益
    kd: float = 0.2      # 微分增益
    
    target_rate: float = 1000.0   # 目标消耗速率（tokens/步）
    
    # 内部状态
    error_integral: float = 0.0
    last_error: float = 0.0
    last_time: float = field(default_factory=time.time)
    
    def compute(self, actual_rate: float) -> float:
        """计算 PID 输出。
        
        Returns:
            调整因子（1.0 = 不变，<1.0 = 减少分配，>1.0 = 增加分配）
        """
        current_time = time.time()
        dt = current_time - self.last_time
        if dt == 0:
            dt = 1.0
        
        # 误差
        error = self.target_rate - actual_rate
        
        # 积分
        self.error_integral += error * dt
        self.error_integral = max(min(self.error_integral, 10000), -10000)  # 防止积分饱和
        
        # 微分
        derivative = (error - self.last_error) / dt
        
        # PID 输出
        output = self.kp * error + self.ki * self.error_integral + self.kd * derivative
        
        # 映射为调整因子
        # 输出为负 → 减少分配；输出为正 → 增加分配
        # 用 sigmoid 映射到 [0.5, 2.0]
        adjustment = 1.0 + math.tanh(output / self.target_rate) * 0.5
        
        # 更新状态
        self.last_error = error
        self.last_time = current_time
        
        return adjustment
    
    def reset(self) -> None:
        """重置控制器状态。"""
        self.error_integral = 0.0
        self.last_error = 0.0
        self.last_time = time.time()


# ═══════════════════════════════════════════════════════════════
# 4. Circuit Breaker — 异常熔断层
# ═══════════════════════════════════════════════════════════════

class CircuitState:
    CLOSED = "closed"       # 正常状态，允许通过
    OPEN = "open"           # 熔断状态，拒绝请求
    HALF_OPEN = "half_open" # 半开状态，试探性允许

@dataclass
class CircuitBreaker:
    """断路器，防止级联资源失控。
    
    核心思想（来自分布式系统的熔断模式）：
    - 监控消耗速率 vs 历史平均
    - 如果速率 > 历史平均 × multiplier，触发熔断
    - 熔断后拒绝请求或降级处理
    - 一段时间后进入半开状态，试探性恢复
    
    Agent 场景：
    - 正常：Agent 每步消耗 1000 tokens
    - 异常：Agent 进入递归循环，每秒 2341 tokens
    - 断路器检测到速率 > 3× 平均值，立即熔断
    """
    multiplier: float = 3.0           # 触发熔断的倍数
    recovery_time: float = 60.0         # 熔断后恢复时间（秒）
    half_open_max: int = 3            # 半开状态最多允许 3 个试探请求
    
    state: str = CircuitState.CLOSED
    last_failure_time: float = 0.0
    half_open_count: int = 0
    
    # 历史平均速率（滑动窗口）
    rate_history: Deque[float] = field(default_factory=lambda: deque(maxlen=20))
    
    def check(self, current_rate: float) -> bool:
        """检查是否允许通过。
        
        Returns:
            True — 允许通过
            False — 触发熔断，拒绝请求
        """
        if self.state == CircuitState.OPEN:
            if time.time() - self.last_failure_time > self.recovery_time:
                self.state = CircuitState.HALF_OPEN
                self.half_open_count = 0
            else:
                return False
        
        if self.state == CircuitState.HALF_OPEN:
            self.half_open_count += 1
            if self.half_open_count > self.half_open_max:
                self.state = CircuitState.OPEN
                return False
            return True
        
        # CLOSED 状态：检查速率
        if len(self.rate_history) > 0:
            avg_rate = sum(self.rate_history) / len(self.rate_history)
            if current_rate > avg_rate * self.multiplier:
                self.state = CircuitState.OPEN
                self.last_failure_time = time.time()
                return False
        
        self.rate_history.append(current_rate)
        return True
    
    def record_success(self) -> None:
        """记录成功，用于半开状态恢复。"""
        if self.state == CircuitState.HALF_OPEN:
            self.state = CircuitState.CLOSED
            self.half_open_count = 0
    
    def record_failure(self) -> None:
        """记录失败，重新熔断。"""
        self.state = CircuitState.OPEN
        self.last_failure_time = time.time()


# ═══════════════════════════════════════════════════════════════
# 5. Trailing Stop Loss — 追踪止损层（借鉴金融风控）
# ═══════════════════════════════════════════════════════════════

@dataclass
class TrailingStopLoss:
    """追踪止损，动态调整剩余预算上限。
    
    核心思想（来自金融交易的追踪止损）：
    - 不是固定上限（如 "最多花 $10"），而是动态上限
    - 上限根据任务进展动态调整：进展好则放宽，进展差则收紧
    - 类比：股票涨了，止损点跟着上移；股票跌了，止损点不动
    
    Agent 场景：
    - 初始：预算 $10，追踪止损 = $8（允许花 $2）
    - Agent 进展顺利：追踪止损上调到 $9（允许花更多）
    - Agent 进展停滞：追踪止损保持 $8（限制追加投入）
    """
    total_budget: float = 10.0        # 总预算
    initial_stop: float = 0.8         # 初始止损比例（80%）
    trailing_step: float = 0.05       # 每次进展上调 5%
    
    highest_progress: float = 0.0     # 最高进展记录
    current_stop_level: float = 0.0   # 当前止损水平
    
    def __post_init__(self):
        self.current_stop_level = self.total_budget * self.initial_stop
    
    def update(self, consumed: float, progress: float) -> Tuple[bool, float]:
        """更新追踪止损。
        
        Args:
            consumed: 已消耗预算
            progress: 当前进展（0-1）
            
        Returns:
            (是否允许继续, 剩余可用预算)
        """
        # 如果进展创新高，上调止损点
        if progress > self.highest_progress:
            self.highest_progress = progress
            new_stop = self.total_budget * (self.initial_stop + progress * (1 - self.initial_stop))
            self.current_stop_level = max(self.current_stop_level, new_stop)
        
        # 检查是否触发止损
        remaining = self.current_stop_level - consumed
        if remaining <= 0:
            return (False, 0.0)
        
        return (True, remaining)
    
    def get_status(self) -> Dict[str, float]:
        """获取当前状态。"""
        return {
            "total": self.total_budget,
            "stop_level": self.current_stop_level,
            "highest_progress": self.highest_progress,
            "available": self.current_stop_level - self.highest_progress * self.total_budget,
        }


# ═══════════════════════════════════════════════════════════════
# 6. FeedbackController — 五层反馈控制整合
# ═══════════════════════════════════════════════════════════════

@dataclass
class FeedbackController:
    """五层反馈控制器：整合信用累积、预测、PID、熔断、止损。
    
    这是 AgentKernel 资源分配的核心升级：
    不是静态分配，而是运行时根据实际消耗动态调整的反馈回路。
    """
    
    # 五层控制器
    credit: BurstCredit = field(default_factory=BurstCredit)
    predictor: ConsumptionPredictor = field(default_factory=ConsumptionPredictor)
    pid: PIDController = field(default_factory=PIDController)
    breaker: CircuitBreaker = field(default_factory=CircuitBreaker)
    stop_loss: TrailingStopLoss = field(default_factory=TrailingStopLoss)
    
    # 状态
    total_allocated: float = 0.0      # 总分配量
    total_consumed: float = 0.0       # 总消耗量
    step_count: int = 0               # 执行步数
    
    def allocate(
        self,
        requested: Dict[str, float],
        context: ExecutionContext,
        task_progress: float = 0.0,
    ) -> Tuple[Dict[str, float], Dict[str, Any]]:
        """智能分配：综合五层反馈回路决定实际分配量。
        
        Args:
            requested: 请求的分配量
            context: 当前执行上下文
            task_progress: 当前任务进展（0-1）
            
        Returns:
            (实际分配量, 决策日志)
        """
        logs = []
        
        # Layer 1: 预测（基于历史）
        predicted = self.predictor.predict()
        
        # Layer 2: 熔断检查（速率异常？）
        if self.step_count > 0:
            current_rate = self.total_consumed / max(self.step_count, 1)
            if not self.breaker.check(current_rate):
                # 熔断触发！只允许最小分配
                logs.append({"layer": "circuit_breaker", "action": "throttled", "reason": "rate_too_high"})
                # 降级为最小分配（10% 的请求）
                allocated = {k: v * 0.1 for k, v in requested.items()}
                return allocated, {"logs": logs, "throttled": True}
        
        # Layer 3: 追踪止损（进展相关）
        can_continue, available = self.stop_loss.update(
            self.total_consumed, task_progress
        )
        if not can_continue:
            logs.append({"layer": "stop_loss", "action": "stopped", "reason": "trailing_stop_triggered"})
            return {k: 0 for k in requested}, {"logs": logs, "stopped": True}
        
        # Layer 4: PID 调节（根据历史误差调整）
        if self.step_count > 0:
            avg_rate = self.total_consumed / self.step_count
            adjustment = self.pid.compute(avg_rate)
            # 应用调整
            adjusted = {k: v * adjustment for k, v in requested.items()}
            logs.append({"layer": "pid", "adjustment": adjustment, "target": self.pid.target_rate})
        else:
            adjusted = requested.copy()
        
        # Layer 5: 信用借用（突发时可用）
        # 如果调整后的分配超过可用预算，尝试借用信用
        final_allocated = {}
        for key, value in adjusted.items():
            budget_key = key if key != "seconds" else "time"  # 简化映射
            # 检查是否可以从信用借用
            shortfall = max(0, value - getattr(context.budget, budget_key, 0))
            if shortfall > 0 and self.credit.can_borrow(shortfall):
                borrowed = self.credit.borrow(shortfall)
                final_allocated[key] = getattr(context.budget, budget_key, 0) + borrowed
                logs.append({"layer": "credit", "action": "borrowed", "amount": borrowed, "key": key})
            else:
                final_allocated[key] = min(value, getattr(context.budget, budget_key, 0))
        
        self.total_allocated += sum(final_allocated.values())
        self.step_count += 1
        
        return final_allocated, {"logs": logs, "predicted": predicted, "available": available}
    
    def record_consumption(
        self,
        allocated: Dict[str, float],
        actual: Dict[str, float],
    ) -> Dict[str, Any]:
        """记录实际消耗，更新所有反馈层。
        
        Returns:
            状态报告
        """
        self.total_consumed += sum(actual.values())
        
        # 更新预测器
        total_actual = sum(actual.values())
        status, ratio = self.predictor.record_actual(total_actual)
        
        # 如果异常，更新熔断器
        if status == "burst":
            self.breaker.record_failure()
        else:
            self.breaker.record_success()
        
        # 更新信用池（未使用的配额转化为信用）
        credit_added = self.credit.add_unused(sum(allocated.values()), total_actual)
        
        # 更新熔断器历史
        if self.step_count > 0:
            rate = total_actual / 1.0  # 简化：每步视为1单位时间
            self.breaker.rate_history.append(rate)
        
        return {
            "status": status,
            "ratio": ratio,
            "credit_added": credit_added,
            "credit_remaining": self.credit.accumulated,
            "anomaly": status == "burst",
        }
    
    def get_dashboard(self) -> Dict[str, Any]:
        """获取控制器状态仪表盘。"""
        return {
            "credit": {
                "accumulated": self.credit.accumulated,
                "max": self.credit.max_limit,
                "usage_percent": self.credit.accumulated / self.credit.max_limit * 100,
            },
            "predictor": {
                "next_prediction": self.predictor.predict() if self.predictor.predicted_history else 0,
                "history_size": len(self.predictor.actual_history),
            },
            "pid": {
                "last_adjustment": self.pid.last_error,
                "target_rate": self.pid.target_rate,
            },
            "breaker": {
                "state": self.breaker.state,
                "rate_history_size": len(self.breaker.rate_history),
            },
            "stop_loss": self.stop_loss.get_status(),
            "totals": {
                "allocated": self.total_allocated,
                "consumed": self.total_consumed,
                "efficiency": self.total_consumed / max(self.total_allocated, 1),
                "steps": self.step_count,
            },
        }


# ═══════════════════════════════════════════════════════════════
# 7. Exploration Budget — 探索预算分离
# ═══════════════════════════════════════════════════════════════

@dataclass
class ExplorationBudget:
    """探索预算与执行预算分离。
    
    核心思想：
    - 探索预算：用于不确定的搜索/调研，有严格上限
    - 执行预算：用于确定的工具调用/代码执行，精确预估
    
    当探索预算用完时，自动降级为确定性执行模式。
    
    Agent 场景：
    - 图片调研（探索性）→ 使用探索预算，有上限
    - 代码生成（执行性）→ 使用执行预算，精确预估
    """
    exploration_quota: float = 0.2    # 探索预算占总预算的 20%
    execution_quota: float = 0.8      # 执行预算占 80%
    
    exploration_used: float = 0.0
    execution_used: float = 0.0
    
    def request_exploration(self, amount: float) -> Tuple[bool, float]:
        """请求探索预算。
        
        Returns:
            (是否允许, 实际分配量)
        """
        remaining = self.exploration_quota - self.exploration_used
        allocated = min(amount, remaining)
        self.exploration_used += allocated
        return (allocated > 0, allocated)
    
    def request_execution(self, amount: float) -> Tuple[bool, float]:
        """请求执行预算。"""
        remaining = self.execution_quota - self.execution_used
        allocated = min(amount, remaining)
        self.execution_used += allocated
        return (allocated > 0, allocated)
    
    def should_degrade(self) -> bool:
        """检查是否需要降级（探索预算耗尽）。"""
        return self.exploration_used >= self.exploration_quota * 0.9
    
    def get_status(self) -> Dict[str, Any]:
        return {
            "exploration": {
                "quota": self.exploration_quota,
                "used": self.exploration_used,
                "remaining": self.exploration_quota - self.exploration_used,
                "percent": self.exploration_used / self.exploration_quota * 100,
            },
            "execution": {
                "quota": self.execution_quota,
                "used": self.execution_used,
                "remaining": self.execution_quota - self.execution_used,
                "percent": self.execution_used / self.execution_quota * 100,
            },
        }
