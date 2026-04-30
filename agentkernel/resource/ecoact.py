"""EcoAct — Adaptive Gearing for Resource-Aware Agents

核心范式：Gauge → Shift → Drive
  Gauge:  感知燃料状态（Fuel Level + Progress Rate）
  Shift:  选择档位（Sprint / Cruise / Eco / Coast）
  Drive:  在档位约束下执行（自动限制思考深度、工具数量、搜索范围）

与 ReAct 对位：
  ReAct: Think → Act → Observe
  EcoAct: Gauge → Shift → Drive
"""
from __future__ import annotations

import asyncio
import enum
import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple


# ═══════════════════════════════════════════════════════════════
# Fuel — 燃料系统（Agent 的"生理指标"）
# ═══════════════════════════════════════════════════════════════

@dataclass
class ComputationFuel:
    """计算燃料：token / cost / api_calls"""
    remaining_tokens: int = 10000
    total_tokens: int = 10000
    remaining_usd: float = 10.0
    total_usd: float = 10.0
    remaining_api_calls: int = 50
    total_api_calls: int = 50
    
    @property
    def token_ratio(self) -> float:
        return self.remaining_tokens / max(self.total_tokens, 1)
    
    @property
    def usd_ratio(self) -> float:
        return self.remaining_usd / max(self.total_usd, 0.01)
    
    @property
    def api_ratio(self) -> float:
        return self.remaining_api_calls / max(self.total_api_calls, 1)
    
    @property
    def min_ratio(self) -> float:
        """返回最紧缺的燃料比例。"""
        return min(self.token_ratio, self.usd_ratio, self.api_ratio)
    
    def consume(self, tokens: int = 0, usd: float = 0, api_calls: int = 0) -> ComputationFuel:
        """消耗燃料，返回新状态（不可变）。"""
        return ComputationFuel(
            remaining_tokens=max(0, self.remaining_tokens - tokens),
            total_tokens=self.total_tokens,
            remaining_usd=max(0, self.remaining_usd - usd),
            total_usd=self.total_usd,
            remaining_api_calls=max(0, self.remaining_api_calls - api_calls),
            total_api_calls=self.total_api_calls,
        )


@dataclass
class TimeFuel:
    """时间燃料：deadline / timeout"""
    remaining_seconds: float = 60.0
    total_seconds: float = 60.0
    
    @property
    def ratio(self) -> float:
        return self.remaining_seconds / max(self.total_seconds, 0.01)


@dataclass
class ProgressRate:
    """任务进展：已完成 / 总步骤"""
    completed_steps: int = 0
    total_steps: int = 5
    
    @property
    def completion_ratio(self) -> float:
        return self.completed_steps / max(self.total_steps, 1)


@dataclass
class FuelState:
    """Agent 的完整燃料状态——Agent 的"生理指标"。"""
    computation: ComputationFuel = field(default_factory=ComputationFuel)
    time: TimeFuel = field(default_factory=TimeFuel)
    progress: ProgressRate = field(default_factory=ProgressRate)
    
    @property
    def overall_fuel_ratio(self) -> float:
        """综合燃料比例（取最紧缺的）。"""
        return min(self.computation.min_ratio, self.time.ratio)
    
    @property
    def efficiency(self) -> float:
        """效率 = 进展 / 消耗比例（越高越好）。"""
        consumed = 1 - self.overall_fuel_ratio
        return self.progress.completion_ratio / max(consumed, 0.01)
    
    def __str__(self) -> str:
        return (f"Fuel(tokens={self.computation.remaining_tokens}/{self.computation.total_tokens}, "
                f"usd=${self.computation.remaining_usd:.2f}/${self.computation.total_usd:.2f}, "
                f"progress={self.progress.completion_ratio:.0%}")


# ═══════════════════════════════════════════════════════════════
# Gear — 档位系统（Agent 的"行为模式"）
# ═══════════════════════════════════════════════════════════════

class Gear(enum.Enum):
    """四个档位，对应四种行为强度。"""
    SPRINT = "sprint"    # 冲刺：高消耗，高产出，用于突破
    CRUISE = "cruise"    # 巡航：正常消耗，正常产出，用于常规
    ECO = "eco"          # 节能：低消耗，低保真，用于紧张
    COAST = "coast"      # 滑行：极低消耗，只读，用于耗尽


@dataclass
class GearProfile:
    """档位行为配置——定义每个档位的具体限制。"""
    max_cot_steps: int           # 最大 Chain-of-Thought 步数
    max_tools_per_step: int      # 每步最大工具调用数
    search_scope: str            # 搜索范围：global/local/directional/none
    output_quality: str          # 输出质量：high/medium/low/minimal
    reasoning_depth: str         # 推理深度：deep/normal/shallow/none
    exploration_budget_ratio: float  # 允许用于探索的预算比例
    burn_rate_multiplier: float  # 燃料消耗倍率（相对于 Cruise）
    
    def __str__(self) -> str:
        return (f"CoT≤{self.max_cot_steps}, tools≤{self.max_tools_per_step}, "
                f"search={self.search_scope}, quality={self.output_quality}, "
                f"burn={self.burn_rate_multiplier}x")


# 档位配置表
GEAR_PROFILES: Dict[Gear, GearProfile] = {
    Gear.SPRINT: GearProfile(
        max_cot_steps=10,
        max_tools_per_step=5,
        search_scope="global",
        output_quality="high",
        reasoning_depth="deep",
        exploration_budget_ratio=0.4,
        burn_rate_multiplier=2.5,
    ),
    Gear.CRUISE: GearProfile(
        max_cot_steps=5,
        max_tools_per_step=2,
        search_scope="local",
        output_quality="medium",
        reasoning_depth="normal",
        exploration_budget_ratio=0.2,
        burn_rate_multiplier=1.0,
    ),
    Gear.ECO: GearProfile(
        max_cot_steps=2,
        max_tools_per_step=1,
        search_scope="directional",
        output_quality="low",
        reasoning_depth="shallow",
        exploration_budget_ratio=0.05,
        burn_rate_multiplier=0.4,
    ),
    Gear.COAST: GearProfile(
        max_cot_steps=0,
        max_tools_per_step=0,
        search_scope="none",
        output_quality="minimal",
        reasoning_depth="none",
        exploration_budget_ratio=0.0,
        burn_rate_multiplier=0.1,
    ),
}


# ═══════════════════════════════════════════════════════════════
# GearBox — 变速箱（换挡逻辑）
# ═══════════════════════════════════════════════════════════════

@dataclass
class GearBox:
    """变速箱：根据燃料状态选择档位。
    
    换挡逻辑：
      燃料充足 + 进展慢 → Sprint（需要突破）
      燃料正常 + 效率OK → Cruise（常规工作）
      燃料紧张 → Eco（节能收敛）
      燃料耗尽 → Coast（滑行输出）
    """
    
    def shift(self, fuel: FuelState) -> Gear:
        fuel_ratio = fuel.overall_fuel_ratio
        progress = fuel.progress.completion_ratio
        efficiency = fuel.efficiency
        
        # Sprint: 燃料 > 70% 但进展 < 30% → 需要冲刺突破
        if fuel_ratio > 0.7 and progress < 0.3:
            return Gear.SPRINT
        
        # Sprint: 燃料 > 50% 且效率极低（探索期）
        if fuel_ratio > 0.5 and efficiency < 0.3:
            return Gear.SPRINT
        
        # Cruise: 燃料 > 40% 且效率正常
        if fuel_ratio > 0.4 and efficiency >= 0.3:
            return Gear.CRUISE
        
        # Eco: 燃料 > 20% 或燃料 > 40% 但效率低
        if fuel_ratio > 0.2:
            return Gear.ECO
        
        # Coast: 燃料耗尽
        return Gear.COAST
    
    def explain_shift(self, fuel: FuelState, gear: Gear) -> str:
        """解释为什么切换到该档位。"""
        fuel_ratio = fuel.overall_fuel_ratio
        progress = fuel.progress.completion_ratio
        efficiency = fuel.efficiency
        
        explanations = {
            Gear.SPRINT: f"燃料充足({fuel_ratio:.0%})但进展慢({progress:.0%})，需要冲刺突破",
            Gear.CRUISE: f"燃料正常({fuel_ratio:.0%})且效率OK({efficiency:.1f})，巡航执行",
            Gear.ECO: f"燃料紧张({fuel_ratio:.0%})，进入节能模式收敛",
            Gear.COAST: f"燃料耗尽({fuel_ratio:.0%})，滑行输出已有结果",
        }
        return explanations.get(gear, "未知原因")


# ═══════════════════════════════════════════════════════════════
# EcoAct Agent — 完整执行循环
# ═══════════════════════════════════════════════════════════════

@dataclass
class StepResult:
    """一步执行的结果。"""
    output: Any
    tokens_consumed: int
    usd_consumed: float
    api_calls: int
    gear_used: Gear


class EcoActAgent:
    """EcoAct 范式的 Agent。
    
    执行循环：Gauge → Shift → Drive → Burn → Gauge → ...
    """
    
    def __init__(self, name: str, fuel: FuelState):
        self.name = name
        self.fuel = fuel
        self.gearbox = GearBox()
        self.history: List[Tuple[Gear, int]] = []  # (档位, 消耗tokens)
    
    async def run(self, task_description: str, estimated_steps: int = 5) -> List[StepResult]:
        """执行任务，在 EcoAct 循环中自动换挡。"""
        self.fuel.progress.total_steps = estimated_steps
        results = []
        
        step = 0
        while self.fuel.overall_fuel_ratio > 0.05 and step < estimated_steps * 2:
            step += 1
            
            # ─── EcoAct 三原语 ───
            
            # 1. GAUGE: 感知燃料状态
            print(f"\n  [Step {step}] Fuel: {self.fuel}")
            
            # 2. SHIFT: 选择档位
            gear = self.gearbox.shift(self.fuel)
            profile = GEAR_PROFILES[gear]
            explanation = self.gearbox.explain_shift(self.fuel, gear)
            print(f"  → Gear: {gear.value.upper()} | {explanation}")
            print(f"  → Profile: {profile}")
            
            # 3. DRIVE: 在档位下执行
            result = await self._drive_step(task_description, gear, profile)
            results.append(result)
            
            # 4. BURN: 消耗燃料
            self.fuel.computation = self.fuel.computation.consume(
                tokens=result.tokens_consumed,
                usd=result.usd_consumed,
                api_calls=result.api_calls,
            )
            self.fuel.progress.completed_steps += 1
            
            self.history.append((gear, result.tokens_consumed))
            
            print(f"  → Burned: {result.tokens_consumed} tokens | Remaining: {self.fuel.computation.remaining_tokens}")
            
            # 检查是否需要终止
            if gear == Gear.COAST and result.tokens_consumed == 0:
                print(f"  → Coast mode with no progress, stopping.")
                break
        
        return results
    
    async def _drive_step(self, task: str, gear: Gear, profile: GearProfile) -> StepResult:
        """模拟在档位下执行一步。
        
        真实实现中，这里会构造档位感知的 prompt，调用 LLM。
        """
        # 根据档位确定消耗
        base_tokens = 500  # 基础消耗
        actual_tokens = int(base_tokens * profile.burn_rate_multiplier)
        
        # 根据档位确定工具调用数
        actual_tools = min(profile.max_tools_per_step, max(1, int(profile.burn_rate_multiplier)))
        
        # 模拟输出
        if gear == Gear.SPRINT:
            output = {"action": "deep_exploration", "tools_used": actual_tools, "search": "global"}
        elif gear == Gear.CRUISE:
            output = {"action": "standard_execution", "tools_used": actual_tools, "search": "local"}
        elif gear == Gear.ECO:
            output = {"action": "shallow_reasoning", "tools_used": actual_tools, "search": "directional"}
        else:
            output = {"action": "read_only_output", "tools_used": 0, "search": "none"}
        
        # 模拟执行时间
        await asyncio.sleep(0.01)
        
        return StepResult(
            output=output,
            tokens_consumed=actual_tokens,
            usd_consumed=actual_tokens * 0.00003,
            api_calls=actual_tools,
            gear_used=gear,
        )
    
    def get_summary(self) -> Dict[str, Any]:
        """执行摘要。"""
        if not self.history:
            return {}
        
        gear_counts = {}
        total_tokens = 0
        for gear, tokens in self.history:
            gear_counts[gear.value] = gear_counts.get(gear.value, 0) + 1
            total_tokens += tokens
        
        return {
            "agent": self.name,
            "total_steps": len(self.history),
            "gear_distribution": gear_counts,
            "total_tokens": total_tokens,
            "remaining_tokens": self.fuel.computation.remaining_tokens,
            "remaining_ratio": self.fuel.overall_fuel_ratio,
            "final_progress": self.fuel.progress.completion_ratio,
            "efficiency": self.fuel.efficiency,
        }
