"""AgentKernel — Budget-Aware Call Graph (BACG) for AI Agents

主要API:
  - BACGRuntime: 统一资源控制运行时
  - CallGraph: 调用图容器
  - BudgetToken: 多维资源令牌
  - BACGAgent: 拓扑感知 Agent
  - create_bacg: 便捷工厂函数
"""
__version__ = "0.1.0"

from .topology.token import BudgetToken
from .topology.graph import CallGraph
from .topology.constraints import (
    IdentityLayer, ResourceLayer, SecurityLayer,
    QualityLayer, ConcurrencyLayer, TopologyLayer,
    ConstraintLayer,
)
from .topology.agent import BACGAgent
from .topology.integration import BACGRuntime, UsageExtractor, create_bacg

__all__ = [
    "BudgetToken",
    "CallGraph",
    "IdentityLayer",
    "ResourceLayer",
    "SecurityLayer",
    "QualityLayer",
    "ConcurrencyLayer",
    "TopologyLayer",
    "ConstraintLayer",
    "BACGAgent",
    "BACGRuntime",
    "UsageExtractor",
    "create_bacg",
]
