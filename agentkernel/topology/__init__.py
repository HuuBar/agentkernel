"""BACG Topology Module — Budget-Aware Call Graph

核心组件：
  - BudgetToken: 多维资源令牌（CHERI 单调性）
  - CallGraph: 属性有向树 + 6 层约束面
  - BACGAgent: 拓扑感知 Agent
  - BACGRuntime: 统一资源控制运行时
"""
from .token import BudgetToken, InsufficientBudget
from .constraints import (
    ConstraintLayer,
    ConstraintNode,
    ConstraintEdge,
    IdentityLayer,
    ResourceLayer,
    SecurityLayer,
    QualityLayer,
    ConcurrencyLayer,
    TopologyLayer,
)
from .graph import CallGraph
from .branching import BranchingModel
from .budget import TopologyBudget, CallNode, CallGraphV1
from .agent import BACGAgent, BACGAgentV2
from .integration import (
    BACGRuntime,
    UsageExtractor,
    BudgetExceeded,
    OperationContext,
    FrameworkAdapter,
    LangGraphAdapter,
    AutoGenAdapter,
    OpenAIAdapter,
    BACGGateway,
    create_bacg,
)

__all__ = [
    "BudgetToken",
    "InsufficientBudget",
    "ConstraintLayer",
    "ConstraintNode",
    "ConstraintEdge",
    "IdentityLayer",
    "ResourceLayer",
    "SecurityLayer",
    "QualityLayer",
    "ConcurrencyLayer",
    "TopologyLayer",
    "CallGraph",
    "BranchingModel",
    "TopologyBudget",
    "CallNode",
    "CallGraphV1",
    "BACGAgent",
    "BACGAgentV2",
    "BACGRuntime",
    "UsageExtractor",
    "BudgetExceeded",
    "OperationContext",
    "FrameworkAdapter",
    "LangGraphAdapter",
    "AutoGenAdapter",
    "OpenAIAdapter",
    "BACGGateway",
    "create_bacg",
]
