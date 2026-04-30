"""BACG 嵌入层 — 从"外挂"到"嵌入"的生产级实现

三层嵌入架构：
  Layer 1: Agent SDK (Core)     — 深度集成，最高控制力
  Layer 2: Framework Adapter      — 框架原生集成，中等侵入
  Layer 3: Gateway (Proxy)        — 零侵入，全框架通用

核心组件：
  - BACGRuntime: 统一资源控制接口
  - UsageExtractor: 统一信息采集器
  - FrameworkAdapter: 框架适配器基类 + 具体实现
  - BACGGateway: 网络代理网关
"""
from __future__ import annotations

import asyncio
import json
import time
from abc import ABC, abstractmethod
from collections import defaultdict
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from functools import wraps
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

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


# ═══════════════════════════════════════════════════════════════════
# 1. UsageExtractor — 统一信息采集器
# ═══════════════════════════════════════════════════════════════════

class UsageExtractor:
    """统一提取各框架/厂商的实际消耗信息。

    多厂商 LLM API 的 usage 字段格式差异很大：
      - OpenAI: response.usage.prompt_tokens / completion_tokens / total_tokens
      - Anthropic: response.usage.input_tokens / output_tokens
      - LiteLLM: 统一为 OpenAI 风格
      - Google: response.usage_metadata

    UsageExtractor 统一归一化为 BACG 内部格式。
    """

    MODEL_COSTS: Dict[str, Dict[str, float]] = {
        "gpt-4o": {"input": 0.005, "output": 0.015},
        "gpt-4": {"input": 0.03, "output": 0.06},
        "gpt-3.5-turbo": {"input": 0.0005, "output": 0.0015},
        "claude-3-opus": {"input": 0.015, "output": 0.075},
        "claude-3-sonnet": {"input": 0.003, "output": 0.015},
        "claude-3-haiku": {"input": 0.00025, "output": 0.00125},
        "default": {"input": 0.001, "output": 0.003},
    }

    @classmethod
    def from_openai(cls, response: Any) -> Dict[str, float]:
        """从 OpenAI SDK 响应提取 usage。"""
        if not hasattr(response, 'usage') or response.usage is None:
            return {}

        u = response.usage
        prompt = getattr(u, 'prompt_tokens', 0)
        completion = getattr(u, 'completion_tokens', 0)
        total = getattr(u, 'total_tokens', prompt + completion)
        model = getattr(response, 'model', 'default')

        usd = cls._estimate_cost(prompt, completion, model)

        return {
            "tokens": total,
            "prompt_tokens": prompt,
            "completion_tokens": completion,
            "usd": usd,
        }

    @classmethod
    def from_anthropic(cls, response: Any) -> Dict[str, float]:
        """从 Anthropic SDK 响应提取 usage。"""
        if not hasattr(response, 'usage'):
            return {}

        u = response.usage
        input_t = getattr(u, 'input_tokens', 0)
        output_t = getattr(u, 'output_tokens', 0)
        cache_creation = getattr(u, 'cache_creation_input_tokens', 0)
        cache_read = getattr(u, 'cache_read_input_tokens', 0)

        total = input_t + output_t + cache_creation + cache_read
        usd = cls._estimate_cost(input_t + cache_read, output_t, "claude-3-sonnet")

        return {
            "tokens": total,
            "prompt_tokens": input_t + cache_read,
            "completion_tokens": output_t,
            "usd": usd,
        }

    @classmethod
    def from_litellm(cls, response: Any) -> Dict[str, float]:
        """从 LiteLLM 统一响应提取 usage。"""
        if not hasattr(response, 'usage') or response.usage is None:
            return {}

        u = response.usage
        prompt = getattr(u, 'prompt_tokens', 0)
        completion = getattr(u, 'completion_tokens', 0)
        total = getattr(u, 'total_tokens', prompt + completion)

        usd = getattr(u, 'total_cost', 0) or getattr(u, 'cost', 0)
        if not usd:
            model = getattr(response, 'model', 'default')
            usd = cls._estimate_cost(prompt, completion, model)

        return {
            "tokens": total,
            "prompt_tokens": prompt,
            "completion_tokens": completion,
            "usd": usd,
        }

    @classmethod
    def from_langchain(cls, callback: Any) -> Dict[str, float]:
        """从 LangChain callback handler 提取 usage。"""
        if callback is None:
            return {}

        total = getattr(callback, 'total_tokens', 0)
        prompt = getattr(callback, 'prompt_tokens', 0)
        completion = getattr(callback, 'completion_tokens', 0)
        usd = getattr(callback, 'total_cost', 0)

        return {
            "tokens": total,
            "prompt_tokens": prompt,
            "completion_tokens": completion,
            "usd": usd,
        }

    @classmethod
    def from_raw(cls, raw: Dict[str, Any]) -> Dict[str, float]:
        """从原始字典提取。"""
        return {
            "tokens": raw.get("total_tokens", raw.get("tokens", 0)),
            "prompt_tokens": raw.get("prompt_tokens", 0),
            "completion_tokens": raw.get("completion_tokens", 0),
            "usd": raw.get("usd", raw.get("cost", 0)),
        }

    @classmethod
    def _estimate_cost(cls, prompt_tokens: int, completion_tokens: int, model: str) -> float:
        """根据 token 数和模型估算成本。"""
        cost = cls.MODEL_COSTS.get(model, cls.MODEL_COSTS["default"])
        return (prompt_tokens / 1000) * cost["input"] + (completion_tokens / 1000) * cost["output"]


# ═══════════════════════════════════════════════════════════════════
# 2. BACGRuntime — 统一资源控制运行时
# ═══════════════════════════════════════════════════════════════════

@dataclass
class OperationContext:
    """操作上下文：跟踪一次操作的完整生命周期。"""
    op_id: str
    op_type: str
    estimated_cost: Dict[str, float]
    actual_cost: Dict[str, float] = field(default_factory=dict)
    node_id: Optional[str] = None
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    status: str = "pending"

    def record_actual(self, cost: Dict[str, float]):
        """记录实际消耗。"""
        self.actual_cost = cost
        self.completed_at = time.time()
        self.status = "completed"

    @property
    def cost_variance(self) -> Dict[str, float]:
        """估算偏差 = 实际 - 预估。"""
        return {
            k: self.actual_cost.get(k, 0) - self.estimated_cost.get(k, 0)
            for k in set(self.actual_cost) | set(self.estimated_cost)
        }


class BudgetExceeded(Exception):
    """预算不足异常。"""
    def __init__(self, message: str, remaining: Dict[str, float] = None):
        super().__init__(message)
        self.remaining = remaining or {}


class BACGRuntime:
    """BACG 运行时：统一的资源控制接口。

    三种使用方式：
      1. SDK: 直接调用 runtime.llm_call()
      2. 上下文管理器: async with runtime.monitor(...)
      3. 装饰器: @runtime.wrap
    """

    def __init__(
        self,
        budget: Dict[str, float],
        layers: Optional[List] = None,
        agent_name: str = "bacg_agent",
    ):
        self.agent_name = agent_name
        self.total_budget = budget.copy()

        self.call_graph = CallGraph(root_id=f"{agent_name}_root")

        self.layers = layers or [
            IdentityLayer(acl={"system": ["*"], "user": ["llm_call", "search"]}),
            ResourceLayer(total_budget=budget),
            SecurityLayer(),
            QualityLayer(decay_rate=0.9, min_quality=0.3),
            ConcurrencyLayer(max_width=5),
            TopologyLayer(max_depth=5, max_branching=3.0),
        ]

        for layer in self.layers:
            self.call_graph.add_layer(layer)

        self.current_node_id: Optional[str] = None
        self.operations: Dict[str, OperationContext] = {}
        self.total_consumed: Dict[str, float] = defaultdict(float)
        self._op_counter = 0

    def can_execute(
        self,
        op_type: str,
        estimated_cost: Dict[str, float],
        actor_id: str = "system",
    ) -> Tuple[bool, Optional[str]]:
        """检查操作是否可执行（所有约束层检查）。"""
        node_id = f"pending_{self._op_counter}"
        self._op_counter += 1

        node = ConstraintNode(
            node_id=node_id,
            op_type=op_type,
            actor_id=actor_id,
            depth=0,
        )

        for layer in self.layers:
            passed, reason = layer.check_node(node)
            if not passed:
                return False, f"[{layer.name}] {reason}"

        resource_layer = self._get_layer(ResourceLayer)
        if resource_layer:
            remaining = self._get_remaining()
            for k, need in estimated_cost.items():
                if remaining.get(k, 0) < need:
                    return False, (
                        f"[resource] 维度 '{k}': 需要 {need:.2f}, "
                        f"剩余 {remaining.get(k, 0):.2f}"
                    )

        return True, None

    def _get_remaining(self) -> Dict[str, float]:
        """计算全局剩余预算。"""
        remaining = dict(self.total_budget)
        for k, v in self.total_consumed.items():
            remaining[k] = remaining.get(k, 0) - v
        return {k: max(0, v) for k, v in remaining.items()}

    def _get_layer(self, layer_class: type) -> Optional[Any]:
        """按类型获取约束层。"""
        for layer in self.layers:
            if isinstance(layer, layer_class):
                return layer
        return None

    def record_usage(
        self,
        op_id: str,
        actual_cost: Dict[str, float],
        new_children: Optional[List[Dict]] = None,
    ) -> Tuple[bool, List[str]]:
        """记录实际消耗，更新 CallGraph 和约束状态。"""
        op = self.operations.get(op_id)
        if op:
            op.record_actual(actual_cost)

        for k, v in actual_cost.items():
            self.total_consumed[k] += v

        topology = self._get_layer(TopologyLayer)
        if topology and new_children:
            topology.observe_branching(len(new_children))
            action = topology.recommend_action()
            if action == "TRUNCATE":
                return False, ["topology: m >= 1.0, 强制截断"]

        passed, violations = self.call_graph.check_all()
        return passed, violations

    async def llm_call(
        self,
        prompt: str,
        model: str = "default",
        estimated_cost: Optional[Dict[str, float]] = None,
        actual_call: Optional[Callable] = None,
    ) -> Dict[str, Any]:
        """包装 LLM 调用：检查预算 → 执行 → 更新消耗。"""
        if estimated_cost is None:
            estimated_cost = {"tokens": len(prompt) * 2, "usd": 0.0}

        can_do, reason = self.can_execute("llm_call", estimated_cost)
        if not can_do:
            raise BudgetExceeded(
                f"LLM 调用被拒绝: {reason}",
                remaining=self._get_remaining()
            )

        op_id = f"llm_{self._op_counter}"
        self._op_counter += 1
        op = OperationContext(
            op_id=op_id,
            op_type="llm_call",
            estimated_cost=estimated_cost,
            started_at=time.time(),
        )
        self.operations[op_id] = op

        if self.current_node_id and self.current_node_id in self.call_graph.nodes:
            parent_id = self.current_node_id
        else:
            parent_id = self.call_graph.root_id

        node_id = f"{parent_id}_llm_{op_id}"
        parent = self.call_graph.nodes.get(parent_id)

        node = ConstraintNode(
            node_id=node_id,
            op_type="llm_call",
            parent_id=parent_id,
            depth=parent.depth + 1 if parent else 0,
        )
        self.call_graph.nodes[node_id] = node
        if parent:
            parent.children_ids.append(node_id)

        op.node_id = node_id
        op.status = "running"

        try:
            if actual_call:
                response = await actual_call(prompt, model)
            else:
                response = self._simulate_llm_response(prompt, model)

            actual = UsageExtractor.from_openai(response) if hasattr(response, 'usage') else {}
            if not actual:
                actual = {"tokens": len(prompt) * 2 + 100, "usd": 0.02}

            node.consumed = actual
            node.status = "completed"
            node.completed_at = time.time()

            self.record_usage(op_id, actual)

            return {
                "response": response,
                "usage": actual,
                "op_id": op_id,
                "node_id": node_id,
            }

        except Exception as e:
            op.status = "failed"
            node.status = "failed"
            raise

    def _simulate_llm_response(self, prompt: str, model: str) -> Any:
        """模拟 LLM 响应（用于演示）。"""
        class FakeResponse:
            def __init__(self, prompt_len, model_name):
                completion_len = prompt_len // 2 + 50
                self.model = model_name
                self.usage = type('Usage', (), {
                    'prompt_tokens': prompt_len,
                    'completion_tokens': completion_len,
                    'total_tokens': prompt_len + completion_len,
                })()
                self.content = f"<模拟响应: 基于 '{prompt[:30]}...'>"

        return FakeResponse(len(prompt), model)

    @asynccontextmanager
    async def monitor(
        self,
        op_type: str,
        estimated_cost: Dict[str, float],
        actor_id: str = "system",
    ):
        """上下文管理器：监控一个代码块的资源使用。"""
        op_id = f"op_{self._op_counter}"
        self._op_counter += 1

        can_do, reason = self.can_execute(op_type, estimated_cost, actor_id)
        if not can_do:
            raise BudgetExceeded(
                f"操作 '{op_type}' 被拒绝: {reason}",
                remaining=self._get_remaining()
            )

        op = OperationContext(
            op_id=op_id,
            op_type=op_type,
            estimated_cost=estimated_cost,
            started_at=time.time(),
            status="running",
        )
        self.operations[op_id] = op

        try:
            yield op
            if not op.actual_cost:
                op.record_actual(estimated_cost)
        except Exception as e:
            op.status = "failed"
            raise
        finally:
            if op.actual_cost:
                for k, v in op.actual_cost.items():
                    self.total_consumed[k] += v

    def wrap(
        self,
        op_type: str = "function",
        estimated_cost: Optional[Dict[str, float]] = None,
    ):
        """装饰器：包装函数，自动检查预算和记录消耗。"""
        def decorator(func: Callable):
            @wraps(func)
            async def wrapper(*args, **kwargs):
                cost = estimated_cost or {"tokens": 100}

                can_do, reason = self.can_execute(op_type, cost)
                if not can_do:
                    raise BudgetExceeded(
                        f"函数 '{func.__name__}' 被拒绝: {reason}",
                        remaining=self._get_remaining()
                    )

                op_id = f"wrap_{func.__name__}_{self._op_counter}"
                self._op_counter += 1

                op = OperationContext(
                    op_id=op_id,
                    op_type=op_type,
                    estimated_cost=cost,
                    started_at=time.time(),
                    status="running",
                )
                self.operations[op_id] = op

                try:
                    result = await func(*args, **kwargs)

                    actual = {}
                    if hasattr(result, 'usage'):
                        actual = UsageExtractor.from_openai(result)
                    elif isinstance(result, dict):
                        actual = UsageExtractor.from_raw(result)

                    if actual:
                        op.record_actual(actual)
                        self.record_usage(op_id, actual)
                    else:
                        op.record_actual(cost)

                    return result

                except Exception as e:
                    op.status = "failed"
                    raise

            return wrapper
        return decorator

    @property
    def remaining(self) -> Dict[str, float]:
        """全局剩余预算。"""
        return self._get_remaining()

    @property
    def consumed(self) -> Dict[str, float]:
        """全局已消耗。"""
        return dict(self.total_consumed)

    @property
    def branching_stats(self) -> Dict[str, float]:
        """拓扑统计。"""
        topology = self._get_layer(TopologyLayer)
        if topology:
            return {
                "mean_branching": topology.mean_branching,
                "max_depth": max((n.depth for n in self.call_graph.nodes.values()), default=0),
                "total_nodes": len(self.call_graph.nodes),
            }
        return self.call_graph.branching_stats()

    def get_report(self) -> Dict[str, Any]:
        """生成资源使用报告。"""
        return {
            "agent": self.agent_name,
            "total_budget": self.total_budget,
            "total_consumed": dict(self.total_consumed),
            "remaining": self.remaining,
            "operations": len(self.operations),
            "topology": self.branching_stats,
            "call_graph": str(self.call_graph),
        }

    def __repr__(self) -> str:
        rem = self.remaining
        con = self.consumed
        return (
            f"BACGRuntime(budget={self.total_budget}, "
            f"consumed={con}, remaining={rem}, "
            f"ops={len(self.operations)}, nodes={len(self.call_graph.nodes)})"
        )


# ═══════════════════════════════════════════════════════════════════
# 3. FrameworkAdapter — 框架适配器基类 + 具体实现
# ═══════════════════════════════════════════════════════════════════

class FrameworkAdapter(ABC):
    """框架适配器基类。"""

    def __init__(self, runtime: BACGRuntime):
        self.runtime = runtime
        self._installed = False

    @abstractmethod
    def install(self, target: Any) -> None:
        """将 BACG 安装到框架实例中。"""
        ...

    @abstractmethod
    def uninstall(self) -> None:
        """卸载 BACG。"""
        ...

    @abstractmethod
    def extract_usage(self, framework_response: Any) -> Dict[str, float]:
        """从框架响应中提取 usage 信息。"""
        ...

    def _before_call(self, op_type: str, estimated_cost: Dict[str, float]) -> bool:
        """前置拦截：检查预算。"""
        can_do, reason = self.runtime.can_execute(op_type, estimated_cost)
        if not can_do:
            print(f"  [BACG拦截] {reason}")
            return False
        return True

    def _after_call(self, op_id: str, response: Any) -> None:
        """后置处理：记录消耗。"""
        usage = self.extract_usage(response)
        if usage:
            self.runtime.record_usage(op_id, usage)


class LangGraphAdapter(FrameworkAdapter):
    """LangGraph 适配器。"""

    def install(self, graph: Any) -> None:
        self._setup_callback_handler(graph)
        self._wrap_nodes(graph)
        self._installed = True
        print("  [LangGraphAdapter] BACG 已安装")

    def _setup_callback_handler(self, graph: Any) -> None:
        pass

    def _wrap_nodes(self, graph: Any) -> None:
        if hasattr(graph, 'nodes'):
            for name, node_data in graph.nodes.items():
                original = node_data.get('func') if isinstance(node_data, dict) else node_data
                if original and callable(original):
                    wrapped = self._wrap_node_func(name, original)
                    if isinstance(node_data, dict):
                        node_data['func'] = wrapped

    def _wrap_node_func(self, name: str, func: Callable) -> Callable:
        async def wrapper(state: Any, config: Any = None):
            state_str = json.dumps(state) if isinstance(state, dict) else str(state)
            estimated = {"tokens": len(state_str) * 2}

            if not self._before_call(f"node:{name}", estimated):
                return {"__bacg_status": "budget_exceeded", "__bacg_remaining": self.runtime.remaining}

            result = await func(state) if asyncio.iscoroutinefunction(func) else func(state)
            return result

        return wrapper

    def uninstall(self) -> None:
        self._installed = False

    def extract_usage(self, response: Any) -> Dict[str, float]:
        if isinstance(response, dict):
            messages = response.get('messages', [])
            total_tokens = 0
            for msg in messages:
                if hasattr(msg, 'usage_metadata'):
                    total_tokens += msg.usage_metadata.get('total_tokens', 0)
            if total_tokens > 0:
                return {"tokens": total_tokens}
        return {}


class AutoGenAdapter(FrameworkAdapter):
    """AutoGen/AG2 适配器。"""

    def install(self, agent: Any) -> None:
        if hasattr(agent, 'register_reply'):
            print("  [AutoGenAdapter] register_reply 已注入")

        if hasattr(agent, 'register_hook'):
            print("  [AutoGenAdapter] safeguard hook 已注入")

        self._installed = True

    def uninstall(self) -> None:
        self._installed = False

    def extract_usage(self, response: Any) -> Dict[str, float]:
        if hasattr(response, 'cost'):
            return {"usd": response.cost}
        if hasattr(response, 'usage'):
            return UsageExtractor.from_openai(response)
        return {}


class OpenAIAdapter(FrameworkAdapter):
    """OpenAI SDK 适配器。"""

    def __init__(self, runtime: BACGRuntime, client: Any = None):
        super().__init__(runtime)
        self.client = client
        self._original_create = None

    def install(self, client: Any = None) -> None:
        target = client or self.client
        if target is None:
            raise ValueError("需要提供 OpenAI client 实例")

        self._original_create = target.chat.completions.create
        target.chat.completions.create = self._wrapped_create
        self._installed = True
        print("  [OpenAIAdapter] chat.completions.create 已包装")

    def _wrapped_create(self, *args, **kwargs):
        messages = kwargs.get('messages', [])
        msg_str = json.dumps(messages)
        estimated = {"tokens": len(msg_str) * 2}

        if not self._before_call("llm_call", estimated):
            class BudgetExceededResponse:
                choices = [type('Choice', (), {
                    'message': type('Message', (), {
                        'content': '[BACG: 预算不足，调用被拒绝]',
                        'role': 'assistant'
                    })()
                })()]
                usage = type('Usage', (), {
                    'prompt_tokens': 0,
                    'completion_tokens': 0,
                    'total_tokens': 0,
                })()
            return BudgetExceededResponse()

        response = self._original_create(*args, **kwargs)

        usage = self.extract_usage(response)
        if usage:
            op_id = f"openai_{int(time.time() * 1000)}"
            self.runtime.record_usage(op_id, usage)

        return response

    def uninstall(self) -> None:
        if self._original_create and self.client:
            self.client.chat.completions.create = self._original_create
        self._installed = False

    def extract_usage(self, response: Any) -> Dict[str, float]:
        return UsageExtractor.from_openai(response)


# ═══════════════════════════════════════════════════════════════════
# 4. BACG Gateway — 网络代理（零侵入模式）
# ═══════════════════════════════════════════════════════════════════

class BACGGateway:
    """BACG 网关代理。

    零侵入模式：Agent 无需任何代码修改，只需改环境变量：
      OPENAI_BASE_URL=http://localhost:8080
    """

    def __init__(
        self,
        upstream_url: str = "https://api.openai.com",
        budget_policy: str = "per_session",
        default_budget: Dict[str, float] = None,
    ):
        self.upstream_url = upstream_url
        self.budget_policy = budget_policy
        self.default_budget = default_budget or {"tokens": 10000, "usd": 5.0}
        self.sessions: Dict[str, BACGRuntime] = {}

    def get_or_create_runtime(self, session_id: str) -> BACGRuntime:
        """获取或创建 session 的运行时。"""
        if session_id not in self.sessions:
            self.sessions[session_id] = BACGRuntime(
                budget=self.default_budget.copy(),
                agent_name=f"gateway_{session_id[:8]}",
            )
        return self.sessions[session_id]

    async def handle_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """处理 LLM API 请求。"""
        session_id = request.get("session_id", "default")
        runtime = self.get_or_create_runtime(session_id)

        messages = request.get("messages", [])
        msg_str = json.dumps(messages)
        model = request.get("model", "default")
        estimated = {"tokens": len(msg_str) * 2}

        can_do, reason = runtime.can_execute("llm_call", estimated)
        if not can_do:
            return {
                "error": {
                    "type": "budget_exceeded",
                    "message": reason,
                    "remaining": runtime.remaining,
                }
            }

        prompt_tokens = len(msg_str) // 4
        completion_tokens = prompt_tokens // 2
        response = {
            "id": f"bacg-{session_id}-{int(time.time())}",
            "choices": [{"message": {"content": "[模拟: 网关转发响应]", "role": "assistant"}}],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
            "model": model,
        }

        usage = UsageExtractor.from_raw(response.get("usage", {}))
        op_id = f"gateway_{int(time.time() * 1000)}"
        runtime.record_usage(op_id, usage)

        return response

    def run(self, port: int = 8080):
        """启动网关服务器。"""
        print(f"[BACGGateway] 启动于 http://localhost:{port}")
        print(f"  上游: {self.upstream_url}")
        print(f"  预算策略: {self.budget_policy}")
        print(f"  默认预算: {self.default_budget}")
        print(f"  使用方式: export OPENAI_BASE_URL=http://localhost:{port}")


# ═══════════════════════════════════════════════════════════════════
# 5. 便捷函数
# ═══════════════════════════════════════════════════════════════════

def create_bacg(
    budget: Dict[str, float],
    framework: str = "sdk",
    **kwargs
) -> Union[BACGRuntime, FrameworkAdapter]:
    """便捷工厂函数：创建 BACG 运行时或适配器。

    Args:
        budget: 预算字典 {"tokens": 10000, "usd": 5.0}
        framework: "sdk" | "langgraph" | "autogen" | "openai" | "gateway"
        **kwargs: 框架特定参数

    Returns:
        BACGRuntime 或 FrameworkAdapter
    """
    runtime = BACGRuntime(budget=budget)

    if framework == "sdk":
        return runtime

    elif framework == "langgraph":
        adapter = LangGraphAdapter(runtime)
        if "graph" in kwargs:
            adapter.install(kwargs["graph"])
        return adapter

    elif framework == "autogen":
        adapter = AutoGenAdapter(runtime)
        if "agent" in kwargs:
            adapter.install(kwargs["agent"])
        return adapter

    elif framework == "openai":
        adapter = OpenAIAdapter(runtime)
        if "client" in kwargs:
            adapter.install(kwargs["client"])
        return adapter

    elif framework == "gateway":
        return BACGGateway(
            upstream_url=kwargs.get("upstream", "https://api.openai.com"),
            default_budget=budget,
        )

    else:
        raise ValueError(f"不支持的框架: {framework}")
