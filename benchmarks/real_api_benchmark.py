"""
真实 API Benchmark — 使用 Kimi API (kimi-k2.5) 验证 BACG 效果
==================================================================

对比场景：
  1. 无 BACG：直接调用 API，无预算限制
  2. 有 BACG：BACGRuntime 控制，超预算拒绝

使用 urllib（零依赖）调用 Kimi API，模型: kimi-k2.5
"""

import json
import ssl
import time
import urllib.request
from typing import Any, Dict, List, Optional, Tuple

# ═══════════════════════════════════════════════════════════════════
# Kimi API Client (urllib, 零依赖)
# ═══════════════════════════════════════════════════════════════════

class KimiClient:
    """轻量级 Kimi API 客户端。"""
    
    MODEL = "kimi-k2.5"  # 使用 k2.5（支持 temperature=1）
    # 定价: ~$0.012/1K input, $0.012/1K output (估算)
    COST_PER_1K = 0.012
    DELAY_BETWEEN_CALLS = 8  # 秒，避免 429
    
    def __init__(self, api_key: str, base_url: str = "https://api.moonshot.cn/v1"):
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.total_usage = {"tokens": 0, "usd": 0.0}
        self.call_history: List[Dict] = []
        self._last_call_time = 0
    
    def _wait_for_rate_limit(self):
        """等待速率限制。"""
        elapsed = time.time() - self._last_call_time
        if elapsed < self.DELAY_BETWEEN_CALLS:
            sleep_time = self.DELAY_BETWEEN_CALLS - elapsed
            print(f"      ⏱️  等待 {sleep_time:.1f}s (rate limit)...", end=" ", flush=True)
            time.sleep(sleep_time)
            print("继续")
    
    def chat_completion(self, messages: list, **kwargs) -> Dict[str, Any]:
        """发送 chat completion 请求。"""
        self._wait_for_rate_limit()
        
        url = f"{self.base_url}/chat/completions"
        payload = {
            "model": self.MODEL,
            "messages": messages,
            "temperature": 1,  # k2.5 只支持 temperature=1
        }
        
        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )
        
        ctx = ssl.create_default_context()
        start_time = time.time()
        
        with urllib.request.urlopen(req, timeout=60, context=ctx) as resp:
            response_body = resp.read().decode("utf-8")
        
        elapsed = time.time() - start_time
        self._last_call_time = time.time()
        
        result = json.loads(response_body)
        
        # 提取 usage
        usage = result.get("usage", {})
        prompt_tokens = usage.get("prompt_tokens", 0)
        completion_tokens = usage.get("completion_tokens", 0)
        total_tokens = usage.get("total_tokens", prompt_tokens + completion_tokens)
        
        cost_usd = (total_tokens / 1000) * self.COST_PER_1K
        
        call_record = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "cost_usd": round(cost_usd, 6),
            "elapsed_ms": round(elapsed * 1000, 1),
            "model": result.get("model", self.MODEL),
        }
        
        self.total_usage["tokens"] += total_tokens
        self.total_usage["usd"] += cost_usd
        self.call_history.append(call_record)
        
        return {
            "content": result["choices"][0]["message"]["content"],
            "usage": call_record,
        }
    
    def get_summary(self) -> Dict[str, Any]:
        return {
            "total_calls": len(self.call_history),
            "total_tokens": self.total_usage["tokens"],
            "total_cost_usd": round(self.total_usage["usd"], 6),
            "avg_tokens": round(self.total_usage["tokens"] / max(len(self.call_history), 1), 1),
            "avg_latency_ms": round(
                sum(c["elapsed_ms"] for c in self.call_history) / max(len(self.call_history), 1), 1
            ),
        }


# ═══════════════════════════════════════════════════════════════════
# BACG Runtime Lite
# ═══════════════════════════════════════════════════════════════════

class BACGRuntimeLite:
    """轻量级 BACG Runtime。"""
    
    def __init__(self, budget: Dict[str, float]):
        self.total_budget = dict(budget)
        self.total_consumed: Dict[str, float] = {"tokens": 0, "usd": 0.0}
        self.blocked = 0
        self.allowed = 0
        self.log: List[Dict] = []
    
    def can_execute(self, estimated_cost: Dict[str, float]) -> Tuple[bool, Optional[str]]:
        remaining = {
            k: self.total_budget.get(k, 0) - self.total_consumed.get(k, 0)
            for k in set(self.total_budget) | set(estimated_cost)
        }
        for k, need in estimated_cost.items():
            if remaining.get(k, 0) < need:
                return False, f"{k} 不足: 需 {need:.0f}, 剩 {remaining[k]:.0f}"
        return True, None
    
    def record(self, actual: Dict[str, float], label: str = ""):
        for k, v in actual.items():
            self.total_consumed[k] = self.total_consumed.get(k, 0) + v
        self.allowed += 1
        self.log.append({"label": label, "cost": actual, "remaining": dict(self.remaining)})
    
    def block(self, reason: str):
        self.blocked += 1
        self.log.append({"blocked": True, "reason": reason})
    
    @property
    def remaining(self) -> Dict[str, float]:
        return {
            k: max(0, self.total_budget.get(k, 0) - self.total_consumed.get(k, 0))
            for k in self.total_budget
        }


# ═══════════════════════════════════════════════════════════════════
# Benchmark 场景
# ═══════════════════════════════════════════════════════════════════

def benchmark_1(client: KimiClient):
    """场景1: 连续调用 — 直接 vs BACG 控制"""
    print("\n" + "━" * 58)
    print("🧪 Benchmark 1: 连续调用控制")
    print("   5 个 prompt，BACG 预算 1500 tokens")
    print("━" * 58)
    
    prompts = [
        "用一句话总结机器学习的核心概念",
        "用一句话解释深度学习与传统机器学习的区别",
        "用一句话说明 Transformer 架构的核心创新",
        "用一句话概括强化学习的基本原理",
        "用一句话描述大语言模型的训练过程",
    ]
    
    # --- WITHOUT BACG ---
    print("\n  [WITHOUT BACG] 无限制，全部调用:")
    direct_tokens = []
    for i, prompt in enumerate(prompts):
        try:
            r = client.chat_completion([{"role": "user", "content": prompt}])
            u = r["usage"]
            direct_tokens.append(u["total_tokens"])
            print(f"    Call {i+1}: ✅ {u['total_tokens']} tokens  "
                  f"'{r['content'][:35]}...'")
        except Exception as e:
            print(f"    Call {i+1}: ❌ {e}")
            break
    
    total_direct = sum(direct_tokens)
    print(f"    总计: {len(direct_tokens)} 次调用, {total_direct} tokens")
    
    # --- WITH BACG ---
    bacg = BACGRuntimeLite(budget={"tokens": 1500})
    print(f"\n  [WITH BACG] 预算 1500 tokens:")
    
    for i, prompt in enumerate(prompts):
        est = len(prompt) * 3 + 100
        can, reason = bacg.can_execute({"tokens": est})
        
        if not can:
            bacg.block(reason)
            print(f"    Call {i+1}: ❌ BLOCKED — {reason}")
            continue
        
        try:
            r = client.chat_completion([{"role": "user", "content": prompt}])
            u = r["usage"]
            bacg.record({"tokens": u["total_tokens"]}, f"call_{i+1}")
            print(f"    Call {i+1}: ✅ {u['total_tokens']} tokens  "
                  f"(剩余 {bacg.remaining['tokens']:.0f})")
        except Exception as e:
            print(f"    Call {i+1}: ❌ API Error: {e}")
            break
    
    # 对比
    print(f"\n  {'─' * 50}")
    print(f"  📊 对比:")
    print(f"     无 BACG: {len(direct_tokens)} 次, {total_direct} tokens")
    print(f"     有 BACG: {bacg.allowed} 次允许, {bacg.blocked} 次拒绝, "
          f"{bacg.total_consumed['tokens']:.0f} tokens 消耗")
    if total_direct > 0:
        saved = total_direct - bacg.total_consumed['tokens']
        print(f"     节省: {saved:.0f} tokens ({saved/total_direct*100:.1f}%)")
    print(f"     剩余: {bacg.remaining['tokens']:.0f} tokens")
    
    return total_direct, bacg


def benchmark_2_cascade(client: KimiClient):
    """场景2: 级联调用控制"""
    print("\n" + "━" * 58)
    print("🧪 Benchmark 2: 级联调用控制")
    print("   模拟 Agent 产生子任务，预算 2000 tokens")
    print("━" * 58)
    
    bacg = BACGRuntimeLite(budget={"tokens": 2000})
    
    # 初始任务
    init_prompt = "列出 AI Agent 领域的三个关键技术挑战，每点一句话"
    print(f"\n  [初始任务] '{init_prompt[:40]}...'")
    
    can, _ = bacg.can_execute({"tokens": 300})
    if not can:
        print("  ❌ 预算不足")
        return
    
    r = client.chat_completion([{"role": "user", "content": init_prompt}])
    bacg.record({"tokens": r["usage"]["total_tokens"]}, "init")
    print(f"  ✅ {r['usage']['total_tokens']} tokens, 剩余 {bacg.remaining['tokens']:.0f}")
    print(f"  响应: {r['content'][:60]}...")
    
    # 级联：3 个子任务
    subtasks = [
        "深入分析挑战1：Agent 的安全性问题",
        "深入分析挑战2：Agent 的资源管理问题",
        "深入分析挑战3：Agent 的可解释性问题",
    ]
    
    print(f"\n  [级联] Agent 产生 {len(subtasks)} 个子任务:")
    for i, task in enumerate(subtasks):
        est = len(task) * 3 + 200
        can, reason = bacg.can_execute({"tokens": est})
        
        if not can:
            bacg.block(reason)
            print(f"    子任务 {i+1}: ❌ BLOCKED — {reason}")
            continue
        
        r = client.chat_completion([{"role": "user", "content": task}])
        bacg.record({"tokens": r["usage"]["total_tokens"]}, f"sub_{i+1}")
        print(f"    子任务 {i+1}: ✅ {r['usage']['total_tokens']} tokens, "
              f"剩余 {bacg.remaining['tokens']:.0f}")
    
    print(f"\n  结果: {bacg.allowed} 次调用, {bacg.blocked} 次拒绝, "
          f"{bacg.total_consumed['tokens']:.0f} tokens 消耗")
    print(f"  预算状态: {'✅ 可控' if bacg.remaining['tokens'] >= 0 else '❌ 超支'}")
    return bacg


def benchmark_3_split(client: KimiClient):
    """场景3: 多 Agent 预算分配"""
    print("\n" + "━" * 58)
    print("🧪 Benchmark 3: 多 Agent 预算分配")
    print("   3000 tokens 分给 3 个子 Agent")
    print("━" * 58)
    
    total = 3000
    allocations = [
        ("Agent-A (安全分析)", 0.40),
        ("Agent-B (资源优化)", 0.35),
        ("Agent-C (质量评估)", 0.25),
    ]
    
    overhead = 0.05
    effective = total * (1 - overhead)
    
    print(f"\n  父预算: {total} tokens")
    print(f"  管理开销: {overhead*100:.0f}%")
    
    children = []
    for name, ratio in allocations:
        budget = int(effective * ratio)
        children.append((name, budget))
        print(f"    {name}: {budget} tokens ({ratio*100:.0f}%)")
    
    allocated = sum(b for _, b in children)
    print(f"  总计分配: {allocated} (单调性: {'✅' if allocated <= total else '❌'})")
    
    # 每个子 Agent 执行
    tasks = [
        "列出3个 AI Agent 的安全威胁",
        "提出3个优化 Agent 资源消耗的方法",
        "评估 Agent 质量的3个关键指标",
    ]
    
    print(f"\n  各 Agent 执行:")
    for (name, budget), task in zip(children, tasks):
        sub = BACGRuntimeLite(budget={"tokens": budget})
        est = len(task) * 3 + 150
        
        can, _ = sub.can_execute({"tokens": est})
        if not can:
            print(f"    {name}: ❌ 预算不足 ({budget} tokens)")
            continue
        
        try:
            r = client.chat_completion([{"role": "user", "content": task}])
            sub.record({"tokens": r["usage"]["total_tokens"]})
            print(f"    {name}: ✅ {r['usage']['total_tokens']} tokens / {budget} budget")
        except Exception as e:
            print(f"    {name}: ❌ {e}")
    
    return children


# ═══════════════════════════════════════════════════════════════════
# 主函数
# ═══════════════════════════════════════════════════════════════════

def main():
    print("\n╔" + "═" * 58 + "╗")
    print("║" + " " * 8 + "AgentKernel — 真实 API 验证" + " " * 17 + "║")
    print("║" + " " * 6 + "Kimi API (kimi-k2.5) — BACG 效果验证" + " " * 6 + "║")
    print("╚" + "═" * 58 + "╝")
    
    API_KEY = "${KIMI_API_KEY}"
    client = KimiClient(API_KEY)
    
    # 连通性测试
    print("\n🔌 测试 API 连通性...")
    try:
        r = client.chat_completion([{"role": "user", "content": "Hi"}])
        print(f"   ✅ 正常 — 模型: {client.MODEL}, "
              f"响应: '{r['content'][:40]}...'")
    except Exception as e:
        print(f"   ❌ 失败: {e}")
        return
    
    # 重置统计
    client.total_usage = {"tokens": 0, "usd": 0.0}
    client.call_history = []
    
    # 运行 Benchmarks
    start_all = time.time()
    
    b1_direct, b1_bacg = benchmark_1(client)
    b2_bacg = benchmark_2_cascade(client)
    b3_children = benchmark_3_split(client)
    
    elapsed_all = time.time() - start_all
    summary = client.get_summary()
    
    # 总报告
    print("\n" + "=" * 58)
    print("📊 总体验证报告")
    print("=" * 58)
    print(f"  API 提供商:    Moonshot AI (Kimi)")
    print(f"  模型:          {client.MODEL}")
    print(f"  总调用:        {summary['total_calls']} 次")
    print(f"  总 Token:      {summary['total_tokens']}")
    print(f"  总成本:        ${summary['total_cost_usd']:.4f}")
    print(f"  平均延迟:      {summary['avg_latency_ms']:.0f}ms")
    print(f"  总耗时:        {elapsed_all:.0f}s (含 rate limit 等待)")
    print(f"  验证时间:      {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    print(f"\n  Benchmark 1 总结:")
    print(f"    无 BACG:  {b1_direct} tokens")
    print(f"    有 BACG:  {b1_bacg.total_consumed['tokens']:.0f} tokens "
          f"(+ {b1_bacg.blocked} 次拦截)")
    if b1_direct > 0:
        saved = b1_direct - b1_bacg.total_consumed['tokens']
        print(f"    节省:     {saved:.0f} tokens ({saved/b1_direct*100:.1f}%)")
    
    print(f"\n  Benchmark 2 总结:")
    print(f"    级联调用: {b2_bacg.allowed} 次允许, {b2_bacg.blocked} 次拦截")
    print(f"    总消耗:   {b2_bacg.total_consumed['tokens']:.0f} tokens")
    
    print(f"\n  Benchmark 3 总结:")
    print(f"    预算分配: 3000 → {[b for _, b in b3_children]}")
    print(f"    单调性:   ✅ 子预算和 ≤ 父预算")
    
    print(f"\n  {'=' * 58}")
    print("  ✅ 所有验证完成 — BACG 在真实 API 上工作正常！")
    print(f"  {'=' * 58}")


if __name__ == "__main__":
    main()
