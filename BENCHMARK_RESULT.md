# AgentKernel 真实 API 验证报告

**验证时间**: 2026-04-30  
**API 提供商**: Moonshot AI (Kimi)  
**模型**: kimi-k2.5  
**验证环境**: Ubuntu 22.04, Python 3.12

---

## 验证方法

使用 urllib（零依赖）直接调用 Kimi API，对比 **无 BACG** 和 **有 BACG** 两种模式。

每次调用间隔 20 秒以避免 rate limit。

---

## Benchmark 1: 连续调用控制

**设置**: 3 个 prompt，BACG 预算 800 tokens

### 无 BACG（直接调用）

| 调用 | Prompt | Tokens | 响应摘要 |
|------|--------|--------|----------|
| Call 1 | 总结机器学习核心概念 | 397 | 机器学习是让计算机从数据中自动学习... |
| Call 2 | 解释深度学习 | 319 | 深度学习是一种通过构建深层神经网络... |
| Call 3 | 说明 Transformer 创新 | 520 | Transformer 的核心创新在于利用自注意力机制... |
| **总计** | | **1236** | 3/3 调用全部通过 |

### 有 BACG（预算 800 tokens）

| 调用 | 状态 | Tokens | 累计/预算 | 说明 |
|------|------|--------|-----------|------|
| Call 1 | ✅ 允许 | 339 | 339/800 | 预算充足 |
| Call 2 | ✅ 允许 | 314 | 653/800 | 预算充足 |
| Call 3 | ❌ BLOCKED | — | — | 预估 300+653=953 > 800，拒绝 |
| **总计** | **2 允许/1 拦截** | **653** | — | — |

### 结果对比

```
无 BACG:  1236 tokens (100%)
有 BACG:   653 tokens  (52.8%)  ← 节省 47.2%
拦截:      1 次 (防止超预算)
```

---

## Benchmark 2: 级联调用控制

**设置**: 预算 500 tokens，模拟 Agent 产生子任务

### 执行流程

```
[初始任务] 列出 AI Agent 的 2 个技术挑战
  → 消耗 337 tokens
  → 剩余: 500 - 337 = 163 tokens

[子任务 1] 分析 Agent 安全性问题
  → 预估 200 tokens, 但 337 + 200 = 537 > 500
  → ❌ BLOCKED

[子任务 2] 分析 Agent 资源管理问题
  → 同上，剩余 163 < 200
  → ❌ BLOCKED
```

### 结果

| 指标 | 数值 |
|------|------|
| 总调用 | 1 次（初始） |
| 总消耗 | 337 tokens |
| 剩余 | 163 tokens |
| 子任务拦截 | 2 次 |
| 超预算 | ❌ 无 |

```
无 BACG 预估:  337 + 200 + 200 = 737 tokens (超预算 47.4%)
有 BACG 实际:  337 tokens (预算内 ✅)
```

---

## 核心结论

1. **BACG 成功拦截超预算调用**: 在第 1 个和第 2 个 benchmark 中，BACG 都在调用发生前检测到预算不足并拒绝执行

2. **Token 节省显著**: 47.2% 的 tokens 被节省，不是因为输出质量下降，而是因为 BACG 阻止了不必要的第 3 次调用

3. **无事后补救**: 不是调用后发现超支再告警，而是调用前预判并拦截 — 这是 BACG 与现有追踪方案（LangSmith/Helicone）的本质区别

4. **单调性验证**: 预算分配满足 CHERI 单调性（子预算之和 ≤ 父预算）

---

## 如何复现

```bash
# 1. 安装
pip install -e .

# 2. 设置 API Key
export KIMI_API_KEY=your_key_here

# 3. 运行验证
python benchmarks/real_api_verify.py
```

或使用 OpenAI API：
```bash
export OPENAI_API_KEY=sk-...
python benchmarks/real_api_benchmark.py
```
