# AgentKernel

**Budget-Aware Call Graph (BACG) — Topology-aware resource allocation for AI Agents**

[![CI](https://github.com/HuuBar/agentkernel/actions/workflows/ci.yml/badge.svg)](https://github.com/HuuBar/agentkernel/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Why AgentKernel?

Current frameworks (LangGraph, AutoGen, CrewAI) solve "how Agents collaborate" but not "how Agent systems run sustainably." They're all ephemeral: no persistent identity, no resource budgets, no self-correction.

AgentKernel fills that gap with **BACG (Budget-Aware Call Graph)** — a topology-aware resource allocation paradigm based on the Galton-Watson branching process.

## Core Insight

Agent consumption is not a scalar — it's a branching process on the Call Graph:

- One Agent call → triggers Z child calls (Z is a random variable)
- Each child call → triggers Z' child calls
- Total consumption = sum of all node consumptions in the call tree

**Expected Total Consumption:**

```
E[Total] = E[C] / (1 - m)   when m < 1 (converges)
```

Where:
- `m = E[Z]` = average branching factor
- `E[C]` = average consumption per call
- If `m >= 1`, expected total is **infinite** → must force termination

## Installation

```bash
pip install -e .
```

For benchmark dependencies:
```bash
pip install -e ".[benchmark]"
```

## Quick Start

```python
from agentkernel import create_bacg, BudgetToken

# Create runtime with budget
runtime = create_bacg(budget={"tokens": 2000}, framework="sdk")

# Check if operation is allowed
can_do, reason = runtime.can_execute("llm_call", {"tokens": 500})
if can_do:
    print("Allowed!")
else:
    print(f"Blocked: {reason}")

# Split budget among child agents
root_token = BudgetToken(dimensions={"tokens": 5000})
children = root_token.split([("agent_a", 0.4), ("agent_b", 0.35), ("agent_c", 0.25)])
# children budgets sum <= 5000 (CHERI monotonicity)
```

## Project Structure

```
agentkernel/
├── agentkernel/
│   ├── __init__.py              # Main API: BACGRuntime, CallGraph, BudgetToken
│   ├── types.py                 # Core type definitions
│   ├── core/                    # Actor Model + Event Sourcing + Governor
│   │   ├── kernel.py            # AgentKernel runtime
│   │   ├── actor.py             # Actor Model concurrency
│   │   ├── event_store.py       # Event Sourcing + CQRS
│   │   └── governor.py          # Agent Contracts
│   ├── topology/                # BACG — Core innovation
│   │   ├── token.py             # BudgetToken (CHERI monotonicity)
│   │   ├── constraints.py       # 6 Constraint Layers
│   │   ├── graph.py             # CallGraph (attribute directed tree)
│   │   ├── branching.py         # BranchingModel (Galton-Watson)
│   │   ├── budget.py            # TopologyBudget
│   │   ├── agent.py             # BACGAgent / BACGAgentV2
│   │   └── integration.py       # BACGRuntime + Framework Adapters
│   ├── constraints/             # CAAE — Constraints as Effects
│   ├── strategies/              # Strategy system
│   ├── resource/                # FCRA + EcoAct
│   └── protocol/                # MCP + A2A adapters
├── examples/                    # 7 example demos
├── tests/                       # Core tests
├── benchmarks/                  # 3 benchmark scenarios
├── website/                     # GitHub Pages site
└── .github/workflows/           # CI/CD
```

## The 6 Constraint Layers

Like Photoshop layers, each constraint layer works independently:

| Layer | Scope | Purpose |
|-------|-------|---------|
| **Identity** | Node | Actor permission control (AWS IAM style) |
| **Resource** | Path | BudgetToken flow (Petri net style) |
| **Security** | Edge | Operation whitelist (Take-Grant model) |
| **Quality** | Path | Quality decay along depth |
| **Concurrency** | Level | Width control (graph coloring) |
| **Topology** | Global | Branching process enforcement |

## Benchmarks

Run benchmarks to see BACG in action:

```bash
make benchmark
```

Or individually:
```bash
python benchmarks/01_openai_vs_bacg.py    # With vs Without BACG
python benchmarks/02_cascade_control.py   # Cascade prevention
python benchmarks/03_multi_agent_budget.py # Multi-agent budget split
```

## Testing

```bash
make test
```

## License

MIT License — open for contribution, fork, and production use.
