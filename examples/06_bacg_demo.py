"""Example 06: BACG Demo — Budget-Aware Call Graph"""
from __future__ import annotations

import asyncio

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agentkernel import create_bacg, BudgetToken, CallGraph
from agentkernel.topology.agent import BACGAgent
from agentkernel.topology.budget import BranchingModel

async def demo_budget_token():
    """Demo 1: BudgetToken monotonicity"""
    print("\n" + "-" * 40)
    print("Demo 1: BudgetToken Split (CHERI Monotonicity)")
    print("-" * 40)

    token = BudgetToken(dimensions={"tokens": 10000, "usd": 5.0})
    print(f"Parent token: {token}")

    children = token.split([
        ("agent_a", 0.4),
        ("agent_b", 0.35),
        ("agent_c", 0.25),
    ])

    for child in children:
        print(f"  Child: {child}")

    total_children = sum(c.dimensions["tokens"] for c in children)
    print(f"Sum of children tokens: {total_children:.0f}")
    print(f"Parent original: 10000")
    print(f"Monotonicity preserved: {total_children <= 10000}")


async def demo_bacg_runtime():
    """Demo 2: BACGRuntime budget enforcement"""
    print("\n" + "-" * 40)
    print("Demo 2: BACGRuntime Budget Enforcement")
    print("-" * 40)

    runtime = create_bacg(
        budget={"tokens": 2000, "usd": 1.0},
        framework="sdk"
    )

    print(f"Runtime: {runtime}")

    # Try operations within budget
    for i in range(3):
        estimated = {"tokens": 500, "usd": 0.2}
        can_do, reason = runtime.can_execute("llm_call", estimated)
        if can_do:
            runtime.total_consumed["tokens"] += estimated["tokens"]
            runtime.total_consumed["usd"] += estimated["usd"]
            print(f"  Op {i+1}: ✅ allowed (consumed {estimated})")
        else:
            print(f"  Op {i+1}: ❌ rejected: {reason}")

    print(f"Remaining: {runtime.remaining}")


async def demo_bacg_agent():
    """Demo 3: BACGAgent topology-aware execution"""
    print("\n" + "-" * 40)
    print("Demo 3: BACGAgent Topology-Aware Execution")
    print("-" * 40)

    model = BranchingModel(
        mean_branching=0.5,
        variance_branching=0.2,
        mean_cost={"tokens": 500, "usd": 0.01},
    )

    # Inject some history
    for _ in range(5):
        model.observe(n_children=0, cost={"tokens": 400, "usd": 0.008})
    for _ in range(3):
        model.observe(n_children=1, cost={"tokens": 600, "usd": 0.012})

    agent = BACGAgent(
        name="ResearchAgent",
        total_budget={"tokens": 5000, "usd": 2.5},
        branching_model=model,
    )

    result, graph, summary = await agent.execute("Research AI safety")
    print(f"\nSummary: {summary}")


async def main():
    print("=" * 50)
    print("BACG Demo: Budget-Aware Call Graph")
    print("=" * 50)

    await demo_budget_token()
    await demo_bacg_runtime()
    await demo_bacg_agent()

    print("\n" + "=" * 50)
    print("✅ All BACG demos complete")
    print("=" * 50)

if __name__ == "__main__":
    asyncio.run(main())
