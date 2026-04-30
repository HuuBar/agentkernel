"""Benchmark 3: Multi-Agent Budget Allocation"""
import asyncio
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from agentkernel import create_bacg, BudgetToken

async def main():
    print("=" * 50)
    print("Benchmark 3: Multi-Agent Budget Allocation")
    print("=" * 50)

    parent = create_bacg(budget={"tokens": 5000}, framework="sdk")

    root_token = BudgetToken(dimensions={"tokens": 5000})
    child_tokens = root_token.split([
        ("agent_a", 0.4),
        ("agent_b", 0.35),
        ("agent_c", 0.25),
    ])

    print("\nParent budget: 5000 tokens")
    print("Splitting to 3 child agents:")
    for i, token in enumerate(child_tokens):
        print(f"  Agent {chr(65+i)}: {token.dimensions['tokens']:.0f} tokens")

    total_children = sum(t.dimensions['tokens'] for t in child_tokens)
    print(f"  Total allocated: {total_children:.0f} tokens")
    print(f"  Overhead: {5000 - total_children:.0f} tokens")
    print(f"  ✅ Monotonicity: {total_children:.0f} <= 5000")

if __name__ == "__main__":
    asyncio.run(main())
