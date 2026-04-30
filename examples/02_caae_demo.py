"""Example 02: CAAE (Constraints as Effects) Demo"""
from __future__ import annotations

import asyncio

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agentkernel.constraints.caae import (
    Effect, Precondition, Postcondition,
    interpret_with_constraints, State,
)

async def main():
    print("=" * 50)
    print("CAAE Demo: Constraints as Effects")
    print("=" * 50)

    # Define a state
    state = State()
    state.budget = {"tokens": 1000, "usd": 5.0}
    state.variables = {"query": "AI research trends"}

    # Define effects
    effects = [
        Effect("llm_call", {"tokens": 200, "usd": 0.5}),
        Effect("tool_invoke", {"tokens": 50, "usd": 0.1}),
        Effect("search", {"tokens": 100, "usd": 0.2}),
    ]

    print(f"\nInitial state: {state}")
    print(f"Effects to apply: {len(effects)}")

    for eff in effects:
        print(f"  Applying: {eff.name} (cost={eff.cost})")
        for k, v in eff.cost.items():
            state.budget[k] = state.budget.get(k, 0) - v
        print(f"  Remaining budget: {state.budget}")

    print(f"\nFinal state: {state}")
    print("✅ CAAE demo complete")

if __name__ == "__main__":
    asyncio.run(main())
