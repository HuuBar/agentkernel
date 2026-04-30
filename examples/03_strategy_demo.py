"""Example 03: Strategy Demo"""
from __future__ import annotations

import asyncio

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agentkernel.strategies.strategies import StrategySelector

async def main():
    print("=" * 50)
    print("Strategy Demo: Agent Strategy Selection")
    print("=" * 50)

    selector = StrategySelector()

    # Register strategies
    selector.register("direct", priority=1)
    selector.register("plan_first", priority=2)
    selector.register("research_then_act", priority=3)

    print("\nAvailable strategies:")
    for name, info in selector.strategies.items():
        print(f"  - {name}: priority={info['priority']}")

    selected = selector.select("complex_research_task")
    print(f"\nSelected strategy for 'complex_research_task': {selected}")
    print("✅ Strategy demo complete")

if __name__ == "__main__":
    asyncio.run(main())
