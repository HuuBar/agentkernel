"""Example 05: EcoAct Demo — Ecological Actor Model"""
from __future__ import annotations

import asyncio

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agentkernel.resource.ecoact import EcoActRuntime

async def main():
    print("=" * 50)
    print("EcoAct Demo: Ecological Actor Model")
    print("=" * 50)

    runtime = EcoActRuntime()

    # Simulate resource cycles
    print("\nSimulating ecological resource cycles:")
    resources = {"cpu": 100, "memory": 500, "tokens": 10000}

    for cycle in range(3):
        print(f"\n  Cycle {cycle + 1}:")
        print(f"    Before: {resources}")

        # Consumption
        resources["tokens"] -= 2000
        resources["cpu"] -= 20

        # Regeneration (ecological principle)
        resources["cpu"] = min(100, resources["cpu"] + 10)

        print(f"    After:  {resources}")

    print("\n✅ EcoAct demo complete")

if __name__ == "__main__":
    asyncio.run(main())
