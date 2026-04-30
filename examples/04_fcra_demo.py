"""Example 04: FCRA (Formal Contract Runtime Assurance) Demo"""
from __future__ import annotations

import asyncio

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agentkernel.resource.fcra import ContractRuntime

async def main():
    print("=" * 50)
    print("FCRA Demo: Formal Contract Runtime Assurance")
    print("=" * 50)

    runtime = ContractRuntime()

    # Define a contract
    contract = {
        "id": "demo_contract",
        "budget": {"tokens": 5000, "usd": 2.0},
        "constraints": {
            "max_depth": 3,
            "max_agents": 5,
        }
    }

    runtime.register_contract(contract)
    print(f"\nRegistered contract: {contract['id']}")
    print(f"  Budget: {contract['budget']}")

    # Check compliance
    compliance = runtime.check_compliance("demo_contract", {"tokens": 3000})
    print(f"\nCompliance check (tokens=3000): {compliance}")

    compliance2 = runtime.check_compliance("demo_contract", {"tokens": 6000})
    print(f"Compliance check (tokens=6000): {compliance2}")

    print("\n✅ FCRA demo complete")

if __name__ == "__main__":
    asyncio.run(main())
