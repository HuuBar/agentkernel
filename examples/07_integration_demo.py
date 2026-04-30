"""Example 07: Integration Demo — BACG + Framework Adapters"""
from __future__ import annotations

import asyncio

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agentkernel import create_bacg, BACGRuntime, UsageExtractor

async def main():
    print("=" * 50)
    print("Integration Demo: BACG Framework Adapters")
    print("=" * 50)

    # Create BACG runtime (SDK mode)
    runtime = create_bacg(
        budget={"tokens": 5000, "usd": 5.0},
        framework="sdk"
    )
    print(f"\n1. Created BACGRuntime: {runtime.agent_name}")
    print(f"   Budget: {runtime.total_budget}")

    # Simulate LLM calls
    print("\n2. Simulating LLM calls with budget control:")
    prompts = [
        "What is machine learning?",
        "Explain neural networks",
        "Describe deep learning",
        "What are transformers?",
    ]

    for prompt in prompts:
        try:
            result = await runtime.llm_call(prompt, estimated_cost={"tokens": 800})
            print(f"   ✅ '{prompt[:30]}...' -> {result['usage']}")
        except Exception as e:
            print(f"   ❌ '{prompt[:30]}...' -> {e}")

    print(f"\n3. Final state:")
    print(f"   Remaining: {runtime.remaining}")
    print(f"   Operations: {len(runtime.operations)}")

    report = runtime.get_report()
    print(f"\n4. Report: {report['agent']}")
    print(f"   Total consumed: {report['total_consumed']}")

    # Test UsageExtractor
    print("\n5. UsageExtractor:")
    raw = {"total_tokens": 150, "prompt_tokens": 100, "completion_tokens": 50, "cost": 0.002}
    extracted = UsageExtractor.from_raw(raw)
    print(f"   Input: {raw}")
    print(f"   Extracted: {extracted}")

    print("\n✅ Integration demo complete")

if __name__ == "__main__":
    asyncio.run(main())
