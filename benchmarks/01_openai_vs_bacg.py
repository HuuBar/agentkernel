"""Benchmark 1: OpenAI SDK with vs without BACG"""
import asyncio
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import os
import time
from agentkernel import BACGRuntime, create_bacg

async def without_bacg():
    """模拟无BACG控制的情况"""
    print("[WITHOUT BACG] 连续调用 LLM API，无预算限制")
    total_tokens = 0
    calls = []
    for i in range(5):
        cost = {"tokens": 500 + i * 50}
        total_tokens += cost["tokens"]
        calls.append(cost)
        print(f"  Call {i+1}: +{cost['tokens']} tokens, total={total_tokens}")
    print(f"  Final: {total_tokens} tokens consumed, no limit enforced")
    return total_tokens

async def with_bacg():
    """有BACG控制的情况"""
    print("\n[WITH BACG] BACGRuntime 控制，预算=2000 tokens")
    runtime = create_bacg(
        budget={"tokens": 2000},
        framework="sdk"
    )
    total_allowed = 0
    rejected = 0
    for i in range(5):
        estimated = {"tokens": 500 + i * 50}
        can_do, reason = runtime.can_execute("llm_call", estimated)
        if can_do:
            total_allowed += estimated["tokens"]
            runtime.total_consumed["tokens"] += estimated["tokens"]
            print(f"  Call {i+1}: ✅ allowed, total={total_allowed}")
        else:
            rejected += 1
            print(f"  Call {i+1}: ❌ REJECTED: {reason}")
    print(f"  Final: {total_allowed} tokens allowed, {rejected} calls rejected")
    return total_allowed, rejected

async def main():
    print("=" * 50)
    print("Benchmark 1: OpenAI SDK — With vs Without BACG")
    print("=" * 50)

    without = await without_bacg()
    with_bacg_result, rejected = await with_bacg()

    print(f"\n{'=' * 50}")
    print("SUMMARY:")
    print(f"  Without BACG: {without} tokens (uncontrolled)")
    print(f"  With BACG:    {with_bacg_result} tokens, {rejected} calls rejected")
    if without > 0:
        print(f"  Savings:      {without - with_bacg_result} tokens ({(without - with_bacg_result) / without * 100:.1f}%)")
    print(f"{'=' * 50}")

if __name__ == "__main__":
    asyncio.run(main())
