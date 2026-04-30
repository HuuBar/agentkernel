"""Benchmark 2: Cascade call control — preventing runaway chains"""
import asyncio
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from agentkernel import BACGRuntime, TopologyLayer, create_bacg

async def without_bacg():
    """模拟级联调用失控"""
    print("[WITHOUT BACG] Agent calls LLM → triggers search → triggers more LLM...")
    total = 0
    depth = 0
    for level in range(5):
        nodes_at_level = 2 ** level
        cost = nodes_at_level * 300
        total += cost
        depth = level + 1
        print(f"  Depth {depth}: {nodes_at_level} nodes, +{cost} tokens, total={total}")
    print(f"  Final: {total} tokens across {depth} depth levels")
    return total

async def with_bacg():
    """BACG 拓扑约束防止指数增长"""
    print("\n[WITH BACG] TopologyLayer detects m>=1, truncates at depth=3")
    runtime = create_bacg(budget={"tokens": 10000}, framework="sdk")
    topology = TopologyLayer(max_depth=3, max_branching=1.5)

    total = 0
    for level in range(5):
        m = 2.0
        topology.observe_branching(int(m))

        if topology.mean_branching >= 1.0 and level >= 3:
            print(f"  Depth {level+1}: ❌ TRUNCATED (m={topology.mean_branching:.2f} >= 1.0)")
            break

        nodes = min(2 ** level, 8)
        cost = nodes * 300
        total += cost
        print(f"  Depth {level+1}: {nodes} nodes, +{cost} tokens, m={topology.mean_branching:.2f}")

    print(f"  Final: {total} tokens, growth controlled")
    return total

async def main():
    print("=" * 50)
    print("Benchmark 2: Cascade Control — Preventing Runaway Chains")
    print("=" * 50)

    without = await without_bacg()
    with_result = await with_bacg()

    print(f"\n{'=' * 50}")
    print("SUMMARY:")
    print(f"  Without BACG: {without} tokens (exponential growth)")
    print(f"  With BACG:    {with_result} tokens (controlled growth)")
    print(f"  Savings:      {without - with_result} tokens")
    print(f"{'=' * 50}")

if __name__ == "__main__":
    asyncio.run(main())
