"""Example 01: Basic AgentKernel usage — spawn, govern, and communicate."""
from __future__ import annotations

import asyncio
import logging

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agentkernel import AgentKernel, AgentId, Message, MessageKind, ContractMode, ResourceBudget
from agentkernel.core.actor import Actor
from agentkernel.core.governor import AgentContract

logging.basicConfig(level=logging.INFO)


class DataAnalyst(Actor):
    """A simple agent that simulates data analysis work."""

    async def receive(self, msg: Message) -> dict:
        task = msg.payload.get("task", "unknown")
        await asyncio.sleep(0.05)
        return {
            "status": "completed",
            "task": task,
            "insights": ["trend_up", "anomaly_detected"],
            "tokens_used": 1200,
        }


class Manager(Actor):
    """A manager agent that delegates to sub-agents and aggregates results."""

    def __init__(self, agent_id: AgentId, kernel: AgentKernel):
        super().__init__(agent_id)
        self.kernel = kernel
        self.results: dict = {}

    async def receive(self, msg: Message) -> dict:
        analyst1 = await self.kernel.spawn_agent(
            DataAnalyst,
            agent_id=AgentId("analyst", "team_a"),
            contract_spec={
                "budget": {"max_tokens": 5000, "max_cost_usd": 0.5},
                "mode": ContractMode.ECONOMICAL,
            },
        )
        analyst2 = await self.kernel.spawn_agent(
            DataAnalyst,
            agent_id=AgentId("analyst", "team_b"),
            contract_spec={
                "budget": {"max_tokens": 5000, "max_cost_usd": 0.5},
                "mode": ContractMode.ECONOMICAL,
            },
        )

        self.kernel.actors.send(analyst1, Message(
            kind=MessageKind.TASK,
            sender=self.state.agent_id,
            recipient=analyst1,
            payload={"task": "analyze_q1_revenue"},
        ))
        self.kernel.actors.send(analyst2, Message(
            kind=MessageKind.TASK,
            sender=self.state.agent_id,
            recipient=analyst2,
            payload={"task": "analyze_q2_revenue"},
        ))

        await asyncio.sleep(0.2)
        return {"status": "delegated", "workers": [analyst1.full, analyst2.full]}


async def main():
    kernel = AgentKernel()
    await kernel.start()

    manager = await kernel.spawn_agent(
        Manager,
        agent_id=AgentId("manager", "executive"),
        contract_spec={
            "budget": {"max_tokens": 20_000, "max_cost_usd": 2.0},
            "mode": ContractMode.BALANCED,
        },
        kernel=kernel,
    )

    result = await kernel.ask(manager, {"objective": "quarterly_review"})
    print("\n=== Result ===")
    print(result.payload)

    active = kernel.governor.repository.list_active()
    print(f"\nActive contracts: {len(active)}")
    for c in active:
        print(f"  - {c.id}: mode={c.mode.value}, tokens={c.consumed_tokens}/{c.budget.max_tokens}")

    all_events = kernel.events.get_all_events()
    print(f"\nEvents persisted: {len(all_events)}")
    event_types = {}
    for e in all_events:
        event_types[e.event_type.value] = event_types.get(e.event_type.value, 0) + 1
    for et, count in sorted(event_types.items()):
        print(f"  - {et}: {count}")

    await kernel.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
