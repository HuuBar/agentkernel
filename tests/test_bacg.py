"""BACG Core Tests"""
import pytest
import asyncio
from agentkernel import (
    BudgetToken, CallGraph, BACGRuntime, create_bacg,
    TopologyLayer, ResourceLayer, IdentityLayer,
)
from agentkernel.topology.constraints import ConstraintNode


class TestBudgetToken:
    """Test BudgetToken monotonicity and operations."""

    def test_split_monotonicity(self):
        """Token split must maintain monotonicity."""
        token = BudgetToken(dimensions={"tokens": 1000, "usd": 5.0})
        children = token.split([
            ("child_a", 0.5),
            ("child_b", 0.3),
        ])

        assert len(children) == 2
        total = sum(c.dimensions["tokens"] for c in children)
        assert total <= 1000 * 1.001  # Allow small floating point error

    def test_can_afford(self):
        token = BudgetToken(dimensions={"tokens": 100, "usd": 0.5})
        assert token.can_afford({"tokens": 50})
        assert not token.can_afford({"tokens": 150})

    def test_consume(self):
        token = BudgetToken(dimensions={"tokens": 100})
        remaining = token.consume({"tokens": 30})
        assert remaining["tokens"] == 70
        assert token.dimensions["tokens"] == 70


class TestCallGraph:
    """Test CallGraph construction and transformations."""

    def test_create_graph(self):
        graph = CallGraph(root_id="test_root")
        assert "test_root" in graph.nodes
        assert graph.nodes["test_root"].op_type == "root"

    def test_add_layers(self):
        graph = CallGraph()
        graph.add_layer(IdentityLayer())
        graph.add_layer(ResourceLayer())
        assert len(graph.layers) == 2

    def test_branching_stats_empty(self):
        graph = CallGraph()
        stats = graph.branching_stats()
        assert stats["mean"] == 0
        assert stats["total_nodes"] == 1  # Just root


class TestBACGRuntime:
    """Test BACGRuntime resource control."""

    def test_can_execute_within_budget(self):
        runtime = BACGRuntime(budget={"tokens": 1000})
        can_do, reason = runtime.can_execute("llm_call", {"tokens": 100})
        assert can_do is True
        assert reason is None

    def test_can_execute_over_budget(self):
        runtime = BACGRuntime(budget={"tokens": 100})
        runtime.total_consumed["tokens"] = 80
        can_do, reason = runtime.can_execute("llm_call", {"tokens": 50})
        assert can_do is False
        assert "resource" in reason or "需要" in reason

    def test_remaining(self):
        runtime = BACGRuntime(budget={"tokens": 1000})
        runtime.total_consumed["tokens"] = 300
        remaining = runtime.remaining
        assert remaining["tokens"] == 700


class TestTopologyLayer:
    """Test TopologyLayer branching constraints."""

    def test_branching_check(self):
        layer = TopologyLayer(max_depth=3, max_branching=2.0)
        layer.observe_branching(2)
        layer.observe_branching(3)
        assert layer.mean_branching > 0

    def test_recommend_action(self):
        layer = TopologyLayer()
        # No observations yet
        assert layer.recommend_action() == "NORMAL"

        # Simulate high branching
        for _ in range(10):
            layer.observe_branching(5)
        assert layer.recommend_action() == "TRUNCATE"


def test_integration_import():
    """Test that all major exports work."""
    from agentkernel import (
        BudgetToken, CallGraph, BACGRuntime,
        BACGAgent, UsageExtractor, create_bacg,
    )
    assert BudgetToken is not None
    assert CallGraph is not None
    assert BACGRuntime is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
