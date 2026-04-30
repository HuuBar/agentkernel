#!/bin/bash
set -e

echo "=========================================="
echo "AgentKernel Benchmark Suite"
echo "=========================================="
echo ""
echo "This suite compares Agent behavior WITH and WITHOUT BACG."
echo ""

if [ -z "$OPENAI_API_KEY" ]; then
    echo "⚠️  OPENAI_API_KEY not set. Running in SIMULATION mode."
    echo "   Set it to run against real APIs:"
    echo "   export OPENAI_API_KEY=sk-..."
    echo ""
fi

python benchmarks/01_openai_vs_bacg.py
python benchmarks/02_cascade_control.py
python benchmarks/03_multi_agent_budget.py

echo ""
echo "=========================================="
echo "Benchmark complete. See results above."
echo "=========================================="
