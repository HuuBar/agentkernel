.PHONY: install test benchmark demo clean lint format

install:
	pip install -e .

install-benchmark:
	pip install -e ".[benchmark]"

test:
	python -m pytest tests/ -v

benchmark:
	@echo "=== Running BACG Benchmarks ==="
	@echo "These benchmarks compare WITH and WITHOUT BACG"
	@echo ""
	python benchmarks/01_openai_vs_bacg.py
	python benchmarks/02_cascade_control.py
	python benchmarks/03_multi_agent_budget.py

demo:
	@echo "=== Running BACG Demos ==="
	python examples/01_basic.py
	python examples/06_bacg_demo.py

lint:
	ruff check agentkernel/ tests/ benchmarks/

format:
	black agentkernel/ tests/ benchmarks/

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
