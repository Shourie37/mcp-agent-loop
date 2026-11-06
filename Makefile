PY ?= python3

.PHONY: test typecheck demo eval workflow check clean

test:
	$(PY) -m pytest -q

typecheck:
	$(PY) -m mypy

demo:
	$(PY) -m mcp_agent_loop demo

eval:
	$(PY) -m mcp_agent_loop eval

workflow:
	$(PY) -m mcp_agent_loop workflow --auto-approve

check: typecheck test

clean:
	rm -rf traces .pytest_cache .mypy_cache
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
