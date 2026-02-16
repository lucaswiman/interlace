.PHONY: test clean docs docs-clean docs-html docs-clean-build

# Include reusable venv setup from parent directory
include ../venv.mk

test: $(VENV_BIN)activate
	timeout 300 $(PYTEST)

# Read The Docs documentation targets
docs: docs-html
	@echo "Documentation built in docs/_build/html"

docs-html:
	cd docs && $(MAKE) clean html

docs-clean:
	cd docs && rm -rf _build

docs-clean-build: docs-clean docs-html

clean: docs-clean
	rm -rf __pycache__ .pytest_cache .eggs *.egg-info dist build .uv-cache .venv
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
