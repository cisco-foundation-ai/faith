.PHONY: test test_all lint format lint_format

# Run only unit tests (excluding slow tests).
test:
	pytest tests --cov=src --suppress-no-test-exit-code --cov-report=term-missing --durations=50

# Run all tests including slow tests.
test_all:
	pytest tests --cov=src --runslow --suppress-no-test-exit-code --cov-report=term-missing --durations=50

# Lint using ruff, black, isort, and mypy
lint:
	( \
	    black --check src tests && \
		flake8 --config=.github/linters/.flake8 src tests && \
	    mypy --ignore-missing-imports --disallow-untyped-defs --show-error-codes src tests && \
	    run_ri_lint \
	)

# Format using ruff, isort, and black
format:
	black src tests
	run_ri_format

# Run both format and lint
lint_format: format lint
