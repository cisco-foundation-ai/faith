.PHONY: superlint test test_all lint format lint_format

# Rule-function to check if a given command is installed.
define CHECK_COMMAND
.PHONY: check_installed_$(1)
check_installed_$(1):
	@command -v $(1) >/dev/null 2>&1 || (echo "Error: $(1) is not installed or not in PATH. Please install $(1)." >&2; exit 1)
endef
$(foreach cmd,docker yq,$(eval $(call CHECK_COMMAND,$(cmd))))


# Read the environment variables for super-linter from the GitHub Actions
# workflow file to ensure consistency between local and CI linting.
GITHUB_WORKFLOW_FILE := .github/workflows/lint.yml
SUPER_LINTER_ENV_FLAGS := $(shell yq eval \
    ".jobs.super-linter.steps[] | select(.uses | contains(\"super-linter/super-linter\")).env | to_entries | .[] | select(.key != \"GITHUB_TOKEN\") | select(.key != \"VALIDATE_ALL_CODEBASE\") | \"-e \(.key)=\(.value)\"" \
    "$(GITHUB_WORKFLOW_FILE)")

# Define the version of super-linter to use.
SUPER_LINTER_VERSION := "v8.2.0"

# Get the absolute path to the root of the main git repository.
GIT_MAIN := $(shell git rev-parse --path-format=absolute --git-common-dir | sed 's/\/\.git//')

superlint: check_installed_docker check_installed_yq
	@[ ! -n "$$(git status --porcelain)" ] || \
		( \
			echo "Error: There are uncommitted changes. Please commit or stash them before running superlint."; \
			exit 1; \
		)
	@docker run \
		$(SUPER_LINTER_ENV_FLAGS) \
		-e DEFAULT_BRANCH=main \
		-e LOG_LEVEL=NOTICE \
		-e RUN_LOCAL=true \
		-e VALIDATE_ALL_CODEBASE=false \
		-v "$(shell pwd):/tmp/lint" \
		-v "$(GIT_MAIN):$(GIT_MAIN)" \
		ghcr.io/super-linter/super-linter:$(SUPER_LINTER_VERSION)

FORMATTERS := \
	FIX_CLANG_FORMAT \
	FIX_GITHUB_ACTIONS_ZIZMOR \
	FIX_JSON_PRETTIER \
	FIX_MARKDOWN \
	FIX_MARKDOWN_PRETTIER \
	FIX_NATURAL_LANGUAGE \
	FIX_PYTHON_BLACK \
	FIX_PYTHON_ISORT \
	FIX_PYTHON_RUFF \
	FIX_PYTHON_RUFF_FORMAT \
	FIX_TYPESCRIPT_PRETTIER \
	FIX_YAML_PRETTIER
FORMAT_FLAGS := $(foreach arg, $(FORMATTERS), -e $(arg)=true)

superformat: check_installed_docker check_installed_yq
	@[ ! -n "$$(git status --porcelain)" ] || \
		( \
			echo "Error: There are uncommitted changes. Please commit or stash them before running superlint."; \
			exit 1; \
		)
	@docker run \
		$(SUPER_LINTER_ENV_FLAGS) \
		$(FORMAT_FLAGS) \
		-e DEFAULT_BRANCH=main \
		-e LOG_LEVEL=NOTICE \
		-e RUN_LOCAL=true \
		-e VALIDATE_ALL_CODEBASE=false \
		-v "$(shell pwd):/tmp/lint" \
		-v "$(GIT_MAIN):$(GIT_MAIN)" \
		ghcr.io/super-linter/super-linter:$(SUPER_LINTER_VERSION)

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
