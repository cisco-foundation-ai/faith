---
id: contributing
title: Contributing
sidebar_label: Contributing
sidebar_position: 6
---

# Contributing

Contributions include bug reports, fixes, tests, benchmarks, and documentation.
This page covers the development workflow; for the full
policy, see [`CONTRIBUTING.md`](https://github.com/cisco-foundation-ai/faith/blob/main/CONTRIBUTING.md)
and the [Code of Conduct](https://github.com/cisco-foundation-ai/faith/blob/main/CODE_OF_CONDUCT.md)
in the repository.

## Development setup

Work from a Git checkout and select Python 3.11 or 3.12 explicitly. For framework
development, install FAITH in editable mode with the `test` extra:

```shell title="Framework development setup"
uv venv --python 3.12 .venv
source .venv/bin/activate
uv pip install -e '.[test]'
```

The `test` extra includes all engine and deployment extras, including the vLLM
stack, plus pytest and its plugins. Use an environment supported by those
dependencies; this is not a general-purpose macOS or CPU-only installation.
For documentation-only changes, use the smaller setup below. See
[Installation](./getting-started/installation.md#for-development) for supported
environments and Git-derived versioning. A normal Git checkout does not require
`SETUPTOOLS_SCM_PRETEND_VERSION`.

### Documentation-only development

The documentation checks require FAITH's base dependencies, Python 3.11 or 3.12,
and Node.js 20.18.1 or later. They use test doubles for external services; no model
engine, GPU, container daemon, or cloud credentials are required. From the
repository root, create a separate environment without replacing an existing
development environment:

```shell title="Documentation development setup"
uv venv --python 3.12 .venv-docs
source .venv-docs/bin/activate
uv pip install -e .
cd website
npm ci
npm run check-repo
npm run build
npm run serve -- --host 127.0.0.1
```

Open the local address printed by the server. Keep the Python environment active
when running `check-repo`. The build also validates the rendered site's internal
links, anchors, and documentation-page reachability. Use `npm start` for a
development server that refreshes on edits; rerun the checks and production
build before submitting changes.

## Running tests

Tests run through the `Makefile`:

```shell
make test       # runs the standard suite with coverage
make test_all   # also runs slow tests (--runslow)
```

Both invoke `pytest` against `tests/` with coverage reporting. The Makefile
expects Bash with `globstar` support and reads lint configuration through `yq`
when loaded. Slow tests may require additional resources. Include tests for
changed behavior; `make test_all` does not enable the separate opt-in container
integration tests below.

### Read-only workspace integration tests

The dedicated `read-only-workspace` job in
[`.github/workflows/unit-test.yml`](https://github.com/cisco-foundation-ai/faith/blob/main/.github/workflows/unit-test.yml)
tests workspace protection separately from the standard suite. The integration
tests are opt-in and require Docker, the `docker` and `kubernetes` extras, and
pytest. From the repository root:

```shell
docker build --target agent -t faith-read-only-agent:test src/faith/__benchmarks__/vulnerability-localization
docker build --target runtime -t faith-read-only-runtime:test src/faith/__benchmarks__/vulnerability-localization
FAITH_TEST_READ_ONLY=1 python -m pytest tests/agentic/environments/test_read_only_integration.py -q
```

These tests exercise the Docker backend and the actual Kubernetes runtime helper
in disposable containers, including workspace-write denial, upload rejection,
helper restart, and recovery with a replacement environment. They do not deploy
to a live Kubernetes cluster or validate an operator's cluster configuration.

## Linting and formatting

FAITH uses [super-linter](https://github.com/super-linter/super-linter). The local
targets read settings from the repository's lint workflow. Both require Docker,
`yq`, and a clean working tree. Preserve existing work before using them:

```shell
make lint_all   # run all linters over the codebase
make format     # apply the configured automatic fixes
```

Python code is formatted with Black, isort, and Ruff; Markdown, YAML, JSON, and
TypeScript use their configured formatters. `format` uses changed-file selection
and can modify files; review the resulting diff. It is not equivalent to the
whole-codebase check performed by `lint_all`.

## License headers

FAITH is licensed under Apache-2.0. New Python and YAML source files require
license headers. The following Docker-based target annotates matching files
throughout the checkout, not only newly added files; review its diff:

```shell
make fix_license
```

## Reporting issues

Before opening an issue, search the
[issue list](https://github.com/cisco-foundation-ai/faith/issues) to check it
has not already been reported or fixed. Include a clear title, relevant
configuration and version information, and a minimal reproduction when possible.
Remove credentials and sensitive dataset content from logs before sharing them.

:::warning[Security issues]

**Do not report security bugs through GitHub issues.** Follow the process in
[`SECURITY.md`](https://github.com/cisco-foundation-ai/faith/blob/main/SECURITY.md)
instead.

:::

## Sending pull requests

- Check existing PRs and issues first, in case the change is already discussed
  or implemented.
- Include tests for any behavior you change.
- FAITH follows **semantic versioning** — breaking changes may be held for the
  next major release.

The team aims to review incoming issues and pull requests within 10 days.

## Other ways to help

- Triage open issues and help authors fill in missing detail.
- Review open pull requests and test patches against real usage.
- Add a missing test case, or improve these docs.

## Adding a benchmark

See [Creating new benchmarks](./benchmarks/creating-benchmarks.md) for benchmark
schemas, Q&A and agentic examples, and execution with `--custom-benchmarks`.
