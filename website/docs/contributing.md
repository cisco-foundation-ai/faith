---
id: contributing
title: Contributing
sidebar_label: Contributing
sidebar_position: 6
---

# Contributing

Contributions to FAITH are welcome — bug reports, fixes, tests, new benchmarks,
and docs improvements. This page covers the development workflow; for the full
policy, see [`CONTRIBUTING.md`](https://github.com/cisco-foundation-ai/faith/blob/main/CONTRIBUTING.md)
and the [Code of Conduct](https://github.com/cisco-foundation-ai/faith/blob/main/CODE_OF_CONDUCT.md)
in the repository.

## Development setup

Install FAITH in **editable mode** so you can change the code and re-run without
reinstalling. Use the `[test]` extra, which pulls in everything plus pytest and
its plugins. The linters aren't part of it — they run in Docker, via the targets
[below](#linting-and-formatting):

```shell
uv venv .venv && source .venv/bin/activate
SETUPTOOLS_SCM_PRETEND_VERSION=0.0.0 uv pip install -e '.[test]'
```

The `SETUPTOOLS_SCM_PRETEND_VERSION` is needed when setuptools-scm can't infer a
version from your checkout. See [Installation](./getting-started/installation.md)
for details and other extras.

## Running tests

Tests run through the `Makefile`:

```shell
make test       # runs the standard suite with coverage
make test_all   # also runs slow tests (--runslow)
```

Both invoke `pytest` against `tests/` with coverage reporting. We expect new
pull requests to **include tests for any affected behavior**.

## Linting and formatting

FAITH lints and formats through [super-linter](https://github.com/super-linter/super-linter)
so local checks match CI exactly. Both targets require **Docker** and **`yq`**,
and a clean working tree (commit or stash first):

```shell
make lint_all   # run all linters over the codebase
make format     # auto-fix formatting (Black, isort, Ruff, Prettier, and more)
```

Python code is formatted with Black, isort, and Ruff; Markdown, YAML, JSON, and
TypeScript are handled by their respective formatters.

## License headers

FAITH is Apache-2.0 and every source file carries a license header. If you add
new `.py` or `.yaml`/`.yml` files, apply the headers with:

```shell
make fix_license
```

## Reporting issues

Before opening an issue, search the
[issue list](https://github.com/cisco-foundation-ai/faith/issues) to check it
isn't already reported or fixed. Include a clear title and description, as much
relevant context as possible, and a test case if you can.

:::warning Security issues
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

Want to contribute a new benchmark rather than change the framework? See
[Creating New Benchmarks](./benchmarks/creating-benchmarks.md), which walks
through defining a benchmark and running it with `--custom-benchmarks`.
