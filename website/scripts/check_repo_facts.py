#!/usr/bin/env python3
# Copyright 2025 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

"""Verify the site's claims that require importing faith to resolve.

The cheap path-existence checks live in check-repo-facts.js. This script covers
the facts that only faith itself can answer:

  * CLI flags and their defaults, from the argparse parser in faith.cli.cli;
  * benchmark state / categories / primary metric, from each benchmark.yaml
    (whose `!from` tags need faith's extended YAML loader);
  * the BigQuery METRICS_SCHEMA column list.

The site lives at website/ inside the faith repo, so it checks itself against
the enclosing checkout by default:

    PYTHONPATH=src python3 website/scripts/check_repo_facts.py

Pass --repo to check against a different faith tree. Exits 1 and lists every
mismatch. Needs faith importable on PYTHONPATH; the BigQuery schema check is
read statically with `ast`, so google-cloud-bigquery need not be installed.
"""

from __future__ import annotations

import argparse
import ast
import re
import sys
from pathlib import Path
from typing import Any

SITE = Path(__file__).resolve().parent.parent
FAILURES: list[str] = []
CHECKED = 0

# Floor on how many FlagTable rows the CLI pages should yield. Guards against a
# parser that silently stops matching; raise it as flags are documented.
EXPECTED_DOC_FLAGS = 28


def claim(ok: bool, where: str, what: str) -> None:
    """Record one checked fact, and its failure message if it doesn't hold."""
    global CHECKED  # pylint: disable=global-statement
    CHECKED += 1
    if not ok:
        FAILURES.append(f"{where}: {what}")


def _render_default(value: object) -> set[str]:
    """The set of spellings that legitimately describe an argparse default.

    A default reaches us wrapped in whatever type the flag parses into, and the
    docs sensibly show the user-facing form. So `DefaultValue(3500)` may be
    written `3500`, an enum as its bare member name, and a single-element list
    of `SampleRatio(0, 1)` (the --n-shot default) as `0`.
    """
    text = re.sub(r"^DefaultValue\((.*)\)$", r"\1", str(value))
    spellings = {text, text.split(".")[-1]}

    if isinstance(value, (list, tuple)):
        inner: set[str] = set()
        for item in value:
            inner |= _render_default(item)
        spellings |= inner
        spellings.add(" ".join(str(i) for i in value))
    elif ratio := re.fullmatch(
        r"SampleRatio\(numerator=(\d+), denominator=(\d+)\)", text
    ):
        num, den = ratio.groups()
        spellings |= {num, f"{num}/{den}"} if den == "1" else {f"{num}/{den}"}

    return spellings


# ---------------------------------------------------------------------------
# 1. CLI flags: every --flag in a FlagTable must exist on the subcommand whose
#    page it appears on, and any documented `default:` must match argparse.
# ---------------------------------------------------------------------------
def _argparse_flags() -> dict[str, object]:
    """Every --flag faith's CLI accepts, mapped to its argparse default."""
    # cli.py calls parse_args() at import time, so give it a valid argv first.
    # That's also why this import can't sit at module scope.
    sys.argv = [
        "faith",
        "query",
        "--model-paths",
        "x",
        "--datastore-location",
        "/tmp/x",
    ]
    # pylint: disable=import-outside-toplevel
    from faith.cli import cli

    # Reading the parser's registered actions is the only way to recover flags
    # and defaults, and argparse exposes no public API for it.
    # pylint: disable=protected-access
    subs = next(
        a
        for a in cli._cli_parser._actions
        if isinstance(a, argparse._SubParsersAction)  # noqa: SLF001
    )

    # Flag -> default, unioned across subcommands (a flag means the same thing
    # wherever it appears; run-all/qed just re-expose query's flags).
    flags: dict[str, object] = {}
    for subparser in subs.choices.values():
        for action in subparser._actions:  # noqa: SLF001
            for opt in action.option_strings:
                if opt.startswith("--") and opt != "--help":
                    flags.setdefault(opt, action.default)
    return flags


def _flag_entries(text: str) -> list[tuple[str, str]]:
    """(flag, entry-text) per FlagTable row, spanning multiple lines.

    Rows are JS object literals in MDX, and most span several lines with nested
    `{...}` in `required:` and `annotations:`. Scanning for `flag: '--x'` and
    then walking braces outward keeps every row in scope; matching `{...}` with
    a regex would silently drop exactly the multi-line rows that carry the
    interesting flags.
    """
    entries: list[tuple[str, str]] = []
    for match in re.finditer(r"flag: '(--[a-z0-9-]+)'", text):
        start = text.rfind("{", 0, match.start())
        if start == -1:
            continue
        depth = 0
        for i in range(start, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    entries.append((match.group(1), text[start : i + 1]))
                    break
    return entries


def check_cli_flags(repo_root: Path) -> None:
    """Check documented flags and defaults against faith's argparse parser."""
    source = _argparse_flags()

    default_re = re.compile(r"default: '([^']*)'")

    # Flags whose argparse default is None because the real default is resolved
    # downstream: from the flag, then an env var, then a literal. The docs quite
    # reasonably document the *effective* default, so record the env var and
    # literal that produce it and exempt the flag from the argparse comparison.
    effective_defaults = {
        # summarize.py: bigquery_table or $FAITH_BIGQUERY_TABLE or "metrics"
        "--bigquery-table": (
            "metrics",
            "FAITH_BIGQUERY_TABLE",
            "faith/cli/subcmd/summarize.py",
        ),
    }

    seen_flags = 0
    for page in sorted((SITE / "docs" / "cli").glob("*.md")):
        for flag, entry_text in _flag_entries(page.read_text()):
            seen_flags += 1
            if flag not in source:
                claim(
                    False,
                    page.name,
                    f"documented flag {flag} does not exist in argparse",
                )
                continue
            doc_default = default_re.search(entry_text)
            if not doc_default:
                continue
            want = doc_default.group(1)

            # The exemption applies only while argparse really has no default.
            # Gating on that keeps a flag from staying exempt after it grows a
            # genuine argparse default that the docs then never get compared to.
            if flag in effective_defaults and source[flag] is None:
                literal, env_var, origin = effective_defaults[flag]
                # Don't compare against argparse — instead confirm the doc
                # matches the literal, and that the module still resolves the
                # flag that way, so this exemption can't go stale. Match the
                # whole `x or os.getenv(...) or "literal"` chain: a bare search
                # for the literal anywhere in the file would keep passing after
                # the fallback moved or changed shape.
                claim(
                    want == literal,
                    page.name,
                    f"{flag} effective default documented as '{want}' but {origin} uses '{literal}'",
                )
                fallback = re.compile(
                    rf"\bos\.getenv\(\s*\"{re.escape(env_var)}\"\s*\)"
                    rf"\s*or\s*\"{re.escape(literal)}\"",
                )
                claim(
                    fallback.search((repo_root / "src" / origin).read_text())
                    is not None,
                    origin,
                    f'no longer falls back to ${env_var} then "{literal}" '
                    f"for {flag}",
                )
                continue

            claim(
                want in _render_default(source[flag]),
                page.name,
                f"{flag} default documented as '{want}' but argparse says "
                f"'{source[flag]}'",
            )

    # A parser that matches nothing passes silently, which is the failure mode
    # that matters here: this check once read only single-line rows and skipped
    # 15 of 28 flags while still reporting success. Pin the count so a format
    # change surfaces as a failure instead of as quiet, invisible coverage loss.
    claim(
        seen_flags >= EXPECTED_DOC_FLAGS,
        "docs/cli/",
        f"only found {seen_flags} FlagTable entries, expected at least "
        f"{EXPECTED_DOC_FLAGS} — did the FlagTable format change?",
    )


# ---------------------------------------------------------------------------
# 2. Benchmark catalog: state, categories and primary metric per benchmark.
# ---------------------------------------------------------------------------
def _check_reasoning_group(catalog: str, configs: list[Any]) -> None:
    """Check the all-reasoning group row against the benchmarks it should name.

    The catalog states which benchmarks the reasoning group collects, and that
    only `enabled` benchmarks join a group. Read that one row and compare the
    two sets: searching the whole page for the name after the row (``
    `all-reasoning`.*`name` `` with re.S) would be satisfied by any later
    mention anywhere on the page, so a benchmark merely listed in its own
    section would pass.
    """
    enabled_reasoning = {
        c.metadata.name
        for c in configs
        if str(c.metadata.state) == "enabled"
        and "reasoning" in (c.metadata.categories or [])
    }
    row = re.search(r"^\| `all-reasoning` \| [^|]* \|([^|]*)\|", catalog, re.M)
    if row is None:
        claim(
            False, "benchmarks/catalog.md", "no `all-reasoning` row in the groups table"
        )
        return
    listed = set(re.findall(r"`([a-z0-9-]+)`", row.group(1)))
    claim(
        listed == enabled_reasoning,
        "benchmarks/catalog.md",
        f"the all-reasoning row lists {sorted(listed)} but the enabled "
        f"reasoning benchmarks are {sorted(enabled_reasoning)}",
    )


def _check_all_groups(
    catalog: str, configs: list[Any], real_names: set[str | None]
) -> None:
    """Check the `all` and `all-security` group rows against the benchmarks.

    Both rows state a fact that a routine change falsifies: adding one enabled
    benchmark makes the `all` count wrong, and adding a non-security one makes
    the `all-security` row's exception list wrong. Neither is visible from the
    site, so nothing but this would notice.

    The `all-security` row names its exceptions in prose ("all except
    `mmlu-all`, which is tagged `general`"), so its backticked tokens are
    filtered to those that are real benchmark names — otherwise the category
    name would count as an exception.

    `real_names` is `set[str | None]` because BenchmarkMetadata.name is
    optional; the caller builds it from every config. The None can't match a
    backticked token, so it drops out of the intersection on its own.
    """
    enabled = {c.metadata.name for c in configs if str(c.metadata.state) == "enabled"}

    row = re.search(r"^\| `all` \| [^|]* \| *([0-9,]+) benchmarks *\|", catalog, re.M)
    if row is None:
        claim(
            False,
            "benchmarks/catalog.md",
            "no `all` row with an 'N benchmarks' count in the groups table",
        )
    else:
        documented = int(row.group(1).replace(",", ""))
        claim(
            documented == len(enabled),
            "benchmarks/catalog.md",
            f"the all row says {documented} benchmarks but "
            f"{len(enabled)} are enabled",
        )

    non_security = {
        c.metadata.name
        for c in configs
        if str(c.metadata.state) == "enabled"
        and "security" not in (c.metadata.categories or [])
    }
    sec_row = re.search(r"^\| `all-security` \| [^|]* \|([^|]*)\|", catalog, re.M)
    if sec_row is None:
        claim(
            False, "benchmarks/catalog.md", "no `all-security` row in the groups table"
        )
        return
    excepted = set(re.findall(r"`([a-z0-9-]+)`", sec_row.group(1))) & real_names
    claim(
        excepted == non_security,
        "benchmarks/catalog.md",
        f"the all-security row excepts {sorted(excepted)} but the enabled "
        f"benchmarks not tagged security are {sorted(non_security)}",
    )


def _check_intro_suites(catalog: str) -> None:
    """Check the intro's suite table against the catalog's suite sections.

    The intro groups benchmarks into suites, a grouping that exists only in the
    docs — no benchmark.yaml names a suite — so the catalog's own `##` headings
    are the source of truth. The catalog is itself checked per-benchmark against
    benchmark.yaml above, which makes this transitive: a new benchmark needs a
    catalog section, and a new catalog section needs an intro row.

    A suite section is identified by structure — it contains the catalog's
    per-task table — not by a list of headings to ignore. The catalog has
    non-suite sections ("Benchmark groups", "Model backends", "Bringing your own
    benchmark") and will grow more; a blocklist would have to be edited every
    time one is added, and forgetting to would fail this check for no reason.

    The intro's table is located via its heading rather than by scanning the
    page, so a table added elsewhere can't satisfy this check.
    """
    intro = (SITE / "docs" / "intro.md").read_text()
    section = re.search(
        r"^## Supported benchmarks\n(.*?)(?=^## |\Z)", intro, re.M | re.S
    )
    if section is None:
        claim(False, "intro.md", "no '## Supported benchmarks' section")
        return

    listed = set(re.findall(r"^\| \[([^\]]+)\]\(", section.group(1), re.M))
    suites = {
        heading
        # `[^\n]+` for the heading, not `.+`: re.S makes `.` match newlines, so a
        # greedy `.+` swallows the rest of the page as one "heading".
        for heading, body in re.findall(
            r"^## ([^\n]+)\n(.*?)(?=^## |\Z)", catalog, re.M | re.S
        )
        if "| Task | Answer type |" in body
    }

    claim(
        bool(listed) and bool(suites),
        "intro.md",
        "found no linked suites in the intro table or no task tables in the "
        "catalog — did either page stop being structured this way?",
    )
    claim(
        listed == suites,
        "intro.md",
        f"the suite table lists {sorted(listed)} but catalog.md documents "
        f"{sorted(suites)}",
    )


def check_benchmarks(repo: Path) -> None:
    """Check the catalog's state, categories and primary metric per benchmark."""
    # Deferred so the argv fixup in check_cli_flags happens first.
    # pylint: disable=import-outside-toplevel
    from faith.benchmark.config import load_config_from_path

    catalog = (SITE / "docs" / "benchmarks" / "catalog.md").read_text()

    # Metric label in the catalog's "Primary metric" column -> metric path.
    metric_names = {
        "Accuracy": "accuracy.mean",
        "Subject-weighted accuracy": "subject_weighted_accuracy.mean",
        "Mean CVSS score": "mean_cvss_score.mean",
    }

    rows = dict(
        (m.group(1), m.group(2).strip())
        for m in re.finditer(
            r"^\| `([a-z0-9-]+)` \| [^|]+ \| [^|]+ \| ([^|]+) \|", catalog, re.M
        )
    )

    # A benchmark is a directory holding a benchmark.yaml, which is how
    # listing.py:_get_benchmark_paths enumerates them. Listing directories
    # instead is a weaker proxy that breaks on a shared-asset directory: it
    # would call load_config_from_path on one and hit that function's assert.
    # __benchmarks__ already holds shared *files* (base-template.yaml and
    # friends), so a directory of them is a matter of time.
    configs = [
        load_config_from_path(f.parent)
        for f in sorted((repo / "src/faith/__benchmarks__").glob("**/benchmark.yaml"))
    ]

    # The per-benchmark loop below only checks that every benchmark reaches the
    # catalog. Assert the reverse too, or a row for a benchmark that was renamed
    # or deleted stays on the page indefinitely, checked by nothing.
    real_names = {c.metadata.name for c in configs}
    for phantom in sorted(set(rows) - real_names):
        claim(
            False,
            "benchmarks/catalog.md",
            f"catalog lists '{phantom}', which is not a benchmark in "
            f"src/faith/__benchmarks__",
        )

    for config in configs:
        name = config.metadata.name
        state = str(config.metadata.state)

        # test_only benchmarks are intentionally absent from the catalog.
        if state == "test_only":
            claim(
                name not in rows,
                "benchmarks/catalog.md",
                f"'{name}' is test_only but is listed in the catalog",
            )
            continue

        if name not in rows:
            claim(
                False,
                "benchmarks/catalog.md",
                f"benchmark '{name}' is missing from the catalog",
            )
            continue

        want = config.output_processing.primary_metric
        got = metric_names.get(rows[name], rows[name])
        claim(
            got == want,
            "benchmarks/catalog.md",
            f"'{name}' primary metric documented as '{rows[name]}' but "
            f"benchmark.yaml says '{want}'",
        )

    _check_reasoning_group(catalog, configs)
    _check_all_groups(catalog, configs, real_names)
    _check_intro_suites(catalog)


# ---------------------------------------------------------------------------
# 3. BigQuery schema. Parsed statically so google-cloud-bigquery isn't needed.
# ---------------------------------------------------------------------------
def _parse_metrics_schema(source: str) -> list[tuple[str, str, str]]:
    """(name, type, mode) per METRICS_SCHEMA field, read without importing it.

    Static parsing keeps google-cloud-bigquery out of the checker's dependencies.
    """
    fields: list[tuple[str, str, str]] = []
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, ast.Assign):
            continue
        if not any(
            isinstance(t, ast.Name) and t.id == "METRICS_SCHEMA" for t in node.targets
        ):
            continue
        for call in node.value.elts:  # type: ignore[attr-defined]
            args = [a.value for a in call.args if isinstance(a, ast.Constant)]
            # SchemaField(name, type, ...) — anything else isn't a field we can
            # read, and skipping silently would drop it from the comparison, so
            # let the caller's `bool(fields)` claim and the name-set diffs below
            # notice. Two positional constants is the shape we understand.
            if len(args) < 2:
                continue
            mode = next(
                (
                    k.value.value
                    for k in call.keywords
                    if k.arg == "mode" and isinstance(k.value, ast.Constant)
                ),
                "NULLABLE",
            )
            fields.append((str(args[0]), str(args[1]), str(mode)))
    return fields


def check_bigquery_schema(repo: Path) -> None:
    """Check the documented column list against METRICS_SCHEMA in the source."""
    bq = repo / "src/faith/ingestion/storage/bigquery.py"
    fields = _parse_metrics_schema(bq.read_text())

    claim(
        bool(fields), "ingestion/storage/bigquery.py", "could not parse METRICS_SCHEMA"
    )

    doc = (SITE / "docs" / "bigquery" / "schema.md").read_text()
    documented = {
        m.group(1) for m in re.finditer(r"^\| `(\w+)` \| `?\w+`? \|", doc, re.M)
    }
    source_names = {f[0] for f in fields}

    for missing in sorted(source_names - documented):
        claim(
            False,
            "bigquery/schema.md",
            f"column '{missing}' exists in METRICS_SCHEMA but isn't documented",
        )
    for extra in sorted(documented - source_names):
        claim(
            False,
            "bigquery/schema.md",
            f"documented column '{extra}' is not in METRICS_SCHEMA",
        )

    for name, ftype, mode in fields:
        # Capture the cells loosely and validate them below. A pattern that
        # demanded `\w+` in both would simply not match a reworded Required
        # cell, and then `continue` would skip the type check along with it —
        # turning an editorial change into silent loss of coverage.
        row = re.search(rf"^\| `{re.escape(name)}` \|([^|]*)\|([^|]*)\|", doc, re.M)
        if not row:
            # The name sets above already report a column that's missing
            # outright, so reaching here means the row exists in some shape the
            # table parser can't read.
            claim(
                False,
                "bigquery/schema.md",
                f"could not parse the table row for column '{name}'",
            )
            continue
        doc_type = row.group(1).strip().strip("`")
        doc_required = row.group(2).strip()
        claim(
            doc_type.upper() == ftype.upper(),
            "bigquery/schema.md",
            f"column '{name}' type documented as '{doc_type}' but source says '{ftype}'",
        )
        # The docs render mode as a reader-friendly Required Yes/No column
        # rather than BigQuery's REQUIRED/NULLABLE spelling. An unrecognised
        # spelling fails rather than passing: `.get` returning None can't equal
        # any mode, and the message quotes the cell so the cause is obvious.
        want_required = {"Yes": "REQUIRED", "No": "NULLABLE"}.get(doc_required)
        claim(
            want_required == mode.upper(),
            "bigquery/schema.md",
            f"column '{name}' documented as Required={doc_required!r} but "
            f"source mode is '{mode}'",
        )


def main() -> int:
    """Run every check and report the tally. Returns the process exit code."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo",
        type=Path,
        default=SITE.parent,
        help="path to a faith checkout (default: the repo this site lives in)",
    )
    args = parser.parse_args()
    repo = args.repo.resolve()

    if not (repo / "src/faith").is_dir():
        print(f"No faith package at {repo}/src/faith.", file=sys.stderr)
        return 2

    check_cli_flags(repo)
    check_benchmarks(repo)
    check_bigquery_schema(repo)

    if FAILURES:
        print(
            f"✗ {len(FAILURES)} of {CHECKED} repo facts are stale:\n", file=sys.stderr
        )
        for f in FAILURES:
            print(f"  {f}", file=sys.stderr)
        print(f"\nRepo: {repo}", file=sys.stderr)
        return 1
    print(f"✓ {CHECKED} repo facts verified against {repo}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
