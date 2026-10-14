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
  * the BigQuery METRICS_SCHEMA column list;
  * the landing page's demo command, benchmark names, engines and flags, which
    live in JSX and so are invisible to the markdown checks.

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
import shlex
import sys
import tomllib
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
def _cli_subparsers() -> dict[str, argparse.ArgumentParser]:
    """faith's subcommand parsers, keyed by the name typed on the command line."""
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

    # Reading the parser's registered actions is the only way to recover
    # subcommands, flags and defaults, and argparse exposes no public API for it.
    # pylint: disable=protected-access
    subs = next(
        a
        for a in cli._cli_parser._actions
        if isinstance(a, argparse._SubParsersAction)  # noqa: SLF001
    )
    return dict(subs.choices)


def _parser_flags(parser: argparse.ArgumentParser) -> dict[str, object]:
    """The --flags one subcommand accepts, mapped to their argparse defaults."""
    # pylint: disable=protected-access
    return {
        opt: action.default
        for action in parser._actions  # noqa: SLF001
        for opt in action.option_strings
        if opt.startswith("--") and opt != "--help"
    }


def _argparse_flags() -> dict[str, object]:
    """Every --flag faith's CLI accepts, mapped to its argparse default."""
    # Flag -> default, unioned across subcommands (a flag means the same thing
    # wherever it appears; run-all/qed just re-expose query's flags).
    flags: dict[str, object] = {}
    for parser in _cli_subparsers().values():
        for opt, default in _parser_flags(parser).items():
            flags.setdefault(opt, default)
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
def _check_reasoning_group(groups: str, configs: list[Any]) -> None:
    """Check the all-reasoning group row against the benchmarks it should name.

    groups.md states which benchmarks the reasoning group collects, and that
    only `enabled` benchmarks join a group. Read that one row and compare the
    two sets: searching the whole page for the name after the row (``
    `all-reasoning`.*`name` `` with re.S) would be satisfied by any later
    mention anywhere on the page, so a benchmark merely named elsewhere on it
    would pass.
    """
    enabled_reasoning = {
        c.metadata.name
        for c in configs
        if str(c.metadata.state) == "enabled"
        and "reasoning" in (c.metadata.categories or [])
    }
    row = re.search(r"^\| `all-reasoning` \| [^|]* \|([^|]*)\|", groups, re.M)
    if row is None:
        claim(
            False, "benchmarks/groups.md", "no `all-reasoning` row in the groups table"
        )
        return
    listed = set(re.findall(r"`([a-z0-9-]+)`", row.group(1)))
    claim(
        listed == enabled_reasoning,
        "benchmarks/groups.md",
        f"the all-reasoning row lists {sorted(listed)} but the enabled "
        f"reasoning benchmarks are {sorted(enabled_reasoning)}",
    )


def _check_all_groups(
    groups: str, configs: list[Any], real_names: set[str | None]
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

    Both sets are enabled *and* non-agentic, matching the exclude_agentic=True
    that group expansion passes. Counting every enabled benchmark is what this
    check did until an agentic one existed, at which point it started agreeing
    with a page that overstated `all` — a check that ratifies the error it was
    written to catch.
    """
    enabled = {
        c.metadata.name
        for c in configs
        if str(c.metadata.state) == "enabled" and c.agentic_config is None
    }

    row = re.search(r"^\| `all` \| [^|]* \| *([0-9,]+) benchmarks *\|", groups, re.M)
    if row is None:
        claim(
            False,
            "benchmarks/groups.md",
            "no `all` row with an 'N benchmarks' count in the groups table",
        )
    else:
        documented = int(row.group(1).replace(",", ""))
        claim(
            documented == len(enabled),
            "benchmarks/groups.md",
            f"the all row says {documented} benchmarks but "
            f"{len(enabled)} enabled non-agentic benchmarks exist",
        )

    non_security = {
        c.metadata.name
        for c in configs
        if str(c.metadata.state) == "enabled"
        and c.agentic_config is None
        and "security" not in (c.metadata.categories or [])
    }
    sec_row = re.search(r"^\| `all-security` \| [^|]* \|([^|]*)\|", groups, re.M)
    if sec_row is None:
        claim(
            False, "benchmarks/groups.md", "no `all-security` row in the groups table"
        )
        return
    excepted = set(re.findall(r"`([a-z0-9-]+)`", sec_row.group(1))) & real_names
    claim(
        excepted == non_security,
        "benchmarks/groups.md",
        f"the all-security row excepts {sorted(excepted)} but the enabled "
        f"benchmarks not tagged security are {sorted(non_security)}",
    )


def _suite_name(heading: str) -> str:
    """The suite name from a catalog `###` heading, minus its trailing markup.

    Each suite heading carries a `<SourceLinks .../>` element for its canonical
    sources and an explicit `{#anchor}`, because MDX would otherwise derive the
    anchor from the whole heading — including the element — and break the
    `#vulnerability-localization`-style links that point at these sections from
    elsewhere.
    """
    heading = re.sub(r"\s*<SourceLinks\b.*?/>", "", heading, flags=re.S)
    heading = re.sub(r"\s*\\?\{#[a-z0-9-]+\}", "", heading)
    return heading.strip()


def _check_metrics_page(configs: list[Any], metric_names: dict[str, str]) -> None:
    """Check the metrics reference against benchmark.yaml and against the catalog.

    Two claims, both prompted by the metrics page silently falling behind when
    agentic benchmarks landed: its per-benchmark table kept a row per Q&A
    benchmark and grew no rows for the two new ones.

    First, every row that names a single benchmark must quote that benchmark's
    real `primary_metric`. Rows whose first column isn't one backticked benchmark
    name — "Most MCQA + short-answer" covers the long tail — are skipped rather
    than guessed at.

    Second, every metric label the catalog uses must have a definition section
    here, matched by prefix so "Accuracy" satisfies "Accuracy (strict)". This is
    the direction that actually broke: a benchmark arrives with a new metric, gets
    a catalog row, and nothing notices that no one ever wrote down what the metric
    means.
    """
    metrics = (SITE / "docs" / "analytics" / "metrics.md").read_text()

    rows = dict(re.findall(r"^\| `([a-z0-9-]+)` \| `([a-z0-9_.]+)` \|", metrics, re.M))
    by_name = {c.metadata.name: c for c in configs}
    for name, documented in sorted(rows.items()):
        if name not in by_name:
            claim(
                False,
                "analytics/metrics.md",
                f"the metrics table lists '{name}', which is not a benchmark in "
                f"src/faith/__benchmarks__",
            )
            continue
        want = by_name[name].output_processing.primary_metric
        claim(
            documented == want,
            "analytics/metrics.md",
            f"'{name}' primary metric documented as '{documented}' but "
            f"benchmark.yaml says '{want}'",
        )

    headings = re.findall(r"^### (.+)$", metrics, re.M)
    for label in sorted(metric_names):
        claim(
            any(h.startswith(label) for h in headings),
            "analytics/metrics.md",
            f"the catalog reports '{label}' as a primary metric but no "
            f"'## Metric definitions' section defines it",
        )


def _check_stopped_reasons() -> None:
    """Check troubleshooting's stopped_reason table against the StoppedReason enum.

    The table is the page's whole diagnostic: a reader matches the value they got
    against a row. A member added to the enum and not to the table reads as an
    undocumented value, and a row naming a member that no longer exists sends the
    reader looking for something they will never see. Neither shows up in a build,
    since both are prose.

    Compare the serialized lowercase values in the dedicated table.
    `max_calls` / `max_turns` share a row, so the row is split on `/`.
    """
    # pylint: disable=import-outside-toplevel
    from faith._types.record.agent import StoppedReason

    page = (SITE / "docs" / "troubleshooting.md").read_text()
    section = re.search(r"^\| `stopped_reason` \|.*\n((?:\|.*\n)+)", page, re.M)
    table = re.findall(r"^\| `([a-z_ /`]+)` \|", section[1] if section else "", re.M)
    documented = {
        member.strip().strip("`") for row in table for member in row.split("/")
    }
    real = {str(sr) for sr in StoppedReason}

    for member in sorted(real):
        claim(
            member in documented,
            "troubleshooting.md",
            f"StoppedReason.{member.upper()} exists but no stopped_reason row documents it",
        )
    for member in sorted(documented - real):
        claim(
            False,
            "troubleshooting.md",
            f"the stopped_reason table documents '{member}', which is not a "
            f"StoppedReason member",
        )


# Catalog groupings also provide the two benchmark-kind entry points in the
# introduction. Keep their coverage checked without duplicating the suite list.
CATALOG_KINDS = {"Q&A benchmarks": "qa", "Agentic benchmarks": "agentic"}


def _catalog_suites(catalog: str) -> dict[str, str]:
    """Map each catalog suite to the kind of benchmark it holds.

    A suite section is identified by structure — it contains the catalog's
    per-benchmark table — not by a list of headings to ignore. The catalog carries
    prose sections alongside its tables and will grow more; a blocklist would
    have to be edited every time one is added, and forgetting to would fail this
    check for no reason.

    Suites sit at `###` because the catalog brackets them under `##` groupings by
    how a benchmark runs, so the enclosing `##` heading is what says whether a
    suite is agentic. A suite under some third grouping fails loudly rather than
    being dropped: a silently skipped suite is exactly the drift this catches.
    """
    suites: dict[str, str] = {}
    # Empty until the first `##`, which no suite can sit under — a table before any
    # grouping fails the claim below rather than being tagged from a stale heading.
    group = ""
    # `[^\n]+` for the heading, not `.+`: re.S makes `.` match newlines, so a
    # greedy `.+` swallows the rest of the page as one "heading".
    for hashes, heading, body in re.findall(
        r"^(#{2,3}) ([^\n]+)\n(.*?)(?=^#{2,3} |\Z)", catalog, re.M | re.S
    ):
        if hashes == "##":
            group = heading.strip()
        elif "| Benchmark | Size | Primary metric |" in body:
            name = _suite_name(heading)
            claim(
                group in CATALOG_KINDS,
                "benchmarks/catalog.md",
                f"suite '{name}' sits under '## {group}', which is not one of "
                f"{sorted(CATALOG_KINDS)}",
            )
            suites[name] = CATALOG_KINDS.get(group, "?")
    return suites


def _check_intro_benchmark_links(catalog: str) -> None:
    """Keep both benchmark kinds discoverable from the shortened introduction.

    The catalog is checked per benchmark against benchmark.yaml. The intro
    now links to that inventory instead of maintaining a second suite list.
    """
    intro = (SITE / "docs" / "intro.md").read_text()
    suites = _catalog_suites(catalog)
    claim(
        set(suites.values()) == set(CATALOG_KINDS.values()),
        "benchmarks/catalog.md",
        "the catalog must document both Q&A and agentic suites",
    )
    for anchor in ("qa-benchmarks", "agentic-benchmarks"):
        claim(
            f"(./benchmarks/catalog.md#{anchor})" in intro,
            "intro.md",
            f"missing direct link to catalog.md#{anchor}",
        )


def check_benchmarks(repo: Path) -> None:
    """Check the catalog's state, categories and primary metric per benchmark."""
    # Deferred so the argv fixup in check_cli_flags happens first.
    # pylint: disable=import-outside-toplevel
    from faith.benchmark.config import load_config_from_path

    catalog = (SITE / "docs" / "benchmarks" / "catalog.md").read_text()
    groups = (SITE / "docs" / "benchmarks" / "groups.md").read_text()

    # Metric label in the catalog's "Primary metric" column -> metric path.
    metric_names = {
        "Accuracy": "accuracy.mean",
        "Subject-weighted accuracy": "subject_weighted_accuracy.mean",
        "Mean CVSS score": "mean_cvss_score.mean",
        "File-set F1": "mean_file_f1.mean",
        "True-negative rate": "mean_true_negative_rate.mean",
    }

    rows = dict(
        (m.group(1), m.group(2).strip())
        for m in re.finditer(
            r"^\| `([a-z0-9-]+)` \| [^|]+ \| ([^|]+) \|", catalog, re.M
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

    _check_reasoning_group(groups, configs)
    _check_all_groups(groups, configs, real_names)
    _check_intro_benchmark_links(catalog)
    _check_metrics_page(configs, metric_names)
    _check_stopped_reasons()


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

    doc = (SITE / "docs" / "analytics" / "bigquery" / "schema.md").read_text()
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


# ---------------------------------------------------------------------------
# 4. The architecture page's <FileTree>, checked path by path.
# ---------------------------------------------------------------------------
def _tree_node_error(
    name: str, kind: str, dirs: set[str], files: set[str]
) -> str | None:
    """Why a documented tree node is wrong, or None if it checks out.

    A child's name in the JSX is bare (`vllm.py`), not a full path, so a node is
    real if some path in the package is, or ends with, that name.
    """

    def in_pool(pool: set[str]) -> bool:
        bare = name.rstrip("/")
        return any(p == bare or p.endswith("/" + bare) for p in pool)

    want, other = (dirs, files) if kind == "dir" else (files, dirs)
    if in_pool(want):
        return None
    # Distinguish a wrong `kind` from a path that is simply gone. Reporting "does
    # not exist" for a path that does exist as the other kind would send a reader
    # hunting for a deletion that never happened.
    if in_pool(other):
        actually = "directory" if kind == "file" else "file"
        return f"the code tree lists '{name}' as a {kind}, but it is a {actually}"
    return (
        f"the code tree lists {kind} '{name}', which does not exist under "
        f"src/faith/ (is it renamed or deleted?)"
    )


def check_code_tree(repo: Path) -> None:
    """Check that every path in the architecture page's code tree exists.

    The tree is hand-maintained JSX listing ~50 files and directories under
    `src/faith/`, and nothing verified any of them: a renamed or deleted module
    stayed on the page indefinitely. The tree is deliberately partial, so this
    checks one direction only — every documented path must exist, but a real path
    may go undocumented.

    `kind` is checked too, so a file that becomes a package (or the reverse) is
    caught rather than passing on the name alone.

    The note under the tree quantifies how partial it is ("31 of the package's 160
    modules"), and both numbers drift the moment anyone adds a module, so they are
    checked against the tree and the package rather than trusted.
    """
    page = SITE / "docs" / "reference" / "architecture.md"
    text = page.read_text()

    tree = re.search(r"<FileTree\b(.*?)^/>", text, re.M | re.S)
    if tree is None:
        claim(False, "reference/architecture.md", "no <FileTree> element found")
        return

    # Each node is `{name: 'x', kind: 'file'|'dir', ...}`. Nesting doesn't matter
    # here: the note already carries the parent, so names are matched against the
    # whole subtree rather than reconstructing each node's path from the JSX.
    nodes = re.findall(r"\{name:\s*'([^']+)',\s*kind:\s*'(file|dir)'", tree.group(1))
    claim(
        len(nodes) >= 40,
        "reference/architecture.md",
        f"parsed only {len(nodes)} nodes from the code tree — did its shape change?",
    )

    # Build the candidate sets once from the real package, then classify each node.
    root = repo / "src" / "faith"
    paths = [(p.relative_to(root).as_posix(), p.is_dir()) for p in root.rglob("*")]
    real_dirs = {rel for rel, is_dir in paths if is_dir}
    real_files = {rel for rel, is_dir in paths if not is_dir}

    for name, kind in nodes:
        problem = _tree_node_error(name, kind, real_dirs, real_files)
        claim(problem is None, "reference/architecture.md", problem or "")

    # The note's two counts. Modules mean importable .py files, excluding the
    # __init__.py that only marks a package — that is what a reader counts when they
    # look at the tree, and counting them would inflate the total by ~20.
    #
    # `_version.py` is excluded because it is not in the repository: the build
    # backend writes it into the package on install (`write_to` in pyproject.toml)
    # and it is gitignored. Counting it made this check depend on whether faith
    # happened to be installed from this tree — passing in a clean checkout and
    # failing in CI, which installs the package before running.
    listed = sum(1 for _, kind in nodes if kind == "file")
    real = sum(
        1
        for rel in real_files
        if rel.endswith(".py")
        and not rel.endswith("__init__.py")
        and rel != "_version.py"
    )
    note = re.search(r"partial tree — (\d+) of the package's (\d+) modules", text)
    claim(
        note is not None,
        "reference/architecture.md",
        "the note under the code tree no longer states how partial the tree is",
    )
    if note is not None:
        claim(
            int(note.group(1)) == listed,
            "reference/architecture.md",
            f"the note says the tree lists {note.group(1)} modules, but it lists "
            f"{listed}",
        )
        claim(
            int(note.group(2)) == real,
            "reference/architecture.md",
            f"the note says the package has {note.group(2)} modules, but it has "
            f"{real}",
        )


def check_flow_refs(repo: Path) -> None:
    """Check that every `ref` in the flow diagram names a real path.

    `conceptData.js` labels each step of the run-flow diagram with the source file
    or directory it lives in, and its own comment promises the shapes differ
    because the code paths do. Nothing enforced that the paths are real. A ref may
    be a file or a directory (a trailing slash marks the latter), so both are
    accepted here — unlike the code tree, the diagram doesn't assert which.

    Some refs carry a `:line` suffix, which points the reader at a specific line
    and is not part of the path, so it is stripped before the check.
    """
    data = (SITE / "src" / "components" / "FlowConcepts" / "conceptData.js").read_text()
    refs = sorted(
        {
            re.sub(r":\d+$", "", m)
            for m in re.findall(r"^\s*ref:\s*\"([^\"]+)\"", data, re.M)
        }
    )

    # A scraped check that finds nothing passes for the wrong reason, so the empty
    # case is a failure rather than a no-op. Not a minimum count: how many steps the
    # diagram has is a design choice, and pinning a number here would mean editing
    # this file every time someone adds or removes a card.
    claim(
        bool(refs),
        "FlowConcepts/conceptData.js",
        "no 'ref:' values matched in the flow data — did its shape change?",
    )
    for ref in refs:
        claim(
            (repo / "src" / "faith" / ref).exists(),
            "FlowConcepts/conceptData.js",
            f"the flow diagram labels a step '{ref}', which does not exist under "
            f"src/faith/",
        )


def _dataclass_fields(module: ast.Module, name: str) -> dict[str, Any]:
    """Field name → literal default for one dataclass, read without importing it.

    A field's default may be written directly (`max_turns: int = 100`) or wrapped
    in `field(default=..., metadata=...)`, which is how the optional ones in these
    configs are spelled. Both are read; a `default_factory` has no literal to
    report and comes back as None, same as an absent default.
    """
    cls = next(
        (n for n in module.body if isinstance(n, ast.ClassDef) and n.name == name),
        None,
    )
    if cls is None:
        return {}
    fields: dict[str, Any] = {}
    for node in cls.body:
        if not isinstance(node, ast.AnnAssign) or not isinstance(node.target, ast.Name):
            continue
        default = None
        if isinstance(node.value, ast.Constant):
            default = node.value.value
        elif isinstance(node.value, ast.Call):
            default = next(
                (
                    k.value.value
                    for k in node.value.keywords
                    if k.arg == "default" and isinstance(k.value, ast.Constant)
                ),
                None,
            )
        fields[node.target.id] = default
    return fields


def check_agentic_config(repo: Path) -> None:
    """Check the `agentic_config` reference against the dataclasses behind it.

    `creating-benchmarks.md` documents the configuration fields and limit
    defaults. Compare those claims with the dataclass definitions.

    The sub-section names are checked as a set, so adding a field to AgenticConfig
    fails here until the page documents it — which is the direction that matters.
    A field the page omits is a gap; prose the code doesn't have is a lie.

    Read statically with `ast`, as the BigQuery schema is, rather than by importing
    the dataclass: importing resolves against whatever faith is on PYTHONPATH, which
    silently ignores --repo and would check the page against the wrong tree.
    """
    agentic = repo / "src/faith/_types/config/agentic.py"
    module = ast.parse(agentic.read_text())
    sections = _dataclass_fields(module, "AgenticConfig")
    defaults = _dataclass_fields(module, "AgenticLimitsConfig")

    claim(
        bool(sections) and bool(defaults),
        "_types/config/agentic.py",
        "could not parse AgenticConfig / AgenticLimitsConfig",
    )

    text = (SITE / "docs" / "benchmarks" / "creating-benchmarks.md").read_text()

    documented = set(re.findall(r"^\| `(\w+)` \| ", text, re.M))
    for name in sections:
        claim(
            name in documented,
            "benchmarks/creating-benchmarks.md",
            f"agentic_config has a '{name}' section that the page never names",
        )

    for name, default in defaults.items():
        # None is documented as prose, not as a literal: "unset" is what an author
        # writes (i.e. nothing), and `None` is a Python spelling of it.
        shown = "unset" if default is None else f"`{default}`"
        row = re.search(rf"^\| `{name}` \| ([^|]+?) \|", text, re.M)
        claim(
            row is not None,
            "benchmarks/creating-benchmarks.md",
            f"the limits table has no row for '{name}'",
        )
        if row is not None:
            claim(
                row.group(1).strip() == shown,
                "benchmarks/creating-benchmarks.md",
                f"the limits table gives '{name}' as {row.group(1).strip()}, but "
                f"its default is {shown}",
            )


def check_agentic_runtime_reference(repo: Path) -> None:
    """Check the registered tool list and the independent probe/task budgets."""
    module = ast.parse((repo / "src/faith/agentic/registries.py").read_text())
    registry = next(
        (
            node
            for node in module.body
            if isinstance(node, ast.ClassDef) and node.name == "AgenticToolType"
        ),
        None,
    )
    registered = {
        target.id.lower()
        for node in (registry.body if registry is not None else [])
        if isinstance(node, ast.Assign)
        for target in node.targets
        if isinstance(target, ast.Name) and target.id.isupper()
    }
    claim(bool(registered), "agentic/registries.py", "no registered tools found")

    authoring = (SITE / "docs/benchmarks/creating-benchmarks.md").read_text()
    tool_section = re.search(
        r"^#### Registered tools\n(.*?)^### Limits\n", authoring, re.M | re.S
    )
    documented = set(
        re.findall(r"^\| `(\w+)` \|", tool_section[1] if tool_section else "", re.M)
    )
    claim(
        documented == registered,
        "benchmarks/creating-benchmarks.md",
        f"registered tool table differs: missing={sorted(registered - documented)}, "
        f"unregistered={sorted(documented - registered)}",
    )

    query = ast.parse((repo / "src/faith/cli/subcmd/query.py").read_text())
    probe_budget = next(
        (
            node.value.value
            for node in query.body
            if isinstance(node, ast.Assign)
            and isinstance(node.value, ast.Constant)
            and any(
                isinstance(target, ast.Name) and target.id == "_TOOL_PROBE_MAX_TOKENS"
                for target in node.targets
            )
        ),
        None,
    )
    claim(
        isinstance(probe_budget, int),
        "cli/subcmd/query.py",
        "could not read the probe completion-token budget",
    )
    serving = (SITE / "docs/configuration/serving-agentic-models.md").read_text()
    probe_claim = re.search(
        r"probe sets `max_completion_tokens` to `([\d,]+)`", serving
    )
    claim(
        probe_claim is not None
        and int(probe_claim[1].replace(",", "")) == probe_budget,
        "configuration/serving-agentic-models.md",
        f"probe budget must match _TOOL_PROBE_MAX_TOKENS={probe_budget!r}",
    )
    generation = ast.parse((repo / "src/faith/_types/model/generation.py").read_text())
    default_budget = _dataclass_fields(generation, "GenParams").get(
        "max_completion_tokens"
    )
    task_claim = re.search(r"`max_completion_tokens` defaults to `([\d,]+)`", serving)
    claim(
        task_claim is not None
        and int(task_claim[1].replace(",", "")) == default_budget,
        "configuration/serving-agentic-models.md",
        f"task default must match GenParams.max_completion_tokens={default_budget}",
    )


def check_group_exclusions(repo: Path) -> None:
    """Check the documented group names, and that groups still exclude agentic.

    Two facts the states table on creating-benchmarks.md now leans on. The page
    says a group covers the question-answering benchmarks only, which is true
    because choices_to_benchmarks passes exclude_agentic=True; drop that argument
    and the page silently starts overstating what `--benchmarks all` will run —
    the expensive direction, since an agentic task costs a container and many
    model calls.

    Read statically with `ast`, as the other source-side checks here are, rather
    than by importing faith.benchmark.listing: importing resolves against whatever
    faith is on PYTHONPATH, which ignores --repo and would check the page against
    the wrong tree.
    """
    listing = repo / "src/faith/benchmark/listing.py"
    module = ast.parse(listing.read_text())

    groups = next(
        (
            n.value
            for n in module.body
            if isinstance(n, ast.AnnAssign)
            and isinstance(n.target, ast.Name)
            and n.target.id == "_BENCHMARK_GROUPS"
        ),
        None,
    )
    real_groups = (
        {
            k.value
            for k in groups.keys
            # A group name is a string key; anything else isn't one.
            if isinstance(k, ast.Constant) and isinstance(k.value, str)
        }
        if isinstance(groups, ast.Dict)
        else set()
    )
    claim(
        bool(real_groups),
        "benchmark/listing.py",
        "could not read the group names out of _BENCHMARK_GROUPS",
    )

    # The exclusion lives on the group expansion specifically: benchmark_choices
    # and find_benchmarks filter on state alone, which is why the page qualifies
    # only the group column.
    expansion = next(
        (
            n
            for n in module.body
            if isinstance(n, ast.FunctionDef) and n.name == "choices_to_benchmarks"
        ),
        None,
    )
    excludes = expansion is not None and any(
        isinstance(n, ast.keyword) and n.arg == "exclude_agentic"
        # An explicit `exclude_agentic=False` would be a lie dressed as a citation.
        and isinstance(n.value, ast.Constant) and n.value.value is True
        for n in ast.walk(expansion)
    )
    claim(
        excludes,
        "benchmarks/groups.md",
        "the page says no group expands to an agentic benchmark, but "
        "choices_to_benchmarks no longer passes exclude_agentic=True",
    )

    text = (SITE / "docs" / "benchmarks" / "creating-benchmarks.md").read_text()
    for name in sorted(real_groups):
        claim(
            f"`{name}`" in text,
            "benchmarks/creating-benchmarks.md",
            f"'{name}' is a benchmark group the page never names",
        )


def check_extras(repo: Path) -> None:
    """Check the extras table against pyproject's optional-dependencies.

    Both directions, unlike most checks here. An undocumented extra is invisible
    to a reader who needs it — which is how `docker` and `kubernetes` sat unlisted
    while the table's own `all` row claimed to cover everything above it — and a
    documented one that no longer exists sends them to a failing install.
    """
    with (repo / "pyproject.toml").open("rb") as fh:
        pyproject = tomllib.load(fh)
    real = set(pyproject["project"]["optional-dependencies"])

    page = (SITE / "docs" / "getting-started" / "installation.md").read_text()
    table = page.split("### Available extras", 1)[-1].split("###", 1)[0]
    documented = set(re.findall(r"^\| `([a-z][a-z-]*)` \|", table, re.M))

    for name in sorted(real - documented):
        claim(
            False,
            "getting-started/installation.md",
            f"pyproject defines an extra '{name}' that the extras table omits",
        )
    for name in sorted(documented - real):
        claim(
            False,
            "getting-started/installation.md",
            f"the extras table lists '{name}', which pyproject does not define",
        )
    # The loop above records nothing when the two sets agree, and a check that can
    # pass by finding nothing is the failure mode this file has been bitten by.
    claim(
        bool(documented),
        "getting-started/installation.md",
        "no extras matched in the extras table — did its shape change?",
    )


# The landing page keeps its every claim in one module: the hero command, the phase
# cards, the capability cards and the backend grid are all plain data in here.
LANDING = SITE / "src" / "components" / "landing" / "shared.js"


def _published_benchmarks(repo: Path) -> set[str]:
    """Every benchmark a reader can actually run, by name.

    `test_only` benchmarks are excluded: they exist to exercise faith's own tests,
    so naming one on a public page points a reader at a benchmark that isn't for
    them. Directories are found the way `check_benchmarks` finds them — see the
    comment there for why the glob is on `benchmark.yaml` and not on directories.
    """
    # pylint: disable=import-outside-toplevel
    from faith.benchmark.config import load_config_from_path

    configs = [
        load_config_from_path(f.parent)
        for f in sorted((repo / "src/faith/__benchmarks__").glob("**/benchmark.yaml"))
    ]
    return {
        c.metadata.name
        for c in configs
        if c.metadata.name is not None and str(c.metadata.state) != "test_only"
    }


def _check_demo_command(text: str, published: set[str], engines: set[str]) -> None:
    """Check the hero terminal's command against faith's own parser.

    It is the command a newcomer is likeliest to copy, and in five lines it names a
    subcommand, four flags, two benchmarks and an engine — every one of which the
    repo is free to rename without the page noticing.
    """
    where = "landing/shared.js"
    literal = re.search(r"export const CMD_TEXT = `(.*?)`;", text, re.S)
    if literal is None:
        claim(False, where, "no CMD_TEXT template literal to check")
        return

    # Line continuations are shell punctuation, not arguments. They reach us as two
    # characters, since a template literal has to escape the backslash.
    tokens = [t for t in literal.group(1).split() if t.strip("\\")]
    claim(
        tokens[:1] == ["faith"],
        where,
        f"the demo command starts with {tokens[:1]} rather than `faith`",
    )
    subcommand = tokens[1] if len(tokens) > 1 else ""
    parser = _cli_subparsers().get(subcommand)
    if parser is None:
        claim(
            False,
            where,
            f"the demo command runs `faith {subcommand}`, which is not a subcommand",
        )
        return

    # Flag -> the values written after it, up to the next flag.
    accepted = _parser_flags(parser)
    used: dict[str, list[str]] = {}
    current = ""
    for token in tokens[2:]:
        if token.startswith("--"):
            current = token
            used[current] = []
            claim(
                token in accepted,
                where,
                f"the demo command passes `{token}`, which `faith {subcommand}` "
                f"does not accept",
            )
        elif current:
            used[current].append(token)
    claim(
        bool(used),
        where,
        "no flags parsed out of the demo command — did its shape change?",
    )

    for name in used.get("--benchmarks", []):
        claim(
            name in published,
            where,
            f"the demo command runs '{name}', which is not a published benchmark "
            f"in src/faith/__benchmarks__",
        )
    for engine in used.get("--model-engine", []):
        claim(
            engine.lower() in engines,
            where,
            f"the demo command passes --model-engine {engine}, which is not a "
            f"ModelEngine member",
        )


def _check_landing_benchmarks(text: str, published: set[str]) -> None:
    """Check the benchmark names in the hero evaluation results.

    The scores are labeled as example values and are not checked against measured
    results. Benchmark names must still resolve to published benchmarks.
    """
    where = "landing/shared.js"
    named = re.findall(r'bench: "([^"]+)"', text)
    claim(
        bool(named),
        where,
        "no result rows matched — did EVALUATION_RESULTS change shape?",
    )
    for name in sorted(set(named)):
        claim(
            name in published,
            where,
            f"the results table names '{name}', which is not a published " f"benchmark",
        )


def _check_landing_backends(text: str, engines: set[str]) -> None:
    """Check the backend grid, and the phase card that lists the same engines.

    Set equality rather than containment, in both places: an engine added to the
    enum and not to the page leaves the page quietly incomplete, and a subset check
    is blind to exactly that.
    """
    where = "landing/shared.js"
    grid = re.findall(r'name: "([^"]+)",\s*kind:', text)
    claim(bool(grid), where, "no BACKENDS rows matched — did the grid change shape?")
    claim(
        {n.lower() for n in grid} == engines,
        where,
        f"the backend grid lists {sorted(grid)}, but ModelEngine has "
        f"{sorted(engines)}",
    )

    # The Query phase card repeats the same four names in prose, where a reader
    # meets them before reaching the grid.
    sentence = re.search(r'Q&A supports ([^."]+)\.', text)
    if sentence is None:
        claim(False, where, "no 'Q&A supports …' sentence found in the phase cards")
        return
    listed = [
        w for w in re.findall(r"[A-Za-z][A-Za-z0-9]*", sentence.group(1)) if w != "and"
    ]
    claim(
        {w.lower() for w in listed} == engines,
        where,
        f"the Query card says FAITH works with {listed}, but ModelEngine has "
        f"{sorted(engines)}",
    )


def _check_landing_flags(repo: Path, text: str) -> None:
    """Check the flags and filenames the cards name in running text.

    These are `<code>` chips inside JSX, so none of the markdown checks sees them,
    and each one is something the card is telling a reader to type.
    """
    where = "landing/shared.js"
    accepted = _argparse_flags()
    chips = re.findall(r"<code>\s*(--[a-z0-9-]+)\s*</code>", text)
    claim(bool(chips), where, "no --flag chips matched — did the cards change shape?")
    for flag in sorted(set(chips)):
        claim(
            flag in accepted,
            where,
            f"a card names `{flag}`, which faith's CLI does not accept",
        )

    # `benchmark.yaml` is joined onto the path inline where a config is loaded, so
    # there is no symbol to compare against: the filename in that source is the fact.
    loader = (repo / "src/faith/benchmark/config.py").read_text()
    # `.ya?ml`, so that a card written `benchmark.yml` fails against the loader
    # rather than matching nothing and falling to the shape guard below.
    filenames = re.findall(r"<code>\s*([a-z_]+\.ya?ml)\s*</code>", text)
    claim(
        bool(filenames),
        where,
        "no *.yaml filename chip matched — did the capability cards change?",
    )
    for name in sorted(set(filenames)):
        claim(
            f'/ "{name}"' in loader,
            where,
            f"a card tells readers to define a `{name}`, but load_config_from_path "
            f"does not build that filename",
        )


def check_landing_page(repo: Path) -> None:
    """Check the landing page's demo command, benchmarks, engines and flags.

    The page is JSX, so the path checks in check-repo-facts.js never see it, and
    `SuiteList` was the only component this file read. That left the site's most
    read page free to name a subcommand, benchmark, engine or flag that the repo
    had since renamed.
    """
    # pylint: disable=import-outside-toplevel
    from faith._types.model.engine import ModelEngine

    text = LANDING.read_text()
    published = _published_benchmarks(repo)
    engines = {m.name.lower() for m in ModelEngine}

    _check_demo_command(text, published, engines)
    _check_landing_benchmarks(text, published)
    _check_landing_backends(text, engines)
    _check_landing_flags(repo, text)


def _check_quickstart_example() -> None:
    where = "getting-started/quickstart.md"
    quickstart = (SITE / "docs" / where).read_text()
    command = re.search(r"```shell\n(faith run-all\b.*?)```", quickstart, re.S)
    claim(command is not None, where, "no run-all example found")
    if command is not None:
        tokens = shlex.split(command[1].replace("\\\n", ""))
        args = _cli_subparsers()["run-all"].parse_args(tokens[2:])
        claim(
            "accuracy" in args.stats,
            where,
            "the run-all example must select accuracy score columns with --stats",
        )
        claim(
            args.sample_size is not None and args.sample_size > 0,
            where,
            "the smoke-test example must bound the sample size",
        )
        claim(
            not str(args.datastore_location).startswith("gs:"),
            where,
            "the introductory example should work without Cloud Storage setup",
        )
        claim(
            ".[vllm,metrics]" in quickstart,
            where,
            "the local run-all example needs both inference and table extras",
        )


def check_usage_examples() -> None:
    """Validate the quickstart arguments and the authoring metric example."""
    # pylint: disable=import-outside-toplevel
    import yaml

    from faith._types.config.scoring import OutputProcessingConfig
    from faith._types.record.stats import AnswerFormat, StatsRecord
    from faith.benchmark.agentic import AgenticMetricsAggregator
    from faith.cli.subcmd.eval import _agg_trials
    from faith.ingestion.parser import ExperimentConfig, parse_metrics_data

    _check_quickstart_example()
    where = "benchmarks/creating-benchmarks.md"
    authoring = (SITE / "docs" / where).read_text()
    example = re.search(
        r"### Scores and the primary metric\n.*?```yaml\n(.*?)```",
        authoring,
        re.S,
    )
    claim(example is not None, where, "no primary-metric example found")
    if example is None:
        return
    config = OutputProcessingConfig.from_dict(
        yaml.safe_load(example[1])["output_processing"]
    )
    # This example documents file F1, whose per-record score is a scalar value.
    # Aggregate one perfect score through the real trial and export paths so
    # the documented name is checked against generated metrics, not a literal.
    grade = StatsRecord(
        label=["example.py"],
        prediction=["example.py"],
        answer_format=AnswerFormat.PROPER,
        scores={name: {"value": 1.0} for name in config.score_fns},
    )
    trial = AgenticMetricsAggregator(config)([grade])
    records = parse_metrics_data(
        {"stats": _agg_trials({"trial": trial}.values())},
        "docs-example",
        ExperimentConfig("example-model", "example-model", "example-benchmark"),
        config.primary_metric,
    )
    claim(
        sum(record["is_primary"] is True for record in records) == 1,
        where,
        f"primary_metric={config.primary_metric!r} must identify a generated metric",
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
    check_code_tree(repo)
    check_flow_refs(repo)
    check_agentic_config(repo)
    check_agentic_runtime_reference(repo)
    check_group_exclusions(repo)
    check_extras(repo)
    check_landing_page(repo)
    check_usage_examples()

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
