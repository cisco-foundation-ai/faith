---
id: overview
title: CLI overview
sidebar_label: Overview
sidebar_position: 1
---

# CLI overview

The primary way to run FAITH's benchmarks is through the `faith` command,
installed into your virtual environment. It exposes the following subcommands:

| Subcommand | Alias | Purpose | Requires |
| --- | --- | --- | --- |
| [`query`](./query.md) | — | Run benchmarks on models; record every exchange | a model-engine extra (`vllm`, `openai`, …), plus `docker` or `kubernetes` for an [agentic benchmark](../benchmarks/catalog.md#agentic-benchmarks) |
| [`eval`](./eval.md) | — | Analyze benchmark logs and compute metrics | a model-engine extra if the benchmark uses an LLM judge |
| [`summarize`](./summarize.md) | `digest` | Collate saved metrics for table or CSV output, or BigQuery ingestion | `[metrics]` for table output, `[bigquery]` for BigQuery |
| [`run-all`](#run-all-aka-qed) | `qed` | Run `query`, `eval`, and `summarize` sequentially | the union of the above; `[metrics]` unless `--summary-filepath` or `--output-format bigquery` is set |

## The three phases

Each phase reads what the previous one wrote, so each can be re-run on its own.
`query` records exchanges without scoring them, `eval` turns those logs into a
`metrics.json` beside each experiment, and `summarize` collates every
`metrics.json` under a directory. Grading implementation changes can be applied
by [forcing re-grading](./eval.md#regrading-saved-results) of saved logs; `eval`
uses the saved configuration, not an edited benchmark YAML. Changing the output
format requires only re-running `summarize`. Neither phase regenerates benchmark
answers, but an LLM-based grader can invoke its judge model.

`eval` and `summarize` read a local `--experiment-path`, not the `gs://` URI
that `query` can write to, so results kept in cloud storage need
[downloading](../analytics/storage.md#downloading-results) first.

## `run-all` (a.k.a. `qed`)

The `run-all` subcommand runs all three stages sequentially. It accepts:

- every flag [`query`](./query.md) takes: the **Model** and **Experiment**
  flags plus `--datastore-location` and `--replacement-strategy`,
- the **Summarize behavior** and **BigQuery output** flags from
  [`summarize`](./summarize.md).

It does **not** accept `--experiment-path`. Both later stages use
`--datastore-location`: `eval` processes each experiment produced by `query`, and
`summarize` collates the whole datastore. `--replacement-strategy` is shared,
with one value governing both the query and the eval stage; the two cannot be
given different values.

Because `summarize` runs last, a missing `[metrics]` extra surfaces only after
the query and eval stages have finished: the run does all of its work and then
fails while printing the summary. Install it, or send the summary to a file with
`--summary-filepath`.

## Related reference

Each subcommand's own flags are on its page above. What those flags refer to is
documented separately:

- [Model config files](../configuration/model-configs.md): specify a model's
  engine and generation settings in YAML instead of by flag.
- [Environment config files](../configuration/environment-configs.md): where an
  agentic benchmark runs its agents.
- [Storage structure](../analytics/storage.md): where logs, experiments, and
  metrics are stored locally or in cloud storage. `query` writes trial records
  and experiment metadata, `eval` writes metrics, and `summarize` reads both
  metadata and metrics.
