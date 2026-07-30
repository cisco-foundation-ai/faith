---
id: overview
title: CLI Overview
sidebar_label: Overview
sidebar_position: 1
---

# CLI Overview

The primary way to run FAITH's benchmarks is through the `faith` command,
installed into your virtual environment. It exposes the following subcommands:

| Subcommand | Alias | Purpose | Requires |
| --- | --- | --- | --- |
| [`query`](./query.md) | — | Run benchmarks on models; record prompts/responses | a model-engine extra (`vllm`, `openai`, …) |
| [`eval`](./eval.md) | — | Analyze benchmark logs and compute metrics | — |
| [`summarize`](./summarize.md) | `digest` | Aggregate metrics into a table, CSV, or BigQuery | `[metrics]` for table output, `[bigquery]` for BigQuery |
| `run-all` | `qed` | Run `query`, `eval`, and `summarize` sequentially | union of the above |

## The three phases

FAITH separates a benchmark run into three phases so each can be re-run
independently:

1. **`query`** records the raw model input/output records into a datastore.
   It does not score anything — that's the next phase's job.
2. **`eval`** reads those logs and computes metrics, writing `metrics.json`.
3. **`summarize`** collates all metrics in a directory into a summary you can
   read, export, or ingest.

## `run-all` (a.k.a. `qed`)

The `run-all` subcommand runs all three stages sequentially. It accepts:

- every flag [`query`](./query.md) takes — the **Model** and **Experiment**
  flags plus `--datastore-location` and `--replacement-strategy`,
- the **Summarize behavior** and **BigQuery output** flags from
  [`summarize`](./summarize.md).

It does **not** accept `--experiment-path`. Both later stages work off
`--datastore-location`: `eval` runs on each experiment `query` just wrote, and
`summarize` collates the whole datastore. `--replacement-strategy` is shared —
one value governs both the query and the eval stage; you can't set them apart.

This is the most convenient way to go from "a model and a benchmark" to
"a summary of results" in one command.

## Related reference

- [Model Config Files](./model-configs.md) — specify per-model engine and
  generation settings in YAML.
- [Storage Structure](../architecture/storage.md) — where logs, experiments,
  and metrics land on disk / in cloud storage.
