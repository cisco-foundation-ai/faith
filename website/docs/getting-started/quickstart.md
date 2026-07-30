---
id: quickstart
title: Quickstart
sidebar_label: Quickstart
sidebar_position: 2
---

# Quickstart

This guide runs a small benchmark end-to-end. It assumes you have
[installed FAITH](./installation.md) with an appropriate extra
(e.g. `vllm` for local models or `openai`/`openrouter` for hosted APIs).

## 1. Activate your environment

```shell
source .venv/bin/activate
```

## 2. Run a benchmark

The primary entry point is the `faith` command. To run the full pipeline —
query, eval, and summarize — in one shot, use `run-all`:

```shell
faith run-all \
  --model-paths meta-llama/Llama-3.1-8B-Instruct \
  --benchmarks ctibench-mcqa seceval \
  --datastore-location gs://my-bucket/results/run-001 \
  --model-engine vllm \
  --generation-mode chat_comp \
  --prompt-format chat \
  --num-trials 3 \
  --n-shot 0
```

This will:

1. **Query** the model over the `ctibench-mcqa` and `seceval` benchmarks.
2. **Eval** the resulting logs to compute metrics, writing a `metrics.json`
   beside each experiment.
3. **Summarize** the metrics — printed as a table by default; pass
   `--output-format csv` or `bigquery` to change that.

Each phase is also available as its own subcommand
([`query`](../cli/query.md), [`eval`](../cli/eval.md),
[`summarize`](../cli/summarize.md)) if you want to run them independently.

:::tip
**Sampling for a quick test** — add `--sample-size 25` to run against only 25
questions per benchmark, handy for a smoke test before committing to a full run.
:::

### Keeping a long run alive

A full run can take hours, easily outlasting an SSH connection. If you're
benchmarking on a remote machine, start the run inside a long-lived terminal
session so a disconnect doesn't kill it. From your local machine, open (or
reconnect to) a `screen` session named `NAME`:

```shell
ssh -t user@REMOTE "screen -RdS NAME"
```

This creates the session if it doesn't exist and reattaches if it does,
disconnecting any previous connection. Useful shortcuts:

- **Detach:** `ctrl-a d`
- **Scroll within a session:** `ctrl-a ESC`
- **Force-disconnect a frozen SSH session:** type `~ .`

## 3. Querying a hosted API model

To use an API-based model instead of a local one, set the engine and the
relevant credential environment variable. For OpenAI:

```shell
export OPENAI_API_KEY=sk-...

faith query \
  --model-paths gpt-4.1-2025-04-14 \
  --model-engine openai \
  --benchmarks seceval \
  --generation-mode chat_comp \
  --prompt-format chat \
  --datastore-location out/ \
  --sample-size 25
```

No GPUs are needed — the model runs on OpenAI's side. Usage is billed to your
OpenAI account, so `--sample-size` is worth keeping while you experiment.

:::tip
FAITH ships a packaged config for this model, so you can replace
`--model-paths`, `--model-engine`, and `--prompt-format` with a single
`--model-configs instruct/gpt-4.1-2025-04-14`. The config also sets the
engine's context length and generation parameters, which the flags above leave
at their defaults. See [Model Config Files](../cli/model-configs.md).
:::

## 4. Inspect the results

Benchmark logs are written under your `--datastore-location`, organized by
benchmark, model, prompt format, generation mode, and shot count. Each
completed run produces an `experiment.json` (how the run was configured) and a
`metrics.json` (computed metrics). See
[Storage Structure](../architecture/storage.md) for the full layout.

## Where to go next

- [CLI Reference](../cli/overview.md) — every subcommand and flag
- [Model Config Files](../cli/model-configs.md) — reproducible per-model setups
- [Benchmark Catalog](../benchmarks/catalog.md) — every available benchmark
- [BigQuery Analytics](../bigquery/quickstart.md) — leaderboards and dashboards
