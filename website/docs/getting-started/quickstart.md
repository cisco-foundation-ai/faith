---
id: quickstart
title: Quickstart
sidebar_label: Quickstart
sidebar_position: 2
---

import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# Quickstart

Run a small Q&A evaluation, then inspect its scores and saved files. Complete
[Installation](./installation.md) first. Choose local GPU inference or a model
API below; for tool-using tasks, use the
[agentic smoke test](#run-an-agentic-benchmark).

## 1. Activate your environment

```shell
source .venv/bin/activate
```

## 2. Run a benchmark

Choose the backend you installed. Both examples use `run-all` to query, grade,
and summarize **25 questions per benchmark**, with one trial. Use a new output
directory for this first run.

<Tabs queryString="engine">

<TabItem value="vllm" label="Local GPU" default>

Requires a supported Linux GPU host and `.[vllm,metrics]`. The model download
may require `HF_TOKEN` and publisher access approval; see
[credentials](./installation.md#host-tools-and-credentials).

```shell
faith run-all \
  --model-paths meta-llama/Llama-3.1-8B-Instruct \
  --benchmarks ctibench-mcqa seceval \
  --datastore-location out/run-001 \
  --model-engine vllm \
  --generation-mode chat_comp \
  --prompt-format chat \
  --num-trials 1 \
  --sample-size 25 \
  --n-shot 0 \
  --stats accuracy
```

</TabItem>

<TabItem value="openai" label="Model API">

Requires `.[openai,metrics]` and an `OPENAI_API_KEY` supplied through your
[approved secret mechanism](./installation.md#host-tools-and-credentials).
No local GPU is needed. Requests are billed to your OpenAI account; retain
`--sample-size` to limit the initial evaluation cost.

```shell
faith run-all \
  --model-paths gpt-4.1-2025-04-14 \
  --benchmarks ctibench-mcqa seceval \
  --datastore-location out/run-001 \
  --model-engine openai \
  --generation-mode chat_comp \
  --prompt-format chat \
  --num-trials 1 \
  --sample-size 25 \
  --n-shot 0 \
  --stats accuracy
```

</TabItem>

</Tabs>

## 3. Inspect the results

On a complete run, the console table has an entry for each benchmark with
accuracy columns. `accuracy_mean` is the mean trial accuracy; with one trial,
it is that trial's score. `--stats accuracy` selects these columns; without
`--stats`, the table contains only run metadata.

Under `out/run-001`, each experiment contains:

- `experiment.json`: configuration and execution metadata, saved when the
  query stage completes.
- `metrics.json`: computed metrics from `eval`.
- `trials/…/benchmark-log.json`: the saved questions and model outputs.

Verify that each benchmark has 25 evaluated records and inspect error counts
before interpreting the score. See the
[completeness checks](../cli/eval.md#checking-result-completeness) and
[metric definitions](../analytics/metrics.md). The
[storage reference](../analytics/storage.md#directory-layout) explains the full
directory hierarchy.

## Run an agentic benchmark \{#run-an-agentic-benchmark}

This smoke test runs one vulnerability-localization task. Install
`.[openai,docker,metrics]`, start Docker, and complete the
[Docker preflight](../configuration/environment-configs.md#docker-preflight),
including [runner download access](../configuration/environment-configs.md#runner-network-and-cache).

The agent executes commands in a local container; the model runs remotely.
Set `OPENAI_API_KEY` through your approved secret mechanism. One task can make
multiple billed model requests, so keep the sample size small. The benchmark's
[limits](../benchmarks/creating-benchmarks.md#limits) bound its execution.

```shell title="Agentic smoke query"
# OPENAI_API_KEY must already be set in the runner environment.

faith query \
  --model-paths gpt-4.1-2025-04-14 \
  --model-engine openai \
  --benchmarks vulnerability-localization-a \
  --generation-mode chat_comp \
  --prompt-format chat \
  --num-trials 1 \
  --sample-size 1 \
  --datastore-location out/agentic-smoke
```

Both vulnerability-localization phases require a read-only workspace. The supplied
image supports this policy. Custom deployments must meet the
[environment requirements](../configuration/environment-configs.md#read-only-workspace-requirements).

`query` writes logs but does not create metrics. Grade and summarize the smoke
test separately:

```shell title="Agentic smoke results"
faith eval --experiment-path out/agentic-smoke
faith summarize --experiment-path out/agentic-smoke \
  --stats mean_file_f1 error_rate stopped_reason
```

Expect one evaluated task, no recorded execution error, and a termination
breakdown explaining how the task ended. A low F1 score alone is not a setup
failure. Inspect `benchmark-log.json` for tool output and errors; a successful
command exit is not sufficient validation. See the
[completeness checks](../cli/eval.md#checking-result-completeness).

For other deployments, see [serving a local model](../configuration/serving-agentic-models.md)
or [using Kubernetes](../configuration/environment-configs.md#cluster-prerequisites).
All agentic runs require the `openai` engine, `chat_comp` generation mode, and
`chat` prompt format; see the [full requirements](../cli/query.md#agentic-runs).

## After the first run

- **Evaluate more samples:** Remove `--sample-size` for a full benchmark, or
  increase `--num-trials` to measure variation across trials. Check expected
  runtime and API cost first.
- **Export results:** See [CSV output](../cli/summarize.md#summarize-behavior-flags)
  or [BigQuery setup](../analytics/bigquery/quickstart.md).
- **Use cloud storage:** Configure [Google Cloud tooling](./installation.md#cloud-storage-optional)
  before selecting a `gs://` datastore. See the [CLI overview](../cli/overview.md#the-three-phases)
  for local downloads needed by later phases.
- **Run phases separately:** Use [`query`](../cli/query.md), [`eval`](../cli/eval.md),
  and [`summarize`](../cli/summarize.md) independently.

### Keeping a long run alive

<details>
<summary>Keep a remote run active across SSH disconnections</summary>

A full run can take several hours. When benchmarking on a remote machine, use
a persistent terminal session so an SSH disconnection does not terminate the
run. From your local machine, open or reconnect to a `screen` session named
`NAME`:

```shell
ssh -t user@REMOTE "screen -RdS NAME"
```

This creates the session if it does not exist or reattaches to an existing
session, disconnecting any previous attachment. Session controls:

- **Detach:** `ctrl-a d`
- **Scroll within a session:** `ctrl-a ESC`
- **Force-disconnect a frozen SSH session:** type `~ .`

</details>

### API configuration options \{#query-a-model-over-an-api}

<details>
<summary>Use a local endpoint or a packaged model configuration</summary>

With `vllm`, FAITH loads model weights in-process. With `openai`, it connects to
a running OpenAI-compatible chat API. To use a local server through the same
client, add `--engine-kwargs base_url='"http://localhost:8100/v1"'`.
For tool-using tasks, follow [Serve a model for agentic runs](../configuration/serving-agentic-models.md).

For the hosted API example, replace `--model-paths`, `--model-engine`, and
`--prompt-format` with `--model-configs instruct/gpt-4.1-2025-04-14` to use a
packaged configuration. This also sets context length and generation parameters
instead of leaving them at the CLI defaults. See
[Model config files](../configuration/model-configs.md) for settings and override
precedence.

</details>

## Where to go next

- [CLI commands](../cli/overview.md): every subcommand and flag
- [Model config files](../configuration/model-configs.md): per-model settings
  and override precedence
- [Benchmark catalog](../benchmarks/catalog.md): every available benchmark
- [BigQuery analytics](../analytics/bigquery/quickstart.md): leaderboards and
  dashboards
- [Troubleshooting & FAQ](../troubleshooting.md): failed runs and unexpected
  scores
