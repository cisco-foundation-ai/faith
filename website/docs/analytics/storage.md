---
id: storage
title: Storage structure
sidebar_label: Storage structure
sidebar_position: 1
---

import {CombinedLayout} from '@site/src/components/StorageLayout';

# Storage structure

The datastore contains experiment metadata, trial records, and computed
metrics. Retain these artifacts in storage appropriate to the evaluation's
recovery and retention requirements, such as a persistent local volume or
Cloud Storage bucket.

## Agentic task checkpoints

Agentic trials atomically update their local `benchmark-log.json` after each
completed task, before starting the next one. Each record includes the full
trajectory, failure details when applicable, and its model-data hash. No new
checkpoint format or command-line flag is required; Q&A persistence is unchanged.

To resume, rerun against the same retained datastore and trial identity with a
replacement strategy that permits reuse. Existing reconciliation rules still
apply: non-failure trajectories can be reused, failures and missing results are
retried, and the `always` replacement strategy reruns all selected tasks. A task
interrupted before its checkpoint commits runs again from the beginning; there
is no mid-task or tool-call resume.

This is separate from [sandbox recovery](../benchmarks/creating-benchmarks.md#sandbox-recovery),
which can rebuild an unavailable environment during a running task while retaining
the model conversation. Neither mechanism restores lost files or process state.

During resume, checkpoints retain untouched records from the previous log,
including later completed tasks. Only successful completion of a nonempty trial
removes records outside the newly selected task set. An empty selection leaves
the prior log unchanged. Checkpoint write errors stop the trial and propagate to
the caller, leaving the previous valid checkpoint intact.

:::warning
These are **local** checkpoints, not per-task GCS uploads. They do not survive
deletion of temporary pod or container storage. Use retained storage for recovery
across pod replacement; remote sync behavior and credentials are unchanged.
:::

Each checkpoint rewrites the accumulated JSON log, so long trajectories increase
write cost and memory use. Atomic replacement also needs temporary disk space
for the new log alongside the previous one. Use one writer per trial log; this
does not add coordination for concurrent runs targeting the same trial.

## Uploading results

When you use a GCP URI (beginning with `gs://`) as the `--datastore-location`,
benchmark logs are uploaded to GCP periodically during the run and synced once more when the run finishes.

To manually upload local benchmark results to GCP:

```shell
gcloud storage rsync -R -J -P ~/benchmarks/<BENCHMARK> gs://<BENCHMARK_PATH>
```

Replace `<BENCHMARK>` with your local benchmark directory and
`<BENCHMARK_PATH>` with the destination path in your GCP bucket.

## Downloading results

:::warning
Logs can be large because they include responses and trajectories across
benchmarks, models, and trials. Download only the experiment subdirectories
needed for the analysis when a full copy is unnecessary.
:::

### Downloading metrics only

To examine a benchmark, you usually only need the metrics files. Download all
metrics for a benchmark from GCP efficiently, skipping the large
`benchmark-log.json` files:

```shell
gcloud storage rsync -R -J -P -x '.*/benchmark-log\.json$' gs://<BENCHMARK_PATH> ~/benchmarks/<BENCHMARK>
```

This fetches experiment descriptions, benchmark metrics, and summary files
without the full per-query logs.

## Directory layout

Each experiment uses the structure below. Hover or focus a path-segment button
to highlight its description. The final `agentic_...` segment is present only
for agentic benchmarks.

<CombinedLayout />

:::caution[Experiment identity]
An experiment directory is not a unique run ID. Its path does not encode every
model-engine setting, model revision, or data revision. The agentic hash uses
the benchmark's authored configuration, so a deployment override supplied with
`--environment-config` does not change that hash. The resolved environment is
recorded in `experiment.json` instead.

Use separate datastore roots when comparing deployments or revisions that
would otherwise share a path. Record reuse is controlled by the
[`query` replacement strategy](../cli/query.md#other-flags), not by a guarantee
that every configuration change creates a new directory.
:::

### Files

- **`experiment.json`** — describes how the benchmark was run and its
  parameters. Not created until all benchmark trials complete.
- **`metrics.json`** — per-trial metrics and across-trial statistics produced by
  `eval` from available trial logs. See [trial coverage](./metrics.md#trial-coverage-and-weighting).

The **`trials`** subdirectory contains Q&A records or agentic trajectories. Its
`trial-seed` and `benchmark-trial-hash` subfolders both identify the trial:

- the **seed** identifies the trial's sampling seed. Zero-based trial *n* uses
  `--seed` + *n*, so a run with
  `--seed 373363 --num-trials 3` writes `trials/373363/`, `trials/373364/`,
  and `trials/373365/`;
- the **hash** is the last 16 characters of a hash of the benchmark spec (name,
  generation mode, prompt format, and shot count). It does not identify the full
  dataset, rendered prompts, or an agent's subsequent tool interactions.

The same seed and hash do not establish that two trials executed identical
samples or model requests. Compare the saved configurations and records.

Each trial contains:

- **`benchmark-log.json`** — the trial's Q&A records or agentic task trajectories,
  including retained records when results are reused.

### An agentic record \{#agentic-records}

On an [agentic run](../benchmarks/catalog.md#agentic-benchmarks), each record in
`benchmark-log.json` carries the task's whole trajectory under
`model_data.agent_trajectory`:

| Field | Description |
| --- | --- |
| `messages` | Full conversation messages in order, including tool-result content |
| `tool_calls` / `tool_responses` | Per-call metadata and truncated result copies; full tool-result content remains in `messages` |
| `stopped_reason` | how the task ended (see [troubleshooting](../troubleshooting.md#agentic-runs)) |
| `error` | why it failed, on `error` and `environment_error` only |

A failed task records the error message on the trajectory and again as
`model_data.error`, whose `title` is `Agent Run Error` or
`Agent Environment Error`. Search these titles to locate failed records, then
inspect the trajectory's `error` or the wrapped error's details.

`metrics.json` reports error counts and termination reasons, not the full error
messages. Retain the trial logs for failure diagnosis.
