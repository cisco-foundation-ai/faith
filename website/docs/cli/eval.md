---
id: eval
title: The eval subcommand
sidebar_label: eval
sidebar_position: 3
---

import FlagTable from '@site/src/components/FlagTable';

# The `eval` subcommand

The `eval` subcommand grades the records [`query`](./query.md) wrote and
computes metrics over them. It writes a `metrics.json` beside each
`experiment.json` it processes, and writes each record's grade back into the
trial log it came from: `query` records every answer with its `stats` empty, and
`eval` is what fills them in.

## Flags

<FlagTable
  rows={[
    {
      flag: '--experiment-path',
      type: 'Path',
      required: true,
      desc: 'Local path to the experiment logs to analyze. If a directory is given, every experiment under it is analyzed.',
    },
    {
      flag: '--replacement-strategy',
      type: 'string',
      default: 'if_hash_differs',
      desc: 'Which existing grades to keep when re-grading: `never` grades only records that have none; `always` regrades everything; `if_hash_differs` regrades a record when its question or its model output no longer matches the hash stored with it, including when no hash was stored.',
    },
  ]}
/>

`eval` does not regenerate answers or rerun agent trajectories. However, a
benchmark configured with an LLM judge can load or query the judge model during
grading. Its engine dependencies, credentials, and inference costs still apply.

Which metrics a benchmark produces, and what `metrics.json` contains, are in the
[metrics reference](../analytics/metrics.md): see
[where metrics live](../analytics/metrics.md#where-metrics-live) for the file's
shape.

## Re-grading saved results \{#regrading-saved-results}

`eval` reconstructs the benchmark from the `benchmark_config` and benchmark
specification stored in `experiment.json`. It uses the installed grading code,
but does not reload the original `benchmark.yaml` or accept a replacement
benchmark configuration flag.

The default `if_hash_differs` policy compares the stored record's input and
model-output hashes. It does not detect changes to extractor code, scoring code,
or grading configuration. To apply an implementation correction under the same
saved configuration, preserve a copy of the experiment directory, then run:

```shell
faith eval --experiment-path ./results/run-001 \
  --replacement-strategy always
```

This rewrites record-level grades in the trial logs and replaces `metrics.json`.
If the grading configuration itself must change, prepare a separate derived
experiment with an explicitly revised snapshot and record the changes before
forcing re-grading. Editing the original YAML alone has no effect. Preserve the
original logs and snapshot for comparison; this is a manual migration, not an
automatic `eval` workflow.

## Checking result completeness

Evaluation uses only trial logs listed in `experiment.json`. Missing log files
are skipped, and empty trials are omitted from the aggregate. A partially
populated log can therefore produce metrics without a completeness error.

Before comparing or publishing results:

1. Check that every listed trial log exists and that the listed trials match the
   intended run. Unlisted logs are not evaluated.
2. Compare `per_trial_metrics` and its `query_count` values with the intended
   trial and sample/task counts. Inspect `scored_count`, `error_count`, and, for
   agentic runs, termination reasons as well.
3. Rerun `eval` after changing or resuming query logs, then run `summarize`.
   Existing metrics are not automatically validated against the current logs.

See [sample counts and trial weighting](../analytics/metrics.md#trial-coverage-and-weighting)
for how incomplete or failed trials affect the reported means.

## Example

```shell
faith eval --experiment-path ./results/run-001
```

:::note
`--experiment-path` is parsed as a local filesystem path, so a `gs://` URI does
not work: it collapses to `gs:/my-bucket/...`, matches nothing, and reports no
error. To grade results held in GCS, copy them down first
(`gcloud storage cp -r`). `--datastore-location`, on `query` and `run-all`, is
the only flag that reads and writes GCS directly.
:::
