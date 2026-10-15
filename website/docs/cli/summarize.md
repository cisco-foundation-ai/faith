---
id: summarize
title: The summarize subcommand
sidebar_label: summarize
sidebar_position: 4
---

import FlagTable from '@site/src/components/FlagTable';

# The `summarize` subcommand

The `summarize` subcommand (alias `digest`) collates all benchmark metrics in a
given directory into a summary. You can output to the console/CSV (default) or
ingest directly to BigQuery for advanced analytics.

`summarize` reads existing `metrics.json` files; it does not regrade records or
check whether those metrics still match the trial logs. After resuming a run or
changing its logs, rerun [`eval`](./eval.md#checking-result-completeness) and
verify coverage before exporting. Table/CSV rows expose only a subset of the
configuration, so identical displayed labels do not establish equivalent runs.

:::note
Console table output requires the `[metrics]` extra (it supplies the `tabulate`
backend). BigQuery ingestion requires the `[bigquery]` extra. CSV output needs
neither.
:::

## Required flags

<FlagTable
  rows={[
    {
      flag: '--experiment-path',
      type: 'Path',
      required: true,
      desc: 'Local directory holding the runs to collate: every `metrics.json` under it is included, along with the `experiment.json` beside it. Unlike `eval`, it must be a directory, not a single experiment file. A `gs://` URI is not understood here.',
    },
  ]}
/>

## Summarize behavior flags

<FlagTable
  rows={[
    {
      flag: '--output-format',
      type: 'string',
      default: 'table',
      desc: 'Set to `bigquery` to ingest into BigQuery instead of emitting a table or CSV. Between `table` and `csv` the value is ignored; the caution below says what decides.',
    },
    {
      flag: '--stats',
      type: 'list[string]',
      default: '[]',
      desc: 'Space-separated stat names to include (for table/CSV output), each expanding into one column per aggregate: `--stats accuracy` yields `accuracy_mean`, `accuracy_std`, `accuracy_min`, `accuracy_max`, `accuracy_median`, `accuracy_p_25` and `accuracy_p_75`. Names come from the keys under `stats` in `metrics.json`. Defaults to none, leaving only the run columns.',
    },
    {
      flag: '--summary-filepath',
      type: 'Path',
      default: 'None',
      desc: 'Write the summary to this CSV file instead of printing a table. Fails if the path already exists. (The `query`/`run-all` help text calls this an "output directory"; it is a file path.)',
    },
  ]}
/>

:::caution
For the non-BigQuery formats, where the summary goes is decided by
`--summary-filepath`, not `--output-format`. Set it and a CSV is written, even
with `--output-format table`; leave it unset and a Markdown table is printed to
stdout, even with `--output-format csv`.
:::

## Summarizing an agentic run

Use `--stats` to select keys under `stats` in `metrics.json`. For an agentic run,
include `stopped_reason` alongside the benchmark's score:

```shell
faith summarize --experiment-path ./results \
  --stats mean_file_f1 stopped_reason
```

Tasks ending with `error` or `environment_error` are excluded from score
aggregation. Tasks ending with `no_tool_calls`, `max_calls`, `max_turns`, or
`timeout` remain included; in vulnerability localization, a missing submission
scores zero. A successful command exit does not establish that every task
completed successfully. Inspect the termination breakdown and error metrics
alongside the score.

For an agentic run with recorded termination reasons, `--stats stopped_reason`
produces three columns:

| Column | Value |
| --- | --- |
| `stopped_reason_total` | Dictionary of task counts by reason, summed over trials |
| `stopped_reason_mean_per_trial` | Dictionary of task counts by reason, divided by the number of trials |
| `stopped_reason_mean_per_query` | Dictionary of task counts by reason, divided by the total query count |

Each dictionary includes all seven reasons, including those with zero counts.
Reasons are not expanded into separate columns. Q&A runs do not emit this
breakdown.

For the meaning of each reason, see
[troubleshooting](../troubleshooting.md#agentic-runs). For the underlying metrics
structure and scoring exclusions, see
[how many samples the score describes](../analytics/metrics.md#how-many-samples-the-score-describes).

## BigQuery output flags

Used when `--output-format bigquery`:

<FlagTable
  rows={[
    {flag: '--bigquery-project', type: 'string', desc: 'GCP project ID. Also settable via `FAITH_BIGQUERY_PROJECT`.'},
    {flag: '--bigquery-dataset', type: 'string', desc: 'BigQuery dataset name. Also settable via `FAITH_BIGQUERY_DATASET`.'},
    {flag: '--bigquery-table', type: 'string', default: 'metrics', desc: 'BigQuery table name. Also settable via `FAITH_BIGQUERY_TABLE`.'},
  ]}
/>

### Quick BigQuery setup

```bash
# One-time setup
export FAITH_BIGQUERY_PROJECT=my-project
export FAITH_BIGQUERY_DATASET=faith_results
bq mk --dataset ${FAITH_BIGQUERY_PROJECT}:${FAITH_BIGQUERY_DATASET}

# Ingest results
uv pip install -e '.[bigquery]'
faith summarize --experiment-path ./results \
  --output-format bigquery
```

With metrics in BigQuery you can build leaderboards comparing models across
benchmarks, track performance over time, query with SQL, and power dashboards
in Looker Studio. See
[BigQuery analytics](../analytics/bigquery/quickstart.md) for the full guide.
