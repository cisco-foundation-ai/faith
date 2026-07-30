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
      desc: 'Local path to the experiment logs to analyze. If a directory is provided, all experiments within it are analyzed. This is a local filesystem path: a `gs://` URI is not understood here, unlike `--datastore-location` on `query`.',
    },
  ]}
/>

## Summarize behavior flags

<FlagTable
  rows={[
    {
      flag: '--output-format',
      type: 'table | csv | bigquery',
      default: 'table',
      desc: 'Only `bigquery` changes behavior, selecting BigQuery ingestion. Between `table` and `csv` the value is ignored: what decides is whether `--summary-filepath` is set (see the note below).',
    },
    {
      flag: '--stats',
      type: 'list[string]',
      default: '[]',
      desc: 'Space-separated metric names to include (for table/CSV output). Defaults to none: the summary carries only the run columns unless stats are named.',
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
For the non-BigQuery formats, `--summary-filepath` — not `--output-format` —
decides where the summary goes. Set it and a CSV is written, even with
`--output-format table`; leave it unset and a Markdown table is printed to
stdout, even with `--output-format csv`.
:::

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
in Looker Studio. See [BigQuery Analytics](../bigquery/quickstart.md) for the
full guide.
