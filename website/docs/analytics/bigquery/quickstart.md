---
id: quickstart
title: BigQuery quickstart
sidebar_label: Quickstart
sidebar_position: 1
---

# BigQuery quickstart

Evaluate a bounded sample locally, verify its metrics, and ingest the results
into BigQuery.

## Prerequisites

- FAITH installed with the extras used below: `uv pip install -e '.[vllm,metrics,bigquery]'`
- A GPU environment suitable for the selected vLLM model; see
  [installation](../../getting-started/installation.md)
- A Google Cloud project with the BigQuery API enabled
- GCloud CLI authenticated: `gcloud auth application-default login`

## Setup

### 1. Create a BigQuery dataset

Set your project and dataset names as environment variables, then create the
dataset using the `bq` CLI:

```bash
export FAITH_BIGQUERY_PROJECT=my-project
export FAITH_BIGQUERY_DATASET=faith_results

bq mk --dataset ${FAITH_BIGQUERY_PROJECT}:${FAITH_BIGQUERY_DATASET}
```

The metrics table (default: `metrics`) is created automatically on first
ingestion.

**Using a custom table name:**

```bash
export FAITH_BIGQUERY_TABLE=metrics_prod
# Or use: --bigquery-table metrics_prod
```

Any table name will be auto-created with the correct schema.

### 2. Run an evaluation

```bash
faith run-all \
  --model-paths meta-llama/Llama-3.1-8B-Instruct@name=llama3-8b \
  --benchmarks ctibench-mcqa seceval \
  --datastore-location ./results/run-001 \
  --model-engine vllm \
  --generation-mode chat_comp \
  --prompt-format chat \
  --num-trials 1 \
  --sample-size 25 \
  --n-shot 0 \
  --stats accuracy
```

This creates:

```plaintext
./results/run-001/
├── ctibench-mcqa/
│   └── llama3-8b/.../
│       ├── experiment.json
│       └── metrics.json
└── seceval/...
```

### 3. Ingest to BigQuery

Before ingestion, inspect trial/task counts and error metrics using the
[`eval` completeness checks](../../cli/eval.md#checking-result-completeness).
`summarize` does not validate freshness or completeness.

You can ingest at any path level: the tool recursively finds all
`metrics.json` files. `--experiment-path` is a local path. If results are instead
stored in GCS, download them to a stable local location first:

```bash
gcloud storage cp -r gs://my-bucket/results/run-001 ./results/
```

Choose **one** of the following scopes. These commands are alternatives, not
consecutive steps; running all three attempts to ingest overlapping results.

```bash
# Entire run (all benchmarks, all models)
faith summarize --experiment-path ./results/run-001 \
  --output-format bigquery

# Single benchmark (all models)
faith summarize --experiment-path ./results/run-001/ctibench-mcqa \
  --output-format bigquery

# Single model
faith summarize --experiment-path ./results/run-001/ctibench-mcqa/llama3-8b \
  --output-format bigquery
```

The tool will:

- Find all `metrics.json` files recursively
- Parse experiment configuration from paired `experiment.json` files
- Create the `metrics` table if it does not exist
- Check the first record's `metrics_file_uri`: if it already exists in the
  table, abort the whole batch with an error
- Insert all metrics in a single batch

:::caution
Duplicate detection is not batch-wide. If the first URI is new, later files can
be inserted again even when they already exist in the table. Use disjoint
ingestion scopes, retain stable source paths, and avoid concurrent submissions
of the same results. Downloaded GCS results are identified by their local path,
not their original GCS URI. See [`metrics_file_uri`](./schema.md#metrics_file_uri).
:::

## Query your data

### BigQuery console

```sql
SELECT
  model_key,
  benchmark,
  metric_name,
  metric_value,
  num_shots,
  temperature
FROM `my-project.faith_results.metrics`
WHERE is_primary = true
ORDER BY benchmark, metric_value DESC;
```

This lists stored primary metrics; it does not establish a controlled model
comparison. Verify cohort configuration and coverage before ranking results.

### Using `bq` CLI

```bash
bq query --use_legacy_sql=false \
  'SELECT model_key, benchmark, metric_value
   FROM `'${FAITH_BIGQUERY_PROJECT}.${FAITH_BIGQUERY_DATASET}'.metrics`
   WHERE is_primary = true
   ORDER BY metric_value DESC
   LIMIT 10'
```

## Example queries

### Filter by experiment configuration

```sql
SELECT
  model_key,
  benchmark,
  metric_value
FROM `my-project.faith_results.metrics`
WHERE benchmark = 'ctibench-mcqa'
  AND metric_name = 'accuracy.mean'
  AND temperature = 0.0
  AND num_shots = 5
  AND num_shots_pool_size = 1  -- Fixed examples (not resampled)
  AND is_primary = true
ORDER BY benchmark, metric_value DESC;
```

The fields above constrain only part of the run. Compare the saved snapshots
for settings absent from the table, including agentic limits and grading
configuration; see [comparison limits](./schema.md#provenance-and-comparison-limits).

### Scores by ingestion date

For a verified cohort ingested once per result file:

```sql
SELECT
  DATE(ingest_time) as date,
  model_key,
  benchmark,
  AVG(metric_value) as avg_score
FROM `my-project.faith_results.metrics`
WHERE is_primary = true
  AND benchmark = 'ctibench-mcqa'
  AND metric_name = 'accuracy.mean'
  AND temperature = 0.0
  AND num_shots = 0
GROUP BY date, model_key, benchmark
ORDER BY date DESC, benchmark;
```

This is an average of ingested rows by date, not a task-weighted score or an
execution-time history. Duplicate ingestions can change the average, and
reingesting an older run places it at the new ingestion date.

## When ingestion fails

The BigQuery failures — an unset project or dataset, a missing IAM permission,
`Metrics already ingested from: …`, and `No metrics.json files found` — are in
[Troubleshooting](../../troubleshooting.md#results--analytics) with the rest of
the run's symptoms, so one page covers a failure whichever phase it surfaces in.
A missing `[bigquery]` extra is under
[installation](../../troubleshooting.md#installation).

## Next steps

- **Full schema:** [Metrics table schema](./schema.md)
- **Build dashboards:** [Building dashboards with Looker Studio](./dashboards/setup.md)
