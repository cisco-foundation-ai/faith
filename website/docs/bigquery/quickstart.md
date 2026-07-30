---
id: quickstart
title: BigQuery Quickstart
sidebar_label: Quickstart
sidebar_position: 1
---

# BigQuery quickstart

Get started with FAITH BigQuery analytics in 5 minutes.

## Prerequisites

- FAITH installed with BigQuery support: `uv pip install -e '.[bigquery]'`
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
  --model-paths meta-llama/Llama-3.1-8B-Instruct \
  --benchmarks ctibench-mcqa seceval \
  --datastore-location gs://my-bucket/results/run-001 \
  --model-engine vllm \
  --generation-mode chat_comp \
  --prompt-format chat \
  --num-trials 3 \
  --n-shot 0
```

This creates:

```plaintext
gs://my-bucket/results/run-001/
├── ctibench-mcqa/
│   └── Llama-3.1-8B-Instruct/.../
│       ├── experiment.json
│       └── metrics.json
└── seceval/...
```

### 3. Ingest to BigQuery

You can ingest at any path level — the tool recursively finds all
`metrics.json` files. `--experiment-path` is a **local** path, so pull the run
down from GCS first if that's where `query` wrote it:

```bash
gcloud storage cp -r gs://my-bucket/results/run-001 ./results/
```

```bash
# Entire run (all benchmarks, all models)
faith summarize --experiment-path ./results/run-001 \
  --output-format bigquery

# Single benchmark (all models)
faith summarize --experiment-path ./results/run-001/ctibench-mcqa \
  --output-format bigquery

# Single model
faith summarize --experiment-path ./results/run-001/ctibench-mcqa/Llama-3.1-8B \
  --output-format bigquery
```

The tool will:

- Find all `metrics.json` files recursively
- Parse experiment configuration from paired `experiment.json` files
- Create the `metrics` table if it doesn't exist
- Refuse to re-ingest an already-ingested run: if the first record's
  `metrics_file_uri` is already in the table, the whole insert is aborted with
  an error rather than skipped silently
- Insert all metrics in a single batch

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
WHERE temperature = 0.0
  AND num_shots = 5
  AND num_shots_pool_size = 1  -- Fixed examples (not resampled)
  AND is_primary = true
ORDER BY benchmark, metric_value DESC;
```

### Performance over time

```sql
SELECT
  DATE(ingest_time) as date,
  model_key,
  benchmark,
  AVG(metric_value) as avg_score
FROM `my-project.faith_results.metrics`
WHERE is_primary = true
GROUP BY date, model_key, benchmark
ORDER BY date DESC, benchmark;
```

## Troubleshooting

### "BigQuery dependencies not installed"

Install the BigQuery extra:

```bash
uv pip install -e '.[bigquery]'
```

### "BigQuery project not specified" or "BigQuery dataset not specified"

Set the required environment variables:

```bash
export FAITH_BIGQUERY_PROJECT=my-project
export FAITH_BIGQUERY_DATASET=faith_results
# Or use flags: --bigquery-project my-project --bigquery-dataset faith_results
```

If the dataset doesn't exist yet, create it:

```bash
bq mk --dataset ${FAITH_BIGQUERY_PROJECT}:${FAITH_BIGQUERY_DATASET}
```

### Permission denied

Authenticate and verify permissions:

```bash
gcloud auth application-default login
```

Required IAM permissions:

- `bigquery.tables.create`
- `bigquery.tables.updateData`
- `bigquery.jobs.create`

### "Metrics already ingested from: ..."

The tool prevents duplicate ingestion by checking `metrics_file_uri`. To
re-ingest, first delete existing rows:

```sql
DELETE FROM `my-project.faith_results.metrics`
WHERE metrics_file_uri = 'gs://my-bucket/results/.../metrics.json';
```

### "No metrics.json files found"

Verify the path:

```bash
gcloud storage ls -r gs://my-bucket/results/run-001/
```

Ensure files are organized as:

```plaintext
experiment-path/
└── benchmark/
    └── model/.../
        ├── metrics.json
        └── experiment.json
```

## Next steps

- **Full schema:** [Metrics Table Schema](./schema.md)
- **Build dashboards:** [Building Dashboards with Looker Studio](./dashboards/setup.md)
