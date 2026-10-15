---
id: schema
title: Metrics table schema
sidebar_label: Schema
sidebar_position: 2
---

# BigQuery metrics table schema

The `metrics` table stores FAITH benchmark results in a normalized format with
one row per metric value.

## Table structure

- **Partitioning:** Daily partitions by `ingest_time`
- **Clustering:** `benchmark`, `model_key`, `metric_name`

## Fields

### Identifiers

| Field | Type | Required | Description |
| --- | --- | --- | --- |
| `metrics_file_uri` | STRING | Yes | Source `metrics.json` path or URI used for duplicate checks; not a content hash or uniqueness constraint. |
| `model_key` | STRING | Yes | Display name for the model (from config or shortened path). |
| `source_uri` | STRING | No | Full model path/URI (e.g. `meta-llama/Llama-3.1-8B-Instruct`). |
| `benchmark` | STRING | Yes | Benchmark name (e.g. `ctibench-mcqa`, `mmlu-security`). |

### Metric data

| Field | Type | Required | Description |
| --- | --- | --- | --- |
| `metric_name` | STRING | Yes | Metric name with flattened path (e.g. `accuracy.mean`). |
| `metric_value` | FLOAT64 | Yes | Computed metric value from the benchmark evaluation. |
| `is_primary` | BOOL | No | Whether this is the benchmark's primary/headline metric (e.g. accuracy for MMLU). |

### Generation parameters

| Field | Type | Required | Description |
| --- | --- | --- | --- |
| `temperature` | FLOAT64 | No | Configured sampling temperature; zero does not guarantee identical outputs. |
| `top_p` | FLOAT64 | No | Nucleus-sampling probability mass threshold (0.0–1.0). |
| `max_completion_tokens` | INT64 | No | Maximum number of tokens the model can generate. |
| `context_length` | INT64 | No | Model's context window size in tokens. |
| `generation_mode` | STRING | No | How the model generates output (e.g. `chat_comp`, `logits`). |
| `prompt_format` | STRING | No | How prompts are formatted (e.g. `chat`, `base`). |

### Experiment configuration

| Field | Type | Required | Description |
| --- | --- | --- | --- |
| `num_shots` | INT64 | No | Number of few-shot examples shown per prompt. |
| `num_shots_pool_size` | INT64 | No | Pool size for resampling few-shot examples (1 = fixed, >1 = resampled). |

### Metadata

| Field | Type | Required | Description |
| --- | --- | --- | --- |
| `faith_version` | STRING | No | FAITH package version used to prepare the ingestion records, not necessarily the version that ran or graded the experiment. |
| `ingest_time` | TIMESTAMP | Yes | Timestamp assigned while preparing each metric row for ingestion; not the experiment execution time. |

## Field details

### `num_shots` and `num_shots_pool_size`

FAITH's `n_shot` parameter supports two formats:

- **Integer** (e.g. `"5"`): fixed examples → `num_shots=5, num_shots_pool_size=1`
- **Fraction** (e.g. `"3/5"`): resampled examples → `num_shots=3, num_shots_pool_size=5`

Query examples:

```sql
-- All 5-shot experiments
WHERE num_shots = 5

-- Fixed examples only (no resampling)
WHERE num_shots_pool_size = 1

-- Resampled examples
WHERE num_shots_pool_size > 1
```

### `is_primary`

Each benchmark defines one primary metric (e.g. `accuracy.mean` for MMLU). This
field is `true` for that metric, `false` for others, and `NULL` if no primary
metric is defined.

Use it for leaderboards:

```sql
SELECT model_key, benchmark, metric_value
FROM metrics
WHERE is_primary = true
ORDER BY benchmark, metric_value DESC
```

### `metrics_file_uri`

The source locator supplied to the parser. The CLI reads local files and records
their absolute paths, including when the files were downloaded from GCS. It does
not recover the original GCS URI. Library callers can instead pass a GCS URI:

```plaintext
gs://bucket/path/to/gen_params_hash/metrics.json
```

or

```plaintext
/local/path/to/gen_params_hash/metrics.json
```

An agentic benchmark hashes its agentic config into a further `agentic_<hash>`
directory below `gen_params_<hash>`, so its metrics file sits one level deeper.

Copying the same result to another local path changes `metrics_file_uri` without
changing the experiment. Conversely, re-grading in the same directory preserves
the URI even if metric values change. It is not a run-content or grading-version
identifier.

The ingestion duplicate check examines only the first record's URI in a batch.
An existing first URI rejects the entire batch; a new first URI does not prevent
duplicates for later files. Use a stable source location and avoid overlapping
or concurrent ingestion batches. See the [ingestion guidance](./quickstart.md#3-ingest-to-bigquery).

## Provenance and comparison limits

The table contains flattened values from `metrics.json`'s `stats` object, not
its per-trial records or the complete experiment snapshot. The exported fields
do not include task selection, dataset revision, random seeds, full engine and
generation settings, agent/tool limits, environment configuration, or grading
identity. Equal `model_key`, `benchmark`, temperature, and shot counts therefore
do not establish comparable experiments.

Retain the paired `experiment.json`, `metrics.json`, and trial logs. Verify those
artifacts when selecting a comparison cohort, and include query counts and
error metrics alongside performance scores. `faith_version` describes the
ingester, and `ingest_time` describes record preparation; neither establishes
when or with which code version the original run was executed or graded.

## Example row

```json
{
  "model_key": "my-model-v1",
  "source_uri": "org/my-model-v1",
  "benchmark": "ctibench-mcqa",
  "metric_name": "accuracy.mean",
  "metric_value": 0.8875,
  "is_primary": true,
  "num_shots": 5,
  "num_shots_pool_size": 1,
  "temperature": 0.0,
  "top_p": 1.0,
  "max_completion_tokens": 64,
  "context_length": 8192,
  "generation_mode": "chat_comp",
  "prompt_format": "chat",
  "metrics_file_uri": "gs://bucket/results/ctibench-mcqa/.../metrics.json",
  "faith_version": "1.0.0",
  "ingest_time": "2026-02-11T10:30:00Z"
}
```

## Next steps

- **Getting started:** [BigQuery quickstart](./quickstart.md)
- **Ready-to-run SQL:** [Example queries & views](./examples.md) — the
  `docs/bigquery/queries/` and `docs/bigquery/views/` files in the repository
- **Build dashboards:** [Building dashboards with Looker Studio](./dashboards/setup.md)
