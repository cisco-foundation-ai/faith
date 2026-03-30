# BigQuery Metrics Table Schema

The `metrics` table stores FAITH benchmark results in a normalized format with
one row per metric value.

## Table Structure

**Partitioning:** Daily partitions by `ingest_time`  
**Clustering:** `benchmark`, `model_key`, `metric_name`

## Fields

### Identifiers

| Field              | Type   | Required | Description                                                         |
| ------------------ | ------ | -------- | ------------------------------------------------------------------- |
| `metrics_file_uri` | STRING | Yes      | URI of source metrics.json file. Unique identifier for idempotency. |
| `model_key`        | STRING | Yes      | Display name for the model (from config or shortened path)          |
| `source_uri`       | STRING | No       | Full model path/URI (e.g., "meta-llama/Llama-3.1-8B-Instruct")      |
| `benchmark`        | STRING | Yes      | Benchmark name (e.g., "ctibench-mcqa", "mmlu-security")             |

### Metric Data

| Field          | Type    | Required | Description                                                                       |
| -------------- | ------- | -------- | --------------------------------------------------------------------------------- |
| `metric_name`  | STRING  | Yes      | Metric name with flattened path (e.g., "accuracy.mean")                           |
| `metric_value` | FLOAT64 | Yes      | Computed metric value from the benchmark evaluation                               |
| `is_primary`   | BOOL    | No       | Whether this is the benchmark's primary/headline metric (e.g., accuracy for MMLU) |

### Generation Parameters

| Field                   | Type    | Required | Description                                               |
| ----------------------- | ------- | -------- | --------------------------------------------------------- |
| `temperature`           | FLOAT64 | No       | Sampling temperature (0.0 = deterministic)                |
| `top_p`                 | FLOAT64 | No       | Probability mass threshold for nucleus sampling (0.0-1.0) |
| `max_completion_tokens` | INT64   | No       | Maximum number of tokens the model can generate           |
| `context_length`        | INT64   | No       | Model's context window size in tokens                     |
| `generation_mode`       | STRING  | No       | How model generates output (e.g., "chat_comp", "logits")  |
| `prompt_format`         | STRING  | No       | How prompts are formatted (e.g., "chat", "base")          |

### Experiment Configuration

| Field                 | Type  | Required | Description                                                        |
| --------------------- | ----- | -------- | ------------------------------------------------------------------ |
| `num_shots`           | INT64 | No       | Number of few-shot examples shown per prompt                       |
| `num_shots_pool_size` | INT64 | No       | Pool size for resampling few-shot examples (1=fixed, >1=resampled) |

### Metadata

| Field           | Type      | Required | Description                                |
| --------------- | --------- | -------- | ------------------------------------------ |
| `faith_version` | STRING    | No       | FAITH version that generated these metrics |
| `ingest_time`   | TIMESTAMP | Yes      | Timestamp when metrics were ingested       |

## Field Details

### `num_shots` and `num_shots_pool_size`

FAITH's `n_shot` parameter supports two formats:

- **Integer** (e.g., `"5"`): Fixed examples → `num_shots=5,
num_shots_pool_size=1`
- **Fraction** (e.g., `"3/5"`): Resampled examples → `num_shots=3,
num_shots_pool_size=5`

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

Each benchmark defines one primary metric (e.g., "accuracy.mean" for MMLU).
This field is `true` for that metric, `false` for others, and `NULL` if no
primary metric is defined.

Use for leaderboards:

```sql
SELECT model_key, benchmark, metric_value
FROM metrics
WHERE is_primary = true
ORDER BY benchmark, metric_value DESC
```

### `metrics_file_uri`

The unique identifier for each experiment run, preventing duplicate ingestion.
Format:

```plaintext
gs://bucket/path/to/gen_params_hash/metrics.json
```

or

```plaintext
/local/path/to/gen_params_hash/metrics.json
```

## Example Row

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

## Next Steps

- **Pre-built views:** See [views/](views/) for SQL view definitions
  (leaderboard, latest_runs)
- **Query examples:** See [queries/](queries/) for production-ready query
  templates
- **Getting started:** See [quickstart.md](quickstart.md) for setup and basic
  queries
