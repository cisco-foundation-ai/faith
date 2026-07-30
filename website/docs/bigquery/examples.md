---
id: examples
title: Example Queries & Views
sidebar_label: Queries & Views
sidebar_position: 4
---

# Example queries & views

{/* The SQL blocks below are generated from the .sql files by
    `npm run sync-sql` — edit those files, not this page. `npm run check-repo`
    fails if they drift. Each block's title="..." names its source, relative to
    docs/bigquery/. */}

FAITH ships a set of ready-to-run SQL under `docs/bigquery/queries/` and
`docs/bigquery/views/` in the repository. **Queries** answer a question on the
spot; **views** are reusable, named objects you create once and query (or wire
into a dashboard) repeatedly.

Replace `YOUR_PROJECT` / `YOUR_DATASET` (or `PROJECT_ID` / `DATASET_ID` in the
view definitions) with your own values before running.

## Queries

### Leaderboard

Compare models across benchmarks, showing primary metrics sorted by
performance. Great for identifying the top-performing model on each benchmark.

```sql title="queries/leaderboard.sql"
SELECT
    benchmark,
    model_key,
    metric_name,
    metric_value,
    temperature,
    num_shots,
    ingest_time
FROM `YOUR_PROJECT.YOUR_DATASET.metrics`
WHERE is_primary = TRUE
ORDER BY benchmark ASC, metric_value DESC
LIMIT 100;
```

```bash
bq query --use_legacy_sql=false < queries/leaderboard.sql
```

### Compare models

Compare several models on the same benchmark under the same configuration —
useful for A/B testing and model selection. Edit the `model_key` list and the
`benchmark`/config filters to taste.

```sql title="queries/compare_models.sql"
SELECT
    model_key,
    benchmark,
    metric_name,
    metric_value,
    temperature,
    num_shots,
    ingest_time
FROM `YOUR_PROJECT.YOUR_DATASET.metrics`
WHERE
    model_key IN ('model-a', 'model-b', 'model-c')
    AND is_primary = TRUE
    AND temperature = 0.0
    AND num_shots = 0
ORDER BY benchmark ASC, metric_value DESC;
```

### Latest runs

Show the 20 most recent ingestions with their primary metrics — handy for
monitoring recent evaluation activity.

```sql title="queries/latest_runs.sql"
SELECT
    ingest_time,
    model_key,
    benchmark,
    metric_name,
    metric_value,
    temperature,
    num_shots
FROM `YOUR_PROJECT.YOUR_DATASET.metrics`
WHERE is_primary = TRUE
ORDER BY ingest_time DESC
LIMIT 20;
```

## Views

Views power the [Looker Studio dashboards](./dashboards/setup.md) and give you clean,
deduplicated tables to query. Create them once:

```bash
bq query --project_id=YOUR_PROJECT --dataset_id=YOUR_DATASET \
  --use_legacy_sql=false < views/leaderboard.sql
```

### `leaderboard` view

Shows the **most recent** primary metric for each model/benchmark combination —
i.e. the current headline score per model per benchmark.

```sql title="views/leaderboard.sql"
CREATE OR REPLACE VIEW `PROJECT_ID.DATASET_ID.leaderboard` AS
SELECT
    model_key,
    benchmark,
    metric_name,
    metric_value,
    temperature,
    num_shots,
    num_shots_pool_size,
    ingest_time
FROM (
    SELECT
        *,
        ROW_NUMBER() OVER (
            PARTITION BY model_key, benchmark
            ORDER BY ingest_time DESC
        ) AS rn
    FROM `PROJECT_ID.DATASET_ID.metrics`
    WHERE is_primary = TRUE
) AS ranked
WHERE ranked.rn = 1
ORDER BY benchmark ASC, metric_value DESC;
```

### `latest_runs` view

Deduplicates metrics to the **latest ingestion of each `metrics_file_uri`**,
keeping all metrics (not just primary). Use it for dashboards that should show
current results with full detail.

```sql title="views/latest_runs.sql"
CREATE OR REPLACE VIEW `PROJECT_ID.DATASET_ID.latest_runs` AS
SELECT
    metrics_file_uri,
    model_key,
    benchmark,
    metric_name,
    metric_value,
    is_primary,
    temperature,
    top_p,
    num_shots,
    num_shots_pool_size,
    generation_mode,
    prompt_format,
    ingest_time
FROM (
    SELECT
        *,
        ROW_NUMBER() OVER (
            PARTITION BY metrics_file_uri, metric_name
            ORDER BY ingest_time DESC
        ) AS rn
    FROM `PROJECT_ID.DATASET_ID.metrics`
) AS ranked
WHERE ranked.rn = 1
ORDER BY ingest_time DESC;
```

## Next steps

- [Build dashboards](./dashboards/setup.md) from these views in Looker Studio.
- [Schema reference](./schema.md) for every column you can query.
