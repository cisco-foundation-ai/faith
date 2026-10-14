---
id: examples
title: Example queries & views
sidebar_label: Queries & views
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

List primary metrics sorted by benchmark and value. This query does not select
comparable configurations or remove repeated ingestions. Add filters for one
benchmark and a verified comparison cohort before using it to rank models.

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
bq query --use_legacy_sql=false < docs/bigquery/queries/leaderboard.sql
```

### Compare models

Select several models at temperature zero with zero shots. The supplied query
does not filter to one benchmark or constrain other settings. Add a `benchmark`
filter and the required configuration filters; verify the saved experiment
snapshots for [settings not exported to BigQuery](./schema.md#provenance-and-comparison-limits).

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

Show the 20 most recently prepared primary-metric rows. These are not necessarily
20 distinct runs, and ingestion order is not experiment execution order.

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

Views provide the row-selection rules used by the
[Looker Studio dashboards](./dashboards/setup.md). Their grouping and freshness
limits are described below. From the repository root, create them with:

```bash
bq query --project_id=YOUR_PROJECT --dataset_id=YOUR_DATASET \
  --use_legacy_sql=false < docs/bigquery/views/leaderboard.sql
```

### `leaderboard` view

Selects the most recently ingested primary metric for each `(model_key,
benchmark)` pair. This collapses all other configuration differences, including
agent settings, task selections, and grading revisions. Reingesting an older
experiment can make it the selected result.

For controlled comparisons, filter the base `metrics` table to a verified cohort
**before** the window function ranks rows, or use a custom view with the required
configuration grouping. Filtering the supplied view afterward cannot recover
rows it has already discarded.

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

Selects the latest row independently for each `(metrics_file_uri, metric_name)`
pair, including non-primary metrics. It does not select a complete ingestion
snapshot per file: if a later ingestion omits a metric, that metric's older row
remains. The result can therefore combine values from separate ingestions.

Use immutable result locations for separate grading revisions and verify source
artifacts before treating this view as a current snapshot. Its configuration
columns are a subset of the [metrics schema](./schema.md), not the full run
configuration.

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
