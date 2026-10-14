---
id: setup
title: Dashboards — setup
sidebar_label: Setup
sidebar_position: 1
---

# Building dashboards with Looker Studio

This cookbook teaches you how to create interactive dashboards to visualize your
FAITH benchmark results using Google Looker Studio (formerly Data Studio).

It is split across three pages:

1. **Setup** (this page) — create the BigQuery views and connect Looker Studio.
2. [**Visualizations**](./visualizations.md) — build the essential charts.
3. [**Advanced techniques**](./advanced.md) — calculated fields and blending
   multiple data sources.

## Prerequisites

- FAITH metrics ingested to BigQuery (see [Quickstart](../quickstart.md))
- A Google account with access to your BigQuery project
- A BigQuery dataset with metrics data

## Part 1: Set up BigQuery views (one-time setup)

Before building dashboards, select the view whose row-selection rules match the
analysis. The
`leaderboard.sql` and `latest_runs.sql` view definitions ship with FAITH under
`docs/bigquery/views/` in the repository — see [Example queries &
views](../examples.md) for the full SQL.

### Create the leaderboard view

In the BigQuery console or `bq` CLI, run:

```bash
# From the repository root, after replacing the project and dataset placeholders
bq query --use_legacy_sql=false < docs/bigquery/views/leaderboard.sql
```

Or copy the SQL from `views/leaderboard.sql` and run it in the BigQuery console,
replacing `PROJECT_ID` and `DATASET_ID` with your values.

### Create the latest runs view

```bash
bq query --use_legacy_sql=false < docs/bigquery/views/latest_runs.sql
```

**Verify:** In the BigQuery console you should now see `leaderboard` and
`latest_runs` in your dataset.

## Part 2: Connect Looker Studio to BigQuery

### 1. Create a new report

1. Go to [Looker Studio](https://lookerstudio.google.com/)
2. Click **Create** → **Report**

### 2. Connect your data

Choose one of the pre-built views:

#### Leaderboard view (latest headline per model and benchmark)

1. When prompted to add data, select **BigQuery** as your data source
2. Choose **My Projects** → `YOUR_PROJECT` → `YOUR_DATASET`
3. Select the **leaderboard** view and click **Add**

**What you get:** primary metrics only (the headline scores per benchmark),
latest results per model/benchmark combination, and key configuration
parameters (temperature, num_shots, etc.).

The view collapses configuration differences before dashboard filters are
applied. For controlled model comparisons, first create a cohort-specific view
that filters the base table before ranking; see the
[leaderboard view limitations](../examples.md#leaderboard-view).

#### Latest runs view (for detailed analysis)

Follow the same steps but select the **latest_runs** view instead.

**What you get:** the latest row per source file and metric name, including
non-primary metrics, with selected configuration fields. This is not a complete
latest-ingestion snapshot: metrics absent from a newer ingestion can retain older
values. See the [latest-runs view limitations](../examples.md#latest_runs-view).

**Best for:** research dashboards, debugging, detailed performance analysis.

For either source, validate trial/task coverage and compare saved experiment
snapshots before interpreting score differences. The exported table does not
include the full agent, environment, dataset, or grading configuration.

---

Next: [**Visualizations →**](./visualizations.md)
