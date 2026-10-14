---
id: visualizations
title: Dashboards — visualizations
sidebar_label: Visualizations
sidebar_position: 2
---

# Essential visualizations

Once you've [set up the views and connected Looker Studio](./setup.md), these
are the core charts to build. Each one names the data source, dimensions, and
metrics to configure.

Use only a verified comparison cohort. The supplied `leaderboard` view collapses
configuration differences; dashboard filters cannot restore discarded runs.
Create a cohort-specific view when needed, as described in
[setup](./setup.md#leaderboard-view-latest-headline-per-model-and-benchmark).

## Leaderboard table

**Purpose:** rank models within a benchmark.

1. Add a **Table** chart to your report.
2. In the **Setup** tab:
   - Set **Data source** to the `leaderboard` view.
   - Set **Dimension** to `benchmark` and `model_key`.
   - Set **Metric** to `metric_value` with **Aggregation** = Average.
   - In **Sort**, select `metric_value`, order descending.
3. In the **Style** tab:
   - Under **Conditional Formatting**, click **Add formatting**.
   - Select **Color Scale (Gradient)**.
   - Set **Format based on** to `metric_value`.
   - Use the default gradient (low = red, high = green).

**Required filter:** select one benchmark and its performance metric. Do not
interpret a value-sorted list across different benchmarks as a common ranking.

## Model comparison bar chart

**Purpose:** compare multiple models side-by-side.

1. Add a **Bar Chart**.
2. In the **Setup** tab:
   - Set **Data source** to the `leaderboard` view.
   - Set **Dimension** to `model_key`.
   - Set **Metric** to `metric_value` with **Aggregation** = Average.
   - In **Sort**, select `metric_value`, order descending.
3. In the **Style** tab, adjust colors as desired.

**Required filters:** select one `benchmark` and one `metric_name` before
averaging `metric_value`. A `model_key` multi-select is optional. Without the
benchmark filter, the chart averages unrelated benchmark scores for each model.

## Scores by ingestion date (line chart)

**Purpose:** inspect scores by ingestion date within a comparable cohort.

1. Add a **Time Series Chart**.
2. In the **Setup** tab:
   - Set **Data source** to the `latest_runs` view.
   - Set **Dimension** to `ingest_time` (as a date range dimension).
   - Set **Breakdown dimension** to `model_key`.
   - Set **Metric** to `metric_value` with **Aggregation** = Average.
   - Filter to one `benchmark` and one performance `metric_name` (for example,
     `mean_file_f1.mean` for vulnerability-localization-a).
3. In the **Style** tab, enable data labels if desired.

Without the metric filter, the chart can average scores, query counts, error
rates, and other unrelated values together. A date bucket's average also weights
each retained run equally, not by its task count.

`ingest_time` is not execution time. Reingestion changes where a result appears,
and `latest_runs` does not preserve every historical ingestion. Use a separate
runtime and revision-aware source for model-development trends.

## Benchmark coverage scorecards

**Purpose:** count models and benchmarks with retained primary metrics. Add **Scorecard**
charts:

1. **Models with Primary Metrics** — Data source `leaderboard`, Metric `model_key`,
   Aggregation = Count Distinct.
2. **Benchmarks with Primary Metrics** — Data source `leaderboard`, Metric `benchmark`,
   Aggregation = Count Distinct.
3. **Latest Ingest** — Data source `leaderboard`, Metric `ingest_time`,
   Aggregation = Max.

These cards omit runs without a primary metric, including all-error agentic
runs. They do not measure completed trials or tasks. Report `queries.total` and
`error_rate.mean` separately and inspect per-trial counts in the saved artifacts.

---

Next: [**Advanced techniques →**](./advanced.md)
