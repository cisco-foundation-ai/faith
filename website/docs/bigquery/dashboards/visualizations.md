---
id: visualizations
title: Dashboards — Visualizations
sidebar_label: Visualizations
sidebar_position: 2
---

# Essential visualizations

Once you've [set up the views and connected Looker Studio](./setup.md), these
are the core charts to build. Each one names the data source, dimensions, and
metrics to configure.

## Leaderboard table

**Purpose:** rank models by performance across benchmarks.

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

**Optional filter:** add a dropdown for `benchmark` to switch between
benchmarks.

## Model comparison bar chart

**Purpose:** compare multiple models side-by-side.

1. Add a **Bar Chart**.
2. In the **Setup** tab:
   - Set **Data source** to the `leaderboard` view.
   - Set **Dimension** to `model_key`.
   - Set **Metric** to `metric_value` with **Aggregation** = Average.
   - In **Sort**, select `metric_value`, order descending.
3. In the **Style** tab, adjust colors as desired.

**Optional filters:** a multi-select for `model_key`, and a single-select for
`benchmark`.

## Performance over time (line chart)

**Purpose:** track model improvements across evaluation runs.

1. Add a **Time Series Chart**.
2. In the **Setup** tab:
   - Set **Data source** to the `latest_runs` view.
   - Set **Dimension** to `ingest_time` (as a date range dimension).
   - Set **Breakdown dimension** to `model_key`.
   - Set **Metric** to `metric_value` with **Aggregation** = Average.
3. In the **Style** tab, enable data labels if desired.

**Best for:** monitoring regressions/improvements after model updates and
comparing training runs over time.

## Benchmark coverage scorecards

**Purpose:** quick stats on evaluation coverage. Add multiple **Scorecard**
charts:

1. **Total Models Evaluated** — Data source `leaderboard`, Metric `model_key`,
   Aggregation = Count Distinct.
2. **Total Benchmarks** — Data source `leaderboard`, Metric `benchmark`,
   Aggregation = Count Distinct.
3. **Latest Ingest** — Data source `leaderboard`, Metric `ingest_time`,
   Aggregation = Max.

---

Next: [**Advanced techniques →**](./advanced.md)
