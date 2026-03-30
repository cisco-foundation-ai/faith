# Building Dashboards with Looker Studio

This cookbook teaches you how to create interactive dashboards to visualize your
FAITH benchmark results using Google Looker Studio (formerly Data Studio).

## Prerequisites

- FAITH metrics ingested to BigQuery (see [quickstart.md](quickstart.md))
- Google account with access to your BigQuery project
- BigQuery dataset with metrics data

## Part 1: Set Up BigQuery Views (One-Time Setup)

Before building dashboards, create the pre-built views for better performance.

### Create the Leaderboard View

In the BigQuery console or `bq` CLI, run:

```bash
# From the views/ directory
bq query --use_legacy_sql=false < docs/bigquery/views/leaderboard.sql
```

Or copy the SQL from
[views/leaderboard.sql](views/leaderboard.sql)
and run it in the BigQuery console, replacing `PROJECT_ID` and `DATASET_ID`
with your values.

### Create the Latest Runs View

In the BigQuery console or `bq` CLI, run:

```bash
bq query --use_legacy_sql=false < docs/bigquery/views/latest_runs.sql
```

Or copy the SQL from
[views/latest_runs.sql](views/latest_runs.sql)
and run it in the BigQuery console, replacing `PROJECT_ID` and `DATASET_ID`
with your values.

**Verify:** In the BigQuery console, you should now see `leaderboard` and
`latest_runs` in your dataset.

## Part 2: Connect Looker Studio to BigQuery

### 1. Create a New Report

1. Go to [Looker Studio](https://lookerstudio.google.com/)
2. Click **Create** → **Report**

### 2. Connect Your Data

Choose one of the pre-built views:

#### Leaderboard View (Recommended for Most Dashboards)

1. When prompted to add data, select **BigQuery** as your data source
2. Choose **My Projects** → `YOUR_PROJECT` → `YOUR_DATASET`
3. Select the **leaderboard** view, and click **Add**

**What you get:**

- Primary metrics only (the headline scores for each benchmark)
- Latest results per model/benchmark combination
- Key configuration parameters (temperature, num_shots, etc.)

**Best for:** Executive dashboards, leaderboards, model comparisons

#### Latest Runs View (For Detailed Analysis)

Follow the same steps as above but select the **latest_runs** view instead.

**What you get:**

- All metrics (including non-primary metrics)
- Latest ingestion per unique experiment run
- Full configuration details

**Best for:** Research dashboards, debugging, detailed performance analysis

## Part 3: Essential Visualizations

### Leaderboard Table

**Purpose:** Rank models by performance across benchmarks

**Setup:**

1. Add a **Table** chart to your report
2. In the chart editor's **Setup** tab:
   - Set **Data source** to `leaderboard` view
   - Set **Dimension** to `benchmark` and `model_key`
   - Set **Metric** to `metric_value` with **Aggregation** set to Average
   - In **Sort**, select `metric_value` and set order to descending
3. In the **Style** tab:
   - Under **Conditional Formatting**, click **Add formatting**
   - Select **Color Scale (Gradient)** as the format type
   - Set **Format based on** to `metric_value`
   - Use the default gradient (low values = red, high values = green)

**Optional Filters:**

- Add a dropdown filter for `benchmark` to let users switch between benchmarks

### Model Comparison Bar Chart

**Purpose:** Compare multiple models side-by-side

**Setup:**

1. Add a **Bar Chart** to your report
2. In the chart editor's **Setup** tab:
   - Set **Data source** to `leaderboard` view
   - Set **Dimension** to `model_key`
   - Set **Metric** to `metric_value` with **Aggregation** set to Average
   - In **Sort**, select `metric_value` and set order to descending
3. In the **Style** tab, adjust colors as desired

**Optional Filters:**

- Add a multi-select filter for `model_key` to choose which models to compare.
- Add a single-select filter for `benchmark`.

### Performance Over Time (Line Chart)

**Purpose:** Track model improvements across evaluation runs

**Setup:**

1. Add a **Time Series Chart** to your report
2. In the chart editor's **Setup** tab:
   - Set **Data source** to `latest_runs` view
   - Set **Dimension** to `ingest_time` (configure as a date range dimension)
   - Set **Breakdown dimension** to `model_key`
   - Set **Metric** to `metric_value` with **Aggregation** set to Average
3. In the **Style** tab, enable data labels (if desired)

**Best for:**

- Monitoring regression/improvements after model updates
- Comparing training runs over time

### Benchmark Coverage Scorecard

**Purpose:** Quick stats on evaluation coverage

**Setup:**

Add multiple **Scorecard** charts to show key metrics:

1. **Total Models Evaluated**
   - In the chart editor's **Setup** tab, set **Data source** to `leaderboard`
     view
   - Set **Metric** to `model_key` with **Aggregation** set to Count Distinct
2. **Total Benchmarks**
   - In the **Setup** tab, set **Data source** to `leaderboard` view
   - Set **Metric** to `benchmark` with **Aggregation** set to Count Distinct
3. **Latest Ingest**
   - In the **Setup** tab, set **Data source** to `leaderboard` view
   - Set **Metric** to `ingest_time` with **Aggregation** set to Max

## Part 4: Advanced Techniques

### Using Calculated Fields

Calculated fields let you create new metrics derived from existing data. Use them
to add custom logic like categorizing scores, computing percentages, or creating
conditional values without modifying your BigQuery data.

**How to create a calculated field:**

1. In your Looker Studio report, go to **Resource** >
   **Manage added data sources**
2. Click **Edit** for your data source, then click **Add a field** >
   **Add calculated field**
3. Enter a name for your field (e.g., "Score Category")
4. In the formula box, write your calculation using the formula syntax
5. Set the field type (Text, Number, etc.)
6. Click **Save** and then **Done**

The new field will now appear in your list of available dimensions or metrics.

#### Example 1: Score Category

Create human-readable labels for metric ranges to make dashboards easier to
interpret at a glance.

Name: `Score Category`

Formula:

```sql
CASE
  WHEN metric_value >= 0.9 THEN 'Excellent'
  WHEN metric_value >= 0.7 THEN 'Good'
  WHEN metric_value >= 0.5 THEN 'Fair'
  ELSE 'Poor'
END
```

Type: Text

Use this as a dimension to group or filter by performance tier.

#### Example 2: Normalized Score (0-100 scale)

Convert decimal metrics to a percentage scale for easier comprehension.

Name: `Score Percentage`

Formula:

```sql
metric_value * 100
```

Type: Number (with **Percent** formatting)

Use this as a metric to display values as percentages instead of decimals.

### Connecting Multiple Data Sources

You can add both pre-built views to the same dashboard for different purposes:

1. **leaderboard** view: Use for high-level comparisons and rankings
2. **latest_runs** view: Use for detailed analysis with all metrics

To add a second data source:

1. Click **Resource** → **Manage added data sources**
2. Click **Add a data source**
3. Select BigQuery and choose the other view
4. Use Looker Studio's blending feature to combine data across sources

## Resources

- [Looker Studio Documentation](https://support.google.com/looker-studio)
- [BigQuery Best Practices](https://cloud.google.com/bigquery/docs/best-practices)
- [FAITH Schema Reference](schema.md)
