---
id: advanced
title: Dashboards — advanced techniques
sidebar_label: Advanced techniques
sidebar_position: 3
---

# Advanced techniques

These techniques build on the [essential visualizations](./visualizations.md) to
derive new metrics and combine data across views.

## Calculated fields

Calculated fields create new metrics derived from existing data — categorizing
scores, computing percentages, or conditional values — without modifying
BigQuery.

**How to create one:**

1. Go to **Resource** → **Manage added data sources**.
2. Click **Edit** for your data source, then **Add a field** → **Add calculated
   field**.
3. Name the field, write the formula, set the field type, and click **Save**.

### Example 1: Score category

Apply this example only to one performance metric whose range is 0–1 and whose
higher values indicate better performance. Counts and error rates require
different interpretation.

Name: `Score Category`

```sql
CASE
  WHEN metric_value >= 0.9 THEN 'Excellent'
  WHEN metric_value >= 0.7 THEN 'Good'
  WHEN metric_value >= 0.5 THEN 'Fair'
  ELSE 'Poor'
END
```

Type: Text. Use as a dimension to group or filter by performance tier.

### Example 2: Score displayed as a percentage

Name: `Score Percentage`

```sql
metric_value
```

Type: **Percent**. Apply this only to a selected 0–1 performance metric; a value
of 0.8875 displays as 88.75%. Do not also multiply by 100. The Percent type expects
a ratio; see Google's [data type reference](https://docs.cloud.google.com/looker/docs/studio/data-types).

If a numeric 0–100 scale is required instead, use `metric_value * 100` with type
**Number**, without Percent formatting.

## Connecting multiple data sources

Add both pre-built views to the same dashboard for different purposes:

1. **leaderboard** — high-level comparisons and rankings.
2. **latest_runs** — detailed analysis with all metrics.

To add a second data source: **Resource** → **Manage added data sources** →
**Add a data source** → select BigQuery and choose the other view. Use Looker
Studio's blending feature to combine data across sources.

Do not join the views only on `model_key`: one model can have several benchmarks
and many metric rows, which can multiply rows and distort aggregates. Select the
required benchmark and metric first and verify the join's uniqueness. The
supplied `leaderboard` view does not expose `metrics_file_uri`, so it cannot
identify a particular result file when joining coverage or error metrics; use a
custom source retaining that key for per-run comparisons.

## Resources

- [Looker Studio Documentation](https://support.google.com/looker-studio)
- [BigQuery Best Practices](https://cloud.google.com/bigquery/docs/best-practices)
- [FAITH schema reference](../schema.md)
