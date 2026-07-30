---
id: advanced
title: Dashboards — Advanced techniques
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

### Example 2: Normalized score (0–100 scale)

Name: `Score Percentage`

```sql
metric_value * 100
```

Type: Number (with **Percent** formatting). Use as a metric to display values as
percentages instead of decimals.

## Connecting multiple data sources

Add both pre-built views to the same dashboard for different purposes:

1. **leaderboard** — high-level comparisons and rankings.
2. **latest_runs** — detailed analysis with all metrics.

To add a second data source: **Resource** → **Manage added data sources** →
**Add a data source** → select BigQuery and choose the other view. Use Looker
Studio's blending feature to combine data across sources.

## Resources

- [Looker Studio Documentation](https://support.google.com/looker-studio)
- [BigQuery Best Practices](https://cloud.google.com/bigquery/docs/best-practices)
- [FAITH Schema Reference](../schema.md)
