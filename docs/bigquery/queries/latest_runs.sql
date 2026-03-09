-- Copyright 2025 Cisco Systems, Inc. and its affiliates
--
-- SPDX-License-Identifier: Apache-2.0

-- Latest Runs Query: Most recently ingested runs
--
-- Shows the 20 most recent ingestions with their primary metrics.
-- Useful for monitoring recent evaluation activity.
--
-- Usage:
--   bq query --use_legacy_sql=false < queries/latest_runs.sql

SELECT
    ingest_time,
    model_key,
    benchmark,
    metric_name,
    metric_value,
    temperature,
    num_shots,
    seed
FROM `YOUR_PROJECT.YOUR_DATASET.metrics`
WHERE is_primary = TRUE
ORDER BY ingest_time DESC
LIMIT 20;
