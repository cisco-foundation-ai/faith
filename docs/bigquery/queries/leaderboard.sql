-- Copyright 2025 Cisco Systems, Inc. and its affiliates
--
-- SPDX-License-Identifier: Apache-2.0

-- Leaderboard Query: Compare models across benchmarks
--
-- Shows primary metrics for all models, sorted by performance.
-- Great for identifying top-performing models on each benchmark.
--
-- Usage:
--   bq query --use_legacy_sql=false < queries/leaderboard.sql

SELECT 
  benchmark,
  model_key,
  metric_name,
  metric_value,
  temperature,
  num_shots,
  ingest_time
FROM `YOUR_PROJECT.YOUR_DATASET.metrics`
WHERE is_primary = true
ORDER BY benchmark, metric_value DESC
LIMIT 100;
