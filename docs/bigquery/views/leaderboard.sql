-- Copyright 2025 Cisco Systems, Inc. and its affiliates
--
-- SPDX-License-Identifier: Apache-2.0

-- Leaderboard view: Latest primary metrics for each model/benchmark combination
--
-- This view shows the most recent score for each model on each benchmark,
-- filtered to only show primary metrics (accuracy.mean, etc.).
--
-- Usage:
--   bq query --project_id=YOUR_PROJECT --dataset_id=YOUR_DATASET \
--     --use_legacy_sql=false < views/leaderboard.sql

CREATE OR REPLACE VIEW `${PROJECT_ID}.${DATASET_ID}.leaderboard` AS
SELECT 
  model_key,
  benchmark,
  metric_name,
  metric_value,
  temperature,
  num_shots,
  num_shots_pool_size,
  seed,
  ingest_time
FROM (
  SELECT 
    *,
    ROW_NUMBER() OVER (
      PARTITION BY model_key, benchmark 
      ORDER BY ingest_time DESC
    ) as rn
  FROM `${PROJECT_ID}.${DATASET_ID}.metrics`
  WHERE is_primary = true
)
WHERE rn = 1
ORDER BY benchmark, metric_value DESC;
