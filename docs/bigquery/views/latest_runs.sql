-- Copyright 2025 Cisco Systems, Inc. and its affiliates
--
-- SPDX-License-Identifier: Apache-2.0

-- Latest runs view: Most recent ingestion for each unique metrics file
--
-- This view deduplicates metrics to show only the latest ingestion of each
-- metrics_file_uri, useful for dashboards that should show current results.
--
-- Usage:
--   bq query --project_id=YOUR_PROJECT --dataset_id=YOUR_DATASET \
--     --use_legacy_sql=false < views/latest_runs.sql

CREATE OR REPLACE VIEW `${PROJECT_ID}.${DATASET_ID}.latest_runs` AS
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
  num_trials,
  seed,
  sample_size,
  generation_mode,
  prompt_format,
  ingest_time
FROM (
  SELECT 
    *,
    ROW_NUMBER() OVER (
      PARTITION BY metrics_file_uri, metric_name
      ORDER BY ingest_time DESC
    ) as rn
  FROM `${PROJECT_ID}.${DATASET_ID}.metrics`
)
WHERE rn = 1
ORDER BY ingest_time DESC;
