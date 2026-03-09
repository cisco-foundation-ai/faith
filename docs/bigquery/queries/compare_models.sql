-- Copyright 2025 Cisco Systems, Inc. and its affiliates
--
-- SPDX-License-Identifier: Apache-2.0

-- Compare Models Query: Side-by-side model comparison
--
-- Compare multiple models on the same benchmark with the same configuration.
-- Useful for A/B testing and model selection.
--
-- Usage:
--   bq query --use_legacy_sql=false < queries/compare_models.sql
--
-- Customize:
--   - Update the model_key list to your models of interest
--   - Update the benchmark filter if needed

SELECT 
  model_key,
  benchmark,
  metric_name,
  metric_value,
  temperature,
  num_shots,
  ingest_time
FROM `YOUR_PROJECT.YOUR_DATASET.metrics`
WHERE model_key IN ('model-a', 'model-b', 'model-c')
  AND is_primary = true
  AND temperature = 0.0
  AND num_shots = 0
ORDER BY benchmark, metric_value DESC;
