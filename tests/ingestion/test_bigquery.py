# Copyright 2025 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for BigQuery storage backend."""

from unittest.mock import Mock, patch

import pytest

from faith.ingestion.storage.bigquery import METRICS_SCHEMA, BigQueryClient


class TestBigQueryClient:
    """Tests for BigQueryClient."""

    def test_init(self):
        """Test client initialization."""
        with patch("faith.ingestion.storage.bigquery.bigquery.Client"):
            client = BigQueryClient(
                project_id="test-project",
                dataset_id="test_dataset",
                table_id="test_table",
            )

            assert client.project_id == "test-project"
            assert client.dataset_id == "test_dataset"
            assert client.table_id == "test_table"
            assert client.table_ref == "test-project.test_dataset.test_table"

    def test_schema_structure(self):
        """Test that schema has expected columns."""
        schema_names = [field.name for field in METRICS_SCHEMA]

        # Required columns
        assert "model_key" in schema_names
        assert "benchmark" in schema_names
        assert "metric_name" in schema_names
        assert "metric_value" in schema_names
        assert "metrics_file_uri" in schema_names
        assert "faith_version" in schema_names
        assert "ingest_time" in schema_names

        # Optional but important columns
        assert "temperature" in schema_names
        assert "num_shots" in schema_names
        assert "num_shots_pool_size" in schema_names
        assert "seed" in schema_names
        assert "sample_size" in schema_names
        assert "is_primary" in schema_names

    def test_check_metrics_file_exists_false(self):
        """Test check_metrics_file_exists returns False when file hasn't been ingested."""
        mock_client = Mock()
        mock_result = Mock()
        mock_row = Mock(count=0)
        mock_result.__iter__ = Mock(return_value=iter([mock_row]))
        mock_client.query.return_value.result.return_value = mock_result

        client = BigQueryClient(
            project_id="test-project", dataset_id="test_dataset", client=mock_client
        )

        exists = client.check_metrics_file_exists("gs://bucket/model/metrics.json")
        assert exists is False

    def test_check_metrics_file_exists_true(self):
        """Test check_metrics_file_exists returns True when file exists."""
        mock_client = Mock()
        mock_result = Mock()
        mock_row = Mock(count=5)
        mock_result.__iter__ = Mock(return_value=iter([mock_row]))
        mock_client.query.return_value.result.return_value = mock_result

        client = BigQueryClient(
            project_id="test-project", dataset_id="test_dataset", client=mock_client
        )

        exists = client.check_metrics_file_exists("gs://bucket/model/metrics.json")
        assert exists is True

    def test_insert_metrics_duplicate_check(self):
        """Test that insert_metrics raises error for duplicate metrics_file_uri."""
        mock_client = Mock()

        # Mock table existence check (table already exists)
        mock_client.get_table.return_value = Mock()

        # Mock check_metrics_file_exists to return True (duplicate)
        mock_result = Mock()
        mock_row = Mock(count=1)
        mock_result.__iter__ = Mock(return_value=iter([mock_row]))
        mock_client.query.return_value.result.return_value = mock_result

        client = BigQueryClient(
            project_id="test-project", dataset_id="test_dataset", client=mock_client
        )

        metrics = [
            {
                "metrics_file_uri": "gs://bucket/model/metrics.json",
                "metric_name": "accuracy",
            }
        ]

        with pytest.raises(ValueError, match="Metrics already ingested"):
            client.insert_metrics(metrics, check_duplicates=True)

    def test_insert_metrics_success(self):
        """Test successful metrics insertion."""
        mock_client = Mock()

        # Mock check_run_exists to return False (not duplicate)
        mock_result = Mock()
        mock_row = Mock(count=0)
        mock_result.__iter__ = Mock(return_value=iter([mock_row]))
        mock_client.query.return_value.result.return_value = mock_result
        mock_client.get_table.return_value = Mock()  # Table exists
        mock_client.insert_rows_json.return_value = []  # No errors

        client = BigQueryClient(
            project_id="test-project", dataset_id="test_dataset", client=mock_client
        )

        metrics = [
            {
                "metrics_file_uri": "gs://bucket/model/metrics.json",
                "metric_name": "accuracy",
                "metric_value": 0.85,
            },
            {
                "metrics_file_uri": "gs://bucket/model/metrics.json",
                "metric_name": "f1",
                "metric_value": 0.82,
            },
        ]

        count = client.insert_metrics(metrics, check_duplicates=True)
        assert count == 2
        mock_client.insert_rows_json.assert_called_once()
