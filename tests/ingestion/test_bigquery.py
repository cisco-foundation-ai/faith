# Copyright 2025 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for BigQuery storage backend."""

from unittest.mock import Mock, patch

import pytest
from google.api_core.exceptions import NotFound

from faith.ingestion.storage.bigquery import METRICS_SCHEMA, BigQueryClient


def test_init():
    """Test client initialization."""
    mock_client = Mock(get_table=Mock(return_value=Mock()))

    with patch(
        "faith.ingestion.storage.bigquery.bigquery.Client", return_value=mock_client
    ):
        client = BigQueryClient(
            project_id="test-project",
            dataset_id="test_dataset",
            table_id="test_table",
        )

        assert client.table_ref == "test-project.test_dataset.test_table"


def test_schema_structure():
    """Test that schema has expected columns."""
    assert {field.name for field in METRICS_SCHEMA} == {
        "model_key",
        "benchmark",
        "metric_name",
        "metric_value",
        "metrics_file_uri",
        "faith_version",
        "ingest_time",
        "temperature",
        "top_p",
        "num_shots",
        "num_shots_pool_size",
        "is_primary",
        "source_uri",
        "max_completion_tokens",
        "context_length",
        "generation_mode",
        "prompt_format",
    }


def test_check_metrics_file_exists_false():
    """Test check_metrics_file_exists returns False when file hasn't been ingested."""
    mock_client = Mock(
        query=Mock(return_value=Mock(result=Mock(return_value=Mock(total_rows=0)))),
        get_table=Mock(return_value=Mock()),
    )

    with patch(
        "faith.ingestion.storage.bigquery.bigquery.Client", return_value=mock_client
    ):
        client = BigQueryClient(project_id="test-project", dataset_id="test_dataset")

    assert not client.check_metrics_file_exists("gs://bucket/model/metrics.json")


def test_check_metrics_file_exists_true():
    """Test check_metrics_file_exists returns True when file exists."""
    mock_client = Mock(
        query=Mock(return_value=Mock(result=Mock(return_value=Mock(total_rows=1)))),
        get_table=Mock(return_value=Mock()),
    )

    with patch(
        "faith.ingestion.storage.bigquery.bigquery.Client", return_value=mock_client
    ):
        client = BigQueryClient(project_id="test-project", dataset_id="test_dataset")

    assert client.check_metrics_file_exists("gs://bucket/model/metrics.json")


def test_insert_metrics_duplicate_check():
    """Test that insert_metrics raises error for duplicate metrics_file_uri."""
    mock_client = Mock(
        query=Mock(return_value=Mock(result=Mock(return_value=Mock(total_rows=1)))),
        get_table=Mock(return_value=Mock()),
    )

    with patch(
        "faith.ingestion.storage.bigquery.bigquery.Client", return_value=mock_client
    ):
        client = BigQueryClient(project_id="test-project", dataset_id="test_dataset")

        metrics = [
            {
                "metrics_file_uri": "gs://bucket/model/metrics.json",
                "metric_name": "accuracy",
            }
        ]

        with pytest.raises(ValueError, match="Metrics already ingested"):
            client.insert_metrics(metrics, check_duplicates=True)


def test_insert_metrics_success():
    """Test successful metrics insertion."""
    mock_job = Mock(result=Mock(return_value=None), output_rows=2)
    mock_client = Mock(
        query=Mock(return_value=Mock(result=Mock(return_value=Mock(total_rows=0)))),
        get_table=Mock(return_value=Mock()),
        load_table_from_json=Mock(return_value=mock_job),
    )

    with patch(
        "faith.ingestion.storage.bigquery.bigquery.Client", return_value=mock_client
    ):
        client = BigQueryClient(project_id="test-project", dataset_id="test_dataset")

        assert (
            client.insert_metrics(
                [
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
                ],
                check_duplicates=True,
            )
            == 2
        )
        mock_client.load_table_from_json.assert_called_once()
        mock_job.result.assert_called_once()


def test_ensure_table_creates_table():
    """Test that table is created when it doesn't exist."""
    mock_client = Mock(
        get_table=Mock(side_effect=[NotFound("test"), Mock()]),
        create_table=Mock(return_value=Mock()),
    )

    with patch(
        "faith.ingestion.storage.bigquery.bigquery.Client", return_value=mock_client
    ):
        _ = BigQueryClient(project_id="test-project", dataset_id="test_dataset")

        mock_client.create_table.assert_called_once()
        assert mock_client.create_table.call_args[0][0].clustering_fields == [
            "benchmark",
            "model_key",
            "metric_name",
        ]
