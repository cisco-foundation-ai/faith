# Copyright 2025 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

"""BigQuery storage backend for FAITH metrics ingestion."""

import logging
from typing import Any

from google.api_core.exceptions import NotFound
from google.cloud import bigquery
from google.cloud.bigquery import SchemaField

logger = logging.getLogger(__name__)


# BigQuery table schema for FAITH metrics
METRICS_SCHEMA = [
    SchemaField(
        "model_key",
        "STRING",
        mode="REQUIRED",
        description="Model name (from @name annotation or path)",
    ),
    SchemaField(
        "source_uri",
        "STRING",
        mode="NULLABLE",
        description="Full model path/URI from experiment.json",
    ),
    SchemaField("benchmark", "STRING", mode="REQUIRED", description="Benchmark name"),
    SchemaField(
        "metric_name",
        "STRING",
        mode="REQUIRED",
        description="Metric name (e.g., accuracy.mean)",
    ),
    SchemaField("metric_value", "FLOAT64", mode="REQUIRED", description="Metric value"),
    SchemaField(
        "is_primary",
        "BOOL",
        mode="NULLABLE",
        description="Whether this is the primary metric",
    ),
    SchemaField(
        "num_shots",
        "INT64",
        mode="NULLABLE",
        description="Number of few-shot examples shown per prompt",
    ),
    SchemaField(
        "num_shots_pool_size",
        "INT64",
        mode="NULLABLE",
        description="Pool size for resampling (1=fixed examples)",
    ),
    SchemaField("num_trials", "INT64", mode="NULLABLE", description="Number of trials"),
    SchemaField(
        "seed", "INT64", mode="NULLABLE", description="Random seed for reproducibility"
    ),
    SchemaField(
        "sample_size",
        "INT64",
        mode="NULLABLE",
        description="Dataset sample size (null=full benchmark)",
    ),
    SchemaField(
        "temperature", "FLOAT64", mode="NULLABLE", description="Generation temperature"
    ),
    SchemaField(
        "top_p",
        "FLOAT64",
        mode="NULLABLE",
        description="Probability mass threshold for nucleus sampling (0.0-1.0)",
    ),
    SchemaField(
        "max_completion_tokens",
        "INT64",
        mode="NULLABLE",
        description="Max completion tokens",
    ),
    SchemaField(
        "context_length", "INT64", mode="NULLABLE", description="Model context length"
    ),
    SchemaField(
        "generation_mode", "STRING", mode="NULLABLE", description="Generation mode"
    ),
    SchemaField(
        "prompt_format", "STRING", mode="NULLABLE", description="Prompt format"
    ),
    SchemaField(
        "metrics_file_uri",
        "STRING",
        mode="REQUIRED",
        description="URI of source metrics.json",
    ),
    SchemaField(
        "faith_version",
        "STRING",
        mode="NULLABLE",
        description="FAITH version that ran the experiment",
    ),
    SchemaField(
        "ingest_time",
        "TIMESTAMP",
        mode="REQUIRED",
        description="When metrics were ingested",
    ),
]


class BigQueryClient:
    """Client for inserting FAITH metrics into BigQuery.

    Handles table creation, schema management, and metric insertion with
    idempotency checks to prevent duplicate ingestion.
    """

    def __init__(
        self,
        project_id: str,
        dataset_id: str,
        table_id: str = "metrics",
    ):
        """Initialize BigQuery client.

        Args:
            project_id: GCP project ID
            dataset_id: BigQuery dataset name
            table_id: BigQuery table name (default: "metrics")
        """
        self.client = bigquery.Client(project=project_id)
        self._table_ref = f"{project_id}.{dataset_id}.{table_id}"
        self._ensure_table_exists()

    @property
    def table_ref(self) -> str:
        """Get the full table reference."""
        return self._table_ref

    def _ensure_table_exists(self) -> None:
        """Create the metrics table if it doesn't exist.

        The table is created with:
        - Partitioning by ingest_time (daily)
        - Clustering by benchmark, model_key, metric_name

        Raises:
            GoogleAPIError: If table creation fails
        """
        try:
            self.client.get_table(self.table_ref)
            return
        except NotFound:
            pass

        # Create table with schema
        table = bigquery.Table(self.table_ref, schema=METRICS_SCHEMA)

        # Partition by ingest_time (daily partitions)
        table.time_partitioning = bigquery.TimePartitioning(
            type_=bigquery.TimePartitioningType.DAY, field="ingest_time"
        )

        # Cluster by commonly queried columns
        table.clustering_fields = ["benchmark", "model_key", "metric_name"]

        table = self.client.create_table(table)

    def check_metrics_file_exists(self, metrics_file_uri: str) -> bool:
        """Check if metrics from a given file URI have already been ingested.

        This provides simple idempotency protection to prevent duplicate ingestion.

        Args:
            metrics_file_uri: Metrics file URI to check

        Returns:
            True if file has been ingested, False otherwise

        Note:
            metrics_file_uri uniquely identifies each experiment run
            (model + benchmark + generation parameters).
        """
        result = self.client.query(
            f"""
                SELECT 1
                FROM `{self.table_ref}`
                WHERE metrics_file_uri = @metrics_file_uri
                LIMIT 1
            """,
            job_config=bigquery.QueryJobConfig(
                query_parameters=[
                    bigquery.ScalarQueryParameter(
                        "metrics_file_uri", "STRING", metrics_file_uri
                    )
                ]
            ),
        ).result()
        return result.total_rows > 0

    def insert_metrics(
        self, metrics: list[dict[str, Any]], check_duplicates: bool = True
    ) -> int:
        """Insert metric records into BigQuery using Load Jobs.

        Args:
            metrics: List of metric records (output from parse_metrics_file)
            check_duplicates: Whether to check for existing metrics_file_uri

        Returns:
            Number of rows inserted

        Raises:
            ValueError: If metrics_file_uri already exists and check_duplicates=True
            GoogleAPIError: If load job fails
        """
        if not metrics:
            return 0

        # Check for duplicates if requested
        if check_duplicates and metrics:
            metrics_file_uri = metrics[0].get("metrics_file_uri")
            if metrics_file_uri and self.check_metrics_file_exists(metrics_file_uri):
                raise ValueError(
                    f"Metrics already ingested from: {metrics_file_uri}. "
                    "Delete existing rows or use check_duplicates=False."
                )

        # Load data using Load Jobs (atomic, all-or-nothing)
        # If job fails, GoogleAPIError is raised and no rows are inserted
        self.client.load_table_from_json(metrics, self.table_ref).result()

        return len(metrics)
