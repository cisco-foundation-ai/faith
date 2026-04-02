# Copyright 2025 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

"""Implementation of the subcommand `summarize` to summarize experiment metrics."""

import os
from enum import auto
from pathlib import Path
from typing import Sequence

from faith._internal.io.json import read_json_file
from faith._internal.io.pandas import safe_df_to_csv
from faith._internal.io.resource_provider import ResourceProvider
from faith._types.enums import CIStrEnum
from faith.experiment.summarize import build_summary
from faith.ingestion.parser import (
    parse_experiment_config,
    parse_metrics_file,
    parse_primary_metric,
)


class OutputFormat(CIStrEnum):
    """Output format for summarize command."""

    TABLE = auto()
    CSV = auto()
    BIGQUERY = auto()


def summarize_experiments(
    output_format: OutputFormat,
    experiment_path: Path,
    selected_stats: Sequence[str],
    summary_filepath: Path | None,
    *,
    bigquery_project: str | None = None,
    bigquery_dataset: str | None = None,
    bigquery_table: str | None = None,
) -> None:
    """Summarize the experiments in the given path.

    Args:
        output_format: Output format (table, csv, or bigquery)
        experiment_path: Path to experiment results
        selected_stats: Stats to include (for table/csv output)
        summary_filepath: Output file path (for csv output)
        bigquery_project: GCP project ID (for bigquery output)
        bigquery_dataset: BigQuery dataset name (for bigquery output)
        bigquery_table: BigQuery table name (default: metrics)
    """
    if output_format == OutputFormat.BIGQUERY:
        _summarize_to_bigquery(
            experiment_path,
            bigquery_project,
            bigquery_dataset,
            bigquery_table,
        )
    else:
        _summarize_to_table_or_csv(
            experiment_path,
            selected_stats,
            summary_filepath,
        )


def _summarize_to_table_or_csv(
    experiment_path: Path,
    selected_stats: Sequence[str],
    summary_filepath: Path | None,
) -> None:
    """Summarize experiments to table or CSV."""
    if summary_filepath is not None and summary_filepath.exists():
        raise FileExistsError(f"Output path {summary_filepath} already exists")

    summary = build_summary(experiment_path, selected_stats)
    if summary_filepath is not None:
        safe_df_to_csv(summary, summary_filepath)
    else:
        print(summary.to_markdown(index=False))


def _resolve_bigquery_config(
    bigquery_project: str | None,
    bigquery_dataset: str | None,
    bigquery_table: str | None,
) -> tuple[str, str, str]:
    """Resolve BigQuery configuration from arguments and environment variables."""
    project = bigquery_project or os.getenv("FAITH_BIGQUERY_PROJECT")
    dataset = bigquery_dataset or os.getenv("FAITH_BIGQUERY_DATASET")
    table = bigquery_table or os.getenv("FAITH_BIGQUERY_TABLE") or "metrics"

    assert project, (
        "BigQuery project not specified. "
        "Set FAITH_BIGQUERY_PROJECT environment variable or "
        "use --bigquery-project flag"
    )
    assert dataset, (
        "BigQuery dataset not specified. "
        "Set FAITH_BIGQUERY_DATASET environment variable or "
        "use --bigquery-dataset flag"
    )

    return project, dataset, table


def _find_metrics_files(experiment_path: Path) -> list[Path]:
    """Find all metrics.json files in the experiment directory."""
    if not (metrics_paths := list(experiment_path.glob("**/metrics.json"))):
        raise FileNotFoundError(f"No metrics.json files found in {experiment_path}")
    return metrics_paths


def _process_metrics_file(metrics_path: Path):
    """Parse a single metrics file and return the metrics records."""
    experiment_file = metrics_path.parent / "experiment.json"
    if not experiment_file.exists():
        raise FileNotFoundError(
            f"Missing experiment.json for {metrics_path}. Expected: {experiment_file}"
        )

    with ResourceProvider(str(experiment_file)) as path:
        experiment_data = read_json_file(path)
    experiment_config = parse_experiment_config(experiment_data)
    primary_metric_name = parse_primary_metric(experiment_data)
    return parse_metrics_file(metrics_path, experiment_config, primary_metric_name)


def _summarize_to_bigquery(
    experiment_path: Path,
    bigquery_project: str | None,
    bigquery_dataset: str | None,
    bigquery_table: str | None,
) -> None:
    """Summarize experiments to BigQuery."""
    # We disable the import-outside-toplevel pylint rule here because BigQuery
    # dependencies are only installed as an optional package extra to allow
    # for a smaller install footprint for users who only need table/csv output.
    # pylint: disable=import-outside-toplevel
    try:
        from faith.ingestion.storage.bigquery import BigQueryClient
    except ImportError as e:
        raise ImportError(
            "BigQuery dependencies not installed. "
            "Install with: pip install faith[bigquery]"
        ) from e

    # Resolve configuration
    config = _resolve_bigquery_config(
        bigquery_project, bigquery_dataset, bigquery_table
    )

    # Find and parse all metrics files first (fail fast before any DB writes)
    all_metrics = [
        metric
        for metrics_path in _find_metrics_files(experiment_path)
        for metric in _process_metrics_file(metrics_path)
    ]

    # Insert all metrics in one batch (all or nothing)
    BigQueryClient(*config).insert_metrics(all_metrics, check_duplicates=True)
