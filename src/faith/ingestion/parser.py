# Copyright 2025 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

"""FAITH metrics parsing for ingestion to BigQuery.

This module extracts metrics from FAITH's native output files (metrics.json and
experiment.json) and flattens them into a normalized schema for BigQuery analytics.
"""

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from dataclasses_json import DataClassJsonMixin

from faith import __version__ as faith_version
from faith._internal.io.datastore import DatastoreContext
from faith._internal.io.json import read_json_file
from faith.benchmark.types import BenchmarkSpec


@dataclass
class ExperimentConfig(DataClassJsonMixin):
    """Structured configuration extracted from FAITH's experiment.json."""

    model_key: str
    source_uri: str
    benchmark: str
    temperature: float | None = None
    top_p: float | None = None
    max_completion_tokens: int | None = None
    context_length: int | None = None
    generation_mode: str | None = None
    prompt_format: str | None = None
    num_shots: int | None = None
    num_shots_pool_size: int | None = None


def _extract_model_info(model_config: dict[str, Any]) -> tuple[str, str]:
    """Extract model key and source URI from model config."""
    source_uri = model_config["path"]
    return model_config.get("name") or source_uri, source_uri


def _extract_generation_params(model_config: dict[str, Any]) -> dict:
    """Extract generation parameters from model config."""
    generation = model_config.get("generation") or {}
    engine = model_config.get("engine") or {}

    return {
        "temperature": generation.get("temperature"),
        "top_p": generation.get("top_p"),
        "max_completion_tokens": generation.get("max_completion_tokens"),
        "context_length": engine.get("context_length"),
    }


def _extract_n_shot_values(
    benchmark_spec: BenchmarkSpec,
) -> tuple[int | None, int | None]:
    """Extract n_shot numerator and denominator from BenchmarkSpec."""
    if benchmark_spec.n_shot and benchmark_spec.n_shot.numerator > 0:
        return benchmark_spec.n_shot.numerator, benchmark_spec.n_shot.denominator
    if benchmark_spec.n_shot and benchmark_spec.n_shot.numerator == 0:
        return 0, 1
    return None, None


def parse_experiment_config(experiment_data: dict[str, Any]) -> ExperimentConfig:
    """Parse experiment.json data into a structured configuration object.

    Extracts key configuration parameters from FAITH's experiment.json
    including model generation settings, benchmark parameters, and run metadata.

    Args:
        experiment_data: Raw dictionary from experiment.json

    Returns:
        ExperimentConfig with parsed configuration

    Raises:
        AssertionError: If required fields (model.path, benchmark.name) are missing
    """
    exp_params = experiment_data.get("experiment_params") or {}
    model_config = exp_params.get("model") or {}
    bench_config = exp_params.get("benchmark") or {}

    model_key, source_uri = _extract_model_info(model_config)
    benchmark_spec = BenchmarkSpec.from_dict(bench_config)
    gen_params = _extract_generation_params(model_config)
    num_shots, num_shots_pool_size = _extract_n_shot_values(benchmark_spec)

    return ExperimentConfig(
        model_key=model_key,
        source_uri=source_uri,
        benchmark=benchmark_spec.name,
        temperature=gen_params["temperature"],
        top_p=gen_params["top_p"],
        max_completion_tokens=gen_params["max_completion_tokens"],
        context_length=gen_params["context_length"],
        generation_mode=str(benchmark_spec.generation_mode),
        prompt_format=str(benchmark_spec.prompt_format),
        num_shots=num_shots,
        num_shots_pool_size=num_shots_pool_size,
    )


def parse_primary_metric(experiment_data: dict[str, Any]) -> str | None:
    """Extract primary metric name from experiment.json.

    Args:
        experiment_data: Raw dictionary from experiment.json

    Returns:
        Primary metric name (e.g., "accuracy.mean") or None if not defined
    """
    benchmark_config = experiment_data.get("benchmark_config") or {}
    output_processing = benchmark_config.get("output_processing") or {}
    return output_processing.get("primary_metric")


def _flatten_metrics(obj: dict, prefix: str = "") -> dict[str, float]:
    """Recursively flatten nested metrics dictionary.

    Args:
        obj: Dictionary to flatten
        prefix: Current key prefix for nested keys

    Returns:
        Dictionary of flattened metric_name -> metric_value
    """
    result = {}
    for key, value in obj.items():
        new_key = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            result.update(_flatten_metrics(value, new_key))
        else:
            result[new_key] = value
    return result


def _derive_file_uri(path: str | Path) -> str:
    """Derive a canonical URI string from a file path.

    Args:
        path: File path (local or GCS URI)

    Returns:
        Canonical URI string
    """
    if isinstance(path, str) and path.startswith("gs://"):
        return path

    return str(Path(path).absolute())


def _build_metric_record(
    metric_name: str,
    metric_value: float,
    metrics_file_uri: str,
    experiment_config: ExperimentConfig,
    primary_metric_name: str | None,
    ingest_time: str,
) -> dict[str, Any]:
    """Build a single BigQuery metric record."""
    is_primary = (metric_name == primary_metric_name) if primary_metric_name else None

    return {
        "metric_name": metric_name,
        "metric_value": metric_value,
        "is_primary": is_primary,
        "metrics_file_uri": metrics_file_uri,
        "faith_version": faith_version,
        "ingest_time": ingest_time,
    } | experiment_config.to_dict()


def parse_metrics_data(
    metrics_data: dict[str, Any],
    metrics_file_uri: str,
    experiment_config: ExperimentConfig,
    primary_metric_name: str | None,
) -> list[dict[str, Any]]:
    """Parse metrics.json data into normalized BigQuery records.

    Flattens nested metrics and enriches with experiment configuration.

    Args:
        metrics_data: Raw dictionary from metrics.json
        metrics_file_uri: URI of the metrics file (unique identifier for the run)
        experiment_config: Parsed experiment configuration
        primary_metric_name: Name of the primary metric for this benchmark

    Returns:
        List of metric records (one dict per metric)
    """
    stats = metrics_data.get("stats")
    stats_dict = stats if isinstance(stats, dict) else {}

    return [
        _build_metric_record(
            metric_name,
            metric_value,
            metrics_file_uri,
            experiment_config,
            primary_metric_name,
            datetime.utcnow().isoformat() + "Z",
        )
        for metric_name, metric_value in _flatten_metrics(stats_dict).items()
    ]


def parse_metrics_file(
    metrics_path: str | Path,
    experiment_config: ExperimentConfig,
    primary_metric_name: str | None,
) -> list[dict[str, Any]]:
    """Parse FAITH metrics.json file into normalized BigQuery records.

    Convenience function that combines loading and parsing metrics.

    Args:
        metrics_path: Path to metrics.json (local file or GCS URI)
        experiment_config: Parsed experiment configuration
        primary_metric_name: Name of the primary metric for this benchmark

    Returns:
        List of metric records (one dict per metric)

    Raises:
        FileNotFoundError: If metrics.json doesn't exist
    """
    with DatastoreContext.from_path(str(metrics_path)) as ds:
        return parse_metrics_data(
            read_json_file(ds.pull()),
            _derive_file_uri(metrics_path),
            experiment_config,
            primary_metric_name,
        )
