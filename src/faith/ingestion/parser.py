# Copyright 2025 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

"""
FAITH metrics parsing for ingestion to BigQuery.

This module extracts metrics from FAITH's native output files (metrics.json and
experiment.json) and flattens them into a normalized schema for BigQuery analytics.
"""

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from faith import __version__ as faith_version
from faith._internal.io.datastore import DatastoreContext
from faith._internal.io.json import read_json_file
from faith.benchmark.types import BenchmarkSpec


@dataclass
class ExperimentConfig:
    """Structured configuration extracted from FAITH's experiment.json."""

    model_name: str
    model_path: str
    benchmark_name: str
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    max_completion_tokens: Optional[int] = None
    context_length: Optional[int] = None
    generation_mode: Optional[str] = None
    prompt_format: Optional[str] = None
    num_shots: Optional[int] = None
    num_shots_pool_size: Optional[int] = None
    num_trials: Optional[int] = None
    seed: Optional[int] = None
    sample_size: Optional[int] = None
    primary_metric_name: Optional[str] = None


def _extract_model_info(model_config: Dict[str, Any]) -> tuple[str, str]:
    """Extract model name and path from model config."""
    model_path = model_config.get("path")
    assert model_path, "experiment.json missing model.path"

    model_name = model_config.get("name") or model_path
    return model_name, model_path


def _extract_generation_params(model_config: Dict[str, Any]) -> dict:
    """Extract generation parameters from model config."""
    generation = model_config.get("generation", {})
    engine = model_config.get("engine", {})

    return {
        "temperature": generation.get("temperature"),
        "top_p": generation.get("top_p"),
        "max_completion_tokens": generation.get("max_completion_tokens"),
        "context_length": engine.get("context_length"),
    }


def _extract_cli_arg(run_args: list, flag: str) -> Optional[int]:
    """Extract an integer CLI argument value from run_args."""
    if flag not in run_args:
        return None

    idx = run_args.index(flag)
    if idx + 1 >= len(run_args):
        return None

    try:
        return int(run_args[idx + 1])
    except (ValueError, TypeError):
        return None


def _extract_metadata_args(
    experiment_data: Dict[str, Any],
) -> tuple[Optional[int], Optional[int], Optional[int]]:
    """Extract CLI arguments from metadata."""
    metadata = experiment_data.get("metadata", {})
    run_args = metadata.get("run_args", [])
    return (
        _extract_cli_arg(run_args, "--num-trials"),
        _extract_cli_arg(run_args, "--seed"),
        _extract_cli_arg(run_args, "--sample-size"),
    )


def _extract_n_shot_values(
    benchmark_spec: BenchmarkSpec,
) -> tuple[Optional[int], Optional[int]]:
    """Extract n_shot numerator and denominator from BenchmarkSpec."""
    if benchmark_spec.n_shot and benchmark_spec.n_shot.numerator > 0:
        return benchmark_spec.n_shot.numerator, benchmark_spec.n_shot.denominator
    if benchmark_spec.n_shot and benchmark_spec.n_shot.numerator == 0:
        return 0, 1
    return None, None


def _extract_primary_metric(experiment_data: Dict[str, Any]) -> Optional[str]:
    """Extract primary_metric from benchmark_config metadata."""
    benchmark_config = experiment_data.get("benchmark_config", {})
    benchmark_metadata = benchmark_config.get("metadata", {})
    return benchmark_metadata.get("primary_metric")


def parse_experiment_config(
    experiment_data: Dict[str, Any],
) -> ExperimentConfig:
    """
    Parse experiment.json data into a structured configuration object.

    Extracts key configuration parameters from FAITH's experiment.json
    including model generation settings, benchmark parameters, and run metadata.

    Args:
        experiment_data: Raw dictionary from experiment.json

    Returns:
        ExperimentConfig with extracted parameters

    Raises:
        AssertionError: If required fields (model.path, benchmark.name) are missing
    """
    exp_params = experiment_data.get("experiment_params", {})
    model_config = exp_params.get("model", {})
    bench_config = exp_params.get("benchmark", {})

    model_name, model_path = _extract_model_info(model_config)
    benchmark_spec = BenchmarkSpec.from_dict(bench_config)
    gen_params = _extract_generation_params(model_config)
    num_trials, seed, sample_size = _extract_metadata_args(experiment_data)
    num_shots, num_shots_pool_size = _extract_n_shot_values(benchmark_spec)
    primary_metric_name = _extract_primary_metric(experiment_data)

    return ExperimentConfig(
        model_name=model_name,
        model_path=model_path,
        benchmark_name=benchmark_spec.name,
        temperature=gen_params["temperature"],
        top_p=gen_params["top_p"],
        max_completion_tokens=gen_params["max_completion_tokens"],
        context_length=gen_params["context_length"],
        generation_mode=str(benchmark_spec.generation_mode),
        prompt_format=str(benchmark_spec.prompt_format),
        num_shots=num_shots,
        num_shots_pool_size=num_shots_pool_size,
        num_trials=num_trials,
        seed=seed,
        sample_size=sample_size,
        primary_metric_name=primary_metric_name,
    )


def _flatten_metrics(obj: Any, prefix: str = "") -> List[tuple[str, float]]:
    """
    Recursively flatten nested metrics dictionary.

    Args:
        obj: Dictionary or value to flatten
        prefix: Current key prefix for nested keys

    Returns:
        List of (metric_name, metric_value) tuples

    Examples:
        >>> _flatten_metrics({"accuracy": {"mean": 0.85, "std": 0.02}})
        [('accuracy.mean', 0.85), ('accuracy.std', 0.02)]
    """
    results = []
    if isinstance(obj, dict):
        for key, value in obj.items():
            new_key = f"{prefix}.{key}" if prefix else key
            if isinstance(value, (int, float)):
                results.append((new_key, float(value)))
            elif isinstance(value, dict):
                results.extend(_flatten_metrics(value, new_key))
    return results


def _derive_file_uri(path: str | Path) -> str:
    """
    Derive a canonical URI string from a file path.

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
    model_key: str,
    source_uri: str,
    metrics_file_uri: str,
    experiment_config: ExperimentConfig,
    ingest_time: str,
) -> Dict[str, Any]:
    """Build a single BigQuery metric record."""
    metric_unit = "ratio" if 0 <= metric_value <= 1 else None
    is_primary = (
        (metric_name == experiment_config.primary_metric_name)
        if experiment_config.primary_metric_name
        else None
    )

    return {
        "model_key": model_key,
        "source_uri": source_uri,
        "benchmark": experiment_config.benchmark_name,
        "metric_name": metric_name,
        "metric_value": metric_value,
        "metric_unit": metric_unit,
        "is_primary": is_primary,
        "metrics_file_uri": metrics_file_uri,
        "faith_version": faith_version,
        "temperature": experiment_config.temperature,
        "top_p": experiment_config.top_p,
        "max_completion_tokens": experiment_config.max_completion_tokens,
        "context_length": experiment_config.context_length,
        "generation_mode": experiment_config.generation_mode,
        "prompt_format": experiment_config.prompt_format,
        "num_shots": experiment_config.num_shots,
        "num_shots_pool_size": experiment_config.num_shots_pool_size,
        "num_trials": experiment_config.num_trials,
        "seed": experiment_config.seed,
        "sample_size": experiment_config.sample_size,
        "ingest_time": ingest_time,
    }


def parse_metrics_data(
    metrics_data: Dict[str, Any],
    metrics_file_uri: str,
    experiment_config: ExperimentConfig,
    model_key: Optional[str] = None,
    source_uri: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Parse metrics.json data into normalized BigQuery records.

    Flattens nested metrics and enriches with experiment configuration.

    Args:
        metrics_data: Raw dictionary from metrics.json
        metrics_file_uri: URI of the metrics file (unique identifier for the run)
        experiment_config: Parsed experiment configuration
        model_key: Optional model name override
        source_uri: Optional model path override

    Returns:
        List of metric records (one dict per metric)

    Raises:
        AssertionError: If 'stats' field is missing or invalid
    """
    final_model_key = (
        model_key if model_key is not None else experiment_config.model_name
    )
    final_source_uri = (
        source_uri if source_uri is not None else experiment_config.model_path
    )

    stats = metrics_data.get("stats", {})
    assert isinstance(stats, dict), "metrics.json missing or invalid 'stats' dict"

    flattened = _flatten_metrics(stats)
    ingest_time = datetime.utcnow().isoformat() + "Z"

    return [
        _build_metric_record(
            metric_name,
            metric_value,
            final_model_key,
            final_source_uri,
            metrics_file_uri,
            experiment_config,
            ingest_time,
        )
        for metric_name, metric_value in flattened
    ]


def parse_metrics_file(
    metrics_path: str | Path,
    experiment_config: ExperimentConfig,
    model_key: Optional[str] = None,
    source_uri: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Parse FAITH metrics.json file into normalized BigQuery records.

    Convenience function that combines loading and parsing metrics.

    Args:
        metrics_path: Path to metrics.json (local file or GCS URI)
        experiment_config: Parsed experiment configuration
        model_key: Optional model name override
        source_uri: Optional full model path/URI

    Returns:
        List of metric records (one dict per metric)

    Raises:
        FileNotFoundError: If metrics.json doesn't exist
        AssertionError: If required fields are missing
    """
    # Load metrics file
    with DatastoreContext.from_path(str(metrics_path)) as ds:
        metrics_data = read_json_file(ds.pull())

    # Use metrics_file_uri as the unique identifier for this run
    # This is truly unique as it includes model + benchmark + gen_params
    metrics_file_uri = _derive_file_uri(metrics_path)

    # Parse metrics data
    return parse_metrics_data(
        metrics_data,
        metrics_file_uri,
        experiment_config,
        model_key,
        source_uri,
    )
