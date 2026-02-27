# Copyright 2025 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

"""Core functionality for computing aggregate metrics from benchmark trials."""

from collections.abc import ValuesView
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tqdm import tqdm

from faith._internal.config.model_response import model_response_format_config
from faith._internal.io.json import read_json_file, write_as_json
from faith._internal.io.logging import LoggingTransform
from faith._internal.iter.transform import IdentityTransform
from faith._internal.metrics.aggregations import (
    agg_breakdown_counts,
    agg_trial_stats,
    is_breakdown_dict,
)
from faith._internal.records.io import load_records_from_json
from faith._internal.records.types import Record
from faith.benchmark.benchmark import Benchmark, BenchmarkSpec
from faith.benchmark.load import load_benchmark


@dataclass
class RecordHandlingParams:
    """Parameters defining the behavior of metrics computation and logging."""

    annotate_prediction_stats: bool
    recompute_stats: bool


def _agg_trials(tms: ValuesView[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate statistics from a collection of trial metrics dictionaries `tms`."""
    num_trials = len(tms)
    total_queries = sum(tm.get("query_count", 0) for tm in tms)
    return (
        {
            "num_trials": num_trials,
            "queries": {
                "total": total_queries,
                "mean": total_queries / num_trials if num_trials > 0 else float("nan"),
            },
        }
        | {
            stat: agg_trial_stats(pts)
            for stat in {k for tm in tms for k in tm.keys() if not k.endswith("_count")}
            if len(pts := [tm[stat] for tm in tms if stat in tm]) > 0
        }
        | {
            stat.removesuffix("_count"): {
                "total": agg_breakdown_counts(pts, factor=1.0),
                "mean_per_trial": agg_breakdown_counts(pts, factor=1.0 / num_trials),
                "mean_per_query": agg_breakdown_counts(pts, factor=1.0 / total_queries),
            }
            for stat in {
                k
                for tm in tms
                for k in tm.keys()
                if k.endswith("_count") and is_breakdown_dict(tm[k])
            }
            if len(pts := [tm[stat] for tm in tms if stat in tm]) > 0
        }
    )


def evaluate_experiment_logs(
    benchmark: Benchmark,
    trial_logs: dict[str, Path],
    *,
    annotate_prediction_stats: bool = False,
    **kwargs: Any,
) -> dict[str, Any]:
    """Evaluate an experiment from trial logs using a log grader and grade aggregator."""
    per_trial_metrics = {
        trial_key: trial_metrics
        for trial_key, trial_log_filepath in tqdm(
            trial_logs.items(), desc="Processing trials", unit="trial", leave=False
        )
        if (
            trial_metrics := load_records_from_json(trial_log_filepath)
            >> benchmark.log_grader(**kwargs)
            >> (
                LoggingTransform(trial_log_filepath)
                if annotate_prediction_stats
                else IdentityTransform[Record]()
            )
            >> benchmark.grade_aggregator()
        ).get("query_count", 0)
        > 0
    }
    return {
        "per_trial_metrics": per_trial_metrics,
        "stats": _agg_trials(per_trial_metrics.values()),
    }


def compute_experiment_metrics(
    experiment_path: Path,
    record_params: RecordHandlingParams,
) -> dict[str, Any]:
    """Compute metrics for the experiment at the given path."""
    experiment_summary = read_json_file(experiment_path)
    experiment_metrics = evaluate_experiment_logs(
        load_benchmark(
            BenchmarkSpec.from_dict(
                experiment_summary["experiment_params"]["benchmark"]
            ),
            experiment_summary["benchmark_config"],
        ),
        {
            trial_key: trial_log_filepath
            for trial_key, trial_metadata in experiment_summary["trial_records"].items()
            if (
                trial_log_filepath := experiment_path.parent
                / trial_metadata["trial_log_path"]
            ).exists()
        },
        annotate_prediction_stats=record_params.annotate_prediction_stats,
        model_format_config=model_response_format_config(
            experiment_summary["experiment_params"]["model"].get(
                "response_pattern", None
            )
        ),
        recompute_stats=record_params.recompute_stats,
    )
    write_as_json(experiment_path.parent / "metrics.json", experiment_metrics)
    return experiment_metrics
