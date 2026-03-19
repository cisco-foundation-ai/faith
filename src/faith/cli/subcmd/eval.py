# Copyright 2025 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

"""Core functionality for computing aggregate metrics from benchmark trials."""

from collections.abc import ValuesView
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tqdm import tqdm

from faith._internal.io.json import read_json_file, read_logs_from_json, write_as_json
from faith._internal.io.logging import LoggingTransform
from faith._internal.iter.common import GetAttrTransform
from faith._internal.iter.transform import IdentityTransform
from faith._internal.metrics.aggregations import (
    agg_breakdown_counts,
    agg_trial_stats,
    is_breakdown_dict,
)
from faith._internal.records.sort import SortByTransform
from faith._internal.types.stats import MetricSummary
from faith._types.config.benchmark import BenchmarkConfig
from faith._types.config.patterns import AnswerFormat, Disambiguation, PatternDef
from faith._types.record.sample_record import SampleRecord
from faith._types.record.stats import StatsRecord
from faith.benchmark.benchmark import Benchmark, BenchmarkSpec
from faith.benchmark.load import load_benchmark


@dataclass(frozen=True)
class RecordHandlingParams:
    """Parameters defining the behavior of metrics computation and logging."""

    annotate_prediction_stats: bool
    recompute_stats: bool


def _agg_trials(tms: ValuesView[MetricSummary]) -> MetricSummary:
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


def compute_experiment_metrics(
    benchmark: Benchmark,
    trial_logs: dict[str, Path],
    *,
    annotate_prediction_stats: bool = False,
    **kwargs: Any,
) -> MetricSummary:
    """Compute a benchmark experiment's metrics from its trial logs."""
    per_trial_metrics = {
        trial_key: trial_metrics
        for trial_key, trial_log_filepath in tqdm(
            trial_logs.items(), desc="Processing trials", unit="trial", leave=False
        )
        if (
            trial_metrics := [
                SampleRecord.from_dict(d)
                for d in read_logs_from_json(trial_log_filepath)
            ]
            >> benchmark.log_grader(**kwargs)
            >> (
                (
                    SortByTransform[int]("data", "benchmark_sample_index")
                    | LoggingTransform[SampleRecord](trial_log_filepath)
                )
                if annotate_prediction_stats
                else IdentityTransform[SampleRecord]()
            )
            >> GetAttrTransform[SampleRecord, StatsRecord | None]("stats")
            >> benchmark.grade_aggregator()
        ).get("query_count", 0)
        > 0
    }
    return {
        "per_trial_metrics": per_trial_metrics,
        "stats": _agg_trials(per_trial_metrics.values()),
    }


def evaluate_experiment(
    experiment_path: Path,
    record_params: RecordHandlingParams,
    *,
    metrics_output_path: Path | None = None,
) -> MetricSummary:
    """Evaluate an experiment from trial logs at the given path."""
    experiment_summary = read_json_file(experiment_path)
    format_pattern = experiment_summary["experiment_params"]["model"].get(
        "response_pattern"
    )
    metrics = compute_experiment_metrics(
        load_benchmark(
            BenchmarkSpec.from_dict(
                experiment_summary["experiment_params"]["benchmark"]
            ),
            BenchmarkConfig.from_dict(experiment_summary["benchmark_config"]),
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
        model_format_config=(
            PatternDef(
                pattern=format_pattern or r"(?s).*",
                disambiguation=Disambiguation.MATCH_ALL,
                format_type=AnswerFormat.PROPER,
            )
            if format_pattern is not None
            else None
        ),
        recompute_stats=record_params.recompute_stats,
    )
    if metrics_output_path is not None:
        write_as_json(metrics_output_path, metrics)
    return metrics
