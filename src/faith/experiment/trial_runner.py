# Copyright 2025 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

"""Runners that execute a single trial of a benchmark.

A `TrialRunner` knows *how* to execute one trial of a given kind of benchmark,
writing its `SampleRecord`s to a trial-log path. This lives in the orchestration
layer (it composes record pipelines and queries the model), keeping the
`Benchmark` types themselves a passive description of *what* to run rather than
*how* to run it.
"""

from abc import ABC, abstractmethod
from pathlib import Path

from faith._internal.io.logging import LoggingTransform
from faith._internal.iter.mux import MuxTransform
from faith._internal.iter.transform import DevNullReducer, IdentityTransform
from faith._types.model.generation import GenParams
from faith._types.record.sample import RecordStatus, SampleRecord
from faith.benchmark.benchmark import Benchmark, QABenchmark
from faith.experiment.params import DataSamplingParams
from faith.model.base import BaseModel
from faith.record_pipelines.formatting import SampleFormatter
from faith.record_pipelines.hashing import HashModelDataTransform
from faith.record_pipelines.params import RecordHandlingParams
from faith.record_pipelines.prediction import model_querier
from faith.record_pipelines.reconciliation import record_reconciler
from faith.record_pipelines.sorting import SortByTransform
from faith.record_pipelines.trial_log import read_trial_log


class TrialRunner(ABC):
    """Executes one trial of a benchmark, writing its records to a trial log."""

    @abstractmethod
    def run_trial(
        self,
        benchmark: Benchmark,
        trial_log_path: Path,
        *,
        model: BaseModel,
        generation: GenParams,
        sampling_params: DataSamplingParams,
        record_params: RecordHandlingParams,
    ) -> None:
        """Run a single trial of `benchmark`, writing records to `trial_log_path`.

        Args:
            benchmark: The benchmark trial to execute.
            trial_log_path: Absolute path of the trial log to read/write.
            model: The initialized model to query.
            generation: Generation parameters for model queries.
            sampling_params: Data sampling parameters.
            record_params: Record reconciliation parameters.
        """


class QATrialRunner(TrialRunner):
    """Runs the standard question-answer pipeline for a QA benchmark trial.

    Builds the benchmark's dataset, formats each sample into a log record,
    reconciles against any prior trial log, queries the model for stale records,
    and writes the resulting records back to the trial log.
    """

    def run_trial(
        self,
        benchmark: Benchmark,
        trial_log_path: Path,
        *,
        model: BaseModel,
        generation: GenParams,
        sampling_params: DataSamplingParams,
        record_params: RecordHandlingParams,
    ) -> None:
        assert isinstance(
            benchmark, QABenchmark
        ), f"QATrialRunner requires a QABenchmark, got {type(benchmark).__name__}."
        _ = (
            benchmark.build_dataset(**sampling_params.to_dict()).iter_data()
            >> SampleFormatter(benchmark, model.tokenizer)
            >> record_reconciler(
                read_trial_log(trial_log_path),
                record_params.replacement_strategy,
                benchmark.generation_mode,
            )
            >> MuxTransform(
                {
                    # Pass through fresh records unchanged.
                    RecordStatus.FRESH: IdentityTransform[SampleRecord](),
                    # For stale records, query the model and update the record.
                    RecordStatus.STALE: model_querier(
                        model, benchmark.generation_mode, generation
                    ),
                }
            )
            >> HashModelDataTransform()
            >> SortByTransform[int]("data", "benchmark_sample_index")
            >> LoggingTransform[SampleRecord](trial_log_path)
            >> DevNullReducer[SampleRecord]()
        )


def trial_runner_for(benchmark: Benchmark) -> TrialRunner:
    """Select the appropriate TrialRunner for a benchmark.

    Currently only question-answer benchmarks are supported; agentic benchmarks
    will add their own runner here.
    """
    if isinstance(benchmark, QABenchmark):
        return QATrialRunner()
    raise ValueError(
        f"No trial runner available for benchmark type {type(benchmark).__name__}."
    )
