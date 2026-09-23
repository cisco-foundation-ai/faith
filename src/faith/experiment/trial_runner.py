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
from typing import Any

from faith._internal.iter.mux import MuxTransform
from faith._internal.iter.transform import DevNullReducer, IdentityTransform
from faith._types.model.generation import GenParams
from faith._types.record.sample import RecordStatus, SampleRecord
from faith.agentic.core import ReActLoop
from faith.agentic.orchestration import build_environment, build_tool_registry
from faith.benchmark.agentic import AgenticBenchmark
from faith.benchmark.benchmark import Benchmark, QABenchmark
from faith.experiment.params import DataSamplingParams
from faith.model.base import BaseModel
from faith.record_pipelines.agentic_execution import AgentLoopRunner
from faith.record_pipelines.agentic_formatting import TaskFormatter
from faith.record_pipelines.formatting import SampleFormatter
from faith.record_pipelines.params import RecordHandlingParams
from faith.record_pipelines.prediction import model_querier
from faith.record_pipelines.reconciliation import (
    agentic_record_reconciler,
    record_reconciler,
)
from faith.record_pipelines.trial_log import persist_records, read_trial_log


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
            >> persist_records(trial_log_path)
            >> DevNullReducer[SampleRecord]()
        )


class AgenticTrialRunner(TrialRunner):
    """Runs the agent loop over each task of an agentic benchmark trial.

    Loads the benchmark's tasks, formats each into a bare record, reconciles
    against any prior trial log, runs a fresh ReAct loop per stale task, and
    writes the records to the trial log. Scoring is deferred to eval time
    (AgentTrajectoryLogGrader), so the trial log holds raw trajectories, not
    scores.
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
        assert isinstance(benchmark, AgenticBenchmark), (
            f"AgenticTrialRunner requires an AgenticBenchmark, got "
            f"{type(benchmark).__name__}."
        )
        # The loop drives chat.completions.create interactively, so it needs the
        # raw client rather than the batch `query` surface QA uses. Agentic always
        # runs via the openai engine (see the CLI fork), so this is exact by design.
        assert (
            hasattr(model, "client")
            and hasattr(model, "api_max_attempts")
            and hasattr(model, "api_retry_sleep_secs")
        ), (
            f"AgenticTrialRunner requires a model exposing an OpenAI client, but "
            f"{type(model).__name__} does not; agentic runs must use the openai engine."
        )

        agentic_config = benchmark.agentic_config
        limits = agentic_config.limits
        generation_params: dict[str, Any] = {
            "temperature": generation.temperature,
            "top_p": generation.top_p,
            **generation.kwargs,
        }
        # -1 is the "omit, use the API default" sentinel (see APIBasedModel.query);
        # forwarding it verbatim would make the API reject reasoning configs like o3.
        if generation.max_completion_tokens > 0:
            generation_params["max_completion_tokens"] = (
                generation.max_completion_tokens
            )
        loop = ReActLoop(
            model_api=model.client,
            model=model.name_or_path,
            environment=build_environment(agentic_config),
            tool_registry=build_tool_registry(agentic_config.tools),
            max_tool_calls=limits.max_tool_calls,
            max_turns=limits.max_turns,
            max_consecutive_no_tool_calls=limits.max_consecutive_no_tool_calls,
            timeout_seconds=limits.timeout_seconds,
            system_prompt=agentic_config.system_prompt,
            # Retry patience is a transport setting configured on the model spec,
            # so the loop must not fall back to its own defaults here.
            api_max_attempts=model.api_max_attempts,
            api_retry_sleep_secs=model.api_retry_sleep_secs,
            generation_params=generation_params,
        )
        # Index tasks so the query stage can recover the Task to run for each
        # stale record (a stale record may be one read back from a prior log and
        # so carries no task); the index is the pipeline's ordered source too.
        tasks_by_index = dict(
            benchmark.load_tasks(sample_size=sampling_params.sample_size)
        )
        _ = (
            iter(tasks_by_index.items())
            >> TaskFormatter(benchmark.version, agentic_config.system_prompt)
            >> agentic_record_reconciler(
                read_trial_log(trial_log_path),
                record_params.replacement_strategy,
            )
            >> MuxTransform(
                {
                    # Pass through fresh records unchanged.
                    RecordStatus.FRESH: IdentityTransform[SampleRecord](),
                    # For stale records, run the agent loop and store its trajectory.
                    RecordStatus.STALE: AgentLoopRunner(loop, tasks_by_index),
                }
            )
            >> persist_records(trial_log_path)
            >> DevNullReducer[SampleRecord]()
        )


def trial_runner_for(benchmark: Benchmark) -> TrialRunner:
    """Select the appropriate TrialRunner for a benchmark."""
    if isinstance(benchmark, AgenticBenchmark):
        return AgenticTrialRunner()
    if isinstance(benchmark, QABenchmark):
        return QATrialRunner()
    raise ValueError(
        f"No trial runner available for benchmark type {type(benchmark).__name__}."
    )
