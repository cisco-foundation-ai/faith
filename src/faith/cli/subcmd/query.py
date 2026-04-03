# Copyright 2025 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

"""Functions to execute queries from benchmarks on models and collect their responses."""

import itertools
import logging
import os
import subprocess
import sys
import time
from collections.abc import Iterable, Iterator, Sequence
from datetime import datetime
from functools import partial
from pathlib import Path
from zoneinfo import ZoneInfo

from tqdm import tqdm

from faith import __version__
from faith._internal.functools.compose import compose
from faith._internal.io.json import read_logs_from_json, write_as_json
from faith._internal.io.logging import LoggingTransform
from faith._internal.io.store import Store
from faith._internal.iter.mux import MuxTransform
from faith._internal.iter.transform import DevNullReducer, IdentityTransform
from faith._internal.multiprocessing.gpu_scheduling import (
    GPUJob,
    run_gpu_jobs_in_parallel,
)
from faith._types.model.engine import ModelEngine
from faith._types.model.generation import GenerationMode
from faith._types.model.spec import ModelSpec
from faith._types.record.sample import RecordStatus, SampleRecord
from faith.benchmark.listing import choices_to_benchmarks, find_benchmarks
from faith.experiment.experiment import BenchmarkExperiment
from faith.experiment.params import DataSamplingParams, ExperimentParams
from faith.model.factory import create_model
from faith.model.resolver import ResolvedModelPath
from faith.record_pipelines.formatting import SampleFormatter
from faith.record_pipelines.hashing import HashModelDataTransform
from faith.record_pipelines.params import RecordHandlingParams
from faith.record_pipelines.prediction import model_querier
from faith.record_pipelines.reconciliation import record_reconciler
from faith.record_pipelines.sorting import SortByTransform

logger = logging.getLogger(__name__)

# The maximum number of logit tokens to be returned by the model.
# Set to -1 to generate logits for all tokens. This may result in massive records.
MAX_LOGITS = 100


def get_command_output(command: str) -> str:
    """Execute a shell command and return its output."""
    result = subprocess.run(
        command,
        shell=True,
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip()


def current_timestamp() -> str:
    """Get the current timestamp as a string in the 'America/Los_Angeles' timezone."""
    return datetime.now(tz=ZoneInfo("America/Los_Angeles")).strftime(
        "%Y-%m-%d %H:%M:%S %Z"
    )


def read_trial_log(trial_log_path: Path) -> Iterable[SampleRecord]:
    """Read the trial log records from the given path."""
    if trial_log_path.exists():
        return [SampleRecord.from_dict(d) for d in read_logs_from_json(trial_log_path)]
    return []  # No records if the log file doesn't exist yet.


def _run_single_model(
    model_spec: ModelSpec,
    exp_params: ExperimentParams,
    sampling_params: DataSamplingParams,
    record_params: RecordHandlingParams,
    datastore: Store,
) -> Iterator[Path]:
    """
    Run benchmarks for a single model.

    Args:
        model_spec: Specification of the model to run, including path, engine,
            and generation parameters.
        exp_params: Experiment parameters
        sampling_params: Data sampling parameters
        record_params: Record handling parameters
        datastore: The datastore to use for storing experiment results.

    Yields:
        Path to experiment.json for each completed experiment
    """
    # pylint: disable=too-many-locals
    if model_spec.tokenizer is not None:
        logger.warning(
            "Using a tokenizer other than the model's tokenizer is not recommended and may lead to incorrect queries."
        )

    # Initialize the model from its annotated path.
    with ResolvedModelPath(model_spec) as name_or_path:
        try:
            model = create_model(
                model_spec.engine.engine_type,
                name_or_path=name_or_path,
                tokenizer_name_or_path=model_spec.tokenizer,
                num_gpus=model_spec.engine.num_gpus,
                seed=exp_params.initial_seed,
                context_len=model_spec.engine.context_length,
                num_log_probs=(
                    MAX_LOGITS
                    if exp_params.generation_mode == GenerationMode.LOGITS
                    else None
                ),
                reasoning_spec=model_spec.reasoning,
                **model_spec.engine.kwargs,
            )
        except BaseException as e:
            # Exceptions from model initialization may not be picklable, so we re-raise
            # them as RuntimeErrors with the original message.
            # pylint: disable-next=raise-missing-from
            raise RuntimeError(f"Failed to initialize model: {e}")

        prompt_formatter = model_spec.prompt_format
        assert (
            prompt_formatter in model.supported_formats
        ), f"Prompt format '{prompt_formatter}' is not supported by the model '{model.name_or_path}'. Supported formats: {model.supported_formats}"

        # Create the benchmark experiments.
        experiments = [
            BenchmarkExperiment(
                benchmark_path,
                exp_params.generation_mode,
                prompt_formatter,
                n_shot,
                model_spec.name,
                model_spec.generation,
                datastore,
                exp_params.num_trials,
                initial_seed=exp_params.initial_seed,
            )
            for benchmark_path in itertools.chain[Path](
                choices_to_benchmarks(exp_params.benchmark_names or []),
                [
                    benchmark_path
                    for p in exp_params.custom_benchmark_paths or []
                    for benchmark_path in find_benchmarks(p)
                ],
            )
            for n_shot in exp_params.n_shot
        ]
        for experiment in tqdm(
            experiments, desc="Benchmarks", unit=" benchmark", leave=False
        ):
            # Pull prior results from the experiment's datastore before the trials.
            exp_datastore = experiment.datastore
            exp_datastore.pull(raise_on_error=True)

            # Record the model parameters.
            experiment_start_time = time.perf_counter()
            run_record = {
                "metadata": {
                    "start_time": current_timestamp(),
                    "version": __version__,
                    "run_args": sys.argv,
                },
                "experiment_params": {
                    "benchmark": experiment.benchmark_spec.to_dict(),
                    "model": model_spec.to_dict(),
                },
                "benchmark_config": experiment.benchmark_config.to_dict(),
                "trial_records": {},
            }

            # Execute the trials of the experiment.
            for benchmark, trial_path in tqdm(
                experiment, desc="Trials", unit=" trial", leave=False
            ):
                trial_start_time = time.perf_counter()
                _ = (
                    benchmark.build_dataset(**sampling_params.to_dict()).iter_data()
                    >> SampleFormatter(benchmark, model.tokenizer)
                    >> record_reconciler(
                        read_trial_log(exp_datastore.path / trial_path),
                        record_params.replacement_strategy,
                        benchmark.generation_mode,
                    )
                    >> MuxTransform(
                        {
                            # Pass through fresh records unchanged.
                            RecordStatus.FRESH: IdentityTransform[SampleRecord](),
                            # For stale records, query the model and update the record.
                            RecordStatus.STALE: model_querier(
                                model, benchmark.generation_mode, model_spec.generation
                            ),
                        }
                    )
                    >> HashModelDataTransform()
                    >> SortByTransform[int]("data", "benchmark_sample_index")
                    >> LoggingTransform[SampleRecord](exp_datastore.path / trial_path)
                    >> DevNullReducer[SampleRecord]()
                )
                run_record["trial_records"][str(trial_path.parent)] = {
                    "runtime_seconds": time.perf_counter() - trial_start_time,
                    "trial_log_path": str(trial_path),
                }

            # Conclude the experiment.
            run_record["metadata"]["runtime_seconds"] = (
                time.perf_counter() - experiment_start_time
            )
            run_record["metadata"]["end_time"] = current_timestamp()

            # Save the record of the run.
            experiment_path = exp_datastore.path / "experiment.json"
            write_as_json(experiment_path, run_record)
            yield experiment_path
        del model


def run_experiment_queries(
    model_specs: Sequence[ModelSpec],
    exp_params: ExperimentParams,
    sampling_params: DataSamplingParams,
    record_params: RecordHandlingParams,
    datastore: Store,
    parallelize_models: bool = True,
) -> Iterator[Path]:
    """Query over given benchmarks for all specified models and generation parameters."""
    assert (
        exp_params.benchmark_names or exp_params.custom_benchmark_paths
    ), "Please specify at least one benchmark to query over using '--benchmarks' or '--custom-benchmark-paths'."

    # Running the function _run_single_model requires a blood-brain barrier.
    # All data passed to it must be picklable to allow for multiprocessing so its
    # arguments are limited to simple data structures that can be serialized.
    # Similarly the function must not rely on any global state that may not be shared
    # across processes, and its return value must be picklable.
    if parallelize_models:
        for model_spec in model_specs:
            assert model_spec.engine.engine_type == ModelEngine.VLLM, (
                f"Parallel model execution is only supported for the vLLM engine, "
                f"but model '{model_spec.path}' uses {model_spec.engine.engine_type}."
            )
        logger.info("Parallel execution enabled")
        for outcome in run_gpu_jobs_in_parallel(
            [
                GPUJob(
                    id=job_id,
                    fn=partial(
                        compose(list, _run_single_model),
                        model_spec=model_spec,
                        exp_params=exp_params,
                        sampling_params=sampling_params,
                        record_params=record_params,
                        datastore=datastore,
                    ),
                    need=model_spec.engine.num_gpus,
                )
                for model_spec in model_specs
                if (job_id := model_spec.name or str(model_spec.path)) is not None
            ]
        ):
            if outcome.result is not None:
                yield from outcome.result
            else:
                logger.error(
                    "Model job '%s' failed; error details:\n\n%s",
                    outcome.job_id,
                    outcome.error,
                )
    else:
        # Sequential model execution.
        for model_spec in tqdm(model_specs, desc="Models:", unit=" model"):
            yield from _run_single_model(
                model_spec=model_spec,
                exp_params=exp_params,
                sampling_params=sampling_params,
                record_params=record_params,
                datastore=datastore,
            )

    # Cleanup GPU processes after all models complete.
    _killall_gpu_processes()


def _killall_gpu_processes() -> None:
    """Manually shutdown the vLLM model processes on GPU."""
    # NOTE: This workaround kills ALL of the processes on GPU. Use responsibly!
    try:
        pids = get_command_output(
            "nvidia-smi --query-compute-apps=pid --format=csv,noheader"
        )
        if len(pids) > 0:
            os.system(f"sudo kill -9 {pids}")
            logger.info("Cleaned up GPU processes")
    except subprocess.CalledProcessError as e:
        logger.warning("Failed to cleanup GPU processes: %s", e)
