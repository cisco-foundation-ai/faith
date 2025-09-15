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
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Iterable, Iterator, Type
from zoneinfo import ZoneInfo

from tqdm import tqdm

from faith import __version__
from faith._internal.io.datastore import ReadOnlyDataContext
from faith._internal.io.json import write_as_json
from faith._internal.io.logging import LoggingTransform
from faith._internal.io.paths import canonical_segment
from faith._internal.iter.transform import DevNullReducer, IsoTransform
from faith._internal.types.flags import GenerationMode, PathWithAnnotations
from faith.benchmark.benchmark import Benchmark
from faith.benchmark.listing import choices_to_benchmarks
from faith.experiment.experiment import BenchmarkExperiment
from faith.experiment.params import DataSamplingParams, ExperimentParams
from faith.model.base import BaseModel, ChatResponse, GenerationError
from faith.model.params import EngineParams, GenParams

logger = logging.getLogger(__name__)

# The maximum number of logit tokens to be returned by the model.
# Set to -1 to generate logits for all tokens. This may result in massive records.
MAX_LOGITS = 100


def get_command_output(command: str) -> str:
    """Execute a shell command and return its output."""
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    return result.stdout.strip()


def current_timestamp() -> str:
    """Get the current timestamp as a string in the 'America/Los_Angeles' timezone."""
    return datetime.now(tz=ZoneInfo("America/Los_Angeles")).strftime(
        "%Y-%m-%d %H:%M:%S %Z"
    )


def query_over_benchmark(
    benchmark: Benchmark,
    sampling_params: DataSamplingParams,
    model: BaseModel,
    gen_params: GenParams,
) -> Iterable[dict[str, Any]]:
    """Query over a benchmark for the specified model and generation parameters."""
    bench_formatter = benchmark.formatter
    answer_leadin = None
    if benchmark.generation_mode in [GenerationMode.LOGITS, GenerationMode.NEXT_TOKEN]:
        assert (
            model.tokenizer is not None
        ), f"Model tokenizer required for {str(benchmark.generation_mode)}."
        answer_leadin = benchmark.answer_leadin(model.tokenizer)

    # Translate the answer symbols to the model's tokenizer's token IDs.
    answer_symbol_ids = {}
    if benchmark.generation_mode == GenerationMode.LOGITS:
        assert hasattr(
            benchmark, "answer_token_map"
        ), f"Model tokenizer required for {str(benchmark.generation_mode)}."
        assert (
            model.tokenizer is not None
        ), "Model tokenizer is required for logits generation."
        answer_symbol_ids = benchmark.answer_token_map(model.tokenizer)

    mode_map = {
        GenerationMode.LOGITS: _ModelMethod.LOGITS,
        GenerationMode.NEXT_TOKEN: _ModelMethod.NEXT_TOKEN,
        GenerationMode.CHAT_COMPLETION: _ModelMethod.GENERATION,
    }
    model_querier = mode_map[benchmark.generation_mode].create_transform(
        model=model,
        gen_params=gen_params,
    )

    return (
        {
            "metadata": {
                "version": benchmark.version,
                "data_hash": example.sha256(),
            },
            "data": example.to_dict(),
            "model_data": {
                "prompt": bench_formatter.render_conversation(example, answer_leadin),
                "answer_symbol_ids": answer_symbol_ids,
            },
        }
        for example in benchmark.build_dataset(**sampling_params.to_dict()).iter_data()  # type: ignore[arg-type]
    ) >> model_querier


class _PredictionTransform(IsoTransform[dict[str, Any]]):
    """Base class for prediction transforms that generate model outputs."""

    def __init__(self, model: BaseModel, gen_params: GenParams):
        """Initialize the prediction transform for a model."""
        super().__init__()
        self._model = model
        self._gen_params = gen_params


class _LogitsTransform(_PredictionTransform):
    """Transform for generating logits from a model."""

    def __init__(self, model: BaseModel, gen_params: GenParams):
        """Initialize the logits transform for the model."""
        super().__init__(model, gen_params)

    def __call__(self, iter: Iterable[dict[str, Any]]) -> Iterable[dict[str, Any]]:
        """Generate the next-token logits for each input in the `iter`."""
        inputs = list(iter)
        logit_responses = self._model.logits(
            inputs=[example["model_data"]["prompt"] for example in inputs],
            temperature=self._gen_params.temperature,
            top_p=self._gen_params.top_p,
            **self._gen_params.kwargs,
        )
        for record, logit_response in zip(inputs, logit_responses):
            if isinstance(logit_response, list):
                record["model_data"]["logits"] = [
                    [tp.to_dict() for tp in tok_dist] for tok_dist in logit_response
                ]
            elif isinstance(logit_response, GenerationError):
                record["model_data"]["error"] = logit_response.to_dict()
            yield record


class _NextTokenTransform(_PredictionTransform):
    """Transform for generating next token predictions from a model."""

    def __init__(self, model: BaseModel, gen_params: GenParams):
        """Initialize the next token transform for the model."""
        super().__init__(model, gen_params)

    def __call__(self, iter: Iterable[dict[str, Any]]) -> Iterable[dict[str, Any]]:
        """Generate next token predictions for each input in the `iter`."""
        inputs = list(iter)
        responses = self._model.next_token(
            inputs=[example["model_data"]["prompt"] for example in inputs],
            temperature=self._gen_params.temperature,
            top_p=self._gen_params.top_p,
            **self._gen_params.kwargs,
        )
        for record, response in zip(inputs, responses):
            if isinstance(response, ChatResponse):
                record["model_data"]["next_token"] = response.to_dict()
            elif isinstance(response, GenerationError):
                record["model_data"]["error"] = response.to_dict()
            yield record


class _GenerationTransform(_PredictionTransform):
    """Transform for generating chat completions from a model."""

    def __init__(self, model: BaseModel, gen_params: GenParams):
        """Initialize the generation transform for the model."""
        super().__init__(model, gen_params)

    def __call__(self, iter: Iterable[dict[str, Any]]) -> Iterable[dict[str, Any]]:
        """Generate chat completion responses for each input in the `iter`."""
        inputs = list(iter)
        responses = self._model.query(
            inputs=[example["model_data"]["prompt"] for example in inputs],
            temperature=self._gen_params.temperature,
            max_completion_tokens=self._gen_params.max_completion_tokens,
            top_p=self._gen_params.top_p,
            **self._gen_params.kwargs,
        )
        for record, response in zip(inputs, responses):
            if isinstance(response, ChatResponse):
                record["model_data"]["chat_comp"] = response.to_dict()
            elif isinstance(response, GenerationError):
                record["model_data"]["error"] = response.to_dict()
            yield record


class _ModelMethod(Enum):
    """Enumeration of model methods for generating predictions.

    Each enum value corresponds to a specific generative method of the model,
    allowing for flexible handling of different prediction types.
    """

    LOGITS = (_LogitsTransform,)
    NEXT_TOKEN = (_NextTokenTransform,)
    GENERATION = (_GenerationTransform,)

    def __init__(self, transform_cls: Type[_PredictionTransform]) -> None:
        """Initialize the model method with the corresponding transform class."""
        self._transform_cls = transform_cls

    def create_transform(
        self, model: BaseModel, gen_params: GenParams
    ) -> IsoTransform[dict[str, Any]]:
        """Create an instance of the transform for the model and generation parameters."""
        return self._transform_cls(model, gen_params)


def run_experiment_queries(
    exp_params: ExperimentParams,
    sampling_params: DataSamplingParams,
    engine_params: EngineParams,
    gen_params: GenParams,
    datastore_path: Path,
) -> Iterator[Path]:
    """Query over given benchmarks for all specified models and generation parameters."""
    assert (
        exp_params.benchmark_names or exp_params.custom_benchmark_paths
    ), "Please specify at least one benchmark to query over using '--benchmarks' or '--custom-benchmark-paths'."

    for annotated_model_path in tqdm(
        exp_params.model_paths, desc="Models:", unit=" model"
    ):
        # Initialize the model from its annotated path.
        is_model_a_file = annotated_model_path.get_value("is_file")
        with ReadOnlyDataContext(
            annotated_model_path.raw_path, is_model_a_file
        ) as model_path:
            model_name = annotated_model_path.get_value("name")
            if model_name is None:
                model_name = canonical_segment(str(model_path))
            model_tokenizer = annotated_model_path.get_value("tokenizer")
            if model_tokenizer is not None:
                logger.warning(
                    "Using a tokenizer other than the model's tokenizer is not recommended and may lead to incorrect queries."
                )

            model = engine_params.engine_type.create_model(
                name_or_path=str(model_path),
                tokenizer_name_or_path=model_tokenizer,
                num_gpus=engine_params.num_gpus,
                seed=exp_params.initial_seed,
                context_len=engine_params.context_length,
                num_log_probs=(
                    MAX_LOGITS
                    if exp_params.generation_mode == GenerationMode.LOGITS
                    else None
                ),
                **engine_params.kwargs,
            )

            prompt_formatter = exp_params.prompt_format
            assert (
                prompt_formatter in model.supported_formats
            ), f"Prompt format '{prompt_formatter}' is not supported by the model '{model.name_or_path}'. Supported formats: {model.supported_formats}"

            # Create the benchmark experiments.
            # Note: experiments is instantiated as a list so all are validated before iterating.
            experiments = [
                BenchmarkExperiment(
                    benchmark,
                    exp_params.generation_mode,
                    prompt_formatter,
                    n_shot,
                    model_name,
                    gen_params,
                    datastore_path,
                    exp_params.num_trials,
                    initial_seed=exp_params.initial_seed,
                )
                for benchmark in itertools.chain[str | PathWithAnnotations](
                    choices_to_benchmarks(exp_params.benchmark_names or []),
                    exp_params.custom_benchmark_paths or [],
                )
                for n_shot in exp_params.n_shot
            ]
            for experiment in tqdm(
                experiments, desc="Benchmarks", unit=" benchmark", leave=False
            ):
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
                        "model": {
                            "name": model_name,
                            "path": str(model_path),
                            "engine": engine_params.to_dict(),
                            "generation": gen_params.to_dict(),
                        },
                    },
                    "benchmark_config": experiment.benchmark_config,
                    "trial_records": {},
                }

                # Execute the trials of the experiment.
                for benchmark, trial_path in tqdm(
                    experiment, desc="Trials", unit=" trial", leave=False
                ):
                    trial_start_time = time.perf_counter()
                    (
                        query_over_benchmark(
                            benchmark,
                            sampling_params,
                            model,
                            gen_params,
                        )
                        >> LoggingTransform[dict[str, Any]](
                            experiment.experiment_dir / trial_path
                        )
                        >> DevNullReducer[dict[str, Any]]()
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
                experiment_path = experiment.experiment_dir / "experiment.json"
                write_as_json(experiment_path, run_record)
                yield experiment_path
            del model

    # Manually shutdown the vLLM model and exit as it hangs sometimes
    # NOTE: This workaround kills ALL of the processes on GPU. Use responsibly!
    pids = get_command_output(
        "nvidia-smi --query-compute-apps=pid --format=csv,noheader"
    )
    if len(pids) > 0:
        os.system(f"sudo kill -9 {pids}")
