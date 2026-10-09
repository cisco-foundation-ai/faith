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
from collections.abc import Iterator, Sequence
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from tqdm import tqdm

from faith import __version__
from faith._internal.functools.compose import compose
from faith._internal.io.json import write_as_json
from faith._internal.io.store import Store
from faith._internal.multiprocessing.gpu_scheduling import (
    GPUJob,
    run_gpu_jobs_in_parallel,
)
from faith._types.benchmark.sample_ratio import SampleRatio
from faith._types.model.engine import REDACTED, ModelEngine, is_secret_name
from faith._types.model.generation import GenerationMode, GenParams
from faith._types.model.prompt import PromptFormatter
from faith._types.model.spec import ModelSpec
from faith.agentic.core.react_loop import validate_agentic_generation_params
from faith.benchmark.agentic import (
    AGENTIC_GENERATION_MODE,
    AGENTIC_N_SHOT,
    AGENTIC_PROMPT_FORMAT,
)
from faith.benchmark.config import load_config_from_path
from faith.benchmark.listing import choices_to_benchmarks, find_benchmarks
from faith.experiment.experiment import BenchmarkExperiment
from faith.experiment.params import DataSamplingParams, ExperimentParams
from faith.experiment.trial_runner import trial_runner_for
from faith.model.api_retry import completion_with_retries, describe_api_error
from faith.model.factory import create_model
from faith.model.resolver import ResolvedModelPath
from faith.record_pipelines.params import RecordHandlingParams

logger = logging.getLogger(__name__)

# The maximum number of logit tokens to be returned by the model.
# Set to -1 to generate logits for all tokens. This may result in massive records.
MAX_LOGITS = 100

# Repeating an agentic trial only averages over agent-trajectory nondeterminism,
# and each trial is expensive (per task: a container plus tens of model calls), so
# warn past this many trials that the cost scales with num_tasks * num_trials.
_AGENTIC_TRIAL_WARN_THRESHOLD = 5

# Room for a reasoning model to think its way to the probe's tool call. Far below
# an agentic run's own budget, but a small value here would make the check itself
# the failure it reports.
_TOOL_PROBE_MAX_TOKENS = 1024

_TOOL_PROBE_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "probe",
            "description": "Probe tool.",
            "parameters": {
                "type": "object",
                "properties": {"x": {"type": "string"}},
                "required": ["x"],
            },
        },
    }
]

# A 400 naming one of these is the endpoint declining tool calling, which is the
# verdict the probe is here to reach; any other parameter name means faith sent
# something the model rejects, which is faith's bug to report as its own.
_TOOL_PROBE_CAPABILITY_PARAMS = frozenset({"tools", "tool_choice"})

_TOOL_PROBE_REMEDY = (
    "Serve the model with tool calling enabled (for vLLM: --enable-auto-tool-choice "
    "and a --tool-call-parser suited to the model), and check that the model's chat "
    'template renders tools, tool_calls, and role: "tool" — a template without them '
    "cannot emit a tool call under any parser."
)


def _is_agentic_benchmark(benchmark_path: Path) -> bool:
    """Whether the benchmark at `benchmark_path` is agentic.

    Only called on paths that came from benchmark listing, which loads every config
    unguarded, so a malformed one has already raised. The reload is a cache hit.
    """
    return load_config_from_path(benchmark_path).agentic_config is not None


def _requested_benchmark_paths(exp_params: ExperimentParams) -> Sequence[Path]:
    """The benchmark paths requested by name and by custom path, in that order.

    De-duplicated by resolved path, since overlapping `--custom-benchmarks` roots
    would otherwise run the same benchmark twice, at full cost for an agentic one.
    """
    requested = itertools.chain[Path](
        choices_to_benchmarks(exp_params.benchmark_names or []),
        [
            benchmark_path
            for p in exp_params.custom_benchmark_paths or []
            for benchmark_path in find_benchmarks(p)
        ],
    )
    return list({p.resolve(): p for p in requested}.values())


def _requests_agentic_benchmark(exp_params: ExperimentParams) -> bool:
    """Whether any requested benchmark is confirmably agentic."""
    return any(
        _is_agentic_benchmark(benchmark_path)
        for benchmark_path in _requested_benchmark_paths(exp_params)
    )


def _build_experiments(
    benchmark_paths: Sequence[Path],
    exp_params: ExperimentParams,
    prompt_formatter: PromptFormatter,
    model_spec: ModelSpec,
    datastore: Store,
) -> Iterator[BenchmarkExperiment]:
    """Build the experiments for one model, forking QA and agentic per benchmark.

    QA benchmarks fan out over every requested `--n-shot` value; agentic benchmarks
    ignore that cross-product (few-shot is meaningless for a tool-using agent) and
    pin their spec to the agentic constants, yielding exactly one experiment each.
    """
    for benchmark_path in benchmark_paths:
        if _is_agentic_benchmark(benchmark_path):
            generation_mode = AGENTIC_GENERATION_MODE
            prompt_format = AGENTIC_PROMPT_FORMAT
            n_shots: Sequence[SampleRatio] = [AGENTIC_N_SHOT]
        else:
            generation_mode = exp_params.generation_mode
            prompt_format = prompt_formatter
            n_shots = exp_params.n_shot

        for n_shot in n_shots:
            yield BenchmarkExperiment(
                benchmark_path,
                generation_mode,
                prompt_format,
                n_shot,
                model_spec.name,
                model_spec.generation,
                datastore,
                exp_params.num_trials,
                initial_seed=exp_params.initial_seed,
            )


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


def _is_secret_flag(arg: str) -> bool:
    """Whether `arg` names a credential, ignoring flag dashes and dash-vs-underscore."""
    return is_secret_name(arg.lstrip("-").replace("-", "_"))


def redact_argv(argv: Sequence[str]) -> list[str]:
    """Mask credential values in a command line, keeping every flag and key visible.

    A credential reaches argv either within a single element (`api_key=sk-...`) or as
    the element following its flag (`--api-key sk-...`), so both shapes are masked.
    """
    redacted: list[str] = []
    mask_next = False
    for arg in argv:
        if mask_next:
            redacted.append(REDACTED)
            mask_next = False
            continue
        key, sep, _ = arg.partition("=")
        if sep and _is_secret_flag(key):
            redacted.append(f"{key}{sep}{REDACTED}")
        else:
            redacted.append(arg)
            mask_next = not sep and _is_secret_flag(arg)
    return redacted


def current_timestamp() -> str:
    """Get the current timestamp as a string in the 'America/Los_Angeles' timezone."""
    return datetime.now(tz=ZoneInfo("America/Los_Angeles")).strftime(
        "%Y-%m-%d %H:%M:%S %Z"
    )


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

        # Checked here rather than beside the other agentic guards because it needs
        # the live client, which only exists once the model is built.
        if _requests_agentic_benchmark(exp_params):
            _check_tool_calling(model, model_spec.generation)

        # Create the benchmark experiments, forking QA vs agentic per benchmark.
        experiments = list(
            _build_experiments(
                _requested_benchmark_paths(exp_params),
                exp_params,
                prompt_formatter,
                model_spec,
                datastore,
            )
        )
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
                    "run_args": redact_argv(sys.argv),
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
                trial_runner_for(benchmark).run_trial(
                    benchmark,
                    exp_datastore.path / trial_path,
                    model=model,
                    generation=model_spec.generation,
                    sampling_params=sampling_params,
                    record_params=record_params,
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


def _check_agentic_compat(
    model_specs: Sequence[ModelSpec],
    exp_params: ExperimentParams,
    parallelize_models: bool,
) -> None:
    """Reject agentic runs in configurations that can only fail later (or worse).

    Every check is fail-fast: they fire before any experiment runs, so a mixed
    QA+agentic run doesn't burn the QA benchmarks' time and cost before the agentic
    one hits an incompatibility.
    """
    if not _requests_agentic_benchmark(exp_params):
        return

    # Agentic runs stay single-process: their Docker/OpenAI clients aren't picklable
    # and can't cross the _run_single_model multiprocessing barrier. Reject the flag
    # here with a clear error rather than the confusing vLLM-only assertion downstream.
    if parallelize_models:
        raise ValueError(
            "--experimental-parallelize-models is incompatible with agentic "
            "benchmarks, which run single-process. Rerun without the flag."
        )

    # Any run containing an agentic benchmark is openai-only, including the specs that
    # would only serve its QA benchmarks: agentic drives chat.completions.create
    # interactively, and per-(model, benchmark) engine selection is not supported.
    # Reject up front rather than after a mixed run's QA benchmarks have executed.
    for model_spec in model_specs:
        if model_spec.engine.engine_type != ModelEngine.OPENAI:
            raise ValueError(
                f"Agentic benchmarks require the openai engine, but model "
                f"'{model_spec.path}' uses {model_spec.engine.engine_type}. "
                f"Rerun with an openai-engine model."
            )

    # The agentic experiments pin these two, but the openai model is still built from
    # the user's values, and it rejects both alternatives at init (logits and the base
    # prompt format are unsupported), so the run would die before any task ran.
    if exp_params.generation_mode != AGENTIC_GENERATION_MODE:
        raise ValueError(
            f"Agentic benchmarks require --generation-mode="
            f"{AGENTIC_GENERATION_MODE}, but got {exp_params.generation_mode}."
        )
    for model_spec in model_specs:
        if model_spec.prompt_format != AGENTIC_PROMPT_FORMAT:
            raise ValueError(
                f"Agentic benchmarks require the {AGENTIC_PROMPT_FORMAT} prompt "
                f"format, but model '{model_spec.path}' uses "
                f"{model_spec.prompt_format}."
            )


def _rejected_request_param(error: Exception) -> str | None:
    """The generation parameter a 4xx blames faith for sending, if any.

    Duck-typed rather than caught as `openai.BadRequestError`, since openai is an
    optional extra: its exceptions carry the error body's `param`. A rejected
    `tools` or `tool_choice` is excluded because that *is* the capability the probe
    exists to measure, not a parameter faith got wrong.
    """
    status = getattr(error, "status_code", None)
    if not isinstance(status, int) or not 400 <= status < 500:
        return None
    param = getattr(error, "param", None)
    if not isinstance(param, str) or param in _TOOL_PROBE_CAPABILITY_PARAMS:
        return None
    return param or None


def _check_tool_calling(model: Any, generation: GenParams) -> None:
    """Reject a model whose endpoint cannot emit tool calls, before any task runs.

    The agentic loop reads its actions from `message.tool_calls` and nowhere else, so
    an endpoint that never populates it yields a full run of trajectories that all
    stop with NO_TOOL_CALLS, a well-formed metrics.json, and a score of 0.0 — a
    broken setup that looks like a measurement rather than an error.

    The probe requires a tool call rather than merely offering one: a healthy model
    may decline an optional throwaway tool, which is indistinguishable from being
    unable to call it.

    It is assembled from `generation` through the same path the run itself uses, so
    a model the run can serve cannot be rejected by the check that gates it.
    """
    probe_params = generation.as_request_params()
    validate_agentic_generation_params(probe_params)
    probe_params["max_completion_tokens"] = _TOOL_PROBE_MAX_TOKENS
    try:
        response = completion_with_retries(
            model.client,
            max_attempts=model.api_max_attempts,
            retry_sleep_secs=model.api_retry_sleep_secs,
            model=model.name_or_path,
            messages=[{"role": "user", "content": "Call the probe tool with x='hi'."}],
            tools=_TOOL_PROBE_TOOLS,
            tool_choice="required",
            **probe_params,
        )
    except Exception as e:
        if bad_param := _rejected_request_param(e):
            raise ValueError(
                f"Probing '{model.name_or_path}' for tool calls failed because faith "
                f"sent a parameter the model does not support: {bad_param}. This is a "
                f"faith bug, not a problem with the endpoint's tool support: {describe_api_error(e)}"
            ) from None
        raise ValueError(
            f"Agentic benchmarks require an endpoint that can emit tool calls, but "
            f"probing '{model.name_or_path}' for one failed: {describe_api_error(e)}. {_TOOL_PROBE_REMEDY}"
        ) from None

    choice = response.choices[0]
    if choice.message.tool_calls:
        return

    # Truncation says nothing about tool support: the model may simply not have
    # reached its call within the probe's budget, which is far below the run's own.
    # Rejecting the run here is what wrongly failed a correctly configured server.
    if choice.finish_reason == "length":
        logger.warning(
            "Could not confirm '%s' emits tool calls: the probe was truncated at "
            "%d tokens before producing one. Continuing, since an agentic run's own "
            "budget is larger.",
            model.name_or_path,
            _TOOL_PROBE_MAX_TOKENS,
        )
        return

    raise ValueError(
        f"Agentic benchmarks require an endpoint that can emit tool calls, but "
        f"'{model.name_or_path}' returned none when required to "
        f"(finish_reason={choice.finish_reason}). {_TOOL_PROBE_REMEDY}"
    )


def _warn_ignored_agentic_nshot(exp_params: ExperimentParams) -> None:
    """Warn once when an agentic run drops a requested --n-shot fan-out.

    Few-shot examples are meaningless for a tool-using agent, so agentic experiments
    pin n-shot=0. In a mixed run the requested values still apply to the QA
    benchmarks, hence "for agentic benchmarks".
    """
    if list(exp_params.n_shot) == [AGENTIC_N_SHOT] or not _requests_agentic_benchmark(
        exp_params
    ):
        return

    logger.warning(
        "Ignoring --n-shot=%s for agentic benchmarks, which pin n-shot=0 (it still "
        "applies to any QA benchmarks in a mixed run).",
        ",".join(str(n) for n in exp_params.n_shot),
    )


def _warn_expensive_agentic_trials(exp_params: ExperimentParams) -> None:
    """Warn once per agentic benchmark when its trial count is costly.

    Each agentic trial runs every task in its own environment with tens of model
    calls, so cost scales with num_tasks * num_trials. This fires once up front
    (not per model), since the trial count is shared across all models.
    """
    if exp_params.num_trials <= _AGENTIC_TRIAL_WARN_THRESHOLD:
        return

    for benchmark_path in _requested_benchmark_paths(exp_params):
        if _is_agentic_benchmark(benchmark_path):
            logger.warning(
                "Running %d agentic trials for '%s'; each trial runs every task "
                "in its own environment with tens of model calls, so total cost "
                "scales with num_tasks * num_trials.",
                exp_params.num_trials,
                benchmark_path.name,
            )


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

    _check_agentic_compat(model_specs, exp_params, parallelize_models)
    _warn_ignored_agentic_nshot(exp_params)
    _warn_expensive_agentic_trials(exp_params)

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
