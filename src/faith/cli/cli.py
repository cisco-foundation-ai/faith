# Copyright 2025 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

# PYTHON_ARGCOMPLETE_OK
#
# DO NOT ADD EXPENSIVE DEPENDENCIES HERE.
# This is a wrapper script around different stages of benchmarking.
# To allow for autocompletion, imports must be lightweight.
# Within each stage's sub-command, you can import the necessary dependencies.

"""General purpose CLI script for running benchmarks on models."""

import argparse
import ast
import dataclasses
import logging
import os
from collections.abc import Iterator
from functools import partial
from pathlib import Path
from typing import Any

import argcomplete
import colorlog

from faith._internal.io.datastore import Datastore, DatastoreContext
from faith._internal.iter.transform import DevNullReducer
from faith._internal.threading.periodic import PeriodicTaskContext
from faith._types.benchmark.sample_ratio import SampleRatio
from faith._types.model.engine import EngineParams, ModelEngine
from faith._types.model.generation import GenerationMode, GenParams
from faith._types.model.prompt import PromptFormatter
from faith._types.model.spec import ModelSpec, Reasoning
from faith.benchmark.listing import benchmark_choices
from faith.cli.flags.annotated_path import AnnotatedPath
from faith.cli.flags.arg_value import DefaultValue, TypeWithDefault, UserValueType
from faith.cli.flags.token_parsing import parse_begin_end_tokens
from faith.cli.subcmd.summarize import OutputFormat
from faith.experiment.params import DataSamplingParams, ExperimentParams
from faith.model.listing import choice_to_model, model_choices

_cli_parser = argparse.ArgumentParser(
    description="General purpose script for running benchmarks on models.",
)
_cli_subparsers = _cli_parser.add_subparsers(required=True)

########################################
# Definition of the query sub-command. #
########################################


def _cli_query(args: argparse.Namespace, datastore: Datastore) -> Iterator[Path]:
    """Helper function to run the query sub-command over the CLI arguments."""
    # We disable the import-outside-toplevel pylint rule here since each
    # sub-command has different dependencies and importing them as part of the
    # main CLI script makes autocompletion slow.
    # pylint: disable=import-outside-toplevel
    from faith.cli.subcmd.query import run_experiment_queries

    # Build the list of model specs from both --model-paths and --model-configs.
    model_specs: list[ModelSpec] = []

    global_engine_kwargs = {
        k: ast.literal_eval(v)
        for k, _, v in [
            arg_str.partition("=") for arg_str in (args.engine_kwargs or [])
        ]
    }
    global_gen_kwargs = {
        k: ast.literal_eval(v)
        for k, _, v in [
            arg_str.partition("=") for arg_str in (args.generation_kwargs or [])
        ]
    }
    if args.model_paths:
        if args.prompt_format is None:
            raise RuntimeError(
                "error: --prompt-format is required when using --model-paths."
            )
        global_engine = EngineParams(
            engine_type=args.model_engine,
            num_gpus=args.num_gpus.value,
            context_length=args.model_context_len.value,
            kwargs=global_engine_kwargs,
        )
        global_gen = GenParams(
            temperature=args.temperature.value,
            top_p=args.top_p.value,
            max_completion_tokens=args.max_completion_tokens.value,
            kwargs=global_gen_kwargs,
        )
        model_specs.extend(
            [
                ModelSpec(
                    path=annotated_path.raw_path,
                    engine=dataclasses.replace(global_engine, num_gpus=num_gpus),
                    prompt_format=args.prompt_format,
                    generation=global_gen,
                    **{
                        k: v
                        for k, v in annotated_path.values().items()
                        if k != "num_gpus"
                    },
                )
                for annotated_path in args.model_paths
                if (
                    num_gpus := annotated_path.get_value("num_gpus")
                    or global_engine.num_gpus
                )
                >= 0
            ]
        )

    if args.model_configs:
        gen_overrides: dict[str, Any] = {}
        if not args.temperature.is_default:
            gen_overrides["temperature"] = args.temperature.value
        if not args.top_p.is_default:
            gen_overrides["top_p"] = args.top_p.value
        if not args.max_completion_tokens.is_default:
            gen_overrides["max_completion_tokens"] = args.max_completion_tokens.value

        engine_overrides: dict[str, Any] = {}
        if not args.model_context_len.is_default:
            engine_overrides["context_length"] = args.model_context_len.value
        if not args.num_gpus.is_default:
            engine_overrides["num_gpus"] = args.num_gpus.value

        spec_overrides: dict[str, Any] = {}
        if args.prompt_format is not None:
            spec_overrides["prompt_format"] = args.prompt_format

        model_specs.extend(
            [
                dataclasses.replace(
                    spec,
                    **spec_overrides,
                    engine=dataclasses.replace(
                        spec.engine,
                        kwargs=spec.engine.kwargs | global_engine_kwargs,
                        **(
                            engine_overrides
                            | (
                                {
                                    "num_gpus": annotated_config_path.get_value(
                                        "num_gpus"
                                    )
                                }
                                if annotated_config_path.get_value("num_gpus")
                                is not None
                                else {}
                            )
                        ),
                    ),
                    generation=dataclasses.replace(
                        spec.generation,
                        kwargs=spec.generation.kwargs | global_gen_kwargs,
                        **gen_overrides,
                    ),
                )
                for annotated_config_path in args.model_configs
                if (
                    spec := ModelSpec.from_file(
                        choice_to_model(annotated_config_path.raw_path)
                    )
                )
                is not None
            ]
        )

    if not model_specs:
        raise RuntimeError(
            "error: at least one model must be specified via either --model-paths or --model-configs."
        )

    return run_experiment_queries(
        model_specs,
        ExperimentParams(
            benchmark_names=args.benchmarks,
            custom_benchmark_paths=args.custom_benchmarks,
            generation_mode=args.generation_mode,
            n_shot=args.n_shot,
            num_trials=args.num_trials,
            initial_seed=args.seed,
        ),
        DataSamplingParams(
            sample_size=args.sample_size,
        ),
        datastore,
        parallelize_models=args.experimental_parallelize_models,
    )


def _query_main(args: argparse.Namespace) -> None:
    """Query model(s) over the questions in one or more benchmarks."""
    with DatastoreContext.from_path(args.datastore_location) as datastore:
        with PeriodicTaskContext(
            partial(datastore.push, raise_on_error=False), interval=150
        ):
            _ = _cli_query(args, datastore) >> DevNullReducer[Path]()


def _add_experiment_args(parser: argparse.ArgumentParser) -> None:
    """Add arguments for the experiment's configuration to `parser`."""
    group = parser.add_argument_group(
        "experiment", "Arguments for experiment's configuration"
    )

    group.add_argument(
        "--benchmarks",
        type=str,
        default=None,
        nargs="*",
        help="The benchmarks to query the model over.",
        choices=benchmark_choices(),
    )
    group.add_argument(
        "--custom-benchmarks",
        type=Path,
        default=None,
        nargs="*",
        help="Paths to custom benchmarks (each with a benchmark config). These benchmarks are added to those requested with '--benchmarks'.",
    )
    group.add_argument(
        "--generation-mode",
        type=GenerationMode,
        default=GenerationMode.CHAT_COMP,
        help="The generation mode to use for the model. For a list of available modes, see the mode list.",
        choices=list(GenerationMode),
    )
    group.add_argument(
        "--n-shot",
        type=SampleRatio.from_string,
        default=[SampleRatio.from_string("0")],
        nargs="+",
        help="The number of in-context examples per question. [Default: 0]",
    )
    group.add_argument(
        "--num-trials",
        type=int,
        default=1,
        help="Number of trials to run for the benchmark. [Default: 1]",
    )
    group.add_argument(
        "--seed",
        type=int,
        default=373_363,
        help="Random seed for reproducibility. [Default: 373,363]",
    )
    group.add_argument(
        "--sample-size",
        type=int,
        default=None,
        help="Sample size for the benchmark. If not supplied, no sampling occurs. [Default: None]",
    )


def _add_model_args(parser: argparse.ArgumentParser) -> None:
    """Add arguments for the model's configuration to `parser`."""
    group = parser.add_argument_group(
        "model",
        "Arguments for the model's configuration",
    )

    model_source = group.add_mutually_exclusive_group(required=True)
    model_source.add_argument(
        "--model-paths",
        type=AnnotatedPath(
            name=lambda x: x,
            num_gpus=TypeWithDefault[int | None](int, None),
            reasoning=TypeWithDefault[Reasoning | None](parse_begin_end_tokens, None),
            response_pattern=TypeWithDefault[str | None](str, None),
            tokenizer=TypeWithDefault[str | None](str, None),
        ),
        nargs="*",
        help=(
            "The paths/names of the models to benchmark. "
            "When a name is given, the model-engine is used to resolve that name. "
            "When a path is given, the path is treated as a local path by the engine. "
            "If the path is a gcp uri, the model is copied to a local temp directory. "
            "Requires --model-engine to be specified. "
            "Mutually exclusive with --model-configs."
        ),
    )
    model_configs_action = model_source.add_argument(
        "--model-configs",
        type=AnnotatedPath(num_gpus=TypeWithDefault[int | None](int, None)),
        nargs="*",
        help=(
            "Paths to YAML model configuration files or names of packaged model configs. "
            "Each file fully specifies a model's path, engine, and generation parameters. "
            "Mutually exclusive with --model-paths. "
            f"Available packaged configs: {', '.join(model_choices())}."
        ),
    )
    model_configs_action.completer = lambda **_: model_choices()  # type: ignore[attr-defined]
    group.add_argument(
        "--prompt-format",
        type=PromptFormatter,
        default=None,
        help="The prompt format to use for the model. Required for --model-paths; acts as an override for --model-configs.",
        choices=list(PromptFormatter),
    )
    group.add_argument(
        "--model-engine",
        type=ModelEngine,
        help="The type of model to use. For a list of available models, see the model list.",
        choices=list(ModelEngine),
    )
    group.add_argument(
        "--model-context-len",
        type=UserValueType(int),
        default=DefaultValue(3500),
        help="The context length of the model. [Default: 3500]",
    )
    group.add_argument(
        "--max-completion-tokens",
        type=UserValueType(int),
        default=DefaultValue(500),
        help="Maximum number of new tokens for model to generate in its answer. [Default: 500]",
    )
    group.add_argument(
        "--temperature",
        type=UserValueType(float),
        default=DefaultValue(0.0),
        help="Temperature for sampling. [Default: 0.0 (deterministic)]",
    )
    group.add_argument(
        "--top-p",
        type=UserValueType(float),
        default=DefaultValue(1.0),
        help="Top-p for sampling. [Defaults: 1.0 (deterministic)]",
    )
    group.add_argument(
        "--num-gpus",
        type=UserValueType(int),
        default=DefaultValue(1),
        help="Number of GPUs to use for querying LLMs. [Default: 1]",
    )
    group.add_argument(
        "--experimental-parallelize-models",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Run multiple models in parallel with GPU pooling. [Default: False]",
    )
    group.add_argument(
        "--engine-kwargs",
        type=str,
        help="Additional arguments to pass to vLLM for model generation.",
        nargs="*",
    )
    group.add_argument(
        "--generation-kwargs",
        type=str,
        help="Additional keyword arguments to pass to the model for text generation.",
        nargs="*",
    )


def _add_datastore_args(parser: argparse.ArgumentParser) -> None:
    """Add arguments for the datastore's configuration to `parser`."""
    parser.add_argument(
        "--datastore-location",
        type=str,
        required=True,
        help="The location for the datastore where benchmarking logs are stored.  This can be a local directory or a GCP URI (beginning with gs://). If a GCP URI is provided, logs will be stored in a tmp dir and periodically uploaded to GCP.",
    )


_query_cmd_parser = _cli_subparsers.add_parser(
    "query", help="Query the model(s) on the questions from the benchmark(s)."
)
_add_experiment_args(_query_cmd_parser)
_add_model_args(_query_cmd_parser)
_add_datastore_args(_query_cmd_parser)
_query_cmd_parser.set_defaults(func=_query_main)

#######################################
# Definition of the eval sub-command. #
#######################################


def _eval_main(args: argparse.Namespace) -> None:
    """Compute all metrics for all benchmarks from the models' query responses."""
    # We disable the import-outside-toplevel pylint rule here since each
    # sub-command has different dependencies and importing them as part of the
    # main CLI script makes autocompletion slow.
    # pylint: disable=import-outside-toplevel
    from tqdm import tqdm

    from faith.cli.subcmd.eval import RecordHandlingParams, evaluate_experiment

    experiment_paths = (
        args.experiment_path.glob("**/experiment.json")
        if args.experiment_path.is_dir()
        else [args.experiment_path]
    )

    for experiment_path in tqdm(
        experiment_paths, desc="Processing experiments", unit="experiment"
    ):
        evaluate_experiment(
            experiment_path,
            RecordHandlingParams(
                annotate_prediction_stats=args.cache_prediction_stats,
                recompute_stats=args.force_compute_stats,
            ),
            metrics_output_path=experiment_path.parent / "metrics.json",
        )


def _add_eval_args(parser: argparse.ArgumentParser) -> None:
    """Add arguments for the eval configuration to `parser`."""
    group = parser.add_argument_group("eval", "Arguments for metrics configuration")

    group.add_argument(
        "--cache-prediction-stats",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether to store prediction statistics in the original logs to avoid recomputing in the future; this is recommended when using GPT-4o to extract predictions.",
    )
    group.add_argument(
        "--force-compute-stats",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Forces recomputation of stats even if they are already present in the logs.",
    )


def _add_logs_source_args(parser: argparse.ArgumentParser) -> None:
    """Add arguments for loading experiment logs to `parser`."""
    group = parser.add_argument_group(
        "logs_source",
        "Arguments for loading experiment logs",
    )

    group.add_argument(
        "--experiment-path",
        type=Path,
        required=True,
        help="The path to the experiment logs to analyze. If a directory is provided, all experiments within that directory will be analyzed.",
    )


_eval_parser = _cli_subparsers.add_parser(
    "eval", help="Compute metrics from query responses for each model & benchmark."
)
_add_eval_args(_eval_parser)
_add_logs_source_args(_eval_parser)
_eval_parser.set_defaults(func=_eval_main)

############################################
# Definition of the summarize sub-command. #
############################################


def _summarize_main(args: argparse.Namespace) -> None:
    """Summarize benchmark metrics."""
    # We disable the import-outside-toplevel pylint rule here since each
    # sub-command has different dependencies and importing them as part of the
    # main CLI script makes autocompletion slow.
    # pylint: disable=import-outside-toplevel
    from faith.cli.subcmd.summarize import summarize_experiments

    summarize_experiments(
        args.output_format,
        args.experiment_path,
        args.stats,
        args.summary_filepath,
        bigquery_project=args.bigquery_project,
        bigquery_dataset=args.bigquery_dataset,
        bigquery_table=args.bigquery_table,
    )


def _add_summarize_args(parser: argparse.ArgumentParser) -> None:
    """Add arguments for experiment summarization to `parser`."""
    group = parser.add_argument_group(
        "summarize", "Arguments for experiment summarization"
    )

    group.add_argument(
        "--output-format",
        type=OutputFormat,
        default="table",
        choices=["table", "csv", "bigquery"],
        help="Output format: 'table' (print to console), 'csv' (save to file), or 'bigquery' (ingest to BigQuery).",
    )
    group.add_argument(
        "--stats",
        type=str,
        default=[],
        nargs="*",
        help="The stats to summarize (for table/csv output). This should be a comma-separated list of metric names.",
    )
    group.add_argument(
        "--summary-filepath",
        type=Path,
        default=None,
        help="The path to the output directory where the summary will be saved (for csv output).",
    )

    # BigQuery-specific arguments
    bigquery_group = parser.add_argument_group(
        "bigquery", "Arguments for BigQuery output (requires --output-format bigquery)"
    )
    bigquery_group.add_argument(
        "--bigquery-project",
        type=str,
        default=None,
        help="GCP project ID for BigQuery. Can also be set via FAITH_BIGQUERY_PROJECT environment variable.",
    )
    bigquery_group.add_argument(
        "--bigquery-dataset",
        type=str,
        default=None,
        help="BigQuery dataset name. Can also be set via FAITH_BIGQUERY_DATASET environment variable.",
    )
    bigquery_group.add_argument(
        "--bigquery-table",
        type=str,
        default=None,
        help="BigQuery table name (default: metrics). Can also be set via FAITH_BIGQUERY_TABLE environment variable.",
    )


_summarize_parser = _cli_subparsers.add_parser(
    "summarize",
    aliases=["digest"],
    help="Summarize (digest) a set of benchmark metrics.",
)
_add_summarize_args(_summarize_parser)
_add_logs_source_args(_summarize_parser)
_summarize_parser.set_defaults(func=_summarize_main)

##########################################
# Definition of the run-all sub-command. #
##########################################


def _run_all_main(args: argparse.Namespace) -> None:
    """Run end-to-end benchmarking on models."""
    # We disable the import-outside-toplevel pylint rule here since each
    # sub-command has different dependencies and importing them as part of the
    # main CLI script makes autocompletion slow.
    # pylint: disable=import-outside-toplevel
    from faith.cli.subcmd.eval import RecordHandlingParams, evaluate_experiment
    from faith.cli.subcmd.summarize import summarize_experiments

    with DatastoreContext.from_path(args.datastore_location) as datastore:
        with PeriodicTaskContext(
            partial(datastore.push, raise_on_error=False), interval=150
        ):
            for experiment_path in _cli_query(args, datastore):
                evaluate_experiment(
                    experiment_path,
                    RecordHandlingParams(
                        annotate_prediction_stats=args.cache_prediction_stats,
                        recompute_stats=args.force_compute_stats,
                    ),
                    metrics_output_path=experiment_path.parent / "metrics.json",
                )
            summarize_experiments(
                OutputFormat.TABLE, datastore.path, args.stats, args.summary_filepath
            )


_run_all_parser = _cli_subparsers.add_parser(
    "run-all", aliases=["qed"], help="Run end-to-end benchmarks."
)
_add_experiment_args(_run_all_parser)
_add_model_args(_run_all_parser)
_add_datastore_args(_run_all_parser)
_add_eval_args(_run_all_parser)
_add_summarize_args(_run_all_parser)
_run_all_parser.set_defaults(func=_run_all_main)

# Enable tab completion for the command line interface and add the main entry point.

argcomplete.autocomplete(_cli_parser)
_cli_args = _cli_parser.parse_args()


def _cli_main() -> None:
    """Main entry point for the benchmarking commands."""

    # Suppress the transformers logger's output.
    logging.getLogger("transformers").setLevel(logging.ERROR)

    # Suppress the vLLM logger's output.
    logging.getLogger("vllm").setLevel(logging.ERROR)
    os.environ["VLLM_CONFIGURE_LOGGING"] = "1"  # Set to 0 to disable vLLM logging.

    # Configuring log level for model API calls.
    logging.getLogger("httpx").setLevel(logging.ERROR)

    # Create the logger for the CLI with its formatting and level.
    handler = colorlog.StreamHandler()
    handler.setFormatter(
        colorlog.ColoredFormatter(
            "%(log_color)s%(levelname)s%(reset)s %(asctime)s [%(name)s:%(lineno)d]: %(message)s",
            datefmt="%m-%d %H:%M:%S",
            log_colors={
                "DEBUG": "cyan",
                "INFO": "green",
                "WARNING": "yellow",
                "ERROR": "red",
                "CRITICAL": "bold_red",
            },
        )
    )
    logging.basicConfig(level=logging.INFO, handlers=[handler])

    _cli_args.func(_cli_args)


if __name__ == "__main__":
    _cli_main()
