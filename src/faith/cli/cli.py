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
import logging
import os
from pathlib import Path
from typing import Iterator

import argcomplete
import colorlog

from faith._internal.io.datastore import resolve_storage_path
from faith._internal.iter.transform import DevNullReducer
from faith._internal.types.flags import (
    AnnotatedPath,
    GenerationMode,
    SampleRatio,
    TypeWithDefault,
)
from faith.benchmark.formatting.prompt import PromptFormatter
from faith.benchmark.listing import benchmark_choices
from faith.cli.flags import parse_begin_end_tokens
from faith.experiment.params import DataSamplingParams, ExperimentParams
from faith.model.model_engine import ModelEngine
from faith.model.params import EngineParams, GenParams

_cli_parser = argparse.ArgumentParser(
    description="General purpose script for running benchmarks on models.",
)
_cli_subparsers = _cli_parser.add_subparsers(required=True)

########################################
# Definition of the query sub-command. #
########################################


def _cli_query(args: argparse.Namespace, datastore_path: Path) -> Iterator[Path]:
    """Helper function to run the query sub-command over the CLI arguments."""
    # We disable the import-outside-toplevel pylint rule here since each
    # sub-command has different dependencies and importing them as part of the
    # main CLI script makes autocompletion slow.
    # pylint: disable=import-outside-toplevel
    from faith.cli.subcmd.query import run_experiment_queries

    return run_experiment_queries(
        ExperimentParams(
            benchmark_names=args.benchmarks,
            custom_benchmark_paths=args.custom_benchmarks,
            generation_mode=args.generation_mode,
            prompt_format=args.prompt_format,
            n_shot=args.n_shot,
            model_paths=args.model_paths,
            num_trials=args.num_trials,
            initial_seed=args.seed,
        ),
        DataSamplingParams(
            sample_size=args.sample_size,
        ),
        EngineParams(
            engine_type=args.model_engine,
            num_gpus=args.num_gpus,
            context_length=args.model_context_len,
            kwargs=(
                {
                    k: ast.literal_eval(v)
                    for k, _, v in [
                        arg_str.partition("=") for arg_str in args.engine_kwargs
                    ]
                }
                if args.engine_kwargs is not None
                else {}
            ),
        ),
        GenParams(
            temperature=args.temperature,
            top_p=args.top_p,
            max_completion_tokens=args.max_completion_tokens,
            kwargs={
                k: ast.literal_eval(v)
                for k, _, v in [
                    arg_str.partition("=") for arg_str in (args.generation_kwargs or [])
                ]
            },
        ),
        datastore_path,
    )


def _query_main(args: argparse.Namespace) -> None:
    """Query model(s) over the questions in one or more benchmarks."""
    with resolve_storage_path(args.datastore_location) as datastore_path:
        _ = _cli_query(args, datastore_path) >> DevNullReducer[Path]()


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
        required=True,
        help="The generation mode to use for the model. For a list of available modes, see the mode list.",
        choices=list(GenerationMode),
    )
    group.add_argument(
        "--prompt-format",
        type=PromptFormatter.from_string,
        required=True,
        help="The prompt format to use for the model. For a list of available formats, see the format list.",
        choices=list(PromptFormatter),
    )
    group.add_argument(
        "--n-shot",
        type=SampleRatio.from_string,
        required=True,
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

    group.add_argument(
        "--model-paths",
        type=AnnotatedPath(
            name=lambda x: x,
            is_file=TypeWithDefault[bool](bool, False),
            reasoning_tokens=TypeWithDefault[
                tuple[str | list[int], str | list[int]] | None
            ](parse_begin_end_tokens, None),
            response_pattern=TypeWithDefault[str | None](str, None),
            tokenizer=TypeWithDefault[str | None](str, None),
        ),
        required=True,
        nargs="+",
        help="The paths/names of the models to benchmark. For OpenAI and Hugging Face models, see the model lists. For custom models, specify the path to the model.",
    )
    group.add_argument(
        "--model-engine",
        type=ModelEngine.from_string,
        required=True,
        help="The type of model to use. For a list of available models, see the model list.",
        choices=list(ModelEngine),
    )
    group.add_argument(
        "--model-context-len",
        type=int,
        default=3500,
        help="The context length of the model. [Default: 3500]",
    )
    group.add_argument(
        "--max-completion-tokens",
        default=500,
        type=int,
        help="Maximum number of new tokens for model to generate in its answer. [Default: 500]",
    )
    group.add_argument(
        "--temperature",
        default=0.0,
        type=float,
        help="Temperature for sampling. [Default: 0.0 (deterministic)]",
    )
    group.add_argument(
        "--top-p",
        default=1.0,
        type=float,
        help="Top-p for sampling. [Defaults: 1.0 (deterministic)]",
    )
    group.add_argument(
        "--num-gpus",
        type=int,
        default=1,
        help="Number of GPUs to use for querying LLMs. [Default: 1]",
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

    from faith.cli.subcmd.eval import RecordHandlingParams, compute_experiment_metrics

    filepaths = [args.experiment_path]
    if args.experiment_path.is_dir():
        filepaths = list(args.experiment_path.glob("**/experiment.json"))

    for filepath in tqdm(filepaths, desc="Processing experiments", unit="experiment"):
        compute_experiment_metrics(
            filepath,
            RecordHandlingParams(
                annotate_prediction_stats=args.cache_prediction_stats,
                recompute_stats=args.force_compute_stats,
            ),
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

    summarize_experiments(args.experiment_path, args.stats, args.summary_filepath)


def _add_summarize_args(parser: argparse.ArgumentParser) -> None:
    """Add arguments for experiment summarization to `parser`."""
    group = parser.add_argument_group(
        "summarize", "Arguments for experiment summarization"
    )

    group.add_argument(
        "--stats",
        type=str,
        default=[],
        nargs="*",
        help="The stats to summarize. This should be a comma-separated list of metric names.",
    )
    group.add_argument(
        "--summary-filepath",
        type=Path,
        default=None,
        help="The path to the output directory where the summary will be saved.",
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
    from faith.cli.subcmd.eval import RecordHandlingParams, compute_experiment_metrics
    from faith.cli.subcmd.summarize import summarize_experiments

    with resolve_storage_path(args.datastore_location) as datastore_path:
        for experiment_path in _cli_query(args, datastore_path):
            compute_experiment_metrics(
                experiment_path,
                RecordHandlingParams(
                    annotate_prediction_stats=args.cache_prediction_stats,
                    recompute_stats=args.force_compute_stats,
                ),
            )
        summarize_experiments(datastore_path, args.stats, args.summary_filepath)


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

    # Suppress the transformers logger's ouptut.
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
