# Copyright 2025 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

import re
import subprocess
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from faith._internal.io.json import write_as_json
from faith._internal.iter.transform import DevNullReducer
from faith._internal.multiprocessing.gpu_scheduling import JobOutcome
from faith._types.benchmark.sample_ratio import SampleRatio
from faith._types.model.engine import EngineParams, ModelEngine
from faith._types.model.generation import GenerationMode
from faith._types.model.prompt import PromptFormatter
from faith._types.model.spec import ModelSpec
from faith._types.record.sample import ReplacementStrategy, SampleRecord
from faith.cli.subcmd.query import (
    _run_single_model,
    current_timestamp,
    get_command_output,
    read_trial_log,
    run_experiment_queries,
)
from faith.experiment.params import DataSamplingParams, ExperimentParams
from faith.record_pipelines.params import RecordHandlingParams

MINIMAL_SAMPLE_RECORD = SampleRecord.from_dict(
    {
        "metadata": {"version": "1.0"},
        "data": {
            "benchmark_sample_index": 0,
            "benchmark_sample_hash": "abc123",
            "subject": None,
            "system_prompt": None,
            "instruction": None,
            "question": "What is 1+1?",
            "choices": None,
            "label": "2",
            "formatted_question": "What is 1+1?",
            "formatted_answer": "2",
            "question_prompt": "What is 1+1?",
        },
        "model_data": {
            "prompt": "What is 1+1?",
            "answer_symbol_ids": {},
        },
    }
)


def test_current_timestamp() -> None:
    ts = current_timestamp()
    assert re.match(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2} \w+", ts)
    assert ts.endswith("PST") or ts.endswith("PDT")


def test_get_command_output_returns_stripped_stdout() -> None:
    assert get_command_output("echo hello") == "hello"


def test_get_command_output_raises_on_failure() -> None:
    with pytest.raises(subprocess.CalledProcessError):
        get_command_output("false")


def test_read_trial_log(tmp_path: Path) -> None:
    assert not list(read_trial_log(tmp_path / "missing.json"))

    log_path = tmp_path / "benchmark-log.json"
    write_as_json(log_path, [MINIMAL_SAMPLE_RECORD])

    assert list(read_trial_log(log_path)) == [MINIMAL_SAMPLE_RECORD]


def test_run_single_model_creation_failure(tmp_path: Path) -> None:
    with (
        patch(
            "faith.cli.subcmd.query.create_model",
            side_effect=ValueError("GPU OOM"),
        ),
        pytest.raises(RuntimeError, match="Failed to initialize model: GPU OOM"),
    ):
        _ = (
            _run_single_model(
                model_spec=ModelSpec(
                    path=str(tmp_path),
                    engine=EngineParams(engine_type=ModelEngine.OPENAI),
                    prompt_format=PromptFormatter.CHAT,
                ),
                exp_params=ExperimentParams(
                    benchmark_names=[],
                    custom_benchmark_paths=[],
                    generation_mode=GenerationMode.CHAT_COMP,
                    n_shot=[SampleRatio(0)],
                    num_trials=1,
                    initial_seed=0,
                ),
                sampling_params=DataSamplingParams(),
                record_params=RecordHandlingParams(ReplacementStrategy.NEVER),
                datastore=Mock(),
            )
            >> DevNullReducer[Path]()
        )


def test_run_single_model_unsupported_prompt_format(tmp_path: Path) -> None:
    with (
        patch(
            "faith.cli.subcmd.query.create_model",
            return_value=Mock(supported_formats={PromptFormatter.BASE}),
        ),
        pytest.raises(AssertionError, match="not supported"),
    ):
        _ = (
            _run_single_model(
                model_spec=ModelSpec(
                    path=str(tmp_path),
                    engine=EngineParams(engine_type=ModelEngine.OPENAI),
                    prompt_format=PromptFormatter.CHAT,
                ),
                exp_params=ExperimentParams(
                    benchmark_names=[],
                    custom_benchmark_paths=[],
                    generation_mode=GenerationMode.CHAT_COMP,
                    n_shot=[SampleRatio(0)],
                    num_trials=1,
                    initial_seed=0,
                ),
                sampling_params=DataSamplingParams(),
                record_params=RecordHandlingParams(ReplacementStrategy.NEVER),
                datastore=Mock(),
            )
            >> DevNullReducer[Path]()
        )


def test_run_single_model_experiment_loop(tmp_path: Path) -> None:
    exp_dir = tmp_path / "exp"
    trial_path = Path("trials/0/abc123/benchmark-log.json")

    with (
        patch("faith.cli.subcmd.query.find_benchmarks", return_value=[tmp_path]),
        patch(
            "faith.cli.subcmd.query.create_model",
            return_value=Mock(
                supported_formats={PromptFormatter.CHAT},
                tokenizer=None,
                query=Mock(return_value=[]),
            ),
        ),
        patch(
            "faith.cli.subcmd.query.BenchmarkExperiment",
            return_value=Mock(
                datastore=Mock(path=exp_dir),
                benchmark_spec=Mock(to_dict=Mock(return_value={"name": "test"})),
                benchmark_config=Mock(to_dict=Mock(return_value={})),
                __iter__=Mock(
                    return_value=iter(
                        [
                            (
                                Mock(
                                    build_dataset=Mock(
                                        return_value=Mock(
                                            iter_data=Mock(return_value=[])
                                        )
                                    ),
                                    generation_mode=GenerationMode.CHAT_COMP,
                                ),
                                trial_path,
                            ),
                        ]
                    )
                ),
            ),
        ),
    ):
        assert list(
            _run_single_model(
                model_spec=ModelSpec(
                    path=str(tmp_path),
                    engine=EngineParams(engine_type=ModelEngine.OPENAI),
                    prompt_format=PromptFormatter.CHAT,
                ),
                exp_params=ExperimentParams(
                    benchmark_names=[],
                    custom_benchmark_paths=[tmp_path],
                    generation_mode=GenerationMode.CHAT_COMP,
                    n_shot=[SampleRatio(0)],
                    num_trials=1,
                    initial_seed=0,
                ),
                sampling_params=DataSamplingParams(),
                record_params=RecordHandlingParams(ReplacementStrategy.NEVER),
                datastore=Mock(),
            )
        ) == [exp_dir / "experiment.json"]
        assert (exp_dir / "experiment.json").exists()


def test_run_experiment_queries_sequential(tmp_path: Path) -> None:
    with (
        patch(
            "faith.cli.subcmd.query._run_single_model",
            return_value=iter([tmp_path / "experiment.json"]),
        ),
        patch("faith.cli.subcmd.query._killall_gpu_processes"),
    ):
        assert list(
            run_experiment_queries(
                model_specs=[
                    ModelSpec(
                        path="test-model",
                        engine=EngineParams(engine_type=ModelEngine.OPENAI),
                        prompt_format=PromptFormatter.CHAT,
                    )
                ],
                exp_params=ExperimentParams(
                    benchmark_names=["some_benchmark"],
                    custom_benchmark_paths=None,
                    generation_mode=GenerationMode.CHAT_COMP,
                    n_shot=[],
                    num_trials=1,
                    initial_seed=0,
                ),
                sampling_params=DataSamplingParams(),
                record_params=RecordHandlingParams(ReplacementStrategy.NEVER),
                datastore=Mock(),
                parallelize_models=False,
            )
        ) == [tmp_path / "experiment.json"]


def test_run_experiment_queries_parallel_success(tmp_path: Path) -> None:
    with (
        patch(
            "faith.cli.subcmd.query.run_gpu_jobs_in_parallel",
            return_value=[
                JobOutcome(job_id="m1", ok=True, result=[tmp_path / "experiment.json"]),
            ],
        ),
        patch("faith.cli.subcmd.query._killall_gpu_processes"),
    ):
        assert list(
            run_experiment_queries(
                model_specs=[
                    ModelSpec(
                        path="test-model",
                        engine=EngineParams(engine_type=ModelEngine.VLLM),
                        prompt_format=PromptFormatter.CHAT,
                    )
                ],
                exp_params=ExperimentParams(
                    benchmark_names=["some_benchmark"],
                    custom_benchmark_paths=None,
                    generation_mode=GenerationMode.CHAT_COMP,
                    n_shot=[],
                    num_trials=1,
                    initial_seed=0,
                ),
                sampling_params=DataSamplingParams(),
                record_params=RecordHandlingParams(ReplacementStrategy.NEVER),
                datastore=Mock(),
                parallelize_models=True,
            )
        ) == [tmp_path / "experiment.json"]


def test_run_experiment_queries_requires_benchmarks() -> None:
    exp_params = ExperimentParams(
        benchmark_names=None,
        custom_benchmark_paths=None,
        generation_mode=GenerationMode.CHAT_COMP,
        n_shot=[],
        num_trials=1,
        initial_seed=0,
    )
    with pytest.raises(AssertionError, match="at least one benchmark"):
        list(
            run_experiment_queries(
                model_specs=[],
                exp_params=exp_params,
                sampling_params=DataSamplingParams(),
                record_params=RecordHandlingParams(ReplacementStrategy.NEVER),
                datastore=Mock(),
            )
        )


def test_run_experiment_queries_parallel_requires_vllm() -> None:
    exp_params = ExperimentParams(
        benchmark_names=["some_benchmark"],
        custom_benchmark_paths=None,
        generation_mode=GenerationMode.CHAT_COMP,
        n_shot=[],
        num_trials=1,
        initial_seed=0,
    )
    openai_spec = ModelSpec(
        path="test-model",
        engine=EngineParams(engine_type=ModelEngine.OPENAI),
        prompt_format=PromptFormatter.CHAT,
    )
    with pytest.raises(AssertionError, match="only supported for the vLLM engine"):
        list(
            run_experiment_queries(
                model_specs=[openai_spec],
                exp_params=exp_params,
                sampling_params=DataSamplingParams(),
                record_params=RecordHandlingParams(ReplacementStrategy.NEVER),
                datastore=Mock(),
                parallelize_models=True,
            )
        )
