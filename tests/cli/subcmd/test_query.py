# Copyright 2025 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

import dataclasses
import re
import shutil
import subprocess
import sys
from pathlib import Path
from unittest.mock import Mock, patch

import httpx
import pytest
from openai import BadRequestError, InternalServerError

from faith._internal.io.json import read_json_file, write_as_json
from faith._internal.io.store import StoreContext
from faith._internal.iter.transform import DevNullReducer
from faith._internal.multiprocessing.gpu_scheduling import JobOutcome
from faith._types.benchmark.sample_ratio import SampleRatio
from faith._types.config.agentic import AgenticEnvironmentConfig
from faith._types.config.benchmark import BenchmarkConfig
from faith._types.model.engine import REDACTED, EngineParams, ModelEngine
from faith._types.model.generation import GenerationMode, GenParams
from faith._types.model.prompt import PromptFormatter
from faith._types.model.spec import ModelSpec
from faith._types.record.sample import ReplacementStrategy, SampleRecord
from faith.cli.subcmd.query import (
    _AGENTIC_TRIAL_WARN_THRESHOLD,
    _TOOL_PROBE_MAX_TOKENS,
    _TOOL_PROBE_REMEDY,
    _build_experiments,
    _check_tool_calling,
    _requested_benchmark_paths,
    _run_single_model,
    _warn_expensive_agentic_trials,
    _warn_ignored_agentic_nshot,
    current_timestamp,
    get_command_output,
    redact_argv,
    run_experiment_queries,
)
from faith.experiment.params import DataSamplingParams, ExperimentParams
from faith.record_pipelines.params import RecordHandlingParams
from faith.record_pipelines.trial_log import read_trial_log

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


@pytest.mark.parametrize(
    "argv,expected",
    [
        (["api_key=sk-abc"], ["api_key=<redacted>"]),
        (["--api-key", "sk-abc"], ["--api-key", "<redacted>"]),
        (["--API-KEY", "sk-abc"], ["--API-KEY", "<redacted>"]),
        (
            ["--engine-kwargs", "max_model_len=131072"],
            ["--engine-kwargs", "max_model_len=131072"],
        ),
        (["--api-key"], ["--api-key"]),
        (["api_key=a=b=c"], ["api_key=<redacted>"]),
        (["--model-path"], ["--model-path"]),
        (
            ["--engine-kwargs", "api_key=1", "--engine-kwargs", "AUTH_TOKEN=2"],
            [
                "--engine-kwargs",
                "api_key=<redacted>",
                "--engine-kwargs",
                "AUTH_TOKEN=<redacted>",
            ],
        ),
        # A generation bound is not a credential, however much its name looks like one.
        (["--max-completion-tokens", "512"], ["--max-completion-tokens", "512"]),
        (
            ["--engine-kwargs", "max_tokens=512", "stop_token_ids=[128001]"],
            ["--engine-kwargs", "max_tokens=512", "stop_token_ids=[128001]"],
        ),
    ],
)
def test_redact_argv_masks_every_credential_shape(
    argv: list[str], expected: list[str]
) -> None:
    """A credential is masked whether it follows `=` or its flag, and keys survive."""
    assert redact_argv(argv) == expected


def test_redact_argv_masks_only_the_secret_in_a_full_command_line() -> None:
    """A realistic command line round-trips with the credential as its only change.

    `--max-completion-tokens` is here because it is the case that made this worth
    fixing: a substring match on `token` redacted it, so the record silently lost a
    parameter that decides what the run measured.
    """
    assert redact_argv(
        [
            "faith",
            "query",
            "--model-path",
            "openai/gpt-4",
            "--max-completion-tokens",
            "512",
            "--engine-kwargs",
            'api_key="sk-abc123"',
            "--engine-kwargs",
            "base_url=https://example.invalid/v1",
            "--output-dir",
            "/tmp/out",
        ]
    ) == [
        "faith",
        "query",
        "--model-path",
        "openai/gpt-4",
        "--max-completion-tokens",
        "512",
        "--engine-kwargs",
        "api_key=<redacted>",
        "--engine-kwargs",
        "base_url=https://example.invalid/v1",
        "--output-dir",
        "/tmp/out",
    ]


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
    bench = _write_benchmark(tmp_path / "qa", agentic=False)

    with (
        patch("faith.cli.subcmd.query.find_benchmarks", return_value=[bench]),
        patch.object(
            sys, "argv", ["faith", "query", "--engine-kwargs", 'api_key="sk-abc123"']
        ),
        # The per-trial pipeline is exercised by QATrialRunner's own tests; here
        # we only verify the experiment loop drives the runner and writes the
        # run record, so the runner is mocked out.
        patch(
            "faith.cli.subcmd.query.trial_runner_for",
            return_value=Mock(run_trial=Mock(return_value=None)),
        ),
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
                    custom_benchmark_paths=[bench],
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
        # The recorded command line must not carry the key the model spec redacts.
        assert read_json_file(exp_dir / "experiment.json")["metadata"]["run_args"] == [
            "faith",
            "query",
            "--engine-kwargs",
            "api_key=<redacted>",
        ]


def test_run_experiment_queries_sequential(tmp_path: Path) -> None:
    bench = _write_benchmark(tmp_path / "qa", agentic=False)
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
                    benchmark_names=[],
                    custom_benchmark_paths=[bench],
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
    bench = _write_benchmark(tmp_path / "qa", agentic=False)
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
                    benchmark_names=[],
                    custom_benchmark_paths=[bench],
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


def _copy_benchmark(name: str, dir_path: Path) -> Path:
    """Copy the named testdata benchmark into `dir_path`, which the test may nest."""
    dir_path.mkdir(parents=True, exist_ok=True)
    source = Path(__file__).parent / "testdata" / "query" / name / "benchmark.yaml"
    shutil.copy(source, dir_path)
    return dir_path


def _write_benchmark(dir_path: Path, *, agentic: bool) -> Path:
    """Copy a minimal benchmark.yaml (agentic or QA) into `dir_path`."""
    return _copy_benchmark("fake-agentic" if agentic else "fake-qa", dir_path)


def _exp_params(custom_paths: list[Path], *, num_trials: int = 1) -> ExperimentParams:
    return ExperimentParams(
        benchmark_names=None,
        custom_benchmark_paths=custom_paths,
        generation_mode=GenerationMode.LOGITS,  # distinct from the agentic constant
        n_shot=[SampleRatio(0), SampleRatio.from_string("1")],
        num_trials=num_trials,
        initial_seed=0,
    )


def _model_spec() -> ModelSpec:
    return ModelSpec(
        path="test-model",
        engine=EngineParams(engine_type=ModelEngine.OPENAI),
        prompt_format=PromptFormatter.CHAT,
    )


def test_build_experiments_pins_agentic_spec_and_skips_nshot(tmp_path: Path) -> None:
    """One agentic experiment is built with the pinned spec, ignoring --n-shot fan-out.

    The QA formatter passed in is BASE, i.e. deliberately *not* the agentic pin, so a
    build that forwards the caller's formatter instead of pinning it cannot pass.
    """
    bench = _write_benchmark(tmp_path / "agentic", agentic=True)
    experiments = list(
        _build_experiments(
            [bench],
            _exp_params([bench]),
            PromptFormatter.BASE,
            _model_spec(),
            Mock(sub_store=Mock(return_value=Mock())),
        )
    )
    # Two --n-shot values requested, but agentic collapses to a single experiment.
    assert len(experiments) == 1
    spec = experiments[0].benchmark_spec
    assert spec.generation_mode == GenerationMode.CHAT_COMP
    assert spec.prompt_format == PromptFormatter.CHAT
    assert str(spec.n_shot) == "0"


def test_build_experiments_fans_qa_over_nshot(tmp_path: Path) -> None:
    """QA benchmarks still fan out one experiment per requested --n-shot value.

    The mirror of the agentic case: QA takes the caller's generation mode and
    formatter verbatim, so both differ from the agentic pins here.
    """
    bench = _write_benchmark(tmp_path / "qa", agentic=False)
    experiments = list(
        _build_experiments(
            [bench],
            _exp_params([bench]),
            PromptFormatter.BASE,
            _model_spec(),
            Mock(sub_store=Mock(return_value=Mock())),
        )
    )
    assert len(experiments) == 2  # one per n_shot value
    assert [
        (e.benchmark_spec.generation_mode, e.benchmark_spec.prompt_format)
        for e in experiments
    ] == [(GenerationMode.LOGITS, PromptFormatter.BASE)] * 2


@pytest.mark.parametrize(
    ("num_trials", "expected_warnings"),
    [
        # One warning per agentic benchmark, so two agentic benchmarks warn twice;
        # a single-benchmark case cannot tell "per benchmark" from "once total".
        (50, 2),
        (_AGENTIC_TRIAL_WARN_THRESHOLD, 0),
    ],
    ids=["over_threshold", "at_threshold"],
)
def test_warn_expensive_agentic_trials_warns_once_per_agentic_benchmark(
    num_trials: int,
    expected_warnings: int,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A costly agentic --num-trials warns per agentic benchmark, and only those."""
    benches = [
        _write_benchmark(tmp_path / "agentic1", agentic=True),
        _write_benchmark(tmp_path / "agentic2", agentic=True),
        _write_benchmark(tmp_path / "qa", agentic=False),
    ]
    with caplog.at_level("WARNING"):
        _warn_expensive_agentic_trials(_exp_params(benches, num_trials=num_trials))

    warned = [r.message for r in caplog.records if "agentic trials" in r.message]
    assert len(warned) == expected_warnings
    # Only the agentic benchmarks are named; the QA one is never warned about.
    assert not any("qa" in m for m in warned)


def _pinned_exp_params(custom_paths: list[Path]) -> ExperimentParams:
    """Exp params whose spec flags all match the agentic pins (warning stays silent)."""
    return ExperimentParams(
        benchmark_names=None,
        custom_benchmark_paths=custom_paths,
        generation_mode=GenerationMode.CHAT_COMP,
        n_shot=[SampleRatio(0)],
        num_trials=1,
        initial_seed=0,
    )


_VLLM_SPEC = ModelSpec(
    path="test-model",
    engine=EngineParams(engine_type=ModelEngine.VLLM),
    prompt_format=PromptFormatter.CHAT,
)
_BASE_FORMAT_SPEC = ModelSpec(
    path="test-model",
    engine=EngineParams(engine_type=ModelEngine.OPENAI),
    prompt_format=PromptFormatter.BASE,
)


@pytest.mark.parametrize(
    ("model_spec", "pinned_params", "parallelize", "message"),
    [
        # Agentic runs stay single-process: their clients aren't picklable.
        (_model_spec(), True, True, "incompatible with agentic"),
        # Only the openai engine exposes the interactive client the loop drives.
        (_VLLM_SPEC, True, False, "require the openai engine"),
        # The openai model rejects logprobs at init, so the run dies before any
        # task executes; --generation-mode=logits is fatal, not merely ignored.
        (_model_spec(), False, False, "require --generation-mode"),
        # Likewise --prompt-format=base: the openai model supports chat only.
        (_BASE_FORMAT_SPEC, True, False, "require the chat prompt format"),
    ],
    ids=["parallelize", "non_openai_engine", "logits_mode", "base_prompt_format"],
)
def test_run_experiment_queries_rejects_an_incompatible_agentic_run(
    model_spec: ModelSpec,
    pinned_params: bool,
    parallelize: bool,
    message: str,
    tmp_path: Path,
) -> None:
    """Each agentic incompatibility is rejected up front, before any trial runs.

    Fail-fast matters most in a mixed QA+agentic run: without it the QA benchmarks
    burn their full time and cost before the agentic one hits the incompatibility.
    """
    bench = _write_benchmark(tmp_path / "agentic", agentic=True)
    exp_params = _pinned_exp_params([bench]) if pinned_params else _exp_params([bench])
    with pytest.raises(ValueError, match=message):
        list(
            run_experiment_queries(
                model_specs=[model_spec],
                exp_params=exp_params,
                sampling_params=DataSamplingParams(),
                record_params=RecordHandlingParams(ReplacementStrategy.NEVER),
                datastore=Mock(),
                parallelize_models=parallelize,
            )
        )


def test_run_experiment_queries_rejects_a_qa_only_model_in_a_mixed_run(
    tmp_path: Path,
) -> None:
    """The engine restriction is whole-run, so a spec serving only QA is rejected too.

    A vLLM spec paired with an openai one over a QA and an agentic benchmark would
    work if engines were chosen per benchmark, since _build_experiments already forks
    on that. They are not, and this pins the restriction so narrowing the check to
    only the agentic benchmarks' models is a test failure rather than a silent change.
    """
    with pytest.raises(ValueError, match="require the openai engine"):
        list(
            run_experiment_queries(
                model_specs=[_VLLM_SPEC, _model_spec()],
                exp_params=_pinned_exp_params(
                    [
                        _write_benchmark(tmp_path / "qa", agentic=False),
                        _write_benchmark(tmp_path / "agentic", agentic=True),
                    ]
                ),
                sampling_params=DataSamplingParams(),
                record_params=RecordHandlingParams(ReplacementStrategy.NEVER),
                datastore=Mock(),
                parallelize_models=False,
            )
        )


def test_warn_ignored_agentic_nshot(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """A requested --n-shot fan-out is warned about, since agentic pins n-shot=0."""
    bench = _write_benchmark(tmp_path / "agentic", agentic=True)
    with caplog.at_level("WARNING"):
        _warn_ignored_agentic_nshot(_exp_params([bench]))
    [record] = [r for r in caplog.records if "for agentic benchmarks" in r.message]
    assert "--n-shot=0,1" in record.message


def test_warn_ignored_agentic_nshot_silent_when_flags_match_pins(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """A plain agentic run whose n-shot equals the pin logs nothing."""
    bench = _write_benchmark(tmp_path / "agentic", agentic=True)
    with caplog.at_level("WARNING"):
        _warn_ignored_agentic_nshot(_pinned_exp_params([bench]))
    assert not [r for r in caplog.records if "for agentic benchmarks" in r.message]


def test_warn_ignored_agentic_nshot_silent_without_agentic(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """No agentic benchmark requested -> no warning even with an n-shot fan-out."""
    bench = _write_benchmark(tmp_path / "qa", agentic=False)
    with caplog.at_level("WARNING"):
        _warn_ignored_agentic_nshot(_exp_params([bench]))
    assert not [r for r in caplog.records if "for agentic benchmarks" in r.message]


def test_build_experiments_resolves_a_deployments_environment(tmp_path: Path) -> None:
    """The override reaches the experiment's config, the only record of where a run ran.

    Asserted on the built experiment rather than on the loader, since the summary
    is written from `experiment.benchmark_config` after the merge has happened.
    """
    bench = _write_benchmark(tmp_path / "agentic", agentic=True)
    deployment = AgenticEnvironmentConfig(
        type="kubernetes", config={"warmpool": "vulnerability-localization-pool"}
    )
    [experiment] = list(
        _build_experiments(
            [bench],
            dataclasses.replace(_exp_params([bench]), environment_config=deployment),
            PromptFormatter.BASE,
            _model_spec(),
            Mock(sub_store=Mock(return_value=Mock())),
        )
    )
    agentic_config = experiment.benchmark_config.agentic_config
    assert agentic_config is not None
    assert agentic_config.environment == deployment


@pytest.mark.parametrize(
    ("variables", "expected"),
    [
        ({"API_KEY": "fake-run-key", "LANG": "C"}, {"API_KEY": REDACTED, "LANG": "C"}),
        (["API_KEY=fake-run-key", "LANG=C"], [f"API_KEY={REDACTED}", "LANG=C"]),
    ],
    ids=["mapping", "docker-list"],
)
def test_run_record_redacts_environment_secrets_but_execution_receives_them(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
    variables: dict[str, str] | list[str],
    expected: dict[str, str] | list[str],
) -> None:
    """Exercise the real config merge and experiment.json writer, without a model call."""
    bench = _write_benchmark(tmp_path / "agentic", agentic=True)
    deployment = AgenticEnvironmentConfig(
        type="docker",
        config={
            "environment": variables,
            "nested": [{"AUTH_TOKEN": "fake-run-token"}],
        },
    )
    runner = Mock()
    with (
        StoreContext.from_path(str(tmp_path / "results")) as store,
        patch("faith.cli.subcmd.query._check_tool_calling"),
        patch(
            "faith.cli.subcmd.query.create_model",
            return_value=Mock(supported_formats={PromptFormatter.CHAT}),
        ),
        patch("faith.cli.subcmd.query.trial_runner_for", return_value=runner),
    ):
        [record_path] = _run_single_model(
            model_spec=_model_spec(),
            exp_params=dataclasses.replace(
                _pinned_exp_params([bench]), environment_config=deployment
            ),
            sampling_params=DataSamplingParams(),
            record_params=RecordHandlingParams(ReplacementStrategy.NEVER),
            datastore=store,
        )
    assert isinstance(record_path, Path)
    run_record = read_json_file(record_path)
    assert "fake-run-key" not in str(run_record) + caplog.text
    assert "fake-run-token" not in str(run_record) + caplog.text
    recorded = run_record["benchmark_config"]
    env = recorded["agentic_config"]["environment"]
    assert env["config"]["environment"] == expected
    assert env["config"]["nested"] == [{"AUTH_TOKEN": REDACTED}]
    # The eval command still needs to be able to decode a recorded config.
    assert BenchmarkConfig.from_dict(recorded).agentic_config is not None
    benchmark = runner.run_trial.call_args.args[0]
    assert benchmark.agentic_config.environment.config["environment"] == variables


def test_run_experiment_queries_rejects_an_environment_config_without_agentic(
    tmp_path: Path,
) -> None:
    """A QA-only run has nothing to apply the flag to, so it must not be ignored."""
    bench = _write_benchmark(tmp_path / "qa", agentic=False)
    exp_params = dataclasses.replace(
        _pinned_exp_params([bench]),
        environment_config=AgenticEnvironmentConfig(type="kubernetes"),
    )
    with pytest.raises(
        ValueError, match="--environment-config only applies to agentic benchmarks"
    ):
        list(
            run_experiment_queries(
                model_specs=[_model_spec()],
                exp_params=exp_params,
                sampling_params=DataSamplingParams(),
                record_params=RecordHandlingParams(ReplacementStrategy.NEVER),
                datastore=Mock(),
                parallelize_models=False,
            )
        )


def test_requested_benchmark_paths_dedups_overlapping_roots(tmp_path: Path) -> None:
    """Overlapping --custom-benchmarks roots yield each benchmark once.

    A duplicated agentic benchmark would run twice at full cost, so the de-dup is
    part of the "once per benchmark" contract, not a cosmetic detail.
    """
    bench = _write_benchmark(tmp_path / "nested" / "agentic", agentic=True)
    exp_params = _exp_params([tmp_path, tmp_path / "nested", bench])
    assert list(_requested_benchmark_paths(exp_params)) == [bench]


def _probe_model(
    tool_calls: list[str] | None = None,
    finish_reason: str = "stop",
    side_effect: Exception | None = None,
) -> Mock:
    """A model whose client answers the tool probe with the given response."""
    model = Mock(name_or_path="test-model", api_max_attempts=3, api_retry_sleep_secs=0)
    model.client.chat.completions.create = Mock(
        return_value=Mock(
            choices=[
                Mock(
                    message=Mock(tool_calls=tool_calls),
                    finish_reason=finish_reason,
                )
            ]
        ),
        side_effect=side_effect,
    )
    return model


def test_tool_calling_check_passes_when_the_endpoint_emits_a_call() -> None:
    """An endpoint that returns a tool call is what the agentic loop needs."""
    _check_tool_calling(
        _probe_model(tool_calls=["probe"], finish_reason="tool_calls"), GenParams()
    )


def test_tool_calling_check_requires_the_call_rather_than_offering_a_tool() -> None:
    """The probe forces a call, so an empty response cannot be a healthy refusal.

    Under tool_choice="auto" a correctly configured model may legitimately decline
    a throwaway probe tool, which is indistinguishable from being unable to call
    one — and rejecting the run on that basis has failed working servers.
    """
    model = _probe_model(tool_calls=["probe"], finish_reason="tool_calls")
    _check_tool_calling(model, GenParams())
    assert model.client.chat.completions.create.call_args.kwargs["tool_choice"] == (
        "required"
    )


def test_tool_calling_check_rejects_an_endpoint_that_emits_no_call() -> None:
    """No tool call when one was required means the endpoint cannot make them.

    This is the whole point of the check: without it the run completes, every task
    stops with NO_TOOL_CALLS, and metrics.json reports 0.0 as if it were measured.
    """
    with pytest.raises(ValueError, match="returned none when required to"):
        _check_tool_calling(_probe_model(tool_calls=None), GenParams())


def test_tool_calling_check_warns_rather_than_fails_on_a_truncated_probe(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Truncation says nothing about tool support, so it must not fail the run.

    A reasoning model can think past the probe's budget without ever reaching its
    call. Failing here is precisely the mistake that rejected a working server.
    """
    with caplog.at_level("WARNING"):
        _check_tool_calling(
            _probe_model(tool_calls=None, finish_reason="length"), GenParams()
        )

    assert [r.getMessage() for r in caplog.records] == [
        f"Could not confirm 'test-model' emits tool calls: the probe was truncated "
        f"at {_TOOL_PROBE_MAX_TOKENS} tokens before producing one. Continuing, since "
        f"an agentic run's own budget is larger."
    ]


def test_tool_calling_check_surfaces_a_failed_probe_request() -> None:
    """A server rejecting the probe outright is reported with its own error.

    A vLLM served without tool support rejects tool_choice="required" rather than
    answering emptily, so the underlying message is the operator's best clue.
    """
    with pytest.raises(ValueError, match="probing 'test-model' for one failed"):
        _check_tool_calling(
            _probe_model(side_effect=RuntimeError("tool_choice required unsupported")),
            GenParams(),
        )


def _unsupported_parameter_error() -> BadRequestError:
    """The 400 an OpenAI reasoning model returns for `max_tokens` (job 1887).

    Built through the SDK so `param` is populated the way a live rejection
    populates it, rather than by transcribing attributes the probe reads.
    """
    body = {
        "message": (
            "Unsupported parameter: 'max_tokens' is not supported with this model. "
            "Use 'max_completion_tokens' instead."
        ),
        "type": "invalid_request_error",
        "param": "max_tokens",
        "code": "unsupported_parameter",
    }
    request = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")
    return BadRequestError(
        f"Error code: 400 - {{'error': {body}}}",
        response=httpx.Response(400, json={"error": body}, request=request),
        body=body,
    )


def test_the_probe_request_uses_the_reasoning_model_token_parameter() -> None:
    """The probe sends max_completion_tokens, never max_tokens.

    Reasoning models reject `max_tokens` outright, so sending it made the check
    that gates agentic runs reject the whole gpt-5 and o-series families.
    """
    model = _probe_model(tool_calls=["probe"], finish_reason="tool_calls")
    _check_tool_calling(model, GenParams(temperature=1.0, max_completion_tokens=64))

    kwargs = model.client.chat.completions.create.call_args.kwargs
    assert "max_tokens" not in kwargs
    assert kwargs["max_completion_tokens"] == _TOOL_PROBE_MAX_TOKENS
    assert kwargs["temperature"] == 1.0


def test_the_probe_omits_a_sentinel_token_budget_from_the_generation_config() -> None:
    """A -1 max_completion_tokens is the run's "omit" sentinel, not a budget.

    The probe substitutes its own budget rather than forwarding the sentinel, which
    the API rejects.
    """
    model = _probe_model(tool_calls=["probe"], finish_reason="tool_calls")
    _check_tool_calling(model, GenParams(max_completion_tokens=-1))

    kwargs = model.client.chat.completions.create.call_args.kwargs
    assert kwargs["max_completion_tokens"] == _TOOL_PROBE_MAX_TOKENS


def test_a_parameter_rejection_is_reported_as_a_faith_bug_not_missing_tools() -> None:
    """A 400 blaming a parameter faith sent must not be read as no tool support.

    Sending the wrong parameter name told operators of a perfectly tool-capable
    endpoint to go fix their chat template and tool-call parser (job 1887).
    """
    error = _unsupported_parameter_error()
    with pytest.raises(ValueError) as excinfo:
        _check_tool_calling(_probe_model(side_effect=error), GenParams())

    message = str(excinfo.value)
    assert message == (
        "Probing 'test-model' for tool calls failed because faith sent a parameter "
        "the model does not support: max_tokens. This is a faith bug, not a problem "
        "with the endpoint's tool support: BadRequestError (HTTP 400) "
        "code=unsupported_parameter param=max_tokens"
    )
    assert _TOOL_PROBE_REMEDY not in message


def test_a_rejected_tool_parameter_is_still_reported_as_missing_tool_support() -> None:
    """A 400 blaming `tools` or `tool_choice` is the verdict the probe exists for.

    Such a server really cannot emit tool calls, so the remedy that names the
    parser and chat template is the right advice for it.
    """
    error = BadRequestError(
        "Error code: 400 - tool_choice is not supported",
        response=httpx.Response(
            400,
            json={"error": {"message": "no", "param": "tool_choice"}},
            request=httpx.Request("POST", "https://vllm.invalid/v1/chat/completions"),
        ),
        body={"message": "no", "param": "tool_choice"},
    )
    with pytest.raises(ValueError) as excinfo:
        _check_tool_calling(_probe_model(side_effect=error), GenParams())

    assert _TOOL_PROBE_REMEDY in str(excinfo.value)


def test_a_server_error_naming_a_parameter_is_not_blamed_on_faith() -> None:
    """Only a 4xx means faith sent something wrong; a 5xx is the server's own fault.

    An overloaded endpoint can echo request fields back in a 500, and calling that
    a faith bug would send the operator to the wrong place.
    """
    body = {"message": "upstream overloaded", "param": "temperature"}
    error = InternalServerError(
        "Error code: 503 - upstream overloaded",
        response=httpx.Response(
            503,
            json={"error": body},
            request=httpx.Request("POST", "https://vllm.invalid/v1/chat/completions"),
        ),
        body=body,
    )
    with pytest.raises(ValueError) as excinfo:
        _check_tool_calling(_probe_model(side_effect=error), GenParams())

    assert _TOOL_PROBE_REMEDY in str(excinfo.value)


def test_a_qa_only_run_is_not_probed_for_tool_calling(tmp_path: Path) -> None:
    """The probe costs an API call, so a run with no agentic benchmark skips it.

    Driven through _run_single_model rather than the check itself, since where the
    probe is gated is the thing at issue. The model is tool-blind, so a probe would
    have failed the run rather than passing it silently.
    """
    exp_dir = tmp_path / "exp"
    bench = _write_benchmark(tmp_path / "qa", agentic=False)
    model = _probe_model(tool_calls=None)
    model.supported_formats = {PromptFormatter.CHAT}
    model.tokenizer = None

    with (
        patch("faith.cli.subcmd.query.create_model", return_value=model),
        patch(
            "faith.cli.subcmd.query.trial_runner_for",
            return_value=Mock(run_trial=Mock(return_value=None)),
        ),
        patch(
            "faith.cli.subcmd.query.BenchmarkExperiment",
            return_value=Mock(
                datastore=Mock(path=exp_dir),
                benchmark_spec=Mock(to_dict=Mock(return_value={"name": "test"})),
                benchmark_config=Mock(to_dict=Mock(return_value={})),
                __iter__=Mock(return_value=iter([])),
            ),
        ),
    ):
        assert list(
            _run_single_model(
                model_spec=_model_spec(),
                exp_params=_pinned_exp_params([bench]),
                sampling_params=DataSamplingParams(),
                record_params=RecordHandlingParams(ReplacementStrategy.NEVER),
                datastore=Mock(),
            )
        ) == [exp_dir / "experiment.json"]

    assert model.client.chat.completions.create.call_args_list == []


def test_run_experiment_queries_parallel_requires_vllm(tmp_path: Path) -> None:
    exp_params = ExperimentParams(
        benchmark_names=[],
        custom_benchmark_paths=[_write_benchmark(tmp_path / "qa", agentic=False)],
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
