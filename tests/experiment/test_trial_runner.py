# Copyright 2025 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Callable
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from faith._internal.io.json import write_as_json
from faith._types.config.agentic import AgenticLimitsConfig
from faith._types.config.patterns import AnswerFormat
from faith._types.model.generation import GenParams
from faith._types.record.agent import AgentResult, StoppedReason
from faith._types.record.sample import ReplacementStrategy, SampleRecord
from faith._types.record.stats import StatsRecord
from faith.agentic.core.task import Task
from faith.benchmark.agentic import AgenticBenchmark
from faith.benchmark.benchmark import QABenchmark
from faith.experiment.trial_runner import (
    AgenticTrialRunner,
    QATrialRunner,
    TrialRunner,
    trial_runner_for,
)
from faith.record_pipelines.params import RecordHandlingParams
from faith.record_pipelines.trial_log import read_trial_log

# Note: the per-trial pipeline itself (build_dataset >> format >> query >> log)
# is covered by the individual record_pipelines tests and test_formatting; these
# tests cover the runner dispatch that selects which pipeline to run.

# Every ReActLoop budget the runner is responsible for forwarding.
_LOOP_BUDGET_KEYS = (
    "max_tool_calls",
    "max_turns",
    "max_consecutive_no_tool_calls",
    "timeout_seconds",
    "recover_on_environment_failure",
    "api_max_attempts",
    "api_retry_sleep_secs",
)

_SYSTEM_PROMPT = "You are a security researcher."

_LIMITS = AgenticLimitsConfig(
    max_tool_calls=7,
    max_turns=9,
    max_consecutive_no_tool_calls=2,
    timeout_seconds=30.0,
    recover_on_environment_failure=False,
)

# Every value below is distinct from every other and from the ReActLoop default
# for the same field, so a budget cross-wired to the wrong field cannot pass.
_MODEL = {
    "name_or_path": "test-model",
    "api_max_attempts": 4,
    "api_retry_sleep_secs": 0.25,
}


@pytest.mark.parametrize(
    ("benchmark", "expected"),
    [
        (Mock(spec=QABenchmark), QATrialRunner),
        (Mock(spec=AgenticBenchmark), AgenticTrialRunner),
    ],
    ids=["qa", "agentic"],
)
def test_trial_runner_for_dispatches_on_benchmark_type(
    benchmark: Mock, expected: type
) -> None:
    """Each benchmark type dispatches to the runner that knows its pipeline."""
    runner = trial_runner_for(benchmark)
    assert isinstance(runner, expected)
    assert isinstance(runner, TrialRunner)


def test_trial_runner_for_rejects_unknown_benchmark() -> None:
    """A benchmark with no matching runner raises a helpful error."""
    with pytest.raises(ValueError, match="No trial runner available"):
        trial_runner_for(Mock())


def _agentic_benchmark(tasks: list[Task]) -> Mock:
    """A stand-in AgenticBenchmark that loads `tasks` and carries an agentic config."""
    benchmark = Mock(spec=AgenticBenchmark)
    benchmark.version = "1.0"
    benchmark.load_tasks.return_value = list(enumerate(tasks))
    benchmark.agentic_config = Mock(
        environment=Mock(),
        tools=[],
        system_prompt=_SYSTEM_PROMPT,
        limits=_LIMITS,
    )
    return benchmark


def _run_agentic_trial(
    benchmark: Mock,
    trial_log_path: Path,
    generation: GenParams | None = None,
    replacement_strategy: ReplacementStrategy = ReplacementStrategy.NEVER,
    run_log: list[str] | None = None,
    raise_on: str | None = None,
    model: Mock | None = None,
    before_run: Callable[[Task], None] | None = None,
) -> dict:
    """Run the agentic runner with orchestration builders stubbed to fakes.

    Returns the kwargs `ReActLoop` was constructed with, so tests can assert what
    the runner passed through (e.g. the generation params). If `run_log` is
    given, the id of each task the loop actually runs is appended to it, so tests
    can assert which tasks were (re)run rather than passed through fresh. If
    `raise_on` names a task id, the loop raises on that task. `before_run` can
    observe a checkpoint or interrupt execution at a deterministic task boundary.
    """

    # The loop yields a canned SUBMITTED result echoing the task id as the answer.
    def fake_run(task: Task) -> AgentResult:
        if run_log is not None:
            run_log.append(task.id)
        if before_run is not None:
            before_run(task)
        if task.id == raise_on:
            raise RuntimeError("environment blew up")
        return AgentResult(
            task_id=task.id,
            answer=task.id,
            messages=[],
            tool_calls=[],
            tool_responses=[],
            num_tool_calls=0,
            execution_time_seconds=0.0,
            stopped_reason=StoppedReason.SUBMITTED,
        )

    captured: dict = {}

    def fake_react_loop(*_args, **kwargs) -> Mock:
        captured.update(kwargs)
        return Mock(run=fake_run)

    with (
        patch("faith.experiment.trial_runner.build_environment", lambda cfg: Mock()),
        patch(
            "faith.experiment.trial_runner.build_tool_registry", lambda names: Mock()
        ),
        patch("faith.experiment.trial_runner.ReActLoop", fake_react_loop),
    ):
        AgenticTrialRunner().run_trial(
            benchmark,
            trial_log_path,
            model=model or Mock(client=Mock(), **_MODEL),
            generation=generation or GenParams(),
            sampling_params=Mock(),
            record_params=RecordHandlingParams(replacement_strategy),
        )
    return captured


def test_agentic_run_trial_writes_a_record_per_task(tmp_path) -> None:
    """The runner writes one trial-log record per task, carrying the trajectory."""
    tasks = [
        Task(id="t1", prompt="do t1", ground_truth=["a"]),
        Task(id="t2", prompt="do t2", ground_truth=["b"]),
    ]
    log_path = tmp_path / "benchmark-log.json"

    _run_agentic_trial(_agentic_benchmark(tasks), log_path)

    records = list(read_trial_log(log_path))
    assert [r.data.benchmark_sample_index for r in records] == [0, 1]
    assert [r.data.question for r in records] == ["do t1", "do t2"]
    assert [r.data.label for r in records] == [["a"], ["b"]]
    # The whole AgentResult rides in the agentic sibling field, unscored (stale).
    trajectories = [r.model_data.agent_trajectory for r in records]
    assert [t.answer for t in trajectories if t is not None] == ["t1", "t2"]
    assert all(r.stats is None for r in records)
    # HashModelDataTransform stamps a model_data_hash so IF_HASH_DIFFERS works.
    assert all(r.metadata.model_data_hash is not None for r in records)


@pytest.mark.parametrize(
    ("strategy", "expected_rerun"),
    [
        # NEVER: both tasks are fresh in the log, so no task is re-run.
        (ReplacementStrategy.NEVER, []),
        # ALWAYS: every task is re-run even though the log is complete.
        (ReplacementStrategy.ALWAYS, ["t1", "t2"]),
    ],
    ids=["never_resumes", "always_reruns"],
)
def test_agentic_run_trial_reruns_per_replacement_strategy(
    strategy: ReplacementStrategy, expected_rerun: list[str], tmp_path
) -> None:
    """A resume re-runs only the tasks the strategy says are stale."""
    tasks = [
        Task(id="t1", prompt="do t1", ground_truth=["a"]),
        Task(id="t2", prompt="do t2", ground_truth=["b"]),
    ]
    benchmark = _agentic_benchmark(tasks)
    log_path = tmp_path / "log.json"

    first_run: list[str] = []
    _run_agentic_trial(benchmark, log_path, run_log=first_run)
    assert sorted(first_run) == ["t1", "t2"]

    rerun: list[str] = []
    _run_agentic_trial(
        benchmark, log_path, replacement_strategy=strategy, run_log=rerun
    )

    assert sorted(rerun) == expected_rerun
    # Either way the log still holds one complete record per task.
    records = list(read_trial_log(log_path))
    assert [r.data.benchmark_sample_index for r in records] == [0, 1]
    trajectories = [r.model_data.agent_trajectory for r in records]
    assert [t.answer for t in trajectories if t is not None] == ["t1", "t2"]


@pytest.mark.parametrize(
    ("generation", "expected"),
    [
        (
            GenParams(
                temperature=0.3,
                top_p=0.9,
                max_completion_tokens=256,
                kwargs={"seed": 7},
            ),
            {
                "temperature": 0.3,
                "top_p": 0.9,
                "max_completion_tokens": 256,
                "seed": 7,
            },
        ),
        # -1 is the "omit, use the API default" sentinel shipped by reasoning
        # configs like o3; forwarding it verbatim would make the API reject them.
        (
            GenParams(temperature=0.3, top_p=0.9, max_completion_tokens=-1),
            {"temperature": 0.3, "top_p": 0.9},
        ),
    ],
    ids=["forwarded", "sentinel_omitted"],
)
def test_agentic_run_trial_passes_generation_params_to_loop(
    generation: GenParams, expected: dict, tmp_path
) -> None:
    """The user's generation params reach the loop under the QA key names.

    Without this the agent would silently use the API's defaults (e.g.
    temperature 1.0), defeating deterministic eval.
    """
    benchmark = _agentic_benchmark([Task(id="t1", prompt="p", ground_truth=["a"])])

    captured = _run_agentic_trial(
        benchmark, tmp_path / "log.json", generation=generation
    )

    assert captured["generation_params"] == expected


def test_agentic_run_trial_threads_the_configured_limits_and_retries(tmp_path) -> None:
    """The loop runs on the configured limits and the model spec's retry settings.

    The loop declares its own defaults for both, so anything the runner fails to
    forward is silently replaced rather than reported.
    """
    benchmark = _agentic_benchmark([Task(id="t1", prompt="p", ground_truth=["a"])])

    captured = _run_agentic_trial(benchmark, tmp_path / "log.json")

    assert {k: captured[k] for k in _LOOP_BUDGET_KEYS} == {
        "max_tool_calls": 7,
        "max_turns": 9,
        "max_consecutive_no_tool_calls": 2,
        "timeout_seconds": 30.0,
        "recover_on_environment_failure": False,
        "api_max_attempts": 4,
        "api_retry_sleep_secs": 0.25,
    }


def test_agentic_run_trial_threads_the_configured_system_prompt(tmp_path) -> None:
    """The benchmark's standing instructions reach the loop and the record.

    Same failure mode as the budgets: the loop defaults to no system prompt, so a
    prompt the runner drops means the agent silently runs unbriefed on a benchmark
    whose yaml says otherwise. The record needs it too, because data_hash is what
    makes an edited prompt re-run the task instead of reusing the old trajectory.
    """
    benchmark = _agentic_benchmark([Task(id="t1", prompt="p", ground_truth=["a"])])
    log_path = tmp_path / "log.json"

    captured = _run_agentic_trial(benchmark, log_path)

    assert captured["system_prompt"] == _SYSTEM_PROMPT
    assert [r.data.system_prompt for r in read_trial_log(log_path)] == [_SYSTEM_PROMPT]


@pytest.mark.parametrize(
    ("benchmark", "model", "message"),
    [
        # A non-openai model has no interactive client for the loop to drive.
        (
            _agentic_benchmark([Task(id="t1", prompt="p", ground_truth=None)]),
            Mock(spec=[]),
            "openai engine",
        ),
        (
            Mock(spec=QABenchmark),
            Mock(client=Mock(), **_MODEL),
            "requires an AgenticBenchmark",
        ),
    ],
    ids=["non_openai_model", "non_agentic_benchmark"],
)
def test_agentic_run_trial_rejects_an_unrunnable_trial(
    benchmark: Mock, model: Mock, message: str, tmp_path
) -> None:
    """The runner fails fast rather than part-running an unsupported trial."""
    with pytest.raises(AssertionError, match=message):
        AgenticTrialRunner().run_trial(
            benchmark,
            tmp_path / "log.json",
            model=model,
            generation=Mock(),
            sampling_params=Mock(),
            record_params=RecordHandlingParams(ReplacementStrategy.NEVER),
        )


def test_agentic_run_trial_records_a_failed_task_as_an_errored_trajectory(
    tmp_path,
) -> None:
    """A task whose loop raises still lands in the log, marked ERROR and retryable.

    Ordinary task failures remain visible and retryable rather than aborting
    the trial or silently dropping a row.
    """
    tasks = [
        Task(id="t1", prompt="do t1", ground_truth=["a"]),
        Task(id="t2", prompt="do t2", ground_truth=["b"]),
    ]
    benchmark = _agentic_benchmark(tasks)
    log_path = tmp_path / "log.json"

    _run_agentic_trial(benchmark, log_path, raise_on="t1")

    records = {r.data.benchmark_sample_index: r for r in read_trial_log(log_path)}
    assert sorted(records) == [0, 1]
    failed = records[0].model_data.agent_trajectory
    assert failed is not None
    assert failed.stopped_reason == StoppedReason.ERROR
    assert failed.error == "Agent run failed: RuntimeError: environment blew up"

    # The errored task is stale on resume; the successful one is not re-run.
    rerun: list[str] = []
    _run_agentic_trial(benchmark, log_path, run_log=rerun)
    assert rerun == ["t1"]


def test_agentic_run_trial_discards_stats_of_a_replaced_trajectory(tmp_path) -> None:
    """A re-run task's prior grades are dropped, not carried onto the new trajectory.

    persist_records re-stamps model_data_hash, so stats left in place would look
    current at eval time and report the failed run's scores.
    """
    benchmark = _agentic_benchmark([Task(id="t1", prompt="do t1", ground_truth=["a"])])
    log_path = tmp_path / "log.json"

    _run_agentic_trial(benchmark, log_path, raise_on="t1")
    graded: list[SampleRecord] = list(read_trial_log(log_path))
    graded[0].stats = StatsRecord(
        label=None, prediction=None, answer_format=AnswerFormat.INVALID
    )
    write_as_json(log_path, [r.to_dict() for r in graded])

    _run_agentic_trial(benchmark, log_path)

    records = list(read_trial_log(log_path))
    trajectories = [r.model_data.agent_trajectory for r in records]
    assert [t.answer for t in trajectories if t is not None] == ["t1"]
    assert [r.stats for r in records] == [None]
