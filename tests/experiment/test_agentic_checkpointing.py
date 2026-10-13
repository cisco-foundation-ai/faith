# Copyright 2025 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

"""Agentic checkpoint/resume behavior through the real trial pipeline."""

import json
from dataclasses import replace
from unittest.mock import Mock, patch

import httpx
import pytest

from faith._internal.algo.hash import dict_sha256
from faith._internal.io.json import write_as_json
from faith._types.config.scoring import OutputProcessingConfig, ScoreFnConfig
from faith._types.model.generation import GenParams
from faith._types.record.agent import StoppedReason
from faith._types.record.sample import ReplacementStrategy
from faith.agentic.core import AnswerExtractor, Task, ToolRegistry
from faith.agentic.tools.submit import SubmitTool
from faith.benchmark.grading.agentic_graders import AgentTrajectoryLogGrader
from faith.experiment.params import DataSamplingParams
from faith.experiment.trial_runner import AgenticTrialRunner
from faith.record_pipelines.params import RecordHandlingParams
from faith.record_pipelines.trial_log import read_trial_log
from tests.experiment.test_trial_runner import _agentic_benchmark, _run_agentic_trial
from tests.model.test_api_retry import _completion, _Environment, _model


def _tasks():
    return [Task(id=name, prompt=f"do {name}", ground_truth=[name]) for name in "ABC"]


def _interrupt_b(task):
    if task.id == "B":
        raise KeyboardInterrupt("interrupted during B")


def test_complete_task_is_on_disk_before_the_next_task_starts(tmp_path):
    benchmark = _agentic_benchmark(_tasks())
    path = tmp_path / "benchmark-log.json"

    def check_and_interrupt(task):
        if task.id == "B":
            (saved,) = read_trial_log(path)
            assert saved.model_data.agent_trajectory is not None
            assert saved.model_data.agent_trajectory.task_id == "A"
            assert saved.metadata.model_data_hash == dict_sha256(
                saved.model_data.to_dict()
            )
            _interrupt_b(task)

    with pytest.raises(KeyboardInterrupt, match="during B"):
        _run_agentic_trial(benchmark, path, before_run=check_and_interrupt)

    executed: list[str] = []
    _run_agentic_trial(benchmark, path, run_log=executed)
    assert executed == ["B", "C"]
    records = list(read_trial_log(path))
    assert [record.data.benchmark_sample_index for record in records] == [0, 1, 2]

    # A resumed log remains gradable with identical results to an uninterrupted run.
    other_path = tmp_path / "uninterrupted.json"
    _run_agentic_trial(benchmark, other_path)
    grader = AgentTrajectoryLogGrader(
        OutputProcessingConfig(score_fns={"match": ScoreFnConfig(type="jaccard")}),
        extractor=Mock(spec=AnswerExtractor, extract=lambda _tool, answer: [answer]),
    )
    resumed_stats = [record.stats for record in records >> grader]
    assert resumed_stats == [
        record.stats for record in read_trial_log(other_path) >> grader
    ]
    for stats in resumed_stats:
        assert stats is not None and stats.scores is not None
        assert stats.scores["match"]["value"] == 1.0


def test_second_interruption_preserves_later_fresh_and_failed_records(tmp_path):
    benchmark = _agentic_benchmark(_tasks())
    path = tmp_path / "benchmark-log.json"
    _run_agentic_trial(benchmark, path, raise_on="B")
    previous = list(read_trial_log(path))
    assert previous[1].model_data.error is not None

    with pytest.raises(KeyboardInterrupt, match="during B"):
        _run_agentic_trial(benchmark, path, before_run=_interrupt_b)

    # Includes the previous failure details/hash, not a partially reset B record.
    assert list(read_trial_log(path)) == previous
    executed: list[str] = []
    _run_agentic_trial(benchmark, path, run_log=executed)
    assert executed == ["B"]


def test_task_failure_is_checkpointed_before_the_next_task(tmp_path):
    benchmark = _agentic_benchmark(_tasks())
    path = tmp_path / "benchmark-log.json"
    with pytest.raises(KeyboardInterrupt, match="during B"):
        _run_agentic_trial(benchmark, path, raise_on="A", before_run=_interrupt_b)
    (record,) = read_trial_log(path)
    assert record.model_data.error is not None
    assert record.model_data.agent_trajectory is not None
    assert record.model_data.agent_trajectory.error is not None
    assert record.model_data.error.title == "Agent Run Error"
    assert record.model_data.agent_trajectory.stopped_reason == StoppedReason.ERROR
    assert "environment blew up" in record.model_data.agent_trajectory.error
    assert record.metadata.model_data_hash == dict_sha256(record.model_data.to_dict())
    executed: list[str] = []
    _run_agentic_trial(benchmark, path, run_log=executed)
    assert executed == ["A", "B", "C"]


def _assert_failed_api_checkpoint(record):
    trajectory = record.model_data.agent_trajectory
    assert trajectory.stopped_reason == StoppedReason.ERROR
    assert "2 total attempts" in trajectory.error and "HTTP 500" in trajectory.error
    assert any(message.get("content") == "done" for message in trajectory.messages)
    assert record.metadata.model_data_hash == dict_sha256(record.model_data.to_dict())


def test_late_api_failure_preserves_checkpoints_and_resumes_only_failed_tasks(tmp_path):
    benchmark = _agentic_benchmark(_tasks())
    path = tmp_path / "benchmark-log.json"
    requests = []
    resuming = False

    def handler(request):
        messages = json.loads(request.content)["messages"]
        task = next(m["content"] for m in messages if m["role"] == "user")[-1]
        requests.append(task)
        if not resuming and task == "B":
            saved = list(read_trial_log(path))[0].model_data.agent_trajectory
            assert saved is not None and saved.task_id == "A"
            if requests.count("B") == 1:
                return httpx.Response(200, json=_completion(submit=False))
            return httpx.Response(500, json={"error": {"code": "server_error"}})
        if not resuming and task == "C":
            raise KeyboardInterrupt("after checkpointing B")
        return httpx.Response(200, json=_completion())

    with (
        _model(handler, attempts=2) as model,
        patch("time.sleep") as sleep,
        patch(
            "faith.experiment.trial_runner.build_environment",
            return_value=_Environment(),
        ),
        patch(
            "faith.experiment.trial_runner.build_tool_registry",
            return_value=ToolRegistry([SubmitTool()]),
        ),
    ):

        def run():
            AgenticTrialRunner().run_trial(
                benchmark,
                path,
                model=model,
                generation=GenParams(),
                sampling_params=DataSamplingParams(),
                record_params=RecordHandlingParams(ReplacementStrategy.IF_HASH_DIFFERS),
            )

        with pytest.raises(KeyboardInterrupt, match="after checkpointing B"):
            run()
        assert requests == ["A", "B", "B", "B", "C"]
        assert [call.args for call in sleep.call_args_list] == [(0.25,)]
        saved = list(read_trial_log(path))
        assert len(saved) == 2
        _assert_failed_api_checkpoint(saved[1])
        requests.clear()
        resuming = True
        run()
    assert requests == ["B", "C"]
    final = list(read_trial_log(path))
    assert len(final) == 3 and final[0] == saved[0]
    assert all(record.model_data.error is None for record in final)


@pytest.mark.parametrize("failure_point", ["fsync", "replace"])
def test_failed_checkpoint_preserves_old_log_and_stops_execution(
    tmp_path, monkeypatch, failure_point
):
    benchmark = _agentic_benchmark(_tasks())
    path = tmp_path / "benchmark-log.json"
    _run_agentic_trial(benchmark, path, raise_on="B")
    previous = path.read_bytes()

    def fail_write(*_args):
        raise OSError("checkpoint disk failure")

    def fail_during_b(task):
        if task.id == "B":
            monkeypatch.setattr(
                f"faith._internal.io.json.os.{failure_point}", fail_write
            )

    executed: list[str] = []
    with pytest.raises(OSError, match="checkpoint disk failure"):
        _run_agentic_trial(benchmark, path, run_log=executed, before_run=fail_during_b)
    assert executed == ["B"]  # No C, and the failure was not swallowed as a task error.
    assert path.read_bytes() == previous
    assert list(tmp_path.iterdir()) == [
        path
    ]  # Atomic writer removed its temporary file.


@pytest.mark.parametrize("strategy", list(ReplacementStrategy))
@pytest.mark.parametrize("changed_prompt", [False, True])
def test_checkpointing_preserves_replacement_strategies(
    tmp_path, strategy, changed_prompt
):
    benchmark = _agentic_benchmark(_tasks())
    path = tmp_path / "benchmark-log.json"
    _run_agentic_trial(benchmark, path)
    if changed_prompt:
        # The task/sample identity stays the same but data_hash changes.
        benchmark.agentic_config.system_prompt = "A different standing instruction."
    executed: list[str] = []
    _run_agentic_trial(benchmark, path, replacement_strategy=strategy, run_log=executed)
    should_run = strategy == ReplacementStrategy.ALWAYS or (
        strategy == ReplacementStrategy.IF_HASH_DIFFERS and changed_prompt
    )
    assert executed == (["A", "B", "C"] if should_run else [])
    assert len(list(read_trial_log(path))) == 3


@pytest.mark.parametrize("reason", list(StoppedReason))
def test_checkpointing_preserves_terminal_outcome_freshness(tmp_path, reason):
    benchmark = _agentic_benchmark(_tasks()[:1])
    path = tmp_path / "benchmark-log.json"
    _run_agentic_trial(benchmark, path)
    (record,) = read_trial_log(path)
    assert record.model_data.agent_trajectory is not None
    record.model_data.agent_trajectory = replace(
        record.model_data.agent_trajectory,
        stopped_reason=reason,
        error="failure" if reason.is_failure else None,
    )
    write_as_json(path, [record])
    executed: list[str] = []
    _run_agentic_trial(benchmark, path, run_log=executed)
    assert executed == (["A"] if reason.is_failure else [])


def test_changed_selection_retains_old_records_until_success_then_prunes(tmp_path):
    tasks = _tasks()
    benchmark = _agentic_benchmark(tasks)
    path = tmp_path / "benchmark-log.json"
    _run_agentic_trial(benchmark, path, raise_on="B")
    # New task D first, then failed B; original dataset positions are retained.
    benchmark.load_tasks.return_value = [
        (3, Task(id="D", prompt="do D", ground_truth=["D"])),
        (1, tasks[1]),
    ]
    with pytest.raises(KeyboardInterrupt, match="during B"):
        _run_agentic_trial(benchmark, path, before_run=_interrupt_b)
    assert [r.data.benchmark_sample_index for r in read_trial_log(path)] == [0, 1, 2, 3]
    executed: list[str] = []
    _run_agentic_trial(benchmark, path, run_log=executed)
    assert executed == ["B"]
    assert [r.data.benchmark_sample_index for r in read_trial_log(path)] == [1, 3]


@pytest.mark.parametrize("strategy", list(ReplacementStrategy))
def test_changed_task_at_same_index_replaces_without_duplicate_records(
    tmp_path, strategy
):
    benchmark = _agentic_benchmark(_tasks())
    path = tmp_path / "benchmark-log.json"
    _run_agentic_trial(benchmark, path)
    # Same position, different sample hash: never reuse the old task's answer,
    # even under NEVER (which only applies to records with matching identities).
    benchmark.load_tasks.return_value = [
        (0, Task(id="new", prompt="different task", ground_truth=["new"]))
    ]
    executed: list[str] = []
    _run_agentic_trial(benchmark, path, replacement_strategy=strategy, run_log=executed)
    assert executed == ["new"]
    (record,) = read_trial_log(path)
    assert record.data.benchmark_sample_index == 0
    assert record.model_data.agent_trajectory is not None
    assert record.model_data.agent_trajectory.task_id == "new"


@pytest.mark.parametrize("has_previous_log", [False, True])
def test_empty_selection_matches_existing_logger_behavior(tmp_path, has_previous_log):
    benchmark = _agentic_benchmark(_tasks())
    path = tmp_path / "benchmark-log.json"
    if has_previous_log:
        _run_agentic_trial(benchmark, path)
    previous = path.read_bytes() if has_previous_log else None
    benchmark.load_tasks.return_value = []
    _run_agentic_trial(benchmark, path)
    assert (path.read_bytes() if path.exists() else None) == previous
