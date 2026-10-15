# Copyright 2025 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for the agent-loop query stage."""

from unittest.mock import Mock

import pytest

from faith._types.record.agent import AgentResult, StoppedReason
from faith._types.record.model import ModelRecord
from faith._types.record.model_response import GenerationError
from faith._types.record.sample import SampleRecord
from faith.agentic.core import ReActLoop, ToolRegistry
from faith.agentic.core.environment import Environment
from faith.agentic.core.task import Task
from faith.record_pipelines.agentic_execution import AgentLoopRunner
from faith.record_pipelines.agentic_formatting import TaskFormatter
from tests._types.record.fake_agent_result_maker import (
    make_fake_agent_result as _result,
)

_TASK = Task(id="task-1", prompt="Localize the vulnerability.", ground_truth=["a.c"])


def _record(task: Task = _TASK, *, index: int = 0) -> SampleRecord:
    """The bare record the formatter hands this stage, built as the pipeline does."""
    (record,) = list(iter({index: task}.items()) >> TaskFormatter("v1", None))
    return record


def _run_returning(result: AgentResult) -> ModelRecord:
    """Run one record through the stage with a loop returning `result`."""
    loop = Mock()
    loop.run.return_value = result
    (out,) = list([_record()] >> AgentLoopRunner(loop, {0: _TASK}))
    return out.model_data


def test_agent_loop_runner_runs_loop_for_stale_records() -> None:
    """AgentLoopRunner looks each record's task up by index, runs the loop, and stores the trajectory."""
    result = _result(["a.c"])
    loop = Mock()
    loop.run.return_value = result

    (out,) = list([_record()] >> AgentLoopRunner(loop, {0: _TASK}))

    loop.run.assert_called_once_with(_TASK)
    assert out.model_data.agent_trajectory == result


def test_an_errored_run_records_a_generation_error() -> None:
    """Test the harness failure reaches the grader as QA's generation error."""
    assert _run_returning(
        _result(None, StoppedReason.ERROR, error="container died")
    ).error == GenerationError(title="Agent Run Error", details="container died")


@pytest.mark.parametrize(
    "stopped_reason",
    [
        StoppedReason.SUBMITTED,
        StoppedReason.NO_TOOL_CALLS,
        StoppedReason.MAX_CALLS,
        StoppedReason.MAX_TURNS,
        StoppedReason.TIMEOUT,
    ],
)
def test_a_normal_termination_records_no_error(stopped_reason: StoppedReason) -> None:
    # Hitting a limit is an incomplete answer, not a failure: marking it errored
    # would withhold it from the accuracy metrics it belongs in.
    assert _run_returning(_result(None, stopped_reason)).error is None


def _errored_trajectory(task_id: str, details: str) -> AgentResult:
    """The salvage trajectory the stage substitutes for a run it could not make."""
    return AgentResult(
        task_id=task_id,
        answer=None,
        messages=[],
        tool_calls=[],
        tool_responses=[],
        num_tool_calls=0,
        execution_time_seconds=0.0,
        stopped_reason=StoppedReason.ERROR,
        error=details,
    )


def test_a_loop_that_raises_is_salvaged_as_an_errored_trajectory() -> None:
    """A raised exception is stored as an errored trajectory AND an error.

    Without the error the salvaged row would be graded as a wrong answer; the
    trajectory preserves the failure details so resume can retry the task.
    """
    loop = Mock()
    loop.run.side_effect = RuntimeError("boom")

    (out,) = list([_record()] >> AgentLoopRunner(loop, {0: _TASK}))

    details = "Agent run failed: RuntimeError: boom"
    assert out.model_data.error == GenerationError(
        title="Agent Run Error", details=details
    )
    assert out.model_data.agent_trajectory == _errored_trajectory("task-1", details)


def test_a_record_with_no_loaded_task_is_salvaged_too() -> None:
    """An index carrying no task takes the same salvage path, keyed by the index.

    There is no Task to read an id from, so the trajectory falls back to the
    index; a resume retries the row rather than losing it.
    """
    loop = Mock()

    (out,) = list([_record()] >> AgentLoopRunner(loop, {}))

    details = "Agent run failed: KeyError: 'No task loaded for sample index 0'"
    loop.run.assert_not_called()
    assert out.model_data.error == GenerationError(
        title="Agent Run Error", details=details
    )
    assert out.model_data.agent_trajectory == _errored_trajectory("0", details)


def test_every_record_is_yielded_even_when_one_fails() -> None:
    """A failed row still reaches the log, in input order, so resume can retry it.

    An ordinary failure must not prevent later tasks from executing.
    """
    tasks = {i: Task(id=f"t{i}", prompt="p", ground_truth=None) for i in range(3)}
    records = [_record(task, index=i) for i, task in tasks.items()]
    loop = Mock()
    loop.run.side_effect = lambda task: (
        _result(None, StoppedReason.ERROR, error="died")
        if task.id == "t1"
        else _result([task.id])
    )

    out = list(records >> AgentLoopRunner(loop, tasks))

    assert [r.data.benchmark_sample_index for r in out] == [0, 1, 2]
    trajectories = [r.model_data.agent_trajectory for r in out]
    assert [t.stopped_reason for t in trajectories if t is not None] == [
        StoppedReason.SUBMITTED,
        StoppedReason.ERROR,
        StoppedReason.SUBMITTED,
    ]


class _DeadEnvironment(Environment):
    """Environment that sets up and tears down without doing anything."""

    def setup(self, task: Task) -> None:
        pass

    def teardown(self) -> None:
        pass


def test_an_unreachable_api_quarantines_the_row_end_to_end() -> None:
    """A run whose API never recovers is excluded from grading, not scored wrong.

    The only test driving a real ReActLoop through this stage: everything else
    here substitutes a Mock loop, so this is what covers the two agreeing on the
    shape of a failed run. Each half is covered in isolation (the loop's ERROR
    termination, the stage's GenerationError); only together do they make a
    broken run ungradeable.
    """
    model_api = Mock()
    model_api.chat.completions.create = Mock(side_effect=ConnectionError("down"))
    loop = ReActLoop(
        model_api=model_api,
        model="test-model",
        environment=_DeadEnvironment(),
        tool_registry=ToolRegistry(),
        api_max_attempts=2,
        api_retry_sleep_secs=0,
    )

    (out,) = list([_record()] >> AgentLoopRunner(loop, {0: _TASK}))

    trajectory = out.model_data.agent_trajectory
    assert trajectory is not None
    assert trajectory.stopped_reason is StoppedReason.ERROR
    assert model_api.chat.completions.create.call_count == 2
    assert out.model_data.error is not None
    assert out.model_data.error.title == "Agent Run Error"
    assert "Model API call failed" in (out.model_data.error.details or "")
