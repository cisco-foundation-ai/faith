# Copyright 2025 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

"""Query-side stage that runs the agent loop and records its trajectory."""

import logging
from collections.abc import Iterable

from faith._internal.iter.transform import IsoTransform
from faith._types.record.agent import AgentResult, StoppedReason
from faith._types.record.model_response import GenerationError
from faith._types.record.sample import SampleRecord
from faith.agentic.core import ReActLoop, Task

logger = logging.getLogger(__name__)

# Separate titles so the log distinguishes a broken cluster from a broken run.
_ERROR_TITLES = {
    StoppedReason.ERROR: "Agent Run Error",
    StoppedReason.ENVIRONMENT_ERROR: "Agent Environment Error",
}


def _errored_trajectory(task_id: str, error: str) -> AgentResult:
    """An empty trajectory recording that the task never ran."""
    return AgentResult(
        task_id=task_id,
        answer=None,
        messages=[],
        tool_calls=[],
        tool_responses=[],
        num_tool_calls=0,
        execution_time_seconds=0.0,
        stopped_reason=StoppedReason.ERROR,
        error=error,
    )


def _record_trajectory(record: SampleRecord, result: AgentResult) -> None:
    """Store the trajectory, recording a failed run as a GenerationError.

    Only a failure reason qualifies, so the aggregator withholds a harness or
    infrastructure failure from the accuracy metrics; hitting a limit is a normal
    incomplete result.
    """
    record.model_data.agent_trajectory = result
    record.model_data.error = (
        GenerationError(
            title=_ERROR_TITLES[result.stopped_reason], details=result.error
        )
        if result.stopped_reason.is_failure
        else None
    )


class AgentLoopRunner(IsoTransform[SampleRecord]):
    """Transform that runs the agent loop for each record and stores its trajectory.

    The agentic analog of `model_querier`: muxed onto stale records only, it looks
    each record's task up by its benchmark sample index and writes the resulting
    AgentResult into `model_data.agent_trajectory`.
    """

    def __init__(self, loop: ReActLoop, tasks_by_index: dict[int, Task]):
        self._loop = loop
        self._tasks_by_index = tasks_by_index

    def __call__(self, records: Iterable[SampleRecord]) -> Iterable[SampleRecord]:
        """Run the agent loop for each record, populating its trajectory."""
        for record in records:
            index = record.data.benchmark_sample_index
            task = self._tasks_by_index.get(index)
            # This stage runs in a worker thread whose exceptions are not
            # re-raised, so an unhandled one would silently drop the row from the
            # trial log; an ERROR trajectory is retried on the next resume instead.
            try:
                if task is None:
                    raise KeyError(f"No task loaded for sample index {index}")
                _record_trajectory(record, self._loop.run(task))
            except Exception as e:  # pylint: disable=broad-exception-caught
                logger.error("Agent run for task index %d failed: %s", index, e)
                _record_trajectory(
                    record,
                    _errored_trajectory(
                        task.id if task else str(index),
                        f"Agent run failed: {type(e).__name__}: {e}",
                    ),
                )
            yield record
