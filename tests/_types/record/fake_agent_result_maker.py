# Copyright 2025 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

from faith._types.json_types import JsonValue
from faith._types.record.agent import AgentResult, StoppedReason


def make_fake_agent_result(
    answer: JsonValue,
    stopped_reason: StoppedReason = StoppedReason.SUBMITTED,
    error: str | None = None,
) -> AgentResult:
    """Create a fake single-submission `AgentResult` for testing purposes."""
    return AgentResult(
        task_id="task-1",
        answer=answer,
        messages=[{"role": "user", "content": "go"}],
        tool_calls=[
            {"tool_name": "submit", "timestamp": 1.0, "args": {"answer": answer}}
        ],
        tool_responses=[
            {
                "tool_name": "submit",
                "timestamp": 1.1,
                "response_truncated": "ok",
                "response_length": len("ok"),
            }
        ],
        num_tool_calls=1,
        execution_time_seconds=1.5,
        stopped_reason=stopped_reason,
        error=error,
    )
