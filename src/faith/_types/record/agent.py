# Copyright 2025 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

"""Record types for a single agentic run's outcome.

These live in the record layer (next to ModelRecord) rather than with the loop
that produces them, so ModelRecord can hold an AgentResult as its agentic
sibling field without the record layer importing up into the agentic package.
"""

from dataclasses import dataclass, field
from enum import auto
from typing import Any, NotRequired, TypedDict

from dataclasses_json import DataClassJsonMixin, config

from faith._types.enums import CIStrEnum
from faith._types.json_types import JsonValue


class ToolCallEntry(TypedDict):
    """One emitted tool call. `args` marks an executed call, `error` a rejected one."""

    tool_name: str
    timestamp: float
    args: NotRequired[dict[str, JsonValue]]
    raw_arguments: NotRequired[str]
    error: NotRequired[str]


class ToolResponseEntry(TypedDict):
    """The result string fed back for one tool call, truncated for the log.

    `response_length` is the untruncated length, so a payload-size question is
    answerable from this entry alone; the full string is in AgentResult.messages.
    """

    tool_name: str
    timestamp: float
    response_truncated: str
    response_length: int


class StoppedReason(CIStrEnum):
    """Why a ReActLoop run terminated.

    ERROR and ENVIRONMENT_ERROR are the two failure modes, and AgentResult.error is
    non-None if and only if the reason is one of them; every other member is a
    normal termination, a hit limit (TIMEOUT included) being a normal incomplete
    result. Both are excluded from accuracy, and they are separate because only
    ENVIRONMENT_ERROR says the cluster is what needs attention.

    Only SUBMITTED yields an answer; NO_TOOL_CALLS is a normal termination but
    not a success (the model replied with prose, and kept doing so after being
    nudged, instead of calling submit).
    """

    SUBMITTED = auto()
    NO_TOOL_CALLS = auto()
    MAX_CALLS = auto()
    MAX_TURNS = auto()
    TIMEOUT = auto()
    ERROR = auto()
    ENVIRONMENT_ERROR = auto()

    @property
    def is_failure(self) -> bool:
        """Whether this reason means the run failed rather than finished."""
        return self in (StoppedReason.ERROR, StoppedReason.ENVIRONMENT_ERROR)


@dataclass
class AgentResult(DataClassJsonMixin):
    """The persisted output of running an agent on one task.

    The final answer, the full trajectory, and why the run stopped.

    Attributes:
        answer: Agent's final submitted answer, exactly as the model supplied it
            to the terminal tool (any JSON type: str, list, dict, number, ...),
            or None if the agent never submitted. Interpretation is left to the
            benchmark's extractor, which knows the expected answer shape.
        answer_tool: Name of the terminal tool that ended the run, or None if the
            agent never submitted. Passed to the extractor alongside answer, since
            the same value grades differently under a submit vs. an abstain tool.
        tool_calls: One entry per tool call the model emitted, in order. Executed
            calls record the parsed args; calls that never ran record the raw
            arguments and an "error" marker instead.
        tool_responses: One entry per tool call, holding the (truncated) result
            string returned to the model.
        num_tool_calls: Total number of tool calls the model emitted, including
            malformed attempts that never executed a tool. Equals len(tool_calls).
        stopped_reason: Why the agent stopped (see StoppedReason).
        final_turn_offered: Whether the run exhausted its tool-call budget and was
            given one extra turn to submit. Distinguishes a submission the nudge
            rescued from an ordinary one, and a run that was nudged and still
            refused from a record predating the nudge.
        error: Failure message, set if and only if stopped_reason is a failure
            (see StoppedReason.is_failure).
    """

    task_id: str
    answer: JsonValue
    messages: list[dict[str, Any]]
    tool_calls: list[ToolCallEntry]
    tool_responses: list[ToolResponseEntry]
    num_tool_calls: int
    execution_time_seconds: float
    stopped_reason: StoppedReason
    answer_tool: str | None = field(
        default=None, metadata=config(exclude=lambda x: x is None)
    )
    final_turn_offered: bool = False
    error: str | None = field(
        default=None, metadata=config(exclude=lambda x: x is None)
    )

    def __post_init__(self) -> None:
        # Also runs on from_dict, so a log record violating the invariant fails
        # the read rather than being graded as if it were consistent.
        if self.stopped_reason.is_failure != (self.error is not None):
            raise ValueError(
                f"error must be set if and only if stopped_reason is a failure; "
                f"got stopped_reason={self.stopped_reason}, error={self.error!r}."
            )
