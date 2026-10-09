# Copyright 2025 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for ReActLoop.

The loop drives a model against a tool registry inside an environment, so the
doubles here stand in for all three: `_FakeModelAPI` replays canned turns,
`_FakeTool` routes its arguments through the environment, and `_FakeEnvironment`
records what ran, with its failure modes as constructor parameters.
"""

import json
import time
from typing import Any, ClassVar
from unittest.mock import ANY, Mock, patch

import pytest

from faith.agentic.core import ReActLoop, Task, ToolExecutionError, ToolRegistry
from faith.agentic.core.capabilities import CommandExecution, FileRead
from faith.agentic.core.environment import Environment, EnvironmentUnavailableError
from faith.agentic.core.react_loop import (
    BUDGET_EXHAUSTED_ERROR,
    INVALID_ARGS_ERROR,
    INVALID_JSON_ERROR,
    INVALID_SUBMIT_ERROR,
    MAX_CONSECUTIVE_RECOVERIES,
    MAX_LOGGED_RESPONSE_CHARS,
    MAX_TOOL_RESULT_CHARS,
    MAX_TOTAL_RECOVERIES,
    UNKNOWN_TOOL_ERROR,
    AgentResult,
    StoppedReason,
)
from faith.agentic.core.tool import Tool

_TASK = Task(id="t", prompt="Test", ground_truth="x")


def _json_error(arguments: str) -> str:
    """The message the loop feeds back for arguments that are not JSON.

    The decoder's half is read off json itself rather than transcribed, so the
    whole message is still pinned exactly without this test owning wording that
    is CPython's to change.
    """
    try:
        json.loads(arguments)
    except json.JSONDecodeError as e:
        return f"Error: Model returned invalid JSON for tool arguments: {e}"
    raise AssertionError(f"{arguments!r} is valid JSON")


# Arguments that are not JSON at all, and the message fed back for them.
_MALFORMED_JSON = "{not valid json"
_MALFORMED_JSON_ERROR = _json_error(_MALFORMED_JSON)

_LIST_ARGUMENTS = '{"command": "ls"}'
_SUBMIT_ARGUMENTS = '{"answer": "flag{x}"}'

_TEARDOWN_ERROR = RuntimeError("cleanup boom")


class _FakeEnvironment(Environment):
    """Environment recording its lifecycle and the commands tools ran through it.

    Failure and slowness are constructor parameters rather than subclasses, so a
    test naming one needs no new class: setup_error/teardown_error raise, and the
    delays burn wall-clock time for the timeout and timing contracts.

    `command_failures` scripts a sandbox that dies: the first N commands raise
    EnvironmentUnavailableError. `rebuild_delay` and `rebuild_error` apply to every
    setup but the first, since recovery's cost and failure are what the timeout and
    rebuild contracts are about — applying them to the initial setup would end the
    run before a tool ever ran.
    """

    def __init__(
        self,
        *,
        setup_error: Exception | None = None,
        rebuild_error: Exception | None = None,
        rebuild_delay: float = 0.0,
        command_delay: float = 0.0,
        command_failures: int = 0,
        teardown_error: Exception | None = None,
        teardown_delay: float = 0.0,
    ) -> None:
        self._setup_error = setup_error
        self._rebuild_error = rebuild_error
        self._rebuild_delay = rebuild_delay
        self._command_delay = command_delay
        self._command_failures = command_failures
        self._teardown_error = teardown_error
        self._teardown_delay = teardown_delay
        self.setup_called = False
        self.teardown_called = False
        self.setup_count = 0
        self.teardown_count = 0
        self.commands_executed: list[str] = []

    def setup(self, task: Task) -> None:
        self.setup_called = True
        self.setup_count += 1
        if self.setup_count > 1:
            time.sleep(self._rebuild_delay)
            if self._rebuild_error is not None:
                raise self._rebuild_error
        if self._setup_error is not None:
            raise self._setup_error

    def run_command(self, command: str) -> tuple[int, str]:
        """Record the command and return canned output, or die if scripted to."""
        time.sleep(self._command_delay)
        if self._command_failures > 0:
            self._command_failures -= 1
            raise EnvironmentUnavailableError("sandbox is gone")
        self.commands_executed.append(command)
        return 0, "test output"

    def teardown(self) -> None:
        self.teardown_called = True
        self.teardown_count += 1
        time.sleep(self._teardown_delay)
        if self._teardown_error is not None:
            raise self._teardown_error


class _LifecycleOnlyEnvironment(Environment):
    """The minimal legal environment: lifecycle only, so no capability at all."""

    def setup(self, task: Task) -> None:
        pass

    def teardown(self) -> None:
        pass


class _FakeTool(Tool):
    """Tool echoing its arguments through the environment's shell.

    is_terminal / answer_field default to a non-terminal tool; terminal tools
    (submit, abstain, ...) opt in per-instance so the loop discovers them via the
    registry rather than by name. `result`, when given, is returned instead of
    the environment's output, so a tool can break the `-> str` contract.
    """

    # Annotated as widely as Tool declares it, so a subclass below can name a
    # different capability set.
    required_capabilities: ClassVar[tuple[type, ...]] = (CommandExecution,)

    def __init__(
        self,
        name: str,
        is_terminal: bool = False,
        answer_field: str | None = "answer",
        result: Any = None,
    ):
        self._name = name
        self._is_terminal = is_terminal
        self._answer_field = answer_field
        self._result = result

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return f"Fake {self._name} tool"

    @property
    def parameters(self) -> dict:
        # No required args, so one fake stands in for tools whose arg shapes differ.
        return {
            "type": "object",
            "properties": {
                "command": {"type": "string", "description": "Command to run"}
            },
        }

    @property
    def is_terminal(self) -> bool:
        return self._is_terminal

    @property
    def answer_field(self) -> str | None:
        return self._answer_field

    def _invoke(self, environment: CommandExecution, args: dict) -> str:
        if self._result is not None:
            return self._result
        _, output = environment.run_command(f"{self._name} {json.dumps(args)}")
        return output


class _FileReadTool(_FakeTool):
    """Tool requiring FileRead, which no environment here satisfies."""

    required_capabilities = (FileRead,)


class _NoCapabilityTool(_FakeTool):
    """Tool requiring nothing of its environment."""

    required_capabilities = ()


class _FakeModelAPI:
    """Fake OpenAI-compatible model API replaying canned responses in order.

    Empty responses once the canned ones are exhausted. Uses Mock to fabricate
    the duck-typed objects the loop reads (response.choices[0].message.tool_calls
    ...), since the loop targets a structural OpenAI-compatible surface with no
    concrete type to construct.
    """

    def __init__(self, responses: list[dict]):
        self.responses = responses
        self.call_count = 0
        self.chat = Mock()
        self.chat.completions = Mock()
        self.chat.completions.create = Mock(side_effect=self._create_response)

    def _create_response(self, **_kwargs: Any) -> Mock:
        """Build the next canned response as a Mock with the API's attributes."""
        if self.call_count >= len(self.responses):
            response_data: dict[str, Any] = {"content": "", "tool_calls": None}
        else:
            response_data = self.responses[self.call_count]
        self.call_count += 1

        message = Mock()
        message.content = response_data.get("content") or ""
        message.tool_calls = [
            _mock_tool_call(call) for call in response_data.get("tool_calls") or []
        ] or None

        choice = Mock()
        choice.message = message
        response = Mock()
        response.choices = [choice]
        return response


def _mock_tool_call(call: dict) -> Mock:
    """One tool call from a canned turn, as the duck-typed object the loop reads."""
    tool_call = Mock()
    tool_call.id = call["id"]
    tool_call.type = call["type"]
    tool_call.function = Mock()
    tool_call.function.name = call["function"]["name"]
    tool_call.function.arguments = call["function"]["arguments"]
    return tool_call


def _turn(content: str, *calls: tuple[str, str, str | None]) -> dict[str, Any]:
    """A canned model turn: prose plus zero or more (id, tool name, arguments)."""
    return {
        "content": content,
        "tool_calls": [
            {
                "id": call_id,
                "type": "function",
                "function": {"name": name, "arguments": arguments},
            }
            for call_id, name, arguments in calls
        ]
        or None,
    }


# A well-formed submission, used to end runs whose subject is an earlier turn.
_SUBMIT_TURN = _turn("Submitting", ("call_submit", "submit", _SUBMIT_ARGUMENTS))


def _loop(
    model: _FakeModelAPI,
    registry: ToolRegistry,
    environment: Environment,
    **kwargs: Any,
) -> ReActLoop:
    """A loop over the given doubles; kwargs are the limits under test."""
    return ReActLoop(
        model_api=model,
        model="test-model",
        environment=environment,
        tool_registry=registry,
        **kwargs,
    )


def test_a_submitting_run_reports_the_whole_trajectory() -> None:
    """One tool call then a submission, recorded end to end.

    Compared as a whole so a drifted or extra field in any log entry fails here
    rather than in whichever downstream consumer reads it.
    """
    env = _FakeEnvironment()
    registry = ToolRegistry(
        [_FakeTool("bash_exec"), _FakeTool("submit", is_terminal=True)]
    )
    model = _FakeModelAPI(
        [
            _turn("Let me check", ("call_1", "bash_exec", _LIST_ARGUMENTS)),
            _turn("Found the answer", ("call_2", "submit", _SUBMIT_ARGUMENTS)),
        ]
    )

    result = _loop(model, registry, env).run(
        Task(id="test_task", prompt="Find the flag", ground_truth="flag{x}")
    )

    assert result == AgentResult(
        task_id="test_task",
        answer="flag{x}",
        answer_tool="submit",
        messages=[
            {"role": "user", "content": "Find the flag"},
            {
                "role": "assistant",
                "content": "Let me check",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "bash_exec",
                            "arguments": _LIST_ARGUMENTS,
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call_1",
                "content": "test output\n[49 tool-calls remaining]",
            },
            {
                "role": "assistant",
                "content": "Found the answer",
                "tool_calls": [
                    {
                        "id": "call_2",
                        "type": "function",
                        "function": {
                            "name": "submit",
                            "arguments": _SUBMIT_ARGUMENTS,
                        },
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call_2", "content": "test output"},
        ],
        tool_calls=[
            {"tool_name": "bash_exec", "args": {"command": "ls"}, "timestamp": ANY},
            {"tool_name": "submit", "args": {"answer": "flag{x}"}, "timestamp": ANY},
        ],
        tool_responses=[
            {
                "tool_name": "bash_exec",
                "response_truncated": "test output\n[49 tool-calls remaining]",
                "response_length": len("test output\n[49 tool-calls remaining]"),
                "timestamp": ANY,
            },
            {
                "tool_name": "submit",
                "response_truncated": "test output",
                "response_length": len("test output"),
                "timestamp": ANY,
            },
        ],
        num_tool_calls=2,
        execution_time_seconds=ANY,
        stopped_reason=StoppedReason.SUBMITTED,
        error=None,
    )
    assert (env.setup_called, env.teardown_called) == (True, True)


@pytest.mark.parametrize(
    "tool,arguments,answer",
    [
        # An explicit null answer is a real submission, not a missing key.
        (_FakeTool("submit", is_terminal=True), '{"answer": null}', None),
        # The submit schema is benchmark-defined, so the answer may be a
        # list/dict and must reach the extractor unchanged.
        (
            _FakeTool("submit_files", is_terminal=True, answer_field="ranked_files"),
            '{"ranked_files": ["a.c", "b.c"]}',
            ["a.c", "b.c"],
        ),
        # answer_field=None: the call itself is the whole signal (e.g. abstain),
        # so no key is read and zero arguments are not a botch. An
        # OpenAI-compatible server may encode "no arguments" as null, "" or {}.
        (_FakeTool("abstain", is_terminal=True, answer_field=None), None, None),
        (_FakeTool("abstain", is_terminal=True, answer_field=None), "", None),
        (_FakeTool("abstain", is_terminal=True, answer_field=None), "{}", None),
    ],
)
def test_a_terminal_call_ends_the_run_with_its_answer(
    tool: _FakeTool, arguments: str | None, answer: Any
) -> None:
    """Any tool declaring is_terminal ends the run, whatever its name.

    Termination is discovered through the registry rather than a hardcoded
    "submit", and the answer is read from the tool's own answer_field.
    """
    model = _FakeModelAPI([_turn("Ending", ("call_1", tool.name, arguments))])

    result = _loop(model, ToolRegistry([tool]), _FakeEnvironment()).run(_TASK)

    assert (
        result.stopped_reason,
        result.answer,
        result.answer_tool,
        result.error,
    ) == (StoppedReason.SUBMITTED, answer, tool.name, None)


def test_tools_after_submit_in_same_turn_do_not_execute() -> None:
    """Submit is terminal: tool calls after it in the same turn don't run."""
    env = _FakeEnvironment()
    registry = ToolRegistry(
        [_FakeTool("submit", is_terminal=True), _FakeTool("bash_exec")]
    )
    model = _FakeModelAPI(
        [
            _turn(
                "Submit and also run a command",
                ("call_submit", "submit", _SUBMIT_ARGUMENTS),
                ("call_bash", "bash_exec", '{"command": "rm -rf x"}'),
            )
        ]
    )

    result = _loop(model, registry, env).run(_TASK)

    assert result.stopped_reason == StoppedReason.SUBMITTED
    # Only submit executed; the post-submit bash_exec never ran.
    assert env.commands_executed == ['submit {"answer": "flag{x}"}']


@pytest.mark.parametrize("calls_per_turn", [1, 5])
def test_the_tool_budget_is_enforced_per_call(calls_per_turn: int) -> None:
    """Exactly max_tool_calls commands reach the environment, whatever the batch.

    The check is per call rather than between turns, so a turn emitting more
    calls than the budget can pay for cannot overshoot it: the surplus calls are
    refused. Every call still gets a result message either way, so no tool_call
    is left unanswered.
    """
    env = _FakeEnvironment()
    registry = ToolRegistry([_FakeTool("bash_exec")])
    model = _FakeModelAPI(
        [
            _turn(
                f"Turn {turn}",
                *[
                    (f"call_{turn}_{i}", "bash_exec", _LIST_ARGUMENTS)
                    for i in range(calls_per_turn)
                ],
            )
            for turn in range(10)
        ]
    )

    result = _loop(model, registry, env, max_tool_calls=3).run(_TASK)

    assert (result.stopped_reason, result.answer) == (StoppedReason.MAX_CALLS, None)
    assert env.commands_executed == ['bash_exec {"command": "ls"}'] * 3
    # Every emitted call is logged, executed or refused, and each has a result.
    assert len(result.tool_responses) == result.num_tool_calls


def test_an_over_budget_call_is_refused_rather_than_executed() -> None:
    """A batch larger than the remaining budget spends it and refuses the rest.

    Three calls against a budget of two: the first two execute and are charged,
    the third is refused with the budget message. The refusal is a logged call
    with a result like any other, but nothing ran for it, so it carries the raw
    arguments and an error marker rather than parsed args, and no remaining-count
    suffix - there is nothing remaining.
    """
    env = _FakeEnvironment()
    registry = ToolRegistry([_FakeTool("bash_exec")])
    model = _FakeModelAPI(
        [
            _turn(
                "All at once",
                *[(f"call_{i}", "bash_exec", _LIST_ARGUMENTS) for i in range(3)],
            )
        ]
    )

    result = _loop(model, registry, env, max_tool_calls=2, max_turns=2).run(_TASK)

    assert env.commands_executed == ['bash_exec {"command": "ls"}'] * 2
    assert result.tool_calls[2] == {
        "tool_name": "bash_exec",
        "raw_arguments": _LIST_ARGUMENTS,
        "error": BUDGET_EXHAUSTED_ERROR,
        "timestamp": ANY,
    }
    assert result.tool_responses[2] == {
        "tool_name": "bash_exec",
        "response_truncated": (
            "ERROR: Tool call budget exhausted (2/2). Please submit your answer now."
        ),
        "response_length": len(
            "ERROR: Tool call budget exhausted (2/2). Please submit your answer now."
        ),
        "timestamp": ANY,
    }
    # The refusal changes what an over-budget call does, not when the run ends:
    # the pre-turn check still reports MAX_CALLS on the next iteration.
    assert (result.stopped_reason, result.num_tool_calls) == (
        StoppedReason.MAX_CALLS,
        3,
    )


def test_a_terminal_call_in_an_over_budget_batch_still_submits() -> None:
    """Terminal tools are exempt from the refusal, so an answer is never discarded.

    The submission arrives in the same batch that exhausted the budget. Refusing
    it would throw away the answer the run exists to produce.
    """
    env = _FakeEnvironment()
    registry = ToolRegistry(
        [_FakeTool("bash_exec"), _FakeTool("submit", is_terminal=True)]
    )
    model = _FakeModelAPI(
        [
            _turn(
                "Last look, then done",
                ("call_1", "bash_exec", _LIST_ARGUMENTS),
                ("call_2", "submit", _SUBMIT_ARGUMENTS),
            )
        ]
    )

    result = _loop(model, registry, env, max_tool_calls=1).run(_TASK)

    assert (result.stopped_reason, result.answer, result.answer_tool) == (
        StoppedReason.SUBMITTED,
        "flag{x}",
        "submit",
    )
    assert env.commands_executed == [
        'bash_exec {"command": "ls"}',
        'submit {"answer": "flag{x}"}',
    ]


def test_a_spent_budget_buys_a_final_turn_that_can_submit() -> None:
    """An agent that explores to the last call is asked for its answer, not cut off.

    The whole point of the nudge: without it this run scores zero despite having
    the answer. final_turn_offered is what tells a rescued submission apart from
    an ordinary one.
    """
    env = _FakeEnvironment()
    registry = ToolRegistry(
        [_FakeTool("bash_exec"), _FakeTool("submit", is_terminal=True)]
    )
    model = _FakeModelAPI(
        [
            _turn("Looking", ("call_1", "bash_exec", _LIST_ARGUMENTS)),
            _turn("Looking again", ("call_2", "bash_exec", _LIST_ARGUMENTS)),
            _SUBMIT_TURN,
        ]
    )

    result = _loop(model, registry, env, max_tool_calls=2).run(_TASK)

    assert (
        result.stopped_reason,
        result.answer,
        result.final_turn_offered,
    ) == (StoppedReason.SUBMITTED, "flag{x}", True)
    # The nudge names the budget and the terminal tools, and precedes the turn
    # that submitted.
    assert result.messages[-3] == {
        "role": "user",
        "content": (
            "You have used all 2 of your tool calls. "
            "Submit your answer now using: submit."
        ),
    }


def test_the_final_turn_is_offered_exactly_once() -> None:
    """A model that will not submit is nudged once, then the run ends as before.

    The flag bounds the extra turn by construction, so a refusal cannot buy a
    second one however many turns the cap would allow.
    """
    env = _FakeEnvironment()
    registry = ToolRegistry(
        [_FakeTool("bash_exec"), _FakeTool("submit", is_terminal=True)]
    )
    model = _FakeModelAPI(
        [
            _turn(f"Looking {turn}", (f"call_{turn}", "bash_exec", _LIST_ARGUMENTS))
            for turn in range(10)
        ]
    )

    result = _loop(model, registry, env, max_tool_calls=2, max_turns=10).run(_TASK)

    assert (
        result.stopped_reason,
        result.final_turn_offered,
        result.num_tool_calls,
    ) == (StoppedReason.MAX_CALLS, True, 3)
    # Two executed plus the final turn's refused call: the extra turn cannot
    # spend budget it does not have.
    assert env.commands_executed == ['bash_exec {"command": "ls"}'] * 2


def test_the_final_turn_offers_only_the_terminal_tools() -> None:
    """On the nudged turn the model sees the tools it is being asked to use.

    Every earlier turn offers the whole registry; the final one offers the
    terminal subset, so the model cannot spend the turn on a call that would only
    be refused.
    """
    env = _FakeEnvironment()
    registry = ToolRegistry(
        [_FakeTool("bash_exec"), _FakeTool("submit", is_terminal=True)]
    )
    model = _FakeModelAPI(
        [
            _turn("Looking", ("call_1", "bash_exec", _LIST_ARGUMENTS)),
            _SUBMIT_TURN,
        ]
    )

    _loop(model, registry, env, max_tool_calls=1).run(_TASK)

    offered = [
        [schema["function"]["name"] for schema in call.kwargs["tools"]]
        for call in model.chat.completions.create.call_args_list
    ]
    assert offered == [["bash_exec", "submit"], ["submit"]]


def test_each_result_tells_the_model_what_budget_is_left() -> None:
    """The remaining count is appended to executed results, and counts down.

    An agent that cannot see its budget cannot pace itself, which is what the
    exhausted runs did. The count is what remains *after* the call it is attached
    to, so the first of three reads 2, not 3.
    """
    registry = ToolRegistry(
        [_FakeTool("bash_exec"), _FakeTool("submit", is_terminal=True)]
    )
    model = _FakeModelAPI(
        [
            _turn(
                "Both at once",
                ("call_1", "bash_exec", _LIST_ARGUMENTS),
                ("call_2", "bash_exec", _LIST_ARGUMENTS),
            ),
            _SUBMIT_TURN,
        ]
    )

    result = _loop(model, registry, _FakeEnvironment(), max_tool_calls=3).run(_TASK)

    contents = [
        message["content"] for message in result.messages if message["role"] == "tool"
    ]
    # The submission is not charged, so it carries no suffix at all.
    assert contents == [
        "test output\n[2 tool-calls remaining]",
        "test output\n[1 tool-calls remaining]",
        "test output",
    ]
    assert result.stopped_reason == StoppedReason.SUBMITTED


def test_a_tool_that_does_not_cap_itself_is_truncated_by_the_harness() -> None:
    """The harness backstop bounds a result no well-behaved tool would return.

    Tools cap their own output; this is the ceiling that keeps one that does not
    from blowing the model API's request limit and failing the task outright. The
    marker carries the true size so the agent knows what it is missing, and the
    trajectory records the untruncated length.
    """
    oversized = "x" * (MAX_TOOL_RESULT_CHARS + 5000)
    registry = ToolRegistry([_FakeTool("bash_exec", result=oversized)])
    model = _FakeModelAPI([_turn("Dumping", ("call_1", "bash_exec", _LIST_ARGUMENTS))])

    result = _loop(
        model, registry, _FakeEnvironment(), max_tool_calls=1, max_turns=1
    ).run(_TASK)

    assert result.messages[2] == {
        "role": "tool",
        "tool_call_id": "call_1",
        "content": (
            "x" * MAX_TOOL_RESULT_CHARS
            + f"\n\n[TRUNCATED BY HARNESS — tool returned {len(oversized)} chars, "
            f"limit {MAX_TOOL_RESULT_CHARS}]"
            "\n[0 tool-calls remaining]"
        ),
    }


@pytest.mark.parametrize("length", [MAX_LOGGED_RESPONSE_CHARS + 100, 12])
def test_a_logged_response_records_its_untruncated_length(length: int) -> None:
    """The excerpt is capped for the log; the length recorded is the real one.

    So a payload-size question is answerable from the trajectory alone, without
    reading back the full string from messages. The short case is the boundary:
    a 12-char response is excerpted to 12 chars, not padded, and its length is
    its own.
    """
    response = "y" * length
    registry = ToolRegistry([_FakeTool("bash_exec", result=response)])
    model = _FakeModelAPI([_turn("Reading", ("call_1", "bash_exec", _LIST_ARGUMENTS))])

    result = _loop(
        model, registry, _FakeEnvironment(), max_tool_calls=1, max_turns=1
    ).run(_TASK)

    # The suffix is part of what the model saw, so it is part of what is logged.
    logged = f"{response}\n[0 tool-calls remaining]"
    assert result.tool_responses[0] == {
        "tool_name": "bash_exec",
        "response_truncated": logged[:MAX_LOGGED_RESPONSE_CHARS],
        "response_length": len(logged),
        "timestamp": ANY,
    }


@pytest.mark.parametrize(
    "system_prompt,expected",
    [
        ("You are a security analyst.", "You are a security analyst."),
        # No prompt and an empty one both mean "send no system message", rather
        # than an empty one the model has to interpret.
        (None, None),
        ("", None),
    ],
)
def test_a_system_prompt_is_seeded_ahead_of_the_task(
    system_prompt: str | None, expected: str | None
) -> None:
    """The benchmark's standing instructions open the conversation, if it has any.

    The task prompt is always the first user message, so a benchmark that carries
    no system prompt is unaffected.
    """
    registry = ToolRegistry([_FakeTool("submit", is_terminal=True)])
    model = _FakeModelAPI([_SUBMIT_TURN])

    result = _loop(
        model, registry, _FakeEnvironment(), system_prompt=system_prompt
    ).run(_TASK)

    opening = [{"role": "system", "content": expected}] if expected else []
    assert result.messages[: len(opening) + 1] == opening + [
        {"role": "user", "content": "Test"}
    ]


def test_multi_call_turn_returns_one_result_per_call() -> None:
    """Every tool_call in a multi-call turn gets exactly one tool message.

    The OpenAI-compatible API requires a tool-result message for each
    tool_call_id in an assistant turn.
    """
    registry = ToolRegistry([_FakeTool("bash_exec")])
    call_ids = ["call_a", "call_b", "call_c"]
    model = _FakeModelAPI(
        [
            _turn(
                "Run several at once",
                *[(call_id, "bash_exec", _LIST_ARGUMENTS) for call_id in call_ids],
            ),
            _turn("Done"),
        ]
    )

    result = _loop(model, registry, _FakeEnvironment(), max_tool_calls=10).run(_TASK)

    tool_messages = [m for m in result.messages if m.get("role") == "tool"]
    assert [m["tool_call_id"] for m in tool_messages] == call_ids


def test_a_run_of_prose_turns_ends_the_run() -> None:
    """A model that will not call a tool terminates without an answer."""
    model = _FakeModelAPI([_turn("I'm done")])

    result = _loop(
        model,
        ToolRegistry(),
        _FakeEnvironment(),
        max_consecutive_no_tool_calls=1,
    ).run(_TASK)

    assert (
        result.stopped_reason,
        result.num_tool_calls,
        result.answer_tool,
        result.error,
    ) == (StoppedReason.NO_TOOL_CALLS, 0, None, None)


def test_a_prose_turn_is_nudged_rather_than_ending_the_run() -> None:
    """One chatty turn must not discard a task the model then completes.

    The nudge names the callable tools from the registry, since which tools exist
    is the benchmark's choice.
    """
    registry = ToolRegistry([_FakeTool("submit", is_terminal=True)])
    model = _FakeModelAPI([_turn("Let me think about this."), _SUBMIT_TURN])

    result = _loop(model, registry, _FakeEnvironment()).run(_TASK)

    assert result.stopped_reason == StoppedReason.SUBMITTED
    assert [m for m in result.messages if m["role"] == "user"] == [
        {"role": "user", "content": "Test"},
        {
            "role": "user",
            "content": "Please respond with a tool call. Available tools: submit.",
        },
    ]


def test_the_prose_turn_counter_resets_on_a_tool_call() -> None:
    """Prose turns must be consecutive to end the run, not merely frequent."""
    registry = ToolRegistry([_FakeTool("bash_exec")])
    model = _FakeModelAPI(
        [
            turn
            for i in range(3)
            for turn in (
                _turn("Thinking"),
                _turn("Running", (f"call_{i}", "bash_exec", _LIST_ARGUMENTS)),
            )
        ]
    )

    result = _loop(
        model, registry, _FakeEnvironment(), max_turns=6, max_tool_calls=3
    ).run(_TASK)

    # Six alternating turns: three prose turns, never two in a row, so the tool
    # budget is what stops the run.
    assert (result.stopped_reason, result.num_tool_calls) == (
        StoppedReason.MAX_CALLS,
        3,
    )


def test_generation_params_spread_into_model_calls() -> None:
    """generation_params reach chat.completions.create on every model turn.

    Without this the agent would silently use the API's defaults (e.g.
    temperature 1.0), defeating deterministic eval.
    """
    model = _FakeModelAPI([_turn("done")])

    _loop(
        model,
        ToolRegistry(),
        _FakeEnvironment(),
        generation_params={"temperature": 0.3, "max_completion_tokens": 256},
    ).run(_TASK)

    # The whole call, since a param that collided with model/messages/tools
    # would otherwise go unnoticed.
    assert model.chat.completions.create.call_args.kwargs == {
        "model": "test-model",
        "messages": ANY,
        "tools": None,
        "tool_choice": None,
        "temperature": 0.3,
        "max_completion_tokens": 256,
    }


@pytest.mark.parametrize(
    ("setup_error", "expected_reason", "expected_error"),
    [
        (
            RuntimeError("Setup failed"),
            StoppedReason.ERROR,
            "Environment setup failed: RuntimeError: Setup failed",
        ),
        (
            EnvironmentUnavailableError("Setup failed"),
            StoppedReason.ENVIRONMENT_ERROR,
            # No type prefix: the caught type is named by the reason already.
            "Environment setup failed: Setup failed",
        ),
    ],
    ids=["run_error", "environment_error"],
)
def test_environment_setup_failure_is_an_error_that_still_tears_down(
    setup_error: Exception, expected_reason: StoppedReason, expected_error: str
) -> None:
    """A setup failure yields an empty trajectory, and cleanup still runs.

    Setup that fails partway (e.g. a container started before a later step
    raised) would otherwise leak the resource. The reason separates a cluster
    that could not serve a sandbox from a sandbox FAITH could not prepare, which
    is what tells a reader which one to go look at.
    """
    env = _FakeEnvironment(setup_error=setup_error)

    result = _loop(_FakeModelAPI([]), ToolRegistry(), env).run(_TASK)

    assert result == AgentResult(
        task_id="t",
        answer=None,
        answer_tool=None,
        messages=[],
        tool_calls=[],
        tool_responses=[],
        num_tool_calls=0,
        execution_time_seconds=ANY,
        stopped_reason=expected_reason,
        error=expected_error,
    )
    assert env.teardown_called


def test_a_tool_execution_error_is_fed_back_and_counted_as_executed() -> None:
    """A ToolExecutionError reaches the model instead of ending the run.

    The loop catches it separately from ToolNotFoundError: the tool DID run, so
    the call is charged against the budget and carries no error marker.
    """
    registry = ToolRegistry(
        [_FakeTool("bash_exec"), _FakeTool("submit", is_terminal=True)]
    )
    model = _FakeModelAPI(
        [_turn("Try this", ("call_1", "bash_exec", _LIST_ARGUMENTS)), _SUBMIT_TURN]
    )
    loop = _loop(model, registry, _FakeEnvironment())

    with patch.object(
        ToolRegistry,
        "execute_tool",
        side_effect=[ToolExecutionError("registry says no"), "submitted"],
    ):
        result = loop.run(_TASK)

    assert (result.stopped_reason, result.answer) == (
        StoppedReason.SUBMITTED,
        "flag{x}",
    )
    # Charged like any executed call, so the response carries the budget suffix.
    assert result.tool_responses[0] == {
        "tool_name": "bash_exec",
        "response_truncated": (
            "Error executing tool: registry says no\n[49 tool-calls remaining]"
        ),
        "response_length": len(
            "Error executing tool: registry says no\n[49 tool-calls remaining]"
        ),
        "timestamp": ANY,
    }
    # Both calls executed, so neither carries an "error" marker.
    assert result.tool_calls == [
        {"tool_name": "bash_exec", "args": {"command": "ls"}, "timestamp": ANY},
        {"tool_name": "submit", "args": {"answer": "flag{x}"}, "timestamp": ANY},
    ]


def test_unknown_tool_name_is_fed_back_not_fatal() -> None:
    """An unregistered tool name is fed back and never charged to the budget.

    Caught narrowly as ToolNotFoundError: a ValueError raised by a tool that DID
    run must not be reported as a bad tool name.
    """
    env = _FakeEnvironment()
    registry = ToolRegistry([_FakeTool("submit", is_terminal=True)])
    model = _FakeModelAPI(
        [_turn("Try this", ("call_1", "does_not_exist", "{}")), _SUBMIT_TURN]
    )

    result = _loop(model, registry, env).run(_TASK)

    assert (result.stopped_reason, result.answer) == (
        StoppedReason.SUBMITTED,
        "flag{x}",
    )
    assert result.tool_calls[0] == {
        "tool_name": "does_not_exist",
        "raw_arguments": "{}",
        "error": UNKNOWN_TOOL_ERROR,
        "timestamp": ANY,
    }
    unknown_tool_response = (
        "Unknown tool: Tool 'does_not_exist' not found. Available tools: submit"
    )
    assert result.tool_responses[0] == {
        "tool_name": "does_not_exist",
        "response_truncated": unknown_tool_response,
        "response_length": len(unknown_tool_response),
        "timestamp": ANY,
    }


def test_an_expired_timeout_outranks_a_spent_tool_budget() -> None:
    """A run over the clock reports timeout, whichever budget also ran out.

    Both limits trip on the same turn here: one tool call is allowed and one is
    made, and the command outlasts the timeout. Timeout is the more informative
    of the two, and the one an operator acts on. It is a limit, not a failure, so
    error stays None.
    """
    env = _FakeEnvironment(command_delay=0.05)
    registry = ToolRegistry([_FakeTool("bash_exec")])
    model = _FakeModelAPI([_turn("Checking", ("call_1", "bash_exec", _LIST_ARGUMENTS))])

    result = _loop(model, registry, env, max_tool_calls=1, timeout_seconds=0.01).run(
        _TASK
    )

    assert (result.stopped_reason, result.error) == (StoppedReason.TIMEOUT, None)


@pytest.mark.parametrize(
    "tool_name,arguments,error,message",
    [
        ("bash_exec", _MALFORMED_JSON, INVALID_JSON_ERROR, _MALFORMED_JSON_ERROR),
        ("submit", _MALFORMED_JSON, INVALID_JSON_ERROR, _MALFORMED_JSON_ERROR),
        # Valid JSON that is not an object: models emit bare scalars.
        (
            "bash_exec",
            '"ls"',
            INVALID_ARGS_ERROR,
            'Error: Tool arguments must be a JSON object; got: "ls"',
        ),
        (
            "submit",
            '"flag{x}"',
            INVALID_ARGS_ERROR,
            'Error: Tool arguments must be a JSON object; got: "flag{x}"',
        ),
        # An object missing the terminal tool's declared answer_field: a botched
        # submission, not a submission of None.
        (
            "submit",
            '{"result": "flag{x}"}',
            INVALID_SUBMIT_ERROR,
            "Error: submit requires a JSON object with a 'answer' field, e.g. "
            '{"answer": ...}; got: {"result": "flag{x}"}',
        ),
    ],
)
def test_a_malformed_call_is_fed_back_and_retriable(
    tool_name: str, arguments: str, error: str, message: str
) -> None:
    """A malformed call is logged with its raw arguments, runs nothing, and retries.

    The attempt stays visible for analytics, the model receives the error and can
    fix it, and because no tool ran it is not charged against the tool budget.
    """
    env = _FakeEnvironment()
    registry = ToolRegistry(
        [_FakeTool("bash_exec"), _FakeTool("submit", is_terminal=True)]
    )
    model = _FakeModelAPI(
        [_turn("Bad call", ("call_bad", tool_name, arguments)), _SUBMIT_TURN]
    )

    result = _loop(model, registry, env, max_tool_calls=5).run(_TASK)

    assert result.tool_calls[0] == {
        "tool_name": tool_name,
        "raw_arguments": arguments,
        "error": error,
        "timestamp": ANY,
    }
    assert result.tool_responses[0] == {
        "tool_name": tool_name,
        "response_truncated": message,
        "response_length": len(message),
        "timestamp": ANY,
    }
    # Nothing ran for the bad call: only the recovering submission reached the
    # environment, and the run recovered to a clean submission.
    assert env.commands_executed == ['submit {"answer": "flag{x}"}']
    assert (result.stopped_reason, result.answer) == (
        StoppedReason.SUBMITTED,
        "flag{x}",
    )


def test_a_long_error_response_records_its_untruncated_length() -> None:
    """The error path excerpts its message but records the real length.

    A separate construction site from an executed call's, and the one whose
    message can grow without a tool involved: the feedback quotes the model's raw
    arguments back, so a large invalid blob produces a large error. Every other
    case here is short enough that the truncated and untruncated lengths agree,
    which is where recording the wrong one would go unnoticed.
    """
    arguments = '"' + "y" * (MAX_LOGGED_RESPONSE_CHARS * 2) + '"'
    registry = ToolRegistry(
        [_FakeTool("bash_exec"), _FakeTool("submit", is_terminal=True)]
    )
    model = _FakeModelAPI(
        [_turn("Bad call", ("call_bad", "bash_exec", arguments)), _SUBMIT_TURN]
    )

    result = _loop(model, registry, _FakeEnvironment(), max_tool_calls=5).run(_TASK)

    message = f"Error: Tool arguments must be a JSON object; got: {arguments}"
    assert result.tool_responses[0] == {
        "tool_name": "bash_exec",
        "response_truncated": message[:MAX_LOGGED_RESPONSE_CHARS],
        "response_length": len(message),
        "timestamp": ANY,
    }


def test_all_malformed_calls_stop_at_max_turns_not_max_calls() -> None:
    """A model emitting only malformed JSON is bounded by max_turns.

    Malformed calls never execute a tool, so they never consume the tool-call
    budget. The turn cap is the independent safety net that still guarantees
    termination, and the stop reason distinguishes this unproductive exit from
    exhausting real tool work. max_turns == max_tool_calls is the tightest valid
    config, so reporting max_turns is only possible if nothing was charged.
    """
    env = _FakeEnvironment()
    registry = ToolRegistry([_FakeTool("bash_exec")])
    model = _FakeModelAPI(
        [
            _turn("Bad call", (f"call_{i}", "bash_exec", _MALFORMED_JSON))
            for i in range(20)
        ]
    )

    result = _loop(model, registry, env, max_tool_calls=3, max_turns=3).run(_TASK)

    assert (result.stopped_reason, result.num_tool_calls) == (
        StoppedReason.MAX_TURNS,
        3,
    )
    assert not env.commands_executed


def test_teardown_failure_is_logged_and_leaves_a_clean_run_alone() -> None:
    """A teardown failure is best-effort cleanup: logged, but not the outcome."""
    env = _FakeEnvironment(teardown_error=_TEARDOWN_ERROR, teardown_delay=0.5)
    registry = ToolRegistry([_FakeTool("submit", is_terminal=True)])
    model = _FakeModelAPI([_SUBMIT_TURN])
    loop = _loop(model, registry, env, max_tool_calls=5)

    with patch("faith.agentic.core.react_loop.logger") as mock_logger:
        result = loop.run(_TASK)

    # A swallowed failure must still be visible to whoever debugs a leaked
    # container.
    assert mock_logger.warning.call_args == (
        ("Environment teardown failed: %s", _TEARDOWN_ERROR),
    )
    # The clean submission is preserved, and cleanup time is not the agent's
    # time, so the slow teardown is excluded from it.
    assert (result.stopped_reason, result.answer, result.error) == (
        StoppedReason.SUBMITTED,
        "flag{x}",
        None,
    )
    # Well under the teardown delay, so this fails if the clock is ever stopped
    # after teardown rather than before it. The margin is wide enough that a
    # loaded machine cannot turn a correct implementation red.
    assert result.execution_time_seconds < 0.25


def test_non_string_tool_result_fails_the_run() -> None:
    """A tool returning a non-string ends the run as ERROR.

    Tool.execute is typed -> str, so a non-string result is a tool bug. The loop
    fails loud, naming the tool and its actual type, rather than feed the model a
    corrupted trajectory.
    """
    registry = ToolRegistry(
        [
            _FakeTool("bash_exec", result={"stdout": "not a string"}),
            _FakeTool("submit", is_terminal=True),
        ]
    )
    model = _FakeModelAPI(
        [_turn("Running", ("call_1", "bash_exec", _LIST_ARGUMENTS)), _SUBMIT_TURN]
    )

    result = _loop(model, registry, _FakeEnvironment(), max_tool_calls=5).run(_TASK)

    # The contract violation aborts the run before it can submit.
    assert (result.stopped_reason, result.answer, result.error) == (
        StoppedReason.ERROR,
        None,
        "Agent loop error: TypeError: Tool bash_exec returned dict, expected str",
    )


@pytest.mark.parametrize(
    "config,message",
    [
        (
            {"max_tool_calls": 0},
            "max_tool_calls (0) and max_turns (100) must both be at least 1.",
        ),
        (
            {"max_tool_calls": 0, "max_turns": 0},
            "max_tool_calls (0) and max_turns (0) must both be at least 1.",
        ),
        # A turn cap below the tool budget would cut a well-behaved model off
        # before it could spend that budget.
        (
            {"max_tool_calls": 50, "max_turns": 10},
            "max_turns (10) must be >= max_tool_calls (50).",
        ),
        (
            {"max_consecutive_no_tool_calls": 0},
            "max_consecutive_no_tool_calls (0) must be at least 1.",
        ),
        # timeout_seconds=0 must not silently mean "no timeout"; None does.
        (
            {"timeout_seconds": 0},
            "timeout_seconds (0) must be positive; omit it for no timeout.",
        ),
        (
            {"timeout_seconds": -1.0},
            "timeout_seconds (-1.0) must be positive; omit it for no timeout.",
        ),
        # max_attempts < 1 makes RetryFunctionWrapper never call, or (negative)
        # retry forever.
        ({"api_max_attempts": 0}, "api_max_attempts must be an integer of at least 1."),
        (
            {"api_max_attempts": -1},
            "api_max_attempts must be an integer of at least 1.",
        ),
        (
            {"api_retry_sleep_secs": -1.0},
            "api_retry_sleep_secs must be finite and non-negative.",
        ),
    ],
)
def test_an_invalid_configuration_is_rejected_at_construction(
    config: dict, message: str
) -> None:
    """A limit that would make every task unrunnable fails before any task runs."""
    with pytest.raises(ValueError) as excinfo:
        _loop(_FakeModelAPI([]), ToolRegistry(), _FakeEnvironment(), **config)
    assert str(excinfo.value) == message


@pytest.mark.parametrize(
    "environment,tools,message",
    [
        (
            _FakeEnvironment(),
            [_FakeTool("bash_exec"), _FileReadTool("read")],
            "Environment '_FakeEnvironment' does not support tool 'read' "
            "(requires FileRead). Registered tools: bash_exec, read.",
        ),
        (
            _LifecycleOnlyEnvironment(),
            [_FakeTool("bash_exec")],
            "Environment '_LifecycleOnlyEnvironment' does not support tool "
            "'bash_exec' (requires CommandExecution). Registered tools: bash_exec.",
        ),
    ],
)
def test_loop_construction_rejects_a_tool_the_environment_cannot_serve(
    environment: Environment, tools: list[Tool], message: str
) -> None:
    """A misconfigured benchmark fails at construction, before any task runs."""
    with pytest.raises(ValueError) as excinfo:
        _loop(_FakeModelAPI([]), ToolRegistry(tools), environment)
    assert str(excinfo.value) == message


@pytest.mark.parametrize(
    "environment,tools",
    [
        # A partial environment is fine as long as the registry stays within it.
        (_FakeEnvironment(), [_FakeTool("bash_exec")]),
        # A tool requiring nothing is served by the barest environment.
        (_LifecycleOnlyEnvironment(), [_NoCapabilityTool("noop")]),
    ],
)
def test_loop_construction_accepts_tools_the_environment_supports(
    environment: Environment, tools: list[Tool]
) -> None:
    """Construction succeeds, so the loop is usable for the run() tests above."""
    _loop(_FakeModelAPI([]), ToolRegistry(tools), environment)


def test_a_transient_api_failure_is_retried_within_the_run() -> None:
    """One flaky model call must not discard the whole trajectory."""
    registry = ToolRegistry([_FakeTool("submit", is_terminal=True)])
    model = _FakeModelAPI([_SUBMIT_TURN])
    healthy_create = model.chat.completions.create.side_effect
    model.chat.completions.create = Mock(
        side_effect=[ConnectionError("transient"), healthy_create()]
    )

    result = _loop(
        model,
        registry,
        _FakeEnvironment(),
        api_max_attempts=2,
        api_retry_sleep_secs=0,
    ).run(_TASK)

    assert (result.stopped_reason, result.answer) == (
        StoppedReason.SUBMITTED,
        "flag{x}",
    )


def test_an_api_failure_on_every_attempt_ends_the_run_as_error() -> None:
    """A model API that never recovers fails the run rather than hanging."""
    model = _FakeModelAPI([])
    model.chat.completions.create = Mock(side_effect=ConnectionError("down"))

    result = _loop(
        model,
        ToolRegistry(),
        _FakeEnvironment(),
        api_max_attempts=3,
        api_retry_sleep_secs=0,
    ).run(_TASK)

    assert result.stopped_reason == StoppedReason.ERROR
    assert result.error is not None
    assert result.error.startswith(
        "Agent loop error: RuntimeError: Model API call failed"
    )
    assert model.chat.completions.create.call_count == 3


def test_an_empty_choices_response_is_retried() -> None:
    """No choices is a transient API symptom, so it retries like a raised error."""
    model = _FakeModelAPI([_turn("I'm done")])
    healthy_create = model.chat.completions.create.side_effect
    empty = Mock()
    empty.choices = []
    model.chat.completions.create = Mock(side_effect=[empty, healthy_create()])

    result = _loop(
        model,
        ToolRegistry(),
        _FakeEnvironment(),
        # The retry is the subject here, so end on the first prose turn rather
        # than spend the mock's canned responses on nudges.
        max_consecutive_no_tool_calls=1,
        api_max_attempts=2,
        api_retry_sleep_secs=0,
    ).run(_TASK)

    assert (result.stopped_reason, result.error) == (
        StoppedReason.NO_TOOL_CALLS,
        None,
    )


def test_a_dead_sandbox_is_rebuilt_without_the_agent_seeing_it() -> None:
    """A recovered failure is invisible: the agent gets the retry's result.

    The whole point of the fix — an infra error fed back as a tool result reads to
    the agent as its own mistake, and it burns budget retrying.
    """
    env = _FakeEnvironment(command_failures=1)
    registry = ToolRegistry(
        [_FakeTool("bash_exec"), _FakeTool("submit", is_terminal=True)]
    )
    model = _FakeModelAPI(
        [_turn("Looking", ("call_1", "bash_exec", _LIST_ARGUMENTS)), _SUBMIT_TURN]
    )

    result = _loop(model, registry, env).run(_TASK)

    assert (result.stopped_reason, result.answer) == (
        StoppedReason.SUBMITTED,
        "flag{x}",
    )
    # No mention of the failure, and charged once rather than twice.
    assert result.tool_responses[0] == {
        "tool_name": "bash_exec",
        "response_truncated": "test output\n[49 tool-calls remaining]",
        "response_length": len("test output\n[49 tool-calls remaining]"),
        "timestamp": ANY,
    }
    assert result.tool_calls[0] == {
        "tool_name": "bash_exec",
        "args": {"command": "ls"},
        "timestamp": ANY,
    }
    # Rebuilt once: the initial setup plus one recovery.
    assert (env.setup_count, env.teardown_count) == (2, 2)


def test_a_sandbox_that_keeps_dying_ends_the_task_as_an_environment_error() -> None:
    """Consecutive failures stop the task rather than rebuilding forever."""
    env = _FakeEnvironment(command_failures=MAX_CONSECUTIVE_RECOVERIES + 1)
    registry = ToolRegistry([_FakeTool("bash_exec")])
    model = _FakeModelAPI([_turn("Looking", ("call_1", "bash_exec", _LIST_ARGUMENTS))])

    result = _loop(model, registry, env).run(_TASK)

    assert result == AgentResult(
        task_id="t",
        answer=None,
        answer_tool=None,
        messages=ANY,
        tool_calls=[],
        tool_responses=[],
        num_tool_calls=0,
        execution_time_seconds=ANY,
        stopped_reason=StoppedReason.ENVIRONMENT_ERROR,
        error=(
            f"Environment failed {MAX_CONSECUTIVE_RECOVERIES + 1} times "
            f"({MAX_CONSECUTIVE_RECOVERIES + 1} consecutively) and was not "
            f"recovered: sandbox is gone"
        ),
    )
    # The bound is on rebuilds attempted, so the last failure buys no rebuild.
    assert env.setup_count == MAX_CONSECUTIVE_RECOVERIES + 1


def test_intermittent_failures_are_bounded_by_the_total_not_the_streak() -> None:
    """Alternating success and failure never trips the consecutive counter."""
    env = _FakeEnvironment()
    registry = ToolRegistry([_FakeTool("bash_exec")])
    model = _FakeModelAPI(
        [
            _turn(f"Try {i}", (f"call_{i}", "bash_exec", _LIST_ARGUMENTS))
            for i in range(MAX_TOTAL_RECOVERIES + 1)
        ]
    )
    # One failure per call, each followed by a success, so the streak resets every
    # time and only the total can end the run.
    with patch.object(
        _FakeEnvironment,
        "run_command",
        side_effect=[
            EnvironmentUnavailableError("sandbox is gone") if i % 2 == 0 else (0, "ok")
            for i in range(2 * (MAX_TOTAL_RECOVERIES + 1))
        ],
    ):
        result = _loop(model, registry, env, max_turns=100).run(_TASK)

    assert result.stopped_reason == StoppedReason.ENVIRONMENT_ERROR
    assert result.error == (
        f"Environment failed {MAX_TOTAL_RECOVERIES + 1} times "
        f"(1 consecutively) and was not recovered: sandbox is gone"
    )


def test_recovery_that_overruns_the_task_timeout_stops_the_task() -> None:
    """The clock is checked during recovery, not only between turns.

    A cluster rebuild is a new claim plus a readiness wait, so three of them can
    spend minutes — churn that would otherwise be unbounded and invisible. Here the
    first rebuild alone outlasts the task, so the second failure ends it well inside
    the retry bounds.
    """
    env = _FakeEnvironment(command_failures=2, rebuild_delay=0.3)
    registry = ToolRegistry([_FakeTool("bash_exec")])
    model = _FakeModelAPI([_turn("Looking", ("call_1", "bash_exec", _LIST_ARGUMENTS))])

    result = _loop(model, registry, env, timeout_seconds=0.2).run(_TASK)

    assert result.stopped_reason == StoppedReason.ENVIRONMENT_ERROR
    assert result.error == (
        "Environment failed and the task's 0.2s timeout expired before it could "
        "be recovered: sandbox is gone"
    )
    # Checked before the rebuild, so the expired clock buys no further setup.
    assert env.setup_count == 2


def test_a_rebuild_that_itself_fails_ends_the_task() -> None:
    """A cluster that cannot stamp a new sandbox is not worth retrying into."""
    env = _FakeEnvironment(
        command_failures=1, rebuild_error=RuntimeError("no capacity")
    )
    registry = ToolRegistry([_FakeTool("bash_exec")])
    model = _FakeModelAPI([_turn("Looking", ("call_1", "bash_exec", _LIST_ARGUMENTS))])

    result = _loop(model, registry, env).run(_TASK)

    assert (result.stopped_reason, result.error) == (
        StoppedReason.ENVIRONMENT_ERROR,
        "Environment failed and could not be rebuilt: no capacity",
    )


def test_recovery_can_be_turned_off_for_a_benchmark_whose_agent_writes() -> None:
    """Rebuilding re-runs setup, which discards whatever the agent wrote.

    Silent recovery is only correct for a read-only workspace: a benchmark whose
    agent builds or patches must fail the task instead of continuing against a
    workspace missing its own work.
    """
    env = _FakeEnvironment(command_failures=1)
    registry = ToolRegistry([_FakeTool("bash_exec")])
    model = _FakeModelAPI([_turn("Looking", ("call_1", "bash_exec", _LIST_ARGUMENTS))])

    result = _loop(model, registry, env, recover_on_environment_failure=False).run(
        _TASK
    )

    assert (result.stopped_reason, result.error) == (
        StoppedReason.ENVIRONMENT_ERROR,
        "Environment failed and recovery is disabled: sandbox is gone",
    )
    assert env.setup_count == 1
