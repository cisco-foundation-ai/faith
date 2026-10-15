# Copyright 2025 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for the built-in agent tools.

Each tool is a declaration (the function-calling schema the model is shown, and
the environment capabilities its ``_invoke`` composes) plus a small amount of
result formatting. These tests pin the schema whole, pin the capability tuple,
and drive ``execute`` for each formatting rule.
"""

import pytest

from faith.agentic.core.capabilities import CommandExecution
from faith.agentic.core.environment import Environment
from faith.agentic.core.task import Task
from faith.agentic.core.tool import Tool
from faith.agentic.tools import BashExecTool, SubmitTool


class _RecordingEnvironment(Environment):
    """Environment double recording each capability call, with canned results."""

    def __init__(self, command_result: tuple[int, str] = (0, "cmd output")) -> None:
        self.commands: list[str] = []
        self._command_result = command_result

    def setup(self, task: Task) -> None:
        pass

    def run_command(self, command: str) -> tuple[int, str]:
        """Record the command and return the canned result."""
        self.commands.append(command)
        return self._command_result

    def teardown(self) -> None:
        pass


@pytest.mark.parametrize(
    ("tool", "expected"),
    [
        (BashExecTool(), (CommandExecution,)),
        (SubmitTool(), ()),
    ],
    ids=["bash_exec", "submit"],
)
def test_declared_capabilities(tool: Tool, expected: tuple[type, ...]) -> None:
    """Each tool declares exactly the capabilities its _invoke uses.

    The loop rejects a tool the environment cannot serve by reading these, so an
    over-declaration makes a workable benchmark unrunnable and a missing one turns
    into an AttributeError mid-run instead of a rejected config.
    """
    assert tool.required_capabilities == expected


@pytest.mark.parametrize(
    ("tool", "expected_name", "expected_parameters"),
    [
        (
            BashExecTool(),
            "bash_exec",
            {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "The bash command to execute",
                    },
                    "max_chars": {
                        "type": "integer",
                        "description": "Max output chars (default 2000)",
                        "default": 2000,
                    },
                },
                "required": ["command"],
            },
        ),
        (
            SubmitTool(),
            "submit",
            {
                "type": "object",
                "properties": {
                    "answer": {
                        "type": "string",
                        "description": "Your final answer to the task",
                    }
                },
                "required": ["answer"],
            },
        ),
    ],
    ids=["bash_exec", "submit"],
)
def test_advertised_name_and_parameters(
    tool: Tool, expected_name: str, expected_parameters: dict
) -> None:
    """The function-call name and the whole JSON Schema the model is shown.

    Compared whole: the model rejects a malformed schema, `execute` enforces the
    `required` list, and the registry keys the tool by `name`, so a stray or
    renamed property is a live failure and not a cosmetic one.
    """
    assert (tool.name, tool.parameters) == (expected_name, expected_parameters)


@pytest.mark.parametrize(
    ("tool", "promise"),
    [
        # Enforced by construction: `cd`/`export` do not persist between calls.
        (BashExecTool(), "fresh shell"),
        # The exit-code annotation _invoke appends, so the model can read it.
        (BashExecTool(), "[Exit code: N]"),
    ],
    ids=["fresh_shell", "exit_code"],
)
def test_the_description_states_what_the_tool_actually_enforces(
    tool: Tool, promise: str
) -> None:
    """Each description carries the promises the model has to plan around.

    Only these claims are pinned, not the surrounding prose: they are the ones
    something else in the system enforces, so silently dropping one leaves the
    model unable to diagnose the resulting failure.
    """
    assert promise in tool.description


@pytest.mark.parametrize(
    ("command_result", "expected"),
    [
        ((0, "ok"), "ok\n[Exit code: 0]"),
        ((1, "boom"), "boom\n[Exit code: 1]"),
        # No output: the annotation is the whole result, with no leading blank line,
        # so the model can still tell success from failure.
        ((0, ""), "[Exit code: 0]"),
        ((1, ""), "[Exit code: 1]"),
    ],
    ids=["output_ok", "output_failed", "silent_ok", "silent_failed"],
)
def test_bash_exec_runs_the_command_and_annotates_the_exit_code(
    command_result: tuple[int, str], expected: str
) -> None:
    """The command reaches the environment and the exit code is always reported."""
    env = _RecordingEnvironment(command_result)
    assert BashExecTool().execute(env, {"command": "ls -la"}) == expected
    assert env.commands == ["ls -la"]


def test_bash_exec_truncation_reports_the_true_total() -> None:
    """A huge output is cut to the default cap and the marker states the real size.

    The total is the load-bearing half: it is what tells the agent to narrow its
    command rather than conclude it saw the whole output.
    """
    env = _RecordingEnvironment((0, "x" * 3_000_000))
    assert BashExecTool().execute(env, {"command": "grep -r x ."}) == (
        "x" * 2000
        + "\n\n[TRUNCATED — 3000000 total chars, showing first 2000]"
        + "\n[Exit code: 0]"
    )


@pytest.mark.parametrize(
    ("args", "expected_chars"),
    [
        ({"command": "ls"}, 2000),
        ({"command": "ls", "max_chars": 100}, 100),
        # A model may send anything the schema does not enforce; each of these falls
        # back to the default rather than truncating to nothing or one character.
        ({"command": "ls", "max_chars": 0}, 2000),
        ({"command": "ls", "max_chars": -5}, 2000),
        ({"command": "ls", "max_chars": None}, 2000),
        ({"command": "ls", "max_chars": "lots"}, 2000),
        ({"command": "ls", "max_chars": True}, 2000),
    ],
    ids=["absent", "requested", "zero", "negative", "null", "string", "bool"],
)
def test_bash_exec_honours_a_usable_max_chars_and_ignores_the_rest(
    args: dict, expected_chars: int
) -> None:
    """max_chars caps the output when it is a positive int, and is ignored otherwise.

    The default is written out rather than imported, so a change to the constant
    fails here instead of silently redefining what the test asserts.
    """
    env = _RecordingEnvironment((0, "y" * 5000))
    assert BashExecTool().execute(env, args) == (
        "y" * expected_chars
        + f"\n\n[TRUNCATED — 5000 total chars, showing first {expected_chars}]"
        + "\n[Exit code: 0]"
    )


def test_submit_confirms_without_touching_the_environment() -> None:
    """submit echoes a confirmation and never calls an environment primitive."""
    env = _RecordingEnvironment()
    assert SubmitTool().execute(env, {"answer": "flag{test123}"}) == (
        "Answer submitted: flag{test123}"
    )
    assert not env.commands


def test_submit_is_terminal_carrying_its_answer_field() -> None:
    """submit is a terminal tool naming the key its answer rides in.

    The loop discovers terminal-ness via these properties, not the tool name, and
    reads the answer from a key the tool must itself accept.
    """
    tool = SubmitTool()
    assert tool.is_terminal
    assert tool.answer_field == "answer"
    assert tool.answer_field in tool.parameters["properties"]


@pytest.mark.parametrize(
    ("tool", "args", "expected"),
    [
        (BashExecTool(), {}, "Error: bash_exec requires 'command' argument"),
        (SubmitTool(), {}, "Error: submit requires 'answer' argument"),
    ],
    ids=["bash_exec", "submit"],
)
def test_a_missing_required_argument_is_rejected_before_the_environment_is_touched(
    tool: Tool, args: dict, expected: str
) -> None:
    """A malformed call is an error string fed back to the model, and never acts.

    Each tool's `required` list is what makes this fire, so it is checked per tool
    rather than only on the base class.
    """
    env = _RecordingEnvironment()
    assert tool.execute(env, args) == expected
    assert not env.commands
