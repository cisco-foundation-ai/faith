# Copyright 2025 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

"""Bash command execution tool."""

from faith.agentic.core.capabilities import CommandExecution
from faith.agentic.core.tool import Tool
from faith.agentic.tools._truncate import truncate

# Matched to the reference harness. Deliberately small: the truncation marker
# tells the agent to narrow its command, which a larger window would not.
DEFAULT_MAX_OUTPUT_CHARS = 2000


class BashExecTool(Tool):
    """Runs a shell command in the environment.

    Lets the agent explore the filesystem, run programs, and compile or inspect
    code. Sandboxing and per-call timeouts are the environment's concern.
    """

    required_capabilities = (CommandExecution,)

    @property
    def name(self) -> str:
        return "bash_exec"

    @property
    def description(self) -> str:
        return (
            "Execute a bash command in the environment. "
            "Returns the command output (stdout + stderr), followed by "
            "'[Exit code: N]'. "
            "Each call runs in a fresh shell, so 'cd' and 'export' do not persist "
            "between calls; chain them in one command instead. "
            "Use this to explore files, run programs, compile code, etc."
        )

    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "The bash command to execute",
                },
                "max_chars": {
                    "type": "integer",
                    "description": "Max output chars (default 2000)",
                    "default": DEFAULT_MAX_OUTPUT_CHARS,
                },
            },
            "required": ["command"],
        }

    def _invoke(self, environment: CommandExecution, args: dict) -> str:
        """Run the command, truncating output before annotating the exit code.

        The exit code is appended last so truncation can never cut it off.
        """
        exit_code, output = environment.run_command(args["command"])
        output = truncate(output, _max_chars(args))
        separator = "\n" if output else ""
        return f"{output}{separator}[Exit code: {exit_code}]"


def _max_chars(args: dict) -> int:
    """The requested output cap, ignoring a non-positive or non-integer value.

    Tool._validate_args only checks that required keys are present, so a model
    sending a string or a huge number reaches here; bool is excluded because it is
    an int in Python and output[:True] is one character.
    """
    requested = args.get("max_chars")
    if isinstance(requested, int) and not isinstance(requested, bool) and requested > 0:
        return requested
    return DEFAULT_MAX_OUTPUT_CHARS
