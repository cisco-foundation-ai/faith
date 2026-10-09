# Copyright 2025 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for the command-bounding helpers shared by both environments.

The two backends are driven through their own tests; these pin the wrapper's
exact shape, since it is what a container actually execs and a mis-shaped bound
fails at run time rather than at import.
"""

import pytest

from faith.agentic.environments._command import (
    bounded_argv,
    bounded_shell,
    command_timeout,
    sdk_timeout,
    timeout_notice,
)


@pytest.mark.parametrize(
    ("configured", "expected"),
    [
        (None, 60),
        # 0 is `timeout`'s own "no bound", so it must survive rather than default.
        (0, 0),
        (10, 10),
        ("30", 30),
    ],
    ids=["absent", "unbounded", "bounded", "from_yaml_string"],
)
def test_command_timeout(configured: object, expected: int) -> None:
    """Only an absent bound takes the default; 0 is a value, not a missing one."""
    assert command_timeout(configured) == expected


@pytest.mark.parametrize(
    ("command", "seconds", "expected"),
    [
        ("ls -l", 60, ["timeout", "-k", "5", "60", "bash", "-c", "ls -l"]),
        # argv needs no quoting: the command is already one element, so shell
        # metacharacters reach bash intact and cannot escape the wrapper.
        (
            "grep 'a b' . ; rm -rf /",
            10,
            ["timeout", "-k", "5", "10", "bash", "-c", "grep 'a b' . ; rm -rf /"],
        ),
        ("ls", 0, ["timeout", "-k", "5", "0", "bash", "-c", "ls"]),
    ],
    ids=["plain", "metacharacters", "unbounded"],
)
def test_bounded_argv(command: str, seconds: int, expected: list[str]) -> None:
    """`timeout` is the exec'd program, so it becomes the exec's own PID 1."""
    assert bounded_argv(command, seconds) == expected


@pytest.mark.parametrize(
    ("command", "seconds", "expected"),
    [
        # shlex.quote leaves a single shell word alone.
        ("ls", 60, "timeout -k 5 60 bash -c ls"),
        ("ls -l", 60, "timeout -k 5 60 bash -c 'ls -l'"),
        # The quoting is load-bearing here in a way it is not for argv: without it
        # either of these would terminate `bash -c`'s argument early.
        ("grep 'a b' .", 60, """timeout -k 5 60 bash -c 'grep '"'"'a b'"'"' .'"""),
        ("ls; rm -rf /", 60, "timeout -k 5 60 bash -c 'ls; rm -rf /'"),
        ("ls", 0, "timeout -k 5 0 bash -c ls"),
    ],
    ids=["one_word", "plain", "embedded_quotes", "semicolon", "unbounded"],
)
def test_bounded_shell(command: str, seconds: int, expected: str) -> None:
    """The command survives as one argument to `bash -c`, whatever it contains."""
    assert bounded_shell(command, seconds) == expected


@pytest.mark.parametrize(
    ("exit_code", "expected"),
    [
        (124, "ERROR: Command timed out after 10s"),
        # Only 124 is an expiry. 125 and 137 are the neighbours most easily
        # confused with one: `timeout`'s own failure, and SIGKILL.
        (0, None),
        (1, None),
        (125, None),
        (137, None),
    ],
    ids=["expired", "success", "failure", "timeout_itself_failed", "sigkill"],
)
def test_timeout_notice(exit_code: int, expected: str | None) -> None:
    """Only `timeout`'s own 124 produces the notice, and it names the bound."""
    assert timeout_notice(exit_code, 10) == expected


@pytest.mark.parametrize(
    ("seconds", "expected"),
    [
        # Floored at the 600 the SDK timeout used to default to, so no existing
        # deployment's client wait gets shorter than it was.
        (60, 600),
        (0, 600),
        (540, 600),
        # 60s of headroom over the longest command the bound now permits.
        (541, 601),
        (900, 960),
    ],
    ids=["default", "unbounded", "at_the_floor", "just_over", "well_over"],
)
def test_sdk_timeout(seconds: int, expected: int) -> None:
    """The SDK's wait is derived from the command bound, never shorter than 600."""
    assert sdk_timeout(seconds) == expected
