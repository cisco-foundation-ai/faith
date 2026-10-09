# Copyright 2025 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

"""Wall-clock bounds for the commands an agent runs, shared by both backends.

Neither SDK can bound the command itself: docker-py's ``exec_run`` takes no
timeout, and the k8s sandbox's ``commands.run`` timeout is an HTTP one whose
expiry raises while the process survives. So the bound is imposed inside the
container with coreutils ``timeout``, which becomes the exec's own PID 1 and
actually kills the process. The two backends differ only in argv-vs-string.
"""

import shlex
from typing import Any

# `timeout`'s exit code when it had to kill the command.
_EXPIRED = 124

# Seconds between the SIGTERM and the SIGKILL that follows if it was ignored.
_GRACE = 5

# The bound applied when a config names none.
_DEFAULT_SECONDS = 60


def command_timeout(configured: Any) -> int:
    """The per-command bound from an environment config, defaulted.

    0 is ``timeout``'s own "no bound", so only None takes the default.
    """
    return _DEFAULT_SECONDS if configured is None else int(configured)


def bounded_argv(command: str, seconds: int) -> list[str]:
    """`command` under a hard wall-clock bound, as argv for `docker exec`."""
    return ["timeout", "-k", str(_GRACE), str(seconds), "bash", "-c", command]


def bounded_shell(command: str, seconds: int) -> str:
    """`command` under a hard wall-clock bound, as one shell string."""
    return f"timeout -k {_GRACE} {seconds} bash -c {shlex.quote(command)}"


def timeout_notice(exit_code: int, seconds: int) -> str | None:
    """The tool-visible message for a bound that expired, if it did."""
    if exit_code != _EXPIRED:
        return None
    return f"ERROR: Command timed out after {seconds}s"


def sdk_timeout(seconds: int) -> int:
    """Headroom for the SDK's own wait over the longest command it can carry.

    Derived rather than configured: its only job was to outlast the exec, which
    is now a known quantity. Floored at the previous default so no existing
    deployment gets a shorter client timeout than it had.
    """
    return max(600, seconds + 60)
