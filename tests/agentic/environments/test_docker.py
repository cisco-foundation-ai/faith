# Copyright 2025 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for DockerEnvironment.

A fake Docker client is injected so these run without a Docker daemon: they
assert on the container/exec calls the environment makes, not on real containers.
The archive shapes themselves are pinned in test_archive.py, and the path guard
in test_workspace.py; here an upload is checked only for reaching put_archive
with the right destination and payload.
"""

import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import ANY, Mock, patch

import pytest
import requests
from docker import errors as docker_errors

from faith.agentic.core.environment import EnvironmentUnavailableError
from faith.agentic.core.task import Task
from faith.agentic.environments._archive import tar_bytes, tar_directory
from faith.agentic.environments.docker import DockerEnvironment


class _FakeContainer:
    """Records exec_run / put_archive / stop calls; returns canned exec results."""

    def __init__(
        self,
        exec_results: dict[str, tuple[int, bytes]] | None = None,
        put_archive_result: bool = True,
        stop_error: Exception | None = None,
        exec_error: Exception | None = None,
    ):
        self.exec_calls: list[tuple[list[str], str]] = []
        self.archives: list[tuple[str, bytes]] = []
        self.stop_attempts = 0
        self.exec_results = exec_results or {}
        self.put_archive_result = put_archive_result
        self.stop_error = stop_error
        self.exec_error = exec_error

    def start(self) -> None:
        """Start the fake container."""

    def remove(self, force: bool = False) -> None:
        """Remove the fake container after failed startup."""
        assert force

    def exec_run(self, cmd: list[str], workdir: str) -> tuple[int, bytes]:
        """Record the exec and return the result canned for cmd[0]."""
        self.exec_calls.append((cmd, workdir))
        if self.exec_error is not None:
            raise self.exec_error
        return self.exec_results.get(cmd[0], (0, b""))

    def put_archive(self, path: str, data: bytes) -> bool:
        """Record the upload."""
        self.archives.append((path, data))
        return self.put_archive_result

    def stop(self, timeout: int) -> None:
        """Record the stop, raising if scripted to."""
        self.stop_attempts += 1
        assert timeout == 5
        if self.stop_error is not None:
            raise self.stop_error


class _FakeDockerClient:
    """Fake ``docker.DockerClient`` recording the containers.run kwargs."""

    def __init__(self, container: _FakeContainer, run_error: Exception | None = None):
        self.run_kwargs: dict[str, Any] | None = None
        self.containers = SimpleNamespace(create=self._run, get=self._get)
        self._container = container
        self._run_error = run_error

    def _run(self, **kwargs: Any) -> _FakeContainer:
        self.run_kwargs = kwargs
        if self._run_error is not None:
            raise self._run_error
        return self._container

    def _get(self, name: str) -> _FakeContainer:
        raise docker_errors.NotFound(name)


def _task(data_path: Path | None = None, environment_config=None) -> Task:
    return Task(
        id="t1",
        prompt="do the thing",
        ground_truth=None,
        data_path=data_path,
        environment_config=environment_config or {},
    )


def _make_env(
    container: _FakeContainer | None = None,
    run_error: Exception | None = None,
    **config: Any,
) -> tuple[DockerEnvironment, _FakeDockerClient, _FakeContainer]:
    container = container or _FakeContainer()
    client = _FakeDockerClient(container, run_error)
    env = DockerEnvironment({"agent_image": "img:latest", **config}, client=client)
    return env, client, container


def _setup_env(
    container: _FakeContainer | None = None, **config: Any
) -> tuple[DockerEnvironment, _FakeContainer]:
    env, _, container = _make_env(container, **config)
    env.setup(_task())
    return env, container


def test_requires_agent_image() -> None:
    """Without an image there is nothing to run, and no default is safe to guess."""
    with pytest.raises(
        ValueError, match="DockerEnvironment config requires 'agent_image'"
    ):
        DockerEnvironment({}, client=_FakeDockerClient(_FakeContainer()))


def test_a_container_that_cannot_start_is_an_environment_failure() -> None:
    """A daemon that cannot start a container is not a broken run, so the log says so.

    The loop classifies on this type alone, and only ENVIRONMENT_ERROR points at
    the infrastructure rather than at FAITH.
    """
    env, _, _ = _make_env(run_error=RuntimeError("no such image"))

    with pytest.raises(EnvironmentUnavailableError) as excinfo:
        env.setup(_task())

    assert str(excinfo.value) == "Failed to start a container: no such image"


@pytest.mark.parametrize(
    "exec_error",
    [
        requests.ConnectionError("gone"),
        docker_errors.NotFound("gone"),
        # Derives from neither hierarchy the other two cover.
        docker_errors.StreamParseError("gone"),
    ],
    ids=["transport_error", "sdk_error", "unrelated_sdk_error"],
)
def test_a_container_dying_during_setup_stays_an_environment_failure(
    tmp_path: Path, exec_error: Exception
) -> None:
    """The upload's own wrap must not downgrade a container that died under it.

    Whatever the SDK raises is a dead container: an escaped one would be recorded
    as a broken run instead of triggering recovery.
    """
    data = tmp_path / "repo.zip"
    data.write_bytes(b"not really a zip")
    env, _, _ = _make_env(_FakeContainer(exec_error=exec_error))

    with pytest.raises(EnvironmentUnavailableError) as excinfo:
        env.setup(_task(data_path=data))

    assert str(excinfo.value) == "Docker container unavailable: gone"


def test_setup_runs_a_detached_isolated_labeled_container() -> None:
    """The whole run() call, since an unasserted kwarg is an unnoticed default.

    Networking is off so a benchmark that never asks for egress cannot reach the
    internet; there are no volumes, so the agent cannot read or mutate host
    files; the labels make a container leaked by a killed run attributable and
    bulk-removable (``docker ps -aq -f label=faith.managed=true``); and the
    process cap, dropped capabilities and no-new-privileges bound what a command
    the model wrote can do to the host.
    """
    env, client, _ = _make_env()
    env.setup(_task())
    assert client.run_kwargs == {
        "name": ANY,
        "image": "img:latest",
        "command": "sleep infinity",
        "detach": True,
        "working_dir": "/workspace",
        "environment": {},
        "network_mode": "none",
        "auto_remove": True,
        "labels": {
            "faith.managed": "true",
            "faith.task-id": "t1",
            "faith.pid": str(os.getpid()),
            "faith.instance-id": ANY,
        },
        "pids_limit": 512,
        "cap_drop": ["ALL"],
        "security_opt": ["no-new-privileges"],
    }
    assert client.run_kwargs["name"] == (
        f"faith-{client.run_kwargs['labels']['faith.instance-id']}"
    )


@pytest.mark.parametrize(
    ("config", "environment_config", "expected"),
    [
        ({"network": "bridge"}, {}, {"network_mode": "bridge"}),
        ({"environment": {"CC": "clang"}}, {}, {"environment": {"CC": "clang"}}),
        ({"mount_path": "/app"}, {}, {"working_dir": "/app"}),
        ({"memory_limit": "2g"}, {}, {"mem_limit": "2g"}),
        ({"cpu_limit": 1.5}, {}, {"nano_cpus": 1_500_000_000}),
        ({"pids_limit": 4096}, {}, {"pids_limit": 4096}),
        # 0 is a value, not a missing key: it asks Docker for no cap at all.
        ({"pids_limit": 0}, {}, {"pids_limit": 0}),
        # Added back on top of the dropped set, which is not itself configurable.
        (
            {"capabilities": ["DAC_OVERRIDE", "CHOWN"]},
            {},
            {"cap_add": ["DAC_OVERRIDE", "CHOWN"], "cap_drop": ["ALL"]},
        ),
        # A per-task image overrides the benchmark-wide one.
        ({}, {"agent_image": "other:v2"}, {"image": "other:v2"}),
        # A non-string override is ignored rather than passed through.
        ({}, {"agent_image": 7}, {"image": "img:latest"}),
    ],
    ids=[
        "network",
        "environment",
        "mount_path",
        "memory_limit",
        "cpu_limit",
        "pids_limit",
        "unlimited_pids",
        "capabilities",
        "per_task_image",
        "non_string_per_task_image",
    ],
)
def test_config_reaches_the_container_run_call(
    config: dict[str, Any], environment_config: dict[str, Any], expected: dict[str, Any]
) -> None:
    env, client, _ = _make_env(**config)
    env.setup(_task(environment_config=environment_config))
    assert client.run_kwargs is not None
    assert {k: client.run_kwargs.get(k) for k in expected} == expected


def test_the_limit_kwargs_are_omitted_when_unset() -> None:
    """docker-py reads a present-but-None mem_limit as a value, not as a default."""
    env, client, _ = _make_env()
    env.setup(_task())
    assert client.run_kwargs is not None
    assert not {"mem_limit", "nano_cpus", "volumes"} & set(client.run_kwargs)


def test_a_working_subdir_moves_the_agents_commands_but_not_setup(
    tmp_path: Path,
) -> None:
    """Setup keeps the root: the subdir only exists once the archive is extracted.

    Both go through the same `_exec`, so the two workdirs are asserted together —
    an extraction that followed the agent into the subdir would look for an
    archive that was uploaded one level up.
    """
    data = tmp_path / "repo.zip"
    data.write_bytes(b"PK\x03\x04fake")
    env, _, container = _make_env(working_subdir="repo")
    env.setup(_task(data_path=data))
    env.run_command("ls")
    assert [workdir for _, workdir in container.exec_calls] == [
        "/workspace",
        "/workspace/repo",
    ]
    assert container.archives == [
        ("/workspace", tar_bytes([("repo.zip", b"PK\x03\x04fake")]))
    ]


def test_setup_copies_a_data_directory_in_as_one_archive(tmp_path: Path) -> None:
    """Copied, not bind-mounted, so the workspace is writable and the host is not."""
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "a.c").write_bytes(b"int main(){}")
    env, client, container = _make_env()
    env.setup(_task(data_path=tmp_path))
    assert container.archives == [("/workspace", tar_directory(tmp_path))]
    assert client.run_kwargs is not None and "volumes" not in client.run_kwargs


def test_setup_copies_a_single_data_file_to_the_workspace_root(tmp_path: Path) -> None:
    data = tmp_path / "repo.tar"
    data.write_bytes(b"payload")
    env, _, container = _make_env()
    env.setup(_task(data_path=data))
    assert container.archives == [("/workspace", tar_bytes([("repo.tar", b"payload")]))]
    # Not a zip, so nothing is extracted.
    assert container.exec_calls == []


def test_setup_unzips_a_zip_data_path_and_removes_the_archive(tmp_path: Path) -> None:
    """The zip must not be left where the agent would see it as task data.

    The argv is bare `bash`, not wrapped in `timeout`: extracting a multi-GB
    snapshot is our own setup work and must not compete with an agent-sized bound.
    """
    data = tmp_path / "repo.zip"
    data.write_bytes(b"PK\x03\x04fake")
    env, _, container = _make_env()
    env.setup(_task(data_path=data))
    assert container.archives == [
        ("/workspace", tar_bytes([("repo.zip", b"PK\x03\x04fake")]))
    ]
    assert container.exec_calls == [
        (
            [
                "bash",
                "-c",
                "unzip -o ./repo.zip; status=$?; rm -f ./repo.zip; exit $status",
            ],
            "/workspace",
        )
    ]


@pytest.mark.parametrize(
    ("container", "data_name", "expected_message"),
    [
        (
            _FakeContainer(exec_results={"bash": (1, b"  End-of-central-directory  ")}),
            "repo.zip",
            "Failed to set up container: Failed to unzip repo.zip: "
            "End-of-central-directory",
        ),
        # put_archive reports failure by returning False rather than raising.
        (
            _FakeContainer(put_archive_result=False),
            "repo.zip",
            "Failed to set up container: Failed to copy files into '/workspace'",
        ),
    ],
    ids=["unzip_failed", "copy_failed"],
)
def test_setup_reports_a_failed_data_upload(
    tmp_path: Path,
    container: _FakeContainer,
    data_name: str,
    expected_message: str,
) -> None:
    """Wrapped in the setup message, so a run log says which stage failed."""
    data = tmp_path / data_name
    data.write_bytes(b"not really a zip")
    env, _, _ = _make_env(container)
    with pytest.raises(RuntimeError) as excinfo:
        env.setup(_task(data_path=data))
    assert str(excinfo.value) == expected_message


@pytest.mark.parametrize(
    ("exec_result", "expected"),
    [
        ((0, b"hello\n"), (0, "hello\n")),
        # Raw exit code: annotating a failure is the bash_exec tool's job.
        ((1, b"nope"), (1, "nope")),
        # Undecodable output must not crash the run.
        ((0, b"\xff"), (0, "�")),
        # 124 is `timeout`'s own code for a command it had to kill. The notice is
        # appended, so output printed before the kill is not lost.
        (
            (124, b"partial output"),
            (124, "partial output\nERROR: Command timed out after 60s"),
        ),
        # A failure that is not an expiry gets no notice, whatever it printed.
        ((125, b"other failure"), (125, "other failure")),
    ],
    ids=[
        "success",
        "nonzero_exit",
        "undecodable_output",
        "timed_out",
        "other_failure",
    ],
)
def test_run_command_bounds_the_agents_command_in_the_workspace(
    exec_result: tuple[int, bytes], expected: tuple[int, str]
) -> None:
    """The agent's command runs under coreutils `timeout` as the exec's PID 1.

    Neither SDK can bound the command itself, so the wrapper is what makes a
    runaway `grep -r` survivable. Argv is compared whole: `timeout` must be the
    exec'd program, not something inside the string bash interprets.
    """
    env, container = _setup_env(_FakeContainer(exec_results={"timeout": exec_result}))
    assert env.run_command("echo hello") == expected
    assert container.exec_calls == [
        (["timeout", "-k", "5", "60", "bash", "-c", "echo hello"], "/workspace")
    ]


@pytest.mark.parametrize(
    ("config", "expected_seconds"),
    [
        ({}, "60"),
        ({"command_timeout_seconds": 10}, "10"),
        # 0 means "no bound" to coreutils `timeout`, so it must survive as 0
        # rather than falling back to the default.
        ({"command_timeout_seconds": 0}, "0"),
    ],
    ids=["default", "configured", "zero_is_kept"],
)
def test_the_command_bound_comes_from_the_config(
    config: dict[str, Any], expected_seconds: str
) -> None:
    """command_timeout_seconds is the command's bound, not the SDK's."""
    env, container = _setup_env(**config)
    env.run_command("ls")
    assert container.exec_calls == [
        (
            ["timeout", "-k", "5", expected_seconds, "bash", "-c", "ls"],
            "/workspace",
        )
    ]


def test_the_timeout_notice_reports_the_configured_bound() -> None:
    """The notice names the bound that fired, so the agent can size its next call."""
    env, _ = _setup_env(
        _FakeContainer(exec_results={"timeout": (124, b"")}),
        command_timeout_seconds=10,
    )
    assert env.run_command("sleep 30") == (
        124,
        "\nERROR: Command timed out after 10s",
    )


def test_a_primitive_before_setup_raises() -> None:
    env, _, _ = _make_env()
    with pytest.raises(RuntimeError) as excinfo:
        env.run_command("ls")
    assert str(excinfo.value) == "No container running. Call setup() first."


@pytest.mark.parametrize(
    "stop_error", [None, RuntimeError("already gone")], ids=["stops", "already_gone"]
)
def test_teardown_stops_the_container_once_and_never_raises(
    stop_error: Exception | None,
) -> None:
    """A container that is already gone still leaves the environment torn down."""
    env, container = _setup_env(_FakeContainer(stop_error=stop_error))
    env.teardown()
    env.teardown()  # The first attempt cleared the container, failed or not.
    assert container.stop_attempts == 1


@pytest.mark.parametrize(
    ("config", "expected_timeout"),
    [
        ({}, 600),
        # Only a bound long enough to outlast the 600s floor moves the client's.
        ({"command_timeout_seconds": 900}, 960),
    ],
    ids=["default", "over_the_floor"],
)
def test_the_client_is_built_once_with_a_timeout_from_the_command_bound(
    config: dict[str, Any], expected_timeout: int
) -> None:
    """No client is built at construction, so importing needs no Docker daemon.

    The SDK timeout is no longer configurable: its only job was to outlast the
    longest exec, which `command_timeout_seconds` now makes a known quantity.
    """
    container = _FakeContainer()
    mock_from_env = Mock(return_value=_FakeDockerClient(container))
    with patch(
        "faith.agentic.environments.docker._require_sdk",
        return_value=SimpleNamespace(from_env=mock_from_env),
    ):
        env = DockerEnvironment({"agent_image": "img:latest", **config})
        mock_from_env.assert_not_called()
        # Recovery re-runs setup, which must reuse the client, not rebuild it.
        env.setup(_task())
        env.setup(_task())
    mock_from_env.assert_called_once_with(timeout=expected_timeout)
