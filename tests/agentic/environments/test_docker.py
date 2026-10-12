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

from faith._types.config.agentic import AgenticEnvironmentConfig, resolve_environment
from faith.agentic.core.environment import EnvironmentUnavailableError
from faith.agentic.core.task import Task
from faith.agentic.environments._archive import tar_bytes, tar_directory
from faith.agentic.environments._build import derived_image_tag
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


class _FakeImages:
    """Fake ``client.images`` collection: records build() kwargs.

    ``present`` names the tags that already exist locally, so ``list`` answers the
    existence check the environment makes before building.
    """

    def __init__(self, present: set[str] | None = None):
        self.present = present if present is not None else set()
        self.build_kwargs: dict | None = None
        self.build_error: Exception | None = None
        self.pulled: list[str] = []
        self.pull_error: Exception | None = None

    def list(self, name: str) -> list[SimpleNamespace]:
        """Return a one-image list if the tag is present, else nothing."""
        return [SimpleNamespace(tags=[name])] if name in self.present else []

    def pull(self, name: str) -> SimpleNamespace:
        """Record the pull, marking the tag present afterwards."""
        self.pulled.append(name)
        if self.pull_error is not None:
            raise self.pull_error
        self.present.add(name)
        return SimpleNamespace(tags=[name])

    def build(self, **kwargs: Any) -> tuple[SimpleNamespace, Any]:
        """Record the build kwargs, marking the tag present afterwards."""
        self.build_kwargs = kwargs
        if self.build_error is not None:
            raise self.build_error
        self.present.add(kwargs["tag"])
        return SimpleNamespace(tags=[kwargs["tag"]]), []


class _FakeDockerClient:
    """Fake ``docker.DockerClient`` recording the containers.run and images.build kwargs."""

    def __init__(
        self,
        container: _FakeContainer,
        present: set[str] | None = None,
        run_error: Exception | None = None,
    ):
        self.run_kwargs: dict[str, Any] | None = None
        self.containers = SimpleNamespace(create=self._run, get=self._get)
        self.images = _FakeImages(present)
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
    client = _FakeDockerClient(container, run_error=run_error)
    env = DockerEnvironment({"agent_image": "img:latest", **config}, client=client)
    return env, client, container


def _setup_env(
    container: _FakeContainer | None = None, **config: Any
) -> tuple[DockerEnvironment, _FakeContainer]:
    env, _, container = _make_env(container, **config)
    env.setup(_task())
    return env, container


def test_requires_a_dockerfile_to_build_or_an_image_to_pull() -> None:
    """With neither there is nothing to run, and no default is safe to guess."""
    with pytest.raises(
        ValueError, match="requires 'agent_dockerfile' .* or 'agent_image'"
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


@pytest.mark.parametrize(
    ("image", "expected_type"),
    # An unquoted `agent_image: 3.12` in a yaml is a float, not a name. Rejected
    # rather than stringified, which would produce a plausible tag nothing serves.
    [(3.12, "float"), (True, "bool"), (["img:latest"], "list")],
    ids=["float", "bool", "list"],
)
def test_a_non_string_agent_image_is_rejected_naming_what_it_got(
    image: object, expected_type: str
) -> None:
    with pytest.raises(ValueError) as excinfo:
        DockerEnvironment(
            {"agent_image": image}, client=_FakeDockerClient(_FakeContainer())
        )
    assert str(excinfo.value) == (
        f"DockerEnvironment config 'agent_image' must be a string, got "
        f"{expected_type}: {image!r}"
    )


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


def test_the_workspace_root_is_fixed_and_a_configured_mount_path_is_ignored() -> None:
    """The path the prompt advertises, so a config must not be able to move it."""
    env, _, _ = _make_env(mount_path="/app")
    assert env.workspace_dir == "/workspace"


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


def _dockerfile(tmp_path: Path) -> Path:
    path = tmp_path / "Dockerfile"
    path.write_text("FROM ubuntu:24.04\n")
    return path


def test_builds_the_image_when_the_derived_tag_is_absent(tmp_path: Path) -> None:
    """A benchmark shipping a Dockerfile needs no manual `docker build` step.

    The whole build call, since an unasserted kwarg is an unnoticed default. The tag
    is FAITH's own, derived from the context, so it never appears in the config; the
    label is how the images left behind by earlier recipes are reclaimed.
    """
    dockerfile = _dockerfile(tmp_path)
    env = DockerEnvironment(
        {"agent_dockerfile": str(dockerfile)},
        client=(client := _FakeDockerClient(_FakeContainer())),
    )
    env.setup(_task())
    assert client.images.build_kwargs == {
        "path": str(tmp_path),
        "dockerfile": "Dockerfile",
        "tag": derived_image_tag(dockerfile),
        "rm": True,
        "labels": {"faith.managed": "true"},
    }


def test_the_built_tag_is_never_the_authors_to_choose(tmp_path: Path) -> None:
    """The point of deriving it: no config value can steer what a build overwrites.

    Asserted against a config trying every plausible spelling of "name it this",
    since the guarantee is that no key does it — not that these three do not.
    """
    dockerfile = _dockerfile(tmp_path)
    env = DockerEnvironment(
        {
            "agent_dockerfile": str(dockerfile),
            "agent_tag": "busybox:1.36",
            "image": "busybox:1.36",
            "tag": "busybox:1.36",
        },
        client=(client := _FakeDockerClient(_FakeContainer())),
    )
    env.setup(_task())
    assert client.images.build_kwargs is not None
    assert client.images.build_kwargs["tag"] == derived_image_tag(dockerfile)


def test_an_unchanged_benchmark_does_not_rebuild(tmp_path: Path) -> None:
    """The reuse path: a second run on the same recipe finds its tag present."""
    dockerfile = _dockerfile(tmp_path)
    client = _FakeDockerClient(
        _FakeContainer(), present={derived_image_tag(dockerfile)}
    )
    DockerEnvironment({"agent_dockerfile": str(dockerfile)}, client=client).setup(
        _task()
    )
    assert (client.images.build_kwargs, client.images.pulled) == (None, [])


def test_switching_recipes_builds_each_once_and_reuses_the_right_image(
    tmp_path: Path,
) -> None:
    """Both recipes exist throughout; selection alone must invalidate the cache."""
    first = _dockerfile(tmp_path)
    second = tmp_path / "Dockerfile.other"
    second.write_text("FROM ubuntu:22.04\n", encoding="utf-8")
    client = _FakeDockerClient(_FakeContainer())

    for recipe in (first, second):
        client.images.build_kwargs = None
        env = DockerEnvironment({"agent_dockerfile": str(recipe)}, client=client)
        env.setup(_task())
        assert client.images.build_kwargs is not None
        assert client.images.build_kwargs["dockerfile"] == recipe.name
        assert client.images.build_kwargs["tag"] == derived_image_tag(recipe)
        assert client.run_kwargs is not None
        assert client.run_kwargs["image"] == derived_image_tag(recipe)
        env.teardown()

    client.images.build_kwargs = None
    env = DockerEnvironment({"agent_dockerfile": str(first)}, client=client)
    env.setup(_task())
    assert client.images.build_kwargs is None
    assert client.run_kwargs is not None
    assert client.run_kwargs["image"] == derived_image_tag(first)
    assert len(client.images.present) == 2
    assert not client.images.pulled
    env.teardown()


def test_an_edited_dockerfile_rebuilds_and_a_reverted_one_reuses(
    tmp_path: Path,
) -> None:
    """The bug this design exists to remove, and the cache win that comes with it.

    A single mutable tag is not an identity: with one, editing the recipe leaves the
    old image sitting under the tag the run asks for, so every host holding it keeps
    running the old image silently — two people on the same commit measure different
    things. Deriving the tag makes the edit ask for an image that has never existed.
    Reverting then asks for the first one again, which is still present.
    """
    dockerfile = _dockerfile(tmp_path)
    original_tag = derived_image_tag(dockerfile)
    client = _FakeDockerClient(_FakeContainer(), present={original_tag})

    dockerfile.write_text("FROM ubuntu:22.04\nRUN echo edited\n")
    DockerEnvironment({"agent_dockerfile": str(dockerfile)}, client=client).setup(
        _task()
    )
    assert client.images.build_kwargs is not None
    edited_tag = client.images.build_kwargs["tag"]

    dockerfile.write_text("FROM ubuntu:24.04\n")
    client.images.build_kwargs = None
    DockerEnvironment({"agent_dockerfile": str(dockerfile)}, client=client).setup(
        _task()
    )

    assert (edited_tag != original_tag, client.images.build_kwargs) == (True, None)


def test_an_edited_context_file_rebuilds_though_the_dockerfile_is_untouched(
    tmp_path: Path,
) -> None:
    """A COPY'd script is part of the image, so the digest must cover the context."""
    dockerfile = _dockerfile(tmp_path)
    script = tmp_path / "setup.sh"
    script.write_text("echo one\n")
    client = _FakeDockerClient(
        _FakeContainer(), present={derived_image_tag(dockerfile)}
    )

    script.write_text("echo two\n")
    DockerEnvironment({"agent_dockerfile": str(dockerfile)}, client=client).setup(
        _task()
    )

    assert client.images.build_kwargs is not None


def test_a_task_image_override_is_pulled_rather_than_built(tmp_path: Path) -> None:
    """A task naming an image is naming one to pull, as the config key of that name is.

    Otherwise the benchmark's Dockerfile would be built under a tag the task chose —
    the fabrication the build/pull split exists to prevent, arriving by another door.
    """
    env = DockerEnvironment(
        {"agent_dockerfile": str(_dockerfile(tmp_path))},
        client=(client := _FakeDockerClient(_FakeContainer())),
    )
    env.setup(_task(environment_config={"agent_image": "other:v2"}))
    assert (client.images.build_kwargs, client.images.pulled) == (None, ["other:v2"])


@pytest.mark.parametrize(
    ("task_config", "expected"),
    [({}, "deployment:v2"), ({"agent_image": "task:v3"}, "task:v3")],
    ids=["no_task_override", "task_override"],
)
def test_a_task_image_outranks_a_deployment_override(
    task_config: dict[str, Any], expected: str
) -> None:
    """All three config layers at once, in order: benchmark < deployment < task.

    The deployment layer arrives already merged (`--environment-config`), so only
    the task layer is still live by the time the container starts.
    """
    resolved = resolve_environment(
        AgenticEnvironmentConfig(type="docker", config={"agent_image": "benchmark:v1"}),
        AgenticEnvironmentConfig(
            type="docker", config={"agent_image": "deployment:v2"}
        ),
    )
    client = _FakeDockerClient(_FakeContainer())
    DockerEnvironment(resolved.config, client=client).setup(
        _task(environment_config=task_config)
    )
    assert client.run_kwargs is not None
    assert client.run_kwargs["image"] == expected


def test_an_image_with_no_dockerfile_is_pulled_rather_than_built() -> None:
    """Nothing to build it from means it was produced elsewhere: fetch it."""
    env, client, _ = _make_env()
    env.setup(_task())
    assert (client.images.pulled, client.images.build_kwargs) == (["img:latest"], None)
    assert client.run_kwargs is not None


@pytest.mark.parametrize(
    "image",
    [
        "ubuntu:24.04",
        "my-local-build:v2",
        "registry.example/team/vulnerability-localization:v2",
    ],
    ids=["public_base_image", "unqualified_local_tag", "registry_host"],
)
def test_a_deployment_image_is_never_built_from_the_benchmarks_dockerfile(
    tmp_path: Path, image: str
) -> None:
    """A deployment naming an image is naming one to pull, whatever the benchmark ships.

    The merged config holds both keys — the benchmark's Dockerfile and the
    deployment's image — and that combination reads as "pull this instead", with no
    key-specific rule in `resolve_environment` to arrange it. Building here would
    substitute a local build for the image the deployment asked to fetch, and the
    image's *name* cannot decide it: `ubuntu:24.04` is as pullable as a registry path.
    """
    resolved = resolve_environment(
        AgenticEnvironmentConfig(
            type="docker",
            config={"agent_dockerfile": str(_dockerfile(tmp_path))},
        ),
        AgenticEnvironmentConfig(type="docker", config={"agent_image": image}),
    )
    client = _FakeDockerClient(_FakeContainer())
    DockerEnvironment(resolved.config, client=client).setup(_task())
    assert (client.images.build_kwargs, client.images.pulled) == (None, [image])


def test_a_pulling_deployment_leaves_the_benchmarks_dockerfile_in_the_merged_config(
    tmp_path: Path,
) -> None:
    """Pull-wins is the environment's reading of the pair, not a deleted key.

    Pinned because the previous design deleted `agent_dockerfile` during the merge —
    a key-specific exception to a generic merge. Precedence belongs where the keys
    are interpreted, so the merge stays a merge.
    """
    dockerfile = str(_dockerfile(tmp_path))
    resolved = resolve_environment(
        AgenticEnvironmentConfig(
            type="docker", config={"agent_dockerfile": dockerfile}
        ),
        AgenticEnvironmentConfig(type="docker", config={"agent_image": "pulled:v2"}),
    )

    assert resolved.config == {
        "agent_dockerfile": dockerfile,
        "agent_image": "pulled:v2",
    }


def test_a_build_is_logged_with_the_recipe_it_came_from(tmp_path: Path) -> None:
    """One INFO line, since a derived tag says nothing about which Dockerfile made it.

    A build no longer needs a warning: it can only produce a tag in FAITH's own
    namespace, so it cannot overwrite anything upstream published. What is worth
    recording is which recipe a run's image came from.
    """
    dockerfile = _dockerfile(tmp_path)
    env = DockerEnvironment(
        {"agent_dockerfile": str(dockerfile)},
        client=_FakeDockerClient(_FakeContainer()),
    )
    with patch("faith.agentic.environments.docker.logger") as mock_logger:
        env.setup(_task())
    mock_logger.info.assert_called_once_with(
        "Image '%s' not found locally; building it from %s.",
        derived_image_tag(dockerfile),
        dockerfile,
    )


def test_a_present_image_is_neither_pulled_nor_built(tmp_path: Path) -> None:
    """A cached tag is used as-is, so a run needs no network."""
    client = _FakeDockerClient(
        _FakeContainer(), present={"registry.example/vulnerability-localization:v2"}
    )
    env = DockerEnvironment(
        {
            "agent_image": "registry.example/vulnerability-localization:v2",
            "agent_dockerfile": str(_dockerfile(tmp_path)),
        },
        client=client,
    )
    env.setup(_task())
    assert (client.images.pulled, client.images.build_kwargs) == ([], None)


def test_a_failed_pull_is_reported_rather_than_built_around(tmp_path: Path) -> None:
    """The deployment named an image it expects to exist; a miss must not fall back.

    Given a Dockerfile to fall back *to*, since without one there is nothing the
    fallback this forbids could have built. Typed as an environment failure: a
    registry the host cannot reach is not a broken run.
    """
    client = _FakeDockerClient(_FakeContainer())
    client.images.pull_error = RuntimeError("manifest unknown")
    env = DockerEnvironment(
        {
            "agent_image": "registry.example/vulnerability-localization:v2",
            "agent_dockerfile": str(_dockerfile(tmp_path)),
        },
        client=client,
    )
    with pytest.raises(EnvironmentUnavailableError) as excinfo:
        env.setup(_task())
    assert str(excinfo.value) == (
        "Failed to pull 'registry.example/vulnerability-localization:v2': "
        "manifest unknown"
    )
    assert client.images.build_kwargs is None


def test_a_failed_build_is_reported_with_the_dockerfile(tmp_path: Path) -> None:
    """Also an environment failure, so a broken recipe is not logged as a broken run."""
    dockerfile = _dockerfile(tmp_path)
    client = _FakeDockerClient(_FakeContainer())
    client.images.build_error = RuntimeError("bad instruction")
    env = DockerEnvironment({"agent_dockerfile": str(dockerfile)}, client=client)
    with pytest.raises(EnvironmentUnavailableError) as excinfo:
        env.setup(_task())
    assert str(excinfo.value) == (
        f"Failed to build '{derived_image_tag(dockerfile)}' from {dockerfile}: "
        "bad instruction"
    )


def test_a_missing_dockerfile_fails_at_construction(tmp_path: Path) -> None:
    """A typo'd path must not surface as an opaque build error mid-run."""
    missing = tmp_path / "nope" / "Dockerfile"
    with pytest.raises(ValueError, match=f"agent_dockerfile does not exist: {missing}"):
        DockerEnvironment(
            {"agent_image": "img:latest", "agent_dockerfile": str(missing)},
            client=_FakeDockerClient(_FakeContainer()),
        )


def test_a_non_string_dockerfile_is_rejected() -> None:
    with pytest.raises(ValueError, match="'agent_dockerfile' must be a path"):
        DockerEnvironment(
            {"agent_image": "img:latest", "agent_dockerfile": 7},
            client=_FakeDockerClient(_FakeContainer()),
        )


def test_a_benchmarks_root_dockerfile_path_is_expanded(tmp_path: Path) -> None:
    """Config authors write `$BENCHMARKS_ROOT/` here as they do for `!from`."""
    dockerfile = _dockerfile(tmp_path)
    client = _FakeDockerClient(_FakeContainer())
    with patch("faith._internal.io.resources.benchmarks_root", return_value=tmp_path):
        DockerEnvironment(
            {"agent_dockerfile": "$BENCHMARKS_ROOT/Dockerfile"}, client=client
        ).setup(_task())
    assert client.images.build_kwargs is not None
    assert client.images.build_kwargs["path"] == str(dockerfile.parent)
