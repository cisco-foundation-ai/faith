# Copyright 2025 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for KubernetesEnvironment.

A fake SandboxClient is injected so these run without a cluster: they assert on
the sandbox lifecycle / command / file calls the environment makes. The real
``k8s-agent-sandbox`` package need not be installed. The archive shapes are
pinned in test_archive.py and the path guard in test_workspace.py, so an upload
is checked here only for reaching ``files.write`` with the right payload and
being extracted by the right command.
"""

import os
import re
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock, patch

import pytest

from faith.agentic.core.environment import EnvironmentUnavailableError
from faith.agentic.core.task import Task
from faith.agentic.environments._archive import tar_directory
from faith.agentic.environments.kubernetes import KubernetesEnvironment

# Setup's runtime probe stages a file under this prefix and reads it back through
# a shell. The doubles route both aside from the recordings the other tests assert
# on, so an upload test still names only the upload it is about.
_PROBE_PREFIX = ".faith-probe-"

# A `cat` of a workspace-root file, which is how the probe reads its marker back.
_CAT = re.compile(r"cat \./(?P<name>[^']+)")


class _FakeFiles:
    """Files double recording writes, and reads canned per path.

    Written content is also kept, so `_FakeCommands` can serve a later `cat` of it
    the way a runtime rooted at the same workspace would.
    """

    def __init__(
        self,
        reads: dict[str, bytes | str] | None = None,
        write_error: Exception | None = None,
    ):
        self._reads = reads or {}
        self.write_error = write_error
        self.writes: list[tuple[str, object]] = []
        self.probe_writes: list[tuple[str, object]] = []
        self.content: dict[str, object] = {}

    def write(self, path: str, content: object) -> None:
        """Record the requested path and content, raising if scripted to."""
        if self.write_error is not None:
            raise self.write_error
        self.content[path] = content
        if path.startswith(_PROBE_PREFIX):
            self.probe_writes.append((path, content))
        else:
            self.writes.append((path, content))

    def read(self, path: str) -> bytes | str:
        """Return the canned content for this path, empty if unscripted."""
        return self._reads.get(path, b"")


class _FakeCommands:
    """Commands double returning a result canned by the command's first word.

    The sandbox's shell starts outside the workspace, so every command FAITH sends
    is prefixed with a `cd`. It is stripped before the result lookup — keying on
    the literal first word would script every command as `cd` — while `runs`
    records what was sent, so a test still pins the prefix.

    Conforming by default: a `cat` is served from what `files.write` recorded,
    which is the behavior setup's probe checks for. `probe_result` scripts a
    runtime that does not conform, since a non-shell or wrongly-rooted runtime
    cannot be expressed as a canned result keyed on a word it never parsed.
    """

    def __init__(
        self,
        results: dict[str, tuple[int, str | None, str | None]] | None = None,
        files: _FakeFiles | None = None,
        probe_result: tuple[int, str | None, str | None] | None = None,
        run_error: Exception | None = None,
    ):
        self._results = results or {}
        self._files = files or _FakeFiles()
        self._probe_result = probe_result
        self.run_error = run_error
        self.runs: list[str] = []
        self.probe_runs: list[str] = []
        self.timeouts: list[int] = []

    def run(self, command: str, timeout: int) -> SimpleNamespace:
        """Record the command and its SDK timeout, returning the scripted result."""
        if self.run_error is not None:
            raise self.run_error
        if _PROBE_PREFIX in command:
            self.probe_runs.append(command)
            return self._probe_response(command)
        self.runs.append(command)
        self.timeouts.append(timeout)
        _, _, executed = command.partition(" && ")
        exit_code, out, err = self._results.get(executed.split()[0], (0, "", ""))
        return SimpleNamespace(exit_code=exit_code, stdout=out, stderr=err)

    def _probe_response(self, command: str) -> SimpleNamespace:
        """The probe's result: scripted if given, else the file read back."""
        if self._probe_result is not None:
            exit_code, out, err = self._probe_result
            return SimpleNamespace(exit_code=exit_code, stdout=out, stderr=err)
        match = _CAT.search(command)
        if match is None:
            return SimpleNamespace(exit_code=0, stdout="", stderr="")
        content = self._files.content.get(match["name"], b"")
        text = content.decode() if isinstance(content, bytes) else str(content)
        return SimpleNamespace(exit_code=0, stdout=text, stderr="")


class _FakeSandbox:
    """Sandbox double: recorded file/command traffic and terminate.

    No status sequence: `create_sandbox` only returns a handle once the sandbox
    is ready, so a sandbox this fake stands in for is ready by construction.
    """

    def __init__(
        self,
        reads: dict[str, bytes | str] | None = None,
        cmd_results: dict[str, tuple[int, str | None, str | None]] | None = None,
        terminate_error: Exception | None = None,
        probe_result: tuple[int, str | None, str | None] | None = None,
        run_error: Exception | None = None,
        write_error: Exception | None = None,
    ):
        self.files = _FakeFiles(reads, write_error)
        self.commands = _FakeCommands(cmd_results, self.files, probe_result, run_error)
        self.terminate_attempts = 0
        self.terminate_error = terminate_error

    def terminate(self) -> None:
        """Count the termination, raising if scripted to."""
        self.terminate_attempts += 1
        if self.terminate_error is not None:
            raise self.terminate_error


class _FakeSandboxClient:
    """Fake ``SandboxClient`` handing back one pre-built sandbox."""

    def __init__(self, sandbox: _FakeSandbox, create_error: Exception | None = None):
        self._sandbox = sandbox
        self._create_error = create_error
        self.create_kwargs: dict[str, Any] | None = None

    def create_sandbox(self, **kwargs: Any) -> _FakeSandbox:
        """Record the create kwargs and hand back the fake sandbox."""
        self.create_kwargs = kwargs
        if self._create_error is not None:
            raise self._create_error
        return self._sandbox


def _in_workspace(command: str, workspace: str = "/workspace") -> str:
    """A command as the sandbox receives it: run from the workspace root.

    Spelled out once so the cases below still read as the command under test, and
    a test that cares about the prefix itself names it literally.
    """
    return f"cd {workspace} && {command}"


def _task(data_path: Path | None = None) -> Task:
    return Task(id="t1", prompt="p", ground_truth=None, data_path=data_path)


def _make_env(
    sandbox: _FakeSandbox | None = None,
    create_error: Exception | None = None,
    **config: Any,
) -> tuple[KubernetesEnvironment, _FakeSandboxClient, _FakeSandbox]:
    sandbox = sandbox or _FakeSandbox()
    client = _FakeSandboxClient(sandbox, create_error)
    env = KubernetesEnvironment({"warmpool": "pool-a", **config}, client=client)
    return env, client, sandbox


def _setup_env(
    sandbox: _FakeSandbox | None = None, **config: Any
) -> tuple[KubernetesEnvironment, _FakeSandbox]:
    env, _, sandbox = _make_env(sandbox, **config)
    env.setup(_task())
    sandbox.files.writes.clear()
    sandbox.commands.runs.clear()
    return env, sandbox


@pytest.mark.parametrize(
    "config, expected_namespace",
    [({}, "default"), ({"namespace": "pool-ns"}, "pool-ns")],
    ids=["default-namespace", "explicit-namespace"],
)
def test_setup_claims_a_sandbox_from_the_configured_warm_pool(
    config: dict[str, Any], expected_namespace: str
) -> None:
    """The claim names the configured warm pool, in the SDK's own default namespace.

    Defaulting to any other namespace would be FAITH asserting where a cluster
    installed the sandbox controller, which it cannot know.
    """
    env, client, _ = _make_env(**config)
    env.setup(_task())
    assert client.create_kwargs == {
        "warmpool": "pool-a",
        "namespace": expected_namespace,
        "sandbox_ready_timeout": 120,
        "shutdown_after_seconds": None,
    }


@pytest.mark.parametrize(
    "config",
    [{}, {"warmpool": None}, {"warmpool": ""}],
    ids=["absent", "null", "empty"],
)
def test_a_missing_warm_pool_is_rejected(config: dict[str, Any]) -> None:
    """The pool names a cluster's own resource, so there is nothing to default to.

    A key present but blank is the yaml shape of "unset", so it must fail the
    same way an absent one does rather than claiming with an empty name.
    """
    with pytest.raises(ValueError) as excinfo:
        KubernetesEnvironment(config, client=_FakeSandboxClient(_FakeSandbox()))
    assert str(excinfo.value) == "KubernetesEnvironment config requires 'warmpool'"


def test_a_claim_that_cannot_be_served_is_an_environment_failure() -> None:
    """A cluster that cannot stamp a pod is not a broken run, so the log says so.

    The loop classifies on this type alone, and only ENVIRONMENT_ERROR points at
    the cluster rather than at FAITH.
    """
    env, _, _ = _make_env(create_error=RuntimeError("no capacity"))

    with pytest.raises(EnvironmentUnavailableError) as excinfo:
        env.setup(_task())

    assert str(excinfo.value) == "Failed to claim a sandbox: no capacity"


def test_a_sandbox_dying_during_setup_stays_an_environment_failure() -> None:
    """The probe's own wrap must not downgrade a sandbox that died under it."""
    env, _, _ = _make_env(_FakeSandbox(run_error=RuntimeError("connection refused")))

    with pytest.raises(EnvironmentUnavailableError) as excinfo:
        env.setup(_task())

    assert str(excinfo.value) == "Sandbox unavailable: connection refused"


@pytest.mark.parametrize(
    "error",
    [RuntimeError("connection refused"), TimeoutError("tunnel not established")],
    ids=["sdk-exception", "bare-timeout"],
)
def test_a_sandbox_unreachable_through_the_file_api_is_an_environment_failure(
    error: Exception,
) -> None:
    """The file API crosses the same tunnel as the command API, and is reached first."""
    env, _, _ = _make_env(_FakeSandbox(write_error=error))

    with pytest.raises(EnvironmentUnavailableError) as excinfo:
        env.setup(_task())

    assert str(excinfo.value) == f"Sandbox unavailable: {error}"


@pytest.mark.parametrize(
    "error",
    [RuntimeError("connection refused"), TimeoutError("tunnel not re-established")],
    ids=["sdk-exception", "bare-timeout"],
)
def test_a_sandbox_lost_mid_task_is_an_environment_failure_whatever_it_raised(
    error: Exception,
) -> None:
    """Only this type gets the task a fresh sandbox instead of a recorded error."""
    env, sandbox = _setup_env()
    sandbox.commands.run_error = error

    with pytest.raises(EnvironmentUnavailableError) as excinfo:
        env.run_command("ls")

    assert str(excinfo.value) == f"Sandbox unavailable: {error}"


def test_setup_probes_the_runtime_through_the_file_and_command_apis() -> None:
    """Both assumptions are checked on the agent's own path, before any task work.

    The pod is claimed from a warm pool FAITH does not own, so that a command string
    reaches a shell and that `files.write` roots at this workspace are requirements
    rather than guarantees. One probe covers both: the marker only comes back if the
    write landed where the read looks and the read was evaluated by a shell.
    """
    env, _, sandbox = _make_env()
    with patch(
        "faith.agentic.environments.kubernetes._probe_file_name",
        return_value=".faith-probe-1-0",
    ):
        env.setup(_task())
    assert sandbox.files.probe_writes == [(".faith-probe-1-0", b"faith-runtime-ok")]
    assert sandbox.commands.probe_runs == [
        _in_workspace("timeout -k 5 60 bash -c 'cat ./.faith-probe-1-0'"),
        _in_workspace("rm -f ./.faith-probe-1-0"),
    ]


@pytest.mark.parametrize(
    "probe_result",
    [
        # A runtime that execs its argument directly cannot run the `cd` prefix,
        # which is how upstream's example runtime fails.
        (127, "", "cd: no such file or directory"),
        # One rooted somewhere other than the configured workspace runs the read
        # but finds nothing at the path `files.write` reported writing.
        (1, "", "cat: ./.faith-probe-1-0: No such file or directory"),
        # One without coreutils `timeout`, or without `bash`, cannot run the wrapper.
        (127, "", "timeout: command not found"),
        # One that answers but does not round-trip content is equally unusable.
        (0, "something else", ""),
    ],
    ids=["no_shell", "wrong_root", "no_timeout", "wrong_content"],
)
def test_setup_rejects_a_runtime_that_fails_the_probe(
    probe_result: tuple[int, str | None, str | None],
) -> None:
    """One clear error at setup, rather than every command of every task failing.

    A non-conforming runtime otherwise fails ~50 commands per task with output that
    reads as the agent's own error, so the run completes and scores zero for a
    reason nothing in the results names. The message names the contract and the
    workspace it was checked against, since the fix is on the cluster side.
    """
    env, _, _ = _make_env(_FakeSandbox(probe_result=probe_result))
    with pytest.raises(RuntimeError) as excinfo:
        env.setup(_task())
    assert str(excinfo.value).startswith(
        "Failed to set up sandbox: sandbox runtime does not meet FAITH's contract: "
        "reading back a file written to /workspace through the sandbox file API "
        "exited "
    )
    assert str(excinfo.value).endswith(
        "The runtime must evaluate commands with a shell (bash and coreutils "
        "timeout on PATH) and root its file API at /workspace."
    )


def test_the_probe_is_removed_even_when_it_fails() -> None:
    """A probe file left at the workspace root would be visible to the agent."""
    sandbox = _FakeSandbox(probe_result=(1, "", "nope"))
    env, _, _ = _make_env(sandbox)
    with patch(
        "faith.agentic.environments.kubernetes._probe_file_name",
        return_value=".faith-probe-1-0",
    ):
        with pytest.raises(RuntimeError):
            env.setup(_task())
    assert sandbox.commands.probe_runs[-1] == _in_workspace("rm -f ./.faith-probe-1-0")


def test_the_probe_is_checked_against_the_configured_workspace() -> None:
    """The contract is "root at *this* workspace", not at a fixed path.

    A configured `mount_path` moves both the write and the read, so a runtime is
    checked against the root this environment was told to use.
    """
    env, _, sandbox = _make_env(mount_path="/app")
    with patch(
        "faith.agentic.environments.kubernetes._probe_file_name",
        return_value=".faith-probe-1-0",
    ):
        env.setup(_task())
    assert sandbox.commands.probe_runs == [
        _in_workspace("timeout -k 5 60 bash -c 'cat ./.faith-probe-1-0'", "/app"),
        _in_workspace("rm -f ./.faith-probe-1-0", "/app"),
    ]


def test_the_probe_reads_from_the_root_it_wrote_to_under_a_working_subdir() -> None:
    """`files.write` writes by basename, so the probe file is at the root, not the subdir.

    A probe that followed the agent into `working_dir` would name a file one level
    up from where it looks, and would reject a conforming runtime for every
    benchmark that sets a `working_subdir` — the opposite of the check's purpose.
    """
    env, _, sandbox = _make_env(working_subdir="repo")
    with patch(
        "faith.agentic.environments.kubernetes._probe_file_name",
        return_value=".faith-probe-1-0",
    ):
        env.setup(_task())  # Must not raise: the fake runtime conforms.
    assert sandbox.commands.probe_runs == [
        _in_workspace("timeout -k 5 60 bash -c 'cat ./.faith-probe-1-0'"),
        _in_workspace("rm -f ./.faith-probe-1-0"),
    ]


def test_probe_names_are_unique_per_process_and_per_setup() -> None:
    """A fixed name would collide with a parallel run sharing the workspace root."""
    env, _, sandbox = _make_env()
    env.setup(_task())
    env.setup(_task())
    probed = [name for name, _ in sandbox.files.probe_writes]
    assert len(set(probed)) == 2
    assert all(name.startswith(f".faith-probe-{os.getpid()}-") for name in probed)


@pytest.mark.parametrize(
    ("config", "expected_workspace"),
    [({}, "/workspace"), ({"mount_path": "/app"}, "/app")],
    ids=["default", "configured_mount_path"],
)
def test_every_command_runs_from_the_workspace(
    config: dict[str, Any], expected_workspace: str
) -> None:
    """The sandbox API takes no working directory, so the command has to carry it.

    Its shell starts at `/` while `files.write` resolves against the workspace, so
    without this an uploaded archive is written to the workspace and then not found
    by the extraction command naming it — no task data is unpacked at all, and the
    staging tar is left where the agent sees it. DockerEnvironment gets the same
    thing from `exec_run`'s `workdir`.
    """
    env, sandbox = _setup_env(**config)
    env.run_command("ls")
    assert sandbox.commands.runs == [
        _in_workspace("timeout -k 5 60 bash -c ls", expected_workspace)
    ]


def test_a_workspace_needing_quoting_is_quoted() -> None:
    """The workspace is config, so it reaches the shell as one word regardless."""
    env, sandbox = _setup_env(mount_path="/two words")
    env.run_command("ls")
    assert sandbox.commands.runs == ["cd '/two words' && timeout -k 5 60 bash -c ls"]


def test_a_working_subdir_moves_the_agents_commands_but_not_setup(
    tmp_path: Path,
) -> None:
    """Setup keeps the root: the subdir only exists once the archive is extracted.

    Both go through the same `_run`, so the two `cd` targets are asserted together —
    an extraction that followed the agent into the subdir would look for an archive
    `files.write` put one level up.
    """
    zip_path = tmp_path / "repo.zip"
    zip_path.write_bytes(b"PKzipbytes")
    env, _, sandbox = _make_env(working_subdir="repo")
    env.setup(_task(data_path=zip_path))
    env.run_command("ls")
    assert sandbox.commands.runs == [
        _in_workspace("unzip -o ./repo.zip; status=$?; rm -f ./repo.zip; exit $status"),
        _in_workspace("timeout -k 5 60 bash -c ls", "/workspace/repo"),
    ]


def test_setup_uploads_a_zip_file_and_unzips_it_in_place(tmp_path: Path) -> None:
    """The zip must not be left where the agent would see it as task data.

    The command is unwrapped: extracting a multi-GB snapshot is our own setup work
    and must not compete with an agent-sized bound.
    """
    zip_path = tmp_path / "repo.zip"
    zip_path.write_bytes(b"PKzipbytes")
    env, _, sandbox = _make_env()
    env.setup(_task(data_path=zip_path))
    assert sandbox.files.writes == [("repo.zip", b"PKzipbytes")]
    assert sandbox.commands.runs == [
        _in_workspace("unzip -o ./repo.zip; status=$?; rm -f ./repo.zip; exit $status")
    ]


def test_setup_uploads_a_directory_as_one_staged_tar(tmp_path: Path) -> None:
    """One tar, since files.write writes by basename and would flatten the layout.

    The removal is unconditional and inside the same command, so a tar left by a
    failed extract is not visible to the agent as task data either.
    """
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "a.c").write_bytes(b"aaa")
    env, _, sandbox = _make_env()
    with patch(
        "faith.agentic.environments.kubernetes._staging_archive_name",
        return_value=".faith-upload-1-0.tar",
    ):
        env.setup(_task(data_path=tmp_path))
    assert sandbox.files.writes == [(".faith-upload-1-0.tar", tar_directory(tmp_path))]
    assert sandbox.commands.runs == [
        _in_workspace(
            "tar -xf ./.faith-upload-1-0.tar; status=$?; "
            "rm -f ./.faith-upload-1-0.tar; exit $status"
        )
    ]


@pytest.mark.parametrize(
    ("data_name", "content", "expected_message"),
    [
        (
            "repo.zip",
            b"corrupt",
            "Failed to set up sandbox: Failed to extract repo.zip: bad zip",
        ),
        (
            "adir",
            None,
            "Failed to set up sandbox: Failed to extract .faith-upload-1-0.tar: "
            "bad tar",
        ),
    ],
    ids=["zip", "directory"],
)
def test_setup_reports_a_failed_extraction(
    tmp_path: Path, data_name: str, content: bytes | None, expected_message: str
) -> None:
    """Wrapped in the setup message, so a run log says which stage failed."""
    data = tmp_path / data_name
    if content is None:
        data.mkdir()
        (data / "a.c").write_bytes(b"aaa")
    else:
        data.write_bytes(content)
    sandbox = _FakeSandbox(
        cmd_results={"unzip": (2, "", " bad zip "), "tar": (2, "", " bad tar ")}
    )
    env, _, _ = _make_env(sandbox)
    with patch(
        "faith.agentic.environments.kubernetes._staging_archive_name",
        return_value=".faith-upload-1-0.tar",
    ):
        with pytest.raises(RuntimeError) as excinfo:
            env.setup(_task(data_path=data))
    assert str(excinfo.value) == expected_message


def test_staging_archive_names_are_unique_per_process_and_per_upload(
    tmp_path: Path,
) -> None:
    """A fixed name would let a task file collide with it.

    Two runs in parallel processes share the workspace-root namespace, so the pid
    is part of the name as well as a per-operation counter.
    """
    (tmp_path / "a.c").write_bytes(b"int main(){}")
    env, _, sandbox = _make_env()
    env.setup(_task(data_path=tmp_path))
    env.setup(_task(data_path=tmp_path))
    staged = [name for name, _ in sandbox.files.writes]
    assert len(set(staged)) == 2
    assert all(
        name.startswith(f".faith-upload-{os.getpid()}-") and name.endswith(".tar")
        for name in staged
    )


@pytest.mark.parametrize(
    ("result", "expected"),
    [
        ((0, "out", "warn"), (0, "outwarn")),
        # Raw exit code: annotating a failure is the bash_exec tool's job.
        ((126, "", "denied"), (126, "denied")),
        # The SDK reports "no output" as None on either stream.
        ((0, None, None), (0, "")),
        # 124 is `timeout`'s code for a command it killed; the notice is appended,
        # so output printed before the kill survives.
        (
            (124, "partial", None),
            (124, "partial\nERROR: Command timed out after 60s"),
        ),
        ((125, "other", None), (125, "other")),
    ],
    ids=["success", "nonzero_exit", "no_output", "timed_out", "other_failure"],
)
def test_run_command_bounds_the_command_and_joins_both_streams(
    result: tuple[int, str | None, str | None], expected: tuple[int, str]
) -> None:
    """The agent's command is bounded exactly as on Docker.

    An unbounded backend would score the same benchmark differently, so the
    wrapper is not Docker-specific. The SDK's own timeout is still passed as a
    backstop for a wedged sandbox, derived from the same value.
    """
    env, sandbox = _setup_env(_FakeSandbox(cmd_results={"timeout": result}))
    assert env.run_command("ls -l") == expected
    assert sandbox.commands.runs == [_in_workspace("timeout -k 5 60 bash -c 'ls -l'")]
    assert sandbox.commands.timeouts == [600]


@pytest.mark.parametrize(
    ("config", "expected_command", "expected_sdk_timeout"),
    [
        # A single shell word needs no quoting, and shlex.quote adds none.
        ({}, "timeout -k 5 60 bash -c ls", 600),
        ({"command_timeout_seconds": 10}, "timeout -k 5 10 bash -c ls", 600),
        # 0 means "no bound" to coreutils `timeout`, so it must survive as 0.
        ({"command_timeout_seconds": 0}, "timeout -k 5 0 bash -c ls", 600),
        ({"command_timeout_seconds": 900}, "timeout -k 5 900 bash -c ls", 960),
    ],
    ids=["default", "configured", "zero_is_kept", "over_the_sdk_floor"],
)
def test_the_command_bound_comes_from_the_config(
    config: dict[str, Any], expected_command: str, expected_sdk_timeout: int
) -> None:
    """One key sets the command's bound; the SDK's is derived from it."""
    env, sandbox = _setup_env(**config)
    env.run_command("ls")
    assert sandbox.commands.runs == [_in_workspace(expected_command)]
    assert sandbox.commands.timeouts == [expected_sdk_timeout]


def test_a_primitive_before_setup_raises() -> None:
    env, _, _ = _make_env()
    with pytest.raises(RuntimeError) as excinfo:
        env.run_command("ls")
    assert str(excinfo.value) == "No sandbox running. Call setup() first."


@pytest.mark.parametrize(
    "terminate_error", [None, RuntimeError("gone")], ids=["terminates", "already_gone"]
)
def test_teardown_terminates_once_and_never_raises(
    terminate_error: Exception | None,
) -> None:
    """A sandbox that is already gone still leaves the environment torn down."""
    env, sandbox = _setup_env(_FakeSandbox(terminate_error=terminate_error))
    env.teardown()
    env.teardown()  # The first attempt cleared the sandbox, failed or not.
    assert sandbox.terminate_attempts == 1


@pytest.mark.parametrize(
    ("config", "expected_connection", "expected_kwargs"),
    [
        ({}, "in_cluster", {}),
        ({"connection": "in_cluster"}, "in_cluster", {}),
        (
            {"connection": "tunnel"},
            "tunnel",
            {"router_namespace": "agent-sandbox-system"},
        ),
        (
            {"connection": "tunnel", "router_namespace": "agent-sandbox"},
            "tunnel",
            {"router_namespace": "agent-sandbox"},
        ),
        # Every other config enum in FAITH accepts any casing.
        ({"connection": "In_Cluster"}, "in_cluster", {}),
    ],
    ids=[
        "default_is_in_cluster",
        "explicit_in_cluster",
        "tunnel",
        "tunnel_with_a_configured_router_namespace",
        "any_casing",
    ],
)
def test_the_client_is_built_once_on_first_setup_with_an_explicit_connection(
    config: dict[str, Any], expected_connection: str, expected_kwargs: dict[str, Any]
) -> None:
    """Construction builds no client, so importing needs no reachable cluster.

    The connection config is asserted because leaving it to the SDK is what breaks
    a run either way: its default shells out to `kubectl port-forward`, which a
    FAITH pod has no `kubectl` for, and its router namespace is a cluster's to say.
    """
    mock_client_class = Mock(return_value=_FakeSandboxClient(_FakeSandbox()))
    connections = {"in_cluster": Mock(return_value="cfg"), "tunnel": Mock()}
    with patch(
        "faith.agentic.environments.kubernetes._require_sdk",
        return_value=(
            mock_client_class,
            connections["in_cluster"],
            connections["tunnel"],
        ),
    ):
        connections[expected_connection].return_value = "cfg"
        env = KubernetesEnvironment({"warmpool": "pool-a", **config})
        mock_client_class.assert_not_called()
        # Recovery re-runs setup, which must reuse the client, not rebuild it.
        env.setup(_task())
        env.setup(_task())
    connections[expected_connection].assert_called_once_with(**expected_kwargs)
    mock_client_class.assert_called_once_with(connection_config="cfg", cleanup=True)


def test_an_unknown_connection_mode_is_rejected_at_construction() -> None:
    """Named modes only, so a typo fails before a run rather than at the first exec."""
    with pytest.raises(ValueError) as excinfo:
        _make_env(connection="in-cluster")
    assert str(excinfo.value) == (
        "KubernetesEnvironment 'connection' is 'in-cluster'; "
        "must be one of in_cluster, tunnel."
    )


@pytest.mark.parametrize(
    ("configured", "expected_timeout"),
    [(30, 30), (45.9, 45)],
    ids=["configured", "float_truncates"],
)
def test_the_readiness_timeout_is_handed_to_the_sdk(
    configured: float, expected_timeout: int
) -> None:
    """`create_sandbox` owns the wait, so the configured bound has to reach it.

    Passed as an int because the SDK counts it down in whole seconds; a float
    would truncate there instead, silently. The default reaching it is covered by
    the claim test above.
    """
    env, client, _ = _make_env(ready_timeout_seconds=configured)
    env.setup(_task())

    assert client.create_kwargs == {
        "warmpool": "pool-a",
        "namespace": "default",
        "sandbox_ready_timeout": expected_timeout,
        "shutdown_after_seconds": None,
    }


@pytest.mark.parametrize("timeout", [0, -1], ids=["zero", "negative"])
def test_a_non_positive_readiness_timeout_is_rejected(timeout: int) -> None:
    """The SDK fails every claim on a non-positive bound, so reject it at parse."""
    with pytest.raises(ValueError) as excinfo:
        _make_env(ready_timeout_seconds=timeout)

    assert str(excinfo.value) == (
        f"KubernetesEnvironment 'ready_timeout_seconds' is {timeout}; "
        f"must be positive."
    )


def test_the_claim_ttl_is_handed_to_the_sdk() -> None:
    """Only the cluster can reclaim a claim FAITH was killed before terminating."""
    env, client, _ = _make_env(shutdown_after_seconds=1800)
    env.setup(_task())

    assert client.create_kwargs == {
        "warmpool": "pool-a",
        "namespace": "default",
        "sandbox_ready_timeout": 120,
        "shutdown_after_seconds": 1800,
    }


@pytest.mark.parametrize("ttl", [0, -1], ids=["zero", "negative"])
def test_a_non_positive_claim_ttl_is_rejected(ttl: int) -> None:
    """The SDK rejects one, but only once a claim is attempted."""
    with pytest.raises(ValueError) as excinfo:
        _make_env(shutdown_after_seconds=ttl)

    assert str(excinfo.value) == (
        f"KubernetesEnvironment 'shutdown_after_seconds' is {ttl}; must be positive."
    )
