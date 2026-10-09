# Copyright 2025 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

"""Kubernetes-sandbox execution environment for agentic evaluation.

Unlike DockerEnvironment, this offers no ``network`` setting: FAITH does not own
the pod spec, and k8s has no per-pod equivalent of ``network_mode: none``.
Network isolation is a property of the SandboxTemplate and the cluster's CNI, not
something FAITH enforces.
"""

import itertools
import os
import shlex
from pathlib import Path
from typing import Any

from faith._types.enums import CIStrEnum
from faith.agentic.core.environment import (
    DEFAULT_WORKSPACE_DIR,
    Environment,
    EnvironmentUnavailableError,
)
from faith.agentic.core.task import Task
from faith.agentic.environments._archive import (
    tar_directory,
    workspace_operand,
)
from faith.agentic.environments._command import (
    bounded_shell,
    command_timeout,
    sdk_timeout,
    timeout_notice,
)
from faith.agentic.environments._read_only import read_only_required, workspace_check
from faith.agentic.environments._workspace import WorkspaceLayout


def _require_sdk() -> tuple[Any, Any, Any]:
    """Import the Agent Sandbox SDK, naming the extra if it is not installed.

    Not at module scope so `registries`, which references this class, loads
    without the extra.
    """
    try:
        # pylint: disable-next=import-outside-toplevel
        from k8s_agent_sandbox import SandboxClient

        # pylint: disable-next=import-outside-toplevel
        from k8s_agent_sandbox.models import (
            SandboxInClusterConnectionConfig,
            SandboxLocalTunnelConnectionConfig,
        )
    except ImportError as e:
        raise ImportError(
            "The 'k8s-agent-sandbox' package is required for "
            "KubernetesEnvironment. Install it from your checkout with: "
            "uv pip install -e '.[kubernetes]'"
        ) from e
    return (
        SandboxClient,
        SandboxInClusterConnectionConfig,
        SandboxLocalTunnelConnectionConfig,
    )


# Mirrors the SDK's own default.
_DEFAULT_NAMESPACE = "default"

# Also the SDK's own default, and the one thing a tunnel needs to find the router.
_DEFAULT_ROUTER_NAMESPACE = "agent-sandbox-system"


class ConnectionMode(CIStrEnum):
    """How the client reaches the sandbox: the `connection` config key.

    Named rather than holding the SDK's classes, so a benchmark.yaml can be
    validated where the extra is not installed.
    """

    IN_CLUSTER = "in_cluster"
    TUNNEL = "tunnel"


# `files.write` writes by basename, so a file FAITH stages for itself lands in the
# workspace root alongside the agent's own writes: name it per process and per
# operation so it cannot collide. Shared by the upload tar and the setup probe.
_staging_counter = itertools.count()

_PROBE_MARKER = "faith-runtime-ok"


def _staging_archive_name() -> str:
    """A workspace-root tar name unique to this process and operation."""
    return f".faith-upload-{os.getpid()}-{next(_staging_counter)}.tar"


def _probe_file_name() -> str:
    """A workspace-root probe file name unique to this process and operation."""
    return f".faith-probe-{os.getpid()}-{next(_staging_counter)}"


class KubernetesEnvironment(WorkspaceLayout, Environment):
    """Runs agent tools inside a Kubernetes Agent Sandbox.

    Constructed from the ``environment.config`` sub-dict of a benchmark.yaml:

    - ``warmpool`` (required): SandboxWarmPool to claim from. Cluster-specific,
      so there is no default worth guessing. Not the SandboxTemplate the pool is
      stamped from -- a claim names the pool, which references the template.
    - ``namespace`` (default ``default``): namespace holding the warm pool. The
      SDK's own default.
    - ``connection`` (default ``in_cluster``): how the client reaches the sandbox.
      ``in_cluster`` uses cluster DNS, which is what a FAITH pod needs; ``tunnel``
      shells out to ``kubectl port-forward`` for a run driven from outside. The
      SDK's own default is ``tunnel``, which needs a ``kubectl`` no sandbox image
      carries.
    - ``router_namespace`` (default ``agent-sandbox-system``, the SDK's own):
      namespace holding the router service a ``tunnel`` port-forwards to. Ignored
      in ``in_cluster`` mode. A cluster whose router sits elsewhere fails every
      claim with ``NotFound`` until this names it.
    - ``mount_path`` (default ``/workspace``): the workspace root, which is both
      the upload destination and the root the file guard resolves against.
    - ``working_subdir`` (default ``""``): injected by `build_environment`, not
      authored here. Setup extraction still runs at the root, since the
      subdirectory only exists once the data is extracted.
    - ``ready_timeout_seconds`` (default 120): how long ``create_sandbox`` waits
      for the claimed sandbox to report ready, replacing the SDK's own 180.
    - ``command_timeout_seconds`` (default 60): injected by `build_environment`
      from the benchmark's `limits`, not authored here. A wall-clock bound on each
      command the agent runs, imposed in-sandbox by coreutils ``timeout``; setup
      extraction is exempt, and 0 means no bound. A runtime may cap execs lower
      than this and report the same exit code, in which case the notice the agent
      sees names this bound rather than the one that fired.
    - ``shutdown_after_seconds`` (optional): TTL after which the cluster deletes
      the claim itself, derived by `build_environment` and overridden by an
      authored value. Enforced cluster-side, so it is the only bound here that
      survives FAITH being killed without teardown. Absolute rather than idle: it
      deletes a working sandbox as readily as an abandoned one.
    - ``read_only_workspace`` (default false): injected from the benchmark. Requires
      the runtime's irreversible, restart-persistent FAITH workspace-seal contract.
      A missing contract or failed permission check is a setup error.

    Unlike DockerEnvironment, which starts the container it executes against, this
    claims a pod from a cluster-side warm pool FAITH does not own, so the sandbox's
    runtime must meet three requirements:

    - ``commands.run`` must evaluate its argument as a **shell** command. FAITH
      sends ``cd <dir> && timeout ... bash -c ...``, which a runtime that execs
      directly reads as a request for a binary named ``cd``. Upstream's
      ``examples/python-runtime-sandbox`` does exactly that.
    - ``files.write`` must resolve against the configured ``mount_path``, since an
      uploaded archive is named by a command that runs there.
    - The binaries FAITH runs on its own behalf -- ``bash``, coreutils ``timeout``,
      ``tar``, ``unzip``, ``rm`` -- must be executable by the user the runtime execs
      as, which is stronger than being on PATH: a runtime that enforces an
      executable allowlist has to name them.

    `setup` probes these rather than trusting them: a non-conforming runtime
    otherwise fails every command of every task with output that reads as the
    agent's own error.
    """

    def __init__(self, config: dict[str, Any], client: Any = None):
        if client is None:
            _require_sdk()
        self._warmpool = config.get("warmpool")
        if not self._warmpool:
            raise ValueError("KubernetesEnvironment config requires 'warmpool'")
        self._namespace = config.get("namespace") or _DEFAULT_NAMESPACE
        self._router_namespace = (
            config.get("router_namespace") or _DEFAULT_ROUTER_NAMESPACE
        )
        connection = config.get("connection") or ConnectionMode.IN_CLUSTER
        try:
            self._connection = ConnectionMode(connection)
        except ValueError as e:
            raise ValueError(
                f"KubernetesEnvironment 'connection' is {connection!r}; must be "
                f"one of {', '.join(ConnectionMode)}."
            ) from e
        super().__init__(
            config.get("mount_path") or DEFAULT_WORKSPACE_DIR,
            config.get("working_subdir") or "",
        )
        # The SDK counts this down in whole seconds and rejects nothing, so a
        # float would silently truncate and a 0 would fail every claim.
        timeout = config.get("ready_timeout_seconds")
        self._ready_timeout = 120 if timeout is None else int(timeout)
        if self._ready_timeout <= 0:
            raise ValueError(
                f"KubernetesEnvironment 'ready_timeout_seconds' is "
                f"{self._ready_timeout}; must be positive."
            )
        self._command_timeout = command_timeout(config.get("command_timeout_seconds"))
        # The SDK rejects a non-positive one only once a claim is attempted.
        shutdown_after = config.get("shutdown_after_seconds")
        self._shutdown_after = None if shutdown_after is None else int(shutdown_after)
        if self._shutdown_after is not None and self._shutdown_after <= 0:
            raise ValueError(
                f"KubernetesEnvironment 'shutdown_after_seconds' is "
                f"{self._shutdown_after}; must be positive."
            )
        self._client = client
        self._sandbox: Any = None
        self._read_only_workspace = read_only_required(config)
        self._sealed = False

    def _get_client(self) -> Any:
        """The Agent Sandbox client, created on first use with claim auto-cleanup.

        Deferred so importing this module never requires a reachable cluster. The
        connection config is passed explicitly: the SDK's default shells out to
        ``kubectl port-forward``, so leaving it unset fails every task in a pod,
        where there is no ``kubectl``.
        """
        if self._client is None:
            sandbox_client, in_cluster, tunnel = _require_sdk()
            connection = (
                in_cluster()
                if self._connection is ConnectionMode.IN_CLUSTER
                else tunnel(router_namespace=self._router_namespace)
            )
            self._client = sandbox_client(connection_config=connection, cleanup=True)
        return self._client

    def setup(self, task: Task) -> None:
        """Claim a sandbox, wait for readiness, and upload the task's data.

        Raises:
            EnvironmentUnavailableError: If the cluster could not serve a sandbox.
            RuntimeError: If a sandbox came up but could not be prepared.
        """
        self._sealed = False
        try:
            # create_sandbox watches the Ready condition itself, so a second wait
            # here would already be satisfied; the timeout is what we configure.
            self._sandbox = self._get_client().create_sandbox(
                warmpool=self._warmpool,
                namespace=self._namespace,
                sandbox_ready_timeout=self._ready_timeout,
                shutdown_after_seconds=self._shutdown_after,
            )
        except Exception as e:
            raise EnvironmentUnavailableError(f"Failed to claim a sandbox: {e}") from e
        try:
            self._probe_runtime()
            if task.data_provider is not None:
                task.data_provider()
            if task.data_path is not None:
                self._upload_data(task.data_path)
            if self._read_only_workspace:
                self.seal_workspace()
        except EnvironmentUnavailableError:
            raise
        except Exception as e:
            raise RuntimeError(f"Failed to set up sandbox: {e}") from e

    def seal_workspace(self) -> None:
        """Require the runtime's irreversible seal and verify its basic guarantees."""
        sandbox = self._require_sandbox()
        connector = getattr(sandbox, "connector", None)
        if connector is None:
            raise RuntimeError("Sandbox runtime does not support read_only_workspace")
        try:
            response = connector.send_request(
                "POST",
                "faith/workspace/read-only",
                json={"version": 1, "workspace": self.workspace_dir},
                timeout=120,
                allowed_statuses={404, 405, 409, 422, 501},
            )
        except Exception as e:
            raise RuntimeError(f"Cannot establish read_only_workspace: {e}") from e
        expected = {
            "version": 1,
            "read_only_workspace": True,
            "uploads_disabled": True,
        }
        if response.status_code != 200 or response.json() != expected:
            raise RuntimeError(
                "Sandbox runtime does not support FAITH read_only_workspace v1; "
                "deploy a compatible runtime and its persistent policy-state volume"
            )
        code, output = self._run(
            bounded_shell(workspace_check(self.workspace_dir, sealed=True), 120)
        )
        if code != 0:
            raise RuntimeError(
                f"read_only_workspace verification failed (exit {code}): {output.strip()}"
            )
        # A shell permission check cannot detect a privileged file-API bypass.
        response = connector.send_request(
            "POST",
            "upload",
            files={"file": (_probe_file_name(), b"read-only probe")},
            timeout=60,
            allowed_statuses={403},
        )
        if response.status_code != 403:
            raise RuntimeError("read_only_workspace runtime still accepts uploads")
        self._sealed = True

    def _probe_runtime(self) -> None:
        """Reject a sandbox runtime that cannot serve the contract, before any task.

        One probe covers every requirement because it is the agent's own path — a
        file placed through the file API, read back by a bounded shell command.
        Distinguishing which requirement failed buys nothing: the fix is the same,
        and the marker either comes back or it does not.
        """
        name = _probe_file_name()
        self._write_file(name, _PROBE_MARKER.encode("utf-8"))
        operand = workspace_operand(name)
        exit_code, output = self._run(
            bounded_shell(f"cat {operand}", self._command_timeout)
        )
        # Its own command, not chained onto the read, whose exit code it would
        # otherwise replace.
        self._run(f"rm -f {operand}")
        if exit_code != 0 or output.strip() != _PROBE_MARKER:
            raise RuntimeError(
                f"sandbox runtime does not meet FAITH's contract: reading back a "
                f"file written to {self.workspace_dir} through the sandbox file "
                f"API exited {exit_code} with {output.strip()!r}, expected 0 and "
                f"{_PROBE_MARKER!r}. The runtime must evaluate commands with a "
                f"shell (bash and coreutils timeout on PATH) and root its file "
                f"API at {self.workspace_dir}."
            )

    def teardown(self) -> None:
        """Terminate the sandbox. Idempotent."""
        if self._sandbox is not None:
            try:
                self._sandbox.terminate()
            # terminate can raise from requests, the k8s client or the SDK: three
            # hierarchies with no shared base, two behind the optional extra. Runs
            # in the loop's `finally`, so raising would mask the run's real result.
            except Exception:  # pylint: disable=broad-exception-caught
                pass
            finally:
                self._sandbox = None

    def _upload_data(self, data_path: Path) -> None:
        """Copy a local data path into the sandbox workspace.

        A directory is uploaded as one tar and extracted in place, since
        ``files.write`` writes by basename and would flatten the layout. A single
        file lands at the workspace root, unzipped in place if it is a zip.
        Mirrors ``DockerEnvironment._upload_data``.
        """
        if data_path.is_dir():
            self._untar(tar_directory(data_path))
            return
        name = data_path.name
        self._write_file(name, data_path.read_bytes())
        if name.endswith(".zip"):
            self._extract(name, f"unzip -o {workspace_operand(name)}")

    def _untar(self, archive: bytes) -> None:
        """Upload a tar to a private staging name and extract it in the workspace."""
        name = _staging_archive_name()
        self._write_file(name, archive)
        self._extract(name, f"tar -xf {workspace_operand(name)}")

    def _extract(self, name: str, command: str) -> None:
        """Run an extraction command in the workspace, then remove the archive.

        The removal is unconditional: an archive left behind by a failed extract
        would be visible to the agent as task data.
        """
        operand = workspace_operand(name)
        exit_code, output = self._run(
            f"{command}; status=$?; rm -f {operand}; exit $status"
        )
        if exit_code != 0:
            raise RuntimeError(f"Failed to extract {name}: {output.strip()}")

    def run_command(self, command: str) -> tuple[int, str]:
        """Run a shell command in the sandbox, returning (exit_code, output).

        Bounded identically to DockerEnvironment: an unbounded backend would score
        the same benchmark differently. The SDK's own ``timeout`` is still passed
        as a backstop for a wedged sandbox, but the in-container wrapper is what
        kills the process.
        """
        if self._read_only_workspace and not self._sealed:
            raise RuntimeError(
                "Read-only workspace is not ready; setup must succeed first"
            )
        exit_code, output = self._run(
            bounded_shell(command, self._command_timeout), cwd=self.working_dir
        )
        notice = timeout_notice(exit_code, self._command_timeout)
        return exit_code, f"{output}\n{notice}" if notice else output

    def _run(self, command: str, cwd: str | None = None) -> tuple[int, str]:
        """Run a command string in the sandbox, unbounded, from the workspace root.

        Setup extraction goes through here, exempt from the agent-sized bound and
        keeping the default root: that is where the archive it names was uploaded.
        Only the agent's own commands pass ``working_dir``.

        The `cd` is what sets the working directory — the sandbox API takes none,
        so a command's starting directory is otherwise the runtime's choice.
        DockerEnvironment gets this from `exec_run`'s `workdir`.

        Raises:
            EnvironmentUnavailableError: If the sandbox is gone or unreachable.
        """
        sandbox = self._require_sandbox()
        try:
            result = sandbox.commands.run(
                f"cd {shlex.quote(cwd or self.workspace_dir)} && {command}",
                timeout=sdk_timeout(self._command_timeout),
            )
        # The SDK's own exceptions derive from RuntimeError, but the per-command
        # `connect()` that re-establishes a dead tunnel raises a bare TimeoutError.
        except Exception as e:
            raise EnvironmentUnavailableError(f"Sandbox unavailable: {e}") from e
        output = (result.stdout or "") + (result.stderr or "")
        return result.exit_code, output

    def _write_file(self, name: str, content: bytes) -> None:
        """Write one workspace-root file through the sandbox's file API.

        Raises:
            EnvironmentUnavailableError: If the sandbox is gone or unreachable.
        """
        if self._sealed:
            raise RuntimeError(
                "The read-only workspace is sealed; uploads are disabled"
            )
        # The file API crosses the same tunnel as the command API, so it fails the
        # same way; classifying it as a broken run instead would skip recovery.
        try:
            self._require_sandbox().files.write(name, content)
        except Exception as e:
            raise EnvironmentUnavailableError(f"Sandbox unavailable: {e}") from e

    def _require_sandbox(self) -> Any:
        if self._sandbox is None:
            raise RuntimeError("No sandbox running. Call setup() first.")
        return self._sandbox
