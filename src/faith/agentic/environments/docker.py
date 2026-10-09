# Copyright 2025 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

"""Docker-container execution environment for agentic evaluation."""

import logging
import os
import uuid
from pathlib import Path
from typing import Any

from faith._types.json_types import JsonValue
from faith.agentic.core.environment import (
    DEFAULT_WORKSPACE_DIR,
    Environment,
    EnvironmentUnavailableError,
)
from faith.agentic.core.task import Task
from faith.agentic.environments._archive import (
    tar_bytes,
    tar_directory,
    workspace_operand,
)
from faith.agentic.environments._command import (
    bounded_argv,
    command_timeout,
    sdk_timeout,
    timeout_notice,
)
from faith.agentic.environments._workspace import WorkspaceLayout

logger = logging.getLogger(__name__)


def _require_sdk() -> Any:
    """Import the Docker SDK, naming the extra if it is not installed.

    Not at module scope so `registries`, which references this class, loads
    without the extra.
    """
    try:
        # pylint: disable-next=import-outside-toplevel
        import docker
    except ImportError as e:
        raise ImportError(
            "The 'docker' package is required for DockerEnvironment. Install "
            "it from your checkout with: uv pip install -e '.[docker]'"
        ) from e
    return docker


_DEFAULT_PIDS_LIMIT = 512


def _hardening_kwargs(config: dict[str, Any]) -> dict[str, Any]:
    """Container flags bounding what a runaway agent command can do.

    Docker's defaults are unlimited processes, root's whole capability set and
    setuid escalation, and the commands run here are model-authored.
    """
    pids_limit = config.get("pids_limit")
    kwargs: dict[str, Any] = {
        "pids_limit": _DEFAULT_PIDS_LIMIT if pids_limit is None else int(pids_limit),
        "cap_drop": ["ALL"],
        "security_opt": ["no-new-privileges"],
    }
    if capabilities := config.get("capabilities"):
        kwargs["cap_add"] = [str(c) for c in capabilities]
    return kwargs


class DockerEnvironment(WorkspaceLayout, Environment):
    """Runs agent tools inside an isolated Docker container.

    One container per task, kept alive with ``sleep infinity`` and executed
    against via ``docker exec``. Task data is copied in rather than bind-mounted,
    so the agent gets a writable workspace and cannot reach the host's copy.

    Constructed from the ``environment.config`` sub-dict of a benchmark.yaml:

    - ``agent_image`` (required): image to run the agent in.
    - ``mount_path`` (default ``/workspace``): workspace root and data destination.
    - ``working_subdir`` (default ``""``): injected by `build_environment`, not
      authored here. Setup still targets the root, since the subdirectory only
      exists once the data is extracted.
    - ``environment`` (default ``{}``): environment variables for the container.
    - ``network`` (default ``"none"``): the container's ``network_mode``. A
      benchmark needing egress opts in with e.g. ``"bridge"``. Fixed for the task's
      lifetime, since Docker cannot change it on a running container.
    - ``memory_limit`` (optional): e.g. ``"2g"``, passed as ``mem_limit``.
    - ``cpu_limit`` (optional float): CPU cores, passed as ``nano_cpus``.
    - ``command_timeout_seconds`` (default 60): injected by `build_environment`
      from the benchmark's `limits`, not authored here. A wall-clock bound on each
      command the agent runs, imposed in-container by coreutils ``timeout``; setup
      extraction is exempt, and 0 means no bound.
    - ``pids_limit`` (default 512): cap on processes in the container, so a
      runaway loop fails its own command rather than the host. 0 for unlimited.
    - ``capabilities`` (default none): Linux capabilities to add back; all are
      dropped otherwise. Only an image running as root has any to lose, and one
      writing files it does not own needs at least ``DAC_OVERRIDE``.

    The agent image must provide ``timeout`` (coreutils) as well as ``bash``, and
    should run as a non-root user -- FAITH cannot set that from outside.

    This is hardened against a runaway agent, not against hostile code: a
    container shares the host kernel, and the daemon that starts it is
    root-equivalent. SECURITY.md compares the two backends.
    """

    def __init__(self, config: dict[str, Any], client: Any = None):
        if client is None:
            _require_sdk()
        self._image = config.get("agent_image")
        if not self._image:
            raise ValueError("DockerEnvironment config requires 'agent_image'")
        super().__init__(
            config.get("mount_path") or DEFAULT_WORKSPACE_DIR,
            config.get("working_subdir") or "",
        )
        self._environment = config.get("environment") or {}
        self._network = config.get("network") or "none"
        self._memory_limit = config.get("memory_limit") or None
        self._cpu_limit = config.get("cpu_limit") or None
        self._command_timeout = command_timeout(config.get("command_timeout_seconds"))
        self._hardening = _hardening_kwargs(config)
        self._client = client
        self._container: Any = None

    def _get_client(self) -> Any:
        """The Docker SDK client, created on first use from the ambient environment.

        Deferred so importing this module never requires a running daemon.
        """
        if self._client is None:
            docker = _require_sdk()
            self._client = docker.from_env(timeout=sdk_timeout(self._command_timeout))
        return self._client

    def setup(self, task: Task) -> None:
        """Start the container for a task and copy its data into the workspace.

        Raises:
            EnvironmentUnavailableError: If the daemon could not start a container.
            RuntimeError: If a container started but could not be prepared.
        """
        image = _image_override(task.environment_config) or self._image
        instance_id = uuid.uuid4().hex
        container_name = f"faith-{instance_id}"
        kwargs: dict[str, Any] = {
            "name": container_name,
            "image": image,
            "command": "sleep infinity",
            "detach": True,
            "working_dir": self.workspace_dir,
            "environment": self._environment,
            "network_mode": self._network,
            "auto_remove": True,
            # A run killed without teardown leaks a `sleep infinity` container
            # holding its resource reservation; these make it attributable and
            # bulk-removable (`docker ps -aq -f label=faith.managed=true`).
            "labels": {
                "faith.managed": "true",
                "faith.task-id": task.id,
                "faith.pid": str(os.getpid()),
                "faith.instance-id": instance_id,
            },
            **self._hardening,
        }
        if self._memory_limit:
            kwargs["mem_limit"] = self._memory_limit
        if self._cpu_limit:
            kwargs["nano_cpus"] = int(float(self._cpu_limit) * 1_000_000_000)
        self._start_container(kwargs)
        try:
            if task.data_path is not None:
                self._upload_data(task.data_path)
        except EnvironmentUnavailableError:
            raise
        except Exception as e:
            raise RuntimeError(f"Failed to set up container: {e}") from e

    def _start_container(self, kwargs: dict[str, Any]) -> None:
        """Retain ownership before startup and clean up if either response is lost."""
        try:
            self._container = self._create_container(kwargs)
            self._container.start()
        except Exception as e:
            self._cleanup_failed_start(
                kwargs["name"], kwargs["labels"]["faith.instance-id"]
            )
            raise EnvironmentUnavailableError(
                f"Failed to start a container: {e}"
            ) from e

    def _create_container(self, kwargs: dict[str, Any]) -> Any:
        """Create without starting, retaining containers.run's pull-if-missing behavior."""
        client = self._get_client()
        try:
            return client.containers.create(**kwargs)
        except _require_sdk().errors.ImageNotFound:
            client.images.pull(kwargs["image"])
            return client.containers.create(**kwargs)

    def _cleanup_failed_start(self, name: str, instance_id: str) -> None:
        """Remove only this attempt's container, including a lost create response."""
        try:
            if self._container is None:
                container = self._get_client().containers.get(name)
                if container.labels.get("faith.instance-id") != instance_id:
                    logger.warning(
                        "Refusing startup cleanup of unowned container %s", name
                    )
                    return
                self._container = container
            self._container.remove(force=True)
            self._container = None
        except _require_sdk().errors.NotFound:
            self._container = None
        except Exception as error:  # pylint: disable=broad-exception-caught
            logger.warning(
                "Could not clean up container %s after startup failure: %s",
                name,
                type(error).__name__,
            )

    def _upload_data(self, data_path: Path) -> None:
        """Copy a local data path into the container workspace.

        A directory's contents are copied in preserving layout, modes, and
        symlinks; a single file lands at the workspace root, unzipped in place if
        it is a zip. Mirrors ``KubernetesEnvironment._upload_data``.
        """
        if data_path.is_dir():
            self._put_archive(self.workspace_dir, tar_directory(data_path))
            return
        name = data_path.name
        self._put_files([(name, data_path.read_bytes())])
        if name.endswith(".zip"):
            operand = workspace_operand(name)
            exit_code, output = self._exec(
                [
                    "bash",
                    "-c",
                    f"unzip -o {operand}; status=$?; rm -f {operand}; exit $status",
                ]
            )
            if exit_code != 0:
                raise RuntimeError(f"Failed to unzip {name}: {output.strip()}")

    def _put_files(self, members: list[tuple[str, bytes]]) -> None:
        """Unpack workspace-relative (path, content) pairs into the workspace.

        Sent as one tar, whose paths ``put_archive`` creates directories for.
        """
        self._put_archive(self.workspace_dir, tar_bytes(members))

    def teardown(self) -> None:
        """Stop the container. Idempotent (``auto_remove=True`` deletes it on stop)."""
        if self._container is not None:
            try:
                self._container.stop(timeout=5)
            except Exception:  # pylint: disable=broad-exception-caught
                pass  # Best-effort cleanup; a gone container is already torn down.
            finally:
                self._container = None

    def run_command(self, command: str) -> tuple[int, str]:
        """Run a shell command in the container, returning (exit_code, output).

        Bounded: this is the agent's path, and one runaway command would otherwise
        hang the task. Expiry is appended to the output rather than replacing it,
        so a command killed after printing something does not lose it.
        """
        exit_code, output = self._exec(
            bounded_argv(command, self._command_timeout), workdir=self.working_dir
        )
        notice = timeout_notice(exit_code, self._command_timeout)
        return exit_code, f"{output}\n{notice}" if notice else output

    def _exec(self, argv: list[str], workdir: str | None = None) -> tuple[int, str]:
        """Run argv in the container, unbounded, from the workspace root by default.

        Setup extraction goes through here: it is our own work, not agent input,
        so it is neither bounded nor run from ``working_dir``.

        Raises:
            EnvironmentUnavailableError: If the container or daemon is gone.
        """
        container = self._require_container()
        try:
            exit_code, output = container.exec_run(
                argv, workdir=workdir or self.workspace_dir
            )
        # One SDK call and no FAITH logic, so anything raised is the container or
        # daemon. Enumerating the SDK's classes misses StreamParseError, which
        # derives from neither of its two base hierarchies.
        except Exception as e:  # pylint: disable=broad-exception-caught
            raise EnvironmentUnavailableError(
                f"Docker container unavailable: {e}"
            ) from e
        return exit_code, output.decode("utf-8", "replace")

    def _put_archive(self, dest: str, archive: bytes) -> None:
        """Copy a tar into the container, raising on the SDK's falsy failure."""
        if not self._require_container().put_archive(dest, archive):
            raise RuntimeError(f"Failed to copy files into '{dest}'")

    def _require_container(self) -> Any:
        if self._container is None:
            raise RuntimeError("No container running. Call setup() first.")
        return self._container


def _image_override(environment_config: dict[str, JsonValue]) -> str | None:
    """Per-task agent image from a task's environment_config, if any."""
    override = environment_config.get("agent_image")
    return override if isinstance(override, str) else None
