# Copyright 2025 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

"""Opt-in Docker and real sidecar tests, without a Kubernetes cluster.

Build the agent/runtime test images as in .github/workflows/unit-test.yml,
then set FAITH_TEST_READ_ONLY=1. All containers and volumes are test-owned, have
no host mounts, and expose no host ports.
FAITH_TEST_AGENT_IMAGE and FAITH_TEST_RUNTIME_IMAGE can select immutable images
without retagging another test run's defaults.
"""

# Optional SDKs are loaded only by the explicitly enabled integration fixtures.
# pylint: disable=import-outside-toplevel,protected-access

import base64
import json
import os
from contextlib import suppress
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from faith._types.record.agent import StoppedReason
from faith.agentic.core import ToolRegistry
from faith.agentic.core.environment import EnvironmentUnavailableError
from faith.agentic.core.task import Task
from faith.agentic.environments.docker import DockerEnvironment
from faith.agentic.environments.kubernetes import KubernetesEnvironment
from faith.agentic.tools.bash_exec import BashExecTool
from faith.agentic.tools.submit import SubmitTool
from tests.agentic.core.test_react_loop import _SUBMIT_TURN, _FakeModelAPI, _loop, _turn

pytestmark = pytest.mark.skipif(
    os.environ.get("FAITH_TEST_READ_ONLY") != "1", reason="opt-in container integration"
)

_AGENT_IMAGE = os.environ.get("FAITH_TEST_AGENT_IMAGE", "faith-read-only-agent:test")
_RUNTIME_IMAGE = os.environ.get(
    "FAITH_TEST_RUNTIME_IMAGE", "faith-read-only-runtime:test"
)
_CAPABILITIES = [
    "CHOWN",
    "DAC_OVERRIDE",
    "FOWNER",
    "SETUID",
    "SETGID",
    "SETPCAP",
    "SYS_CHROOT",
    "SYS_PTRACE",
    "SYS_ADMIN",
]

_HTTP = """
import base64, json, sys, time, urllib.request, urllib.error
request = json.loads(sys.argv[1])
for attempt in range(100):
    try:
        urllib.request.urlopen('http://127.0.0.1:8888/', timeout=1).close()
        break
    except OSError:
        time.sleep(.1)
data = base64.b64decode(request['data']) if request['data'] is not None else None
req = urllib.request.Request('http://127.0.0.1:8888/' + request['endpoint'],
                             data=data, method=request['method'], headers=request['headers'])
try:
    response = urllib.request.urlopen(req, timeout=130)
except urllib.error.HTTPError as error:
    response = error
with response:
    print(json.dumps({'status': response.status, 'body': base64.b64encode(response.read()).decode()}))
"""

_LARGE_UPLOAD_HTTP = r"""
import hashlib, http.client, json
boundary = 'faith-large-upload-boundary'
prefix = (
    f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="large.bin"\r\n'
    'Content-Type: application/octet-stream\r\n\r\n'
).encode()
suffix = f'\r\n--{boundary}--\r\n'.encode()
chunk = bytes(range(256)) * 4096
count = 272
connection = http.client.HTTPConnection('127.0.0.1', 8888, timeout=180)
connection.putrequest('POST', '/upload')
connection.putheader('Content-Type', f'multipart/form-data; boundary={boundary}')
connection.putheader('Content-Length', str(len(prefix) + count * len(chunk) + len(suffix)))
connection.endheaders()
connection.send(prefix)
digest = hashlib.sha256()
for _ in range(count):
    connection.send(chunk)
    digest.update(chunk)
connection.send(suffix)
response = connection.getresponse()
print(json.dumps({'status': response.status, 'body': response.read().decode(),
                  'bytes': count * len(chunk), 'sha256': digest.hexdigest()}))
connection.close()
"""


class _ContainerConnector:
    """Use the real HTTP API without opening a port on the host."""

    def __init__(self, runtime):
        self.runtime = runtime

    def is_sandboxd(self):
        """Select the SDK's HTTP runtime protocol."""
        return False

    def send_request(self, method, endpoint, **kwargs):
        """Run an HTTP request inside the sidecar's isolated network namespace."""
        headers, data = {}, None
        if "json" in kwargs:
            headers["Content-Type"] = "application/json"
            data = json.dumps(kwargs["json"]).encode()
        if "files" in kwargs:
            name, content = kwargs["files"]["file"][:2]
            if isinstance(content, str):
                content = content.encode()
            if hasattr(content, "read"):
                content = content.read()
            boundary = "faith-policy-test-boundary"
            headers["Content-Type"] = f"multipart/form-data; boundary={boundary}"
            data = (
                (
                    f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="{name}"\r\n'
                    "Content-Type: application/octet-stream\r\n\r\n"
                ).encode()
                + content
                + f"\r\n--{boundary}--\r\n".encode()
            )
        code, output = self.runtime.exec_run(
            [
                "python",
                "-c",
                _HTTP,
                json.dumps(
                    {
                        "method": method,
                        "endpoint": endpoint,
                        "headers": headers,
                        "data": (
                            base64.b64encode(data).decode()
                            if data is not None
                            else None
                        ),
                    }
                ),
            ]
        )
        assert code == 0, (output.decode(), self.runtime.logs().decode())
        result = json.loads(output)
        body = base64.b64decode(result["body"])
        if result["status"] >= 400 and result["status"] not in kwargs.get(
            "allowed_statuses", set()
        ):
            raise RuntimeError(f"HTTP {result['status']}: {body.decode()}")
        return SimpleNamespace(
            status_code=result["status"],
            content=body,
            text=body.decode(),
            json=lambda: json.loads(body),
        )


@pytest.fixture(name="docker_client")
def _docker_client():
    import docker

    client = docker.from_env(timeout=180)
    yield client
    client.close()


@pytest.fixture(name="sidecar")
def _sidecar(docker_client):
    from docker.errors import NotFound
    from k8s_agent_sandbox.commands.command_executor import CommandExecutor
    from k8s_agent_sandbox.files.filesystem import Filesystem
    from k8s_agent_sandbox.trace_manager import trace

    volume = docker_client.volumes.create(labels={"faith.test": "read-only"})
    agent = runtime = None
    try:
        agent = docker_client.containers.run(
            _AGENT_IMAGE,
            "sleep infinity",
            detach=True,
            network_mode="none",
            cap_drop=["ALL"],
            security_opt=["no-new-privileges"],
            pids_limit=128,
            labels={"faith.test": "read-only"},
        )
        runtime = docker_client.containers.run(
            _RUNTIME_IMAGE,
            detach=True,
            pid_mode=f"container:{agent.id}",
            network_mode=f"container:{agent.id}",
            cap_drop=["ALL"],
            cap_add=_CAPABILITIES,
            volumes={volume.name: {"bind": "/var/lib/faith-policy", "mode": "rw"}},
            pids_limit=128,
            labels={"faith.test": "read-only"},
        )
        connector = _ContainerConnector(runtime)
        tracer = trace.get_tracer("faith-policy-test")
        sandbox = SimpleNamespace(
            connector=connector,
            commands=CommandExecutor(connector, tracer, "test"),
            files=Filesystem(connector, tracer, "test"),
            terminate=lambda: None,
        )
        yield sandbox, runtime, agent
    finally:
        for container in (runtime, agent):
            if container is not None:
                # Retain exited containers until teardown: auto-removal hides
                # OOM/exit diagnostics and makes stop() raise a second error.
                with suppress(NotFound):
                    container.reload()
                    if not container.attrs["State"]["Running"]:
                        print(container.attrs["State"])
                        print(container.logs(tail=50).decode(errors="replace"))
                    container.remove(force=True)
        volume.remove()


@pytest.fixture(name="data")
def _data(tmp_path: Path):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "input.c").write_text("error\nwarn\n", encoding="utf-8")
    (repo / "alias").symlink_to("input.c")
    return tmp_path


def _task(data):
    return Task(
        id="read-only-integration", prompt="test", ground_truth=None, data_path=data
    )


def _kubernetes(sidecar, required=True):
    sandbox, _, _ = sidecar
    return KubernetesEnvironment(
        {"warmpool": "test", "working_subdir": "repo", "read_only_workspace": required},
        client=SimpleNamespace(create_sandbox=lambda **_: sandbox),
    )


_WRITES = [
    "touch created",
    "printf changed > input.c",
    "sed -i 's/error/changed/' input.c",
    "find . -name input.c -delete",
    "rm input.c",
    "mv input.c renamed",
    "chmod u+w input.c",
    "chmod u+w .",
    "mv /workspace/repo /workspace/renamed",
    "mv /workspace /workspace-renamed",
    "touch /workspace/created",
    "printf changed > alias",
]


def _assert_read_only(env):
    assert env.run_command("rg 'error|warn' . | sort")[0] == 0
    for command in _WRITES:
        code, output = env.run_command(command)
        assert code != 0, (command, output)
        assert env.run_command("cat input.c") == (0, "error\nwarn\n")
    assert env.run_command("touch /tmp/faith-scratch && rm /tmp/faith-scratch")[0] == 0


def test_docker_enforces_read_only_and_reapplies_after_recovery(docker_client, data):
    env = DockerEnvironment(
        {
            "agent_image": _AGENT_IMAGE,
            "working_subdir": "repo",
            "read_only_workspace": True,
        },
        client=docker_client,
    )
    try:
        env.setup(_task(data))
        _assert_read_only(env)
        with pytest.raises(RuntimeError, match="uploads are disabled"):
            env._put_files([("repo/input.c", b"changed")])
        env.teardown()
        env.setup(_task(data))
        _assert_read_only(env)
    finally:
        env.teardown()


def test_sidecar_seals_shell_and_uploads_and_survives_restart(sidecar, data):
    env = _kubernetes(sidecar)
    env.setup(_task(data))
    _assert_read_only(env)
    sandbox, runtime, _ = sidecar
    with pytest.raises(RuntimeError, match="HTTP 403"):
        sandbox.files.write("repo/input.c", "changed")
    runtime.restart(timeout=1)
    assert env.run_command("cat input.c") == (0, "error\nwarn\n")
    with pytest.raises(RuntimeError, match="HTTP 403"):
        sandbox.files.write("repo/input.c", "changed")
    env.seal_workspace()  # Idempotent, not a way to reopen the workspace.
    _assert_read_only(env)


def test_docker_writable_mode_is_unchanged(docker_client):
    env = DockerEnvironment({"agent_image": _AGENT_IMAGE}, client=docker_client)
    try:
        env.setup(_task(None))
        assert env.run_command("touch writable && rm writable")[0] == 0
    finally:
        env.teardown()


def test_sidecar_writable_mode_is_unchanged(sidecar, data):
    env = _kubernetes(sidecar, required=False)
    env.setup(_task(data))
    assert env.run_command("touch writable && rm writable")[0] == 0
    assert env.run_command("printf changed > input.c")[0] == 0


def test_large_http_upload_fits_reference_memory_limit(docker_client, sidecar, data):
    """Exercise multipart spooling and atomic copy, with no sender RAM in the runtime."""
    sandbox, runtime, agent = sidecar
    runtime.update(
        mem_limit="256m", memswap_limit="256m", cpu_period=100_000, cpu_quota=20_000
    )
    env = _kubernetes(sidecar, required=False)
    env.setup(_task(data))
    sandbox.files.write("empty", "")
    assert env.run_command("test ! -s /workspace/empty")[0] == 0
    sender = docker_client.containers.run(
        _RUNTIME_IMAGE,
        ["python", "-c", _LARGE_UPLOAD_HTTP],
        detach=True,
        network_mode=f"container:{agent.id}",
        user="1001",
        cap_drop=["ALL"],
        security_opt=["no-new-privileges"],
        mem_limit="128m",
        memswap_limit="128m",
        pids_limit=64,
        labels={"faith.test": "large-upload-sender"},
    )
    try:
        result = sender.wait(timeout=240)
        assert result["StatusCode"] == 0, sender.logs()
        response = json.loads(sender.logs())
    finally:
        sender.remove(force=True)
    assert response["status"] == 200, response
    assert response["bytes"] > 256 * 1024 * 1024
    assert env.run_command("sha256sum /workspace/large.bin") == (
        0,
        f"{response['sha256']}  /workspace/large.bin\n",
    )
    assert env.run_command("printf healthy") == (0, "healthy")
    env.seal_workspace()
    _assert_read_only(env)
    with pytest.raises(RuntimeError, match="HTTP 403"):
        sandbox.files.write("after-seal", "denied")
    runtime.reload()
    assert runtime.attrs["HostConfig"]["Memory"] == 256 * 1024 * 1024
    assert runtime.attrs["HostConfig"]["MemorySwap"] == 256 * 1024 * 1024
    assert runtime.attrs["HostConfig"]["CpuPeriod"] == 100_000
    assert runtime.attrs["HostConfig"]["CpuQuota"] == 20_000
    assert runtime.attrs["State"]["Running"]
    assert not runtime.attrs["State"]["OOMKilled"]
    assert runtime.attrs["RestartCount"] == 0
    assert env._sandbox is sandbox


@pytest.mark.parametrize(
    ("command", "stdout", "stderr", "exit_code"),
    [
        (r"printf '\377'", "\ufffd", "", 0),
        (r"printf '\377' >&2", "", "\ufffd", 0),
        (r"printf '\377'; printf '\376' >&2; exit 23", "\ufffd", "\ufffd", 23),
        ("printf 'café 世界 🐍'; printf 'échec' >&2", "café 世界 🐍", "échec", 0),
        ("printf 'ordinary ASCII\\n'", "ordinary ASCII\n", "", 0),
    ],
    ids=["invalid-stdout", "invalid-stderr", "invalid-nonzero", "unicode", "ascii"],
)
def test_sidecar_returns_decoded_output_and_actual_exit_code(
    sidecar, data, command, stdout, stderr, exit_code
):
    """An already-executed command is a normal HTTP result, even for binary output."""
    env = _kubernetes(sidecar)
    env.setup(_task(data))
    sandbox, _, _ = sidecar
    response = sandbox.connector.send_request(
        "POST", "execute", json={"command": command}
    )
    assert response.status_code == 200
    assert response.json() == {
        "stdout": stdout,
        "stderr": stderr,
        "exit_code": exit_code,
    }
    assert env.run_command("printf 'still healthy'") == (0, "still healthy")
    assert env._sandbox is sandbox
    _assert_read_only(env)


def test_binary_output_does_not_replay_commands_or_replace_a_healthy_sandbox(
    sidecar, data
):
    """Drive recovery-enabled ReAct through the adapter and real HTTP runtime."""
    env = _kubernetes(sidecar)
    command = (
        r"printf x >> /tmp/faith-executions; printf '\377'; printf '\376' >&2; exit 7"
    )
    model = _FakeModelAPI(
        [
            _turn("", ("binary", "bash_exec", json.dumps({"command": command}))),
            _turn(
                "", ("after", "bash_exec", '{"command": "cat /tmp/faith-executions"}')
            ),
            _SUBMIT_TURN,
        ]
    )
    with patch.object(env, "setup", wraps=env.setup) as setup:
        result = _loop(model, ToolRegistry([BashExecTool(), SubmitTool()]), env).run(
            _task(data)
        )
    assert result.stopped_reason == StoppedReason.SUBMITTED
    setup.assert_called_once()
    # One marker proves the command wasn't re-executed after producing binary data.
    assert result.tool_responses[1]["response_truncated"].startswith(
        "x\n[Exit code: 0]"
    )
    assert result.tool_responses[0]["response_truncated"].startswith(
        "\ufffd\ufffd\n[Exit code: 7]"
    )


@pytest.mark.parametrize("stream", ["stdout", "stderr", "both"])
def test_large_output_fits_reference_limit_and_preserves_exit(sidecar, data, stream):
    """Exercise real HTTP/SDK at 256 MiB, including concurrent pipe saturation."""
    sandbox, runtime, _ = sidecar
    runtime.update(
        mem_limit="256m", memswap_limit="256m", cpu_period=100_000, cpu_quota=20_000
    )
    env = _kubernetes(sidecar)
    env.setup(_task(data))
    generate = "head -c 33554432 /dev/zero | tr '\\0' x"
    command = {
        "stdout": generate,
        "stderr": f"({generate}) >&2",
        "both": f"({generate}) & ({generate}) >&2 & wait",
    }[stream]
    command = f"printf x >> /tmp/faith-output-executions; {command}; exit 23"
    response = sandbox.connector.send_request(
        "POST", "execute", json={"command": command}
    )
    assert response.status_code == 200
    result = response.json()
    assert set(result) == {"stdout", "stderr", "exit_code"}
    assert result["exit_code"] == 23
    notice, prefix = (result["stdout"] + result["stderr"]).split("\n", 1)
    total = 67108864 if stream == "both" else 33554432
    assert f"{total} total chars" in notice
    assert prefix == "x" * 1048576
    assert env.run_command("cat /tmp/faith-output-executions") == (0, "x")
    assert env.run_command("touch denied")[0] != 0
    assert env.run_command("cat input.c") == (0, "error\nwarn\n")
    runtime.reload()
    assert runtime.attrs["State"]["Running"]
    assert not runtime.attrs["State"]["OOMKilled"]
    assert runtime.attrs["RestartCount"] == 0
    assert env._sandbox is sandbox


def test_output_cap_survives_huge_model_limit_without_recovery(sidecar, data):
    """A model cannot raise the runtime cap; default limits retain its notice."""
    env = _kubernetes(sidecar)
    command = (
        "printf x >> /tmp/faith-output-executions; "
        "head -c 2097152 /dev/zero | tr '\\0' x"
    )
    model = _FakeModelAPI(
        [
            _turn("", ("normal", "bash_exec", json.dumps({"command": command}))),
            _turn(
                "",
                (
                    "huge",
                    "bash_exec",
                    json.dumps({"command": command, "max_chars": 10**30}),
                ),
            ),
            _turn(
                "",
                (
                    "after",
                    "bash_exec",
                    '{"command": "cat /tmp/faith-output-executions"}',
                ),
            ),
            _SUBMIT_TURN,
        ]
    )
    tool = BashExecTool()
    tool_outputs = []
    invoke = tool._invoke

    def record_output(environment, args):
        output = invoke(environment, args)
        tool_outputs.append(output)
        return output

    with (
        patch.object(env, "setup", wraps=env.setup) as setup,
        patch.object(tool, "_invoke", side_effect=record_output),
    ):
        result = _loop(model, ToolRegistry([tool, SubmitTool()]), env).run(_task(data))
    assert result.stopped_reason == StoppedReason.SUBMITTED
    setup.assert_called_once()
    normal = tool_outputs[0]
    assert normal.startswith("[RUNTIME OUTPUT TRUNCATED — 2097152 total chars")
    assert "showing first 2000]" in normal
    # Inspect before the record's separate logging cap, without another execution.
    huge_notice, huge_output = tool_outputs[1].split("\n", 1)
    assert "2097152 total chars" in huge_notice
    assert huge_output == "x" * 1048576 + "\n[Exit code: 0]"
    assert result.tool_responses[1]["response_truncated"].startswith(
        "[RUNTIME OUTPUT TRUNCATED — 2097152 total chars"
    )
    assert result.tool_responses[2]["response_truncated"].startswith(
        "xx\n[Exit code: 0]"
    )


@pytest.mark.parametrize("ignore_term,expected_exit", [(False, 124), (True, -9)])
def test_high_output_timeout_still_cleans_children(
    sidecar, data, ignore_term, expected_exit
):
    env = _kubernetes(sidecar)
    env.setup(_task(data))
    env._command_timeout = 1
    trap = "trap '' TERM; " if ignore_term else ""
    command = (
        f"{trap}yes x & child=$!; " "printf %s $child > /tmp/faith-output-child; wait"
    )
    code, output = env.run_command(command)
    # SIGKILL of the timeout process itself is -9 from Popen, as it was from
    # subprocess.run; do not substitute a shell's conventional 128 + signal.
    assert code == expected_exit
    assert output.startswith("[RUNTIME OUTPUT TRUNCATED")
    if expected_exit == 124:
        assert output.endswith("ERROR: Command timed out after 1s")
    # A reparented zombie is dead, even if the minimal test PID 1 has not reaped it.
    code, output = env.run_command(
        "pid=$(cat /tmp/faith-output-child); "
        "test ! -e /proc/$pid/stat || test \"$(awk '{print $3}' /proc/$pid/stat)\" = Z"
    )
    assert code == 0, output
    assert env.run_command("printf healthy") == (0, "healthy")


def test_docker_rejects_a_root_image(docker_client):
    env = DockerEnvironment(
        {"agent_image": "ubuntu:24.04", "read_only_workspace": True},
        client=docker_client,
    )
    try:
        with pytest.raises(RuntimeError, match="non-root agent"):
            env.setup(_task(None))
    finally:
        env.teardown()


def test_sidecar_fails_closed_on_partial_or_corrupt_policy(sidecar, data):
    env = _kubernetes(sidecar)
    env.setup(_task(data))
    sandbox, runtime, _ = sidecar
    code, output = runtime.exec_run(
        [
            "python",
            "-c",
            "from pathlib import Path; Path('/var/lib/faith-policy/policy.json').write_text('{')",
        ]
    )
    assert code == 0, output
    runtime.restart(timeout=1)
    with pytest.raises(EnvironmentUnavailableError, match="503"):
        env.run_command("touch writable")
    with pytest.raises(RuntimeError, match="HTTP 403"):
        sandbox.files.write("repo/input.c", "changed")


def test_upload_api_does_not_follow_symlink_parents(sidecar):
    sandbox, _, _ = sidecar
    sandbox.commands.run("ln -s /tmp /workspace/link")
    with pytest.raises(RuntimeError, match="HTTP 500"):
        sandbox.files.write("link/escaped", "changed")
    assert sandbox.commands.run("test ! -e /tmp/escaped").exit_code == 0
