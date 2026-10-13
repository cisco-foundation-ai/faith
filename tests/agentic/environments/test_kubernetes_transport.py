# Copyright 2025 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

"""Real SDK/Requests regressions without Kubernetes or an idle-timeout race.

Only the Kubernetes API and kubectl process are doubled. A loopback HTTP/1.1
server counts accepted connections and side effects. It is a transport fixture,
not a replacement for the real runtime's filesystem-protection integration tests.
"""

import json
import threading
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock, patch

import pytest

from faith.agentic.core.environment import EnvironmentUnavailableError
from faith.agentic.core.task import Task
from faith.agentic.environments._kubernetes_transport import configure_http_transport
from faith.agentic.environments.kubernetes import KubernetesEnvironment

sdk = pytest.importorskip("k8s_agent_sandbox")
models = pytest.importorskip("k8s_agent_sandbox.models")
client_module = pytest.importorskip("k8s_agent_sandbox.sandbox_client")
connector_module = pytest.importorskip("k8s_agent_sandbox.connector")
exceptions = pytest.importorskip("k8s_agent_sandbox.exceptions")


def _task():
    return Task(id="transport", prompt="test", ground_truth=None)


@pytest.fixture(name="clients")
def _clients(monkeypatch):
    """Real SDK claim construction/termination with only the cluster API faked."""
    created: list[Any] = []

    def create(connection):
        helper = Mock()
        helper.create_sandbox_claim.return_value = {
            "metadata": {"resourceVersion": "1"}
        }
        helper.wait_for_claim_ready.side_effect = (
            f"sandbox-{len(created)}-{i}" for i in range(10)
        )
        helper.get_sandbox.return_value = {"status": {"podIPs": ["127.0.0.1"]}}
        monkeypatch.setattr(client_module, "K8sHelper", lambda: helper)
        client = sdk.SandboxClient(connection_config=connection)
        created.append(client)
        return client

    yield create
    for client in created:
        client.delete_all()


@pytest.mark.parametrize("tunnel", [False, True])
def test_policy_preserves_session_settings_and_unrelated_connectors(clients, tunnel):
    connection = (
        models.SandboxLocalTunnelConnectionConfig()
        if tunnel
        else models.SandboxInClusterConnectionConfig()
    )
    client = clients(connection)
    sandbox = client.create_sandbox("pool")
    unrelated = client.create_sandbox("pool")
    session = sandbox.connector.session
    session.headers.update({"Authorization": "Bearer test-only", "X-Custom": "value"})
    session.cookies.set("test", "value")
    session.auth = ("test-user", "test-password")
    session.verify = "/test/ca.pem"
    session.cert = ("/test/cert.pem", "/test/key.pem")
    existing_hook = Mock()
    session.hooks["response"].append(existing_hook)
    before = dict(session.__dict__)
    headers = dict(session.headers)
    retries = {key: adapter.max_retries for key, adapter in session.adapters.items()}
    with patch.object(session, "request", side_effect=AssertionError("early request")):
        configure_http_transport(sandbox, tunnel=tunnel)
    assert sandbox.connector.session is session
    assert session.__dict__ == before
    assert dict(session.headers) == {
        **headers,
        "Connection": "close" if tunnel else "keep-alive",
    }
    assert unrelated.connector.session.headers["Connection"] == "keep-alive"
    assert session.hooks["response"][-1] is existing_hook
    assert len(session.hooks["response"]) == (2 if tunnel else 1)
    assert not unrelated.connector.session.hooks["response"]
    for key, retry in retries.items():
        assert session.adapters[key].max_retries is retry
        assert "POST" not in retry.allowed_methods


@pytest.mark.parametrize("configured_tunnel", [False, True])
def test_injected_mode_mismatch_fails_before_probe_and_cleans_claim(
    clients, configured_tunnel
):
    connection = (
        models.SandboxInClusterConnectionConfig()
        if configured_tunnel
        else models.SandboxLocalTunnelConnectionConfig()
    )
    client = clients(connection)
    env = KubernetesEnvironment(
        {
            "warmpool": "pool",
            "connection": "tunnel" if configured_tunnel else "in_cluster",
        },
        client=client,
    )
    with (
        patch.object(env, "_probe_runtime") as probe,
        pytest.raises(RuntimeError, match="does not match.*connection"),
    ):
        env.setup(_task())
    probe.assert_not_called()
    client.k8s_helper.delete_sandbox_claim.assert_called_once()
    env.teardown()
    client.k8s_helper.delete_sandbox_claim.assert_called_once()
    assert not client.list_active_sandboxes()


@pytest.mark.parametrize("missing", ["connector", "connection_config", "session"])
def test_unsupported_tunnel_transport_fails_without_runtime_traffic(clients, missing):
    client = clients(models.SandboxLocalTunnelConnectionConfig())
    sandbox = client.create_sandbox("pool")
    # Keep the actual handle intact for SDK cleanup, but omit the hook presented
    # to FAITH. Native custom clients are not required to expose these internals.
    connector = SimpleNamespace(
        connection_config=sandbox.connector.connection_config,
        session=sandbox.connector.session,
    )
    if missing != "connector":
        delattr(connector, missing)
    handle = SimpleNamespace(
        connector=None if missing == "connector" else connector,
        terminate=Mock(wraps=sandbox.terminate),
    )
    env = KubernetesEnvironment(
        {"warmpool": "pool", "connection": "tunnel"},
        client=SimpleNamespace(create_sandbox=lambda **_: handle),
    )
    with (
        patch.object(env, "_probe_runtime") as probe,
        pytest.raises(RuntimeError, match="connector"),
    ):
        env.setup(_task())
    probe.assert_not_called()
    handle.terminate.assert_called_once()
    assert sandbox.connector.session.headers["Connection"] == "keep-alive"


class _RuntimeServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self):
        super().__init__(("127.0.0.1", 0), _RuntimeHandler)
        self.records = []
        self.connections = 0
        self.lock = threading.Lock()
        self.sealed = set()
        self.effects = 0
        self.failure = "disconnect"
        self.keep_open = False
        self.output = "ok"
        self.body_started = threading.Event()
        self.release_body = threading.Event()


class _RuntimeHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    close_connection = True
    server: _RuntimeServer

    def setup(self):
        super().setup()
        self.connection.settimeout(5)
        with self.server.lock:
            self.server.connections += 1
            self.connection_id = self.server.connections

    def log_message(self, *_args):
        """Keep successful HTTP fixture traffic out of test output."""

    def do_POST(self):  # pylint: disable=invalid-name
        """Record every actual request, then serve the minimal runtime contract."""
        body = self.rfile.read(int(self.headers.get("Content-Length", "0")))
        self.server.records.append(
            {
                "connection": self.connection_id,
                "path": self.path,
                "headers": dict(self.headers),
                "body": body,
            }
        )
        if b"side-effect" in body:
            self.server.effects += 1
            if self.server.failure == "disconnect":
                self.close_connection = True  # Side effect occurred, response lost.
            elif self.server.failure == "truncated":
                self._respond(200, {}, truncated=True)
            else:
                self._respond(503, {"detail": "failed after side effect"})
            return
        self._serve(body)

    def _serve(self, body):
        sandbox_id = self.headers.get("X-Sandbox-ID", "native")
        if self.path == "/faith/workspace/read-only":
            self.server.sealed.add(sandbox_id)
            self._respond(
                200,
                {"version": 1, "read_only_workspace": True, "uploads_disabled": True},
            )
        elif self.path == "/upload":
            self._respond(403 if sandbox_id in self.server.sealed else 200, {})
        elif self.path == "/execute":
            command = json.loads(body)["command"]
            probe = ".faith-probe-" in command and "cat ./" in command
            self._respond(
                200,
                {
                    "stdout": "faith-runtime-ok" if probe else "ok",
                    "stderr": "",
                    "exit_code": 0,
                },
            )
        elif self.path == "/redirect":
            self._respond(307, {}, {"Location": "/redirect-target"})
        elif self.path == "/stream":
            self._respond(200, {"stdout": self.server.output})
        else:
            self._respond(200, {})

    def _respond(self, status, payload, headers=None, truncated=False):
        body = json.dumps(payload, ensure_ascii=False).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        for key, value in (headers or {}).items():
            self.send_header(key, value)
        # Also simulate a peer/intermediary that does not propagate close: the
        # mitigation must detach the pool, not rely on observing FIN in time.
        if self.server.keep_open:
            self.close_connection = False
        self.end_headers()
        if truncated:
            self.wfile.write(body[:1])
            self.close_connection = True
            return
        if self.path == "/stream":
            self.wfile.write(body[:23])
            self.wfile.flush()
            self.server.body_started.set()
            if not self.server.release_body.wait(timeout=10):
                self.close_connection = True
                return
            body = body[23:]
        self.wfile.write(body)


@pytest.fixture(name="runtime")
def _runtime(monkeypatch):
    """Loopback HTTP server and one inert, counted kubectl process per tunnel."""
    server = _RuntimeServer()
    thread = threading.Thread(
        target=server.serve_forever, kwargs={"poll_interval": 0.01}
    )
    thread.start()
    processes = []

    def start(*_args, **_kwargs):
        process = Mock()
        process.poll.return_value = None
        processes.append(process)
        return process

    monkeypatch.setattr(connector_module.subprocess, "Popen", start)
    monkeypatch.setattr(
        connector_module.LocalTunnelConnectionStrategy,
        "_get_free_port",
        lambda _: server.server_port,
    )
    monkeypatch.setattr(
        connector_module.LocalTunnelConnectionStrategy, "_is_port_open", lambda *_: True
    )
    # Never route loopback fixtures through a user's configured HTTP proxy.
    monkeypatch.setenv("NO_PROXY", "127.0.0.1")
    try:
        yield server, processes
    finally:
        server.release_body.set()
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
        assert not thread.is_alive()


def _environment(client, *, tunnel=True, protected=False):
    return KubernetesEnvironment(
        {
            "warmpool": "pool",
            "namespace": "test-ns",
            "connection": "tunnel" if tunnel else "in_cluster",
            "read_only_workspace": protected,
        },
        client=client,
    )


def test_real_sdk_setup_recovery_and_all_runtime_endpoints(runtime, clients, tmp_path):
    server, processes = runtime
    client = clients(
        models.SandboxLocalTunnelConnectionConfig(router_namespace="router-ns")
    )
    # Exercise FAITH's owned-client path, not just constructor injection. The real
    # SDK create_sandbox runs; Kubernetes API responses alone are supplied above.
    with patch(
        "faith.agentic.environments.kubernetes._require_sdk",
        return_value=(
            Mock(return_value=client),
            models.SandboxInClusterConnectionConfig,
            models.SandboxLocalTunnelConnectionConfig,
        ),
    ):
        env = KubernetesEnvironment(
            {
                "warmpool": "pool",
                "namespace": "test-ns",
                "connection": "tunnel",
                "read_only_workspace": True,
            }
        )
        data = tmp_path / "task.txt"
        data.write_text("fixture data")
        task = Task(id="transport", prompt="test", ground_truth=None, data_path=data)
        for _ in range(2):
            env.setup(task)
            assert env.run_command("echo ok") == (0, "ok")
            env.teardown()
        env.teardown()
    assert len(processes) == 2
    for process in processes:
        process.terminate.assert_called_once()
    assert client.k8s_helper.create_sandbox_claim.call_count == 2
    assert client.k8s_helper.delete_sandbox_claim.call_count == 2
    assert len(server.records) == 16  # Eight requests per protected task setup/run.
    assert server.connections == len(server.records)
    assert {r["path"] for r in server.records} == {
        "/execute",
        "/upload",
        "/faith/workspace/read-only",
    }
    for record in server.records:
        assert record["headers"]["Connection"] == "close"
        assert record["headers"]["X-Sandbox-Namespace"] == "test-ns"
        assert record["headers"]["X-Sandbox-Port"] == "8888"
        assert (
            record["headers"]["X-Sandbox-Timeout"]
            == {
                "/upload": "60",
                "/execute": "600",
                "/faith/workspace/read-only": "120",
            }[record["path"]]
        )


def test_native_and_tunnel_sessions_stay_independent_under_concurrent_use(
    runtime, clients
):
    server, processes = runtime
    native = _environment(
        clients(
            models.SandboxInClusterConnectionConfig(server_port=server.server_port)
        ),
        tunnel=False,
    )
    tunnel = _environment(clients(models.SandboxLocalTunnelConnectionConfig()))
    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            list(pool.map(lambda env: env.setup(_task()), (native, tunnel)))
            for _ in range(3):
                assert list(
                    pool.map(lambda env: env.run_command("echo ok"), (native, tunnel))
                ) == [(0, "ok"), (0, "ok")]
        native_records = [
            r for r in server.records if "X-Sandbox-ID" not in r["headers"]
        ]
        tunnel_records = [r for r in server.records if "X-Sandbox-ID" in r["headers"]]
        assert len(native_records) == len(tunnel_records) == 6
        assert len({r["connection"] for r in native_records}) == 1
        assert len({r["connection"] for r in tunnel_records}) == 6
        assert len(processes) == 1
        processes[0].terminate.assert_not_called()
    finally:
        native.teardown()
        tunnel.teardown()
    processes[0].terminate.assert_called_once()


@pytest.mark.parametrize("operation", ["execute", "upload"])
@pytest.mark.parametrize("failure", ["disconnect", "503", "truncated"])
def test_ambiguous_post_failure_is_not_replayed(runtime, clients, operation, failure):
    # No ReActLoop here: its opt-in/default recovery is a separate replay layer.
    server, _ = runtime
    server.failure = failure
    env = _environment(clients(models.SandboxLocalTunnelConnectionConfig()))
    try:
        env.setup(_task())
        before = len(server.records)
        with pytest.raises(EnvironmentUnavailableError):
            if operation == "execute":
                env.run_command("side-effect")
            else:
                # pylint: disable-next=protected-access
                env._write_file("effect.txt", b"side-effect")
        assert server.effects == 1
        assert len(server.records) == before + 1
    finally:
        env.teardown()


def test_security_headers_timeout_and_redirect_rejection_survive(runtime, clients):
    server, _ = runtime
    sandbox = clients(models.SandboxLocalTunnelConnectionConfig()).create_sandbox(
        "pool"
    )
    connector = sandbox.connector
    connector.session.headers["Authorization"] = "Bearer test-only"
    configure_http_transport(sandbox, tunnel=True)
    with pytest.raises(exceptions.SandboxRequestError):
        connector.send_request(
            "POST",
            "redirect",
            timeout=3,
            allow_redirects=True,
            headers={"X-Custom": "kept"},
        )
    assert len(server.records) == 1
    headers = server.records[0]["headers"]
    assert headers["Connection"] == "close"
    assert headers["Authorization"] == "Bearer test-only"
    assert headers["X-Custom"] == "kept"
    assert headers["X-Sandbox-Timeout"] == "3"


def test_fresh_sockets_do_not_depend_on_the_peer_closing(runtime, clients):
    # Deterministically catches the header-only gap: no response Connection header
    # and no observable FIN, as if a forwarding peer delayed/omitted the close.
    server, processes = runtime
    server.keep_open = True
    env = _environment(clients(models.SandboxLocalTunnelConnectionConfig()))
    try:
        env.setup(_task())
        for _ in range(10):
            assert env.run_command("echo ok") == (0, "ok")
        assert len(server.records) == server.connections == 13
        assert len({r["connection"] for r in server.records}) == 13
        assert len(processes) == 1
        processes[0].terminate.assert_not_called()
    finally:
        env.teardown()


def test_pool_discard_preserves_in_flight_stream_and_caller_hooks(runtime, clients):
    server, processes = runtime
    server.keep_open = True
    server.output = "complete Unicode body: αβ🙂\n" * 20000
    sandbox = clients(models.SandboxLocalTunnelConnectionConfig()).create_sandbox(
        "pool"
    )
    connector = sandbox.connector
    received = []
    connector.session.hooks["response"].append(
        lambda response, **_: received.append(response)
    )
    configure_http_transport(sandbox, tunnel=True)
    response = connector.send_request("POST", "stream", stream=True, timeout=5)
    try:
        assert server.body_started.wait(timeout=5)
        assert not response.raw.closed  # The HTTP body is still in flight.
        # A second request on the SAME connector clears its own pool while the
        # first body is paused; it must neither reuse nor close that first socket.
        with ThreadPoolExecutor(max_workers=1) as pool:
            other = pool.submit(
                connector.send_request,
                "POST",
                "execute",
                json={"command": "echo ok"},
                timeout=5,
            ).result(timeout=5)
        assert other.json()["stdout"] == "ok"
        assert not response.raw.closed
        server.release_body.set()
        body = b"".join(response.iter_content(chunk_size=1023))
        assert json.loads(body) == {"stdout": server.output}
        assert received == [response, other]
        assert server.connections == len(server.records) == 2
        assert len(processes) == 1
        processes[0].terminate.assert_not_called()
    finally:
        server.release_body.set()
        response.close()
        sandbox.terminate()
    processes[0].terminate.assert_called_once()


def test_native_custom_client_does_not_require_the_sdk():
    with patch.dict("sys.modules", {"k8s_agent_sandbox.models": None}):
        configure_http_transport(SimpleNamespace(), tunnel=False)
        configure_http_transport(
            SimpleNamespace(connector=SimpleNamespace(connection_config=object())),
            tunnel=False,
        )
        with pytest.raises(RuntimeError, match="supported 'kubernetes' extra"):
            configure_http_transport(SimpleNamespace(), tunnel=True)
