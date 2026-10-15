# Copyright 2025 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

"""Per-sandbox compatibility with the pinned Agent Sandbox SDK's HTTP session.

SDK 1.0.2 has no connection-policy option. Its sandbox owns a Requests session,
created without runtime traffic, and upload/execute/seal all use that session.
Keep this implementation-detail dependency here and exercise it with the real
SDK in tests. Do not replace its adapters, security settings, or retry policy.
"""

from functools import partial
from typing import Any


def _discard_http_pools(session: Any, response: Any, **_kwargs: Any) -> Any:
    """Detach the HTTP pools before this response can return its socket to one.

    Requests calls response hooks before reading the body. Session.close clears
    its adapters' pools without interrupting in-flight reads; subsequent requests
    get new pools with the same adapter/retry settings. This does not call the
    SDK connector's close, which would also stop kubectl. Leave the response
    untouched so body reads, cookies, security checks and caller hooks still run.
    """
    session.close()
    return response


def configure_http_transport(sandbox: Any, *, tunnel: bool) -> None:
    """Avoid stale pooled HTTP sockets through port-forward, not tunnel reuse.

    Call before the first runtime request on every newly claimed sandbox. The
    session is owned by that sandbox, not shared across SDK clients. Native
    clients keep pooling; injected tunnel clients must agree with FAITH's mode
    and expose the supported SDK transport. Legacy native test/custom clients
    need not expose a connector or have the optional SDK installed.

    Request ``Connection: close`` and discard response pools: the header alone
    cannot prevent reuse if a peer omits the close response header and its FIN is
    delayed. This costs a new HTTP connection per request but leaves the tunnel
    alive. It adds no replay of ambiguous failed POSTs; FAITH's separate
    environment-recovery policy is unchanged.

    Raises:
        RuntimeError: If the connection mode disagrees or the tunnel transport
            cannot establish the policy with the supported SDK.
    """
    connector = getattr(sandbox, "connector", None)
    connection = getattr(connector, "connection_config", None)
    if connection is None and not tunnel:
        return
    try:
        # Optional dependency: native custom clients still work without the SDK.
        # pylint: disable-next=import-outside-toplevel
        from k8s_agent_sandbox.models import SandboxLocalTunnelConnectionConfig
    except ImportError as e:
        if not tunnel:
            return
        raise RuntimeError(
            "Cannot configure tunnel HTTP connections; install FAITH's "
            "supported 'kubernetes' extra"
        ) from e

    actual_tunnel = isinstance(connection, SandboxLocalTunnelConnectionConfig)
    if actual_tunnel != tunnel:
        raise RuntimeError(
            "Sandbox connector does not match KubernetesEnvironment 'connection': "
            "tunnel requires SandboxLocalTunnelConnectionConfig, including for "
            "an injected client; configure both to use the same connection mode"
        )
    if not tunnel:
        return

    # pylint: disable-next=import-outside-toplevel
    from requests import Session

    session = getattr(connector, "session", None)
    if not isinstance(session, Session):
        raise RuntimeError(
            "Cannot configure tunnel HTTP connections: the sandbox connector "
            "must expose a Requests session; use FAITH's supported "
            "'kubernetes' extra or a compatible injected client"
        )
    session.headers["Connection"] = "close"
    session.hooks["response"].insert(0, partial(_discard_http_pools, session))
