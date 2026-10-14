---
id: kubernetes-deployment
title: Kubernetes deployment
sidebar_label: Kubernetes deployment
---

# Kubernetes deployment

This guide is for operators preparing a Kubernetes Agent Sandbox deployment
for FAITH. To run with the default Docker environment instead, use
[Environment config files](./environment-configs.md#docker).

:::warning
This backend is experimental. It requires a Kubernetes cluster, the Agent Sandbox
controller, a compatible SandboxTemplate and warm pool, and compatible agent and
runtime images. Operators must deploy these resources; FAITH does not install
them. Docker remains the portable default.
:::

## 1. Prepare the cluster \{#cluster-prerequisites}

Before running FAITH, an operator must provide:

1. The Agent Sandbox controller and extensions serving the `v1beta1` Sandbox,
   SandboxClaim, SandboxTemplate, and SandboxWarmPool resources used by the
   pinned `k8s-agent-sandbox==1.0.2` client. Installing the Python extra does not
   install or upgrade those cluster components.
2. The sandbox namespace, a compatible template and warm pool, and agent/runtime
   images that the nodes can pull. The bundled manifest contains two resources,
   not a Namespace or a router deployment. Replace its registry and namespace
   placeholders, and verify image architecture against the target nodes.
3. Kubernetes credentials for the runner: an appropriate service account inside
   the cluster or a working kubeconfig outside it. Connection mode controls the
   runtime transport, not Kubernetes API authentication. Grant only the required
   claim lifecycle, readiness, and resource-read permissions; tunnel mode also
   requires permission to port-forward to the router's pods.
4. Network access to the runtime on port `8888`. In direct mode, allow the FAITH
   runner's namespace; in tunnel mode, allow the **router's** namespace. Where
   both modes are used, authorize both sources. Router egress policy must also
   permit the connection. See Kubernetes
   [NetworkPolicy behavior](https://kubernetes.io/docs/concepts/services-networking/network-policies/).

:::caution
Tunnel mode is not a direct port-forward to the sandbox. Port-forwarding reaches
the router; router-to-sandbox traffic still crosses the cluster network. In the
bundled template, replace `<YOUR_FAITH_NAMESPACE>` with the router namespace for
tunnel-only access, or add a second selector when both connection modes are used.
:::

## 2. Deploy compatible images \{#deployment-controls}

The bundled
[`sandbox-template.yaml`](https://github.com/cisco-foundation-ai/faith/blob/main/src/faith/__benchmarks__/vulnerability-localization/sandbox-template.yaml)
and the adjacent Dockerfile's `agent` and `runtime` targets define compatible
deployment components. Operators must build or obtain compatible images, publish
them to an accessible registry, and configure the template and warm pool.
Custom deployments, including bring-your-own clusters, must preserve these
controls:

- Run the agent as non-root, drop all capabilities, disable privilege escalation,
  and retain the supplied security contexts, including seccomp settings.
- Treat the runtime helper as trusted infrastructure. It requires elevated
  privileges to enter the agent's namespaces, but must execute agent commands
  as the unprivileged agent user without capabilities or privilege escalation.
- Mount a runtime-only `emptyDir` at `/var/lib/faith-policy`, owned by root with
  mode `0700`. Never mount this volume into the agent container. It retains
  protection state across helper restarts within the same pod.
- Refuse execution if protection was interrupted, state is corrupt, or the
  workspace-container identity has changed. Replace the sandbox and repeat
  setup; do not clear policy state to reuse the existing workspace.

Policy-state storage is separate from the runner's result datastore. Its
`emptyDir` is deleted with the pod. Likewise,
[local task checkpoints](../analytics/storage.md#agentic-task-checkpoints) do not
survive deletion of temporary pod storage; use retained datastore storage when
recovery must survive runner replacement.

Operators remain responsible for runtime API access, RBAC, secrets, image
provenance, host and node patching, network-policy enforcement, and container
isolation. FAITH's startup probes are compatibility checks, not a cluster security
audit. See
[`SECURITY.md`](https://github.com/cisco-foundation-ai/faith/blob/main/SECURITY.md#running-agentic-benchmarks)
for the security ownership boundary.

## 3. Configure and check the runner

For an external runner, save the following as `k8s-environment.yaml`, adjusting
the names to match the cluster:

```yaml title="k8s-environment.yaml"
type: kubernetes
config:
  warmpool: vulnerability-localization-pool
  namespace: agent-sandbox
  connection: tunnel
  router_namespace: agent-sandbox-system
```

Read-only deployment checks for these example names:

```shell title="Kubernetes preflight"
kubectl get sandboxwarmpools.extensions.agents.x-k8s.io \
  vulnerability-localization-pool -n agent-sandbox
kubectl get sandboxtemplates.extensions.agents.x-k8s.io \
  vulnerability-localization-sandbox -n agent-sandbox
kubectl get service sandbox-router-svc -n agent-sandbox-system
```

These checks confirm that named resources are visible; they do not prove pod
readiness, runner write permissions, image compatibility, or network-policy
enforcement. Use the [one-task smoke test](../getting-started/quickstart.md#run-an-agentic-benchmark)
with `--environment-config k8s-environment.yaml` after the deployment is ready.
Install `.[openai,kubernetes,metrics]` on that runner instead of the Docker extras.

## Kubernetes settings

| Key | Default | Meaning |
| --- | --- | --- |
| `warmpool` | **required** | Name of the SandboxWarmPool to claim from |
| `namespace` | `default` | Namespace holding the warm pool |
| `connection` | `in_cluster` | How the client reaches the sandbox — `in_cluster` or `tunnel` |
| `router_namespace` | `agent-sandbox-system` | Namespace of the router service used by `tunnel`; ignored by `in_cluster` |
| `ready_timeout_seconds` | `120` | How long to wait for the claimed sandbox to report ready |
| `shutdown_after_seconds` | derived | Absolute claim lifetime; defaults to the task timeout plus 600 seconds, or unset without a task timeout; an explicit positive value overrides this |

`warmpool` is deployment-specific and has no default. Set `namespace` to the
namespace containing the pool if it is not `default`.

`connection: in_cluster` connects directly to the sandbox pod, using its IP when
available and cluster DNS as a fallback. `connection: tunnel` uses
`kubectl port-forward` to `svc/sandbox-router-svc` in `router_namespace`; that
router then proxies requests to the sandbox. It requires `kubectl` on the
runner and a deployed router. FAITH defaults to `in_cluster`; only these two
values are accepted.

The shutdown lifetime is cluster-enforced and absolute, not an idle timeout.
It can delete an active sandbox. For the packaged 1,800-second task budget, the
derived lifetime is 2,400 seconds. An explicit override must allow time for
setup and execution; it does not extend the benchmark's own task budget.

As with Docker, `command_timeout_seconds` is a
[benchmark limit](../benchmarks/creating-benchmarks.md#limits), not an environment
setting.

This backend has no `network` setting. Restricting pod egress requires a
NetworkPolicy selecting the pod and a cluster CNI that enforces it. Network
isolation is an operator responsibility, not a guarantee of workspace protection.

## Runtime compatibility

The runtime must satisfy the [command, file, and workspace-policy contracts](../reference/sandbox-runtime.md).
Use the supplied compatible components described above, or validate a custom
implementation against that reference. A ready claim alone does not establish
runtime compatibility.
