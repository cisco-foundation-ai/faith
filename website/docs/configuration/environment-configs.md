---
id: environment-configs
title: Environment config files
sidebar_label: Environment configs
sidebar_position: 3
---

# Environment config files

Every [agentic benchmark](../benchmarks/creating-benchmarks.md#agentic-benchmarks)
declares an execution environment and provides a portable default, normally
`docker`. A deployment can override this environment for one run with
`--environment-config <file.yaml>`.

Use the benchmark's default Docker environment for a first run; no override file
is required. For other deployment work, see:

- [Kubernetes deployment](./kubernetes-deployment.md): cluster prerequisites,
  images, security controls, and runner configuration.
- [Sandbox runtime API](../reference/sandbox-runtime.md): command, upload,
  and workspace-protection APIs for custom runtime implementations.
- [Benchmark configuration](../benchmarks/creating-benchmarks.md#environments-and-tools):
  the environment and tools declared by a benchmark.

Sandbox recovery is configured separately through
`agentic_config.limits.recover_on_environment_failure`, which defaults to `true`.
It rebuilds the sandbox without restoring agent-written files or process state.
See [sandbox recovery](../benchmarks/creating-benchmarks.md#sandbox-recovery) for
when it must be disabled; an environment override does not change this setting.

## `docker`

Runs each task in a separate container and executes commands through
`docker exec`. FAITH copies task data into the container; it does not mount the
runner's repository. Workspace write access depends on the benchmark's
read-only policy. Requires the
[`docker` extra](../getting-started/installation.md#available-extras).

### Docker preflight

From the repository root, with the FAITH environment activated:

```shell title="Docker preflight"
uv pip install -e '.[openai,docker,metrics]'
python - <<'PY'
import docker

client = docker.from_env(timeout=10)
try:
    assert client.ping(), "Docker daemon did not respond"
    print("Docker daemon reachable from Python")
finally:
    client.close()
PY
```

Run this as the same user and in the same environment as `faith query`. FAITH
uses the Python SDK's `from_env`, not the Docker CLI's selected context. If the
CLI works but this check fails, configure the SDK's daemon endpoint and, where
needed, TLS settings; see the
[Docker client reference](https://docker-py.readthedocs.io/en/7.1.0/client.html).
Do not make the daemon socket world-writable or expose an unauthenticated TCP
socket to resolve a permission error.

The packaged vulnerability-localization configuration requests two CPU cores and
4 GB of container memory per task. Allow additional resources for Docker,
image builds, the runner's cache, and the read-only archive round trip. A
successful daemon ping does not verify that an image can build or a task can run.
Continue with the [one-task smoke test](../getting-started/quickstart.md#run-an-agentic-benchmark).

## Override an environment \{#the-file}

To change deployment settings, save a mapping with a registered `type` and a
backend-specific `config`. This example increases the memory limit while
retaining the benchmark's other Docker settings:

```yaml title="docker-environment.yaml"
type: docker
config:
  memory_limit: "8g"
```

Add `--environment-config docker-environment.yaml` to your `faith query` or
`faith run-all` command. One file applies to the entire run. Passing it when no
requested benchmark is agentic is an error.

The file uses the same extended YAML loader as `benchmark.yaml`, including
`!from` composition. An unregistered `type` or a `config` value that is not a
mapping is rejected before benchmark loading. The error identifies the file.

## How it merges

The file overrides the benchmark's environment section. Deployment values take
precedence for named fields; the `config` mapping merges only when the backend
type matches.

| The file sets | Result |
| --- | --- |
| The same `type` | The two `config` mappings deep-merge; unspecified benchmark values are retained |
| A different `type` | The benchmark's backend-specific `config` is discarded |

Types are compared by registered name. For example, `Docker` and `docker` identify
the same backend and therefore merge.

Nested mappings cannot be cleared through this merge. Adding a container
environment variable retains the benchmark's other variables; `null` means
unset rather than a request to delete inherited values.

## Docker settings

| Key | Default | Meaning |
| --- | --- | --- |
| `agent_dockerfile` | — | Dockerfile to build the agent image from |
| `agent_image` | — | Published image to pull |
| `network` | `none` | The container's `network_mode` |
| `environment` | `{}` | Environment variables for the container |
| `memory_limit` | — | e.g. `"2g"`, passed as `mem_limit` |
| `cpu_limit` | — | CPU cores, passed as `nano_cpus` |
| `pids_limit` | `512` | Maximum number of container processes; `0` disables the limit |
| `capabilities` | `[]` | Linux capabilities to add after dropping all; rejected for read-only workspaces |

Networking is disabled by default. A benchmark requiring egress can select
`network: "bridge"`. The selected network mode applies for the task's lifetime,
including setup and agent execution.

`command_timeout_seconds` belongs in the benchmark's
[`limits`](../benchmarks/creating-benchmarks.md#limits), not the environment config.

### Build or pull

The image configuration determines whether FAITH builds or pulls an image:

- `agent_dockerfile`: FAITH builds the image and tags it as
  `faith-build/<context>:<digest of the build context>`. Build-context changes
  produce a new tag without overwriting a published image. Previous images remain
  available; remove unused FAITH-managed images with
  `docker image prune -a --filter label=faith.managed=true`.
- `agent_image`: FAITH uses the named local image or pulls it from a registry
  accessible to the runner when it is absent locally. It does not build an image.

Setting both keys in one authored file is rejected. Validation applies per file,
not to the merged result: an environment override may supply `agent_image` for a
benchmark that specifies `agent_dockerfile`. In that case, FAITH uses the
override image instead of building the benchmark image.

A task may also carry an `agent_image` in its own `environment_config`, which
also selects a prebuilt image. Task-level configuration cannot request a build.

### Custom images with read-only inputs

The supplied vulnerability-localization image satisfies the read-only policy.
Custom images must provide:

- A non-root agent user with no Linux capabilities or privilege escalation.
  Added capabilities are incompatible with this policy.
- `bash`, GNU coreutils, `find`, and `awk` on `PATH`, with `/proc/self/status`
  available for privilege checks.
- A canonical `/workspace` path with root-owned ancestors that are not writable
  by the agent.
- The tools required by the benchmark's preparation commands and agent workflow.

Protection uses an archive round trip through the runner and requires memory
proportional to workspace size. FAITH does not extract this archive into the
runner's repository or expose that repository through a bind mount.

Docker remains a local, trusted-input backend. Access to the Docker daemon can
bypass container permissions and these workspace protections.

## `kubernetes`

Claims one sandbox per task from a warm pool of Agent Sandboxes. This experimental
backend requires a prepared cluster and the `kubernetes` extra; FAITH does not
deploy the cluster resources. Follow the
[Kubernetes deployment guide](./kubernetes-deployment.md) for setup and the
[backend settings](./kubernetes-deployment.md#kubernetes-settings) for connection
modes, namespaces, readiness, and sandbox lifetime.

### Cluster prerequisites

<span id="deployment-controls" />

Operators must provide compatible images, a template and warm pool, runner
permissions, and runtime network access. See the
[cluster prerequisites](./kubernetes-deployment.md#cluster-prerequisites) and
[deployment controls](./kubernetes-deployment.md#deployment-controls) before a run.

### Runtime compatibility \{#what-the-sandbox-image-must-provide}

<span id="custom-runtime-workspace-policy-api" />

Custom runtimes must implement the [command and file API](../reference/sandbox-runtime.md#what-the-sandbox-image-must-provide)
and, for protected benchmarks, the [workspace policy API](../reference/sandbox-runtime.md#custom-runtime-workspace-policy-api).
These contracts are for runtime implementers; the deployment guide identifies
the supplied compatible images.

## Read-only workspace requirements

The benchmark setting
[`agentic_config.read_only_workspace`](../benchmarks/creating-benchmarks.md#read-only-workspaces)
requires workspace protection. It defaults to `false`; both vulnerability-localization
phases enable it. This requirement is outside the replaceable `environment`
section. Changing backends or supplying an environment override cannot disable it.

Trusted setup prepares the task repository, then protects `/workspace` before
the first agent command. FAITH verifies the protection and rejects environments
that do not support it. Recovery must prepare and protect the replacement
environment again. Custom images must not start untrusted processes before
protection is established.

The policy prevents workspace changes and disables uploads after protection. It
does not restrict shell syntax or impose a command allowlist. Scratch paths
outside the workspace, including `/tmp`, may remain writable. It does not itself
restrict networking, prevent denial of service, or provide a hardened sandbox
for hostile code. Images, runtime helpers, and infrastructure remain trusted.
See the ownership guidance in
[`SECURITY.md`](https://github.com/cisco-foundation-ai/faith/blob/main/SECURITY.md#running-agentic-benchmarks).

## Runner network and cache

Model requests and task-data downloads originate from the FAITH runner, not the
agent container. For the packaged vulnerability-localization benchmarks:

| Component | Required access |
| --- | --- |
| FAITH runner | Model API; published manifest on `raw.githubusercontent.com`; repository archives on `github.com` and its archive download hosts |
| Docker image builder / Kubernetes nodes | Base-image or published-image registries; package repositories when building images |
| Agent workspace | Uploaded task files; no network egress is required by these benchmarks |

Docker's default `network: none` applies to task containers. It does not disable
runner downloads or image-build networking. Changing the container to `bridge`
does not fix a runner-side download failure.

The loader stores fetched payloads under
`<FAITH_CACHE_DIR>/faith/vulnerability_localization`. If `FAITH_CACHE_DIR` is
unset, it uses `<XDG_CACHE_HOME>/faith/vulnerability_localization`, or
`~/.cache/faith/vulnerability_localization` when both variables are unset.
Choose a writable location with space for repository archives. This cache is
separate from `--datastore-location`; moving result logs does not move payloads.

A local `manifest_path` replaces only the manifest download. Missing repository
snapshots still require network access. Reuse only a trusted cache: existing
cache files are reused without rechecking their content hash. Do not treat a
populated cache as proof that an offline deployment is complete or intact.
