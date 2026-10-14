---
id: sandbox-runtime
title: Sandbox runtime API
sidebar_label: Sandbox runtime API
---

# Sandbox runtime API

This reference is for developers implementing a custom Kubernetes sandbox
runtime. For cluster setup, use [Kubernetes deployment](../configuration/kubernetes-deployment.md).
The supplied helper implements this contract; ordinary Docker users do not
need to implement these APIs.

Preserve the [deployment security controls](../configuration/kubernetes-deployment.md#deployment-controls)
and the benchmark's [read-only workspace requirements](../configuration/environment-configs.md#read-only-workspace-requirements).
Workspace protection is not a hardened sandbox for hostile code.

## Command and file API \{#what-the-sandbox-image-must-provide}

The sandbox runtime serves the SDK's `commands.run` and `files.write` operations.
Creation, readiness, and termination use the Kubernetes API. Runtime requirements
are:

- Evaluate command arguments as shell commands. FAITH sends wrappers such as
  `cd <dir> && timeout -k 5 <n> bash -c '<command>'`; direct execution of
  whitespace-separated arguments does not satisfy this contract.
- Resolve relative upload paths against `/workspace` during trusted preparation.
  This path is part of the task contract and is not configurable.
- Provide `bash`, GNU coreutils `timeout`, `tar`, `unzip`, and `rm`, executable
  by the user running preparation and agent commands. A runtime command
  allowlist must permit the preparation operations as well as agent tools.
- Support the [workspace policy API](#custom-runtime-workspace-policy-api) when
  the benchmark requires read-only inputs.

During setup, FAITH uploads a marker and reads it through a bounded shell command.
This checks shell evaluation, upload path resolution, and the command wrapper.
Read-only benchmarks perform additional protection checks after preparation.
Incompatible runtimes fail setup rather than produce results from an invalid
environment.

## Workspace policy API \{#custom-runtime-workspace-policy-api}

After trusted preparation, FAITH sends:

```http
POST /faith/workspace/read-only
Content-Type: application/json

{"version": 1, "workspace": "/workspace"}
```

The runtime must return HTTP `200` with exactly this JSON object:

```json
{"version": 1, "read_only_workspace": true, "uploads_disabled": true}
```

Protection is irreversible for that workspace instance. A repeated request may
confirm an already sealed workspace with the same identity; there is no unseal
operation. Subsequent `/upload` requests must return HTTP `403` before writing,
including uploads that were queued before protection completed.

Serialize command execution, upload writes, and sealing across runtime workers.
Check policy after acquiring the shared lock. Persist the workspace-container
identity and the `sealing` state before changing permissions; record `sealed`
only after protection succeeds. State updates must be atomic and retained across
helper restarts. The supplied helper uses the root-only policy volume and a
filesystem lock.

| Policy state | Required behavior |
| --- | --- |
| Unsealed | Allow trusted preparation |
| Sealing interrupted or state invalid | Refuse execution; replace the sandbox |
| Sealed, with unchanged workspace identity | Execute commands unprivileged; reject uploads with HTTP `403` |
| Workspace-container identity changed | Refuse execution; replace the sandbox and repeat setup |

Normal files, directories, and internal relative symlinks are supported.
Escaping or cyclic symlinks and special files are rejected. The Kubernetes helper
also rejects hard-linked files.

FAITH checks the response object, inspects permissions and agent privileges
through the shell, and attempts an upload that must return HTTP `403`. Returning
the expected JSON alone is insufficient. Missing or incompatible support fails
setup. The supplied helper returns HTTP `422` for an unsupported request version
or workspace, and HTTP `409` for protection failures or incompatible policy state.
