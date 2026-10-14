---
id: troubleshooting
title: Troubleshooting & FAQ
sidebar_label: Troubleshooting
sidebar_position: 98
---

# Troubleshooting & FAQ

Expand the entry matching the observed symptom. For issues not covered here,
contact
[foundation-ai-oss@cisco.com](mailto:foundation-ai-oss@cisco.com).

## Installation

<details>
<summary>A command fails with a missing-dependency error</summary>

**Fix:** install the extra that command needs, e.g.
`uv pip install -e '.[vllm,metrics]'`.

FAITH's dependencies are split into [extras](./getting-started/installation.md#available-extras)
so you only install what you use. Which one you need depends on the failure:

| Symptom | Extra |
| --- | --- |
| `query` cannot import an engine | `vllm`, `openai`, `openrouter`, or `sagemaker` |
| `summarize` cannot print a table | `metrics` (supplies the `tabulate` backend) |
| `summarize` reports `BigQuery dependencies not installed` | `bigquery` |
| `The 'docker' package is required for DockerEnvironment` | `docker` |
| `The 'k8s-agent-sandbox' package is required for KubernetesEnvironment` | `kubernetes` |

The last two arrive later than the rest. FAITH's pre-run agentic checks construct
no environment, so a missing extra is not detected by those checks: it raises when the
first trial builds its environment, after the endpoint has been probed for tool
calling. Install the extra your benchmark's
[environment](./configuration/environment-configs.md) needs before starting a
long run.

</details>

<details>
<summary>Installation selects an unsupported Python version or cannot install vLLM</summary>

Create the environment with `uv venv --python 3.12 .venv`; FAITH requires Python
3.11 or 3.12. Do not replace the operating system's interpreter. If an existing
environment uses a different version, preserve it and create a separate
environment with a supported interpreter.

For API-based runs, install the API and environment extras you need rather than
`all` or `test`, which also pull in the GPU inference stack. The packaged vLLM
extra is not a general-purpose macOS or CPU installation. See the
[installation prerequisites](./getting-started/installation.md#prerequisites).

</details>

## Models & engines

<details>
<summary>vLLM runs out of GPU memory</summary>

**Fix:** shard across more GPUs with `--num-gpus <N>`, or lower
`--model-context-len` (default `3500`).

The model is too large for the available VRAM. In order of what to try first:

- Shard across more GPUs with `--num-gpus <N>`.
- Lower `--model-context-len` — a smaller context uses less memory.
- Quantize the model with an engine kwarg, e.g.
  `--engine-kwargs quantization='"bitsandbytes"'`.
- Use a smaller model, or switch to an API engine (`openai` / `openrouter`),
  which needs no local GPU.

</details>

<details>
<summary>API authentication errors (OpenAI / OpenRouter / SageMaker)</summary>

**Fix:** export the credential your engine expects.

| Engine | Credential |
| --- | --- |
| `openai` | `OPENAI_API_KEY` |
| `openrouter` | `OPENROUTER_API_KEY` |
| `sagemaker` | AWS credentials + a region (`AWS_REGION` or the `aws_region` engine kwarg) |

Provide the credential to the runner's environment using your approved secret
mechanism, then restart the process if necessary. Do not print keys in diagnostic
output or commit them to configuration files. For a local OpenAI-compatible
server, check the `base_url` and the server's own authentication requirements.

</details>

<details>
<summary>Engine kwargs are being rejected or misparsed</summary>

**Fix:** escape-quote string values — `key='"value"'`, not `key=value`.

`--engine-kwargs` and `--generation-kwargs` values are parsed with
`ast.literal_eval`, so a bare word is read as a Python name rather than a
string:

```shell
# right — the string is wrapped in single + double quotes
--engine-kwargs quantization='"bitsandbytes"'
# wrong — parsed as a bare name, not a string
--engine-kwargs quantization=bitsandbytes
```

Numbers and booleans (`5`, `True`) are written plainly.
For a vLLM dtype, use `dtype='"bfloat16"'`, not `dtype=torch.bfloat16`.
The latter is a Python attribute expression, not a literal.

</details>

<details>
<summary>A generation kwarg raises <code>multiple values for keyword argument</code></summary>

Set `temperature`, `top_p`, and `max_completion_tokens` through their dedicated
YAML fields or CLI flags, not inside `generation.kwargs`. Q&A passes these as
explicit arguments, so duplicate kwargs can raise before a model request is
sent. The agentic path handles duplicates differently; do not rely on that
behavior for a configuration shared between Q&A and agentic benchmarks.

Likewise, keep engine fields such as `num_gpus` out of `engine.kwargs`, and do
not put runner-owned request fields (`model`, `messages`, `tools`, or
`tool_choice`) in agentic generation kwargs. See
[configuration precedence](./configuration/model-configs.md#configuration-precedence).

</details>

<details>
<summary>An OpenAI engine timeout setting has no effect</summary>

The FAITH `openai` wrapper does not pass arbitrary engine kwargs to the SDK.
`timeout` and `max_retries` in `engine.kwargs` are currently ignored. The
supported FAITH retry controls are `api_max_attempts` and
`api_retry_sleep_secs`; these are not HTTP timeouts or SDK retry settings.

An agentic task's `timeout_seconds` is checked between turns. It does not cancel
an in-flight request or bound the duration of a retry sequence. See
[engine options and agentic requests](./configuration/model-configs.md#engine-options-and-agentic-requests).

</details>

<details>
<summary><code>Unknown annotation keys</code> / <code>Annotations must be in the form of key=value</code></summary>

**Fix:** check the key is accepted by *that* flag — `--model-configs` takes
only `num_gpus`.

Inline annotations chain onto a path with `@`, and each one must contain an `=`:

```shell
--model-paths meta-llama/Llama-3.1-8B-Instruct@name=llama3-8b@num_gpus=2
```

The accepted keys differ between the two model flags, which is the usual cause
of `Unknown annotation keys`:

| Flag | Accepted annotations |
| --- | --- |
| `--model-paths` | `name`, `num_gpus`, `reasoning`, `response_pattern`, `tokenizer` |
| `--model-configs` | `num_gpus` only |

Everything else for a packaged or YAML config is set inside the config file
itself — see [Model config files](./configuration/model-configs.md).

</details>

## Running benchmarks

<details>
<summary>My results vary between runs</summary>

**Fix:** check that nothing raised `--temperature` above its default of `0.0`.

Sampling above `0.0` is the usual cause. Note that a packaged
[model config](./configuration/model-configs.md) sets its own temperature — `0.3`
for instruct models, `0.6` for reasoning ones — and `--temperature` overrides a
config only when passed explicitly.

`--seed` (default `373363`) seeds data sampling and the per-trial offset, not
generation; only the `vllm` engine receives it as a generation seed. To *measure*
variance rather than remove it, run multiple `--num-trials` — metrics are
aggregated across them (see [Trials](./reference/concepts.md#trial)).

</details>

<details>
<summary>A benchmark name is not recognized</summary>

**Fix:** check the exact identifier in the
[Benchmark catalog](./benchmarks/catalog.md).

You can also pass a [group](./benchmarks/groups.md) like
`all` or `all-security`. Experimental benchmarks are not part of any
group and must be named directly.

</details>

## Agentic runs

<details>
<summary>The Docker CLI works, but FAITH cannot reach the daemon</summary>

Run the [Docker preflight](./configuration/environment-configs.md#docker-preflight)
from the activated FAITH environment and as the same user. FAITH's Python client
reads the SDK's environment settings; selecting a Docker CLI context alone does
not configure it. Check the daemon endpoint, socket access, and TLS settings.

Installing the `docker` extra supplies only the SDK. Start the daemon if it is
not running. Do not resolve access errors by making its socket world-writable,
disabling TLS, or exposing an unauthenticated daemon port.

</details>

<details>
<summary>Vulnerability-localization setup cannot download a manifest or snapshot</summary>

Downloads run on the FAITH runner, not inside the agent container. Check the
runner's access to GitHub and its raw/archive hosts, proxy or certificate
configuration, and write permissions for the
[payload cache](./configuration/environment-configs.md#runner-network-and-cache).
Enabling Docker container networking does not correct runner download failures.

A local `manifest_path` does not supply repository archives. A content-hash
mismatch means the downloaded bytes differ from the expected corpus; do not
change the expected hash merely to make setup succeed. Preserve the error and
investigate the manifest revision and downloaded source.

</details>

<details>
<summary>Kubernetes reports a missing router or times out after a claim becomes ready</summary>

Check the [cluster prerequisites](./configuration/environment-configs.md#cluster-prerequisites).
In `tunnel` mode, `namespace` locates the warm pool, while `router_namespace`
locates `sandbox-router-svc`; these can be different. The pinned SDK
port-forwards to that router, which must then reach the sandbox on port `8888`.
Allow the router's namespace in the sandbox ingress policy and check router
egress restrictions. A ready claim does not prove that this network path works.

For `in_cluster`, the runner needs direct pod-network access and cluster DNS as
a fallback. Check Kubernetes credentials and permissions separately from the
runtime transport. If a warm pool cannot produce ready pods, inspect image-pull,
scheduling, admission-policy, and runtime readiness errors before extending the
timeout. The supplied privileged runtime helper may require an operator-approved
exception to the cluster's pod-security policy; do not remove its required
controls or weaken a shared namespace as a troubleshooting shortcut.

</details>

<details>
<summary>Every agentic task scored zero and nothing reported an error</summary>

**Fix:** read `stats.stopped_reason.total` in `metrics.json`.

Every agentic run tallies how its tasks ended, beside the scores.
`stats.stopped_reason.total` counts each reason across the whole run. Per-trial
counts are also available, under
`per_trial_metrics.<trial>.stopped_reason_count`.

| `stopped_reason` | Meaning | Where to look |
| --- | --- | --- |
| `submitted` | The agent submitted an answer | The benchmark's score and answer-format metrics |
| `no_tool_calls` | Consecutive model turns contained no tool calls, possibly because of truncation or model behavior | The entry below |
| `max_calls` / `max_turns` | The task reached its tool-call or turn limit without submitting | The benchmark's `limits` |
| `timeout` | The task reached `timeout_seconds` at a timeout check | The benchmark's `limits` |
| `error` | A model API, setup, or harness operation failed | The task's recorded `error` |
| `environment_error` | The environment was unavailable and setup or recovery did not succeed | The task's recorded `error` and infrastructure logs |

Tasks ending with `error` or `environment_error` are excluded from score
aggregation. Other termination reasons remain included, even without a
submission; both vulnerability-localization phases score a missing submission
as zero. `error_rate` reports the excluded fraction, while `stopped_reason`
reports all task outcomes, including submissions and limits. See
[how many samples the score describes](./analytics/metrics.md#how-many-samples-the-score-describes).

`metrics.json` contains termination counts, not individual failure messages.
Read the trial's `benchmark-log.json` for each task's error details; see
[an agentic record](./analytics/storage.md#agentic-records) for the fields
and where the file sits.

</details>

<details>
<summary>Every task stops with <code>NO_TOOL_CALLS</code></summary>

**Fix:** check the completion-token budget and whether the model is choosing to
call tools. The trajectory does not retain the response's `finish_reason`.

The loop stops after `max_consecutive_no_tool_calls` (default `3`) consecutive
model turns without tool calls. The pre-run probe confirms tool support only when
it returns a tool call; a truncated probe logs a warning and permits execution
with tool support unverified. Check that warning before diagnosing task behavior.
Two possible causes are:

1. **The completion-token budget is insufficient.** A model may reach its token
   limit during reasoning, before emitting a tool call. The default is `500`
   when no model configuration or CLI override supplies another value. Packaged
   [model configs](./configuration/model-configs.md) set `5000` for instruct
   models and `10000` for reasoning ones. Inspect the resolved configuration and
   increase the budget if responses are being truncated.
2. **The model chose not to call a tool.** The pre-run probe demands a call with
   `tool_choice: "required"`, while the run itself uses `"auto"` — so passing the
   probe proves the endpoint *can* emit a call, not that the model will for this
   task.

The trajectory retains each reply's `role`, `content`, and `tool_calls`, but not
`finish_reason`. An incomplete final `content` can suggest truncation; use
endpoint diagnostics or a controlled request to distinguish truncation from a
model choosing not to call tools. See the
[probe and task token budgets](./configuration/serving-agentic-models.md#faith-probes-before-it-runs).

</details>

<details>
<summary><code>Agentic benchmarks require an endpoint that can emit tool
calls</code></summary>

**Fix:** inspect the complete probe error before changing the server. A failed
probe request can indicate an invalid local request, connectivity or credential
problems, or unsupported endpoint behavior.

Before any agentic task runs, `faith query` sends a test tool with
`tool_choice: "required"`. The error distinguishes these outcomes:

| Message ends with | Meaning |
| --- | --- |
| `for one failed: <error>` | Request construction or the API call failed; inspect the underlying error for configuration, connectivity, authentication, or model-selection problems |
| `returned none when required to (finish_reason=…)` | The endpoint responded without a tool call; verify server configuration, the parser, and the model's chat template |

For the second, work through
[serving a model for agentic runs](./configuration/serving-agentic-models.md).
For vLLM, verify `--enable-auto-tool-choice`, the model-specific
`--tool-call-parser`, and a compatible chat template.

A non-truncated probe response without a tool call prevents task execution.
A truncated response is an exception: FAITH warns and continues with tool
support unverified.

</details>

<details>
<summary>The agent repeatedly retries the same command</summary>

**Fix:** verify that the command is installed in the agent image. Enabling a
tool does not install the commands it invokes.

When constructing the agent loop, FAITH checks that the environment implements
each tool's required Python interfaces. It does not inspect the image for every
command a tool might execute. For example, an image without `ripgrep` can pass
interface validation but fail when the agent invokes `rg`.

A missing command typically returns exit status 127. A nonzero shell exit is
returned as tool output, so the task can continue and consume its remaining
budget. Inspect command output before attributing a low score to the model.

No backend probes for the commands a tool runs. `kubernetes` does probe its
sandbox before any task, but only for
[the runtime contract](./configuration/environment-configs.md#what-the-sandbox-image-must-provide)
— that commands are shell-evaluated and the file API is rooted at `/workspace` —
not for every utility in the image. Execute a small test against the intended
image before starting a full evaluation.

</details>

<details>
<summary>Tasks end with <code>max_calls</code> without a submitted answer</summary>

The trajectory remains in the trial log, but it has no submitted answer. When
the tool-call budget is exhausted, FAITH can offer one final turn restricted to
terminal tools if those tools are registered and the task has not timed out.
A terminal-tool response ends the task with `submitted`; otherwise it can end
with `max_calls`.

For the built-in vulnerability-localization benchmarks, a non-error outcome
without a submission receives a zero performance score; it is not excluded as
an execution error. Inspect trajectories, tool failures, and configured
[limits](./benchmarks/creating-benchmarks.md#limits) before changing the budget.
Also check any budget stated in the system prompt; FAITH does not validate that
text against the configured limits.

</details>

<details>
<summary>A task overshot its <code>timeout_seconds</code></summary>

**Expected, by up to one turn.** `timeout_seconds` is checked *between turns*: it
does not interrupt a model call or a tool call already in flight.

To bound a single command, use `command_timeout_seconds` (default `60`, `0` for
unbounded). It runs each agent command under coreutils `timeout` inside the
container or sandbox, killing a runaway process where it runs. The two limits are
independent — see [limits](./benchmarks/creating-benchmarks.md#limits).

</details>

<details>
<summary>The agent's own files vanished partway through a task</summary>

If the benchmark depends on agent-written files or other sandbox state, set
`agentic_config.limits.recover_on_environment_failure` to `false` in `benchmark.yaml`.

Recovery is enabled by default. After an environment failure, it repeats setup
and retries the same tool call without notifying the model. The conversation
survives, but the old sandbox's file changes and process state do not. Search
runner logs for `Rebuilding environment` to confirm whether recovery occurred;
rebuild attempts are warnings, not trajectory turns or tool calls.

Disable recovery for non-repeatable external side effects as well: a command may
have executed before its response was lost. With recovery disabled, the task
fails as `environment_error` instead of rebuilding the sandbox and is excluded
from scoring. See [sandbox recovery](./benchmarks/creating-benchmarks.md#sandbox-recovery)
for configuration, retry bounds, and the distinction from completed-task checkpoints.

</details>

<details>
<summary><code>--environment-config only applies to agentic benchmarks</code></summary>

**Fix:** drop the flag, or add an agentic benchmark to `--benchmarks`.

The flag only selects where an agent runs, so a Q&A-only run has nothing to apply
it to. It is rejected rather than ignored, since accepting it would read as having
been honored.

Check whether you meant to request an agentic benchmark at all. No group expands
to one, so `--benchmarks all` alongside `--environment-config` produces exactly
this error; name the agentic benchmark directly.

</details>

<details>
<summary><code>Agentic benchmarks require the openai engine, but
model '…' uses vllm</code></summary>

**Fix:** every model in the run must use the `openai` engine — not just the one
serving the agentic benchmark.

The error names one model, but the requirement is not per-model: once any requested
benchmark is agentic, the check applies to every model spec, including one that
would only ever serve the run's Q&A benchmarks. `--prompt-format chat` and
`--generation-mode chat_comp` are required run-wide on the same basis.

So a run pairing a `vllm` model with `mmlu` and an `openai` model with
`vulnerability-localization-a` is rejected, even though that pairing would work.
Mixed engines
in one run are not supported — split it into two runs.

A locally served model reaches the `openai` engine via `base_url`; see
[which engine to use](./configuration/model-configs.md#which-engine_type-to-use).

</details>

## Results & analytics

<details>
<summary><code>No metrics.json files found in &lt;path&gt;</code></summary>

**Fix:** confirm that `--experiment-path` is a local directory containing
evaluated results. Run `eval` first if the query logs have not been graded.

`summarize` searches that directory and its subdirectories for `metrics.json`.
Each metrics file must have an `experiment.json` beside it. The full
benchmark/model directory hierarchy is not required: a directory containing
just the paired files is valid. Keep different experiments in separate
directories to avoid filename collisions.

For results stored in Cloud Storage, inspect the remote files, then
[download the metrics and experiment descriptions](./analytics/storage.md#downloading-metrics-only)
and pass the local download directory to `summarize`. Unlike the
`--datastore-location` option on `query`, the `--experiment-path` option on
`summarize` does not accept a `gs://` URI:

```shell
gcloud storage ls -r gs://<BUCKET>/<PATH>/
```

</details>

<details>
<summary><code>Missing experiment.json for &lt;path&gt;</code></summary>

**Fix:** re-run `query` for that experiment.

`experiment.json` is written only once **all** of an experiment's trials have
completed, so this usually means `query` was interrupted partway through.

</details>

<details>
<summary><code>BigQuery project not specified</code> / <code>BigQuery dataset not
specified</code></summary>

**Fix:** set both, by environment variable or by flag.

```shell
export FAITH_BIGQUERY_PROJECT=my-project
export FAITH_BIGQUERY_DATASET=faith_results
# equivalently: --bigquery-project my-project --bigquery-dataset faith_results
```

Naming a dataset does not create it. If it does not exist:

```shell
bq mk --dataset ${FAITH_BIGQUERY_PROJECT}:${FAITH_BIGQUERY_DATASET}
```

</details>

<details>
<summary>BigQuery ingestion fails with a permission error</summary>

**Fix:** authenticate for application-default credentials, and check the account
holds all three permissions.

```shell
gcloud auth application-default login
```

Ingestion creates the table on first use and then appends to it, so read access
is not enough: it needs `bigquery.tables.create`,
`bigquery.tables.updateData`, and `bigquery.jobs.create`.

</details>

<details>
<summary><code>Metrics already ingested from: …</code></summary>

The batch was rejected because its first record's `metrics_file_uri` already
exists in the table. Later URIs are not checked, so this guard does not provide
batch-wide deduplication. CLI ingestion records absolute local paths, including
for files downloaded from GCS.

Do not retry overlapping ingestion scopes or copy unchanged results to a new
path to bypass the guard. For a new grading revision, preserve the original
artifacts and use a distinct derived result location with recorded provenance.
If replacing existing rows is necessary, first verify the exact stored URI and
preserve the previous values; replacement requires a deliberate database
operation, not an automatic CLI retry. See
[ingestion limits](./analytics/bigquery/quickstart.md#3-ingest-to-bigquery).

</details>

## Still stuck?

- Re-read the relevant [CLI command](./cli/overview.md) page for exact flags
  and defaults.
- Check [Core concepts](./reference/concepts.md) if a term is unclear.
- Email [foundation-ai-oss@cisco.com](mailto:foundation-ai-oss@cisco.com).
