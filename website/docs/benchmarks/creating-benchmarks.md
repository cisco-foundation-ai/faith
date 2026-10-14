---
id: creating-benchmarks
title: Benchmark configuration reference
sidebar_label: Configuration reference
sidebar_position: 2
---

# Benchmark configuration reference

For a step-by-step example, choose the [Q&A walkthrough](./creating-qa-benchmarks.md)
or the [agentic walkthrough](./creating-agentic-benchmarks.md). This page documents
shared metadata, data preparation, agentic settings, tools, and execution limits.

To run a new benchmark with FAITH, construct a local folder for your benchmark.
This folder contains a `benchmark.yaml` file defining the benchmark, along with
any required resources (e.g. data folders or auxiliary configurations). Point a
run at it with `--custom-benchmarks`.

:::note
`--custom-benchmarks` takes plain paths — it does **not** accept `@key=value`
annotations the way `--model-paths` does. The benchmark's name comes from
`metadata.name` in its `benchmark.yaml`.
:::

## Which kind of benchmark

A benchmark declares its kind with exactly one of four sections. Declaring more
than one is a config error.

| Section | The benchmark is | Each sample is |
| --- | --- | --- |
| `mcqa_config` | Multiple choice | A question and its answer choices |
| `saqa_config` | Short answer | A question and the answer text |
| `laqa_config` | Long answer | A question, graded free-form |
| `agentic_config` | [Agentic](#agentic-benchmarks) | A task worked in a sandboxed environment |

The first three are question-answering kinds: they read a dataset and render a
prompt from it. Agentic benchmarks instead use a task loader to provide prompts,
labels, and optional payloads. They do not use the Q&A `source` or `format`
sections; each task runs in an environment until the agent submits or execution
terminates.

That choice decides which of the other sections apply:

| Section | Q&A | Agentic |
| --- | --- | --- |
| `metadata` | Yes | Yes |
| `output_processing.score_fns` | Yes | Yes — at least one required |
| `output_processing.primary_metric` | Yes | Yes |
| `output_processing.answer_formats` | Yes | **Rejected** |
| `source` | Yes | **Rejected** |
| `format` | Yes | **Rejected** |

An agentic benchmark's tasks, prompts, and predictions all come from
`agentic_config`, so setting a rejected section is an error rather than something
ignored — the run stops when the config is parsed.

## What every benchmark declares

### Benchmark metadata

| Field | Used by FAITH | Purpose |
| --- | --- | --- |
| `state` | **Yes** | Whether the benchmark can be selected, and by which groups. See [states](#benchmark-states). Defaults to `enabled`. |
| `categories` | **Yes** | Tags that determine group membership — `security` feeds `all-security`, `reasoning` feeds `all-reasoning`. |
| `name` | **Yes** | Identifier used to select the benchmark and name its results directory. Experiment construction requires a nonempty name. |
| `description` | No | What the benchmark's data covers and why. |
| `license` | No | Any license associated with the benchmark. |
| `urls` | No | Source locations / documentation. |

Only `name`, `state` and `categories` change FAITH's behavior; the rest are
informational and describe the benchmark to users.

#### Benchmark states

| State | In `--benchmarks` | In `all` groups (Q&A only) | As a `--custom-benchmarks` folder |
| --- | --- | --- | --- |
| `enabled` | Yes | Yes | **Yes** |
| `experimental` | Yes, by exact name | No | No — silently skipped |
| `disabled` | No | No | No — silently skipped |
| `test_only` | No | No | No — silently skipped |

:::warning
For a custom benchmark, use `enabled` (or omit `state`, which defaults to it).
Any other state causes the folder to be skipped without an error; no experiment
is created from that folder.
:::

Only the group column depends on the kind, and it does so regardless of state:
**no group ever expands to an agentic benchmark**. `all`, `all-security`, and
`all-reasoning` cover the question-answering benchmarks only, so an agentic one
must be selected by exact name. Agentic tasks require environment setup and may
make multiple model calls. The built-in
`vulnerability-localization-a` and `vulnerability-localization-b` are `enabled`
and individually selectable, yet absent from every group.

### Scores and the primary metric

For both kinds, `output_processing.score_fns` names the custom score functions
to compute. `primary_metric` selects a generated metric using
`<metric_name>.<trial_stat>`. A custom score's metric name includes its
within-trial aggregation prefix: the `file_f1` score produces `mean_file_f1`
and `median_file_f1`.

```yaml
output_processing:
  primary_metric: "mean_file_f1.mean"
  score_fns:
    file_f1:
      type: "file_f1"
```

Here, `mean_file_f1.mean` selects the mean across trials of each trial's mean
file F1 score. A score function's configured name must not contain a dot.
The `<trial_stat>` suffix is one of `mean`, `std`, `min`, `p_25`, `median`,
`p_75`, or `max`. Verify that the complete path exists under `stats` in the
generated `metrics.json`; a non-matching name leaves no exported metric marked
as primary. See the [Metrics reference](../analytics/metrics.md) for the
aggregation stages.

An agentic benchmark must declare at least one `score_fns` entry. The extractor
produces a prediction; the scoring configuration defines how it is evaluated.

## Question-answering benchmarks

### Required columns of the loaded data

The required columns follow from which of the three question-answering sections
the benchmark declares:

| Config | `question` | `choices` | `answer` |
| --- | --- | --- | --- |
| `mcqa_config` | Required | Required | Required — a capital letter |
| `saqa_config` | Required | — | Required — the answer itself |
| `laqa_config` | Required | — | Optional |

- **`question`** — the question only, without answer choices.
- **`choices`** — a list of strings for the multiple-choice options.
- **`answer`** — for multiple choice, a single capital letter indexing into
  `choices` by their implied order; for short answer, the expected answer text.

All three categories also accept an optional `subject` column; it is carried
through to the graded records and made available to the prompt templates.

### Modifying data with `dataframe_transform_expr`

The `dataframe_transform_expr` field in `source.options` lets you transform the
loaded dataset into the required column format. It is a Python expression that
receives a pandas DataFrame `df` and a boolean `is_dev`, and must return a
transformed DataFrame.

The same expression is applied to each split the source declares: once to the
main data with `is_dev = False`, and — if the source sets a `dev_split` — again
to that split with `is_dev = True`. The dev split supplies the in-context
examples for [`--n-shot`](../cli/query.md), so branch on `is_dev` only when the
two splits need different handling.

Rename columns and drop duplicate rows:

```yaml
source:
  options:
    dataframe_transform_expr: |
      df.rename(columns={"Description": "question", "GT": "answer"}).drop(
          [243, 419],
      )[["question", "answer"]]
```

Convert numerical answers to letter choices:

```yaml
source:
  options:
    dataframe_transform_expr: |
      df.assign(
          answer=[chr(65 + x) for x in df["answer"].tolist()],
      )[["question", "subject", "choices", "answer"]]
```

The final DataFrame must contain the required columns for the benchmark type
(see [Required columns](#required-columns-of-the-loaded-data) above).

### Running a Q&A benchmark

Follow the [Q&A authoring walkthrough](./creating-qa-benchmarks.md) for a complete
local dataset, benchmark definition, validation step, and bounded evaluation.
The [Quickstart](../getting-started/quickstart.md) covers local-GPU and API model
selection.

## Agentic benchmarks

An `agentic_config` section declares a benchmark whose tasks are executed by
a tool-using agent in a sandboxed environment:

| Section | Purpose |
| --- | --- |
| `task_loader` | Where the tasks come from, and how their data is fetched |
| `environment` | The machine the agent works in |
| `answer_extractor` | How a prediction is read out of the agent's trajectory |
| `working_subdir` | The subdirectory of the workspace the agent works from |
| `read_only_workspace` | Require filesystem protection of benchmark inputs; defaults to `false` |
| `tools` | Which tools the agent may call |
| `system_prompt` | Standing instructions seeded ahead of the task prompt — the benchmark's role framing, method, and effort expectations |
| `limits` | The budgets that bound a task |

`task_loader`, `environment`, and `answer_extractor` each name an implementation
with `type`, plus a `config` dict passed to it:

```yaml
agentic_config:
  task_loader:
    type: "vulnerability_localization"
    config:
      phase: "a"
  environment:
    type: "docker"
    config:
      agent_dockerfile: "$BENCHMARKS_ROOT/vulnerability-localization/Dockerfile"
      memory_limit: "4g"
      cpu_limit: 2.0
  answer_extractor:
    type: "vulnerability_localization"
  working_subdir: "repo"
  read_only_workspace: true
  tools:
    - "terminal"
    - "submit_vulnerable_files"
    - "submit_no_vulnerability_found"
  limits:
    max_tool_calls: 15
    max_turns: 20
    timeout_seconds: 1800
    command_timeout_seconds: 10
```

`system_prompt` is a plain string, not a template: an agentic task has no render
variables to substitute, and per-task text belongs in the loader, where the row
data is. Omit it or leave it empty to send no system message.

The workspace is always `/workspace`. `working_subdir` names a directory
*inside* it for a benchmark whose data unpacks into one — vulnerability
localization sets `repo`, so the agent works from `/workspace/repo`. Omit it to
work from the workspace root. It is declared here rather than under `environment`
because the loader names the same directory in the prompt it writes and the
environment runs the agent's commands there, and the two must agree. A value
that escapes the workspace — absolute, or climbing out with `..` — is rejected
when the config is parsed.

:::note
Every `type` is a **registered short name**, not an import path. Names resolve
case-insensitively — `Docker` and `docker` are the same backend. Adding a loader,
environment, extractor, or tool means registering it in
`src/faith/agentic/registries.py`. Environment and tool names are checked during
configuration validation. Loader names are resolved when tasks are loaded;
extractor names are resolved when grading begins. Parse-only validation does
not detect every missing registration.
:::

### Custom implementation contracts

Custom Python components must be installed with FAITH; `--custom-benchmarks`
does not import Python files from benchmark folders. See
[Custom agentic components](../reference/agentic-components.md) for interfaces,
payload identity, registration, and tool validation.

### Minimal custom agentic example

<span id="1-define-the-task-data" />
<span id="2-implement-and-register-the-loader-and-extractor" />
<span id="3-configure-and-validate-the-benchmark" />
<span id="4-query-evaluate-and-summarize" />

The [agentic authoring walkthrough](./creating-agentic-benchmarks.md) builds a
one-task benchmark using `bash_exec`, `submit`, and a read-only Docker workspace.
It includes task data, registered Python components, a complete YAML definition,
and query, evaluation, and summary commands.

### Read-only workspaces

Set `agentic_config.read_only_workspace: true` when a benchmark requires its
input workspace to remain unchanged during agent execution. The default is
`false`; both vulnerability-localization phases enable it.

```yaml
agentic_config:
  read_only_workspace: true
```

This requirement is declared outside `environment`. Switching between Docker
and Kubernetes, or supplying `--environment-config`, cannot disable it. Enabling
the policy changes the experiment identity, so previously recorded writable runs
are not resumed as protected runs.

Trusted setup prepares the task data, then protects `/workspace` before the
first agent command. Protection prevents the agent from creating, editing,
deleting, renaming, or changing permissions on workspace files and directories.
Agent commands run as a non-root user without Linux capabilities or privilege
escalation, and subsequent uploads are disabled. FAITH rejects an unsupported
environment or fails setup if protection cannot be established. Recovery must
prepare and protect each replacement environment before retrying a command.

This is filesystem enforcement, not a command allowlist. Pipes, shell syntax,
`rg`, `find`, and `sed` remain available. Scratch locations outside the workspace,
including `/tmp`, may remain writable. The policy does not restrict networking,
prevent denial of service, or provide a hardened sandbox for hostile code.
Images, runtime helpers, and infrastructure remain trusted.

Regular files, directories, and internal relative symlinks are supported.
Escaping or cyclic symlinks and special files are rejected; the Kubernetes
helper also rejects hard-linked files. Custom images must not start untrusted
processes before protection is established.

See [environment requirements](../configuration/environment-configs.md#read-only-workspace-requirements)
for deployment details and
[`SECURITY.md`](https://github.com/cisco-foundation-ai/faith/blob/main/SECURITY.md#running-agentic-benchmarks)
for ownership and security responsibilities.

### Environments and tools

Two environments are registered: `docker`, which runs each task in a container,
and `kubernetes`, which runs each task in an Agent Sandbox on a cluster. Install
the matching [extra](../getting-started/installation.md#available-extras).

A Docker environment selects an image through one of two keys. Setting both in
the same authored configuration is rejected:

| Key | Meaning |
| --- | --- |
| `agent_dockerfile` | Build an image tagged `faith-build/<context>:<digest of the build context>`. |
| `agent_image` | Use a prebuilt image by name; pull it if it is not available locally. |

For backend configuration and deployment overrides, see
[environment config files](../configuration/environment-configs.md).

#### Registered tools

Tools declare their required environment capabilities. Configuration validation
rejects an environment that cannot provide them before setup begins.

| Tool | Purpose |
| --- | --- |
| `bash_exec` | Execute a shell command; general-purpose starter tool |
| `submit` | Submit a string answer and end the task; general-purpose starter tool |
| `terminal` | Execute a shell command using the vulnerability-localization tool name |
| `submit_vulnerable_files` | Submit a list of vulnerable files and end the task |
| `submit_no_vulnerability_found` | Submit a no-vulnerability answer and end the task |

A benchmark may register custom tools instead. At least one enabled tool must
be terminal: a successful submission ends the task and supplies its answer.

### Limits

Configure per-task limits for the benchmark's expected workload. Defaults are:

| Field | Default | What it bounds |
| --- | --- | --- |
| `max_tool_calls` | `50` | Non-terminal tool-call budget; invalid JSON and unknown tool names are uncharged |
| `max_turns` | `100` | Model turns before the turn limit, excluding the additional submission turn described below; must be `>=` `max_tool_calls` |
| `max_consecutive_no_tool_calls` | `3` | Prose-only turns in a row before the run ends |
| `timeout_seconds` | unset | Wall-clock seconds per task; unset means unbounded |
| `command_timeout_seconds` | `60` | Seconds any one agent command may run; `0` for no bound |
| `recover_on_environment_failure` | `True` | Rebuild an unavailable sandbox and retry the same tool call; see [state-loss and retry restrictions](#sandbox-recovery) |

`max_turns` also bounds turns that do not consume the tool-call budget, such as
turns containing invalid JSON or unknown tool names. Configuration validation
requires `max_turns >= max_tool_calls` before any model request.

Invalid JSON and unknown tool names are rejected before execution and do not
consume `max_tool_calls`. Current limitation: argument-validation errors returned
by a known non-terminal tool, including missing required arguments, do consume
this budget even though the requested operation did not execute.

`max_consecutive_no_tool_calls` counts consecutive model turns without tool
calls. FAITH prompts the model to use a tool after earlier such turns and stops
at the configured limit. Set it to `1` to stop after the first turn without a
tool call.

`timeout_seconds` bounds task duration at checks between turns. It is unset by
default; vulnerability localization sets `1800`. It does not interrupt a model
request or tool call already in progress.

`command_timeout_seconds` independently bounds each agent command using GNU
coreutils `timeout` inside the execution environment. Set
`agentic_config.limits.command_timeout_seconds` to `0` to disable this command
bound. It is a benchmark limit, not a deployment-specific environment setting.

#### Sandbox recovery

`agentic_config.limits.recover_on_environment_failure` defaults to `true`. When
a tool raises `EnvironmentUnavailableError`, FAITH tears down the unavailable
environment, repeats setup for the same task in a replacement environment,
including any required workspace protection, and retries the same tool call.
The model is not notified, and the failed attempt does not consume an additional
tool execution from the `max_tool_calls` budget.

:::warning[Recovery rebuilds the sandbox; it does not restore task state]
The model conversation survives, but agent-created or modified files and process
state in the old sandbox do not. Setup prepares the original task workspace,
not a snapshot of the agent's progress.

Disable recovery for benchmarks that depend on agent-created or modified files
or other sandbox state. Also disable it for commands with non-repeatable external
side effects: a lost response does not prove that a command never executed, so
retrying can repeat an external action.
:::

Recovery is suitable for read-only, safely repeatable tasks such as the supplied
vulnerability-localization benchmarks. To disable it, set this field in the
benchmark's existing `benchmark.yaml`, not an `--environment-config` file:

```yaml title="benchmark.yaml (fragment)"
agentic_config:
  limits:
    recover_on_environment_failure: false
```

With recovery disabled, an unavailable environment fails the task as
`environment_error` instead of rebuilding it. The same outcome applies if
recovery fails or exhausts its bounds; these tasks are excluded from scoring.
Recovery permits at most two consecutive rebuilds and three in total, with the
task timeout checked before each attempt. These rebuild counts are not
configurable.

Recovery attempts are logged as warnings, not added to the trajectory as model
turns or tool calls. Search runner logs for `Rebuilding environment` when
diagnosing environment failures.

This differs from [completed-task checkpoints](../analytics/storage.md#agentic-task-checkpoints),
which preserve completed trajectories for reuse after a runner interruption.
Neither mechanism restores a partially completed sandbox.

#### Final submission turn

After `max_tool_calls` is exhausted, FAITH can offer one additional model turn
with only terminal tools so the agent can submit an answer. The tool-budget
check precedes the turn-limit check, so this submission turn can exceed
`max_turns` by one. The task timeout still takes precedence. Reaching `max_turns`
without exhausting the tool budget does not grant an additional turn.

For example, with `max_tool_calls: 1` and `max_turns: 1`, an agent that uses its
first turn to execute a non-terminal tool can receive a second turn to submit.
A successful submission is scored normally.

:::warning
If a system prompt quotes a limit to the agent, the two have to be changed
together. Vulnerability localization's prompt tells the agent it has "up to 15
terminal calls",
matching `max_tool_calls: 15`; nothing checks that they agree.
:::

To run the benchmark you've authored, see the
[agentic walkthrough](../getting-started/quickstart.md#run-an-agentic-benchmark);
for the flags `query` treats differently, see
[agentic runs](../cli/query.md#agentic-runs).
