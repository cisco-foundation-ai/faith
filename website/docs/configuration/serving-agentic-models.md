---
id: serving-agentic-models
title: Serve a model for agentic runs
sidebar_label: Serve a model
sidebar_position: 2
---

# Serve a model for agentic runs

Use a separately served local model through FAITH's `openai` engine. This guide
starts a tool-calling endpoint, runs one agentic task, and checks the result.
FAITH's in-process `vllm` engine does not run agentic benchmarks.

## Prerequisites

- A model server with a supported inference environment, model weights, and GPU
  drivers. Installing FAITH does not install the server executable or drivers.
- A FAITH runner with `.[openai,docker,metrics]` and a running Docker daemon for
  the example below. Complete the [Docker preflight](./environment-configs.md#docker-preflight)
  and check [task-download access](./environment-configs.md#runner-network-and-cache).
- An API address reachable from the runner. The server and runner may be separate
  processes or separate machines.

For a hosted endpoint instead, use the
[agentic quickstart](../getting-started/quickstart.md#run-an-agentic-benchmark).

## 1. Start the model server \{#serve-it-with-tool-calling-enabled}

FAITH reads tool requests from the response's `tool_calls` field, not from
tool-like text in `content`. For vLLM, enable automatic tool choice and select a
parser compatible with the model. For example:

```shell
vllm serve Qwen/Qwen3-8B --port 8100 \
  --enable-auto-tool-choice --tool-call-parser hermes
```

Parser selection depends on the model and vLLM version. Consult
[vLLM's tool-calling reference](https://docs.vllm.ai/en/stable/features/tool_calling/)
for supported parsers and chat-template requirements. Other OpenAI-compatible
servers have their own configuration requirements.

:::warning
A successful HTTP response does not establish tool-calling support. If an
endpoint returns tool requests only as text in `content`, FAITH cannot execute
them. Tasks that repeatedly receive no structured tool calls end with
`no_tool_calls` and remain included in scoring.
:::

## 2. Run one task through the endpoint \{#point-the-run-at-your-server}

Select `--model-engine openai` and set `base_url` to the server's API address:

```shell title="Local agentic smoke query"
faith query \
  --model-paths Qwen/Qwen3-8B@name=qwen3-8b \
  --model-engine openai \
  --engine-kwargs base_url='"http://localhost:8100/v1"' \
  --benchmarks vulnerability-localization-a \
  --prompt-format chat \
  --generation-mode chat_comp \
  --max-completion-tokens 2000 \
  --num-trials 1 \
  --sample-size 1 \
  --datastore-location out/agentic-local
```

No API key is needed for a keyless local server: FAITH supplies a placeholder when
`base_url` is set and `OPENAI_API_KEY` is unset. A
[model config file](./model-configs.md) can carry the `base_url` and generation
parameters instead, so the command reduces to `--model-configs`.

`localhost` refers to the FAITH runner's network namespace, not the agent's
workspace container. A runner on another host or in a pod must use an API
address reachable from that runner. Keep model credentials on the runner; they
do not need to be passed into the agent workspace for inference.

:::tip
Set `--max-completion-tokens` high enough for the model to complete any reasoning
and emit a tool call. A response truncated before a tool call counts as a turn
without tool calls. See
[troubleshooting](../troubleshooting.md#agentic-runs).
:::

## 3. Grade and inspect the result

`query` saves the task trajectory. Grade it and report the score, execution
errors, and termination reason:

```shell title="Local agentic smoke results"
faith eval --experiment-path out/agentic-local
faith summarize --experiment-path out/agentic-local \
  --stats mean_file_f1 error_rate stopped_reason
```

Expect one evaluated task and no recorded execution error. A low F1 score alone
is not a setup failure. Inspect the saved trajectory if the model did not submit
an answer, and verify the [result completeness checks](../cli/eval.md#checking-result-completeness)
before increasing the sample size.

## Endpoint diagnostics

A probe warning about truncation means tool-calling support remains unverified,
even if the run continues. Use the details below or the
[agentic troubleshooting entries](../troubleshooting.md#agentic-runs) to diagnose
probe failures and tasks that produce no tool calls.

### FAITH probes before it runs

<details>
<summary>Probe outcomes and completion-token budgets</summary>

Before starting agentic tasks, `faith query` sends a probe request with a test
tool and `tool_choice: "required"`. A returned tool call confirms that the
endpoint can produce the response structure the agent loop requires. It does
not guarantee that the model will choose to call tools during a task, where
FAITH uses `tool_choice: "auto"`.

A response without a tool call fails the probe unless it reports token-limit
truncation. For example:

```text
ValueError: Agentic benchmarks require an endpoint that can emit tool calls, but
'Qwen/Qwen3-8B' returned none when required to (finish_reason=stop). Serve the
model with tool calling enabled (for vLLM: --enable-auto-tool-choice and a
--tool-call-parser suited to the model), and check that the model's chat template
renders tools, tool_calls, and role: "tool" — a template without them cannot emit
a tool call under any parser.
```

The error identifies the model path used for the request, not its `@name=` log
alias.

The probe sets `max_completion_tokens` to `1,024`. Task requests use the resolved
generation settings independently; `max_completion_tokens` defaults to `500`
when neither a model configuration nor a CLI override supplies another value.
The task budget can therefore be smaller or larger than the probe budget.

Both limits are per response, not per task. FAITH sends the resolved temperature,
top-p, and generation kwargs to the probe as well as task requests; the probe
replaces only the completion limit and tool-selection fields. A passing probe
does not validate later, longer conversation histories. See
[token limits](./model-configs.md#token-limits-and-generation-modes) and
[request settings](./model-configs.md#engine-options-and-agentic-requests).

If the probe reaches its token limit without producing a tool call
(`finish_reason: "length"`), FAITH logs a warning and continues. Tool-calling
support remains unverified. Check the endpoint's tool behavior and the task's
completion-token budget before interpreting evaluation results.

</details>

### The model also has to support tool calling

<details>
<summary>Check the model revision and chat template</summary>

The model and its chat template must support the full tool-calling exchange.
Inspect the template used by the server, commonly in `tokenizer_config.json` or
`chat_template.jinja`, for these requirements:

1. The template accepts a `tools` variable.
2. It renders an assistant turn's `tool_calls`.
3. It handles `role: "tool"` messages.

Verify the specific model revision and template in the deployment. A template
that omits tool results prevents the model from using command output on later
turns, even if the endpoint can produce an initial tool call.

Use a template compatible with the model and record any override as part of the
deployment configuration. Substituting a template from an unrelated model can
change the evaluation behavior.

</details>
