---
id: model-configs
title: Model config files
sidebar_label: Model configs
sidebar_position: 1
---

import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';
import CodeBlock from '@theme/CodeBlock';
import {TwoPaneConfig} from '@site/src/components/ModelConfig/TwoPaneConfig';
import {DEFAULTS} from '@site/src/components/ModelConfig/configs';

# Model config files

Use `--model-configs` to select a YAML configuration for each model. This allows
models in the same run to use different engines and generation settings.
Explicit CLI overrides still apply to every selected configuration, subject to
the precedence rules below.

Each YAML config file must have a top-level `model:` key. Three fields are
required — `model.path`, `model.prompt_format`, and
`model.engine.engine_type` — and all others have defaults.

## Config file schema

```yaml
model:
  name: "" # log name (defaults to the full path with unsafe characters replaced by _)
  path: "" # model path or name [REQUIRED]
  prompt_format: "" # base | chat [REQUIRED]
  response_pattern: null # Q&A answer-extraction regex used during eval
  tokenizer: null # custom tokenizer for in-process vLLM only
  engine:
    engine_type: "" # openai | openrouter | sagemaker | vllm [REQUIRED]
    num_gpus: 1 # in-process vLLM tensor-parallel GPU count
    context_length: 3500 # in-process vLLM context limit
    kwargs: {} # additional arguments accepted by the FAITH engine wrapper
  generation:
    temperature: 0.0 # sampling temperature (default: 0.0)
    top_p: 1.0 # top-p sampling (default: 1.0)
    max_completion_tokens: 500 # completion limit per response, not per agentic task
    kwargs: {} # additional keyword arguments for generation
  reasoning: # answer-extraction delimiters for in-process vLLM only (optional)
    start_delimiter: "<think>"
    end_delimiter: "</think>"
```

## Configuration precedence

FAITH resolves YAML includes first, then applies explicit CLI overrides. Omitted
CLI flags do not replace values from the configuration with CLI defaults.

| Setting | Resolution with `--model-configs` |
| --- | --- |
| Engine type | `model.engine.engine_type`; `--model-engine` does not override it. |
| Prompt format | Explicit `--prompt-format`, otherwise `model.prompt_format`. |
| GPU count | Per-config `@num_gpus=N`, then explicit `--num-gpus`, then the YAML value. See the [GPU-count caveats](../cli/query.md#model-flags). |
| Context and generation fields | Explicit `--model-context-len`, `--temperature`, `--top-p`, and `--max-completion-tokens` replace the corresponding YAML fields. |
| Engine and generation kwargs | CLI kwargs are merged into the corresponding YAML `kwargs` mapping; the CLI value replaces an existing key. This merge is shallow. |

For example, this command selects a packaged configuration, overrides its
temperature, and adds a vLLM sampling option. The per-config GPU annotation takes
precedence over the global GPU flag:

```shell title="Model config overrides"
faith query \
  --model-configs instruct/phi-4@num_gpus=2 \
  --num-gpus 1 \
  --temperature 0.0 \
  --generation-kwargs top_k=20 \
  --benchmarks ctibench-mcqa \
  --sample-size 25 \
  --num-trials 1 \
  --datastore-location out/model-config
```

This example requires a supported vLLM installation and sufficient GPU memory;
it uses two GPUs. A mixed Q&A and agentic run must use an `openai` engine for
every selected model, so the vLLM configuration above is Q&A-only.

The `!from` mapping form recursively merges YAML overrides into an included
mapping. This differs from CLI kwarg merging: a CLI dictionary value replaces
the entire value at that key rather than merging its nested fields. Relative
include paths are resolved from the including YAML file.

:::caution
Use the dedicated generation fields for `temperature`, `top_p`, and
`max_completion_tokens`; do not repeat them in `generation.kwargs`. Q&A calls
can fail with duplicate-keyword errors, while the agentic path resolves these
duplicates differently. Likewise, do not repeat engine-construction arguments
such as `num_gpus` or `context_len` in `engine.kwargs`.
:::

## Examples

<Tabs groupId="model-engine" queryString="model-engine">

<TabItem value="vllm" label="vLLM" default>

Local Hugging Face model on GPU, with reasoning delimiters supplied as token IDs.

```yaml title="Local vLLM model"
model:
  name: "gpt-oss-20b"
  path: "openai/gpt-oss-20b"
  prompt_format: "chat"
  engine:
    engine_type: vllm
    context_length: 8192
  generation:
    temperature: 0.6
    max_completion_tokens: 4096
  reasoning:
    start_delimiter: [200006, 173781, 200005, 35644, 200008]
    end_delimiter: [200007, 200006, 173781, 200005, 17196, 200008]
```

</TabItem>

<TabItem value="openai" label="OpenAI">

Hosted OpenAI model. `api_num_threads` controls concurrent Q&A requests. Agentic
tasks and their model turns run sequentially; this setting does not increase
agentic task concurrency.

```yaml
model:
  path: "gpt-4.1-2025-04-14"
  prompt_format: "chat"
  engine:
    engine_type: openai
    context_length: 8192
    kwargs:
      api_num_threads: 25
  generation:
    temperature: 0.3
    max_completion_tokens: 5000
```

</TabItem>

<TabItem value="openrouter" label="OpenRouter">

Any OpenRouter-routed model, addressed by its provider/model identifier.

```yaml
model:
  path: "anthropic/claude-3.5-sonnet"
  prompt_format: "chat"
  engine:
    engine_type: openrouter
    context_length: 8192
    kwargs:
      api_num_threads: 10
  generation:
    temperature: 0.3
    max_completion_tokens: 2000
```

</TabItem>

</Tabs>

## Token limits and generation modes

`engine.context_length` configures `max_model_len` only for FAITH's in-process
vLLM engine. It does not configure, validate, or truncate requests sent to a
hosted endpoint. Configure a separately served model's context limit on the
server.

For `chat_comp`, a positive `generation.max_completion_tokens` limits each
response. FAITH sends it as `max_tokens` to in-process vLLM and as
`max_completion_tokens` to the OpenAI and OpenRouter wrappers. The default
SageMaker request template renames it to `max_tokens`; a custom template may
use a different mapping. Ensure the prompt and requested completion fit the
backend's context limit, including chat-template and tool-message overhead.

`max_completion_tokens: -1` omits FAITH's completion limit and leaves the backend
default in effect; it does not mean unlimited generation. Prefer a positive
limit for a bounded evaluation. For vLLM, see the
[sampling-parameter reference](https://docs.vllm.ai/en/v0.18.1/api/vllm/sampling_params/#vllm.sampling_params.SamplingParams.max_tokens).

For agentic runs, the limit applies separately to every model turn, not to the
whole task. Conversation history accumulates across turns; FAITH does not
automatically compact it to fit the endpoint. Task limits such as `max_turns`,
`max_tool_calls`, and `timeout_seconds` are separate benchmark settings. The
[tool-calling probe](./serving-agentic-models.md#faith-probes-before-it-runs)
also has its own completion limit.

The `logits` and `next_token` modes are available only through in-process vLLM
and only for benchmarks that support them. `max_completion_tokens` does not
control these modes: `next_token` generates one token, and `logits` uses
`max_answer_tokens` (default `1`) from `generation.kwargs`.

## Prompt formatting and reasoning

For Q&A, `prompt_format: base` produces a text prompt and omits the configured
system message. `chat` produces role-tagged messages, including the system
message when supplied. In-process vLLM requires a tokenizer with a chat
template for `chat`; API engines accept only `chat`. Choosing `chat` does not
install or replace a hosted model's server-side template.

`model.reasoning` controls local vLLM answer extraction; it does not enable a
model's reasoning capability or set a reasoning-token budget. Strings are
encoded with the selected tokenizer, while token-ID lists are used directly.
FAITH extracts the answer after the last end delimiter following the last
start delimiter. If a start delimiter is found without a following end
delimiter, the extracted answer is empty. If no start delimiter is found, the
full response is used.

API engines reject a non-null `model.reasoning`, including on agentic runs.
Use only generation options supported by the selected endpoint for reasoning
controls. `model.tokenizer` is ignored by API engines. `model.response_pattern`
is used for Q&A extraction during `eval`; it neither changes the model request
nor replaces an agentic benchmark's answer extractor.

## Which `engine_type` to use

The engine determines whether FAITH loads a model or connects to an endpoint:

- `vllm` loads model weights in the FAITH process and batches Q&A prompts.
  `num_gpus` and `context_length` configure local inference.
- `openai`, `openrouter`, and `sagemaker` connect to hosted endpoints. FAITH does
  not load model weights for these engines; `num_gpus` and `context_length` do not
  configure the server.

The `openai` engine supports OpenAI-compatible endpoints, including a separately
served local model. Set `base_url` to the endpoint's API address:

```yaml
model:
  path: "Qwen/Qwen3-8B"
  prompt_format: "chat"
  engine:
    engine_type: "openai"
    kwargs:
      base_url: "http://localhost:8100/v1"
```

For a keyless local server, FAITH supplies a placeholder API key when `base_url`
is set and `OPENAI_API_KEY` is unset. The default OpenAI endpoint still requires
a valid API key; FAITH does not supply a placeholder for it.

[Agentic benchmarks](../benchmarks/catalog.md#agentic-benchmarks) require
`engine_type: openai` and `prompt_format: chat`, including for locally served
models. The agent loop uses this engine's underlying client to send a model turn,
execute the returned tool calls, and include their results in the next request.
The `vllm`, `openrouter`, and `sagemaker` engines expose a batch interface and are
not supported by the agentic runner. For a local model, use `openai` with
`base_url` as shown above.

### Engine options and agentic requests

`engine.kwargs` configures the FAITH wrapper, not an unrestricted SDK
constructor. For `openai`, the supported options are `base_url`, `api_key`,
`api_num_threads`, `api_max_attempts`, and `api_retry_sleep_secs`. Unknown options
are currently ignored; for example, `timeout` and `max_retries` in
`engine.kwargs` do not configure the underlying client. Keep credentials in the
runner environment rather than committed YAML files. See the
[per-engine reference](../cli/query.md#per-engine-setup) for other wrappers.

Agentic requests use the resolved `temperature`, `top_p`, completion limit, and
generation kwargs on every turn. FAITH rejects `model`, `messages`, `tools`,
`tool_choice`, `client`, `max_attempts`, and `retry_sleep_secs` in generation
kwargs, including inside `extra_body`, before probing the endpoint. Streaming
is also rejected at either level. FAITH expects a single completion; do not
request `n > 1`.
Generation kwargs must be accepted by both the client and the endpoint; they
are not validated against the model's capabilities when the YAML is loaded.

`api_max_attempts` is the total number of FAITH-level attempts per model call,
including the first attempt. `api_retry_sleep_secs` sets the delay after a
retryable failure. These settings apply to Q&A calls, agentic model turns, and
the initial tool-calling probe; SDK retries are disabled. Permanent errors fail
immediately. See [API retry policy](../cli/query.md#api-retry-policy) for error
classification and server-directed delays. Agentic `timeout_seconds` is checked
between turns and does not interrupt an in-flight model request or its retry
sequence.

## Built-in model configs

FAITH ships model configurations that can be selected by name with
`--model-configs`. Select a model below to view its packaged YAML and a bounded
Q&A example command. Install the selected engine's dependencies and provide
its credentials or local hardware before running the command. Packaged
configurations are not a guarantee of current endpoint availability.

<TwoPaneConfig />

### Shared defaults

Most built-in configs inherit their engine and generation settings from two
files in the package, pulled in with the `!from` tag (e.g. `!from
'$MODELS_ROOT/default_engines.yaml["vllm-engine"]'`). Individual configs
override single fields as needed — for example, `instruct/phi-4` sets
`context_length: 16384` on top of the shared vLLM engine.

<Tabs queryString="defaults">

<TabItem value="engines" label="default_engines.yaml" default>

<CodeBlock language="yaml">{DEFAULTS.engines}</CodeBlock>

</TabItem>

<TabItem value="generation" label="default_generation.yaml">

<CodeBlock language="yaml">{DEFAULTS.generation}</CodeBlock>

</TabItem>

</Tabs>
