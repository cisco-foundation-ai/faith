---
id: query
title: The query subcommand
sidebar_label: query
sidebar_position: 2
---

import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';
import FlagTable from '@site/src/components/FlagTable';

# The `query` subcommand

The `query` subcommand runs benchmarks on models and collects records into a
set of logs stored in a datastore. For Q&A benchmarks, those records contain
the prompts and model responses; for agentic benchmarks, they contain each
task's full agent trajectory, including tool interactions. The records are
processed by subsequent commands ([`eval`](./eval.md) and
[`summarize`](./summarize.md)).

## Model flags

`--model-paths` and `--model-configs` are mutually exclusive, and exactly one is
required: passing both is an error, and so is passing neither.

<FlagTable
  rows={[
    {
      flag: '--model-paths',
      type: 'list[Path|Name]',
      required: {note: 'unless `--model-configs` is set'},
      desc: 'Model paths / names to query; all must be the same model type, with credentials provided via environment variables.',
      annotations: [
        {key: '@name=<NAME>', desc: 'Give the model a short name in the logs.'},
        {key: '@num_gpus=<N>', desc: 'Override the GPU count for this model.'},
        {key: '@tokenizer=Path/Name', desc: 'Use an alternative tokenizer for a vLLM model; ignored by API engines, including agentic runs.'},
        {key: '@reasoning=<start>,<end>', desc: 'Delimiter pair marking a vLLM model’s reasoning block; unsupported by API engines and invalid for agentic runs.'},
        {key: '@response_pattern=<regex>', desc: 'Pattern for extracting an answer from a Q&A response; ignored for agentic benchmarks, which use the benchmark’s answer extractor.'},
      ],
    },
    {
      flag: '--model-configs',
      type: 'list[Path]',
      required: {note: 'unless `--model-paths` is set'},
      desc: 'Paths to YAML [model config files](/docs/configuration/model-configs), or names of packaged configs; each fully specifies a model’s path, engine, and generation parameters.',
      annotations: [
        {key: '@num_gpus=<N>', desc: 'Override the config’s GPU count.'},
      ],
    },
    {
      flag: '--model-engine',
      type: 'string',
      required: {note: 'with `--model-paths`'},
      desc: 'Backend engine to drive the model: `openai`, `openrouter`, `sagemaker`, or `vllm`.',
      agentic: 'The resolved engine must be `openai`, including for a locally served model. With `--model-configs`, set it in the config; this flag is ignored — see [which engine to use](/docs/configuration/model-configs#which-engine_type-to-use).',
    },
    {
      flag: '--prompt-format',
      type: 'string',
      required: {note: 'with `--model-paths`; an override for `--model-configs`'},
      desc: 'How prompts are formatted for the model: `base` or `chat`.',
      agentic: 'The resolved format must be `chat`. With `--model-configs`, set it in the config or pass this flag as an explicit override.',
    },
    {flag: '--num-gpus', type: 'int', default: '1', desc: 'Number of GPUs used by vLLM. With `--model-paths`, a per-model `@num_gpus=` wins except `@num_gpus=0`, which currently falls back to this flag; a model whose count resolves negative is silently dropped. With `--model-configs`, the annotation wins including `0`, and negative counts are not filtered here.', agentic: 'Accepted and ignored — the `openai` engine hosts no model.'},
    {flag: '--model-context-len', type: 'int', default: '3500', desc: 'Context limit for in-process vLLM. API engines ignore it; it does not truncate hosted requests or change a server’s context limit.', agentic: 'Accepted and ignored — configure the model server’s context limit separately.'},
    {flag: '--max-completion-tokens', type: 'int', default: '500', desc: 'Completion limit per response in `chat_comp` mode. `-1` omits FAITH’s limit and uses the backend default; it does not mean unlimited generation. Does not control `logits` or `next_token`.', agentic: 'Applied separately to each model turn, not to the entire task. The initial tool-calling probe has a separate limit.'},
    {flag: '--temperature', type: 'float', default: '0.0', desc: 'Sampling temperature passed to the model backend; `0.0` does not guarantee identical results across runs.'},
    {flag: '--top-p', type: 'float', default: '1.0', desc: 'Top-p for sampling.'},
  ]}
/>

:::note
The defaults above apply to `--model-paths`. With `--model-configs` the config
supplies these values, and a flag overrides it **only if you pass the flag
explicitly** — the default is never layered on top of a config. That holds for
`--temperature`, `--top-p`, `--max-completion-tokens`, `--model-context-len`,
`--num-gpus` and `--prompt-format`. `--model-engine` is ignored entirely with
`--model-configs`: the engine comes from the config and cannot be overridden
from the command line.
:::

### Engine kwargs

`--engine-kwargs` supplies additional options accepted by the selected FAITH
engine wrapper. It is not an unrestricted SDK configuration interface. Supply
key-value pairs after the flag:

```shell
--engine-kwargs key1=True key2=5 key3='"foo"'
```

Values are parsed as Python literals with `ast.literal_eval`, so string values
must retain quotes after shell parsing, as in `'"bar"'`. Python expressions such
as `torch.bfloat16` are not accepted; pass the vLLM dtype as a string:

```shell
--engine-kwargs dtype='"bfloat16"'
```

Similarly, pass keyword args to the text-generation method with
`--generation-kwargs`:

```shell
--generation-kwargs reasoning_effort='"high"'
```

For agentic runs, generation kwargs cannot contain `model`, `messages`, `tools`,
`tool_choice`, `client`, `max_attempts`, or `retry_sleep_secs`. FAITH owns these
request and retry fields and rejects conflicting settings before probing the
endpoint. Configure retries through the engine's `api_max_attempts` and
`api_retry_sleep_secs` instead. These restrictions also apply inside `extra_body`;
streaming is rejected at either level. Other generation options must be supported
by the client and endpoint.

With `--model-configs`, both kwarg sets are shallow-merged into the corresponding
config kwargs. A command-line value replaces the entire value at a matching key,
including a nested dictionary. Use dedicated fields or flags for temperature,
top-p, completion limits, GPU count, and context length instead of duplicating
them in kwargs. See [configuration precedence](../configuration/model-configs.md#configuration-precedence).

### Per-engine setup

Each backend accepts a different set of engine options and credentials.

<Tabs groupId="model-engine" queryString="model-engine">

<TabItem value="vllm" label="vLLM" default>

Runs Hugging Face models locally on GPU. `--model-paths` is the Hugging Face
repository ID (e.g. `meta-llama/Llama-3.1-8B-Instruct`); set `--num-gpus` to
shard across devices. Common engine kwargs:

```shell
--engine-kwargs dtype='"bfloat16"'
```

| Engine kwarg | Default | Description |
|------|---------|-------------|
| `dtype` | `"auto"` | Model weight and activation dtype, for example `"bfloat16"`. Use a quoted string on the CLI. |
| `trust_remote_code` | `False` | Allow custom model code from the repository. |
| `quantization` | — | Quantization backend, e.g. `"bitsandbytes"`. |

Only set `trust_remote_code=True` for model code you have reviewed and trust.
Quantization support depends on the model, installed dependencies, and hardware.
See the [vLLM constructor reference](https://docs.vllm.ai/en/v0.18.1/api/vllm/entrypoints/llm/#vllm.entrypoints.llm.LLM).

</TabItem>

<TabItem value="openai" label="OpenAI">

Set `OPENAI_API_KEY` and use the `openai` engine. `--model-paths` is the OpenAI
model name (e.g. `gpt-4.1-2025-04-14`). Supported engine kwargs:

| Engine kwarg | Default | Description |
|------|---------|-------------|
| `base_url` | SDK default endpoint | API address of an OpenAI-compatible server. |
| `api_key` | `OPENAI_API_KEY` | Explicit credential override; prefer the environment variable to avoid storing a key in YAML or shell history. |
| `api_num_threads` | `5` | Concurrent Q&A request threads; does not change agentic task concurrency. |
| `api_max_attempts` | `10` | Total attempts per call, including the first. |
| `api_retry_sleep_secs` | `1.0` | Sleep between retries, in seconds. |

Agentic tasks run sequentially, with one model request at a time within each
task. Increasing `api_num_threads` does not parallelize this execution path.
The initial tool-calling probe uses the same retry settings as model requests.
Unknown engine kwargs are silently ignored: `timeout` and `max_retries` here do
not configure the SDK. See [API retry policy](#api-retry-policy) and
[engine options and agentic requests](../configuration/model-configs.md#engine-options-and-agentic-requests).

</TabItem>

<TabItem value="openrouter" label="OpenRouter">

Set `OPENROUTER_API_KEY` and use the `openrouter` engine. `--model-paths`
should be the OpenRouter model identifier (e.g. `anthropic/claude-3.5-sonnet`,
`google/gemini-pro`). Supported engine kwargs:

| Engine kwarg | Default | Description |
|------|---------|-------------|
| `api_num_threads` | `5` | Concurrent API threads. |
| `api_max_attempts` | `10` | Total attempts per call, including the first. |
| `api_retry_sleep_secs` | `1.0` | Sleep between retries, in seconds. |

</TabItem>

<TabItem value="sagemaker" label="SageMaker">

Configure the following engine kwargs. For SageMaker, `--model-paths` should
specify the **endpoint name**.

```shell
--engine-kwargs aws_region='"us-west-2"' api_num_threads=10 api_max_attempts=5 endpoint_timeout_secs=120 inference_component_name='"my-component"'
```

| Engine kwarg | Default | Description |
|------|---------|-------------|
| `aws_region` | — | AWS region (or set the `AWS_REGION` env var). |
| `api_num_threads` | `5` | Concurrent API threads. |
| `api_max_attempts` | `10` | Total attempts per call, including the first. |
| `api_retry_sleep_secs` | `1.0` | Sleep between retries. |
| `endpoint_timeout_secs` | `60` | SDK read timeout for an endpoint call, not an overall retry deadline. |
| `inference_component_name` | — | Optional inference component for multi-model endpoints. |
| `request_body_expr` | — | Custom Python expression building the request body from `messages` and `gen_params`; must return a JSON-encodable object. |
| `response_parsing_expr` | — | Custom expression parsing the response body into a dict with optional fields `num_prompt_tokens`, `output_text`, `num_output_tokens`, and `max_token_halt`. |

</TabItem>

</Tabs>

### API retry policy

FAITH owns retries for the API engines above; their SDK retries are disabled.
`api_max_attempts` is a positive integer: `1` disables retries and the default `10`
allows at most ten completion attempts, not ten attempts each with SDK retries.
Q&A calls, agentic turns, and the tool-calling preflight use the model's settings.
Direct `ReActLoop` use also disables retries on real OpenAI clients without
changing the caller's client; custom clients must disable their own retry layer.

FAITH retries known connection/timeouts, remote protocol interruptions, empty
completion choices, HTTP 408/409/429, and 5xx except 501/505. Here 409 is treated as
a temporary conflict, matching OpenAI's lock-timeout behavior. Other HTTP errors
(including 400/401/403/404/422), certificate-verification failures, quota/billing
exhaustion, and otherwise unknown exceptions fail immediately. SageMaker's known
throttling, request-timeout, and model-not-ready codes are also retryable. An
explicit `x-should-retry: false` prevents a retry; `true` does not override a
permanent error.

For SageMaker `ModelError` responses (HTTP 424), FAITH applies the HTTP policy to
`OriginalStatusCode`. Interrupted or incomplete SageMaker response bodies also
retry. Empty-choice recovery applies to SageMaker's default response parser;
custom `response_parsing_expr` values retain control of their response format.

`api_retry_sleep_secs` remains the fixed delay between attempts and must be finite
and non-negative. Valid `Retry-After`/`retry-after-ms` hints can increase that delay
up to 60 seconds; a longer server hint stops retries instead of retrying too soon.
There is no sleep after a permanent failure or the final attempt. Diagnostics
retain exception-chain types, HTTP status, error code and parameter identifiers,
but omit raw exception messages, response bodies, request contents and headers.

These bounds are **not an operation deadline**. Agentic `timeout_seconds` is checked
between turns; it does not cancel in-flight requests, retry sleeps, or the remainder
of a retry sequence. Request timeouts are SDK/transport settings. A timed-out
request may have reached the server, so retrying is not an exactly-once guarantee.
Retries do not restart completed agentic turns or change generation parameters.

## Experiment flags

<FlagTable
  rows={[
    {
      flag: '--benchmarks',
      type: 'list[string]',
      required: {note: 'unless `--custom-benchmarks` is set'},
      desc: 'Benchmarks to query each model over.',
      agentic: 'Select agentic benchmarks by exact identifier. Groups, including `all`, exclude agentic benchmarks without a warning.',
    },
    {
      flag: '--custom-benchmarks',
      type: 'list[Path]',
      required: {note: 'unless `--benchmarks` is set'},
      desc: 'Paths to custom benchmark directories, added to any benchmarks selected with `--benchmarks`.',
    },
    {flag: '--generation-mode', type: 'string', default: 'chat_comp', desc: 'The generation mode: `logits`, `next_token`, or `chat_comp`.', agentic: 'Must be `chat_comp`.'},
    {flag: '--n-shot', type: 'list[int|N/D]', default: '0', desc: 'In-context examples per question. Accepts a plain integer, or an `N/D` fraction to re-sample N of a fixed pool of D each trial.', agentic: 'Pinned to 0 because FAITH does not apply the Q&A few-shot mechanism to agentic tasks. A requested sweep is reduced to one 0-shot run with a warning. In a mixed run the requested values still apply to the Q&A benchmarks.'},
    {flag: '--num-trials', type: 'int', default: '1', desc: 'Number of trials to run.', agentic: 'Unchanged, but a value above 5 warns: cost scales with tasks × trials, and each trial runs every task in its own environment.'},
    {flag: '--seed', type: 'int', default: '373363', desc: 'Random seed for reproducibility.'},
    {flag: '--sample-size', type: 'int', default: 'None', desc: 'Sample size for the benchmark; if omitted, no sampling occurs.'},
    {flag: '--environment-config', type: 'Path', default: 'None', desc: '[Environment config file](/docs/configuration/environment-configs) naming where an agentic benchmark runs its agents, overriding each benchmark’s own portable default. One file per run.', agentic: 'The only agentic-only flag here. Rejected if no requested benchmark is agentic, rather than silently accepted.'},
  ]}
/>

## Other flags

<FlagTable
  rows={[
    {
      flag: '--datastore-location',
      type: 'Path',
      required: true,
      desc: 'Where benchmarking logs are stored — a local directory or a GCP URI (beginning with `gs://`). With a GCP URI, logs are staged in a temp directory and periodically uploaded to GCP.',
    },
    {
      flag: '--replacement-strategy',
      type: 'string',
      default: 'if_hash_differs',
      desc: 'Select records to reuse when querying. Records match by sample index and sample hash. For a matching record, `never` retains the existing record, `always` replaces it, and `if_hash_differs` replaces it when its input-data hash is missing or differs. A retained record is still queried if it lacks output for the requested mode. This query-time comparison does not check the model-output hash used by `eval`.',
      agentic: 'Retained trajectories are reused unless they ended with `error` or `environment_error`. Those failures and missing trajectories are retried even with `never`. Limit outcomes (`max_calls`, `max_turns`, `timeout`) and `no_tool_calls` are reused; use `always` to rerun all selected tasks. A changed sample identity is processed as a new task under every strategy. See [task checkpoints](/docs/analytics/storage#agentic-task-checkpoints) for recovery and persistence limits.',
    },
    {
      flag: '--experimental-parallelize-models',
      type: 'flag',
      default: 'False',
      desc: 'Run multiple models in parallel with GPU pooling. Requires vLLM; selecting an API engine causes an assertion error.',
      agentic: 'Rejected. An agentic run is single-process, since its Docker and OpenAI clients can’t be pickled across the model-parallel boundary.',
    },
  ]}
/>

## Example

```shell
faith query \
  --model-paths fdtn-ai/Foundation-Sec-8B-Instruct@name=foundation-sec-8B-it \
  --model-engine vllm \
  --benchmarks ctibench-mcqa seceval \
  --model-context-len 4096 \
  --max-completion-tokens 1000 \
  --num-trials 1 \
  --prompt-format chat \
  --generation-mode chat_comp \
  --num-gpus 1 \
  --sample-size 25 \
  --n-shot 0 \
  --datastore-location out/
```

## Agentic runs

An [agentic benchmark](../benchmarks/catalog.md#agentic-benchmarks) executes each
task with a tool-using agent in a sandboxed environment. The pipeline uses the
same three commands: `query` records trajectories, [`eval`](./eval.md) grades
the recorded answers, and [`summarize`](./summarize.md) collates saved metrics
for display or export.

The `agentic` note on a flag describes what changes for that flag. Model
annotations have additional agentic restrictions in their descriptions above.
Four settings are validated before any task runs. These requirements apply to
mixed Q&A and agentic runs as well:

| Setting | Required for agentic |
| --- | --- |
| Resolved model engine | `openai`, including for a locally served model |
| Resolved prompt format | `chat` |
| `--generation-mode` | `chat_comp` |
| `--experimental-parallelize-models` | Unset |

The engine and prompt-format checks apply to every model when any selected
benchmark is agentic. FAITH evaluates every selected model against every
selected benchmark; it does not support assigning a model to only part of that
set. Use separate invocations for vLLM Q&A experiments and API-based agentic
experiments. Q&A-only invocations can use different engines through model configs.

A flag with no `agentic` note otherwise behaves as it does for Q&A. Before any
agentic task starts, `query` probes the resolved endpoint for tool-calling
support; see [serving a model for agentic runs](../configuration/serving-agentic-models.md).
For reasoning models, give the agent enough `--max-completion-tokens` to emit a
tool call rather than truncating its internal reasoning.

To run one end-to-end, see the
[agentic walkthrough](../getting-started/quickstart.md#run-an-agentic-benchmark).
To benchmark a model on your own GPU, see
[serving a model for agentic runs](../configuration/serving-agentic-models.md).
For where the agent runs, see
[environment config files](../configuration/environment-configs.md).
