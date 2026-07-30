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

The `query` subcommand runs benchmarks on models and collects the
input/output records into a set of logs stored in a datastore. This command
**only records** the prompts and responses of the model — those records are
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
        {key: '@tokenizer=Path/Name', desc: 'Use an alternative tokenizer for the model.'},
        {key: '@reasoning=<start>,<end>', desc: 'Delimiter pair marking the model’s reasoning block.'},
        {key: '@response_pattern=<regex>', desc: 'Pattern for extracting the answer from the response.'},
      ],
    },
    {
      flag: '--model-configs',
      type: 'list[Path]',
      required: {note: 'unless `--model-paths` is set'},
      desc: 'Paths to YAML [model config files](/docs/cli/model-configs), each fully specifying a model’s path, engine, and generation parameters.',
      annotations: [
        {key: '@num_gpus=<N>', desc: 'Override the config’s GPU count.'},
      ],
    },
    {
      flag: '--model-engine',
      type: 'openai | openrouter | sagemaker | vllm',
      required: {note: 'with `--model-paths`'},
      desc: 'Backend engine to drive the model.',
    },
    {flag: '--num-gpus', type: 'int', default: '1', desc: 'Number of GPUs used by vLLM. A per-model `@num_gpus=` wins over it; a model whose count resolves negative is silently dropped from the run.'},
    {flag: '--model-context-len', type: 'int', default: '3500', desc: 'Context length for the models.'},
    {flag: '--max-completion-tokens', type: 'int', default: '500', desc: 'Maximum new tokens the model may generate.'},
    {flag: '--temperature', type: 'float', default: '0.0', desc: 'Sampling temperature; `0.0` is deterministic.'},
    {flag: '--top-p', type: 'float', default: '1.0', desc: 'Top-p for sampling.'},
    {
      flag: '--prompt-format',
      type: 'base | chat',
      required: {note: 'with `--model-paths`; an override for `--model-configs`'},
      desc: 'How prompts are formatted for the model.',
    },
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

Each framework accepts custom keyword args when instantiating its client. Set
them from the command line with `--engine-kwargs` followed by key-value pairs:

```shell
--engine-kwargs key1=True key2=5 key3='"foo"'
```

Values are converted with `ast.literal_eval`, so **strings must be
escape-quoted** like `'"bar"'`. For example, to pass quantization flags to
vLLM:

```shell
--engine-kwargs dtype=torch.bfloat16 trust_remote_code=True quantization='"bitsandbytes"'
```

Similarly, pass keyword args to the text-generation method with
`--generation-kwargs`:

```shell
--generation-kwargs reasoning_effort='"high"'
```

With `--model-configs`, both kwarg sets are **merged** into the config's own
kwargs rather than replacing them, and the command-line value wins on a key
collision.

### Per-engine setup

Each backend expects a different credential and set of engine kwargs. Pick your
engine below.

<Tabs groupId="model-engine">

<TabItem value="vllm" label="vLLM" default>

Runs HuggingFace models locally on GPU. `--model-paths` is the HuggingFace
repository ID (e.g. `meta-llama/Llama-3.1-8B-Instruct`); set `--num-gpus` to
shard across devices. Common engine kwargs:

```shell
--engine-kwargs dtype=torch.bfloat16 trust_remote_code=True quantization='"bitsandbytes"'
```

| Engine kwarg | Default | Description |
|------|---------|-------------|
| `dtype` | — | Compute dtype (e.g. `torch.bfloat16`). |
| `trust_remote_code` | `False` | Allow custom model code from the repository. |
| `quantization` | — | Quantization backend, e.g. `"bitsandbytes"`. |

</TabItem>

<TabItem value="openai" label="OpenAI">

Set `OPENAI_API_KEY` and use the `openai` engine. `--model-paths` is the OpenAI
model name (e.g. `gpt-4.1-2025-04-14`). Supported engine kwargs:

| Engine kwarg | Default | Description |
|------|---------|-------------|
| `api_num_threads` | `5` | Concurrent API threads. |
| `api_max_attempts` | `10` | Max retry attempts. |
| `api_retry_sleep_secs` | `1.0` | Sleep between retries, in seconds. |

</TabItem>

<TabItem value="openrouter" label="OpenRouter">

Set `OPENROUTER_API_KEY` and use the `openrouter` engine. `--model-paths`
should be the OpenRouter model identifier (e.g. `anthropic/claude-3.5-sonnet`,
`google/gemini-pro`). Supported engine kwargs:

| Engine kwarg | Default | Description |
|------|---------|-------------|
| `api_num_threads` | `5` | Concurrent API threads. |
| `api_max_attempts` | `10` | Max retry attempts. |
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
| `api_max_attempts` | `10` | Max retry attempts. |
| `api_retry_sleep_secs` | `1.0` | Sleep between retries. |
| `endpoint_timeout_secs` | `60` | Endpoint call timeout. |
| `inference_component_name` | — | Optional inference component for multi-model endpoints. |
| `request_body_expr` | — | Custom Python expression building the request body from `messages` and `gen_params`; must return a JSON-encodable object. |
| `response_parsing_expr` | — | Custom expression parsing the response body into a dict with optional fields `num_prompt_tokens`, `output_text`, `num_output_tokens`, and `max_token_halt`. |

</TabItem>

</Tabs>

## Experiment flags

<FlagTable
  rows={[
    {
      flag: '--benchmarks',
      type: 'list[string]',
      required: {note: 'unless `--custom-benchmarks` is set'},
      desc: 'Benchmarks to query each model over.',
    },
    {
      flag: '--custom-benchmarks',
      type: 'list[Path]',
      required: {note: 'unless `--benchmarks` is set'},
      desc: 'Paths to custom benchmark directories. These are *added* to any benchmarks named by `--benchmarks`.',
    },
    {flag: '--generation-mode', type: 'logits | next_token | chat_comp', default: 'chat_comp', desc: 'The generation mode.'},
    {flag: '--n-shot', type: 'list[int|N/D]', default: '0', desc: 'In-context examples per question. Accepts a plain integer, or an `N/D` fraction to re-sample N of a fixed pool of D each trial.'},
    {flag: '--num-trials', type: 'int', default: '1', desc: 'Number of trials to run.'},
    {flag: '--seed', type: 'int', default: '373363', desc: 'Random seed for reproducibility.'},
    {flag: '--sample-size', type: 'int', default: 'None', desc: 'Sample size for the benchmark; if omitted, no sampling occurs.'},
  ]}
/>

## Other flags

<FlagTable
  rows={[
    {
      flag: '--datastore-location',
      type: 'Path | GCP-URI',
      required: true,
      desc: 'Where benchmarking logs are stored — a local directory or a GCP URI (beginning with `gs://`). With a GCP URI, logs are staged in a temp directory and periodically uploaded to GCP.',
    },
    {
      flag: '--replacement-strategy',
      type: 'never | always | if_hash_differs',
      default: 'if_hash_differs',
      desc: 'Whether to replace existing records in the logs when reprocessing them.',
    },
    {
      flag: '--experimental-parallelize-models',
      type: 'flag',
      default: 'False',
      desc: 'Run multiple models in parallel with GPU pooling. **vLLM only** — with any API engine the run aborts immediately with an assertion error.',
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
