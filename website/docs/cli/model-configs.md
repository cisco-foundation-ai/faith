---
id: model-configs
title: Model Config Files
sidebar_label: Model Config Files
sidebar_position: 5
---

import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';
import CodeBlock from '@theme/CodeBlock';
import {TwoPaneConfig} from '@site/src/components/ModelConfig/TwoPaneConfig';
import {DEFAULTS} from '@site/src/components/ModelConfig/configs';

# Model config files

Instead of specifying model parameters through individual CLI flags (which
apply uniformly to all models), you can use `--model-configs` to provide a
complete specification of each model's configuration. This is especially useful
when benchmarking multiple models that require different engine or generation
settings, and for creating reproducible model assessments.

Each YAML config file must have a top-level `model:` key. Three fields are
required — `model.path`, `model.prompt_format`, and
`model.engine.engine_type` — and all others have defaults.

## Config file schema

```yaml
model:
  name: "" # display name (defaults to model path segment)
  path: "" # model path or name [REQUIRED]
  prompt_format: "" # base | chat [REQUIRED]
  response_pattern: null # regex pattern for extracting responses
  tokenizer: null # custom tokenizer path (optional)
  engine:
    engine_type: "" # openai | openrouter | sagemaker | vllm [REQUIRED]
    num_gpus: 1 # number of GPUs (default: 1)
    context_length: 3500 # context length (default: 3500)
    kwargs: {} # additional keyword arguments for the engine
  generation:
    temperature: 0.0 # sampling temperature (default: 0.0)
    top_p: 1.0 # top-p sampling (default: 1.0)
    max_completion_tokens: 500 # max new tokens (default: 500)
    kwargs: {} # additional keyword arguments for generation
  reasoning: # reasoning delimiters (optional)
    start_delimiter: "<think>"
    end_delimiter: "</think>"
```

## Examples

<Tabs groupId="model-engine">

<TabItem value="vllm" label="vLLM" default>

Local HuggingFace model on GPU, with reasoning delimiters supplied as token IDs.

```yaml
model:
  name: "gpt-oss-20b"
  path: "openai/gpt-oss-20b"
  prompt_format: "chat"
  engine:
    engine_type: vllm
    context_length: 8192
  generation:
    temperature: 0.6
    max_completion_tokens: 10000
  reasoning:
    start_delimiter: [200006, 173781, 200005, 35644, 200008]
    end_delimiter: [200007, 200006, 173781, 200005, 17196, 200008]
```

</TabItem>

<TabItem value="openai" label="OpenAI">

Hosted OpenAI model. `api_num_threads` controls request concurrency.

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

## Built-in model configs

FAITH ships with pre-configured model configs, so you don't have to write your
own YAML for common models. Reference one directly by name with
`--model-configs`. Browse the groups below and select a model to see its exact
config and a ready-to-run command:

<TwoPaneConfig />

### Shared defaults

Most built-in configs inherit their engine and generation settings from two
files in the package, pulled in with the `!from` tag (e.g. `!from
'$MODELS_ROOT/default_engines.yaml["vllm-engine"]'`). Individual configs
override single fields as needed — for example, `instruct/phi-4` sets
`context_length: 16384` on top of the shared vLLM engine.

<Tabs>

<TabItem value="engines" label="default_engines.yaml" default>

<CodeBlock language="yaml" title="__models__/default_engines.yaml">{DEFAULTS.engines}</CodeBlock>

</TabItem>

<TabItem value="generation" label="default_generation.yaml">

<CodeBlock language="yaml" title="__models__/default_generation.yaml">{DEFAULTS.generation}</CodeBlock>

</TabItem>

</Tabs>
