---
id: concepts
title: Core concepts
sidebar_label: Core concepts
sidebar_position: 1
---

# Core concepts

This page defines the terms used by FAITH's documentation and CLI, their
accepted values, and the flags or configuration keys that set them.

## Run

A **run** usually means one invocation of `faith query` or `faith run-all`. It
can include multiple models, benchmarks, shot counts, and trials, producing
multiple experiments. An **agent run** is narrower: one execution of one task
by the agent loop.

## Benchmark and sample

A **benchmark** is a named evaluation definition: its data or task loader,
prompt settings, execution configuration, and scoring rules. A **suite** groups
related benchmark definitions in the catalog.

A **sample** is one evaluation item within a benchmark. For Q&A, it is a question
and its associated data. For an agentic benchmark, it is a **task** with a prompt,
ground truth, and any workspace data. A task can require multiple model turns
and tool calls; each execution is recorded as one trajectory.

## Experiment

An **experiment** is one configured run of a single benchmark against a single
model, over one or more [trials](#trial). Its configuration includes the model,
the [generation mode](#generation-mode),
[prompt format](#prompt-format), [shot count](#shot-count-n-shot),
[seed](#seed), and [generation parameters](#generation-parameters-gen-params).
After its query trials complete, metadata is written to `experiment.json`
(timestamps, FAITH version, the command line with recognized credentials
redacted, resolved model and benchmark configurations, and per-trial runtimes).
This record is not an archive of all external data or model-server settings. See
[Storage structure](../analytics/storage.md) for how experiments are laid out
on disk.

## Trial

A **trial** is one repeated pass of a benchmark at a specific seed. Running
multiple trials measures variation across repeated evaluations. Each trial uses
the seed `initial_seed + trial_index`, with a zero-based index, and gets its own
subdirectory (`trials/<seed>/…`). Distinct seeds do not guarantee distinct sample
selections or identical model behavior. Metrics are aggregated across
trials — this is where the `.mean` in a metric like `accuracy.mean` comes from.

- **Flag:** `--num-trials` (default `1`)

## Generation mode

**How the model is queried** to answer each sample. Three modes:

| Mode | What it does |
| --- | --- |
| `chat_comp` | Full chat/text completion — the model generates a free-form response (uses `--max-completion-tokens`). The default. |
| `next_token` | Requests a single next-token prediction. |
| `logits` | Requests next-token logits over the candidate answer tokens — useful for scoring multiple-choice answers directly. |

- **Flag:** `--generation-mode` (default `chat_comp`)

Agentic benchmarks pin this to `chat_comp` — an agent loop only ever uses chat
completions. Passing anything else with an agentic benchmark is an error rather
than a silent override.

## Prompt format

**How a prompt is assembled** before it is sent to the model. Agentic benchmarks
pin this to `chat`, building their conversation from the task prompt and the
agent's own tool results; a model configured for `base` is an error rather than a
silent override.

| Format | What it does |
| --- | --- |
| `chat` | Builds an OpenAI-style list of role messages (system / user / assistant). Use for instruction-tuned / chat models. |
| `base` | Builds a single plain-text string; the system prompt is ignored. Use for base/completion models. |

- **Flag:** `--prompt-format` (required with `--model-paths`; can be set per model in a config file via `prompt_format`)

## Shot count (n-shot)

The number of **in-context examples** prepended to each question (few-shot
prompting). FAITH accepts more than a plain integer:

- An integer `N` uses N fixed examples, repeated for every question.
- A fraction `N/D` sets aside `D` examples and re-samples `N` of them each time,
  allowing the few-shot set to reshuffle.

You can pass several values to sweep across shot counts in one run.

- **Flag:** `--n-shot` (default `0`)

FAITH does not apply its Q&A few-shot mechanism to agentic tasks, so agentic
benchmarks use `0`. A requested shot-count sweep is reduced to a single 0-shot
agentic run with a warning. Q&A benchmarks in the same run still execute for
each requested value.

## Seed

The base **random seed** for data sampling. Each trial derives its seed as
`seed + trial_index`. Reproducing a selection requires the same data, ordering,
sampling settings, and implementation, not only the same seed. Agentic sampling
selects a subset in the loader's original order; without `--sample-size`, it
uses the full task list.

FAITH also passes the initial seed to in-process vLLM. It does not automatically
send `--seed` as a generation parameter to hosted endpoints. Neither a sampling
seed nor a temperature of zero guarantees identical model outputs across runs.

- **Flag:** `--seed` (default `373363`)

## Generation parameters (gen params)

The **decoding parameters** that control the model's output:

| Parameter | Flag | Default |
| --- | --- | --- |
| Temperature | `--temperature` | `0.0` |
| Top-p | `--top-p` | `1.0` |
| Max completion tokens | `--max-completion-tokens` | `500` |

FAITH provides two presets in `default_generation.yaml` for model configs:
**`instruct`** (temperature 0.3) and **`reasoning`** (temperature 0.6, with a
larger token budget for chain-of-thought). Explicitly supplied CLI flags override
the corresponding config values, even when the supplied values equal the CLI
defaults. For example, `--temperature 0.0` overrides a configured temperature
of `0.3`; omitting the flag preserves `0.3`.

The set of gen params is hashed into the storage path (`gen_params_<hash>`) so
runs with different decoding settings use different paths — see
[Storage structure](../analytics/storage.md).

## Model engine

The **backend** that runs or queries a model. Four engines:

| Engine | Backend |
| --- | --- |
| `vllm` | In-process vLLM inference using supported local hardware |
| `openai` | OpenAI or a compatible endpoint; credentials depend on the endpoint |
| `openrouter` | OpenRouter API (requires `OPENROUTER_API_KEY`) |
| `sagemaker` | Models on AWS SageMaker endpoints |

- **Flag:** `--model-engine` (required with `--model-paths`); or `engine_type` in
  a [model config file](../configuration/model-configs.md)

Agentic benchmarks require `openai`. See
[engine selection](../configuration/model-configs.md#which-engine_type-to-use)
for local serving and compatibility restrictions.

## Model key

The **identifier** for a model in metrics, storage, and analytics. It is the
model's configured `name` (via `@name=<NAME>` on `--model-paths`,
or the `name` key in a config). With no name, it falls back to the path with
every character outside `[A-Za-z0-9_.-]` replaced by `_`, so
`meta-llama/Llama-3.1-8B-Instruct` becomes
`meta-llama_Llama-3.1-8B-Instruct` — the path itself is kept separately as
`source_uri`. In BigQuery, `model_key` is a primary field for grouping
results.

## Datastore

The **location where logs and records are stored**, set by
`--datastore-location`. It can be either:

- a **local directory**, or
- a **GCP URI** beginning with `gs://` — logs are staged in a temp directory and
  pushed to Cloud Storage periodically.

Within it, each experiment gets a nested path keyed by benchmark, model, prompt
format, generation mode, shot count, and generation-parameters hash. Agentic
experiments add a hash of the authored `agentic_config`. These paths are not
unique invocation IDs; see [storage identity](../analytics/storage.md#directory-layout)
before changing a deployment or reusing an output directory.

## Primary metric

The single **headline score** a benchmark reports, named by a dotted string like
`accuracy.mean`. Each benchmark declares its own in config. See the
[Metrics reference](../analytics/metrics.md) for what each metric means and how
the `name.stat` structure works.
