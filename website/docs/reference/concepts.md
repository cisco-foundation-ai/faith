---
id: concepts
title: Core Concepts
sidebar_label: Concepts
sidebar_position: 1
---

# Core concepts

FAITH's docs and CLI use a handful of recurring terms. This page defines each
one, the values it accepts, and the flag or config key that sets it.

## Experiment

An **experiment** is one configured run of a single benchmark against a single
model, over one or more [trials](#trial). It fixes everything that makes a
result reproducible — the model, the [generation mode](#generation-mode),
[prompt format](#prompt-format), [shot count](#shot-count-n-shot),
[seed](#seed), and [generation parameters](#generation-parameters-gen-params).
Its metadata is written to `experiment.json` (start time, FAITH version, the
exact command line, the full benchmark config, and per-trial runtimes). See
[Storage Structure](../architecture/storage.md) for how experiments are laid out
on disk.

## Trial

A **trial** is one repeated pass of a benchmark at a specific seed. Running
multiple trials lets you measure run-to-run variance. Each trial uses the seed
`initial_seed + trial_index`, so trials are distinct but reproducible, and each
gets its own subdirectory (`trials/<seed>/…`). Metrics are aggregated across
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

## Prompt format

**How a prompt is assembled** before it's sent to the model:

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

## Seed

The base **random seed** for reproducibility. Each trial derives its own seed as
`seed + trial_index`, which drives how benchmark data is loaded and sampled, so
two runs with the same seed see the same questions in the same order. (Sampling
determinism at the model level comes from `--temperature` / `--top-p`, not this
seed.)

- **Flag:** `--seed` (default `373363`)

## Generation parameters (gen params)

The **decoding parameters** that control the model's output:

| Parameter | Flag | Default |
| --- | --- | --- |
| Temperature | `--temperature` | `0.0` (deterministic) |
| Top-p | `--top-p` | `1.0` |
| Max completion tokens | `--max-completion-tokens` | `500` |

FAITH ships two presets in `default_generation.yaml` that model configs can pull
in: **`instruct`** (temperature 0.3) and **`reasoning`** (temperature 0.6, with a
larger token budget for chain-of-thought). CLI flags override a config's values
only when you set them to something non-default.

The set of gen params is hashed into the storage path (`gen_params_<hash>`) so
runs with different decoding settings don't collide — see
[Storage Structure](../architecture/storage.md).

## Model engine

The **backend** that runs or queries a model. Four engines:

| Engine | Backend |
| --- | --- |
| `vllm` | Local HuggingFace models via vLLM (needs a GPU) |
| `openai` | OpenAI API (needs `OPENAI_API_KEY`) |
| `openrouter` | OpenRouter API — Claude, Gemini, and more (needs `OPENROUTER_API_KEY`) |
| `sagemaker` | Models on AWS SageMaker endpoints |

- **Flag:** `--model-engine` (required with `--model-paths`); or `engine_type` in a [model config file](../cli/model-configs.md)

## Model key

The **identifier** for a model in metrics, storage, and analytics. It's the
model's short `name` if you gave it one (via `@name=<NAME>` on `--model-paths`,
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
format, generation mode, shot count, and gen-params hash.

## Primary metric

The single **headline score** a benchmark reports, named by a dotted string like
`accuracy.mean`. Each benchmark declares its own in config. See the
[Metrics Reference](./metrics.md) for what each metric means and how the
`name.stat` structure works.

## Benchmark category (QA type)

Which **kind of question-answering** a benchmark is — determined by its config,
not a flag:

| Category | Answers look like | Example |
| --- | --- | --- |
| **Multiple choice** | One of A/B/C/D | `seceval`, CyberMetric |
| **Short answer** | A label, string, or set (label-set, string-match, or domain-specific scoring) | `ctibench-rcm`, `ctibench-vsp` |
| **Long answer** | Free-form text, graded by an LLM | *(none in the current catalog)* |

See the [Benchmark Catalog](../benchmarks/catalog.md) for each task's category.
