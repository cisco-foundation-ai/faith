---
id: creating-benchmarks
title: Creating New Benchmarks
sidebar_label: Creating Benchmarks
sidebar_position: 2
---

# Creating new benchmarks

To run a new benchmark with FAITH, construct a local folder for your benchmark.
This folder contains a `benchmark.yaml` file defining the benchmark, along with
any required resources (e.g. data folders or auxiliary configurations).

Once you've created your benchmark configuration, run any model over it with
the `--custom-benchmarks` flag. For example, to query the
`fdtn-ai/Foundation-Sec-8B-Instruct` model over a custom benchmark at
`${HOME}/my-custom-benchmark`:

```shell
faith query --custom-benchmarks ${HOME}/my-custom-benchmark \
  --model-paths fdtn-ai/Foundation-Sec-8B-Instruct@name=foundation-sec-8B-it \
  --model-engine vllm --model-context-len 4096 --max-completion-tokens 1000 \
  --num-trials 1 --prompt-format chat --generation-mode chat_comp --num-gpus 1 \
  --sample-size 25 --n-shot 0 --datastore-location out/
```

This queries the model over a sample of 25 questions from the custom benchmark
and saves results to the local folder `out/`.

:::note
`--custom-benchmarks` takes plain paths — it does **not** accept `@key=value`
annotations the way `--model-paths` does. The benchmark's name comes from
`metadata.name` in its `benchmark.yaml`.
:::

## Structure of a benchmark config

A benchmark configuration has the following structure:

```plaintext
benchmark:
├── metadata:
│   └── ...
├── source:
│   └── ...
├── mcqa_config / saqa_config / laqa_config:
│   └── ...
├── format:
│   └── ...
└── output_processing:
    └── ...
```

### Benchmark metadata

| Field | Used by FAITH | Purpose |
| --- | --- | --- |
| `state` | **Yes** | Whether the benchmark can be selected, and by which groups. See [states](#benchmark-states). Defaults to `enabled`. |
| `categories` | **Yes** | Tags that determine group membership — `security` feeds `all-security`, `reasoning` feeds `all-reasoning`. |
| `name` | **Yes** | The name used to refer to the benchmark. Required — an experiment asserts on it, and it's the root of the results path. |
| `description` | No | What the benchmark's data covers and why. |
| `license` | No | Any license associated with the benchmark. |
| `urls` | No | Source locations / documentation. |

Only `name`, `state` and `categories` change FAITH's behavior; the rest are
informational, but they're what someone reads to understand your benchmark.

#### Benchmark states

| State | In `--benchmarks` | In `all` groups | As a `--custom-benchmarks` folder |
| --- | --- | --- | --- |
| `enabled` | Yes | Yes | **Yes** |
| `experimental` | Yes, by exact name | No | No — silently skipped |
| `disabled` | No | No | No — silently skipped |
| `test_only` | No | No | No — silently skipped |

:::warning
For a custom benchmark, use `enabled` (or omit `state`, which defaults to it).
Any other state means the folder is skipped **without an error** — the run
simply proceeds with nothing from it.
:::

## Specification of source data

### Required columns of the loaded data

A benchmark declares its type with exactly one of `mcqa_config` (multiple
choice), `saqa_config` (short answer), or `laqa_config` (long answer) —
providing more than one is a config error. The required columns follow from that
choice:

| Config | `question` | `choices` | `answer` |
| --- | --- | --- | --- |
| `mcqa_config` | Required | Required | Required — a capital letter |
| `saqa_config` | Required | — | Required — the answer itself |
| `laqa_config` | Required | — | Optional |

- **`question`** — the question only, without answer choices.
- **`choices`** — a list of strings for the multiple-choice options.
- **`answer`** — for multiple choice, a single capital letter indexing into
  `choices` by their implied order; for short answer, the expected answer text.

All three categories also accept an optional `subject` column; it's carried
through to the graded records and made available to the prompt templates.

### Modifying data with `dataframe_transform_expr`

The `dataframe_transform_expr` field in `source.options` lets you transform the
loaded dataset into the required column format. It's a Python expression that
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
(see [Required Columns](#required-columns-of-the-loaded-data) above).
