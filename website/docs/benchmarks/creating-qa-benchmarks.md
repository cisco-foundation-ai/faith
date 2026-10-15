---
id: creating-qa-benchmarks
title: Create a Q&A benchmark
sidebar_label: Q&A walkthrough
---

# Create a Q&A benchmark

Define a one-question multiple-choice benchmark, validate its data, and run an
evaluation. This fixture checks the integration, not model capability. It uses
the packaged MCQA template for prompts, answer extraction, and accuracy scoring.

Complete [Installation](../getting-started/installation.md) with
`.[openai,metrics]` for the API-based run below. No agent workspace or Docker
daemon is needed. Model requests require credentials and are billed to your
provider account.

## 1. Create the dataset

Create a directory outside the FAITH package with these files:

```text
custom-mcqa/
  benchmark.yaml
  questions.json
```

Save the question and its answer choices in `custom-mcqa/questions.json`:

```json title="questions.json"
[
  {
    "question": "Which command prints a file's contents?",
    "choices": ["cat", "rm", "mkdir", "mv"],
    "answer": "A"
  }
]
```

The answer is the capital letter identifying the correct choice, not the choice
text. Other Q&A kinds have different [required columns](./creating-benchmarks.md#required-columns-of-the-loaded-data).

## 2. Define the benchmark

Save this as `custom-mcqa/benchmark.yaml`:

```yaml title="custom-mcqa/benchmark.yaml"
benchmark: !from
  '$BENCHMARKS_ROOT/mcqa-template.yaml["mcqa_common_benchmark"]':
    metadata:
      name: custom-mcqa
      state: enabled
      description: One-question multiple-choice smoke test.
    source:
      files:
        type: json
        benchmark_data_paths: ["questions.json"]
```

`!from` loads the packaged template and applies the supplied overrides.
`$BENCHMARKS_ROOT` resolves to FAITH's installed benchmark resources; the data path
is relative to the custom benchmark directory. Keep `state: enabled`: other
states cause custom folders to be skipped without creating an experiment.

## 3. Validate before querying

From the directory containing `custom-mcqa`, run this Python check. It loads the
configuration and local data without a model request:

```python title="Validate the Q&A example"
from pathlib import Path

from faith.benchmark.config import load_config_from_path
from faith.benchmark.dataset.load import load_data

root = Path("custom-mcqa")
config = load_config_from_path(root)
data, _ = load_data(config.metadata.name, root, config.source)
assert config.mcqa_config is not None
assert config.output_processing.primary_metric == "accuracy.mean"
assert list(data.columns) == ["question", "choices", "answer"]
assert len(data) == 1
assert data.iloc[0]["choices"] == ["cat", "rm", "mkdir", "mv"]
assert data.iloc[0]["answer"] == "A"
```

For datasets with different column names or formats, use
[`dataframe_transform_expr`](./creating-benchmarks.md#modifying-data-with-dataframe_transform_expr)
to produce the required columns.

## 4. Run and inspect the evaluation

Set `OPENAI_API_KEY` in the runner environment as described in
[Installation](../getting-started/installation.md#host-tools-and-credentials).
From the same directory, use a new datastore location and run:

```shell title="Run the Q&A example"
faith run-all \
  --custom-benchmarks ./custom-mcqa \
  --model-configs instruct/gpt-4.1-2025-04-14 \
  --generation-mode chat_comp \
  --sample-size 1 \
  --num-trials 1 \
  --n-shot 0 \
  --datastore-location out/custom-mcqa \
  --stats accuracy error_rate
```

`--custom-benchmarks` accepts plain directory paths, not `@key=value`
annotations. `run-all` queries, grades, and summarizes the saved result.
For a correctly formatted `Answer: A` response, `accuracy.mean` is `1.0`;
actual model output is not guaranteed to match it.

Expect one evaluated record and no execution error. Inspect the saved response
and the [completeness checks](../cli/eval.md#checking-result-completeness) before
expanding the dataset. For a local-GPU model instead, use the model flags and
dependencies in the [Quickstart](../getting-started/quickstart.md).

## Extend the benchmark

- [Q&A configuration](./creating-benchmarks.md#question-answering-benchmarks):
  required columns and data transformations for each question type.
- [Metadata and scoring](./creating-benchmarks.md#what-every-benchmark-declares):
  identifiers, states, categories, and primary metrics.
- [Benchmark groups](./groups.md): selection and exclusions.
