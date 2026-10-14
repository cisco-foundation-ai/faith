---
id: creating-agentic-benchmarks
title: Create an agentic benchmark
sidebar_label: Agentic walkthrough
---

# Create an agentic benchmark

Build a one-task benchmark that reads a manifest and submits the paths tagged
`target`. The example covers loading, tool use, extraction, scoring, and result
inspection. It is a smoke test, not a measure of security capability.

## Before you start

Use a FAITH source checkout and a supported Python environment. The run requires
`.[openai,docker,metrics]`, a running Docker daemon, and model credentials; see
the [agentic quickstart](../getting-started/quickstart.md#run-an-agentic-benchmark).
One task can make multiple billed requests.

Custom Python implementations must be installed with FAITH and registered in
`src/faith/agentic/registries.py`; placing Python files in a custom benchmark
folder does not load them. This walkthrough adds those registrations explicitly.
The [component reference](../reference/agentic-components.md) describes the full
interfaces when you extend this example.

## 1. Define the task data

Create this directory outside the FAITH package:

```text
custom-file-list/
  benchmark.yaml
  tasks/
    case-001/
      manifest.txt
      expected.json
```

```text title="manifest.txt"
src/parser.py target
src/utils.py other
```

```json title="expected.json"
["src/parser.py"]
```

Only `manifest.txt` is uploaded. `expected.json` remains on the runner and
supplies the grading label.

## 2. Implement and register the loader and extractor

Add the following module as `src/faith/agentic/file_list_example.py` in your
FAITH checkout. These example classes are not bundled implementations.

```python title="file_list_example.py"
import hashlib
import json
from pathlib import Path

from faith._types.json_types import JsonValue
from faith.agentic.core import AnswerExtractor, Task, TaskLoader


class FileListTaskLoader(TaskLoader):
    def load_tasks(self, benchmark_path: Path) -> list[Task]:
        tasks = []
        for case in sorted((benchmark_path / "tasks").iterdir()):
            if not case.is_dir():
                continue
            payload = case / "manifest.txt"
            content = payload.read_bytes()
            label = json.loads((case / "expected.json").read_text(encoding="utf-8"))
            if not isinstance(label, list) or not all(
                isinstance(path, str) for path in label
            ):
                raise ValueError(f"{case}: expected.json must contain a string list")
            tasks.append(
                Task(
                    id=case.name,
                    prompt=(
                        f"Read manifest.txt in {self._working_dir} using bash_exec. "
                        "Select paths whose second column is target. "
                        "Call submit with answer containing a JSON array of paths."
                    ),
                    ground_truth=label,
                    data_path=payload,
                    data_identity={
                        "filename": payload.name,
                        "sha256": hashlib.sha256(content).hexdigest(),
                    },
                )
            )
        if not tasks:
            raise ValueError("No task directories found")
        return tasks


class FileListAnswerExtractor(AnswerExtractor):
    def extract(self, tool_name: str | None, answer: JsonValue) -> list[str] | None:
        if tool_name != "submit" or not isinstance(answer, str):
            return None
        try:
            prediction = json.loads(answer)
        except json.JSONDecodeError:
            return None
        if not isinstance(prediction, list) or not all(
            isinstance(path, str) for path in prediction
        ):
            return None
        return prediction
```

The classes inherit the required constructors from `TaskLoader` and
`AnswerExtractor`. The loader uses the supplied sandbox working directory in
the prompt, not the host path. Its digest covers the exact uploaded file;
keep that file unchanged between task loading and environment setup.

In `src/faith/agentic/registries.py`, import the two classes:

```python title="Example registry import"
from faith.agentic.file_list_example import FileListAnswerExtractor, FileListTaskLoader
```

Add these members to the **existing** registry classes, retaining all other
members. Do not create replacement registry classes:

```python title="Example registry members"
# Inside AgenticTaskLoaderType:
FILE_LIST_EXAMPLE = (FileListTaskLoader,)

# Inside AgenticAnswerExtractorType:
FILE_LIST_EXAMPLE = (FileListAnswerExtractor,)
```

Install the modified checkout with `uv pip install -e '.[openai,docker,metrics]'`.
The same loader and extractor registrations must be available wherever query
and evaluation run. Defining classes without registering them is insufficient.

## 3. Configure and validate the benchmark

Save this definition as `custom-file-list/benchmark.yaml`:

```yaml title="custom-file-list/benchmark.yaml"
benchmark:
  metadata:
    name: custom-file-list
    state: enabled
    description: Select target paths from a manifest.
  agentic_config:
    task_loader:
      type: file_list_example
    environment:
      type: docker
      config:
        agent_dockerfile: "$BENCHMARKS_ROOT/vulnerability-localization/Dockerfile"
    answer_extractor:
      type: file_list_example
    read_only_workspace: true
    tools: [bash_exec, submit]
    limits:
      max_tool_calls: 4
      max_turns: 6
      timeout_seconds: 60
      command_timeout_seconds: 5
  output_processing:
    primary_metric: mean_file_f1.mean
    score_fns:
      file_f1:
        type: file_f1
```

The omitted `working_subdir` defaults to the workspace root, so the uploaded
file is `/workspace/manifest.txt`. The example reuses the supplied Dockerfile
for its environment tools and read-only support; it does not load vulnerability
localization data. Alternative images must meet the
[read-only environment requirements](../configuration/environment-configs.md#read-only-workspace-requirements).

Before starting Docker or making a model request, validate configuration,
registration, data loading, and extraction from the directory containing
`custom-file-list`:

```python title="Validate the example before querying"
from pathlib import Path

from faith.agentic.orchestration import (
    build_answer_extractor,
    build_task_loader,
    build_tool_registry,
)
from faith.benchmark.config import load_config_from_path

config = load_config_from_path(Path("custom-file-list"))
agentic = config.agentic_config
assert agentic is not None
tasks = build_task_loader(agentic).load_tasks(Path("custom-file-list"))
extractor = build_answer_extractor(agentic.answer_extractor)
build_tool_registry(agentic.tools)
assert len(tasks) == 1
assert extractor.extract("submit", '["src/parser.py"]') == tasks[0].ground_truth
assert extractor.extract("submit", "not JSON") is None
assert extractor.extract(None, None) is None
```

This validates the Python contracts, not container startup, filesystem
enforcement, or endpoint compatibility.

## 4. Query, evaluate, and summarize

Start Docker and configure credentials as described in the
[agentic quickstart](../getting-started/quickstart.md#run-an-agentic-benchmark).
From the directory containing `custom-file-list`, run:

```shell title="Query the custom example"
faith query \
  --custom-benchmarks ./custom-file-list \
  --model-configs instruct/gpt-4.1-2025-04-14 \
  --generation-mode chat_comp \
  --sample-size 1 \
  --num-trials 1 \
  --datastore-location out/file-list
```

```shell title="Evaluate the custom example"
faith eval --experiment-path out/file-list
```

```shell title="Summarize the custom example"
faith summarize --experiment-path out/file-list \
  --stats mean_file_f1 error_rate stopped_reason
```

The `submit` tool accepts a string. A correct call therefore supplies
`{"answer": "[\"src/parser.py\"]"}`, not a list directly in `answer`.
For a correct submission, `stats.mean_file_f1.mean` is `1.0` and
`stats.stopped_reason.total.submitted` is `1`. Actual model output is not
guaranteed to match the reference submission.

Malformed submitted JSON produces an improper answer and scores zero in this
example. A task that never submits also scores zero if it ends normally, such
as with `no_tool_calls` or a limit. Tasks ending with `error` or
`environment_error` are excluded from the score and reported by `error_rate`;
an all-error trial has no `mean_file_f1` score. Inspect both metrics and the
individual trajectories, not only the command's exit status.

## Documentation validation

<details>
<summary>Run the example without a live model or container</summary>

The documentation checks execute these snippets with a scripted model and an
in-memory environment, including malformed, missing, and failed responses:

```shell
cd website
npm run check-agentic-example
```

The mocked check requires the base FAITH dependencies, but no API credentials,
GPU, running Docker daemon, or `metrics` extra. It validates CSV output in
place of the console table and does not verify a live container deployment.

</details>

## Extend the benchmark

- [Configuration reference](./creating-benchmarks.md#agentic-benchmarks): fields,
  registered tools, and scoring.
- [Read-only workspaces](./creating-benchmarks.md#read-only-workspaces): input protection
  and its security boundary.
- [Limits and sandbox recovery](./creating-benchmarks.md#limits): execution budgets
  and state-loss behavior. Recovery is enabled by default; disable it for tasks
  that depend on agent-written state or non-repeatable external side effects.
- [Custom agentic components](../reference/agentic-components.md): loaders,
  payload identity, extractors, and tools.
