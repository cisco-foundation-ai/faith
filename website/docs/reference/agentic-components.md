---
id: agentic-components
title: Custom agentic components
sidebar_label: Custom agentic components
---

# Custom agentic components

Use this reference when implementing loaders, extractors, or tools. For a complete
example with registration and a test run, follow the
[agentic authoring walkthrough](../benchmarks/creating-agentic-benchmarks.md).

Custom Python implementations must be part of the installed FAITH package.
`--custom-benchmarks` discovers benchmark definitions and data; it does not
import Python files from those directories.

| Component | Contract |
| --- | --- |
| `TaskLoader` | Constructed with `(config, working_dir)`. Implement `load_tasks(benchmark_path) -> list[Task]`; return tasks in a stable order before FAITH applies sampling. |
| `Task` | Supply `id`, `prompt`, and JSON-serializable `ground_truth`. Use stable, unique IDs and keep labels separate from files uploaded to the agent. |
| Task payload | `data_path` names a local file or directory. The built-in environments upload it under `/workspace`; directory contents retain their relative paths. `working_subdir` changes the command working directory, not the upload destination. |
| Deferred preparation | If data must be fetched, supply both `data_path` and a no-argument `data_provider`. Setup calls the provider before upload, including after environment recovery. Make it repeatable and verify the expected payload revision. |
| `AnswerExtractor` | Constructed with `(config)`. Implement `extract(tool_name, answer)` and return the prediction shape required by the score functions, or `None` for an absent or malformed answer. The extractor receives no ground-truth label. |
| `Tool` | Constructed without arguments. Implement `name`, `description`, `parameters`, and `_invoke(environment, args)`; retain the base `execute` method. Declare any required environment capabilities. |

Set `Task.data_identity` to a stable, JSON-serializable content identity, such
as a verified content digest or immutable dataset revision. Compute it before
reconciliation, even when payload retrieval is deferred. It must change when
the agent-visible payload changes and must not include machine-specific cache
paths. Without it, FAITH uses `data_path` as the payload identity; changing file
contents at the same path does not by itself invalidate a saved trajectory.

Reconciliation also uses the task's position in the unsampled list. Sorting
tasks consistently avoids changes caused by filesystem enumeration order;
inserting or reordering tasks can still invalidate matches. See
[task checkpoints](../analytics/storage.md#agentic-task-checkpoints) for resume
behavior and retention limits.

For a custom terminal tool, set `is_terminal = True` and identify the argument
containing its answer with `answer_field`. Use `answer_field = None` only when
the terminal action itself represents the answer, such as abstention. The
extractor must interpret that action explicitly; returning `None` does not
represent a correctly scored abstention.

The base tool validator checks that arguments form an object and that required
keys are present. It does not enforce the full JSON Schema, including property
types or ranges. Validate those constraints in `_invoke`. Required capabilities
are method-only, runtime-checkable protocols, such as `CommandExecution`;
configuration validation rejects an environment that lacks them.
