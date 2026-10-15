---
id: architecture
title: Architecture overview
sidebar_label: Architecture
sidebar_position: 2
---

import Pipeline from '@site/src/components/Pipeline';
import FileTree from '@site/src/components/FileTree';
import ForkMerge from '@site/src/components/FlowConcepts/ForkMerge';

# Architecture overview

FAITH is organized around a **three-phase architecture**. Each phase is a
distinct step with a dedicated CLI subcommand, so you can run them
independently or chain them with `run-all`.

<Pipeline />

## How a run flows

`faith query` selects a trial runner for each benchmark. A benchmark with
`agentic_config` uses the agentic runner; other benchmarks use the Q&A runner.
Both write trial records and experiment metadata through the shared experiment
infrastructure.

`faith eval` loads the saved configuration and selects the appropriate answer
extractor and grader. Q&A extraction depends on the generation mode; agentic
extraction uses the recorded terminal-tool submission. Both paths reuse scoring
and across-trial aggregation utilities. `faith summarize` reads the resulting
metrics and metadata without re-executing the benchmark.

The diagram shows these execution and grading differences, not relative runtime.
Hover a lane or focus one of its cards to highlight it.

<ForkMerge />

## Code structure

The following partial tree identifies the main implementation modules under
`src/faith/`:

<FileTree
  root="src/faith/"
  tree={[
    {
      name: 'cli',
      kind: 'dir',
      note: 'Entry point — the faith command',
      children: [
        {name: 'cli.py', kind: 'file', note: 'Argument parser for all four subcommands'},
        {name: 'subcmd/', kind: 'dir', note: 'Implementation of each subcommand'},
        {name: 'flags/', kind: 'dir', note: 'Custom flag types, e.g. path@key=value'},
      ],
    },
    {
      name: 'model',
      kind: 'dir',
      note: 'Model engines (vLLM, OpenAI, etc.)',
      children: [
        {name: 'base.py', kind: 'file', note: 'The shared engine interface'},
        {name: 'factory.py', kind: 'file', note: 'Creates the appropriate engine'},
        {name: 'vllm.py', kind: 'file', note: 'vLLM local inference'},
        {name: 'openai.py', kind: 'file', note: 'OpenAI API'},
        {name: 'open_router.py', kind: 'file', note: 'OpenRouter API'},
        {name: 'sagemaker.py', kind: 'file', note: 'AWS SageMaker'},
        {name: 'config.py', kind: 'file', note: 'Model config YAML parsing'},
        {name: 'resolver.py', kind: 'file', note: 'Model path resolution'},
      ],
    },
    {
      name: 'benchmark',
      kind: 'dir',
      note: 'Benchmark loading & management',
      children: [
        {name: 'benchmark.py', kind: 'file', note: 'The Benchmark base class and QABenchmark'},
        {name: 'agentic.py', kind: 'file', note: 'AgenticBenchmark and its metrics aggregator'},
        {name: 'load.py', kind: 'file', note: 'Build a benchmark from its spec + config'},
        {name: 'config.py', kind: 'file', note: 'Benchmark YAML parsing'},
        {name: 'listing.py', kind: 'file', note: 'Resolve names and groups (all, !name)'},
        {name: 'qa/', kind: 'dir', note: 'Per-type logic: mcqa, saqa, laqa'},
        {name: 'dataset/', kind: 'dir', note: 'Dataset loading and transforms'},
        {name: 'scores/', kind: 'dir', note: 'Score functions (accuracy, Jaccard, …)'},
        {name: 'grading/', kind: 'dir', note: 'Answer extraction and grading'},
        {name: 'formatting/', kind: 'dir', note: 'Prompt rendering'},
      ],
    },
    {
      name: 'agentic',
      kind: 'dir',
      note: 'Agent execution, tools, environments, and task loading',
      children: [
        {name: 'orchestration.py', kind: 'file', note: 'Builds loaders, environments, tools, and extractors'},
        {name: 'registries.py', kind: 'file', note: 'Maps short names in YAML to implementations'},
        {name: 'core/', kind: 'dir', note: 'The ReAct loop and its interfaces'},
        {name: 'environments/', kind: 'dir', note: 'Where the agent runs: Docker, Kubernetes'},
        {name: 'tools/', kind: 'dir', note: 'Tools the agent may call'},
        {name: 'loaders/', kind: 'dir', note: 'Fetch and verify the tasks for a benchmark'},
        {name: 'extractors/', kind: 'dir', note: 'Read an answer out of a tool submission'},
      ],
    },
    {
      name: 'record_pipelines',
      kind: 'dir',
      note: 'Processing query/response records',
      children: [
        {name: 'prediction.py', kind: 'file', note: 'Generate predictions from the model'},
        {name: 'grading.py', kind: 'file', note: 'Score predictions vs. ground truth'},
        {name: 'formatting.py', kind: 'file', note: 'Format prompts for models'},
        {name: 'agentic_execution.py', kind: 'file', note: 'Run the agent loop — the agentic analog of prediction.py'},
        {name: 'agentic_formatting.py', kind: 'file', note: 'Build agentic records — the analog of formatting.py'},
        {name: 'hashing.py', kind: 'file', note: 'Hash record data to detect staleness'},
        {name: 'screening.py', kind: 'file', note: 'Reconcile records using --replacement-strategy'},
        {name: 'params.py', kind: 'file', note: 'Pipeline parameters'},
      ],
    },
    {
      name: 'experiment',
      kind: 'dir',
      note: 'Experiment tracking & logging',
      children: [
        {name: 'experiment.py', kind: 'file', note: 'Builds the datastore path; yields trials'},
        {name: 'trial_runner.py', kind: 'file', note: 'Runs a single trial'},
        {name: 'summarize.py', kind: 'file', note: 'Collates metrics into a summary'},
      ],
    },
    {
      name: 'ingestion',
      kind: 'dir',
      note: 'BigQuery ingestion',
      children: [
        {name: 'parser.py', kind: 'file', note: 'Parse logs for BigQuery'},
        {name: 'storage/', kind: 'dir', note: 'BigQuery schema and client'},
      ],
    },
    {name: '__benchmarks__', kind: 'dir', note: 'Built-in benchmark definitions'},
    {
      name: '__models__',
      kind: 'dir',
      note: 'Built-in model configs',
      children: [
        {name: 'default_engines.yaml', kind: 'file'},
        {name: 'default_generation.yaml', kind: 'file'},
      ],
    },
    {
      name: '_types',
      kind: 'dir',
      note: 'Type definitions & enums',
      children: [
        {name: 'model/', kind: 'dir', note: 'ModelSpec, ModelEngine, GenerationMode'},
        {name: 'config/', kind: 'dir', note: 'Parsed benchmark.yaml schema'},
        {name: 'enums.py', kind: 'file', note: 'Case-insensitive enum base classes'},
      ],
    },
    {
      name: '_internal',
      kind: 'dir',
      note: 'Shared utilities — hashing, I/O, metrics, iteration',
    },
  ]}
/>

:::note
This is a partial tree — 31 of the package's 131 modules, chosen to show how the
pieces fit together. Browse
[the repository](https://github.com/cisco-foundation-ai/faith) for the full
listing.
:::
