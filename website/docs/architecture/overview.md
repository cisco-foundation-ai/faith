---
id: overview
title: Architecture Overview
sidebar_label: Overview
sidebar_position: 1
---

import Pipeline from '@site/src/components/Pipeline';
import FileTree from '@site/src/components/FileTree';
import {BlueprintFlowExpanded} from '@site/src/components/QueryFlow/BlueprintFlow';

# Architecture overview

FAITH is organized around a **three-phase architecture**. Each phase is a
distinct step with a dedicated CLI subcommand, so you can run them
independently or chain them with `run-all`.

<Pipeline />

## Code structure

The package lives under `src/faith/`:

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
        {name: 'benchmark.py', kind: 'file', note: 'Core QABenchmark class'},
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
      name: 'record_pipelines',
      kind: 'dir',
      note: 'Processing query/response records',
      children: [
        {name: 'prediction.py', kind: 'file', note: 'Generate predictions from the model'},
        {name: 'grading.py', kind: 'file', note: 'Score predictions vs. ground truth'},
        {name: 'formatting.py', kind: 'file', note: 'Format prompts for models'},
        {name: 'hashing.py', kind: 'file', note: 'Hash record data to detect staleness'},
        {name: 'screening.py', kind: 'file', note: 'Cull stale records per --replacement-strategy'},
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
Only the directories and files referenced elsewhere in these docs are listed.
Browse [the repository](https://github.com/cisco-foundation-ai/faith) for the
full tree.
:::

## How a query flows

A single query travels through all three phases — from CLI parsing to stored
metrics. Each node names the component and source file it lives in; click a
decision node (the engine selector, the benchmark source) to fan out its
options.

<BlueprintFlowExpanded />

## Model engines

All engines share a common interface, letting the rest of FAITH stay agnostic
to how a given model is served:

| Engine | Backend |
| --- | --- |
| `vllm` | Local HuggingFace models via vLLM |
| `openai` | OpenAI API |
| `openrouter` | OpenRouter API (Claude, Gemini, and more) |
| `sagemaker` | Models on AWS SageMaker endpoints |

See [Model Config Files](../cli/model-configs.md) for how to configure each.
