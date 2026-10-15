---
id: intro
title: Introduction
sidebar_label: Introduction
sidebar_position: 1
slug: /intro
---

import Pipeline from '@site/src/components/Pipeline';

# Foundation-AI's Test Hub/Harness (FAITH)

FAITH evaluates language models on cybersecurity benchmarks. It runs model
queries, grades saved outputs, and summarizes results through one CLI.
[Q&A benchmarks](./benchmarks/catalog.md#qa-benchmarks) evaluate individual
responses; [agentic benchmarks](./benchmarks/catalog.md#agentic-benchmarks)
evaluate multi-turn tasks executed with tools in a configured environment.

## Start here

- **Run your first evaluation:** [Install FAITH](./getting-started/installation.md),
  then follow the [Quickstart](./getting-started/quickstart.md).
- **Choose what to evaluate:** Browse the [benchmark catalog](./benchmarks/catalog.md)
  or [create a benchmark](./benchmarks/creating-benchmarks.md).
- **Configure a run:** Select a [local or hosted model](./configuration/model-configs.md)
  and, for agentic tasks, an [execution environment](./configuration/environment-configs.md).
- **Understand results:** Read the [metrics reference](./analytics/metrics.md)
  or set up [BigQuery analytics and dashboards](./analytics/bigquery/quickstart.md).
- **Resolve a problem:** Find the symptom in [Troubleshooting](./troubleshooting.md).

## The three-phase workflow

<Pipeline />

Run the phases independently or use `faith run-all` for all three. See the
[CLI overview](./cli/overview.md) for commands and dependencies.

Completed query stages save experiment settings alongside the trial logs.
See [Storage structure](./analytics/storage.md) for the files and the limits of
experiment identity; saved settings do not guarantee identical outputs or
capture every external dependency.

:::info
FAITH is distributed under the Apache-2.0 License.
For questions, contact [foundation-ai-oss@cisco.com](mailto:foundation-ai-oss@cisco.com).
:::
