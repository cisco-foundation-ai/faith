---
id: intro
title: Introduction
sidebar_label: Introduction
sidebar_position: 1
slug: /intro
---

# Foundation-AI's Test Hub/Harness (FAITH)

Welcome to **FAITH** — a framework for running benchmarks for language models,
with a focus on the **cybersecurity** domain.

FAITH provides a uniform way to run several open-source benchmarks for
evaluating the knowledge and capabilities of language models. It handles
everything from querying models (locally with vLLM or via APIs), through
grading responses, to aggregating results for centralized analytics.

## Why FAITH?

- **One interface, many benchmarks.** Run CTIBench, CyberMetric, MMLU,
  SecBench, SecEval, and your own custom benchmarks through a single `faith`
  command.
- **One interface, many model backends.** Query HuggingFace models locally
  with [vLLM](https://github.com/vllm-project/vllm), or hit hosted APIs from
  [OpenAI](https://platform.openai.com/docs/models),
  [OpenRouter](https://openrouter.ai/), and AWS SageMaker.
- **Reproducible experiments.** Every run records its full configuration —
  model, prompt format, generation mode, shot count, seed, and generation
  parameters — so results are traceable and comparable.
- **Analytics-ready.** Ingest results into BigQuery to build leaderboards,
  track performance over time, and power dashboards.

## The three-phase workflow

FAITH runs benchmarks in three stages, each backed by a `faith` subcommand:

import Pipeline from '@site/src/components/Pipeline';

<Pipeline />

You can run them individually, or run all three at once with `faith run-all`.

## Supported benchmarks

FAITH ships with a catalog of cybersecurity-focused benchmarks:

| Suite | Description |
| --- | --- |
| [CTIBench](https://arxiv.org/abs/2406.07599) | Cyber Threat Intelligence: technique extraction, MCQA, root-cause mapping, threat-actor attribution, vulnerability severity prediction |
| [CyberMetric](https://arxiv.org/abs/2402.07688) | 80 / 500 / 2,000 / 10,000-question multiple-choice cybersecurity sets |
| [MMLU](https://arxiv.org/abs/2009.03300) | Full MMLU plus a computer-security subset |
| [SecBench](https://arxiv.org/abs/2412.20787) | English multiple-choice and reasoning-focused security questions |
| [SecEval](https://xuanwuai.github.io/SecEval/) | ~1,000 multiple-choice computer-security questions |
| [CTI-Reasoning](https://huggingface.co/datasets/fdtn-ai/cti-reasoning) | 200 reasoning-oriented cyber threat intelligence questions |
| [CWE-Prediction](https://huggingface.co/datasets/fdtn-ai/cwe-prediction-benchmark) | 3,000 CVE / GHSA advisory descriptions to map to CWE identifiers |
| [VulnLoc](./benchmarks/catalog.md#vulnloc) | Agentic vulnerability localization in real repositories, run in two paired phases |

See the [Benchmark Catalog](./benchmarks/catalog.md) for the full list of task
identifiers.

## Next steps

- [Install FAITH](./getting-started/installation.md)
- [Run your first benchmark](./getting-started/quickstart.md)
- [Explore the CLI](./cli/overview.md)

:::info
FAITH is distributed under the Apache-2.0 License.
Questions? Reach the team at [foundation-ai-oss@cisco.com](mailto:foundation-ai-oss@cisco.com).
:::
