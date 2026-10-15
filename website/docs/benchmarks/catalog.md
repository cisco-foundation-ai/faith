---
id: catalog
title: Benchmark catalog
sidebar_label: Catalog
sidebar_position: 1
---

import SourceLinks from '@site/src/components/SourceLinks';

# Benchmark catalog

FAITH provides a uniform way to run several open-source benchmarks for
evaluating language models in the cybersecurity domain, and supports both **Q&A**
and **agentic** benchmarks, listed separately below. Pass any of the benchmark
identifiers to the `--benchmarks` flag, or a [group name](./groups.md) to run
several Q&A benchmarks at once. Each benchmark below names its primary metric; the
[metrics reference](../analytics/metrics.md) defines each one and lists the
secondary metrics it also reports.

## Q&A benchmarks

Each Q&A sample is a question and its associated data. FAITH generates or reuses
the selected model output, extracts a prediction, and applies the benchmark's
grading rules. These benchmarks do not require an agent workspace, but do
require access to their data, model, and any configured judge dependencies.

Sizes are approximate — FAITH drops malformed and duplicate rows while loading —
and a dash (—) means the dataset does not state a count.

### CTIBench <SourceLinks heading right links={[["paper", "https://arxiv.org/abs/2406.07599"], ["dataset", "https://huggingface.co/datasets/AI4Sec/cti-bench"]]} /> \{#ctibench}

Cyber Threat Intelligence benchmarks. Short-answer tasks, except
`ctibench-mcqa`; `ctibench-ate` takes a set of labels rather than one.

| Benchmark | Size | Primary metric | Description |
| --- | --- | --- | --- |
| `ctibench-ate` | — | Accuracy | **Attack Technique Extraction** — identify MITRE ATT&CK techniques from threat descriptions. |
| `ctibench-mcqa` | — | Accuracy | **Multiple Choice Questions** on cyber threat intelligence. |
| `ctibench-rcm` | — | Accuracy | **Root Cause Mapping** — map CVE descriptions to CWE IDs. |
| `ctibench-taa` | — | Accuracy | **Threat Actor Attribution** — attribute cyber incidents to known threat actors. *(Experimental — see [groups](./groups.md).)* |
| `ctibench-vsp` | — | Mean CVSS score | **Vulnerability Severity Prediction** — predict CVSS v3.1 vector strings. |

### CyberMetric <SourceLinks heading right links={[["paper", "https://arxiv.org/abs/2402.07688"], ["repository", "https://github.com/cybermetric/CyberMetric"]]} /> \{#cybermetric}

Multiple-choice cybersecurity questions.

| Benchmark | Size | Primary metric | Description |
| --- | --- | --- | --- |
| `cybermetric-80` | 80 | Accuracy | 80-question subset. |
| `cybermetric-500` | 500 | Accuracy | 500-question subset. |
| `cybermetric-2000` | 2,000 | Accuracy | 2,000-question subset. |
| `cybermetric-10000` | ~10,000 | Accuracy | ~10,000-question subset. |

### MMLU <SourceLinks heading right links={[["paper", "https://arxiv.org/abs/2009.03300"], ["dataset", "https://huggingface.co/datasets/cais/mmlu"]]} /> \{#mmlu}

Massive Multitask Language Understanding. Both tasks are multiple choice.

| Benchmark | Size | Primary metric | Description |
| --- | --- | --- | --- |
| `mmlu-all` | ~14,000 | Subject-weighted accuracy | Full MMLU benchmark across 57 subjects. |
| `mmlu-security` | 100 | Accuracy | Computer-security subset. |

### SecBench <SourceLinks heading right links={[["paper", "https://arxiv.org/abs/2412.20787"], ["repository", "https://github.com/secbench-git/SecBench"]]} /> \{#secbench}

Security-focused evaluation. Both tasks are multiple choice.

| Benchmark | Size | Primary metric | Description |
| --- | --- | --- | --- |
| `secbench-mcqa-eng` | 595 | Accuracy | English single-answer multiple-choice questions, drawn from SecBench's full set of 2,730. |
| `secbench-mcqa-eng-reasoning` | — | Accuracy | The English questions designed to test reasoning skills. |

### SecEval <SourceLinks heading right links={[["project", "https://xuanwuai.github.io/SecEval/"], ["dataset", "https://huggingface.co/datasets/XuanwuAI/SecEval"]]} /> \{#seceval}

Computer-security multiple-choice questions.

| Benchmark | Size | Primary metric | Description |
| --- | --- | --- | --- |
| `seceval` | ~1,000 | Accuracy | ~1,000 multiple-choice computer-security questions. |

### CTI-Reasoning <SourceLinks heading right links={[["dataset", "https://huggingface.co/datasets/fdtn-ai/cti-reasoning"]]} /> \{#cti-reasoning}

Reasoning-focused, multiple-choice cyber threat intelligence questions.

| Benchmark | Size | Primary metric | Description |
| --- | --- | --- | --- |
| `cti-reasoning` | 200 | Accuracy | Reasoning-oriented multiple-choice questions on cyber threat intelligence. |

### CWE-Prediction <SourceLinks heading right links={[["dataset", "https://huggingface.co/datasets/fdtn-ai/cwe-prediction-benchmark"]]} /> \{#cwe-prediction}

Weakness classification from vulnerability descriptions, answered as short
text.

| Benchmark | Size | Primary metric | Description |
| --- | --- | --- | --- |
| `cwe-prediction` | 3,000 | Accuracy | Map CVE or GHSA advisory descriptions to CWE identifiers. |

## Agentic benchmarks

Each agentic sample is a task executed through a tool-using model loop in a
configured environment. FAITH applies limits on model turns and tool calls,
records the trajectory, and grades the submitted answer. Incomplete tasks and
execution errors are reported separately in the metrics.

Each executed task requires environment setup and may make multiple model
requests. Install the extra for the selected environment, `docker` or
`kubernetes`, and satisfy the task loader's data requirements. The built-in
vulnerability-localization loader fetches source snapshots on demand and
verifies newly downloaded payloads. See
[Creating new benchmarks](./creating-benchmarks.md) for the `agentic_config`
schema.

### Vulnerability Localization <SourceLinks heading right links={[["repository", "https://github.com/cisco-foundation-ai/vulnerability-localization-benchmark"]]} /> \{#vulnerability-localization}

Vulnerability localization in real repositories.

The suite has two benchmark definitions over paired repository snapshots,
identified by the `-a` and `-b` suffixes. Phase A measures vulnerable-file
localization; phase B measures correct abstention after the target fix. Evaluate
both to distinguish localization performance from a tendency to report files
regardless of whether the target vulnerability is present.

| Benchmark | Size | Primary metric | Description |
| --- | --- | --- | --- |
| `vulnerability-localization-a` | — | File-set F1 | **Phase A, pre-fix** — given a snapshot from before the fix and the class of vulnerability it contains, name the files that must change. |
| `vulnerability-localization-b` | — | True-negative rate | **Phase B, post-fix** — the same repositories after the fix, so the vulnerability is gone. |

The default deployment requires the `openai` and `docker` extras, a compatible
model endpoint, a running Docker daemon, and runner access to the published
manifest and repository snapshots. Follow the
[Docker preflight](../configuration/environment-configs.md#docker-preflight)
before a first run. Set `manifest_path` in the benchmark config to use a local
manifest; source snapshots still need to be fetched or present in the runner's
[payload cache](../configuration/environment-configs.md#runner-network-and-cache).
