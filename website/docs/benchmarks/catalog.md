---
id: catalog
title: Benchmark Catalog
sidebar_label: Catalog
sidebar_position: 1
---

# Benchmark catalog

FAITH provides a uniform way to run several open-source benchmarks for
evaluating the knowledge and capabilities of language models in the
cybersecurity domain. Pass any of the task identifiers below to the
`--benchmarks` flag, or a [group name](#benchmark-groups) to run several at
once.

Each table lists the **answer type** (how the model is prompted and graded),
the **size**, and the **primary metric** FAITH reports. Sizes come from each
benchmark's own description and are approximate — the actual evaluated count
can differ because FAITH drops malformed or duplicate rows during loading. A
dash (—) means the benchmark doesn't state a count.

## CTIBench

Cyber Threat Intelligence benchmarks
([paper](https://arxiv.org/abs/2406.07599)).

| Task | Answer type | Size | Primary metric | Description |
| --- | --- | --- | --- | --- |
| `ctibench-ate` | Short answer (multi-label) | — | Accuracy | **Attack Technique Extraction** — identify MITRE ATT&CK techniques from threat descriptions. Also reports a Jaccard index over the predicted technique set. |
| `ctibench-mcqa` | Multiple choice | — | Accuracy | **Multiple Choice Questions** on cyber threat intelligence. |
| `ctibench-rcm` | Short answer | — | Accuracy | **Root Cause Mapping** — map CVE descriptions to CWE IDs. |
| `ctibench-taa` | Short answer | — | Accuracy | **Threat Actor Attribution** — attribute cyber incidents to known threat actors. Grading is alias-aware, since actor names vary: it emits `accuracy_aliased_actors` and `accuracy_related_actors` rather than a plain `accuracy`. *(Experimental — see [groups](#benchmark-groups).)* |
| `ctibench-vsp` | Short answer | — | Mean CVSS score | **Vulnerability Severity Prediction** — predict CVSS v3.1 vector strings. |

## CyberMetric

Multiple-choice cybersecurity questions
([paper](https://arxiv.org/abs/2402.07688)).

| Task | Answer type | Size | Primary metric | Description |
| --- | --- | --- | --- | --- |
| `cybermetric-80` | Multiple choice | 80 | Accuracy | 80-question subset. |
| `cybermetric-500` | Multiple choice | 500 | Accuracy | 500-question subset. |
| `cybermetric-2000` | Multiple choice | 2,000 | Accuracy | 2,000-question subset. |
| `cybermetric-10000` | Multiple choice | ~10,000 | Accuracy | ~10,000-question subset. |

## MMLU

Massive Multitask Language Understanding
([paper](https://arxiv.org/abs/2009.03300)).

| Task | Answer type | Size | Primary metric | Description |
| --- | --- | --- | --- | --- |
| `mmlu-all` | Multiple choice | ~14,000 | Subject-weighted accuracy | Full MMLU benchmark across 57 subjects. |
| `mmlu-security` | Multiple choice | 100 | Accuracy | Computer-security subset. |

## SecBench

Security-focused evaluation ([paper](https://arxiv.org/abs/2412.20787)).

| Task | Answer type | Size | Primary metric | Description |
| --- | --- | --- | --- | --- |
| `secbench-mcqa-eng` | Multiple choice | 595 | Accuracy | English single-answer multiple-choice questions, drawn from SecBench's full set of 2,730. |
| `secbench-mcqa-eng-reasoning` | Multiple choice | — | Accuracy | The English questions designed to test reasoning skills. |

## SecEval

Computer-security multiple-choice questions
([project](https://xuanwuai.github.io/SecEval/)).

| Task | Answer type | Size | Primary metric | Description |
| --- | --- | --- | --- | --- |
| `seceval` | Multiple choice | ~1,000 | Accuracy | ~1,000 multiple-choice computer-security questions. |

## CTI-Reasoning

Reasoning-focused cyber threat intelligence questions
([dataset](https://huggingface.co/datasets/fdtn-ai/cti-reasoning)).

| Task | Answer type | Size | Primary metric | Description |
| --- | --- | --- | --- | --- |
| `cti-reasoning` | Multiple choice | 200 | Accuracy | Reasoning-oriented multiple-choice questions on cyber threat intelligence. |

## CWE-Prediction

Weakness classification from vulnerability descriptions
([dataset](https://huggingface.co/datasets/fdtn-ai/cwe-prediction-benchmark)).

| Task | Answer type | Size | Primary metric | Description |
| --- | --- | --- | --- | --- |
| `cwe-prediction` | Short answer | 3,000 | Accuracy | Map CVE or GHSA advisory descriptions to CWE identifiers. |

## Benchmark groups

Instead of naming every task, you can pass a group to `--benchmarks` to run a
whole set at once:

| Group | Runs | Currently |
| --- | --- | --- |
| `all` | Every enabled benchmark. | 15 benchmarks |
| `all-security` | Enabled benchmarks tagged `security`. | all except `mmlu-all`, which is tagged `general` |
| `all-reasoning` | Enabled benchmarks tagged `reasoning`. | `cti-reasoning`, `secbench-mcqa-eng-reasoning` |

Prefix any identifier with `!` to exclude it — for example
`--benchmarks all !mmlu-all` runs the other 14. Groups and exclusions can be
combined freely, and a benchmark named by any group is dropped if it's also
negated anywhere in the list.

Every benchmark carries a `state`, and only `enabled` ones are collected into a
group. `ctibench-taa` is `experimental`, so it belongs to **no** group and must
be named directly — even though it's still a valid value for `--benchmarks`.

## Model backends

FAITH uses [vLLM](https://github.com/vllm-project/vllm) for querying
[HuggingFace](https://huggingface.co/) models, and provides API-based querying
for [OpenAI models](https://platform.openai.com/docs/models) and models
available through [OpenRouter](https://openrouter.ai/) (including Anthropic
Claude and Google Gemini) and AWS SageMaker.

## Bringing your own benchmark

Need a benchmark that isn't listed here? See
[Creating New Benchmarks](./creating-benchmarks.md) to define your own and run
it with `--custom-benchmarks`.
