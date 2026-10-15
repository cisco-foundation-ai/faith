---
id: groups
title: Benchmark groups
sidebar_label: Groups
sidebar_position: 3
---

# Benchmark groups

Instead of naming every task, you can pass a group to `--benchmarks` to run a
whole set at once:

| Group | Runs | Currently |
| --- | --- | --- |
| `all` | Every enabled question-answering benchmark. | 15 benchmarks |
| `all-security` | Enabled benchmarks tagged `security`. | all except `mmlu-all`, which is tagged `general` |
| `all-reasoning` | Enabled benchmarks tagged `reasoning`. | `cti-reasoning`, `secbench-mcqa-eng-reasoning` |

Prefix any identifier with `!` to exclude it — for example
`--benchmarks all !mmlu-all` runs the other 14. Groups and exclusions can be
combined, and a benchmark named by any group is excluded if it is also
negated anywhere in the list.

Every benchmark carries a `state`, and only `enabled` ones are collected into a
group. `ctibench-taa` is `experimental`, so it belongs to **no** group and must
be named directly, although it remains a valid value for `--benchmarks`.

:::caution
No group expands to an
[agentic benchmark](./catalog.md#agentic-benchmarks), not even `all`. Each
agentic task requires environment setup and may make multiple model calls.
Agentic benchmarks are selected explicitly by exact name. `--benchmarks all`
excludes them without a warning.

Combine group and exact-name selection to include both kinds:
`--benchmarks all vulnerability-localization-a`.
:::
