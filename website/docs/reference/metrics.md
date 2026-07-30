---
id: metrics
title: Metrics Reference
sidebar_label: Metrics
sidebar_position: 2
---

# Metrics reference

The [`eval`](../cli/eval.md) phase turns stored query/response records into
metrics. This page explains what each metric in the
[Benchmark Catalog](../benchmarks/catalog.md) measures and how it's computed.

## How a score becomes a metric

Every benchmark reports a **primary metric** — the single headline number, named
by a dotted string like `accuracy.mean`. That string is a path:

```
accuracy . mean
└─metric┘ └─stat┘
```

- The **metric name** (`accuracy`) is produced per trial from the graded
  records.
- The **stat** (`mean`) is chosen after aggregating across trials. Every scalar
  metric is expanded into `mean`, `std`, `min`, `p_25`, `median`, `p_75`, and
  `max` across trials, so `accuracy.mean` is the average accuracy over all
  trials of a run. (With the default `--num-trials 1`, the mean is simply that
  one trial's value.)

Some metrics come from **built-in scorers** (accuracy, F1, confusion matrix),
and others from **custom score functions** attached to a specific benchmark
(Jaccard, CVSS, alias accuracy). Custom scorers produce a per-sample score in a
fixed range (rescaled to 0–1), then aggregate into a metric.

## Metrics by benchmark

| Benchmark(s) | Primary metric | What it measures |
| --- | --- | --- |
| Most MCQA + short-answer | `accuracy.mean` | Fraction correct **and** properly formatted |
| `ctibench-rcm` | `accuracy.mean` | Same, with `lenient_accuracy` also reported |
| `mmlu-all` | `subject_weighted_accuracy.mean` | Accuracy averaged equally across 57 subjects |
| `ctibench-ate` | `accuracy.mean` | Exact set match; Jaccard is also reported for partial overlap |
| `ctibench-vsp` | `mean_cvss_score.mean` | Closeness of predicted CVSS score to ground truth |
| `ctibench-taa` | `accuracy.mean` | Two alias-aware scorers, `accuracy_aliased_actors` and `accuracy_related_actors`; no plain `accuracy` is emitted (see note) |

:::note
`ctibench-taa`'s configured `primary_metric` is `accuracy.mean`, but its
`domain_specific` grading emits only `accuracy_aliased_actors` and
`accuracy_related_actors` — so nothing actually matches that name. The
benchmark is `experimental`, and this is a bug in its `benchmark.yaml` rather
than something to work around from the CLI.
:::

## Metric definitions

### Accuracy (strict)

The default metric. A sample scores 1 only if the prediction **equals the gold
label** *and* the answer was emitted in the expected format; anything else
scores 0. The reported value is the mean over all samples. This means a
technically-correct answer wrapped in the wrong format still scores 0 — strict
accuracy rewards models that both know the answer and follow the output
contract.

### Lenient accuracy

The same as strict accuracy but **ignoring output format**: a sample scores 1
whenever the prediction matches the label, regardless of formatting. FAITH
reports it alongside strict accuracy for multiple-choice and string-match
benchmarks (such as `ctibench-rcm`), so you can see how much of a model's error
is "wrong answer" versus "right answer, wrong format."

### Subject-weighted accuracy

Used by `mmlu-all`, which spans 57 subjects of very different sizes. Instead of
averaging over all questions (which would let large subjects dominate), FAITH
computes accuracy **within each subject** and then takes the unweighted mean
across subjects — every subject counts equally. It's only computed when a
benchmark has more than one subject.

### Jaccard index

Used by `ctibench-ate`, whose answers are **sets** of MITRE ATT&CK techniques.
For each sample it computes the overlap between the predicted and gold sets —
the size of their intersection divided by the size of their union (two empty
sets count as a perfect match). Reported as the mean and median across samples.
Unlike exact-set accuracy, Jaccard gives partial credit for getting *some* of
the techniques right.

### Micro F1

Also reported for multi-label answers. It pools true positives, false
positives, and false negatives across **all** labels globally, then computes a
single F1 score — so it reflects overall label-level precision and recall rather
than per-question exactness.

### Weighted-average F1 and per-answer F1

For multiple-choice benchmarks, FAITH reports F1 per answer symbol (A/B/C/D) and
a **weighted average** F1 across those classes (each class weighted by how often
it's the correct answer). Missing predictions are bucketed as a distinct empty
class so they aren't silently dropped.

### Mean CVSS score

Used by `ctibench-vsp`, where the model predicts a CVSS vulnerability-severity
vector. Rather than demanding an exact string match, FAITH parses both the
predicted and gold vectors into their numeric **base scores** (0–10) and scores
each sample by how close they are — a perfect match scores best, and the penalty
grows with the gap. An unparsable or missing prediction scores lowest.
Reported as the mean (and median) across samples.

### Alias accuracy

Used by `ctibench-taa` (threat-actor attribution), where the same actor has many
names (e.g. "APT29" vs. "Cozy Bear"). FAITH groups known aliases into
equivalence classes, then scores a prediction as correct if it lands in the
**same alias group** as the gold answer — so naming the right actor by a
different valid name still counts.

### Confusion matrix

For multiple-choice benchmarks, FAITH also emits a confusion matrix counting
every (true answer × predicted answer) pair, including a bucket for missing
predictions. It's a breakdown rather than a single score — useful for spotting
systematic biases, like a model that disproportionately guesses "A."

## Where metrics live

`eval` writes a `metrics.json` next to each experiment's logs, and
[`summarize`](../cli/summarize.md) collates them into a table, CSV, or BigQuery.
See [Storage Structure](../architecture/storage.md) for the on-disk layout and
[Analytics](../bigquery/quickstart.md) for querying metrics at scale.
