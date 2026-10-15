---
id: metrics
title: Metrics reference
sidebar_label: Metrics reference
sidebar_position: 2
---

# Metrics reference

The [`eval`](../cli/eval.md) phase turns stored query/response records into
metrics. This page explains what each metric in the
[Benchmark catalog](../benchmarks/catalog.md) measures and how it is computed.

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
  `max` across trials, so `accuracy.mean` is the average accuracy over trials
  that emit that metric. (With the default `--num-trials 1`, the mean is
  that one trial's value.)

Some metrics come from **built-in scorers** (accuracy, F1, confusion matrix),
and others from **custom score functions** attached to a specific benchmark
(Jaccard, CVSS, alias accuracy, file-set F1, true-negative rate). Custom scorers
produce a per-sample score in a configured range (0–1 by default), then
aggregate into a metric.

The same structure covers agentic benchmarks: the per-sample score is computed
from what the agent submitted through its terminal tool rather than from text
extracted out of a reply, but the `name.stat` path and the across-trial
aggregation are identical.

## Metrics by benchmark

| Benchmark(s) | Primary metric | What it measures |
| --- | --- | --- |
| Most MCQA + short-answer | `accuracy.mean` | Fraction correct **and** properly formatted |
| `ctibench-rcm` | `accuracy.mean` | Same, with `lenient_accuracy` also reported |
| `mmlu-all` | `subject_weighted_accuracy.mean` | Accuracy averaged equally across 57 subjects |
| `ctibench-ate` | `accuracy.mean` | Exact set match; Jaccard is also reported for partial overlap |
| `ctibench-vsp` | `mean_cvss_score.mean` | Closeness of predicted CVSS score to ground truth |
| `ctibench-taa` | `accuracy.mean` | Two alias-aware scorers, `accuracy_aliased_actors` and `accuracy_related_actors`, rather than a plain `accuracy` |
| `vulnerability-localization-a` | `mean_file_f1.mean` | Overlap between the files the agent named and the files the real fix touched |
| `vulnerability-localization-b` | `mean_true_negative_rate.mean` | How often the agent correctly reports that there is nothing to find |

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
across subjects — every subject counts equally. It is only computed when a
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
it is the correct answer). Missing predictions are assigned to a distinct empty
class rather than excluded.

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

### File-set F1

Used by `vulnerability-localization-a`, where the agent submits a list of source
files it believes must change to fix a vulnerability. Both lists are compared as
**sets** of exact path strings — order and duplicates are ignored — and each
sample scores the F1 of the predicted set against the files the real fix
touched. Naming every file in the repository is therefore no better than naming
a few of the right ones: precision counts as much as recall. Submitting nothing
scores 0, since every sample in this phase does have labeled files. Reported as
the mean and median across samples.

### True-negative rate

Used by `vulnerability-localization-b`, which runs the same repositories
**after** the fix, so there is no vulnerability to find and no ground-truth file
list to compare against. A sample scores 1 only when the agent submitted an
answer that named no files; naming any file scores 0, and so does never
submitting an answer at all — running out of turns is not an abstention.
Reported as the mean across samples, which makes it the fraction of tasks the
agent correctly declined.

### Confusion matrix

For multiple-choice benchmarks, FAITH also emits a confusion matrix counting
every (true answer × predicted answer) pair, including a bucket for missing
predictions. It is a breakdown rather than a single score, useful for identifying
systematic biases, like a model that disproportionately guesses "A."

## How many samples the score describes

Score aggregation excludes records with a recorded error. For agentic
benchmarks, these are tasks ending with `error` or `environment_error`.
Non-submission alone does not exclude a task: `no_tool_calls`, `max_calls`,
`max_turns`, and `timeout` remain in the scoring population. The benchmark's
scorer determines the score for a missing answer; both vulnerability-localization
phases assign zero.

Use these fields in `metrics.json` to distinguish model performance from
execution failures:

| Key | Meaning |
| --- | --- |
| `queries.total` | Query records included in evaluation, summed over trials, including errors |
| `error_rate.mean` | Mean across trials of the fraction of query records excluded because of a recorded error |
| `format` | Breakdown of how answers were formatted: `proper`, `improper`, `inferred`, `invalid` |
| `stopped_reason` | **Agentic only** — breakdown of how each agent run ended |

For a nonempty evaluation, `error_rate.mean: 0` indicates that no evaluated
records were excluded for errors; it does not establish that every planned task
was evaluated or submitted an answer. Per-trial `query_count`, `scored_count`,
and `error_count` report the total, included, and excluded record counts.
`error_rate` also carries the
across-trial statistics described above. The agentic `stopped_reason` breakdown
includes both scored outcomes and excluded errors.

### Trial coverage and weighting

`stats.num_trials` counts nonempty trials that contributed evaluation records,
not the number originally requested. Missing log files and empty trials do not
contribute. `queries.total` counts records found in evaluated logs; neither
field establishes that the planned sample/task selection was completed.

Across-trial scalar means are unweighted: a one-task trial and a nine-task trial
contribute equally. For example, trial scores of 1.0 and 0.0 produce a mean of
0.5, even if the pooled score over all ten tasks would be 0.1.

An agentic trial containing only excluded errors still contributes to `num_trials`,
query counts, and error metrics, but does not emit a performance score. Such a
trial is absent from the performance mean, not included with a zero score.
Consequently, performance and error-rate means can describe different sets of
trials. Report per-trial scored/error counts alongside the headline score and
verify the [saved-result completeness checks](../cli/eval.md#checking-result-completeness).

### The `stopped_reason` breakdown

Agentic benchmarks report seven termination reasons:
(`submitted`, `no_tool_calls`, `max_calls`, `max_turns`, `timeout`, `error`,
`environment_error`). This breakdown distinguishes submissions, exhausted
limits, repeated turns without tool calls, and execution failures.

`stats.stopped_reason` contains three dictionaries keyed by reason:

- `total`: counts summed over all trials.
- `mean_per_trial`: total counts divided by the number of trials.
- `mean_per_query`: total counts divided by the total query count, including
  records excluded from scoring.

A parallel `stopped_reason_rate` expands each reason into across-trial statistics
(`mean`, `std`, `min`, and the other scalar aggregates). For example,
`stopped_reason_rate.submitted.mean` is the mean across trials of the fraction of
tasks that submitted. Both breakdowns include errored tasks.

For the table and CSV representation, see
[summarizing an agentic run](../cli/summarize.md#summarizing-an-agentic-run).

For what an individual reason means and what to do about it, see
[troubleshooting](../troubleshooting.md#agentic-runs).

## Where metrics live

`eval` writes a `metrics.json` next to each experiment's logs, and
[`summarize`](../cli/summarize.md) collates them into a table, CSV, or BigQuery.
See [Storage structure](./storage.md) for the on-disk layout and
[Analytics](./bigquery/quickstart.md) for querying metrics at scale.

The file holds per-trial values and their across-trial aggregates:

```json
{
  "per_trial_metrics": {"<trial>": {"accuracy": 0.7, "query_count": 100}},
  "stats": {"accuracy": {"mean": 0.7, "std": 0.0}, "queries": {"total": 100}}
}
```

`per_trial_metrics` is one entry per contributing nonempty trial, holding that
trial's raw values.
`stats` is the across-trial aggregation, and is where a primary metric like
`accuracy.mean` resolves — the `mean` in that path is the stat computed here.
Dictionary-valued counts lose the `_count` suffix in `stats`, where they become
the three-way breakdown above: `stopped_reason_count` per trial becomes
`stats.stopped_reason`. Scalar `scored_count` and `error_count` remain per-trial
fields; `query_count` is aggregated separately as `stats.queries`.
