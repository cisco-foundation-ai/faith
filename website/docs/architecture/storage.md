---
id: storage
title: Storage Structure
sidebar_label: Storage Structure
sidebar_position: 2
---

import {CombinedLayout} from '@site/src/components/StorageLayout';

# Storage structure

After running benchmarks, logs should be stored in long-term storage such as a
GCP bucket.

## Uploading results

When you use a GCP URI (beginning with `gs://`) as the `--datastore-location`,
benchmark logs are automatically uploaded to GCP during the run.

To manually upload local benchmark results to GCP:

```shell
gcloud storage rsync -R -J -P ~/benchmarks/<BENCHMARK> gs://<BENCHMARK_PATH>
```

Replace `<BENCHMARK>` with your local benchmark directory and
`<BENCHMARK_PATH>` with the destination path in your GCP bucket.

## Downloading results

:::warning
An entire benchmark test may be enormous (hundreds of GB) because logs include
all queries and responses over multiple benchmarks, models, and trials.
Downloading a full test may take hours. It's recommended to only download
subtrees of interest in most cases.
:::

### Downloading metrics only

To examine a benchmark, you usually only need the metrics files. Download all
metrics for a benchmark from GCP efficiently, skipping the large
`benchmark-log.json` files:

```shell
gcloud storage rsync -R -J -P -x '.*/benchmark-log\.json$' gs://<BENCHMARK_PATH> ~/benchmarks/<BENCHMARK>
```

This fetches experiment descriptions, benchmark metrics, and summary files
without the full per-query logs.

## Directory layout

Each benchmark run is stored according to the structure below. Every path
segment and its meaning is spelled out; **hover a breadcrumb chip** to highlight
the matching level.

<CombinedLayout />

### Files

- **`experiment.json`** — describes how the benchmark was run and its
  parameters. Not created until all benchmark trials complete.
- **`metrics.json`** — metrics summarizing all trials plus aggregate statistics
  for the entire run.

The **`trials`** sub-folder records all query/responses from the run. Its
`trial-seed` and `benchmark-trial-hash` subfolders both identify the trial:

- the **seed** lets you compare similar trials (exactly the same base
  questions) across runs. Trial *n* uses `--seed` + *n*, so a run with
  `--seed 373363 --num-trials 3` writes `trials/373363/`, `trials/373364/`,
  and `trials/373365/`;
- the **hash** lets you compare exactly the same LLM queries between models, but
  is only useful when benchmark parameters are identical. It's the last 16
  characters of a hash of the benchmark spec.

Each trial contains:

- **`benchmark-log.json`** — a list of records for every query-response
  executed from the benchmark on the model.
