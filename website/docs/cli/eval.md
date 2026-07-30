---
id: eval
title: The eval subcommand
sidebar_label: eval
sidebar_position: 3
---

import FlagTable from '@site/src/components/FlagTable';

# The `eval` subcommand

The `eval` subcommand computes metrics over a set of a benchmark's logs. It
reads the raw records produced by [`query`](./query.md) and writes computed
metrics.


## Flags

<FlagTable
  rows={[
    {
      flag: '--experiment-path',
      type: 'Path',
      required: true,
      desc: 'Local path to the experiment logs to analyze. If a directory is provided, all experiments within it are analyzed. Unlike `--datastore-location`, this is a local filesystem path only — a `gs://` URI is not understood here.',
    },
    {
      flag: '--replacement-strategy',
      type: 'never | always | if_hash_differs',
      default: 'if_hash_differs',
      desc: 'Strategy for replacing existing records in benchmark logs when processing them: `never` keeps existing records and only processes missing ones; `always` reprocesses everything; `if_hash_differs` reprocesses when the data or model data have changed.',
    },
  ]}
/>

## Example

```shell
faith eval --experiment-path ./results/run-001
```

:::note
`--experiment-path` is parsed as a local filesystem path, so a `gs://` URI
won't work: it collapses to `gs:/my-bucket/...` and matches nothing. To grade
results held in GCS, copy them down first (`gcloud storage cp -r`). Only
`query`/`run-all`'s `--datastore-location` reads and writes GCS directly.
:::
