<!--
Copyright 2025 Cisco Systems, Inc. and its affiliates

SPDX-License-Identifier: Apache-2.0
-->

# Foundation-AI's Test Hub/Harness (FAITH)

**FAITH** is a framework for benchmarking the knowledge and capabilities of
language models in the cybersecurity domain. It gives a uniform way to run
open-source security benchmarks across many model backends, compute metrics from
the results, and aggregate them into comparable summaries.

FAITH ships 16 benchmarks drawn from CTIBench, CyberMetric, SecBench, SecEval,
MMLU, CTI-Reasoning and CWE-Prediction. Models can be run locally with
[vLLM](https://github.com/vllm-project/vllm) or queried through the
[OpenAI](https://platform.openai.com/docs/models),
[OpenRouter](https://openrouter.ai/) and
[AWS SageMaker](https://aws.amazon.com/sagemaker/) APIs.

## Documentation

📚 **[Read the full documentation at faith-evals.com](https://faith-evals.com/)**

The documentation source lives in [`website/docs/`](website/docs/) and is
built as a site with [Docusaurus](https://docusaurus.io/) — installation,
the CLI reference, the benchmark catalog, how to add your own benchmark,
and the BigQuery analytics guide.

Quick links:

- [Installation](https://faith-evals.com/docs/getting-started/installation) —
  prerequisites, extras, remote-VM setup
- [Quickstart](https://faith-evals.com/docs/getting-started/quickstart) — a
  benchmark run end-to-end
- [CLI reference](https://faith-evals.com/docs/cli/overview) — every flag
  of `query`, `eval`, `summarize` and `run-all`
- [Benchmark catalog](https://faith-evals.com/docs/benchmarks/catalog) —
  what each benchmark measures and how it's scored
- [Creating benchmarks](https://faith-evals.com/docs/benchmarks/creating-benchmarks)
  — the `benchmark.yaml` format

<details>
<summary>Running the docs site locally</summary>

To preview documentation changes locally before pushing, see [`website/README.md`](website/README.md):

```shell
cd website && npm install && npm run start
```

</details>

## Installation

FAITH runs on **Python 3.11–3.12** (3.12 recommended). Clone the repository and
install from your checkout with [uv](https://docs.astral.sh/uv/):

```shell
uv venv .venv && source .venv/bin/activate
uv pip install -e '.[all]'
```

The dependencies are split into extras so you can keep installs small —
`vllm`, `openai`, `openrouter`, `sagemaker`, `metrics`, `bigquery`, or `all`.
See [Installation](https://faith-evals.com/docs/getting-started/installation)
for the full table, GPU guidance, and per-extra details.

## Quickstart

Run the whole pipeline — query, eval, summarize — in one command:

```shell
faith run-all \
  --model-paths meta-llama/Llama-3.1-8B-Instruct \
  --benchmarks ctibench-mcqa seceval \
  --datastore-location gs://my-bucket/results/run-001 \
  --model-engine vllm \
  --generation-mode chat_comp \
  --prompt-format chat \
  --num-trials 3 \
  --n-shot 0
```

This queries the model over both benchmarks, computes metrics from the logs, and
prints a summary table. Add `--sample-size 25` for a quick smoke test. Each
phase is also its own subcommand if you'd rather run them separately.

See the [Quickstart](https://faith-evals.com/docs/getting-started/quickstart)
for a walkthrough, and
[Storage structure](https://faith-evals.com/docs/analytics/storage) for how
results are laid out.

## Roadmap

See the [open issues](https://github.com/cisco-foundation-ai/faith/issues) for a
list of proposed features (and known issues).

## Contributing

Contributions are what make the open source community such an amazing place to
learn, inspire, and create. Any contributions you make are **greatly
appreciated**. For detailed contributing guidelines, please see
[CONTRIBUTING.md](CONTRIBUTING.md).

## License

Distributed under the `Apache-2.0` License. See [LICENSE](LICENSE) for more
information.

## Contact

Foundation AI - <foundation-ai-oss@cisco.com>

Project Link:
[https://github.com/cisco-foundation-ai/faith](https://github.com/cisco-foundation-ai/faith)

## Acknowledgements

This template was adapted from
[https://github.com/othneildrew/Best-README-Template](https://github.com/othneildrew/Best-README-Template).
