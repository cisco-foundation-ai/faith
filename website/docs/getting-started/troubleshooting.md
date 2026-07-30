---
id: troubleshooting
title: Troubleshooting & FAQ
sidebar_label: Troubleshooting
sidebar_position: 3
---

# Troubleshooting & FAQ

Scan for your symptom and expand it. If you hit something not covered here,
reach the team at
[foundation-ai-oss@cisco.com](mailto:foundation-ai-oss@cisco.com).

## Installation

<details>
<summary>A command fails with a missing-dependency error</summary>

**Fix:** install the extra that command needs, e.g.
`uv pip install -e '.[vllm,metrics]'`.

FAITH's dependencies are split into [extras](./installation.md#available-extras)
so you only install what you use. Which one you need depends on the failure:

| Symptom | Extra |
| --- | --- |
| `query` can't find an engine | `vllm`, `openai`, `openrouter`, or `sagemaker` |
| `summarize` can't print a table | `metrics` (supplies the `tabulate` backend) |
| `summarize` reports `BigQuery dependencies not installed` | `bigquery` |

</details>

## Models & engines

<details>
<summary>vLLM runs out of GPU memory</summary>

**Fix:** shard across more GPUs with `--num-gpus <N>`, or lower
`--model-context-len` (default `3500`).

The model is too large for the available VRAM. Options, roughly in order of
what to try first:

- Shard across more GPUs with `--num-gpus <N>`.
- Lower `--model-context-len` — a smaller context uses less memory.
- Quantize the model with an engine kwarg, e.g.
  `--engine-kwargs quantization='"bitsandbytes"'`.
- Use a smaller model, or switch to an API engine (`openai` / `openrouter`),
  which needs no local GPU.

</details>

<details>
<summary>API authentication errors (OpenAI / OpenRouter / SageMaker)</summary>

**Fix:** export the credential your engine expects.

| Engine | Credential |
| --- | --- |
| `openai` | `OPENAI_API_KEY` |
| `openrouter` | `OPENROUTER_API_KEY` |
| `sagemaker` | AWS credentials + a region (`AWS_REGION` or the `aws_region` engine kwarg) |

Put it in your shell profile so it survives new sessions:

```shell
echo 'export OPENAI_API_KEY="sk-…"' >> ~/.bashrc
source ~/.bashrc
```

</details>

<details>
<summary>Engine kwargs are being rejected or misparsed</summary>

**Fix:** escape-quote string values — `key='"value"'`, not `key=value`.

`--engine-kwargs` and `--generation-kwargs` values are parsed with
`ast.literal_eval`, so a bare word is read as a Python name rather than a
string:

```shell
# right — the string is wrapped in single + double quotes
--engine-kwargs quantization='"bitsandbytes"'
# wrong — parsed as a bare name, not a string
--engine-kwargs quantization=bitsandbytes
```

Numbers and booleans (`5`, `True`) are written plainly.

</details>

<details>
<summary><code>Unknown annotation keys</code> / <code>Annotations must be in the form of key=value</code></summary>

**Fix:** check the key is accepted by *that* flag — `--model-configs` takes
only `num_gpus`.

Inline annotations chain onto a path with `@`, and each one must contain an `=`:

```shell
--model-paths meta-llama/Llama-3.1-8B-Instruct@name=llama3-8b@num_gpus=2
```

The accepted keys differ between the two model flags, which is the usual cause
of `Unknown annotation keys`:

| Flag | Accepted annotations |
| --- | --- |
| `--model-paths` | `name`, `num_gpus`, `reasoning`, `response_pattern`, `tokenizer` |
| `--model-configs` | `num_gpus` only |

Everything else for a packaged or YAML config is set inside the config file
itself — see [Model Config Files](../cli/model-configs.md).

</details>

## Running benchmarks

<details>
<summary>My results vary between runs</summary>

**Fix:** use `--temperature 0.0` with a fixed `--seed`.

If `--temperature` is above `0.0`, generation is non-deterministic. To
*measure* variance rather than eliminate it, run multiple `--num-trials` —
metrics are aggregated across them (see
[Trials](../reference/concepts.md#trial)).

</details>

<details>
<summary>A benchmark name isn't recognized</summary>

**Fix:** check the exact identifier in the
[Benchmark Catalog](../benchmarks/catalog.md).

You can also pass a [group](../benchmarks/catalog.md#benchmark-groups) like
`all` or `all-security`. Note that experimental benchmarks aren't part of any
group and must be named directly.

</details>

## Results & analytics

<details>
<summary><code>No metrics.json files found in &lt;path&gt;</code></summary>

**Fix:** run `eval` against that path before `summarize`.

The three phases each read the previous one's output from the datastore, so
confirm `query` and `eval` both completed against the same
`--datastore-location`. See [Storage Structure](../architecture/storage.md) for
where each file lands.

</details>

<details>
<summary><code>Missing experiment.json for &lt;path&gt;</code></summary>

**Fix:** re-run `query` for that experiment.

`experiment.json` isn't written until **all** trials for a run complete, so this
usually means `query` was interrupted partway through.

</details>

## Still stuck?

- Re-read the relevant [CLI reference](../cli/overview.md) page for exact flags
  and defaults.
- Check [Core Concepts](../reference/concepts.md) if a term is unclear.
- Email [foundation-ai-oss@cisco.com](mailto:foundation-ai-oss@cisco.com).
