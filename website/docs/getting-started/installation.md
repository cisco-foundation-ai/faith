---
id: installation
title: Installation
sidebar_label: Installation
sidebar_position: 1
---

import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# Installation

Install from a Git checkout, selecting only the dependencies needed for your
evaluation.

## Prerequisites

- Python **3.11–3.12** (3.12 recommended).
- Git and a POSIX shell for the commands below.
- [uv](https://docs.astral.sh/uv/), the Python package manager used by FAITH's CI.
  Install it with:

```shell
curl -LsSf https://astral.sh/uv/install.sh | sh
```

uv can [obtain Python 3.12](https://docs.astral.sh/uv/guides/install-python/) if
needed; do not replace the operating system's interpreter. Hardware and service
requirements depend on the workflow selected below.

## 1. Create an environment \{#from-git-source}

Clone the repository and change into its root. Check out the intended revision
before installing:

```shell
git clone https://github.com/cisco-foundation-ai/faith.git
cd faith
uv venv --python 3.12 .venv
source .venv/bin/activate
```

## 2. Install FAITH \{#installing-the-faith-package}

Choose the workflow you will use in the [Quickstart](./quickstart.md). Each
command includes `metrics` for console summaries. Other backends and optional
features are listed under [Available extras](#available-extras).

<Tabs queryString="install-workflow">

<TabItem value="vllm" label="Local GPU" default>

For Q&A evaluation on a Linux host with a compatible accelerator and drivers:

```shell
uv pip install -e '.[vllm,metrics]'
```

Check the [vLLM 0.18.1 GPU requirements](https://docs.vllm.ai/en/v0.18.1/getting_started/installation/gpu/)
before installing. This extra pins vLLM, PyTorch, and bitsandbytes; it is not a
general-purpose CPU or macOS installation.

</TabItem>

<TabItem value="openai" label="Model API">

For Q&A evaluation through an OpenAI-compatible API, without a local GPU:

```shell
uv pip install -e '.[openai,metrics]'
```

Provide the endpoint's credentials before running. Requests to hosted OpenAI
models are billed to your OpenAI account.

</TabItem>

<TabItem value="agentic" label="Agentic with Docker">

For tool-using tasks in local containers, with model inference through an API:

```shell
uv pip install -e '.[openai,docker,metrics]'
```

Install and start Docker Engine or Docker Desktop separately; the `docker`
extra installs only the Python client. See
[Docker's Ubuntu installation guide](https://docs.docker.com/engine/install/ubuntu/)
and complete the [Docker preflight](../configuration/environment-configs.md#docker-preflight)
before a run.

Agentic runs use the `openai` client even for
[separately served local models](../configuration/serving-agentic-models.md).
For a prepared Kubernetes deployment, replace `docker` with `kubernetes` and
follow the [cluster prerequisites](../configuration/environment-configs.md#cluster-prerequisites).
These extras do not install a model server or cluster resources.

</TabItem>

</Tabs>

### Credentials \{#host-tools-and-credentials}

Provide credentials through the runner's environment or your approved secret
mechanism. Hosted OpenAI and OpenRouter engines use `OPENAI_API_KEY` and
`OPENROUTER_API_KEY`. Gated model downloads may require `HF_TOKEN` and publisher
access approval. Do not commit credentials to YAML, deployment files, or scripts.

## 3. Verify the installation

Verify the package installation without starting a model or task:

```shell
python --version
faith --help
faith query --help
uv pip check
```

Successful help output verifies the CLI import path, not model credentials,
daemon connectivity, cluster readiness, or access to benchmark downloads.

Continue to the [Quickstart](./quickstart.md) to run a bounded evaluation and
inspect its results. Local output requires no Google Cloud setup.

## Optional setup and reference

### Available extras

<details>
<summary>All dependency extras</summary>

| Extra | What it installs |
| --- | --- |
| `openai` | Dependencies to query [OpenAI models](https://github.com/openai/openai-python). Requires `OPENAI_API_KEY`. Usage is billed to your OpenAI account. |
| `openrouter` | Dependencies to query models via [OpenRouter](https://openrouter.ai/) (Anthropic Claude, Google Gemini, and more through one API). Requires `OPENROUTER_API_KEY`. |
| `vllm` | Dependencies for local inference with [vLLM](https://docs.vllm.ai/en/latest/) and supported Hugging Face models. |
| `sagemaker` | Dependencies to query models on [AWS SageMaker](https://aws.amazon.com/sagemaker/) endpoints. Requires AWS credentials and a region. |
| `all-engines` | All of the above model-engine backends (large install). |
| `metrics` | Dependencies for [`summarize`](../cli/summarize.md)'s console table output (the `tabulate` backend). Metrics themselves are computed without it. |
| `bigquery` | Dependencies to ingest metrics into BigQuery. |
| `docker` | Dependencies for [agentic benchmarks](../benchmarks/catalog.md#agentic-benchmarks) whose environment runs the agent in a local container. Requires a running Docker daemon. |
| `kubernetes` | Dependencies for [agentic benchmarks](../benchmarks/catalog.md#agentic-benchmarks) whose environment runs the agent in a Kubernetes Agent Sandbox. Requires cluster access, and is selected per run with [`--environment-config`](../configuration/environment-configs.md#kubernetes); no packaged benchmark defaults to it. |
| `all` | Everything above (large install). |
| `test` | The `all` dependencies plus pytest and its plugins (large install). Linters are not installed by pip; `make lint_all` runs them in Docker. |

</details>

### Cloud Storage (optional)

<details>
<summary>Set up Google Cloud tooling for gs:// paths</summary>

Use a `gs://` path with [`--datastore-location`](../cli/query.md) to store results
in a bucket, for example when collecting runs from several machines. FAITH
invokes the external `gsutil` command for `gs://` input or output. Install the
[Google Cloud CLI](https://cloud.google.com/sdk/docs/install), ensure `gsutil` is
on the runner's `PATH`, and configure an identity with access to the bucket.
For an interactive user installation:

```shell
gcloud init
gsutil version
```

Verify bucket access before starting a run. Installing a Python storage package
alone does not provide this CLI dependency. BigQuery ingestion uses the Python
client and has a separate
[Application Default Credentials setup](../analytics/bigquery/quickstart.md#prerequisites).

</details>

### Packaging for a remote computer

<details>
<summary>Build and transfer a source distribution</summary>

```shell
uv venv --python 3.12 .venv
source .venv/bin/activate
uv build --sdist
```

Build from a Git checkout so version and tracked package resources are available.
Transfer the generated `.tar.gz` from `dist/` to the remote machine, create a
Python 3.11 or 3.12 environment there, and install that exact archive with the
required extras. For example, replacing the version placeholder:

```shell
uv pip install './cisco_foundation_ai_test_harness-<VERSION>.tar.gz[openai,docker,metrics]'
```

The archive is not an offline dependency bundle: the remote installer still
needs dependency wheels or access to a package index. Keep the full distribution
archive; an arbitrary source-directory copy is not equivalent to an sdist.

</details>

### For development

<details>
<summary>Editable installs and Git-derived versions</summary>

The source installation above is already editable. A normal Git checkout
supplies the SCM version; do not override it for recorded evaluations. A
deliberate `SETUPTOOLS_SCM_PRETEND_VERSION` override is only a development
fallback for missing version metadata, not a replacement for the source revision.
See [Contributing](../contributing.md) for development and documentation checks.

</details>

### Shell autocompletion (optional)

<details>
<summary>Enable Bash completion</summary>

After installing, you can add Bash autocomplete for the `faith` command:

```shell
activate-global-python-argcomplete
cat <<'EOF' >>~/.bashrc

# Autocomplete functions for benchmark commands.
eval "$(register-python-argcomplete faith)"
EOF
source ~/.bashrc
```

</details>

## Next steps

- [Run your first benchmark](./quickstart.md)
- [Troubleshooting & FAQ](../troubleshooting.md)
