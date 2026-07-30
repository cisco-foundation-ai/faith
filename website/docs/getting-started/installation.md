---
id: installation
title: Installation
sidebar_label: Installation
sidebar_position: 1
---

# Installation

## Prerequisites

FAITH runs on **Python 3.11–3.12** (we recommend 3.12).

The install commands below use [**uv**](https://docs.astral.sh/uv/), Astral's
Python package manager, which is also what FAITH's CI uses. Install it with:

```shell
curl -LsSf https://astral.sh/uv/install.sh | sh
```

We generally recommend running FAITH with **1 or more GPUs** when using vLLM.
vLLM can be installed on CPU-only machines, but this is not directly supported
and may be very slow. For **API-based querying** (e.g. OpenAI, OpenRouter),
GPUs are **not** required.

We also recommend using cloud storage for storing and combining benchmark
results.

### Configuring a remote VM (Ubuntu)

For VMs using Ubuntu images, these steps (or equivalents) install the tools for
running FAITH.

**1. Install Python 3.12 and make it the default interpreter:**

```shell
# Set up Python 3.12
sudo apt update && \
sudo apt install software-properties-common -y && \
sudo add-apt-repository ppa:deadsnakes/ppa -y

DEBIAN_FRONTEND=noninteractive sudo apt update && \
sudo apt install -y python3.12 python3.12-venv python3.12-dev

# Make Python 3.12 the default Python interpreter
sudo update-alternatives --install /usr/bin/python python /usr/bin/python3.12 1 && \
sudo update-alternatives --set python /usr/bin/python3.12
```

**2. Install the Google Cloud CLI (optional).** Needed only if you're storing
results in Cloud Storage — see [`--datastore-location`](../cli/query.md):

```shell
curl -fsSL https://packages.cloud.google.com/apt/doc/apt-key.gpg \
  | sudo gpg --dearmor \
  | sudo tee /etc/apt/keyrings/cloud.google.gpg > /dev/null && \
sudo chmod a+r /etc/apt/keyrings/cloud.google.gpg && \
echo "deb [signed-by=/etc/apt/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" \
  | sudo tee /etc/apt/sources.list.d/google-cloud-sdk.list > /dev/null && \
sudo apt update && \
sudo apt install -y google-cloud-cli

gcloud init
```

**3. Export your API tokens (optional).** Add any tokens you need (OpenAI,
HuggingFace, and so on) to your shell profile so they survive new sessions:

```shell
echo 'export HF_TOKEN="hf_000000000000000000000000"' >> ~/.bashrc
source ~/.bashrc
```

## Installing the FAITH package

To properly use FAITH, install it with the `<EXTRAS>` packages that enable the
functionality you need. Extras are broken out to allow smaller installs for
specific use-cases.

### Available extras

| Extra | What it installs |
| --- | --- |
| `openai` | Dependencies to query [OpenAI models](https://github.com/openai/openai-python). Requires `OPENAI_API_KEY`. Usage is billed to your OpenAI account. |
| `openrouter` | Dependencies to query models via [OpenRouter](https://openrouter.ai/) (Anthropic Claude, Google Gemini, and more through one API). Requires `OPENROUTER_API_KEY`. |
| `vllm` | Dependencies to run models locally with [vLLM](https://docs.vllm.ai/en/latest/). Runs many HuggingFace models. |
| `sagemaker` | Dependencies to query models on [AWS SageMaker](https://aws.amazon.com/sagemaker/) endpoints. Requires AWS credentials and a region. |
| `all-engines` | All of the above model-engine backends (large install). |
| `metrics` | Dependencies to compute metrics from benchmark logs. |
| `bigquery` | Dependencies to ingest metrics into BigQuery. |
| `all` | Everything above (large install). |
| `test` | The `all` dependencies plus pytest and its plugins (large install). Linters aren't pip-installed — `make lint_all` runs them in Docker. |

### From Git source

Clone the repository, check out the desired commit/branch, and install from
your local checkout:

```shell
uv venv .venv && source .venv/bin/activate
uv pip install -e '.[<EXTRAS>]'
```

### Packaging for a remote computer

```shell
uv venv .venv && source .venv/bin/activate
uv build --sdist
```

This creates a distribution in `dist/` that you can install on the remote
machine.

### For development

Install in editable mode so you can make changes and re-run FAITH without
reinstalling. This may require setting the SCM version:

```shell
uv venv .venv && source .venv/bin/activate
SETUPTOOLS_SCM_PRETEND_VERSION=0.0.0 uv pip install -e '.[<EXTRAS>]'
```

## Shell autocompletion (optional)

After installing, you can add Bash autocomplete for the `faith` command:

```shell
activate-global-python-argcomplete
cat <<'EOF' >>~/.bashrc

# Autocomplete functions for benchmark commands.
eval "$(register-python-argcomplete faith)"
EOF
source ~/.bashrc
```

## Next steps

- [Run your first benchmark](./quickstart.md)
- [Troubleshooting & FAQ](./troubleshooting.md)
