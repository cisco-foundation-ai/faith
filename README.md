<!--
Copyright 2025 Cisco Systems, Inc. and its affiliates

SPDX-License-Identifier: Apache-2.0
-->

# Foundation-AI's Test Hub/Harness (FAITH)

Welcome to **Foundation-AI's Test Hub/Harness (FAITH)**;
a framework for running benchmarks for language models.

## About The Project

FAITH provides a uniform way to run several open-source benchmarks
for benchmarking the knowledge and capabilities of language models in the
cybersecurity domain. FAITH currently runs benchmarks for the following
cybersecurity benchmarks:

- [AthenaBench](https://arxiv.org/pdf/2511.01144)
- [CTIBench](https://arxiv.org/abs/2406.07599)
- [CyberMetric](https://arxiv.org/abs/2402.07688)
- [MMLU](https://arxiv.org/abs/2009.03300)
- [SecBench](https://arxiv.org/abs/2412.20787)
- [SecEval](https://xuanwuai.github.io/SecEval/)

FAITH uses [vLLM](https://github.com/vllm-project/vllm) for querying
[HuggingFace](https://huggingface.co/) models as well as providing
API-based querying for [OpenAI models](https://platform.openai.com/docs/models).

## Getting Started

To benchmark models using FAITH, follow these simple steps.

### Prerequisites

We generally recommend running FAITH with 1 or more GPUs when using vLLM.
vLLM can be installed to run on CPU-only machines, but we do not
support this directly. Querying models on CPU-only machines may be very
slow. However, for API-based querying (e.g. OpenAI), GPUs are not required.

FAITH can run with Python 3.10 - 3.12, but we recommend using Python 3.12.
We also recommend using cloud storage for storing and combining benchmarks.

#### Configuring a Remote VM to Run FAITH

For VMs using Ubuntu images, we recommend using the following steps
(or equivalent) to install the tools for running FAITH:

```shell
# Setup Python 3.12
sudo apt update && \
sudo apt install software-properties-common -y && \
sudo add-apt-repository ppa:deadsnakes/ppa -y

DEBIAN_FRONTEND=noninteractive sudo apt update && \
sudo apt install -y python3.12 python3.12-venv python3.12-dev

# Make Python3.12 the default Python interpreter
sudo update-alternatives --install /usr/bin/python python /usr/bin/python3.12 1 && \
sudo update-alternatives --set python /usr/bin/python3.12

# GCloud Installation [Optional].
curl -fsSL https://packages.cloud.google.com/apt/doc/apt-key.gpg \
  | sudo gpg --dearmor \
  | sudo tee /etc/apt/keyrings/cloud.google.gpg > /dev/null && \
sudo chmod a+r /etc/apt/keyrings/cloud.google.gpg && \
echo "deb [signed-by=/etc/apt/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" \
  | sudo tee /etc/apt/sources.list.d/google-cloud-sdk.list > /dev/null && \
sudo apt update && \
sudo apt install -y google-cloud-cli

gcloud init

# Login Setup [Optional].
# Export any tokens (e.g. OpenAI, HuggingFace, etc.) in your .bashrc config.
echo 'export HF_TOKEN="hf_000000000000000000000000"' >> ~/.bashrc
source ~/.bashrc
```

### Installing the FAITH Package

Most commonly you should install FAITH from PyPI using pip as follows

```shell
TODO(blaine): Fill in once we have a PyPI distribution.
```

To properly use FAITH, you'll want to install it with `<EXTRAS>` packages to
enable most functionality. These are broken out as separate extras to allow
for smaller installs for specific use-cases. The extra packages available
for FAITH are:

- `openai`: installs dependencies to query OpenAI models using the
  [openai API](https://github.com/openai/openai-python).
  Note: Benchmarking OpenAI models will be billed to your OpenAI account;
  you must export your OpenAI key as the environment variable `OPENAI_API_KEY`
  to query models using their API.
- `vllm`: installs dependencies to run models with
  [vLLM](https://docs.vllm.ai/en/latest/).
  This can run many models from HuggingFace.
- `all-engines`: installs dependencies to run models with all of the
  following model engine backends [large install].
- `metrics`: installs the dependencies required
  to compute metrics from the benchmark logs.
- `all`: installs all the above dependencies for querying and evaluating
  any supported model [large install].
- `test`: install the `all` dependencies
  and additional lint and pytest dependencies [large install].

After installing, you can also add Bash autocomplete for the benchmark script
by running the following commands:

```shell
activate-global-python-argcomplete
cat <<'EOF' >>~/.bashrc

# Autocomplete functions for benchmark commands.
eval "$(register-python-argcomplete faith)"
EOF
source ~/.bashrc
```

#### From Git Source

To install from a particular commit/branch of the faith package,
clone the repository and checkout the desired commit/branch.
You can then install directly from your local repository:

```shell
python -m venv .venv && source .venv/bin/activate && pip install --upgrade pip
pip install -e '.[<EXTRAS>]'
```

See above for the `<EXTRAS>` packages.

#### Package for a Remote Computer

```shell
python -m venv .venv && source .venv/bin/activate && pip install --upgrade pip
pip install build
python -m build . --sdist
```

This will create a distribution in the folder `dist/`
from which you can install the package.

#### For Development

When installing for devolopment purposes, you can install the package in
editable mode from source. Installing in this way allows you to make alterations
and re-run faith without re-installing.
This may require altering SCM's versioning:

```shell
python -m venv .venv && source .venv/bin/activate && pip install --upgrade pip
SETUPTOOLS_SCM_PRETEND_VERSION=0.0.0 pip install -e '.[<EXTRAS>]'
```

See above for the `<EXTRAS>` packages.

### Step 2: Open Long-Lived Sessions for Benchmarking

Benchmarks can take a lot of time to run. To ensure your session isn't lost
due to network disconnect, you should run your benchmarks in a long-lived
terminal session.

#### Screen

You can open sessions with name `NAME` by running this command from your local
machine:

```shell
ssh -t user@REMOTE "screen -RdS NAME"
```

This creates a session called `NAME` if it doesn't already exist or connects
to it if it does exist (previous connections to this session will be
disconnected).

You can manually disconnect by typing `ctrl-a d`. You can scroll within a
session by typing `ctrl-a ESC`. If your SSH session is frozen you can try
to disconnect from it by typing `~ .`.

### Step 3: Run Benchmark Commands

Once connected to `REMOTE` in a long-lived session, load your virtual
environment:

```sh
source .venv/bin/activate
```

You can then use the following commands:

- `faith query`: queries models over benchmarks. This requires one of
  the following optional dependencies depending on what sort of model engine
  framework you want to use to query the model:
  - `all-engines`: installs dependencies to run models with all of the
    following model engine backends (large install).
  - `openai`: installs dependencies to query OpenAI models using the
    [openai API](https://github.com/openai/openai-python).
    Note: Benchmarking OpenAI models will be billed to your OpenAI account;
    you must export your OpenAI key as the environment variable `OPENAI_API_KEY`
    to query models using their API.
  - `vllm`: installs dependencies to run models with
    [vLLM](https://docs.vllm.ai/en/latest/).
    This can run many models from HuggingFace.
- `faith eval`: analyze benchmark runs. Requires the `[metrics]` option
  when you install the FAITH package.
- `faith summarize`: aggregate benchmark stats. Requires the `[metrics]` option
  when you install the FAITH package.

See the commands below for how to run each of these commands.

## Command Summary

The primary way to run FAITH's benchmarks is through the `faith` command,
which is installed in your `venv` through **pip**.
This command has the following subcommands and flags:

### The `query` subcommand

The `query` subcommand runs benchmarks on models and collects the input/output
records each into a set of logs which are stored in a datastore.
This command only records the prompts and responses of the model,
which need to be processed by subsequent commands.

<details>
<summary>Model Flags</summary>

- `--model-paths` [list[Path|Name]]: a list of model paths / names you want
  to query. All the listed models must be of the same model-type.
  You must provide the required credentials for your models by exporting
  environment variables. [Required]
  - Note: You can annotate each path with `@name=<NAME>` to give
    a shortened name for the model in the benchmarking logs.
  - Note: You can annotate each path with `@tokenizer=Path/Name`
    to specify an alternative tokenizer to use with the model.
- `--model-engine` [`openai`, `vllm`]: the backend engine to drive the model.
  [Required]
- `--num-gpus` [int]: the number of GPUs used by vLLM for querying LLMs.
  [Default: 1]
- `--model-context-len` [int]: the context length for the models.
  [Default: 3500]
- `--max-completion-tokens` [int]: the maximum number of new tokens for model
  to generate in its answer. [Default: 500]
- `--temperature` [float]: the temperature for sampling.
  [Default: 0.0 (deterministic)]
- `--top-p` [float]: top-p for sampling.
  [Default: 1.0 (deterministic)]

#### Model Engine Flags

Each framework (OpenAI, vLLM, etc.) use a number of custom key-word-args
when instatiating their clients. We support (limited) ability to set these
key-word-args from the command-line through the flag `--engine-kwargs`
followed by a list of key-value pairs of the form:

```shell
--engine-kwargs key1=True key2=5 key3='"foo"'
```

Note: the values are converted to Python types through `ast.literal()`
so strings must be passed as escape quoted strings like `'"bar"'`.

As an example, to pass special quantization flags to vLLM,
you can do the following:

```shell
--engine-kwargs dtype=torch.bfloat16 trust_remote_code=True quantization='"bitsandbytes"'
```

Similarly we support (limited) ability to set key-word-args to
the text-generation method of the model through the flag `--generation-kwargs`
which also takes a list of key-value pairs, like:

```shell
--generation-kwargs reasoning_effort='"high"'
```

</details>

<details>
<summary>Experiment Flags</summary>

- `--benchmarks` [list[string]]: a list benchmarks to query each model over.
  [One of --benchmarks or --custom-benchmarks is required]
- `--custom-benchmarks` [list[Path]]: paths to custom benchmark directories
  to query over. [One of --benchmarks or --custom-benchmarks is required]
  - Note: You can annotate each path with `@name=<NAME>` to give a shortened
    name for the benchmark in the logs.
- `--generation-mode` [`logits`, `next_token`, `chat_comp`]: the generation mode
  to use for the model. [Required]
- `--prompt-format` [`base`, `chat`]: the prompt format to use for the model.
  [Required]
- `--n-shot` [list[int]]: the number of in-context examples per question.
  [Defalut: 0]
- `--num-trials` [int]: the number of trials to run for the benchmark.
  [Default: 1]
- `--seed` [int]: the random seed for reproducibility. [Default: 373,363]
- `--sample-size` [int]: the sample size for the benchmark.
  If not supplied, no sampling occurs. [Default: None]

</details>

<details>
<summary>Other Flags</summary>

- `--datastore-location` [Path|GCP-URI]: The location for the datastore where
  benchmarking logs are stored.
  This can be a local directory or a GCP URI (beginning with gs://).
  If a GCP URI is provided, logs will be stored in a tmp dir
  and periodically uploaded to GCP.

</details>

### The `eval` subcommand

The `eval` subcommand computes metrics over a set of benchmark's logs.

<details>
<summary>Required Flags</summary>

- `--experiment-path` [Path]: The path to the experiment logs to analyze.
  If a directory is provided,
  all experiments within that directory will be analyzed. [Required]

</details>

<details>
<summary>Eval Behavior Flags</summary>

- `--cache-prediction-stats` [bool]: whether to store prediction statistics
  in the original logs to avoid recomputing in the future. [Default: true]
- `--force-compute-stats` [bool]: forces recomputation of stats even
  if they are already present in the logs. [Default: false]

</details>

### The `summarize` subcommand

The `summarize` (or `digest`) subcommand coallates all benchmark metrics
in a given directory into a summary table.

<details>
<summary>Required Flags</summary>

- `--experiment-path` [Path]: The path to the experiment logs to analyze.
  If a directory is provided,
  all experiments within that directory will be analyzed. [Required]

</details>

<details>
<summary>Summarize Behavior Flags</summary>

- `--stats` [list[string]]: The stats to summarize.
  This should be a comma-separated list of metric names.
- `--summary-filepath` [Path]: The path to the output directory
  where the summary will be saved. [Default: None]

</details>

### The `run-all` subcommand

The `run-all` (or `qed`) subcommand sequentially runs the `query`, `metric`
and `summarize` stages. It uses the **Model Flags**, **Experiment Flags**,
and `--datastore-location` flag from `query`, the **Eval Behavior Flags**
from `eval` and the **Summary Behavior Flags** from `summarize`.

## Benchmark Results

After running your benchmarks, the logs should be stored in long-term storage
e.g. a GCP bucket.

### Benchmark Upload

TODO(blaine)

### Benchmark Download

An entire benchmark test may be enormous (100's GB)

**Note**: All logs from a large benchmark test can be extensive as they
include all queries and responses over multiple benchmark, models, and trials.
Downloading an entire benchmark test may take hours and use 100's of GB of disk
space. It is recommended that you only download subtrees of interest in most
cases.

### Storage Structure

Each benchmark run is stored according to the following file structure:

```plaintext
<benchmark-root-dir>
└── <benchmark-name>
    └── <model-name>
        └── <prompt-format>
            └── <generation-mode>
                └── <num-shots>
                    └── gen_params_<gen-params-hash>
                        ├── `experiment.json`
                        ├── `metrics.json`
                        └── `trials`
                            └── <trial-seed>
                                └── <benchmark-trial-hash>
                                    └── `benchmark-log.json`
```

Components of the main path:

- `benchmark-root-dir` - the root directory.
- `benchmark-name` - the name of the benchmark / dataset used for testing.
- `model-name` - the name / path of the model being tested.
- `prompt-format` - the method by which the prompts are formatted for the model.
- `generation-mode` - the method by which the model is to generate its response.
- `num-shots` - the number of examples provided as examples in each query.
- `gen-params-hash` - a hash of the generation params used for the model, used
  to prevent clobber other logs when changing generation parameters of a model
  (e.g. temperature).

Files in the benchmark directory:

- **experiment.json** - A file describing how the benchmark was run
  and its parameters.
  This file is not created until the benchmark trials all complete.
- **metrics.json** - A file with metrics summarizing all trials as well as
  aggregate statistics for the entire run.

The `trials` sub-folder contains a record of all benchmark query/reponses in the
course of the run. Its subfolder structure contains a `trial-seed` and
`benchmark-trial-hash` both of which are identifiers for the trial -- the seed
allows you to compare similar trials (exactly the same base questions) across
runs, whereas the hash is useful when you want to compare exactly the same LLM
queries between models but is restricted to only being useful useful when the
benchmark parameters are identical.

The file for each trial is:

- **benchmark-log.json**: a list of records for every query-response executed
  from the benchmark on the model.

### Dowloading Metrics Only

For the purposes of examining a benchmark,
you only need the metrics files from that benchmark.
To download all metrics for a benchmark from GCP,
you can do so efficiently by running the following command:

```sh
gcloud storage rsync -R -J -P -x '.*/benchmark-log\.json$' gs://<BENCHMARK_PATH> ~/benchmarks/<BENCHMARK>
```

replacing `<BENCHMARK_PATH>` with the name of the benchmark test
you're interested in.
This will skip all `benchmark-log.json` files in the benchmark,
only fetching the experiment descriptions, benchmark metrics, and summary files.

## Creating New Benchmarks

To use the FAITH framework to run a new benchmark,
construct a new local folder for your benchmark.
This folder will contain a `benchmark.yaml` file for the benchmark's definition
as well as any required benchmark resources
(e.g. data folders or auxiliary configurations).
Once you create your benchmark configuration (see below for details),
you can run queries over any model over your custom benchmark
using the `--custom-benchmarks` flag.
For example, if the custom benchmark's folder is `${HOME}/my-custom-benchmark`,
you could query the `fdtn-ai/Foundation-Sec-8B-Instruct` model on it by running:

```sh
faith query --custom-benchmarks ${HOME}/my-custom-benchmark@name=my-bench \
--model-paths fdtn-ai/Foundation-Sec-8B-Instruct@name=foundation-sec-8B-it \
--model-engine vllm --model-context-len 4096 --max-completion-tokens 1000 \
--num-trials 1 --prompt-format chat --generation-mode chat_comp --num-gpus 1 \
--sample-size 25 --n-shot 0 --datastore-location out/
```

This will query the `fdtn-ai/Foundation-Sec-8B-Instruct` model
over a sample of 25 questions from this custom benchmark
and save resuts to the local folder `out/`.

### Structure of a Benchmark Config

A benchmark configuration has the following structure:

```plaintext
benchmark:
├── metadata:
    └── ...
├── source:
    └── ...
├── mcqa_config / saqa_config:
    └── ...
├── format:
    └── ...
└── output_processing:
    └── ...
```

<details>
<summary>Benchmark Metadata</summary>

Most fields of the metadata are not used by FAITH.
However, two that are used by FAITH are:

- `state`: The current status of the benchmark
  (_enabled_, _experimental_, _disabled_, _test_only_).
  This controls the allowed set of benchmarks amongst those incorporated
  into FAITH as standard benchmarks.
- `categories`: A list of categories that describe the benchmark
  (e.g. _security_ or _reasoning_).

While not generally used by FAITH, the metadata section
provides information about the benchmark for others using it.
It contains fields used to categorize and describe the benchmark, such as:

- `name`: The given name used to refer to the benchmark.
- `description`: A description of the benchmark's data and purpose.
- `license`: Any license associated with the benchmark.
- `urls`: Source locations / documentation for the benchmark.

</details>

### Specification of Source Data

#### Required Columns of the Loaded Data

For multiple-choice benchmarks,
the final dataframe must have the following columns:

- `question`: A string representing the question only (without answer choices).

- `choices`: A list of strings for the multiple-choice options.

- `answer`: A single capital letter representing the correct answer by the
  implied order given in choices.

#### Modifying Data with the `dataframe_transform_expr`

## Roadmap

See the [open issues](https://github.com/cisco-foundation-ai/faith/issues)
for a list of proposed features (and known issues).

## Contributing

Contributions are what make the open source community such an amazing place to
learn, inspire, and create. Any contributions you make are **greatly
appreciated**. For detailed contributing guidelines, please see
[CONTRIBUTING.md](CONTRIBUTING.md)

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
