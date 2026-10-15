// Shared data for the storage-layout explorations.
export const SEGMENTS = [
  {
    id: "root",
    label: "<benchmark-root-dir>",
    ph: true,
    short: "benchmark-root-dir",
    title: "Root directory",
    text: "The --datastore-location root containing one or more experiment directories.",
  },
  {
    id: "bench",
    label: "<benchmark-name>",
    ph: true,
    short: "benchmark-name",
    title: "Benchmark name",
    text: "The benchmark definition's metadata.name, for example ctibench-mcqa.",
  },
  {
    id: "model",
    label: "<model-name>",
    ph: true,
    short: "model-name",
    title: "Model name",
    text: "The model's configured name; otherwise, its full path with unsafe characters replaced by underscores.",
  },
  {
    id: "prompt",
    label: "<prompt-format>",
    ph: true,
    short: "prompt-format",
    title: "Prompt format",
    text: "How prompts are formatted for the model: base or chat.",
  },
  {
    id: "gen",
    label: "<generation-mode>",
    ph: true,
    short: "generation-mode",
    title: "Generation mode",
    text: "How the model generates its response: logits, next_token, or chat_comp.",
  },
  {
    id: "shots",
    label: "<num-shots>_shot",
    ph: true,
    short: "num-shots",
    title: "Number of shots",
    text: "In-context examples per query, as a literal suffix — 0_shot, 5_shot. An N/D n-shot ratio becomes N_D_shot.",
  },
  {
    id: "params",
    label: "gen_params_<gen-params-hash>",
    ph: true,
    short: "gen-params-hash",
    title: "Generation-params hash",
    text: "Hash of the model's generation parameters. A change to these parameters selects a different experiment directory.",
  },
  {
    id: "agentic",
    label: "agentic_<agentic-config-hash>",
    ph: true,
    short: "agentic-config-hash",
    title: "Agentic-config hash",
    text: "Agentic benchmarks only: hash of the authored agentic_config. Deployment overrides from --environment-config are not part of this hash. Absent for Q&A benchmarks.",
  },
];

// Leaf contents inside the innermost path segment
export const LEAVES = [
  {
    id: "experiment",
    label: "experiment.json",
    title: "experiment.json",
    text: "Describes how the benchmark was run and its parameters. Not created until all trials complete.",
  },
  {
    id: "metrics",
    label: "metrics.json",
    title: "metrics.json",
    text: "Per-trial metrics and across-trial statistics computed by eval from available records.",
  },
  {
    id: "trials",
    label: "trials/",
    title: "trials/",
    text: "Q&A records or agentic trajectories, organized by trial seed and benchmark-spec hash.",
  },
];
