// Shared data for the storage-layout explorations.
export const SEGMENTS = [
  {
    id: "root",
    label: "<benchmark-root-dir>",
    ph: true,
    short: "benchmark-root-dir",
    title: "Root directory",
    text: "The root (your --datastore-location) under which every run is organized.",
  },
  {
    id: "bench",
    label: "<benchmark-name>",
    ph: true,
    short: "benchmark-name",
    title: "Benchmark name",
    text: "Name of the benchmark / dataset used for testing, e.g. ctibench-mcqa.",
  },
  {
    id: "model",
    label: "<model-name>",
    ph: true,
    short: "model-name",
    title: "Model name",
    text: "Name or path of the model being tested — the @name annotation if one was given.",
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
    text: "Hash of the model’s generation params — keeps logs from clobbering each other when parameters (e.g. temperature) change.",
  },
];

// Leaf contents inside gen_params_<hash>/
export const LEAVES = [
  {
    id: "experiment",
    label: "experiment.json",
    kind: "file",
    title: "experiment.json",
    text: "Describes how the benchmark was run and its parameters. Not created until all trials complete.",
  },
  {
    id: "metrics",
    label: "metrics.json",
    kind: "file",
    title: "metrics.json",
    text: "Metrics summarizing all trials, plus aggregate statistics for the entire run.",
  },
  {
    id: "trials",
    label: "trials/",
    kind: "dir",
    title: "trials/",
    text: "Records all query/responses, split by trial seed then hash.",
    children: [
      {
        id: "seed",
        label: "<trial-seed>",
        ph: true,
        title: "trial-seed",
        text: "The seed lets you compare similar trials (same base questions) across runs.",
      },
      {
        id: "trialhash",
        label: "<benchmark-trial-hash>",
        ph: true,
        title: "benchmark-trial-hash",
        text: "The hash lets you compare exactly the same LLM queries between models — useful only when benchmark parameters are identical.",
      },
      {
        id: "log",
        label: "benchmark-log.json",
        kind: "file",
        title: "benchmark-log.json",
        text: "A list of records for every query-response executed from the benchmark on the model.",
      },
    ],
  },
];
