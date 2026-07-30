// Shared data for the "how a query flows" diagram explorations.
// Each phase maps to a CLI subcommand; steps are the components a query
// passes through, with the source file/dir each lives in.
export const PHASES = [
  {
    id: "query",
    label: "faith query",
    blurb: "Run models over benchmarks, collect raw records.",
    steps: [
      {
        role: "Entry point",
        title: "CLI",
        ref: "cli/cli.py",
        body: "Parses arguments and dispatches to the query command.",
        io: "args",
        in: "CLI args",
        out: "dispatch",
      },
      {
        role: "Setup",
        title: "Model factory",
        ref: "model/factory.py",
        body: "Builds the right engine (vLLM, OpenAI, OpenRouter, SageMaker) from --model-engine or the config.",
        io: "engine",
        in: "dispatch",
        out: "engine",
        // Decision: exactly one engine is constructed per run.
        branch: {
          on: "--model-engine or config",
          kind: "select-one",
          options: [
            { label: "vLLM", note: "local GPU" },
            { label: "OpenAI", note: "hosted API" },
            { label: "OpenRouter", note: "routed API" },
            { label: "SageMaker", note: "AWS endpoint" },
          ],
        },
      },
      {
        role: "Data source",
        title: "Benchmark loading",
        ref: "benchmark/load.py",
        body: "Loads benchmark(s) from disk — built-in or --custom-benchmarks.",
        io: "prompts",
        in: "engine",
        out: "prompts",
        // Decision: benchmarks come from one of two sources (or both).
        branch: {
          on: "benchmark source",
          kind: "from-either",
          options: [
            { label: "built-in", note: "__benchmarks__/" },
            { label: "custom", note: "--custom-benchmarks" },
          ],
        },
      },
      {
        role: "Core loop",
        title: "Record pipelines",
        ref: "record_pipelines/",
        body: "Format prompts, generate predictions, and write query/response records.",
        io: "records",
        in: "prompts",
        out: "records",
      },
      {
        role: "Persistence",
        title: "Experiment layer",
        ref: "experiment/",
        body: "Records run metadata and organizes logs into the storage hierarchy.",
        io: "logs",
        in: "records",
        out: "logs/",
        to: "/docs/architecture/storage",
      },
    ],
  },
  {
    id: "eval",
    label: "faith eval",
    blurb: "Read the logs, compute metrics against ground truth.",
    steps: [
      {
        role: "Scoring",
        title: "Metrics",
        ref: "record_pipelines/grading.py",
        body: "Reads the stored logs and computes metrics against ground truth.",
        io: "metrics",
        in: "logs/",
        out: "metrics",
        detail: [
          "Loads the query/response records written during the query phase.",
          "Grades predictions against ground truth via record_pipelines/grading.py.",
          "Emits per-benchmark metrics using _internal/metrics/.",
        ],
      },
    ],
  },
  {
    id: "summarize",
    label: "faith summarize",
    blurb: "Collate metrics across runs; optionally push to BigQuery.",
    steps: [
      {
        role: "Reporting",
        title: "Aggregation",
        ref: "experiment/summarize.py",
        body: "Collates metrics across the run — optionally ingesting into BigQuery.",
        io: "summary",
        in: "metrics",
        out: "summary",
        detail: [
          "Collates the per-benchmark metrics across the run into one summary via experiment/summarize.py.",
          "Optionally ingests the results into BigQuery via ingestion/parser.py.",
        ],
        // Optional terminal branch: results can additionally flow to BigQuery.
        branch: {
          on: "optional",
          kind: "optional-sink",
          options: [{ label: "BigQuery", note: "ingestion/parser.py" }],
        },
      },
    ],
  },
];

// Expanded variant: phase 1 (query) is unchanged, but the eval and summarize
// phases — a single node each in PHASES — are broken into their real internal
// sub-steps so all three phases carry comparable weight. Every `ref` is a real
// path under src/faith/, matching what faith eval / faith summarize actually
// import; nothing is invented.
export const PHASES_EXPANDED = PHASES.map((phase) => {
  if (phase.id === "eval") {
    return {
      ...phase,
      steps: [
        {
          role: "Scoring",
          title: "Grade",
          ref: "record_pipelines/grading.py",
          body: "Reads the stored query/response records and grades each prediction against the benchmark's ground truth.",
          io: "graded",
          in: "logs/",
          out: "graded",
          to: "/docs/architecture/storage",
        },
        {
          role: "Output",
          title: "Compute metrics",
          ref: "_internal/metrics/",
          body: "Aggregates the graded results into per-benchmark metrics.",
          io: "metrics",
          in: "graded",
          out: "metrics",
        },
      ],
    };
  }
  if (phase.id === "summarize") {
    return {
      ...phase,
      steps: [
        {
          role: "Reporting",
          title: "Collate",
          ref: "experiment/summarize.py",
          body: "Collates the per-benchmark metrics across the run into a single summary.",
          io: "summary",
          in: "metrics",
          out: "summary",
        },
        {
          role: "Optional sink",
          title: "BigQuery ingest",
          ref: "ingestion/parser.py",
          body: "Optionally ingests the summarized results into BigQuery for analysis.",
          io: "bigquery",
          in: "summary",
          out: "BigQuery",
          optional: true,
        },
      ],
    };
  }
  return phase;
});
