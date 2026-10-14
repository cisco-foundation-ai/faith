// Facts for the flow-concept diagrams.
//
// Every `ref` here is a real path under src/faith/, which both fact checkers
// verify path by path. Nothing is invented for the sake of the drawing: the shapes
// differ because the code paths differ, and the point of the diagram is to show
// that rather than assert it.
//
// Each body describes its own path and nothing else. The reader can see the two
// lanes side by side, so prose that defines one kind by what the other does ("not
// a choice", "unlike Q&A", "deliberately skips") is narrating the comparison the
// layout already makes — and it reads as defensive. Describe the thing; let the
// shapes do the contrast.
//
// Paths only — no `:line` suffixes. A line number is the one thing on a card that
// can go stale while every check stays green, since a checker can confirm a path
// exists but not that a given line still holds what the card says. Where the line
// is genuinely useful it goes in a comment here, for whoever maintains this file,
// rather than on the card, where it is unverifiable detail the reader can't click.

// Shared entry and summary stages. The lanes show benchmark-specific query
// and grading operations; experiment storage and scoring utilities are shared.
export const TRUNK = {
  head: {
    title: "faith query",
    ref: "cli/subcmd/query.py",
    body: "Resolves experiments and selects each benchmark's trial runner. Benchmarks with agentic_config use the agentic runner.",
  },
  tail: {
    title: "faith summarize",
    ref: "cli/subcmd/summarize.py",
    body: "Reads saved metrics and experiment metadata for table or CSV output, or BigQuery ingestion. Does not regenerate answers or recompute grades.",
  },
};

// Each lane is grouped into the two phases that actually diverge — `faith query`
// and `faith eval`. Grouping matters: eval is a second, separate divergence, not
// a continuation of the first, and drawing one flat column of cards per lane hid
// that. The third phase, `summarize`, is the shared trunk both lanes rejoin at.

// The Q&A lane: one exchange per sample, then grade the text it produced.
export const QA_LANE = {
  id: "qa",
  label: "Q&A",
  shape: "one pass",
  // QABenchmark (benchmark.py:77) samples a dataset, renders a prompt, reads one
  // reply. The line is here rather than in the label: see the note at the top.
  ref: "benchmark/benchmark.py",
  phases: [
    {
      id: "query",
      label: "faith query",
      steps: [
        {
          key: "engine",
          title: "Select an engine",
          ref: "model/factory.py",
          body: "Chat completions are supported by all four engines. The logits and next_token modes require in-process vLLM and a compatible benchmark.",
          chips: ["vllm", "openai", "openrouter", "sagemaker"],
        },
        {
          key: "prompt",
          title: "Render the prompt",
          ref: "record_pipelines/formatting.py",
          body: "Formats each selected sample with the benchmark's templates and any configured few-shot examples.",
          // No chips. A chip earns its place by naming something the body can't:
          // which four engines, which two environments, the actual metric
          // identifiers. "dataset / format / n-shot" were the three nouns from the
          // sentence above set in mono, which reads like config keys and isn't.
        },
        {
          key: "reply",
          title: "Generate a response",
          ref: "record_pipelines/prediction.py",
          body: "Records a completion, next-token prediction, or logits for each selected sample. Existing outputs may be reused under the replacement strategy.",
          accent: true,
        },
      ],
      // What phase 1 leaves on disk, which is what phase 2 has to work from.
      hands: "query/response records",
    },
    {
      id: "eval",
      label: "faith eval",
      steps: [
        {
          key: "grade",
          title: "Extract and grade",
          ref: "record_pipelines/grading.py",
          body: "Extracts a prediction from the saved output and grades it against the label using the configured scoring and answer-format rules.",
        },
        {
          key: "metrics",
          title: "Benchmark metrics",
          ref: "benchmark/qa/",
          body: "Aggregates supported accuracy, F1, formatting, and custom-score metrics. The available metrics depend on the benchmark type.",
          // No chips: the body already lists the scorers, so a pair of them repeated
          // in mono was a subset of the sentence above. A chip is for what the prose
          // can't say.
        },
      ],
      hands: "metrics.json",
    },
  ],
};

// The agentic lane: a loop, and the loop is the whole point.
export const AGENTIC_LANE = {
  id: "agentic",
  label: "Agentic",
  shape: "a loop",
  // AgenticBenchmark (agentic.py:43) subclasses the neutral Benchmark base, not
  // QABenchmark.
  ref: "benchmark/agentic.py",
  phases: [
    {
      id: "query",
      label: "faith query",
      steps: [
        {
          key: "engine",
          title: "Validate the endpoint",
          // The rejection is at query.py:301.
          ref: "cli/subcmd/query.py",
          body: "Requires the openai engine and chat format, then probes the endpoint for structured tool calls before executing tasks.",
          chips: ["openai"],
        },
        {
          key: "env",
          title: "Build the environment",
          ref: "agentic/orchestration.py",
          body: "Constructs the configured environment and tool registry. The trial runner sets up and tears down the environment for each task that executes.",
          chips: ["docker", "kubernetes"],
        },
        {
          key: "loop",
          title: "ReAct loop",
          ref: "agentic/core/react_loop.py",
          body: "Sends the conversation to the model, executes requested tools, and includes their results in subsequent turns. Stops on submission, a configured limit, or an error.",
          loop: true,
        },
      ],
      // Nothing is graded during an agentic query: the loop's output goes to
      // disk raw (experiment/trial_runner.py:110-118).
      hands: "trajectories, unscored",
    },
    {
      id: "eval",
      label: "faith eval",
      steps: [
        {
          key: "extract",
          title: "Read the submission",
          ref: "agentic/extractors/",
          body: "The benchmark's extractor interprets the recorded answer and terminal-tool name. In vulnerability localization, an explicit abstention is an empty file list; no submission is a missing prediction.",
        },
        {
          key: "metrics",
          title: "Custom scores",
          // AgenticMetricsAggregator, agentic.py:118.
          ref: "benchmark/agentic.py",
          body: "Applies the benchmark's score functions and aggregates scores, answer-format rates, errors, and termination reasons. Recorded execution errors are excluded from performance scores.",
          // Real primary metrics: vulnerability-localization-a uses
          // file_f1, vulnerability-localization-b uses true_negative_rate.
          chips: ["file_f1", "true_negative_rate"],
          accent: true,
        },
      ],
      hands: "metrics.json",
    },
  ],
};

// The loop's two internal beats, drawn as a cycle. One word each: they sit inside
// the ring's stroke, and anything longer would have to cross it.
export const LOOP_BEATS = [
  { key: "think", label: "think" },
  { key: "act", label: "act" },
];

// All StoppedReason values from _types/record/agent.py. A submission may still
// extract incorrectly or score zero; termination is not a correctness judgment.
export const EXITS = [
  {
    key: "submitted",
    label: "submitted",
    note: "terminal-tool submission",
    answer: true,
  },
  {
    key: "no_tool_calls",
    label: "no_tool_calls",
    note: "consecutive turns without tool calls",
  },
  { key: "max_calls", label: "max_calls", note: "tool-call budget exhausted" },
  { key: "max_turns", label: "max_turns", note: "model-turn limit reached" },
  {
    key: "timeout",
    label: "timeout",
    note: "task timeout reached between turns",
  },
  { key: "error", label: "error", note: "model, setup, or execution error" },
  {
    key: "environment_error",
    label: "environment_error",
    note: "environment unavailable or recovery failed",
  },
];
