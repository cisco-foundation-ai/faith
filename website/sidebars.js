// @ts-check

// This runs in Node.js - Don't use client-side code here (browser APIs, JSX...)

/**
 * The FAITH documentation sidebar.
 *
 * Each category answers exactly one reader question — how do I start, what does
 * this command do, what benchmarks exist, how do I configure it, what did my run
 * produce, what does this term mean, it broke. Categories are not organized by
 * genre ("reference-y things") or by which tool happens to read a file.
 * Authoring walkthroughs distinguish Q&A and agentic work within one task-based
 * category; shared configuration remains in a single reference.
 *
 * @type {import('@docusaurus/plugin-content-docs').SidebarsConfig}
 */
const sidebars = {
  docsSidebar: [
    "intro",
    {
      type: "category",
      label: "Get started",
      collapsed: false,
      items: ["getting-started/installation", "getting-started/quickstart"],
    },
    {
      // Subcommand reference only. Config-file schemas and the serving guide are
      // separate genres and live under Configuration.
      type: "category",
      label: "CLI commands",
      items: ["cli/overview", "cli/query", "cli/eval", "cli/summarize"],
    },
    {
      type: "category",
      label: "Benchmarks",
      items: [
        "benchmarks/catalog",
        "benchmarks/groups",
        {
          type: "category",
          label: "Creating benchmarks",
          items: [
            "benchmarks/creating-qa-benchmarks",
            "benchmarks/creating-agentic-benchmarks",
            "benchmarks/creating-benchmarks",
            "reference/agentic-components",
          ],
        },
      ],
    },
    {
      // The three things to set up before a run: the model, the server in front
      // of it, and the sandbox the agent runs in.
      type: "category",
      label: "Configuration",
      items: [
        "configuration/model-configs",
        "configuration/serving-agentic-models",
        "configuration/environment-configs",
        "configuration/kubernetes-deployment",
      ],
    },
    {
      // Grouped by the reader's situation — "I have results, now what" — rather
      // than by which phase writes each file. Storage can't sit under a single
      // subcommand: `query` writes it, `eval` writes into it, `summarize` reads
      // it.
      type: "category",
      label: "Results & analytics",
      items: [
        "analytics/storage",
        "analytics/metrics",
        {
          type: "category",
          label: "BigQuery",
          items: [
            "analytics/bigquery/quickstart",
            "analytics/bigquery/schema",
            "analytics/bigquery/examples",
            {
              type: "category",
              label: "Dashboards",
              items: [
                "analytics/bigquery/dashboards/setup",
                "analytics/bigquery/dashboards/visualizations",
                "analytics/bigquery/dashboards/advanced",
              ],
            },
          ],
        },
      ],
    },
    {
      type: "category",
      label: "Reference",
      items: [
        "reference/concepts",
        "reference/architecture",
        "reference/sandbox-runtime",
      ],
    },
    // A whole-lifecycle diagnostic index, not onboarding material: its own
    // sections span installation through analytics. It is a page readers arrive
    // at from elsewhere, never one they read in sequence.
    "troubleshooting",
    "contributing",
  ],
};

export default sidebars;
