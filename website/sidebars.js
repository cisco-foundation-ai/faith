// @ts-check

// This runs in Node.js - Don't use client-side code here (browser APIs, JSX...)

/**
 * The FAITH documentation sidebar.
 *
 * @type {import('@docusaurus/plugin-content-docs').SidebarsConfig}
 */
const sidebars = {
  docsSidebar: [
    "intro",
    {
      type: "category",
      label: "Get Started",
      collapsed: false,
      items: [
        "getting-started/installation",
        "getting-started/quickstart",
        "getting-started/troubleshooting",
      ],
    },
    {
      type: "category",
      label: "CLI Reference",
      items: [
        "cli/overview",
        "cli/query",
        "cli/eval",
        "cli/summarize",
        "cli/model-configs",
      ],
    },
    {
      type: "category",
      label: "Benchmarks",
      items: ["benchmarks/catalog", "benchmarks/creating-benchmarks"],
    },
    {
      type: "category",
      label: "Architecture",
      items: ["architecture/overview", "architecture/storage"],
    },
    {
      type: "category",
      label: "Analytics (BigQuery)",
      items: [
        "bigquery/quickstart",
        "bigquery/schema",
        "bigquery/examples",
        {
          type: "category",
          label: "Dashboards",
          items: [
            "bigquery/dashboards/setup",
            "bigquery/dashboards/visualizations",
            "bigquery/dashboards/advanced",
          ],
        },
      ],
    },
    {
      type: "category",
      label: "Reference",
      items: ["reference/concepts", "reference/metrics"],
    },
    "contributing",
  ],
};

export default sidebars;
