import { Fragment } from "react";
import Link from "@docusaurus/Link";
import clsx from "clsx";
import styles from "./styles.module.css";

// The benchmark suites FAITH ships with. This replaced a two-column Markdown
// table for two reasons:
//
//  - The Suite column was sized to its content, leaving ~85px of usable width,
//    and browsers break at hyphens: `CTI-Reasoning` and `CWE-Prediction` split
//    across two lines while the Description column held 693px.
//  - A table has nowhere to say that Vulnerability Localization is a different
//    *kind* of benchmark. That distinction is the main thing a reader needs
//    from this section, so it gets a column of its own.
//
// Names link to the catalog entry rather than out to the paper. The catalog
// carries the paper and dataset links for each suite, so an internal link
// keeps the reader in the docs and still gets them there in one more hop.
//
// `scripts/check_repo_facts.py` checks both the names and the kinds here against
// catalog.md's own sections, so this list can't fall behind the catalog.
const SUITES = [
  {
    name: "CTIBench",
    to: "/docs/benchmarks/catalog#ctibench",
    kind: "qa",
    text: "Cyber threat intelligence: technique extraction, MCQA, root-cause mapping, threat-actor attribution, vulnerability severity prediction.",
  },
  {
    name: "CyberMetric",
    to: "/docs/benchmarks/catalog#cybermetric",
    kind: "qa",
    text: "80 / 500 / 2,000 / 10,000-question multiple-choice cybersecurity sets.",
  },
  {
    name: "MMLU",
    to: "/docs/benchmarks/catalog#mmlu",
    kind: "qa",
    text: "Full MMLU plus a computer-security subset.",
  },
  {
    name: "SecBench",
    to: "/docs/benchmarks/catalog#secbench",
    kind: "qa",
    text: "English multiple-choice and reasoning-focused security questions.",
  },
  {
    name: "SecEval",
    to: "/docs/benchmarks/catalog#seceval",
    kind: "qa",
    text: "~1,000 multiple-choice computer-security questions.",
  },
  {
    name: "CTI-Reasoning",
    to: "/docs/benchmarks/catalog#cti-reasoning",
    kind: "qa",
    text: "200 reasoning-oriented cyber threat intelligence questions.",
  },
  {
    name: "CWE-Prediction",
    to: "/docs/benchmarks/catalog#cwe-prediction",
    kind: "qa",
    text: "3,000 CVE / GHSA advisory descriptions to map to CWE identifiers.",
  },
  {
    name: "Vulnerability Localization",
    to: "/docs/benchmarks/catalog#vulnerability-localization",
    kind: "agentic",
    text: "Locate the files a security fix must touch in real repositories, in two paired phases.",
  },
];

const KIND_LABEL = { qa: "Q&A", agentic: "agentic" };

/**
 * The shipped benchmark suites, one row each, tagged by kind.
 *
 * Every cell is a direct child of the one grid, with `Fragment` grouping a
 * suite's three cells without adding an element. A per-row wrapper element
 * would make each row its own grid, and `max-content` is resolved per grid — so
 * the tags and descriptions would start at a different x on every row, which is
 * the raggedness this component exists to fix.
 */
export default function SuiteList() {
  return (
    <div className={styles.list}>
      {SUITES.map((s) => (
        <Fragment key={s.name}>
          <Link to={s.to} className={styles.name}>
            {s.name}
          </Link>
          <span className={styles.kindCell}>
            <span
              className={clsx(
                styles.kind,
                s.kind === "agentic" && styles.kindAgentic,
              )}
            >
              {KIND_LABEL[s.kind]}
            </span>
          </span>
          <span className={styles.text}>{s.text}</span>
        </Fragment>
      ))}
    </div>
  );
}
