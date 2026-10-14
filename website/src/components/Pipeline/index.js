import clsx from "clsx";
import Link from "@docusaurus/Link";
import styles from "./styles.module.css";

const STEPS = [
  {
    n: "01",
    title: "Query",
    cmd: "faith query",
    body: "Execute selected samples or tasks and save responses or trajectories to the datastore.",
    to: "/docs/cli/query",
  },
  {
    n: "02",
    title: "Eval",
    cmd: "faith eval",
    body: "Grade saved records and aggregate metrics within and across trials.",
    to: "/docs/cli/eval",
  },
  {
    n: "03",
    title: "Summarize",
    cmd: "faith summarize",
    body: "Read experiment metrics for table or CSV output, or BigQuery ingestion.",
    to: "/docs/cli/summarize",
  },
];

/**
 * The three-phase FAITH pipeline, rendered as a connected, animated flow.
 * Used in place of ASCII box diagrams in the docs.
 */
export default function Pipeline() {
  return (
    <div className={styles.pipeline}>
      {STEPS.map((s, i) => (
        <div key={s.title} className={styles.stepWrap}>
          <Link to={s.to} className={styles.step} style={{ "--i": i }}>
            <span className={styles.num}>{s.n}</span>
            <span className={styles.title}>{s.title}</span>
            <code className={styles.cmd}>{s.cmd}</code>
            <span className={styles.body}>{s.body}</span>
          </Link>
          {i < STEPS.length - 1 && (
            <span
              className={clsx(styles.connector)}
              style={{ "--i": i }}
              aria-hidden="true"
            >
              <span className={styles.connectorLine} />
              <span className={styles.connectorArrow}>→</span>
            </span>
          )}
        </div>
      ))}
    </div>
  );
}
