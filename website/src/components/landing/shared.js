import { useEffect, useRef, useState } from "react";
import clsx from "clsx";
import Link from "@docusaurus/Link";
import Heading from "@theme/Heading";
import styles from "../../pages/index.module.css";

// ---------------------------------------------------------------------------
// Shared data
// ---------------------------------------------------------------------------
// `--datastore-location` is required, so leaving it out of the one command on the
// landing page would print an argparse error for anyone who copied it.
export const CMD_TEXT = `faith run-all \\
  --model-paths meta-llama/Llama-3.1-8B-Instruct \\
  --benchmarks ctibench-mcqa seceval \\
  --model-engine vllm \\
  --prompt-format chat \\
  --sample-size 25 \\
  --num-trials 1 \\
  --stats accuracy \\
  --datastore-location out/`;

export const RUN_OUTPUT = [
  { tag: "query", text: "querying 2 benchmarks · 1 model · logs → out/" },
  { tag: "eval", text: "grading responses and aggregating metrics" },
  { tag: "summarize", text: "reading saved metrics → summary table" },
];

// Example comparison across multiple runs, not measured scores or output from
// CMD_TEXT. Show each model on both benchmarks without a cross-benchmark ranking.
export const EVALUATION_RESULTS = [
  {
    model: "Llama-3.1-70B-Instruct",
    bench: "ctibench-mcqa",
    samples: 25,
    score: 0.92,
  },
  { model: "GPT-4.1", bench: "ctibench-mcqa", samples: 25, score: 0.88 },
  {
    model: "Foundation-Sec-8B",
    bench: "ctibench-mcqa",
    samples: 25,
    score: 0.84,
  },
  {
    model: "Llama-3.1-70B-Instruct",
    bench: "seceval",
    samples: 25,
    score: 0.84,
  },
  { model: "GPT-4.1", bench: "seceval", samples: 25, score: 0.8 },
  { model: "Foundation-Sec-8B", bench: "seceval", samples: 25, score: 0.76 },
];

const LINE_MS = 520;
const ROW_MS = 190;

// ---------------------------------------------------------------------------
// Animation hook — reveals the command, then output and evaluation results.
// ---------------------------------------------------------------------------
export function useRunAnimation() {
  // The command is always present; it fades in as a whole (not typed), then the
  // output streams and the results fill in.
  const [cmdIn, setCmdIn] = useState(false);
  const [lines, setLines] = useState(0);
  const [rows, setRows] = useState(0);
  const containerRef = useRef(null);
  const startedRef = useRef(false);

  useEffect(() => {
    const finished = () => {
      setCmdIn(true);
      setLines(RUN_OUTPUT.length);
      setRows(EVALUATION_RESULTS.length);
    };
    const prefersReduced =
      typeof window !== "undefined" &&
      window.matchMedia &&
      window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    if (prefersReduced) {
      finished();
      return undefined;
    }

    const timers = [];
    let cancelled = false;

    const run = () => {
      if (cancelled) return;
      setCmdIn(false);
      setLines(0);
      setRows(0);

      let t = 250;
      timers.push(setTimeout(() => setCmdIn(true), t));
      t += 700;
      for (let l = 1; l <= RUN_OUTPUT.length; l += 1) {
        timers.push(setTimeout(() => setLines(l), t));
        t += LINE_MS;
      }
      // Beat between the last output line and the results filling in.
      t += 450;
      for (let r = 1; r <= EVALUATION_RESULTS.length; r += 1) {
        timers.push(setTimeout(() => setRows(r), t));
        t += ROW_MS;
      }
    };

    const el = containerRef.current;
    const start = () => {
      if (startedRef.current) return;
      startedRef.current = true;
      run();
    };
    let observer;
    if (el && "IntersectionObserver" in window) {
      observer = new IntersectionObserver(
        (entries) => {
          if (entries.some((e) => e.isIntersecting)) start();
        },
        { threshold: 0.25 },
      );
      observer.observe(el);
    } else {
      start();
    }

    return () => {
      cancelled = true;
      timers.forEach(clearTimeout);
      if (observer) observer.disconnect();
    };
  }, []);

  return { cmdIn, lines, rows, containerRef };
}

// ---------------------------------------------------------------------------
// Reusable panels
// ---------------------------------------------------------------------------
export function TerminalPanel({ cmdIn, lines }) {
  return (
    <div className={styles.terminal}>
      <div className={styles.terminalBar}>
        <span className={styles.dot} data-color="red" />
        <span className={styles.dot} data-color="yellow" />
        <span className={styles.dot} data-color="green" />
        <span className={styles.terminalTitle}>bash — faith</span>
      </div>
      <pre className={styles.terminalBody}>
        <code>
          <span className={clsx(styles.cmdBlock, cmdIn && styles.cmdBlockIn)}>
            <span className={styles.prompt}>$ </span>
            <span className={styles.cmd}>{CMD_TEXT}</span>
          </span>
          {"\n\n"}
          {RUN_OUTPUT.map((l, index) => (
            <span
              key={l.tag}
              className={clsx(
                styles.outLine,
                index < lines && styles.outLineVisible,
              )}
            >
              <span className={styles.outTag}>[{l.tag}]</span>
              {"  "}
              {l.text}
              {"\n"}
            </span>
          ))}
        </code>
      </pre>
    </div>
  );
}

export function ResultsPanel({ rows }) {
  const modelCount = new Set(EVALUATION_RESULTS.map((r) => r.model)).size;
  const benchmarkCount = new Set(EVALUATION_RESULTS.map((r) => r.bench)).size;
  return (
    <div className={styles.board}>
      <div className={styles.boardHead}>
        <div className={styles.boardHeading}>
          <h2 id="evaluation-results-title" className={styles.boardTitle}>
            Evaluation results
          </h2>
          <span className={styles.boardExample}>Example</span>
        </div>
        <p className={styles.boardMeta}>
          {modelCount} models · {benchmarkCount} benchmarks · accuracy.mean
        </p>
      </div>
      <div
        className={styles.boardScroll}
        role="region"
        aria-label="Example evaluation results"
        tabIndex={0}
      >
        <table
          className={styles.boardTable}
          aria-labelledby="evaluation-results-title"
          aria-describedby="evaluation-results-note"
        >
          <caption id="evaluation-results-note" className={styles.boardCaption}>
            Example values, not measured results or output from the command
            above.
          </caption>
          <thead>
            <tr>
              <th scope="col" className={styles.boardModel}>
                Model
              </th>
              <th scope="col" className={styles.boardBench}>
                Benchmark
              </th>
              <th
                scope="col"
                className={clsx(styles.boardNumber, styles.boardSamples)}
              >
                Samples
              </th>
              <th
                scope="col"
                className={clsx(styles.boardNumber, styles.boardAccuracy)}
                title="accuracy.mean"
              >
                Accuracy
              </th>
            </tr>
          </thead>
          <tbody>
            {EVALUATION_RESULTS.map((r, i) => {
              const shown = i < rows;
              return (
                <tr
                  key={`${r.model}-${r.bench}`}
                  className={clsx(styles.boardRow, shown && styles.boardRowIn)}
                >
                  <th scope="row" className={styles.boardModel}>
                    {r.model}
                  </th>
                  <td className={styles.boardBench}>{r.bench}</td>
                  <td className={styles.boardNumber}>{r.samples}</td>
                  <td className={styles.boardNumber}>
                    {shown ? r.score.toFixed(3) : "—"}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
}

/**
 * The Foundation AI mark, inlined.
 *
 * One path, from the team's own `black-logo-no-text.svg` / `white-logo-no-text.svg`
 * — identical but for the fill, so this takes `currentColor` instead and needs
 * neither file nor a per-theme rule. Decorative: the link it sits in already says
 * the name.
 */
function FoundationMark() {
  return (
    <svg
      className={styles.heroMark}
      viewBox="0 0 82.968 82.968"
      aria-hidden="true"
      focusable="false"
    >
      <path d="M 82.968,0 V 82.968 H 0 V 0 Z M 13.235,69.732 H 42.013 V 41.483 h 27.719 V 13.235 H 13.235 Z" />
    </svg>
  );
}

/**
 * The outbound arrow on the GitHub button.
 *
 * Drawn rather than typed: `↗` (U+2197) sits outside every unicode-range the
 * Google Fonts stylesheet declares for IBM Plex Sans, so the browser substituted
 * it from a system font, at a weight and optical size that didn't match the label
 * beside it. This one strokes `currentColor`, so it tracks the button through both
 * themes and its hover state. Decorative: the label says where the link goes.
 */
function ExternalArrow() {
  return (
    <svg
      className={styles.btnArrow}
      viewBox="0 0 12 12"
      aria-hidden="true"
      focusable="false"
    >
      <path d="M3.25 8.75 8.75 3.25M4.75 3.25h4v4" />
    </svg>
  );
}

// Hero content pieces, so the hero's layout can wrap them without restating them.
export function HeroCopy() {
  return (
    <>
      <span className={styles.heroEyebrow}>
        FAITH ·{" "}
        <Link
          className={styles.heroEyebrowOrg}
          href="https://cisco-foundation-ai.github.io/"
        >
          <FoundationMark />
          Cisco Foundation AI
        </Link>
      </span>
      <Heading as="h1" className={styles.heroTitle}>
        The open harness for benchmarking cybersecurity models.
      </Heading>
      <p className={styles.heroTagline}>
        FAITH — Foundation-AI&apos;s Test Hub/Harness — provides a common
        interface to benchmarks that evaluate the cybersecurity knowledge and
        capabilities of language models.
      </p>
      {/* `.buttons` centers its row, which is what the closing CTA wants; here the
          hero's copy column is left-aligned, so this row is too. */}
      <div className={clsx(styles.buttons, styles.buttonsLeft)}>
        <Link
          className="button button--primary"
          to="/docs/getting-started/quickstart"
        >
          Quickstart
        </Link>
        {/* Labeled for the destination, not the content: /docs/intro is the
            documentation entry point, and its body — why FAITH, the three
            phases, the benchmark list — is what the sections below this hero
            already say. "How it works" promised a second telling of the page
            you are on. */}
        <Link className="button button--secondary" to="/docs/intro">
          Documentation
        </Link>
        <Link
          className="button button--secondary"
          to="/docs/benchmarks/catalog"
        >
          Benchmark catalog
        </Link>
      </div>
    </>
  );
}

// ---------------------------------------------------------------------------
// Shared lower sections (identical across variants)
// ---------------------------------------------------------------------------
const PHASES = [
  {
    step: "01",
    title: "Query",
    cmd: "faith query",
    // "Every exchange" rather than "every prompt and response": an agentic task
    // is many turns, so the narrower wording read as Q&A-only.
    body: "Records Q&A responses or agentic trajectories in the datastore. Q&A supports vLLM, OpenAI, OpenRouter, and SageMaker. Agentic execution requires the openai engine.",
    to: "/docs/cli/query",
  },
  {
    step: "02",
    title: "Eval",
    cmd: "faith eval",
    body: "Grades saved records and aggregates metrics within and across trials, writing metrics.json for each experiment.",
    to: "/docs/cli/eval",
  },
  {
    step: "03",
    title: "Summarize",
    cmd: "faith summarize",
    body: "Reads saved metrics and experiment metadata for table or CSV output, or BigQuery ingestion.",
    to: "/docs/cli/summarize",
  },
];

/**
 * The two benchmark kinds.
 *
 * Distilled from the two section intros in the catalog, which are the reference
 * text for this distinction — keep them in step. No suite or task counts here:
 * the catalog grows, and a hand-typed number on this page has nothing checking it.
 */
const KINDS = [
  {
    tag: "one exchange",
    title: "Q&A benchmarks",
    body: "Evaluate model responses to benchmark questions using benchmark-specific grading.",
    to: "/docs/benchmarks/catalog#qa-benchmarks",
  },
  {
    tag: "many turns",
    title: "Agentic benchmarks",
    body: "Evaluate models as they use tools to complete tasks in a configured execution environment.",
    to: "/docs/benchmarks/catalog#agentic-benchmarks",
  },
];

export function Kinds() {
  return (
    <section className={clsx(styles.section, styles.sectionMuted)}>
      <div className="container">
        <div className={styles.sectionHead}>
          <Heading as="h2" className={styles.sectionTitle}>
            Q&A and agentic evaluation
          </Heading>
          <p className={styles.sectionLead}>
            Both use the same <code>query</code>, <code>eval</code>, and{" "}
            <code>summarize</code> workflow.
          </p>
        </div>
        <div className={styles.kindGrid}>
          {KINDS.map((k) => (
            <Link key={k.title} to={k.to} className={styles.kindCard}>
              <span className={styles.kindTag}>{k.tag}</span>
              <Heading as="h3" className={styles.kindTitle}>
                {k.title}
              </Heading>
              <p className={styles.kindBody}>{k.body}</p>
            </Link>
          ))}
        </div>
      </div>
    </section>
  );
}

export function Phases() {
  return (
    <section className={styles.section}>
      <div className="container">
        <div className={styles.sectionHead}>
          <Heading as="h2" className={styles.sectionTitle}>
            A three-phase pipeline
          </Heading>
          <p className={styles.sectionLead}>
            Each phase is a dedicated subcommand, so you can re-run any step
            independently — or chain all three with <code>run-all</code>.
          </p>
        </div>
        <div className={styles.phaseGrid}>
          {PHASES.map((p) => (
            <Link key={p.title} to={p.to} className={styles.phaseCard}>
              <span className={styles.phaseStep}>{p.step}</span>
              <Heading as="h3" className={styles.phaseTitle}>
                {p.title}
              </Heading>
              <code className={styles.phaseCmd}>{p.cmd}</code>
              <p className={styles.phaseBody}>{p.body}</p>
            </Link>
          ))}
        </div>
      </div>
    </section>
  );
}

export const CAPABILITIES = [
  {
    title: "Evaluate multiple models on multiple benchmarks",
    body: (
      <>
        Both <code>--model-paths</code> and <code>--benchmarks</code> take
        lists, so one run covers every pair.
      </>
    ),
    to: "/docs/cli/query",
  },
  {
    title: "Query local or hosted models",
    body: "In-process vLLM for Q&A, plus OpenAI, OpenRouter, and AWS SageMaker endpoints. Agentic runs use an OpenAI-compatible tool-calling endpoint.",
    to: "/docs/configuration/model-configs",
  },
  {
    title: "Define a custom benchmark",
    body: (
      <>
        Define a <code>benchmark.yaml</code> and run it with{" "}
        <code>--custom-benchmarks</code>. Reuse built-in components or implement
        additional loaders, tools, and scoring functions as needed.
      </>
    ),
    to: "/docs/benchmarks/creating-benchmarks",
  },
  {
    title: "Record experiment settings",
    // Shots are deliberately not in this list: few-shot prompting is a Q&A idea,
    // and naming it here made the whole page read as Q&A-only now that agentic
    // benchmarks are on it. "Engine and generation parameters" is the code's own
    // phrase for the two blocks of a model spec (see ModelSpec), so the tail of
    // the list generalizes without inventing a category.
    body: "Completed query stages record resolved model and benchmark configurations and trial seeds. Retain data and deployment revisions separately when needed to reproduce an evaluation.",
    to: "/docs/analytics/storage",
  },
  {
    title: "Analyze at scale in BigQuery",
    body: "Ingest normalized metrics for SQL leaderboards and performance-over-time tracking.",
    to: "/docs/analytics/bigquery/quickstart",
  },
  {
    title: "Build dashboards",
    body: "Power Looker Studio leaderboards and comparison charts from pre-built views.",
    to: "/docs/analytics/bigquery/dashboards/setup",
  },
];

export function Capabilities() {
  return (
    <section className={styles.section}>
      <div className="container">
        <div className={styles.sectionHead}>
          <Heading as="h2" className={styles.sectionTitle}>
            What you can do with FAITH
          </Heading>
        </div>
        <div className={styles.capGrid}>
          {CAPABILITIES.map((c, i) => (
            <Link key={c.title} to={c.to} className={styles.capCard}>
              <span className={styles.capNum}>
                {String(i + 1).padStart(2, "0")}
              </span>
              <div>
                <Heading as="h3" className={styles.capTitle}>
                  {c.title}
                </Heading>
                <p className={styles.capBody}>{c.body}</p>
              </div>
            </Link>
          ))}
        </div>
      </div>
    </section>
  );
}

export const BACKENDS = [
  {
    name: "vLLM",
    kind: "Local",
    note: "Hugging Face models on supported hardware",
  },
  { name: "OpenAI", kind: "API", note: "GPT-family models" },
  { name: "OpenRouter", kind: "API", note: "Claude, Gemini, and more" },
  { name: "SageMaker", kind: "API", note: "Models on AWS endpoints" },
];

export function Backends() {
  return (
    <section className={clsx(styles.section, styles.sectionMuted)}>
      <div className="container">
        <div className={styles.sectionHead}>
          <Heading as="h2" className={styles.sectionTitle}>
            One interface, many model backends
          </Heading>
          <p className={styles.sectionLead}>
            Q&A supports the engines below. Agentic execution requires the
            openai engine with a compatible tool-calling endpoint.
          </p>
        </div>
        <div className={styles.backendGrid}>
          {BACKENDS.map((b) => (
            <div key={b.name} className={styles.backendCard}>
              <div className={styles.backendTop}>
                <span className={styles.backendName}>{b.name}</span>
                <span className={styles.backendKind}>{b.kind}</span>
              </div>
              <p className={styles.backendNote}>{b.note}</p>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}

export function FinalCTA() {
  return (
    <section className={styles.cta}>
      <div className="container">
        <Heading as="h2" className={styles.ctaTitle}>
          Run an initial evaluation
        </Heading>
        <p className={styles.ctaLead}>
          Install FAITH, configure a supported model and benchmark, and follow
          the quickstart to generate and evaluate a small set of records.
        </p>
        <div className={styles.buttons}>
          <Link
            className="button button--primary"
            to="/docs/getting-started/installation"
          >
            Install FAITH
          </Link>
          <Link
            className="button button--secondary"
            href="https://github.com/cisco-foundation-ai/faith"
          >
            View on GitHub
            <ExternalArrow />
          </Link>
        </div>
      </div>
    </section>
  );
}
