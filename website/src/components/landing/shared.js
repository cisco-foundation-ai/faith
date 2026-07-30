import { useEffect, useRef, useState } from "react";
import clsx from "clsx";
import Link from "@docusaurus/Link";
import Heading from "@theme/Heading";
import styles from "../../pages/index.module.css";

// ---------------------------------------------------------------------------
// Shared data
// ---------------------------------------------------------------------------
export const CMD_TEXT = `faith run-all \\
  --model-paths meta-llama/Llama-3.1-8B-Instruct \\
  --benchmarks ctibench-mcqa seceval \\
  --model-engine vllm`;

export const RUN_OUTPUT = [
  { tag: "query", text: "querying 2 benchmarks · 1 model · 3 trials" },
  { tag: "eval", text: "grading responses against ground truth" },
  { tag: "summarize", text: "aggregating metrics → leaderboard" },
];

export const LEADERBOARD = [
  {
    model: "Llama-3.1-70B-Instruct",
    bench: "ctibench-mcqa",
    shots: 5,
    score: 0.912,
  },
  { model: "GPT-4.1", bench: "ctibench-mcqa", shots: 0, score: 0.901 },
  { model: "Llama-3.1-70B-Instruct", bench: "seceval", shots: 5, score: 0.864 },
  {
    model: "Foundation-Sec-8B",
    bench: "ctibench-mcqa",
    shots: 5,
    score: 0.859,
  },
  { model: "GPT-4.1", bench: "seceval", shots: 0, score: 0.842 },
  {
    model: "Llama-3.1-8B-Instruct",
    bench: "ctibench-mcqa",
    shots: 5,
    score: 0.821,
  },
  { model: "Foundation-Sec-8B", bench: "seceval", shots: 5, score: 0.804 },
  { model: "Llama-3.1-8B-Instruct", bench: "seceval", shots: 0, score: 0.791 },
];

const LINE_MS = 520;
const ROW_MS = 190;

// ---------------------------------------------------------------------------
// Animation hook — drives typing → output → flow → leaderboard, loops in view.
// ---------------------------------------------------------------------------
export function useRunAnimation() {
  // The command is always present; it fades in as a whole (not typed), then the
  // output streams and the leaderboard fills in.
  const [cmdIn, setCmdIn] = useState(false);
  const [lines, setLines] = useState(0);
  const [rows, setRows] = useState(0);
  const containerRef = useRef(null);
  const startedRef = useRef(false);

  useEffect(() => {
    const finished = () => {
      setCmdIn(true);
      setLines(RUN_OUTPUT.length);
      setRows(LEADERBOARD.length);
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
      // Beat between the last output line and the leaderboard filling in.
      t += 450;
      for (let r = 1; r <= LEADERBOARD.length; r += 1) {
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
          {lines > 0 && "\n\n"}
          {RUN_OUTPUT.slice(0, lines).map((l) => (
            <span key={l.tag} className={styles.outLine}>
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

export function LeaderboardPanel({ rows }) {
  const boardActive = rows > 0;
  return (
    <div className={clsx(styles.board, boardActive && styles.boardOn)}>
      <div className={styles.boardHead}>
        <span className={styles.boardTitle}>Leaderboard</span>
        <span className={styles.boardMeta}>
          accuracy.mean · 4 models · 2 benchmarks
        </span>
      </div>
      <table className={styles.boardTable}>
        <thead>
          <tr>
            <th className={styles.boardRank}>#</th>
            <th>Model</th>
            <th>Benchmark</th>
            <th className={styles.boardShots}>Shots</th>
            <th className={styles.boardScoreCol}>Score</th>
          </tr>
        </thead>
        <tbody>
          {LEADERBOARD.map((r, i) => {
            const shown = i < rows;
            return (
              <tr
                key={`${r.model}-${r.bench}`}
                className={clsx(styles.boardRow, shown && styles.boardRowIn)}
              >
                <td className={styles.boardRank}>{i + 1}</td>
                <td className={styles.boardModel}>{r.model}</td>
                <td>
                  <span className={styles.boardBench}>{r.bench}</span>
                </td>
                <td className={styles.boardShots}>{r.shots}-shot</td>
                <td className={styles.boardScoreCol}>
                  <div className={styles.boardScore}>
                    <span className={styles.boardBarTrack}>
                      <span
                        className={styles.boardBarFill}
                        style={{ width: shown ? `${r.score * 100}%` : "0%" }}
                      />
                    </span>
                    <span className={styles.boardScoreVal}>
                      {shown ? r.score.toFixed(3) : "—"}
                    </span>
                  </div>
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

// Hero content pieces (so variants can wrap them differently)
export function HeroCopy({ align = "center" }) {
  return (
    <>
      <span className={styles.heroEyebrow}>FAITH · Cisco Foundation AI</span>
      <Heading as="h1" className={styles.heroTitle}>
        The open harness for benchmarking cybersecurity models.
      </Heading>
      <p className={styles.heroTagline}>
        FAITH — Foundation-AI&apos;s Test Hub/Harness — gives you one uniform
        way to run open benchmarks that measure the cybersecurity knowledge and
        capabilities of language models.
      </p>
      <div
        className={clsx(styles.buttons, align === "left" && styles.buttonsLeft)}
      >
        <Link
          className="button button--primary"
          to="/docs/getting-started/quickstart"
        >
          Quickstart
        </Link>
        <Link className="button button--secondary" to="/docs/intro">
          How it works
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
    body: "Run models over benchmarks and record every prompt and response into a datastore — local or on GCS. Works with vLLM, OpenAI, OpenRouter, and SageMaker.",
    to: "/docs/cli/query",
  },
  {
    step: "02",
    title: "Eval",
    cmd: "faith eval",
    body: "Analyze the collected logs and compute metrics against ground truth, writing a metrics.json alongside each run.",
    to: "/docs/cli/eval",
  },
  {
    step: "03",
    title: "Summarize",
    cmd: "faith summarize",
    body: "Collate metrics across runs into a table or CSV — or ingest straight into BigQuery for leaderboards and dashboards.",
    to: "/docs/cli/summarize",
  },
];

export function Phases({ muted = true }) {
  return (
    <section className={clsx(styles.section, muted && styles.sectionMuted)}>
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
    title: "Run open cybersecurity benchmarks",
    body: "CTIBench, CyberMetric, MMLU, SecBench, and SecEval — through one interface.",
    to: "/docs/benchmarks/catalog",
  },
  {
    title: "Query local or hosted models",
    body: "vLLM for HuggingFace models, plus OpenAI, OpenRouter, and AWS SageMaker APIs.",
    to: "/docs/cli/model-configs",
  },
  {
    title: "Bring your own benchmark",
    body: "Define a benchmark.yaml and run it with --custom-benchmarks. No code changes.",
    to: "/docs/benchmarks/creating-benchmarks",
  },
  {
    title: "Keep runs reproducible",
    body: "Every run records model, prompt format, generation mode, shots, seed, and params.",
    to: "/docs/architecture/storage",
  },
  {
    title: "Analyze at scale in BigQuery",
    body: "Ingest normalized metrics for SQL leaderboards and performance-over-time tracking.",
    to: "/docs/bigquery/quickstart",
  },
  {
    title: "Build dashboards",
    body: "Power Looker Studio leaderboards and comparison charts from pre-built views.",
    to: "/docs/bigquery/dashboards/setup",
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
  { name: "vLLM", kind: "Local", note: "HuggingFace models on GPU" },
  { name: "OpenAI", kind: "API", note: "GPT-family models" },
  { name: "OpenRouter", kind: "API", note: "Claude, Gemini, and more" },
  { name: "SageMaker", kind: "API", note: "Models on AWS endpoints" },
];

export function Backends({ muted = true }) {
  return (
    <section className={clsx(styles.section, muted && styles.sectionMuted)}>
      <div className="container">
        <div className={styles.sectionHead}>
          <Heading as="h2" className={styles.sectionTitle}>
            One interface, many model backends
          </Heading>
          <p className={styles.sectionLead}>
            Swap engines with a single flag — the rest of the pipeline stays the
            same.
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
          Run your first benchmark in minutes
        </Heading>
        <p className={styles.ctaLead}>
          Install FAITH, point it at a model and a benchmark, and get scored
          results.
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
            View on GitHub ↗
          </Link>
        </div>
      </div>
    </section>
  );
}
