import clsx from "clsx";
import useDocusaurusContext from "@docusaurus/useDocusaurusContext";
import Layout from "@theme/Layout";
import styles from "./index.module.css";
import {
  useRunAnimation,
  TerminalPanel,
  ResultsPanel,
  HeroCopy,
  Kinds,
  Phases,
  Capabilities,
  Backends,
  FinalCTA,
} from "../components/landing/shared";

// Split hero: headline/buttons on the left, the animated terminal → results
// stacked on the right, above the fold.
function SplitHero() {
  const { cmdIn, lines, rows, containerRef } = useRunAnimation();
  return (
    <header className={clsx(styles.hero, styles.heroSplit)}>
      <div className={styles.heroGlow} aria-hidden="true" />
      <div className={clsx("container", styles.heroSplitGrid)}>
        <div className={styles.heroSplitCopy}>
          <HeroCopy />
        </div>
        <div className={styles.heroSplitDemo} ref={containerRef}>
          <TerminalPanel cmdIn={cmdIn} lines={lines} />
          <ResultsPanel rows={rows} />
        </div>
      </div>
    </header>
  );
}

export default function Home() {
  const { siteConfig } = useDocusaurusContext();
  return (
    <Layout
      title={siteConfig.title}
      description="FAITH — Foundation-AI's Test Hub/Harness for benchmarking language models in cybersecurity."
    >
      <SplitHero />
      <main>
        <Phases />
        <Kinds />
        <Capabilities />
        <Backends />
        <FinalCTA />
      </main>
    </Layout>
  );
}
