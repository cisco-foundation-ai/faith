import React, { useState } from "react";
import clsx from "clsx";
import { TRUNK, QA_LANE, AGENTIC_LANE, LOOP_BEATS, EXITS } from "./conceptData";
import styles from "./ForkMerge.module.css";

// ---------------------------------------------------------------------------
// Concept A — fork and merge.
//
// One trunk splits into two lanes and rejoins. Both the sharing and the
// divergence are structural, so neither needs a sentence: the reader can see
// that `query` forks, that the lanes are different lengths and different shapes,
// and that `summarize` is reached from both.
//
// Each lane is banded by subcommand, because `query` and `eval` diverge
// separately — eval extracts and scores differently on the two paths, so it is
// part of the fork rather than the shared summary stage. The diagram omits
// shared experiment storage and scoring utilities to emphasize the differences.
//
// Everything is HTML and CSS, including the fork and merge connectors (see
// Joint). Hover or focus a lane to dim the other.
// ---------------------------------------------------------------------------

// No gloss beside the chips. A run of four engine names is self-evidently four
// options, so a "1 of 4" note beside them is counting for the reader.
function Chips({ chips }) {
  if (!chips) return null;
  return (
    <div className={styles.chips}>
      {chips.map((c) => (
        <code key={c} className={styles.chip}>
          {c}
        </code>
      ))}
    </div>
  );
}

function LoopCycle() {
  return (
    <div className={styles.cycle}>
      <svg
        className={styles.cycleRing}
        viewBox="0 0 100 100"
        aria-hidden="true"
      >
        {/* The ring is the shape that carries the meaning, so it is drawn rather
            than implied by a border-radius: the dash offset animates around it
            to show the loop running. */}
        <circle className={styles.cycleTrack} cx="50" cy="50" r="34" />
        <circle className={styles.cycleRun} cx="50" cy="50" r="34" />
      </svg>
      {/* Two words inside the ring and the turn cap under it — nothing else. The
          beats' glosses ("one turn of the model", "bash · read · write") were the
          least-read text here, and the ring says what it needs to without them. */}
      <div className={styles.cycleBeats}>
        {LOOP_BEATS.map((b) => (
          <span key={b.key} className={styles.beatLabel}>
            {b.label}
          </span>
        ))}
      </div>
      <span className={styles.cycleCap}>configured limits</span>
    </div>
  );
}

function Exits() {
  return (
    <div className={styles.exits}>
      <span className={styles.exitsTitle}>Task termination reasons</span>
      <div className={styles.exitList}>
        {EXITS.map((e) => (
          <span
            key={e.key}
            className={clsx(styles.exit, e.answer && styles.exitAnswer)}
          >
            <code className={styles.exitLabel}>{e.label}</code>
            <span className={styles.exitNote}>{e.note}</span>
          </span>
        ))}
      </div>
    </div>
  );
}

// The fork (or merge) drawn as three CSS rules: a stub on the trunk's centre
// line, a horizontal beam spanning the two lane centres, and a drop into each
// lane. `kind` only flips which end the stub and the drops attach to.
//
// Only the fork carries drops. The lanes are different heights — that asymmetry
// is the point — so a fixed-length drop up from the merge beam would leave the
// short lane hanging in mid-air. The merge is reached by each column's own rail
// instead (see Rail), which stretches however far that column has to fall.
function Joint({ kind, hot }) {
  return (
    <div className={clsx(styles.joint, styles[`joint_${kind}`])}>
      <span className={styles.jointStem} aria-hidden="true" />
      <span className={styles.jointBeam} aria-hidden="true">
        {kind === "fork" && (
          <>
            <span
              className={clsx(
                styles.jointDrop,
                styles.jointDropQa,
                hot === "agentic" && styles.jointDim,
              )}
            />
            <span
              className={clsx(
                styles.jointDrop,
                styles.jointDropAg,
                hot === "qa" && styles.jointDim,
              )}
            />
          </>
        )}
      </span>
    </div>
  );
}

// The drop from a lane down to the merge beam, as tall as that column's slack.
// Rail length balances the layout; it does not represent elapsed execution time.
function Rail({ dim }) {
  return (
    <div className={clsx(styles.rail, dim && styles.railDim)}>
      <span className={styles.railLine} aria-hidden="true" />
    </div>
  );
}

// No step number and no rail. The cards are already ordered top to bottom inside a
// bounded band, so numbering them was labelling a sequence the layout states — and
// the connector line between them lived in the numbers' gutter, so it goes too.
function Step({ step }) {
  return (
    <li className={styles.step}>
      <div
        className={clsx(
          styles.card,
          step.accent && styles.cardAccent,
          step.loop && styles.cardLoop,
        )}
        tabIndex={0}
      >
        <span className={styles.cardTitle}>{step.title}</span>
        <code className={styles.cardRef}>{step.ref}</code>
        <p className={styles.cardBody}>{step.body}</p>
        <Chips chips={step.chips} />
        {step.loop && <LoopCycle />}
      </div>
    </li>
  );
}

function Lane({ lane, dim, onEnter, onLeave }) {
  return (
    <div
      className={clsx(
        styles.lane,
        styles[`lane_${lane.id}`],
        dim && styles.dim,
      )}
      onMouseEnter={onEnter}
      onMouseLeave={onLeave}
      onFocus={onEnter}
      onBlur={onLeave}
    >
      <div className={styles.laneHead}>
        <span className={styles.laneName}>{lane.label}</span>
        <span className={styles.laneShape}>{lane.shape}</span>
        <code className={styles.laneRef}>{lane.ref}</code>
      </div>

      {/* One block per subcommand. The handoff artifact is deliberately outside
          the block: `query` and `eval` are separate invocations with a file
          written between them, so the block is what one command does and the
          chip below it is what that command leaves behind. */}
      {lane.phases.map((phase) => (
        <React.Fragment key={phase.id}>
          <div className={styles.band}>
            <div className={styles.bandHead}>
              <code className={styles.bandName}>{phase.label}</code>
              <span className={styles.bandRule} aria-hidden="true" />
            </div>

            <ol className={styles.steps}>
              {phase.steps.map((s) => (
                <Step key={s.key} step={s} />
              ))}
            </ol>

            {/* The exits belong to phase 1: they are how the loop ends, and they
                decide whether eval has an answer to extract at all. */}
            {lane.id === "agentic" && phase.id === "query" && <Exits />}
          </div>

          <div className={styles.hands}>
            <span className={styles.handsArrow} aria-hidden="true" />
            <code className={styles.handsLabel}>{phase.hands}</code>
          </div>
        </React.Fragment>
      ))}
    </div>
  );
}

export default function ForkMerge() {
  const [hot, setHot] = useState(null);

  return (
    <div className={styles.wrap}>
      {/* No legend. Each lane is titled in its own colour directly above its own
          cards, so a key mapping colour to kind explains a code the reader has
          already broken. */}
      <div className={styles.canvas}>
        {/* No tag over either trunk node. "One entry, then it forks" and "they
            rejoin" narrate what the connectors already draw; captioning a shape
            with the shape's own name is hedging that the drawing won't land. */}
        <div className={styles.trunk}>
          <div className={clsx(styles.trunkNode, styles.trunkHead)}>
            <div className={styles.trunkInner}>
              <code className={styles.trunkTitle}>{TRUNK.head.title}</code>
              <code className={styles.trunkRef}>{TRUNK.head.ref}</code>
              <p className={styles.trunkBody}>{TRUNK.head.body}</p>
            </div>
          </div>
        </div>

        {/* The fork and the merge are drawn in CSS off the same column geometry
            the lanes use, so they line up exactly at every width and are correct
            in the server-rendered HTML. Measuring the laid-out boxes and drawing
            an SVG overlay was the earlier approach; it collapsed to a single
            vertical line whenever the measurement ran before layout settled. */}
        <Joint kind="fork" hot={hot} />

        {/* Each lane sits in a full-height column with its rail beneath it, so the
            lane keeps its own height while the rail absorbs the difference. That
            is what makes the short path's early finish visible as a long drop
            rather than as an unexplained gap. */}
        <div className={styles.lanes}>
          <div className={clsx(styles.col, styles.col_qa)}>
            <Lane
              lane={QA_LANE}
              dim={hot === "agentic"}
              onEnter={() => setHot("qa")}
              onLeave={() => setHot(null)}
            />
            <Rail dim={hot === "agentic"} />
          </div>
          <div className={clsx(styles.col, styles.col_agentic)}>
            <Lane
              lane={AGENTIC_LANE}
              dim={hot === "qa"}
              onEnter={() => setHot("agentic")}
              onLeave={() => setHot(null)}
            />
            <Rail dim={hot === "qa"} />
          </div>
        </div>

        <Joint kind="merge" hot={hot} />

        <div className={styles.trunk}>
          <div className={clsx(styles.trunkNode, styles.trunkTail)}>
            <div className={styles.trunkInner}>
              <code className={styles.trunkTitle}>{TRUNK.tail.title}</code>
              <code className={styles.trunkRef}>{TRUNK.tail.ref}</code>
              <p className={styles.trunkBody}>{TRUNK.tail.body}</p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
