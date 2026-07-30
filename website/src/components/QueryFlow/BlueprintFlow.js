import React, { useState } from "react";
import clsx from "clsx";
import Link from "@docusaurus/Link";
import { PHASES_EXPANDED } from "./flowData";
import styles from "./BlueprintFlow.module.css";

function decisionText(branch) {
  if (branch.kind === "select-one") return `${branch.on} — pick one`;
  if (branch.kind === "from-either")
    return `${branch.on} — built-in and/or custom`;
  return branch.on; // optional-sink
}

function branchSummary(branch) {
  const n = branch.options.length;
  if (branch.kind === "select-one") return `${n} engines`;
  if (branch.kind === "from-either") return `${n} sources`;
  return "optional";
}

// ---------------------------------------------------------------------------
// A flowchart of the pipeline's actual control flow, not a list of steps:
//   • query fans out at the model factory (pick 1 of 4 engines) and pulls
//     benchmarks from 1 of 2 sources
//   • phases are chained by the data artifact that flows between them
//   • summarize has an optional terminal branch to BigQuery
// Driven by PHASES_EXPANDED, so eval and summarize show their real sub-steps
// rather than a single node each.
// ---------------------------------------------------------------------------

const PHASE_META = {
  query: { cls: "query", tag: "Phase 1" },
  eval: { cls: "eval", tag: "Phase 2" },
  summarize: { cls: "summarize", tag: "Phase 3" },
};

// The artifact handed from one phase to the next.
const HANDOFF = {
  query: { label: "logs/", hint: "raw query + response records on disk" },
  eval: { label: "metrics.json", hint: "scores computed against ground truth" },
};

export function BlueprintFlowExpanded() {
  const [open, setOpen] = useState(() => new Set());
  const [active, setActive] = useState(null);

  const toggle = (id) =>
    setOpen((prev) => {
      const next = new Set(prev);
      next.has(id) ? next.delete(id) : next.add(id);
      return next;
    });

  return (
    <div className={styles.blueprint}>
      {PHASES_EXPANDED.map((phase) => {
        const m = PHASE_META[phase.id];
        return (
          <React.Fragment key={phase.id}>
            <section className={styles[`bp_${m.cls}`]}>
              <header className={styles.bpBand}>
                <span className={styles.bpTag}>{m.tag}</span>
                <code className={styles.bpCmd}>{phase.label}</code>
                <span className={styles.bpBlurb}>{phase.blurb}</span>
                {phase.steps.some((s) => s.branch) && (
                  <span className={styles.bpHint}>
                    click a decision node to expand
                  </span>
                )}
              </header>

              <div className={styles.bpTrack}>
                {phase.steps.map((step, i) => {
                  const id = `${phase.id}/${step.title}`;
                  const isDecision = Boolean(step.branch);
                  const isOpen = open.has(id);
                  const isActive = active === id;
                  const NodeTag = isDecision ? "button" : "div";
                  return (
                    <React.Fragment key={step.title}>
                      {i > 0 && (
                        <span className={styles.bpArrow} aria-hidden="true">
                          →
                        </span>
                      )}
                      <div className={styles.bpCol}>
                        <NodeTag
                          type={isDecision ? "button" : undefined}
                          className={clsx(
                            styles.bpNode,
                            isDecision && styles.bpNodeBtn,
                            isActive && styles.bpNodeActive,
                            isOpen && styles.bpNodeOpen,
                            step.optional && styles.bpNodeOptional,
                          )}
                          data-decision={isDecision ? "true" : undefined}
                          aria-expanded={isDecision ? isOpen : undefined}
                          onClick={isDecision ? () => toggle(id) : undefined}
                          onMouseEnter={() => setActive(id)}
                          onMouseLeave={() =>
                            setActive((cur) => (cur === id ? null : cur))
                          }
                          onFocus={() => setActive(id)}
                          onBlur={() =>
                            setActive((cur) => (cur === id ? null : cur))
                          }
                        >
                          <div className={styles.bpNodeHead}>
                            <span className={styles.bpRole}>{step.role}</span>
                            {isDecision && (
                              <span className={styles.bpBadge}>
                                {branchSummary(step.branch)}
                                <span
                                  className={clsx(
                                    styles.bpChevron,
                                    isOpen && styles.bpChevronOpen,
                                  )}
                                  aria-hidden="true"
                                >
                                  ▾
                                </span>
                              </span>
                            )}
                            {step.optional && !isDecision && (
                              <span className={styles.bpOptTag}>optional</span>
                            )}
                          </div>
                          <span className={styles.bpNodeTitle}>
                            {step.title}
                          </span>
                          <code className={styles.bpNodeRef}>{step.ref}</code>
                          <p className={styles.bpNodeBody}>{step.body}</p>
                          {step.detail && (
                            <ul className={styles.bpDetail}>
                              {step.detail.map((d) => (
                                <li key={d}>{d}</li>
                              ))}
                            </ul>
                          )}
                          {step.to && (
                            <Link
                              to={step.to}
                              className={styles.bpNodeLink}
                              onClick={(e) => e.stopPropagation()}
                            >
                              View storage layout →
                            </Link>
                          )}
                          {step.in && step.out && (
                            <div className={styles.bpIo}>
                              <code className={styles.bpIoIn}>{step.in}</code>
                              <span
                                className={styles.bpIoArrow}
                                aria-hidden="true"
                              >
                                →
                              </span>
                              <code className={styles.bpIoOut}>{step.out}</code>
                            </div>
                          )}
                        </NodeTag>

                        {isDecision && isOpen && (
                          <div className={styles.bpFanWrap} data-open="true">
                            <span
                              className={styles.bpDrop}
                              aria-hidden="true"
                            />
                            <span className={styles.bpDecision}>
                              {decisionText(step.branch)}
                            </span>
                            <div className={styles.bpOptions}>
                              {step.branch.options.map((o) => (
                                <span key={o.label} className={styles.bpOption}>
                                  <span className={styles.bpOptLabel}>
                                    {o.label}
                                  </span>
                                  <code className={styles.bpOptNote}>
                                    {o.note}
                                  </code>
                                </span>
                              ))}
                            </div>
                          </div>
                        )}
                      </div>
                    </React.Fragment>
                  );
                })}
              </div>
            </section>

            {HANDOFF[phase.id] && (
              <div className={styles.bpHandoff}>
                <span className={styles.bpHandoffArrow} aria-hidden="true">
                  ↓
                </span>
                <code className={styles.bpHandoffLabel}>
                  {HANDOFF[phase.id].label}
                </code>
                <span className={styles.bpHandoffHint}>
                  {HANDOFF[phase.id].hint}
                </span>
              </div>
            )}
          </React.Fragment>
        );
      })}
    </div>
  );
}
