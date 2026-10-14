import { useState, useId, Fragment } from "react";
import clsx from "clsx";
import Link from "@docusaurus/Link";
import styles from "./styles.module.css";

// ---------------------------------------------------------------------------
// Tiny inline formatter so row data can stay plain strings while still
// supporting `inline code` and [links](/url) — the only rich bits the flag
// descriptions and notes ever need.
// ---------------------------------------------------------------------------
function renderRich(text, keyPrefix = "r") {
  if (text == null) return null;
  if (typeof text !== "string") return text; // already a React node

  const nodes = [];
  // Split on `code` spans and [label](href) links, keeping the delimiters.
  const pattern = /(`[^`]+`|\[[^\]]+\]\([^)]+\))/g;
  const parts = text.split(pattern);
  parts.forEach((part, i) => {
    if (!part) return;
    const key = `${keyPrefix}-${i}`;
    if (part.startsWith("`") && part.endsWith("`")) {
      nodes.push(<code key={key}>{part.slice(1, -1)}</code>);
      return;
    }
    const link = part.match(/^\[([^\]]+)\]\(([^)]+)\)$/);
    if (link) {
      nodes.push(
        <Link key={key} to={link[2]}>
          {link[1]}
        </Link>,
      );
      return;
    }
    nodes.push(<Fragment key={key}>{part}</Fragment>);
  });
  return nodes;
}

// A labelled caveat hanging under the description. Two kinds use it: `agentic:`,
// for the flags whose meaning changes when a benchmark is agentic, and
// `required:`. Both sit in the Description cell rather than in columns of their
// own — only about half the flags have anything to say under either heading, so
// a column would be mostly empty while squeezing the descriptions.
//
// The label alone is a complete statement for an unconditional `required: true`;
// with body text it reads "required <condition>". That is why this renders the
// label and the body as separate nodes instead of one string.
function Note({ label, variant, children }) {
  return (
    <div className={clsx(styles.note, variant && styles[variant])}>
      <span className={styles.noteLabel}>{label}</span>
      {children && <span>{children}</span>}
    </div>
  );
}

// The trailing period is added here rather than in the row data because the data
// holds a fragment ("unless `--model-configs` is set") that has to complete the
// word "required" to make a sentence. `required: true` has no fragment and the
// label alone is the whole statement, so it gets no period either way.
function RequiredNote({ required, index }) {
  if (!required) return null;
  const note = typeof required === "object" ? required.note : null;
  return (
    <Note label="required" variant="noteQuiet">
      {note ? <>{renderRich(note, `req-${index}`)}.</> : null}
    </Note>
  );
}

function AgenticNote({ note, index }) {
  if (!note) return null;
  return <Note label="agentic">{renderRich(note, `ag-${index}`)}</Note>;
}

export default function FlagTable({ rows }) {
  const tableId = useId();
  const [open, setOpen] = useState(() => new Set());

  const toggle = (i) =>
    setOpen((prev) => {
      const next = new Set(prev);
      if (next.has(i)) next.delete(i);
      else next.add(i);
      return next;
    });

  return (
    <div className={styles.wrap}>
      <table className={styles.table}>
        <thead>
          <tr>
            <th scope="col" className={styles.flagCol}>
              Flag
            </th>
            <th scope="col" className={styles.defaultCol}>
              Default
            </th>
            <th scope="col">Description</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((row, i) => {
            const anns = row.annotations || [];
            const expandable = anns.length > 0;
            const isOpen = open.has(i);
            const annotationId = `${tableId}-annotations-${i}`;
            // The type goes next to the flag name, as the metavar in a usage
            // string: `--n-shot list[int|N/D]`. That is the form the reader has
            // already seen, because these pages are generated from argparse and
            // `--help` prints the same pairing.
            //
            // A boolean gets nothing. `--experimental-parallelize-models flag`
            // says less than the flag name alone, and it is also the widest cell
            // in the table, so suppressing it is free.
            const head = (
              <span className={styles.flagHead}>
                {expandable && (
                  <span
                    className={clsx(
                      styles.chevron,
                      isOpen && styles.chevronOpen,
                    )}
                    aria-hidden="true"
                  >
                    ›
                  </span>
                )}
                <code className={styles.flagName}>{row.flag}</code>
                {row.type !== "flag" && (
                  <code className={styles.metaVal}>{row.type}</code>
                )}
              </span>
            );
            return (
              <Fragment key={row.flag}>
                <tr className={clsx(expandable && styles.rowExpandable)}>
                  <td className={styles.flagCol}>
                    {expandable ? (
                      <button
                        type="button"
                        className={styles.flagToggle}
                        aria-expanded={isOpen}
                        aria-controls={annotationId}
                        onClick={() => toggle(i)}
                      >
                        {head}
                        <span className={styles.annCount}>
                          {anns.length} annotation{anns.length > 1 ? "s" : ""}
                        </span>
                      </button>
                    ) : (
                      head
                    )}
                  </td>
                  <td className={styles.defaultCol}>
                    {row.default ? (
                      <code className={styles.defaultVal}>{row.default}</code>
                    ) : (
                      <span className={styles.dash}>—</span>
                    )}
                  </td>
                  <td>
                    {renderRich(row.desc, `desc-${i}`)}
                    <RequiredNote required={row.required} index={i} />
                    <AgenticNote note={row.agentic} index={i} />
                  </td>
                </tr>
                {expandable && (
                  <tr
                    id={annotationId}
                    hidden={!isOpen}
                    className={clsx(styles.annRow, isOpen && styles.annRowOpen)}
                  >
                    <td colSpan={3} className={styles.annCell}>
                      <div className={styles.annInner}>
                        <dl className={styles.annList}>
                          {anns.map((a) => (
                            <div className={styles.annItem} key={a.key}>
                              <dt>
                                <code className={styles.annKey}>{a.key}</code>
                              </dt>
                              <dd>{renderRich(a.desc, `ann-${i}-${a.key}`)}</dd>
                            </div>
                          ))}
                        </dl>
                      </div>
                    </td>
                  </tr>
                )}
              </Fragment>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}
