import { useState, Fragment } from "react";
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

function Required({ required }) {
  if (!required) return <span className={styles.dash}>—</span>;
  const note = typeof required === "object" ? required.note : null;
  return (
    <>
      <span className="reqPill">Required</span>
      {note && <span className="reqNote">{renderRich(note, "note")}</span>}
    </>
  );
}

export default function FlagTable({ rows }) {
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
            <th className={styles.flagCol}>Flag</th>
            <th className={styles.typeCol}>Type</th>
            <th className={styles.defaultCol}>Default</th>
            <th className={styles.reqCol}>Required</th>
            <th>Description</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((row, i) => {
            const anns = row.annotations || [];
            const expandable = anns.length > 0;
            const isOpen = open.has(i);
            return (
              <Fragment key={row.flag}>
                <tr className={clsx(expandable && styles.rowExpandable)}>
                  <td className={styles.flagCol}>
                    {expandable ? (
                      <button
                        type="button"
                        className={styles.flagToggle}
                        aria-expanded={isOpen}
                        onClick={() => toggle(i)}
                      >
                        <span
                          className={clsx(
                            styles.chevron,
                            isOpen && styles.chevronOpen,
                          )}
                          aria-hidden="true"
                        >
                          ›
                        </span>
                        <code className={styles.flagName}>{row.flag}</code>
                        <span className={styles.annCount}>
                          {anns.length} annotation{anns.length > 1 ? "s" : ""}
                        </span>
                      </button>
                    ) : (
                      <code className={styles.flagName}>{row.flag}</code>
                    )}
                  </td>
                  <td className={styles.typeCol}>
                    <code className={styles.typeVal}>{row.type}</code>
                  </td>
                  <td className={styles.defaultCol}>
                    {row.default ? (
                      <code className={styles.defaultVal}>{row.default}</code>
                    ) : (
                      <span className={styles.dash}>—</span>
                    )}
                  </td>
                  <td className={styles.reqCol}>
                    <Required required={row.required} />
                  </td>
                  <td>{renderRich(row.desc, `desc-${i}`)}</td>
                </tr>
                {expandable && (
                  <tr
                    className={clsx(styles.annRow, isOpen && styles.annRowOpen)}
                  >
                    <td colSpan={5} className={styles.annCell}>
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
