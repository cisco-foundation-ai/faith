import React from "react";
import Link from "@docusaurus/Link";
import clsx from "clsx";
import styles from "./styles.module.css";

/**
 * A row of pill-shaped links to a benchmark's canonical sources.
 *
 * Usage: <SourceLinks links={[["paper", "https://..."], ["dataset", "https://..."]]} />
 *
 * Pass `heading` when the row sits inside a `###` heading, so the pills keep
 * their own small type rather than inheriting the heading's size and weight.
 */
export default function SourceLinks({ links, heading, right, onHeadingLine }) {
  return (
    <span
      className={clsx(
        styles.row,
        heading && styles.rowHeading,
        right && styles.rowRight,
        onHeadingLine && styles.rowOnHeadingLine,
      )}
    >
      {links.map(([label, href]) => (
        <Link key={href} className={styles.pill} to={href}>
          {label}
        </Link>
      ))}
    </span>
  );
}
