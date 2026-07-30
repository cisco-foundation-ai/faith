import { Fragment } from "react";
import clsx from "clsx";
import styles from "./styles.module.css";

// ---------------------------------------------------------------------------
// A styled directory tree — replaces ASCII `├──` code blocks with something
// that reads as UI rather than a fenced snippet. Feed it a nested array:
//   {name, kind: 'dir' | 'file', note?, children?}
// ---------------------------------------------------------------------------
function Node({ node, depth, isLast, ancestorsLast }) {
  const isDir = node.kind === "dir";
  const hasChildren = node.children && node.children.length > 0;

  return (
    <>
      <div
        className={clsx(styles.row, isDir ? styles.rowDir : styles.rowFile)}
        style={{ "--depth": depth }}
      >
        <span className={styles.rails} aria-hidden="true">
          {ancestorsLast.map((al, i) => (
            <span
              key={i}
              className={clsx(styles.rail, al && styles.railBlank)}
            />
          ))}
          {depth > 0 && (
            <span className={clsx(styles.rail, styles.railTee)}>
              {isLast ? "└" : "├"}
            </span>
          )}
        </span>
        <span className={styles.icon} aria-hidden="true">
          {isDir ? "▸" : "·"}
        </span>
        <span className={styles.name}>
          {node.name}
          {isDir && <span className={styles.slash}>/</span>}
        </span>
        {node.note && <span className={styles.note}>{node.note}</span>}
      </div>
      {hasChildren &&
        node.children.map((child, i) => (
          <Node
            key={child.name}
            node={child}
            depth={depth + 1}
            isLast={i === node.children.length - 1}
            ancestorsLast={[...ancestorsLast, isLast]}
          />
        ))}
    </>
  );
}

export default function FileTree({ root, tree, caption }) {
  return (
    <div className={styles.wrap}>
      {root && <div className={styles.rootLabel}>{root}</div>}
      <div className={styles.tree}>
        {tree.map((node, i) => (
          <Node
            key={node.name}
            node={node}
            depth={0}
            isLast={i === tree.length - 1}
            ancestorsLast={[]}
          />
        ))}
      </div>
      {caption && <div className={styles.caption}>{caption}</div>}
    </div>
  );
}
