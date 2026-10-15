import { useState, useId } from "react";
import clsx from "clsx";
import { SEGMENTS, LEAVES } from "./data";
import styles from "./styles.module.css";

// ===========================================================================
// A breadcrumb chip trail on top cross-highlights the numbered spec rows
// below, so hovering either side lights up the matching path segment.
// ===========================================================================
export function CombinedLayout() {
  const layoutId = useId();
  const [activeId, setActive] = useState(null);
  return (
    <div className={styles.combo}>
      <div className={styles.comboTrail}>
        {SEGMENTS.map((s) => (
          <span key={s.id} className={styles.crumbItem}>
            <button
              type="button"
              aria-describedby={`${layoutId}-${s.id}-description`}
              className={clsx(
                styles.crumb,
                activeId === s.id && styles.crumbOn,
              )}
              onMouseEnter={() => setActive(s.id)}
              onMouseLeave={() => setActive(null)}
              onFocus={() => setActive(s.id)}
              onBlur={() => setActive(null)}
            >
              {s.label}
            </button>
            <span className={styles.crumbSep} aria-hidden="true">
              /
            </span>
          </span>
        ))}
        <span className={styles.crumbTail}>…</span>
      </div>

      <div className={styles.comboSpec}>
        {SEGMENTS.map((s, i) => (
          <div
            key={s.id}
            className={clsx(
              styles.specRow,
              activeId === s.id && styles.specRowOn,
            )}
            style={{ "--d": i }}
            onMouseEnter={() => setActive(s.id)}
            onMouseLeave={() => setActive(null)}
          >
            <span className={styles.specNum}>{i + 1}</span>
            <div className={styles.specMain}>
              <code className={styles.specSeg}>
                {s.label}
                <span className={styles.specSlash}>/</span>
              </code>
              <span
                className={styles.specText}
                id={`${layoutId}-${s.id}-description`}
              >
                {s.text}
              </span>
            </div>
          </div>
        ))}
        <div className={styles.specLeafHead}>Inside the innermost segment</div>
        {/* No file/folder glyph on these rows. The one this used to draw for a
            directory was `▸`, which reads as a collapsed disclosure triangle and
            so promised that `trials/` would open — it never did, and the subtree
            it pointed at is covered in prose below the diagram instead. The
            markers were redundant even when they were right: `trials/` ends in a
            slash and the others end in `.json`. */}
        {LEAVES.map((l) => (
          <div key={l.id} className={styles.specLeaf}>
            <code className={styles.specLeafName}>{l.label}</code>
            <span className={styles.specText}>{l.text}</span>
          </div>
        ))}
      </div>
    </div>
  );
}
