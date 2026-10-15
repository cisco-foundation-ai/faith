#!/usr/bin/env node
// Verifies the site's claims about the faith repo that can be checked without
// running Python: every source path the docs point at must actually exist.
//
// This is the cheap half of drift detection. Facts that need Python to resolve
// (argparse flags, benchmark.yaml `!from` tags, the BigQuery schema) live in
// check-repo-facts.py, which imports faith itself.
//
//   node scripts/check-repo-facts.js [--repo <other-faith-checkout>]
//
// The site lives at website/ inside the faith repo, so it checks itself against
// the enclosing checkout by default. Exits 1 and lists every stale fact.

import { readFileSync, existsSync, statSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

// Resolve paths from this file, not the cwd, so the script works from anywhere.
const SITE = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const args = process.argv.slice(2);
const repoIdx = args.indexOf("--repo");
const repo = repoIdx === -1 ? resolve(SITE, "..") : resolve(args[repoIdx + 1]);
const pkg = join(repo, "src/faith");

if (!existsSync(pkg)) {
  console.error(
    `No faith package at ${pkg}.\n` +
      `Expected the site to live at website/ inside the faith repo, ` +
      `or pass --repo <path-to-faith>.`,
  );
  process.exit(2);
}

const failures = [];
const claim = (ok, where, what) => {
  if (!ok) failures.push(`${where}: ${what}`);
};

// ---------------------------------------------------------------------------
// 1. Flow diagram `ref:` values — the source file each step lives in.
//
// A ref may carry a `:line` suffix, which is a pointer for the reader and not
// part of the path, so it is stripped before the existence check. Both quote
// styles are accepted: matching only one silently checks nothing the day the
// file is reformatted, which is exactly what happened to the predecessor of
// this check.
// ---------------------------------------------------------------------------
const conceptData = readFileSync(
  join(SITE, "src/components/FlowConcepts/conceptData.js"),
  "utf8",
);
const refs = new Set(
  [...conceptData.matchAll(/ref: ['"]([^'"]+)['"]/g)].map((m) =>
    m[1].replace(/:\d+$/, ""),
  ),
);
// A scraped check that finds nothing passes for the wrong reason, so the empty
// case is a failure rather than a no-op. Not a minimum count: how many steps the
// diagram has is a design choice, and pinning a number here would mean editing
// this file every time someone adds or removes a card. Zero is the only count
// that says the scrape itself is broken.
if (refs.size === 0) {
  failures.push(
    `FlowConcepts/conceptData.js: no 'ref:' values matched — did its shape change?`,
  );
}
// Count distinct refs, not raw matches: the tally should say how many facts were
// actually checked, and several steps point at the same file.
let checked = refs.size;
for (const ref of refs) {
  claim(
    existsSync(join(pkg, ref)),
    "FlowConcepts/conceptData.js",
    `ref '${ref}' does not exist`,
  );
}

// ---------------------------------------------------------------------------
// 2. FileTree entries on the architecture page. The tree is deliberately
//    partial, so completeness isn't asserted — but each entry must exist at the
//    path the tree places it under, and be the kind (file/dir) it claims.
//    Checking the full path matters: a flat set of basenames would accept
//    model/vllm.py being drawn under cli/.
// ---------------------------------------------------------------------------
const arch = readFileSync(join(SITE, "docs/reference/architecture.md"), "utf8");
const treeBlock = arch.slice(
  arch.indexOf("<FileTree"),
  arch.indexOf("/>", arch.indexOf("<FileTree")),
);

/** Walk the nested `tree={[...]}` literal, yielding [repoRelativePath, kind]. */
const treeEntries = (block) => {
  const out = [];
  const parents = []; // ancestor dir names, innermost last
  let last = null; // most recent entry name, i.e. whose `children:` may follow
  // Tokenise just the shape we care about. `children:` pushes the entry that
  // precedes it; the `]` that closes that array pops it back off.
  for (const t of block.matchAll(
    /name: '([^']+)'|kind: '(\w+)'|children:\s*\[|\]/g,
  )) {
    if (t[1] !== undefined) {
      last = t[1].replace(/\/$/, "");
    } else if (t[2] !== undefined && last !== null) {
      out.push([[...parents, last].join("/"), t[2]]);
    } else if (t[0].startsWith("children:")) {
      parents.push(last);
    } else if (t[0] === "]" && parents.length) {
      parents.pop();
    }
  }
  return out;
};

const entries = treeEntries(treeBlock);
claim(
  entries.length >= 40,
  "reference/architecture.md",
  `only parsed ${entries.length} FileTree entries — did the tree format change?`,
);
checked += entries.length + 1;
for (const [relPath, kind] of entries) {
  const full = join(pkg, relPath);
  if (!existsSync(full)) {
    failures.push(
      `reference/architecture.md: FileTree entry 'src/faith/${relPath}' does not exist`,
    );
    continue;
  }
  const isDir = statSync(full).isDirectory();
  claim(
    isDir === (kind === "dir"),
    "reference/architecture.md",
    `'src/faith/${relPath}' is marked kind: '${kind}' but is a ${isDir ? "directory" : "file"}`,
  );
}

// The inlined SQL on the BigQuery examples page used to be compared with the
// repo's .sql files here. It is now generated from them by
// scripts/sync-example-sql.js, whose --check mode runs first in `check-repo`,
// so drift can't reach this point.

// ---------------------------------------------------------------------------
if (failures.length) {
  console.error(`✗ ${failures.length} of ${checked} repo facts are stale:\n`);
  for (const f of failures) console.error(`  ${f}`);
  console.error(`\nRepo: ${repo}`);
  process.exit(1);
}
console.log(`✓ ${checked} repo facts verified against ${repo}`);
