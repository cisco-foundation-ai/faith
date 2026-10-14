#!/usr/bin/env node
// Regenerates the ```sql blocks on the BigQuery examples page from the repo's
// runnable .sql files, so the page and the files can never disagree.
//
//   node scripts/sync-example-sql.js           # rewrite from this repo
//   node scripts/sync-example-sql.js --check   # CI gate
//
// Each block names its source in the fence info string:
//
//   ```sql title="queries/leaderboard.sql"
//
// which resolves to docs/bigquery/queries/leaderboard.sql in the repo. The
// licence and usage comment preamble is stripped; the statement is inlined
// verbatim.
//
// Generating a committed file, rather than reading the .sql at build time via a
// remark plugin, is deliberate: Vercel fences a build to its Root Directory
// (website/) and forbids `..`, so nothing outside it is readable when the site
// builds. A build-time plugin works in CI, where the whole repo is checked out,
// and fails only on Vercel — green CI, broken deploy.
//
// The site lives at website/ inside the faith repo, so the source of truth is
// the enclosing checkout. --repo exists only to point the site at another tree.

import { readFileSync, writeFileSync, existsSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

// Resolve paths from this file, not the cwd, so the script works from anywhere.
const SITE = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const args = process.argv.slice(2);
const check = args.includes("--check");
const repoIdx = args.indexOf("--repo");
const repo = repoIdx === -1 ? resolve(SITE, "..") : resolve(args[repoIdx + 1]);

const SQL_DIR = join(repo, "docs/bigquery");
const PAGE = join(SITE, "docs/analytics/bigquery/examples.md");

if (!existsSync(SQL_DIR)) {
  console.error(
    `No SQL directory at ${SQL_DIR}\n` +
      `Expected the site to live at website/ inside the faith repo, ` +
      `or pass --repo <path-to-faith-checkout>.`,
  );
  process.exit(2);
}

// Drop the leading licence/usage comment preamble; keep the statement itself.
// Once a non-comment line is seen, everything after it is kept verbatim, so a
// comment *inside* the statement survives.
const statement = (file) =>
  readFileSync(file, "utf8")
    .split("\n")
    .reduce(
      (acc, l) =>
        acc.length || (l.trim() && !l.startsWith("--")) ? [...acc, l] : acc,
      [],
    )
    .join("\n")
    .trim();

const current = readFileSync(PAGE, "utf8");
const problems = [];
let synced = 0;

// Match every sql block, then require the title separately. Folding
// `title="..."` into the pattern would make an untitled block simply not match,
// so its SQL would silently go unsynced with nothing to show for it.
const generated = current.replace(
  /(```sql)([^\n]*)(\n)([\s\S]*?)(```)/g,
  (_, open, meta, nl, _body, close) => {
    const title = meta.match(/title="([^"]+)"/)?.[1];
    if (!title) {
      problems.push(
        'a ```sql block has no title="...", so it has no source file to sync from',
      );
      return `${open}${meta}${nl}${_body}${close}`;
    }
    const src = join(SQL_DIR, title);
    if (!existsSync(src)) {
      problems.push(`no such file in the repo: docs/bigquery/${title}`);
      return `${open}${meta}${nl}${_body}${close}`;
    }
    synced += 1;
    return `${open}${meta}${nl}${statement(src)}\n${close}`;
  },
);

// An authoring error (missing title or missing file) is fatal in both modes:
// rewriting the page around a block we can't resolve would quietly leave stale
// SQL behind a passing exit code.
if (problems.length) {
  console.error(
    `✗ ${problems.length} problem(s) in docs/analytics/bigquery/examples.md:\n`,
  );
  for (const p of problems) console.error(`  ${p}`);
  process.exit(1);
}

if (!synced) {
  console.error(
    `No \`\`\`sql blocks found in ${PAGE} — did the page change shape?`,
  );
  process.exit(1);
}

if (check) {
  if (current !== generated) {
    console.error(
      `examples.md is out of sync with ${SQL_DIR}.\n` +
        `Run: node scripts/sync-example-sql.js${
          repoIdx === -1 ? "" : ` --repo ${repo}`
        }`,
    );
    process.exit(1);
  }
  console.log(`examples.md in sync (${synced} SQL blocks).`);
} else if (current === generated) {
  console.log(`${PAGE} already up to date (${synced} SQL blocks).`);
} else {
  writeFileSync(PAGE, generated);
  console.log(`Wrote ${PAGE} — ${synced} SQL blocks.`);
}
