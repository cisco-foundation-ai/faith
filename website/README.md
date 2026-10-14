# FAITH Documentation Site

The documentation site for **FAITH** — Foundation-AI's Test Hub/Harness.
Built with [Docusaurus](https://docusaurus.io/), a modern static-site
generator.

The site lives at `website/` inside the FAITH repository. That's deliberate: a
doc fix and the code change that prompted it land in the same PR, and the drift
checks below read the surrounding checkout directly, with nothing to configure.

## Local development

```bash
cd website
npm install     # first time only
npm run start
```

`npm run start` launches a dev server and opens a browser window. Most edits
are reflected live without a restart.

## Build

```bash
npm run build
```

Generates static files into `build/`, servable by any static host. Preview the
production build locally with `npm run serve`.

## Project structure

```text
website/
├── docs/                 # All documentation pages (MDX)
│   ├── intro.md
│   ├── getting-started/
│   ├── cli/
│   ├── benchmarks/
│   ├── architecture/
│   ├── bigquery/
│   ├── reference/
│   └── contributing.md
├── src/
│   ├── components/        # Diagram + table components the docs import
│   │   └── landing/       # Landing-page sections (shared.js)
│   ├── css/custom.css     # Brand palette (cyan, --ifm-color-primary)
│   ├── pages/index.js     # Landing page
│   └── theme/             # Docusaurus theme overrides (sidebar)
├── static/img/           # Logo and favicon
├── docusaurus.config.js  # Site config, navbar, footer
├── sidebars.js           # Left-sidebar structure
└── scripts/              # Generators + drift checks (see below)
```

To add a page, drop a Markdown file into the right `docs/` subfolder and add its
ID to `sidebars.js`.

## Staying in sync with the FAITH repository

Parts of this site mirror facts that live in the `faith` source tree. Those parts
are **generated or checked**, never hand-maintained, so they can't quietly drift:

| What                                       | Source of truth                                                              | Tool                                            |
| ------------------------------------------ | ---------------------------------------------------------------------------- | ----------------------------------------------- |
| Packaged model configs                     | `src/faith/__models__/*.yaml`                                                | `npm run sync-configs` (generates `configs.js`) |
| QueryFlow / FileTree source paths          | the `src/faith/` tree                                                        | `npm run check-paths`                           |
| Inlined BigQuery SQL                       | `docs/bigquery/{queries,views}/*.sql`                                        | `npm run sync-sql` (rewrites `examples.md`)     |
| CLI flags and defaults                     | the argparse parser in `faith.cli.cli`                                       | `npm run check-facts`                           |
| Benchmark state / categories / metrics     | each `benchmark.yaml`                                                        | `npm run check-facts`                           |
| Benchmark groups                           | each `benchmark.yaml`                                                        | `npm run check-facts`                           |
| The intro's suite list, names and tags     | `benchmarks/catalog.md`                                                      | `npm run check-facts`                           |
| BigQuery table schema                      | `METRICS_SCHEMA` in `ingestion/storage/bigquery.py`                          | `npm run check-facts`                           |
| `agentic_config` sections and limits       | `_types/config/agentic.py`                                                   | `npm run check-facts`                           |
| Registered agentic tools and token budgets | `agentic/registries.py`, `cli/subcmd/query.py`, `_types/model/generation.py` | `npm run check-facts`                           |
| Groups, and their agentic exclusion        | `benchmark/listing.py`                                                       | `npm run check-facts`                           |
| The install extras table                   | `[project.optional-dependencies]`                                            | `npm run check-facts`                           |

Run everything at once:

```bash
npm run check-repo
```

No arguments: the scripts resolve the faith package from their own location
(`website/scripts/` → `../../src/faith`), so they always check the repository they're
committed in. Pass `--repo <path>` to check against a different faith tree.

`check-repo` is read-only and exits non-zero on any drift, so it works as a CI
gate. To _fix_ drift rather than report it, run the generator without `--check`:
`npm run sync-configs` or `npm run sync-sql`.

The two generators write files that are committed, rather than reading their
sources when the site builds. That's a constraint, not a preference: Vercel
builds the site with its Root Directory set to `website/`, and a build there
can't read outside that directory or use `..`. Pulling `docs/bigquery/*.sql` in
at build time works in CI, where the whole repository is checked out, and fails
only on Vercel — so it would ship green and break on deploy. Generating into a
committed file keeps the source of truth in the repository and the build
self-contained.

Each check reports how many facts it verified, and the parsers assert a floor on
what they matched. That matters more than it sounds: a checker whose pattern stops
matching passes silently, having proven nothing. The count is the signal that
coverage is still there, so treat a sudden drop as a failure even when the run is
green.

The `docs site` workflow (`.github/workflows/docs-site.yml`) runs `check-repo`
and a full site build on every push and pull request that touches `website/**`,
`src/faith/**` or `docs/bigquery/**` — either side of a mirrored fact — plus
`pyproject.toml`, since the checkers import `faith.cli` and break if a
dependency leaves the install. The checkers only need `faith.cli` and
`faith.benchmark`, so that job installs faith with no extras.

The build step is a real gate on link integrity, not just a smoke test:
`onBrokenLinks`, `onBrokenAnchors` and `onBrokenMarkdownLinks` are all `throw`,
so a renamed or deleted page fails CI instead of shipping 404s — and so does a
link to a heading that no longer exists. That check reads the rendered pages, so
it covers links a component builds, not just the ones written in Markdown.

Anything not in that table — prose, editorial framing, the benchmark
descriptions, `--engine-kwargs` examples — is hand-authored and needs a human
review when FAITH changes.

## How the `lint` workflow treats the site

Two of super-linter's linters genuinely can't handle `website/`: markdownlint
reads MDX's JSX as stray inline HTML, and prettier reformats MDX into a shape
markdownlint then rejects. The obvious fix, `FILTER_REGEX_EXCLUDE`, is the wrong
tool — it drops a file from **every** linter at once, so silencing those two also
stopped codespell and textlint from reading 22 doc pages. That blind spot was
hiding a real typo.

Each exclusion now lives with the linter that owns it, so nothing is hidden from
a linter that had no complaint:

| Where                               | Scope              | Why                                         |
| ----------------------------------- | ------------------ | ------------------------------------------- |
| `.markdownlintignore`               | `website/docs/`    | MDX parses as invalid Markdown              |
| `.prettierignore`                   | `website/docs/`    | reformats MDX into a shape MD rejects       |
| `.prettierignore`                   | `configs.js`       | generated; reformatting it looks like drift |
| `.github/linters/.stylelintrc.json` | `website/src/` CSS | CSS Modules conventions (below)             |
| `.github/linters/eslint.config.mjs` | `website/src/` JSX | React 17+ automatic JSX runtime             |

Codespell and textlint now read every page under `docs/`, which is the point.

### Why the scripts are `.js` and not `.mjs`

Super-linter never lints `.mjs`. It derives a file's type from its raw
extension (`lib/functions/detectFiles.sh`) and then dispatches on
`FILE_TYPE == "js"` (`lib/functions/buildFileList.sh`), so an `.mjs` file
reaches neither ESLint nor prettier — silently, with the run still green.

The scripts in `scripts/` were `.mjs`, so all three went unlinted. Renaming
them to `.js` fixes it, but they use ESM, which needs `"type": "module"`.
Setting that in `website/package.json` breaks the site build: Docusaurus
relies on extensionless imports (`./configs`), which ESM rejects as not fully
specified. Hence `scripts/package.json` — a nested manifest scopes
`"type": "module"` to that directory, leaving the build untouched.

Worth knowing before adding another `.mjs` anywhere under `website/`: CI will
not check it, and nothing will say so.

Neither config file can carry its own rationale — JSON has no comments — so:

- **`selector-class-pattern`, `keyframes-name-pattern`** off. CSS Modules classes
  are referenced as JS identifiers (`styles.heroGlow`), so camelCase is correct;
  the Infima theme classes this file overrides are BEM (`.button--primary`). The
  kebab-case default rejects both, 309 times over.
- **`selector-pseudo-class-no-unknown`** allows `:global()`, a CSS Modules
  construct stylelint doesn't recognize.
- **`no-descending-specificity`** off. It fires 7 times on the Docusaurus theme
  overrides in `custom.css`, where element-scoped rules (`h1 code`) deliberately
  follow class-scoped ones (`.markdown :not(pre) > code:not([class])`) and
  losing the cascade is the intent.
- **`media-feature-range-notation: "prefix"`** keeps `max-width: 996px` rather
  than autofixing to `width <= 996px`, which needs Safari 16.4+/Chrome 104+ —
  narrower than this site's `browserslist`.
- **`property-no-vendor-prefix`** off: autofix was stripping
  `-webkit-mask-image`, silently breaking the hero grid's fade in Safari.
- **`react/react-in-jsx-scope`** off and **`react/prop-types`** downgraded to a
  warning. Docusaurus uses the automatic JSX runtime, so `import React` is
  obsolete; the missing propTypes are worth fixing but shouldn't gate a build.

`eslint.config.mjs` **imports** super-linter's bundled config instead of copying
it, because pointing `LINTER_RULES_PATH` at a config file makes super-linter drop
its own entirely. If that path ever moves the import fails loudly, rather than
quietly reverting to a thinner rule set.

## Deployment

The config targets **Vercel**: `baseUrl` is `'/'`, since Vercel serves the site
at the domain root. All that's left is to connect the project:

1. Import the repository in Vercel and set **Root Directory** to `website`. Vercel
   handles a subdirectory site natively, so no `vercel.json` is needed.
2. Leave the framework preset as auto-detected (Docusaurus).
3. Confirm **Settings → Environment Variables → Enable access to System
   Environment Variables** is on. See the `url` note below for why.

That's it — and it gives a preview deploy per pull request, which GitHub Pages
doesn't.

`url` only affects absolute links (sitemap, canonical tags, social-card
metadata) and is read from `VERCEL_PROJECT_PRODUCTION_URL`, which Vercel sets at
build time on every deploy, previews included — so preview builds emit canonical
links pointing at production rather than at their own throwaway hostname. The
value carries no scheme, hence the `https://` prefix in the config.

Without system environment variables enabled that variable is unset and the
build still **succeeds**, silently falling back to `http://localhost:3000`: a
valid URL as far as Docusaurus is concerned, so nothing errors, but the sitemap
and every canonical tag would then point at localhost. Local builds hit the same
fallback, where absolute URLs don't matter.

### Serving under a path instead (e.g. GitHub Pages)

A project site at `https://<org>.github.io/faith/` is served under a path, not
the root, so it needs `baseUrl: '/faith/'` and `url: 'https://<org>.github.io'`.
Getting this wrong is the usual cause of a site that loads with no CSS. You'd
also need a workflow that builds with `working-directory: website`, uploads
`website/build`, and **Settings → Pages → Source** set to **GitHub Actions**.
