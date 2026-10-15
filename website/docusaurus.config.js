// @ts-check
// `@type` JSDoc annotations allow editor autocompletion and type checking
// (when paired with `@ts-check`).
// See: https://docusaurus.io/docs/api/docusaurus-config

import prismTheme from "./prism-theme.mjs";

// This runs in Node.js - Don't use client-side code here (browser APIs, JSX...)

// `url` only feeds absolute links: the sitemap, canonical tags and social-card
// metadata. Vercel sets VERCEL_PROJECT_PRODUCTION_URL to the project's
// production domain on every deploy, so preview builds emit canonical links to
// production rather than to their own throwaway hostname. The fallback is only
// reached by local builds, where absolute URLs don't matter.
const productionUrl = process.env.VERCEL_PROJECT_PRODUCTION_URL
  ? `https://${process.env.VERCEL_PROJECT_PRODUCTION_URL}`
  : "http://localhost:3000";

/** @type {import('@docusaurus/types').Config} */
const config = {
  title: "FAITH",
  tagline:
    "Foundation-AI's Test Hub/Harness — benchmark language models for cybersecurity",
  favicon: "img/favicon.ico",

  // Future flags, see https://docusaurus.io/docs/api/docusaurus-config#future
  future: {
    v4: true, // Improve compatibility with the upcoming Docusaurus v4
  },

  // The production origin, used for absolute links only (see above).
  url: productionUrl,
  // Vercel serves the site at the domain root. (A GitHub Pages project site
  // would instead need '/faith/', the repo name, plus a matching `url`.)
  baseUrl: "/",

  organizationName: "cisco-foundation-ai", // GitHub org/user name.
  projectName: "faith", // repo name.
  trailingSlash: false,

  // Fail the build on a dangling link rather than warn: the docs-site workflow
  // treats a successful build as proof the site is intact, and a renamed page
  // otherwise ships its 404s green.
  onBrokenLinks: "throw",

  // Same reasoning one level down: a link to #a-heading-that-moved resolves to
  // the right page and then silently lands at the top of it, so nothing about a
  // green build says the reader ended up where the link promised.
  onBrokenAnchors: "throw",

  markdown: {
    hooks: {
      onBrokenMarkdownLinks: "throw",
    },
  },

  i18n: {
    defaultLocale: "en",
    locales: ["en"],
  },

  stylesheets: [
    "https://fonts.googleapis.com/css2?family=Cormorant+Garamond:ital,wght@0,300;0,400;0,500;0,600;0,700;1,400&family=DM+Mono:ital,wght@0,300;0,400;0,500&family=IBM+Plex+Sans:wght@300;400;500;600&display=swap",
  ],

  themes: [
    [
      "@easyops-cn/docusaurus-search-local",
      /** @type {import('@easyops-cn/docusaurus-search-local').PluginOptions} */
      ({
        hashed: true,
        indexBlog: false,
        // searchBarPosition is left at its default ("right"), which anchors the
        // results panel to right:0 so it opens leftward. That is what the field's
        // place on the right side of the navbar needs; anchoring it left:0 there
        // would hang a 560px panel off the right edge of the window.
        docsRouteBasePath: "/docs",
        highlightSearchTermsOnTargetPage: false,
        searchResultLimits: 8,
        searchBarShortcut: true,
        searchBarShortcutHint: true,
      }),
    ],
  ],

  presets: [
    [
      "classic",
      /** @type {import('@docusaurus/preset-classic').Options} */
      ({
        docs: {
          sidebarPath: "./sidebars.js",
          routeBasePath: "docs",
          editUrl:
            "https://github.com/cisco-foundation-ai/faith/tree/main/website/",
        },
        blog: false,
        theme: {
          customCss: "./src/css/custom.css",
        },
      }),
    ],
  ],

  themeConfig:
    /** @type {import('@docusaurus/preset-classic').ThemeConfig} */
    ({
      // No `image:` until a real social card exists. It previously named
      // img/faith-social-card.jpg, which was never committed, so every page
      // advertised an og:image that 404s — worse than having none, and
      // invisible to the link checker, which only validates routes. A card
      // wants to be a ~1200x630 raster; the 40x40 logo.svg won't do, since
      // Slack and Twitter don't render SVG.

      colorMode: {
        defaultMode: "dark",
        respectPrefersColorScheme: false,
      },
      navbar: {
        title: "FAITH",
        items: [
          // No doc-section tabs. The sidebar already indexes every page,
          // nested, with the current one marked; a second set of tabs into the
          // same tree is a duplicate index that has to be kept in sync by hand
          // (each tab needs an activeBaseRegex naming the routes it owns, and a
          // page missing from all of them highlights nothing).
          //
          // One "Docs" link is not that index — it is a single entry point, and
          // it earns its place on the pages with no sidebar (the landing page,
          // 404), where the header would otherwise offer no way in. No
          // activeBaseRegex, so it marks itself active only on the one page it
          // actually points at; claiming all 25 doc routes would make it look
          // like a tab that should go somewhere new and never does.
          { to: "/docs/intro", label: "Docs", position: "left" },
          // Declared before the GitHub item because right-position items render
          // in declaration order, left to right.
          //
          // No version chip. It could only be a hand-typed constant here —
          // faith-internal's version.txt is not kept current — and it links to
          // the releases page the GitHub mark beside it already reaches. Worth
          // adding once the site lives in the faith repo, where the config can
          // read version.txt at build time and cannot drift.
          { type: "search", position: "right" },
          {
            // No `label`: the mark is drawn by CSS (see .header-github-link) so
            // this reads as an icon button matching the color-mode toggle beside
            // it, rather than a text link wedged between a box and an icon.
            // aria-label carries the name for anyone who can't see the mark. The
            // mobile drawer renders this same item as a row in a list of words, so
            // custom.css gives that copy the word — a `label` here would put text
            // in the bar, which the bar would then have to hide.
            href: "https://github.com/cisco-foundation-ai/faith",
            position: "right",
            className: "header-github-link",
            "aria-label": "FAITH on GitHub",
          },
        ],
      },
      footer: {
        style: "light",
        links: [
          {
            title: "Docs",
            items: [
              { label: "Introduction", to: "/docs/intro" },
              {
                label: "Installation",
                to: "/docs/getting-started/installation",
              },
              { label: "Quickstart", to: "/docs/getting-started/quickstart" },
            ],
          },
          {
            title: "Reference",
            items: [
              { label: "CLI commands", to: "/docs/cli/overview" },
              { label: "Benchmark catalog", to: "/docs/benchmarks/catalog" },
              {
                label: "BigQuery analytics",
                to: "/docs/analytics/bigquery/quickstart",
              },
            ],
          },
          {
            title: "More",
            items: [
              {
                label: "GitHub",
                href: "https://github.com/cisco-foundation-ai/faith",
              },
              { label: "Foundation AI", href: "https://www.cisco.com/" },
            ],
          },
        ],
        copyright: `Copyright © ${new Date().getFullYear()} Cisco Systems, Inc. and its affiliates. Built with Docusaurus.`,
      },
      prism: {
        theme: prismTheme,
        darkTheme: prismTheme,
        additionalLanguages: ["bash", "yaml", "sql", "python"],
      },
    }),
};

export default config;
