// @ts-check
// `@type` JSDoc annotations allow editor autocompletion and type checking
// (when paired with `@ts-check`).
// See: https://docusaurus.io/docs/api/docusaurus-config

import { themes as prismThemes } from "prism-react-renderer";

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
    "https://fonts.googleapis.com/css2?family=Cormorant+Garamond:ital,wght@0,300;0,400;0,500;0,600;0,700;1,400&family=DM+Mono:ital,wght@0,300;0,400;0,500&family=Poppins:wght@300;400;500;600&display=swap",
  ],

  themes: [
    [
      "@easyops-cn/docusaurus-search-local",
      /** @type {import('@easyops-cn/docusaurus-search-local').PluginOptions} */
      ({
        hashed: true,
        indexBlog: false,
        // The search bar lives in the left sidebar, so the results dropdown
        // must open rightward (left:0) instead of the navbar default (right:0).
        searchBarPosition: "left",
        docsRouteBasePath: "/docs",
        highlightSearchTermsOnTargetPage: true,
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
          {
            // Every nav tab points into the same docs sidebar, so a plain
            // docSidebar "Docs" item would highlight on ALL doc pages —
            // including Benchmarks/Architecture/Analytics. Scope each tab's
            // active state to the doc sections it actually owns.
            //
            // Docs owns everything the other three tabs don't, so it has to
            // name each of those sections; a page missing from all four
            // regexes leaves every tab unhighlighted.
            to: "/docs/intro",
            label: "Docs",
            position: "left",
            activeBaseRegex:
              "/docs/(intro|getting-started|cli|reference|contributing)",
          },
          {
            to: "/docs/benchmarks/catalog",
            label: "Benchmarks",
            position: "left",
            activeBaseRegex: "/docs/benchmarks",
          },
          {
            to: "/docs/architecture/overview",
            label: "Architecture",
            position: "left",
            activeBaseRegex: "/docs/architecture",
          },
          {
            to: "/docs/bigquery/quickstart",
            label: "Analytics",
            position: "left",
            activeBaseRegex: "/docs/bigquery",
          },
          {
            href: "https://github.com/cisco-foundation-ai/faith",
            label: "GitHub",
            position: "right",
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
              { label: "CLI Commands", to: "/docs/cli/overview" },
              { label: "Benchmark Catalog", to: "/docs/benchmarks/catalog" },
              { label: "BigQuery Analytics", to: "/docs/bigquery/quickstart" },
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
        theme: prismThemes.github,
        darkTheme: prismThemes.dracula,
        additionalLanguages: ["bash", "yaml", "sql", "python"],
      },
    }),
};

export default config;
