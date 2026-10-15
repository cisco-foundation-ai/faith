// Copyright 2025 Cisco Systems, Inc. and its affiliates
//
// SPDX-License-Identifier: Apache-2.0

// Extend super-linter's bundled ESLint config rather than replacing it.
// Pointing LINTER_RULES_PATH at a config file makes super-linter drop its own
// entirely (lib/functions/linterRules.sh: it's an either/or), so importing the
// default from its path inside the container keeps their rules and layers ours
// on top. If that path ever moves, this fails loudly with a module-resolution
// error rather than silently reverting to a thinner rule set.
import base from "/action/lib/.automation/eslint.config.mjs";

export default [
  ...base,
  {
    files: ["**/*.js", "**/*.jsx", "**/*.mjs"],
    settings: { react: { version: "detect" } },
    rules: {
      // Docusaurus builds with the automatic JSX runtime, so JSX compiles
      // without React in scope. The rule is obsolete for React 17+ and fired
      // 207 times on website/src.
      "react/react-in-jsx-scope": "off",
      // The site's components take props without declaring propTypes. Worth
      // fixing, but it's 40 errors of unrelated work; warn so it stays visible
      // without gating the build.
      "react/prop-types": "warn",
    },
  },
];
