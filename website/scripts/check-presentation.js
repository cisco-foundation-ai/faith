import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import test from "node:test";
import prismTheme from "../prism-theme.mjs";

const site = fileURLToPath(new URL("../", import.meta.url));
const css = fs.readFileSync(path.join(site, "src/css/custom.css"), "utf8");

function luminance(hex) {
  let digits = hex.replace("#", "");
  if (digits.length === 3) digits = [...digits].map((c) => c + c).join("");
  const channels = digits.match(/../g).map((c) => {
    const value = parseInt(c, 16) / 255;
    return value <= 0.04045 ? value / 12.92 : ((value + 0.055) / 1.055) ** 2.4;
  });
  return channels[0] * 0.2126 + channels[1] * 0.7152 + channels[2] * 0.0722;
}

function contrast(first, second) {
  const a = luminance(first);
  const b = luminance(second);
  return (Math.max(a, b) + 0.05) / (Math.min(a, b) + 0.05);
}

test("contrast calculation matches the black/white reference", () => {
  assert.equal(contrast("#000", "#fff"), 21);
  assert.equal(contrast("#123456", "#123456"), 1);
});

for (const mode of ["light", "dark"]) {
  test(`${mode} syntax colors meet 4.5:1 on the actual code background`, () => {
    const block = css.match(
      new RegExp(`\\[data-theme="${mode}"\\] \\{([\\s\\S]*?)\\n\\}`),
    );
    assert.ok(block, `Missing ${mode} palette`);
    const palette = Object.fromEntries(
      [...block[1].matchAll(/(--[\w-]+):\s*(#[\da-f]+);/gi)].map((m) => [
        m[1],
        m[2],
      ]),
    );
    const foregrounds = [
      prismTheme.plain,
      ...prismTheme.styles.map((rule) => rule.style),
    ];
    for (const style of foregrounds) {
      assert.ok(
        style.opacity === undefined || style.opacity === 1,
        "Token opacity reduces contrast",
      );
      if (!style.color) continue;
      const name = style.color.match(/^var\((--[\w-]+)\)$/)?.[1];
      assert.ok(palette[name], `Unresolved syntax color ${style.color}`);
      const ratio = contrast(palette[name], palette["--fai-code-bg"]);
      assert.ok(ratio >= 4.5, `${mode} ${name}: ${ratio.toFixed(2)}:1`);
    }
  });
}

test("both Prism modes use the tested palette and code background", () => {
  const config = fs.readFileSync(
    path.join(site, "docusaurus.config.js"),
    "utf8",
  );
  assert.match(config, /theme: prismTheme/);
  assert.match(config, /darkTheme: prismTheme/);
  assert.equal(prismTheme.plain.backgroundColor, "var(--fai-code-bg)");
});

test("each documentation tab set has a distinct explicit URL parameter", () => {
  const visit = (directory) =>
    fs.readdirSync(directory, { withFileTypes: true }).flatMap((entry) => {
      const name = path.join(directory, entry.name);
      return entry.isDirectory() ? visit(name) : [name];
    });
  let count = 0;
  const documents = visit(path.join(site, "docs")).filter((name) =>
    /\.mdx?$/.test(name),
  );
  for (const name of documents) {
    const parameters = new Set();
    for (const [tag] of fs
      .readFileSync(name, "utf8")
      .matchAll(/<Tabs\b[^>]*>/g)) {
      const parameter = tag.match(/queryString="([\w-]+)"/)?.[1];
      assert.ok(
        parameter,
        `${name}: tab selection cannot be shared in the URL`,
      );
      assert.ok(
        !parameters.has(parameter),
        `${name}: duplicate tab URL parameter ${parameter}`,
      );
      parameters.add(parameter);
      count += 1;
    }
  }
  assert.ok(count > 0, "No documentation tab sets found");
});

test("the documented Node minimum matches the site and locked dependencies", () => {
  const packageJson = JSON.parse(
    fs.readFileSync(path.join(site, "package.json"), "utf8"),
  );
  const lock = JSON.parse(
    fs.readFileSync(path.join(site, "package-lock.json"), "utf8"),
  );
  const contributing = fs.readFileSync(
    path.join(site, "docs/contributing.md"),
    "utf8",
  );
  const minimum = packageJson.engines.node.match(/^>=(\d+\.\d+\.\d+)$/)?.[1];
  assert.ok(minimum, "Declare an explicit minimum Node patch version");
  assert.equal(lock.packages[""].engines.node, packageJson.engines.node);
  assert.ok(contributing.includes(`Node.js ${minimum} or later`));
  // These transitive packages raised the minimum beyond Docusaurus's floor.
  for (const name of [
    "node_modules/@easyops-cn/docusaurus-search-local/node_modules/cheerio",
    "node_modules/undici",
  ]) {
    assert.equal(
      lock.packages[name].engines.node,
      packageJson.engines.node,
      `${name}: review the documented Node minimum if the locked requirement changes`,
    );
  }
});
