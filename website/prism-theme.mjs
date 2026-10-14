// Resolve syntax colors from the active site palette so highlighted text is
// checked against the same background as the rendered code block.
export default {
  plain: {
    color: "var(--fai-text)",
    backgroundColor: "var(--fai-code-bg)",
  },
  styles: [
    {
      types: ["comment", "prolog", "doctype", "cdata"],
      style: { color: "var(--fai-text2)", fontStyle: "italic" },
    },
    {
      types: ["punctuation", "operator"],
      style: { color: "var(--fai-text)" },
    },
    {
      types: ["keyword", "atrule", "selector", "builtin", "tag"],
      style: { color: "var(--fai-accent)" },
    },
    {
      types: ["string", "char", "attr-value", "regex", "url"],
      style: { color: "var(--fai-syntax-string)" },
    },
    {
      types: [
        "property",
        "key",
        "attr-name",
        "number",
        "boolean",
        "constant",
        "symbol",
        "variable",
      ],
      style: { color: "var(--fai-syntax-value)" },
    },
    {
      types: ["function", "function-variable", "class-name"],
      style: { color: "var(--fai-syntax-function)" },
    },
    { types: ["deleted"], style: { color: "var(--fai-syntax-deleted)" } },
    { types: ["inserted"], style: { color: "var(--fai-syntax-inserted)" } },
    { types: ["namespace"], style: { opacity: 1 } },
    { types: ["important", "bold"], style: { fontWeight: "bold" } },
    { types: ["italic"], style: { fontStyle: "italic" } },
  ],
};
