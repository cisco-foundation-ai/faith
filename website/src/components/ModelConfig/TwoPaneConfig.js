import { useState, useMemo, useId } from "react";
import clsx from "clsx";
import CodeBlock from "@theme/CodeBlock";
import { GROUPS } from "./configs";
import styles from "./TwoPaneConfig.module.css";

// ---------------------------------------------------------------------------
// Shared helpers
// ---------------------------------------------------------------------------

// Bucket a group's models into vendor "families" so 30+ configs collapse into
// a handful of expandable rows. Order matters — first match wins.
const FAMILY_RULES = [
  [/^foundation-sec/, "Foundation-Sec"],
  [/^llama-primus/, "Llama-Primus"],
  [/^llama/, "Llama"],
  [/^qwen3/, "Qwen3.5"],
  [/^qwen2/, "Qwen2.5"],
  [/^gemma/, "Gemma"],
  [/^falcon/, "Falcon3"],
  [/^mistral/, "Mistral"],
  [/^phi/, "Phi"],
  [/^gpt-oss/, "GPT-OSS"],
  [/^gpt-4|^gpt-5|^o3/, "OpenAI"],
  [/^deephat/, "DeepHat"],
  [/^lily/, "Lily"],
  [/^whiterabbitneo/, "WhiteRabbitNeo"],
];

function familyOf(name) {
  for (const [re, label] of FAMILY_RULES) if (re.test(name)) return label;
  return "Other";
}

function useFamilies(group) {
  return useMemo(() => {
    const map = new Map();
    for (const m of group.models) {
      const fam = familyOf(m.name);
      if (!map.has(fam)) map.set(fam, []);
      map.get(fam).push(m);
    }
    return [...map.entries()].map(([label, models]) => ({ label, models }));
  }, [group]);
}

function GroupToggle({ groupId, onSelect, controls }) {
  return (
    <div
      className={styles.groupToggle}
      role="group"
      aria-label="Configuration category"
    >
      {GROUPS.map((g) => (
        <button
          key={g.id}
          type="button"
          aria-pressed={groupId === g.id}
          aria-controls={controls}
          className={clsx(
            styles.groupBtn,
            groupId === g.id && styles.groupBtnOn,
          )}
          onClick={() => onSelect(g)}
        >
          {g.label}
          <span className={styles.groupCount}>{g.models.length}</span>
        </button>
      ))}
    </div>
  );
}

function Yaml({ model, id }) {
  return (
    <div className={styles.codeWrap} id={id}>
      <CodeBlock language="yaml" title={`${model.id}.yaml`}>
        {model.yaml}
      </CodeBlock>
    </div>
  );
}

// ===========================================================================
// Two panes: searchable, collapsible model-family groups on the left, and the
// selected model's YAML plus a ready-to-run command on the right.
// ===========================================================================
export function TwoPaneConfig() {
  const pickerId = useId();
  const listId = `${pickerId}-models`;
  const yamlId = `${pickerId}-yaml`;
  const commandId = `${pickerId}-command`;
  const statusId = `${pickerId}-status`;
  const [groupId, setGroupId] = useState(GROUPS[0].id);
  const group = GROUPS.find((g) => g.id === groupId);
  const [modelId, setModelId] = useState(group.models[0].id);
  const [query, setQuery] = useState("");
  const [collapsed, setCollapsed] = useState(() => new Set());
  const model = group.models.find((m) => m.id === modelId) || group.models[0];

  const q = query.trim().toLowerCase();
  const searching = q.length > 0;
  const families = useFamilies(group)
    .map((fam) => ({
      ...fam,
      models: q ? fam.models.filter((m) => m.name.includes(q)) : fam.models,
    }))
    .filter((fam) => fam.models.length);
  const matchCount = families.reduce(
    (count, fam) => count + fam.models.length,
    0,
  );

  const onGroup = (g) => {
    setGroupId(g.id);
    setModelId(g.models[0].id);
    setQuery("");
    setCollapsed(new Set());
  };

  const toggleFam = (label) =>
    setCollapsed((prev) => {
      const next = new Set(prev);
      next.has(label) ? next.delete(label) : next.add(label);
      return next;
    });

  return (
    <>
      <div className={clsx(styles.card, styles.twoPane)}>
        <aside className={styles.pane} aria-label="Built-in model selector">
          <GroupToggle
            groupId={groupId}
            onSelect={onGroup}
            controls={`${listId} ${yamlId} ${commandId}`}
          />
          <input
            type="text"
            className={styles.search}
            placeholder="Search models…"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            aria-label="Search models"
            aria-controls={listId}
            aria-describedby={statusId}
          />
          <div
            id={statusId}
            className={styles.filterStatus}
            role="status"
            aria-atomic="true"
          >
            {matchCount} {group.label.toLowerCase()} configuration
            {matchCount === 1 ? "" : "s"} match
            {matchCount === 1 ? "es" : ""}. Selected: {model.id}.
          </div>
          <div className={styles.scrollList} id={listId}>
            {families.map((fam) => {
              const isOpen = searching || !collapsed.has(fam.label);
              return (
                <div key={fam.label} className={styles.famBlock}>
                  <button
                    type="button"
                    className={styles.famLabel}
                    aria-expanded={isOpen}
                    aria-disabled={searching}
                    aria-controls={`${pickerId}-family-${fam.label}`}
                    onClick={() => !searching && toggleFam(fam.label)}
                  >
                    <span
                      className={clsx(
                        styles.famChevron,
                        isOpen && styles.famChevronOpen,
                      )}
                      aria-hidden="true"
                    >
                      ▸
                    </span>
                    <span className={styles.famName}>{fam.label}</span>
                    <span className={styles.famCount}>{fam.models.length}</span>
                  </button>
                  <div
                    id={`${pickerId}-family-${fam.label}`}
                    className={styles.familyModels}
                    hidden={!isOpen}
                  >
                    {fam.models.map((m) => (
                      <button
                        key={m.id}
                        type="button"
                        aria-pressed={model.id === m.id}
                        aria-controls={`${yamlId} ${commandId}`}
                        className={clsx(
                          styles.listItem,
                          model.id === m.id && styles.listItemOn,
                        )}
                        onClick={() => setModelId(m.id)}
                      >
                        {m.name}
                      </button>
                    ))}
                  </div>
                </div>
              );
            })}
            {!families.length && (
              <div className={styles.empty}>
                No matches in {group.label.toLowerCase()}. Clear the filter or
                select another category. The selected configuration remains
                displayed.
              </div>
            )}
          </div>
        </aside>
        <Yaml model={model} id={yamlId} />
      </div>

      <div className={styles.example} id={commandId}>
        <span className={styles.exampleLabel}>Q&A example</span>
        <CodeBlock language="shell">
          {`faith query --model-configs ${model.id} \\\n  --benchmarks ctibench-mcqa \\\n  --sample-size 25 \\\n  --num-trials 1 \\\n  --datastore-location out/model-config`}
        </CodeBlock>
      </div>
    </>
  );
}
