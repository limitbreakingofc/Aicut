import React, { useMemo, useRef, useState } from "react";
import { Blocks, CircleCheck, Copy, Play, Terminal, TriangleAlert, Wrench } from "lucide-react";
import { runAction } from "../../lib/client";
import type { ActionSpec, ModuleInfo, NotifyFn } from "../../lib/types";
import { copyToClipboard } from "../../lib/ui";
import { pageMeta } from "../../app/navigation";
import {
  Badge,
  BusyIcon,
  CodeBlock,
  Field,
  MoreButton,
  PageHeader,
  Panel,
  Switch,
  useContextMenu,
  type MenuItems,
} from "../../components";
import "./core.css";

interface ModulesViewProps {
  modules: ModuleInfo[];
  actions: ActionSpec[];
  onRefresh: () => void;
  onNotify: NotifyFn;
}

export const ModulesView: React.FC<ModulesViewProps> = ({ modules, actions, onRefresh, onNotify }) => {
  const menu = useContextMenu();
  const playgroundRef = useRef<HTMLDivElement>(null);
  const [selectedActionId, setSelectedActionId] = useState<string>(actions[0]?.id ?? "system.info");
  const [jsonInput, setJsonInput] = useState<string>("{}");
  const [resultText, setResultText] = useState<string>("");
  const [running, setRunning] = useState(false);

  const selectedSpec = actions.find((a) => a.id === selectedActionId) ?? actions[0] ?? null;

  const actionsByModule = useMemo(() => {
    const map = new Map<string, ActionSpec[]>();
    for (const a of actions) map.set(a.module_id, [...(map.get(a.module_id) ?? []), a]);
    return map;
  }, [actions]);

  async function toggleModule(mod: ModuleInfo) {
    try {
      await runAction("system.set_module_enabled", {
        module_id: mod.manifest.id,
        enabled: !mod.enabled,
      });
      onRefresh();
      onNotify("info", mod.manifest.name, !mod.enabled ? "Módulo ativado para UI e IAs." : "Módulo desativado.");
    } catch (e) {
      onNotify("error", "Erro ao alterar módulo", (e as Error).message);
    }
  }

  async function handleRunPlayground() {
    if (!selectedSpec) return;
    setRunning(true);
    setResultText("");
    try {
      const parsed = JSON.parse(jsonInput || "{}");
      const out = await runAction(selectedSpec.id, parsed);
      setResultText(JSON.stringify(out, null, 2));
    } catch (e) {
      setResultText(JSON.stringify({ error: (e as Error).message }, null, 2));
    } finally {
      setRunning(false);
    }
  }

  const openInPlayground = (actionId: string) => {
    setSelectedActionId(actionId);
    setJsonInput("{}");
    playgroundRef.current?.scrollIntoView({ behavior: "smooth", block: "start" });
  };

  const moduleMenu = (mod: ModuleInfo): MenuItems => [
    { type: "label", label: `${mod.manifest.name} · v${mod.manifest.version}` },
    !mod.manifest.core && {
      label: mod.enabled ? "Desativar módulo" : "Ativar módulo",
      icon: mod.enabled ? TriangleAlert : CircleCheck,
      danger: mod.enabled,
      onSelect: () => void toggleModule(mod),
    },
    mod.action_ids.length > 0 && {
      label: "Abrir no Action Playground",
      icon: Terminal,
      onSelect: () => openInPlayground(mod.action_ids[0]),
    },
    { type: "separator" },
    { label: "Copiar ID do módulo", icon: Copy, onSelect: () => void copyToClipboard(mod.manifest.id, onNotify, "ID do módulo") },
    {
      label: "Copiar lista de actions",
      icon: Copy,
      onSelect: () => void copyToClipboard(mod.action_ids.join("\n"), onNotify, "Actions"),
    },
  ];

  const ann = selectedSpec?.annotations;

  return (
    <>
      <PageHeader
        {...pageMeta("modules")}
        actions={
          <span className="row gap-xs">
            <Badge>{modules.length} módulos</Badge>
            <Badge tone="accent">{actions.length} actions</Badge>
          </span>
        }
      />

      <div className="table-wrap">
        <table className="table">
          <thead>
            <tr>
              <th>módulo</th>
              <th className="col-fit">versão</th>
              <th className="col-fit">saúde</th>
              <th className="col-fit num">actions</th>
              <th className="col-fit c-opt">ferramentas</th>
              <th className="col-fit">ativo</th>
              <th className="col-fit" aria-label="Ações" />
            </tr>
          </thead>
          <tbody>
            {modules.map((mod) => (
              <tr key={mod.manifest.id} onContextMenu={(e) => menu.open(e, moduleMenu(mod))}>
                <td className="cell-title">
                  <div className="row gap-xs">
                    <Blocks size={14} className="text-accent" aria-hidden />
                    <span className="strong">{mod.manifest.name}</span>
                    <span className="muted text-xs">{mod.manifest.id}</span>
                    {mod.manifest.core && <Badge tone="info">core</Badge>}
                  </div>
                  <div className="secondary text-xs truncate" title={mod.manifest.description}>
                    {mod.manifest.description}
                  </div>
                </td>
                <td className="col-fit muted">v{mod.manifest.version}</td>
                <td className="col-fit">
                  <Badge tone={mod.health.healthy ? "success" : "warning"} icon={mod.health.healthy ? CircleCheck : TriangleAlert}>
                    {mod.health.message}
                  </Badge>
                </td>
                <td className="col-fit num">
                  <button
                    type="button"
                    className="btn btn-ghost btn-xs"
                    disabled={mod.action_ids.length === 0}
                    title="Abrir no Action Playground"
                    onClick={() => mod.action_ids[0] && openInPlayground(mod.action_ids[0])}
                  >
                    {mod.action_ids.length}
                  </button>
                </td>
                <td className="col-fit c-opt">
                  {mod.tool_ids.length > 0 ? (
                    <span className="row gap-xs muted text-xs">
                      <Wrench size={12} aria-hidden /> {mod.tool_ids.join(", ")}
                    </span>
                  ) : (
                    <span className="muted">—</span>
                  )}
                </td>
                <td className="col-fit">
                  {mod.manifest.core ? (
                    <span className="muted text-xs">sempre</span>
                  ) : (
                    <Switch
                      checked={mod.enabled}
                      onChange={() => void toggleModule(mod)}
                      label={mod.enabled ? `Desativar ${mod.manifest.name}` : `Ativar ${mod.manifest.name}`}
                    />
                  )}
                </td>
                <td className="col-fit">
                  <MoreButton items={() => moduleMenu(mod)} />
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <div ref={playgroundRef} className="scroll-anchor">
      <Panel
        title="Action Playground"
        icon={Terminal}
        actions={<span className="muted text-xs">teste qualquer Action / Tool exposta às IAs</span>}
      >
        <div className="grid-2 playground">
          <div className="stack">
            <Field label="Action registrada" htmlFor="pg-action">
              <select
                id="pg-action"
                className="select"
                value={selectedSpec?.id ?? ""}
                onChange={(e) => {
                  setSelectedActionId(e.target.value);
                  setJsonInput("{}");
                }}
              >
                {Array.from(actionsByModule.entries()).map(([moduleId, list]) => (
                  <optgroup key={moduleId} label={moduleId}>
                    {list.map((a) => (
                      <option key={a.id} value={a.id}>
                        {a.id} — {a.title}
                      </option>
                    ))}
                  </optgroup>
                ))}
              </select>
            </Field>

            {selectedSpec && (
              <div className="stack stack-sm">
                <p className="secondary text-sm">{selectedSpec.description}</p>
                {ann && (
                  <div className="row wrap gap-xs">
                    {ann.read_only && <Badge tone="info">somente leitura</Badge>}
                    {ann.destructive && <Badge tone="danger">destrutiva</Badge>}
                    {ann.idempotent && <Badge>idempotente</Badge>}
                    {ann.long_running && <Badge tone="warning">longa duração</Badge>}
                  </div>
                )}
              </div>
            )}

            <Field label="Parâmetros (JSON)" htmlFor="pg-json" hint="Ctrl+Enter executa">
              <textarea
                id="pg-json"
                className="textarea"
                rows={8}
                spellCheck={false}
                value={jsonInput}
                onChange={(e) => setJsonInput(e.target.value)}
                onKeyDown={(e) => {
                  if ((e.ctrlKey || e.metaKey) && e.key === "Enter") {
                    e.preventDefault();
                    void handleRunPlayground();
                  }
                }}
              />
            </Field>

            <button
              type="button"
              className="btn btn-primary"
              disabled={running || !selectedSpec}
              onClick={handleRunPlayground}
            >
              <BusyIcon busy={running} icon={Play} /> Executar action
            </button>
          </div>

          <div className="stack">
            <Field label="Saída estruturada">
              <CodeBlock
                code={resultText || "// execute uma action para inspecionar o retorno JSON"}
                minHeight={180}
                maxHeight={300}
                onCopy={resultText ? () => void copyToClipboard(resultText, onNotify, "Saída") : undefined}
              />
            </Field>
            {selectedSpec && (
              <Field label="Input JSON Schema (exposto ao MCP & OpenAPI)">
                <CodeBlock
                  code={JSON.stringify(selectedSpec.input_schema, null, 2)}
                  maxHeight={200}
                  onCopy={() =>
                    void copyToClipboard(JSON.stringify(selectedSpec.input_schema, null, 2), onNotify, "JSON Schema")
                  }
                />
              </Field>
            )}
          </div>
        </div>
      </Panel>
      </div>
    </>
  );
};
