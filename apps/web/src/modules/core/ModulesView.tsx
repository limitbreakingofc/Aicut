import React, { useState } from "react";
import { Blocks, CheckCircle2, Play, Terminal } from "lucide-react";
import { runAction } from "../../lib/client";
import type { ActionSpec, ModuleInfo } from "../../lib/types";

interface ModulesViewProps {
  modules: ModuleInfo[];
  actions: ActionSpec[];
  onRefresh: () => void;
  onNotify: (level: "info" | "success" | "warning" | "error", title: string, message: string) => void;
}

export const ModulesView: React.FC<ModulesViewProps> = ({
  modules,
  actions,
  onRefresh,
  onNotify,
}) => {
  const [selectedActionId, setSelectedActionId] = useState<string>(actions[0]?.id ?? "system.info");
  const [jsonInput, setJsonInput] = useState<string>("{}");
  const [resultText, setResultText] = useState<string>("");
  const [running, setRunning] = useState(false);

  const selectedSpec = actions.find((a) => a.id === selectedActionId) ?? actions[0] ?? null;

  async function toggleModule(mod: ModuleInfo) {
    try {
      await runAction("system.set_module_enabled", {
        module_id: mod.manifest.id,
        enabled: !mod.enabled,
      });
      onRefresh();
      onNotify(
        "info",
        mod.manifest.name,
        !mod.enabled ? "Módulo ativado para UI e IAs." : "Módulo desativado.",
      );
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

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 18 }}>
      <div className="page-header">
        <div>
          <h1 className="page-title">Módulos & Action Playground</h1>
          <p className="page-subtitle">
            Arquitetura plugável: cada módulo registra suas próprias Actions, ferramentas externas e JSON Schemas compartilhados entre você e as IAs.
          </p>
        </div>
      </div>

      {/* Installed Modules Grid */}
      <div className="grid-2">
        {modules.map((mod) => (
          <div key={mod.manifest.id} className="panel" style={{ display: "flex", flexDirection: "column", gap: 10 }}>
            <div style={{ display: "flex", alignItems: "flex-start", justifyContent: "space-between", gap: 10 }}>
              <div>
                <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
                  <Blocks size={16} color="var(--accent)" />
                  <strong>{mod.manifest.name}</strong>
                  <span className="pill">v{mod.manifest.version}</span>
                  {mod.manifest.core && <span className="pill accent">Core</span>}
                </div>
                <p style={{ margin: "6px 0 0", fontSize: 13, color: "var(--text-secondary)" }}>
                  {mod.manifest.description}
                </p>
              </div>
              {!mod.manifest.core && (
                <button
                  type="button"
                  className={`btn btn-sm ${mod.enabled ? "" : "btn-primary"}`}
                  onClick={() => toggleModule(mod)}
                >
                  {mod.enabled ? "Desativar" : "Ativar"}
                </button>
              )}
            </div>

            <div style={{ display: "flex", flexWrap: "wrap", gap: 6, alignItems: "center" }}>
              <span className={`pill ${mod.health.healthy ? "success" : "warning"}`}>
                <CheckCircle2 size={11} /> {mod.health.message}
              </span>
              <span className="pill">{mod.action_ids.length} actions</span>
              {mod.tool_ids.length > 0 && (
                <span className="pill">Ferramentas: {mod.tool_ids.join(", ")}</span>
              )}
            </div>
          </div>
        ))}
      </div>

      {/* Dynamic Action Playground */}
      <div className="panel" style={{ display: "flex", flexDirection: "column", gap: 14 }}>
        <div className="panel-header" style={{ marginBottom: 0 }}>
          <h2 className="panel-title">
            <Terminal size={16} /> Action Playground (Testar qualquer Action / Tool de IA)
          </h2>
        </div>

        <div className="grid-2">
          <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
            <div className="field">
              <label>Action registrada</label>
              <select
                className="select"
                value={selectedSpec?.id ?? ""}
                onChange={(e) => {
                  setSelectedActionId(e.target.value);
                  setJsonInput("{}");
                }}
              >
                {actions.map((a) => (
                  <option key={a.id} value={a.id}>
                    {a.id} — {a.title}
                  </option>
                ))}
              </select>
            </div>

            {selectedSpec && (
              <div style={{ fontSize: 12.5, color: "var(--text-secondary)" }}>
                {selectedSpec.description}
              </div>
            )}

            <div className="field">
              <label>Parâmetros (JSON)</label>
              <textarea
                className="textarea"
                rows={7}
                value={jsonInput}
                onChange={(e) => setJsonInput(e.target.value)}
              />
            </div>

            <button
              type="button"
              className="btn btn-primary"
              disabled={running || !selectedSpec}
              onClick={handleRunPlayground}
            >
              <Play size={14} /> Executar Action
            </button>
          </div>

          <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
            <div className="field">
              <label>Saída Estruturada</label>
              <pre className="code-block" style={{ minHeight: 170, maxHeight: 260 }}>
                {resultText || "// Execute uma Action para inspecionar o retorno JSON"}
              </pre>
            </div>
            {selectedSpec && (
              <div className="field">
                <label>Input JSON Schema (exposto automaticamente ao MCP & OpenAPI)</label>
                <pre className="code-block" style={{ maxHeight: 160 }}>
                  {JSON.stringify(selectedSpec.input_schema, null, 2)}
                </pre>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};
