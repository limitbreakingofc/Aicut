import React, { useState } from "react";
import {
  CheckCircle2,
  Cpu,
  Download,
  Folder,
  HardDrive,
  Loader2,
  Moon,
  RefreshCw,
  Sun,
  Wrench,
} from "lucide-react";
import { getPublicServerUrl, runAction } from "../../lib/client";
import type { HardwareProfile, ToolStatus, WorkspaceConfig } from "../../lib/types";

interface SettingsViewProps {
  hardware: HardwareProfile | null;
  tools: ToolStatus[];
  workspace: WorkspaceConfig | null;
  theme: "dark" | "light";
  onToggleTheme: () => void;
  onRefreshTools: () => Promise<void>;
  onNotify: (level: "info" | "success" | "warning" | "error", title: string, message: string) => void;
}

export const SettingsView: React.FC<SettingsViewProps> = ({
  hardware,
  tools,
  workspace,
  theme,
  onToggleTheme,
  onRefreshTools,
  onNotify,
}) => {
  const [checking, setChecking] = useState(false);
  const [updatingAll, setUpdatingAll] = useState(false);
  const [busyTool, setBusyTool] = useState<string | null>(null);

  async function handleCheckUpdates() {
    setChecking(true);
    try {
      const res = await runAction("tools.check_updates");
      await onRefreshTools();
      onNotify("info", "Verificação concluída", res.summary);
    } catch (e) {
      onNotify("error", "Erro ao verificar atualizações", (e as Error).message);
    } finally {
      setChecking(false);
    }
  }

  async function handleInstallTool(toolId: string, label: string) {
    setBusyTool(toolId);
    try {
      const res = await runAction("tools.install", { tool_id: toolId });
      await onRefreshTools();
      onNotify("success", `${label} pronto!`, res.summary);
    } catch (e) {
      onNotify("error", `Falha ao instalar/atualizar ${label}`, (e as Error).message);
    } finally {
      setBusyTool(null);
    }
  }

  async function handleUpdateAll() {
    setUpdatingAll(true);
    try {
      await runAction("tools.update_all");
      await onRefreshTools();
      onNotify("success", "Ferramentas atualizadas", "As melhores versões para seu hardware foram instaladas.");
    } catch (e) {
      onNotify("error", "Erro ao atualizar ferramentas", (e as Error).message);
    } finally {
      setUpdatingAll(false);
    }
  }

  const accelLabel: Record<string, string> = {
    cuda_nvenc: "NVIDIA CUDA / NVENC",
    amf_directml: "AMD AMF / DirectML",
    qsv_openvino: "Intel QuickSync / OpenVINO",
    metal: "Apple Silicon Metal",
    cpu_only: "CPU Nativa Multithread",
  };

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 18 }}>
      <div className="page-header">
        <div>
          <h1 className="page-title">Configurações & Ferramentas</h1>
          <p className="page-subtitle">
            O Aicut detecta automaticamente o hardware do seu PC e instala/atualiza as melhores versões do yt-dlp, FFmpeg e futuras ferramentas dos módulos.
          </p>
        </div>
        <button type="button" className="btn" onClick={onToggleTheme}>
          {theme === "dark" ? <Sun size={15} /> : <Moon size={15} />}
          {theme === "dark" ? "Tema Claro" : "Tema Escuro"}
        </button>
      </div>

      {/* Hardware Profile Card */}
      {hardware && (
        <div className="panel" style={{ display: "flex", flexDirection: "column", gap: 14 }}>
          <div className="panel-header" style={{ marginBottom: 0 }}>
            <h2 className="panel-title">
              <Cpu size={16} color="var(--accent)" /> Hardware Detectado Automaticamente
            </h2>
            <span className="pill accent">
              {hardware.platform_tag} · {accelLabel[hardware.recommended_accel] ?? hardware.recommended_accel}
            </span>
          </div>

          <div className="grid-4">
            <div style={{ padding: 12, borderRadius: "var(--radius-md)", background: "var(--bg-elevated)" }}>
              <div style={{ fontSize: 11.5, color: "var(--text-muted)" }}>Sistema & Arquitetura</div>
              <div style={{ fontWeight: 600, marginTop: 4 }}>{hardware.os_version}</div>
              <div style={{ fontSize: 12, color: "var(--text-secondary)", fontFamily: "var(--font-mono)" }}>
                {hardware.arch} ({hardware.platform_tag})
              </div>
            </div>

            <div style={{ padding: 12, borderRadius: "var(--radius-md)", background: "var(--bg-elevated)" }}>
              <div style={{ fontSize: 11.5, color: "var(--text-muted)" }}>Processador (CPU)</div>
              <div style={{ fontWeight: 600, marginTop: 4 }}>{hardware.cpu_brand}</div>
              <div style={{ fontSize: 12, color: "var(--text-secondary)" }}>
                {hardware.physical_cores} núcleos físicos · {hardware.logical_cores} threads
              </div>
            </div>

            <div style={{ padding: 12, borderRadius: "var(--radius-md)", background: "var(--bg-elevated)" }}>
              <div style={{ fontSize: 11.5, color: "var(--text-muted)" }}>Memória RAM</div>
              <div style={{ fontWeight: 600, marginTop: 4 }}>
                {(hardware.total_memory_mb / 1024).toFixed(1)} GB
              </div>
              <div style={{ fontSize: 12, color: "var(--text-secondary)" }}>
                {(hardware.available_memory_mb / 1024).toFixed(1)} GB livre agora
              </div>
            </div>

            <div style={{ padding: 12, borderRadius: "var(--radius-md)", background: "var(--bg-elevated)" }}>
              <div style={{ fontSize: 11.5, color: "var(--text-muted)" }}>GPU / Aceleração</div>
              <div style={{ fontWeight: 600, marginTop: 4 }}>
                {hardware.gpus.length > 0 ? hardware.gpus[0].name : "CPU Integrada"}
              </div>
              <div style={{ fontSize: 12, color: "var(--text-secondary)" }}>
                {accelLabel[hardware.recommended_accel] ?? hardware.recommended_accel}
              </div>
            </div>
          </div>

          <div style={{ fontSize: 12.5, color: "var(--text-secondary)" }}>
            {hardware.recommendation_summary}
          </div>
        </div>
      )}

      {/* Extensible Tools & Auto-Updater */}
      <div className="panel" style={{ display: "flex", flexDirection: "column", gap: 14 }}>
        <div className="panel-header" style={{ marginBottom: 0 }}>
          <h2 className="panel-title">
            <Wrench size={16} color="var(--accent)" /> Gerenciador de Ferramentas (yt-dlp, FFmpeg e Futuros Módulos)
          </h2>
          <div style={{ display: "flex", gap: 8 }}>
            <button type="button" className="btn btn-sm" disabled={checking} onClick={handleCheckUpdates}>
              {checking ? <Loader2 size={14} /> : <RefreshCw size={14} />}
              Verificar Atualizações
            </button>
            <button
              type="button"
              className="btn btn-primary btn-sm"
              disabled={updatingAll}
              onClick={handleUpdateAll}
            >
              {updatingAll ? <Loader2 size={14} /> : <Download size={14} />}
              Instalar / Atualizar Tudo
            </button>
          </div>
        </div>

        <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
          {tools.map((tool) => {
            const isBusy = busyTool === tool.id || updatingAll;
            return (
              <div
                key={tool.id}
                style={{
                  display: "flex",
                  alignItems: "center",
                  justifyContent: "space-between",
                  gap: 16,
                  padding: "14px 16px",
                  borderRadius: "var(--radius-md)",
                  background: "var(--bg-elevated)",
                  border: "1px solid var(--border-subtle)",
                }}
              >
                <div style={{ display: "flex", flexDirection: "column", gap: 5, minWidth: 0 }}>
                  <div style={{ display: "flex", alignItems: "center", gap: 8, flexWrap: "wrap" }}>
                    <strong style={{ fontSize: 14.5 }}>{tool.display_name}</strong>
                    <span className="pill">módulo: {tool.owner_module}</span>
                    {tool.installed ? (
                      <span className={`pill ${tool.update_available ? "warning" : "success"}`}>
                        <CheckCircle2 size={11} />
                        {tool.installed_version ? `v${tool.installed_version}` : "Instalado"}
                        {tool.update_available && tool.latest_version ? ` → ${tool.latest_version} disponível` : ""}
                      </span>
                    ) : (
                      <span className="pill danger">Não instalado</span>
                    )}
                    {tool.managed_by_aicut && <span className="pill accent">Gerenciado pelo Aicut</span>}
                  </div>
                  <div style={{ fontSize: 12.5, color: "var(--text-secondary)" }}>{tool.description}</div>
                  <div style={{ fontSize: 12, color: "var(--text-muted)" }}>
                    <strong>Build ideal p/ seu PC:</strong> {tool.recommended_build} — {tool.hardware_note}
                  </div>
                  {tool.binary_path && (
                    <div style={{ fontSize: 11.5, fontFamily: "var(--font-mono)", color: "var(--text-muted)" }}>
                      {tool.binary_path}
                    </div>
                  )}
                </div>

                <button
                  type="button"
                  className={`btn btn-sm ${!tool.installed || tool.update_available ? "btn-primary" : ""}`}
                  disabled={isBusy}
                  onClick={() => handleInstallTool(tool.id, tool.display_name)}
                >
                  {isBusy ? <Loader2 size={14} /> : <Download size={14} />}
                  {!tool.installed
                    ? "Instalar Melhor Versão"
                    : tool.update_available
                      ? "Atualizar"
                      : "Reinstalar / Atualizar"}
                </button>
              </div>
            );
          })}
        </div>
      </div>

      {/* Workspace Paths & Server */}
      {workspace && (
        <div className="panel" style={{ display: "flex", flexDirection: "column", gap: 12 }}>
          <h2 className="panel-title">
            <Folder size={16} color="var(--accent)" /> Workspace & Servidor Local
          </h2>
          <div className="grid-2" style={{ fontSize: 12.5 }}>
            <div style={{ display: "flex", flexDirection: "column", gap: 4 }}>
              <span style={{ color: "var(--text-muted)" }}>Pasta de Downloads (yt-dlp)</span>
              <code className="code-block" style={{ padding: "7px 10px" }}>{workspace.downloads_dir}</code>
            </div>
            <div style={{ display: "flex", flexDirection: "column", gap: 4 }}>
              <span style={{ color: "var(--text-muted)" }}>Pasta de Importações (Modo Cópia)</span>
              <code className="code-block" style={{ padding: "7px 10px" }}>{workspace.imports_dir}</code>
            </div>
            <div style={{ display: "flex", flexDirection: "column", gap: 4 }}>
              <span style={{ color: "var(--text-muted)" }}>Legendas & Timelines Geradas (Exports)</span>
              <code className="code-block" style={{ padding: "7px 10px" }}>{workspace.exports_dir}</code>
            </div>
            <div style={{ display: "flex", flexDirection: "column", gap: 4 }}>
              <span style={{ color: "var(--text-muted)" }}>Modelos de IA Local (Whisper GGML)</span>
              <code className="code-block" style={{ padding: "7px 10px" }}>{workspace.models_dir}</code>
            </div>
            <div style={{ display: "flex", flexDirection: "column", gap: 4 }}>
              <span style={{ color: "var(--text-muted)" }}>Binários Gerenciados (yt-dlp / FFmpeg / whisper.cpp)</span>
              <code className="code-block" style={{ padding: "7px 10px" }}>{workspace.bin_dir}</code>
            </div>
            <div style={{ display: "flex", flexDirection: "column", gap: 4 }}>
              <span style={{ color: "var(--text-muted)" }}>Endpoint Local de IA (MCP / REST / OpenAPI)</span>
              <code className="code-block" style={{ padding: "7px 10px" }}>
                <HardDrive size={12} style={{ verticalAlign: "middle", marginRight: 6 }} />
                {getPublicServerUrl()}
              </code>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
