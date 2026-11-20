import { Activity, Cpu, Film, Keyboard, TriangleAlert } from "lucide-react";
import { Spinner, useContextMenu } from "../components";
import type { HardwareProfile, RouteId } from "../lib/types";

interface StatusBarProps {
  connected: boolean;
  serverUrl: string;
  activeJobs: number;
  mediaCount: number;
  toolsNeedAttention: number;
  hardware: HardwareProfile | null;
  onNavigate: (route: RouteId) => void;
  onOpenShortcuts: () => void;
  onOpenPalette: () => void;
  onClearFinished: () => void;
  onRefreshTools: () => void;
  onCopy: (text: string, label: string) => void;
}

/** tmux/VS Code-style status line: every segment is clickable and has a context menu. */
export function StatusBar({
  connected,
  serverUrl,
  activeJobs,
  mediaCount,
  toolsNeedAttention,
  hardware,
  onNavigate,
  onOpenShortcuts,
  onOpenPalette,
  onClearFinished,
  onRefreshTools,
  onCopy,
}: StatusBarProps) {
  const menu = useContextMenu();
  const host = serverUrl.replace(/^https?:\/\//, "");
  const gpu = hardware?.gpus[0];
  const hwLabel = hardware
    ? `${gpu ? gpu.name : hardware.cpu_brand}${hardware.cuda_version ? ` · CUDA ${hardware.cuda_version}` : ""}`
    : null;

  const serverMenu = [
    { type: "label" as const, label: `Core local · ${host}` },
    { label: "Abrir Conectar IA", onSelect: () => onNavigate("ai-connect") },
    { type: "separator" as const },
    { label: "Copiar URL do servidor", onSelect: () => onCopy(serverUrl, "URL do servidor") },
    { label: "Copiar endpoint MCP", onSelect: () => onCopy(`${serverUrl}/mcp`, "Endpoint MCP") },
    { label: "Copiar URL OpenAPI", onSelect: () => onCopy(`${serverUrl}/openapi.json`, "URL OpenAPI") },
  ];

  return (
    <footer className="statusbar">
      <button
        type="button"
        className="status-item status-mode"
        data-state={connected ? "online" : "offline"}
        title={connected ? "Core local conectado" : "Tentando reconectar ao core local"}
        onClick={() => onNavigate("ai-connect")}
        onContextMenu={(e) => menu.open(e, serverMenu)}
      >
        {connected ? "● ONLINE" : "○ OFFLINE"}
      </button>

      <button
        type="button"
        className="status-item status-optional"
        title="Servidor MCP / REST local"
        onClick={() => onNavigate("ai-connect")}
        onContextMenu={(e) => menu.open(e, serverMenu)}
      >
        mcp {host}
      </button>

      <button
        type="button"
        className="status-item"
        data-tone={activeJobs > 0 ? "accent" : undefined}
        title="Fila de tarefas"
        onClick={() => onNavigate("jobs")}
        onContextMenu={(e) =>
          menu.open(e, [
            { type: "label", label: "Fila de tarefas" },
            { label: "Abrir fila", icon: Activity, shortcut: "Alt+5", onSelect: () => onNavigate("jobs") },
            { label: "Limpar finalizadas", onSelect: onClearFinished },
          ])
        }
      >
        {activeJobs > 0 ? <Spinner size={12} /> : <Activity aria-hidden />}
        {activeJobs > 0 ? `${activeJobs} em andamento` : "fila ociosa"}
      </button>

      <button type="button" className="status-item" title="Biblioteca" onClick={() => onNavigate("library")}>
        <Film aria-hidden />
        {mediaCount} {mediaCount === 1 ? "mídia" : "mídias"}
      </button>

      {toolsNeedAttention > 0 && (
        <button
          type="button"
          className="status-item"
          data-tone="warning"
          title="Ferramentas para instalar ou atualizar"
          onClick={() => onNavigate("settings")}
          onContextMenu={(e) =>
            menu.open(e, [
              { label: "Abrir ferramentas", onSelect: () => onNavigate("settings") },
              { label: "Recarregar status", onSelect: onRefreshTools },
            ])
          }
        >
          <TriangleAlert aria-hidden />
          {toolsNeedAttention} {toolsNeedAttention === 1 ? "ferramenta" : "ferramentas"}
        </button>
      )}

      <div className="status-spacer" />

      {hwLabel && (
        <button
          type="button"
          className="status-item status-right status-optional"
          title={hardware?.recommendation_summary}
          onClick={() => onNavigate("settings")}
          onContextMenu={(e) =>
            menu.open(e, [
              { label: "Ver hardware e ferramentas", icon: Cpu, onSelect: () => onNavigate("settings") },
              { label: "Copiar resumo do hardware", onSelect: () => onCopy(hwLabel, "Resumo do hardware") },
            ])
          }
        >
          <Cpu aria-hidden />
          {hwLabel}
        </button>
      )}

      <button type="button" className="status-item status-right" title="Atalhos de teclado" onClick={onOpenShortcuts}>
        <Keyboard aria-hidden />? atalhos
      </button>
      <button type="button" className="status-item status-right" title="Paleta de comandos" onClick={onOpenPalette}>
        ctrl+k
      </button>
    </footer>
  );
}
