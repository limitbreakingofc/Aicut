import { Activity, Cpu, Keyboard, TriangleAlert } from "lucide-react";
import { Spinner, useContextMenu } from "../components";
import type { HardwareProfile } from "../lib/types";
import type { SettingsTabId } from "./navigation";

interface StatusBarProps {
  connected: boolean;
  activeJobs: number;
  toolsNeedAttention: number;
  hardware: HardwareProfile | null;
  onOpenSettingsTab: (tab: SettingsTabId) => void;
  onOpenShortcuts: () => void;
  onClearFinished: () => void;
  onRefreshTools: () => void;
  onCopy: (text: string, label: string) => void;
}

/** Slim, non-redundant status bar: live core status, active jobs, tool alerts, GPU and shortcuts hint. */
export function StatusBar({
  connected,
  activeJobs,
  toolsNeedAttention,
  hardware,
  onOpenSettingsTab,
  onOpenShortcuts,
  onClearFinished,
  onRefreshTools,
  onCopy,
}: StatusBarProps) {
  const menu = useContextMenu();
  const gpu = hardware?.gpus[0];
  const hwLabel = hardware
    ? `${gpu ? gpu.name : hardware.cpu_brand}${hardware.cuda_version ? ` · CUDA ${hardware.cuda_version}` : ""}`
    : null;

  return (
    <footer className="statusbar">
      <button
        type="button"
        className="status-item status-mode"
        data-state={connected ? "online" : "offline"}
        title={connected ? "Core local conectado — clique para ver Integrações MCP" : "Reconectando ao core local"}
        onClick={() => onOpenSettingsTab("integrations")}
      >
        {connected ? "● ONLINE" : "○ OFFLINE"}
      </button>

      <button
        type="button"
        className="status-item"
        data-tone={activeJobs > 0 ? "accent" : undefined}
        title="Ver fila de tarefas em Configurações → Tarefas"
        onClick={() => onOpenSettingsTab("jobs")}
        onContextMenu={(e) =>
          menu.open(e, [
            { type: "label", label: "Tarefas" },
            { label: "Abrir fila de tarefas", icon: Activity, onSelect: () => onOpenSettingsTab("jobs") },
            { label: "Limpar finalizadas", onSelect: onClearFinished },
          ])
        }
      >
        {activeJobs > 0 ? <Spinner size={12} /> : <Activity aria-hidden />}
        {activeJobs > 0 ? `${activeJobs} em andamento` : "fila ociosa"}
      </button>

      {toolsNeedAttention > 0 && (
        <button
          type="button"
          className="status-item"
          data-tone="warning"
          title="Ferramentas pendentes em Configurações → Ferramentas"
          onClick={() => onOpenSettingsTab("tools")}
          onContextMenu={(e) =>
            menu.open(e, [
              { label: "Abrir Ferramentas", onSelect: () => onOpenSettingsTab("tools") },
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
          onClick={() => onOpenSettingsTab("system")}
          onContextMenu={(e) =>
            menu.open(e, [
              { label: "Ver Sistema & Hardware", icon: Cpu, onSelect: () => onOpenSettingsTab("system") },
              { label: "Copiar hardware", onSelect: () => onCopy(hwLabel, "Hardware") },
            ])
          }
        >
          <Cpu aria-hidden />
          {hwLabel}
        </button>
      )}

      <button type="button" className="status-item status-right" title="Atalhos de teclado (?)" onClick={onOpenShortcuts}>
        <Keyboard aria-hidden />?
      </button>
    </footer>
  );
}
