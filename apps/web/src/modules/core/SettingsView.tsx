import React, { useState } from "react";
import {
  CircleCheck,
  Copy,
  Cpu,
  Download,
  Folder,
  HardDrive,
  MemoryStick,
  Monitor,
  Palette,
  RefreshCw,
  Server,
  Wrench,
} from "lucide-react";
import { getPublicServerUrl, runAction } from "../../lib/client";
import type { HardwareProfile, NotifyFn, ToolStatus, WorkspaceConfig } from "../../lib/types";
import { copyToClipboard } from "../../lib/ui";
import { pageMeta } from "../../app/navigation";
import { THEMES, type ThemeId } from "../../app/themes";
import {
  Badge,
  BusyIcon,
  ChoiceCards,
  MoreButton,
  PageHeader,
  Panel,
  StatTile,
  useContextMenu,
  type MenuItems,
} from "../../components";
import "./core.css";

interface SettingsViewProps {
  hardware: HardwareProfile | null;
  tools: ToolStatus[];
  workspace: WorkspaceConfig | null;
  theme: ThemeId;
  onThemeChange: (theme: ThemeId) => void;
  onRefreshTools: () => Promise<void>;
  onNotify: NotifyFn;
}

const ACCEL_LABEL: Record<string, string> = {
  cuda_nvenc: "NVIDIA CUDA / NVENC",
  amf_directml: "AMD AMF / DirectML",
  qsv_openvino: "Intel QuickSync / OpenVINO",
  metal: "Apple Silicon Metal",
  cpu_only: "CPU Nativa Multithread",
};

export const SettingsView: React.FC<SettingsViewProps> = ({
  hardware,
  tools,
  workspace,
  theme,
  onThemeChange,
  onRefreshTools,
  onNotify,
}) => {
  const menu = useContextMenu();
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

  const installLabel = (tool: ToolStatus) =>
    !tool.installed ? "Instalar melhor versão" : tool.update_available ? "Atualizar" : "Reinstalar / atualizar";

  const toolMenu = (tool: ToolStatus): MenuItems => [
    { type: "label", label: `${tool.display_name} · ${tool.owner_module}` },
    {
      label: installLabel(tool),
      icon: Download,
      disabled: busyTool === tool.id || updatingAll,
      onSelect: () => void handleInstallTool(tool.id, tool.display_name),
    },
    { type: "separator" },
    tool.binary_path && {
      label: "Copiar caminho do binário",
      icon: Copy,
      onSelect: () => void copyToClipboard(tool.binary_path ?? "", onNotify, "Caminho do binário"),
    },
    tool.installed_version && {
      label: "Copiar versão instalada",
      icon: Copy,
      onSelect: () => void copyToClipboard(tool.installed_version ?? "", onNotify, "Versão"),
    },
    { label: "Copiar build recomendada", icon: Copy, onSelect: () => void copyToClipboard(tool.recommended_build, onNotify, "Build") },
  ];

  const paths: Array<{ label: string; value: string; icon: typeof Folder }> = workspace
    ? [
        { label: "Downloads (yt-dlp)", value: workspace.downloads_dir, icon: Download },
        { label: "Importações (modo cópia)", value: workspace.imports_dir, icon: Folder },
        { label: "Legendas & timelines geradas", value: workspace.exports_dir, icon: Folder },
        { label: "Modelos de IA local", value: workspace.models_dir, icon: Cpu },
        { label: "Binários gerenciados", value: workspace.bin_dir, icon: Wrench },
        { label: "Endpoint local de IA (MCP / REST / OpenAPI)", value: getPublicServerUrl(), icon: Server },
      ]
    : [];

  return (
    <>
      <PageHeader {...pageMeta("settings")} />

      <Panel title="Aparência" icon={Palette} description="Esquemas de cor inspirados em terminais. A escolha fica salva neste computador.">
        <ChoiceCards
          ariaLabel="Esquema de cores"
          className="theme-picker"
          columns={4}
          value={theme}
          onChange={onThemeChange}
          options={THEMES.map((t) => ({
            id: t.id,
            title: t.label,
            description: t.description,
            preview: (
              <span className="theme-swatch" style={{ background: t.swatch[0] }} aria-hidden>
                <span style={{ background: t.swatch[1] }} />
                <span style={{ background: t.swatch[2] }} />
              </span>
            ),
          }))}
        />
      </Panel>

      {hardware && (
        <Panel
          title="Hardware detectado"
          icon={Cpu}
          actions={
            <Badge tone="accent">
              {hardware.platform_tag} · {ACCEL_LABEL[hardware.recommended_accel] ?? hardware.recommended_accel}
              {hardware.cuda_version ? ` (CUDA ${hardware.cuda_version})` : ""}
            </Badge>
          }
        >
          <div className="grid-4">
            <StatTile
              icon={Monitor}
              label="Sistema"
              value={hardware.os_version}
              detail={`${hardware.arch} (${hardware.platform_tag})`}
            />
            <StatTile
              icon={Cpu}
              label="Processador"
              value={hardware.cpu_brand}
              detail={`${hardware.physical_cores} núcleos · ${hardware.logical_cores} threads`}
            />
            <StatTile
              icon={MemoryStick}
              label="Memória RAM"
              value={`${(hardware.total_memory_mb / 1024).toFixed(1)} GB`}
              detail={`${(hardware.available_memory_mb / 1024).toFixed(1)} GB livre agora`}
            />
            <StatTile
              icon={HardDrive}
              label="GPU / aceleração"
              value={hardware.gpus.length > 0 ? hardware.gpus[0].name : "CPU integrada"}
              detail={`${ACCEL_LABEL[hardware.recommended_accel] ?? hardware.recommended_accel}${
                hardware.cuda_version ? ` · CUDA ${hardware.cuda_version}` : ""
              }`}
            />
          </div>
          <p className="secondary text-sm">{hardware.recommendation_summary}</p>
        </Panel>
      )}

      <Panel
        title="Ferramentas gerenciadas"
        icon={Wrench}
        flush
        actions={
          <>
            <button type="button" className="btn btn-sm" disabled={checking} onClick={handleCheckUpdates}>
              <BusyIcon busy={checking} icon={RefreshCw} size={13} />
              Verificar atualizações
            </button>
            <button type="button" className="btn btn-primary btn-sm" disabled={updatingAll} onClick={handleUpdateAll}>
              <BusyIcon busy={updatingAll} icon={Download} size={13} />
              Instalar / atualizar tudo
            </button>
          </>
        }
      >
        <div className="table-wrap">
          <table className="table tools-table">
            <thead>
              <tr>
                <th>ferramenta</th>
                <th className="col-fit">status</th>
                <th className="col-fit c-opt">módulo</th>
                <th>build ideal p/ seu pc</th>
                <th className="col-fit" aria-label="Ações" />
              </tr>
            </thead>
            <tbody>
              {tools.map((tool) => {
                const isBusy = busyTool === tool.id || updatingAll;
                return (
                  <tr key={tool.id} onContextMenu={(e) => menu.open(e, toolMenu(tool))}>
                    <td className="tool-cell">
                      <div className="row gap-xs">
                        <span className="strong">{tool.display_name}</span>
                        {tool.managed_by_aicut && <Badge tone="info">gerenciado</Badge>}
                      </div>
                      <div className="secondary text-xs">{tool.description}</div>
                      {tool.binary_path && <div className="muted text-xs break-all">{tool.binary_path}</div>}
                    </td>
                    <td className="col-fit">
                      {tool.installed ? (
                        <Badge tone={tool.update_available ? "warning" : "success"} icon={CircleCheck}>
                          {tool.installed_version ? `v${tool.installed_version}` : "instalado"}
                          {tool.update_available && tool.latest_version ? ` → ${tool.latest_version}` : ""}
                        </Badge>
                      ) : (
                        <Badge tone="danger">não instalado</Badge>
                      )}
                    </td>
                    <td className="col-fit muted c-opt">{tool.owner_module}</td>
                    <td className="tool-build">
                      <div className="text-sm">{tool.recommended_build}</div>
                      <div className="muted text-xs">{tool.hardware_note}</div>
                    </td>
                    <td className="col-fit">
                      <span className="row gap-xs">
                        <button
                          type="button"
                          className={`btn btn-sm ${!tool.installed || tool.update_available ? "btn-primary" : ""}`}
                          disabled={isBusy}
                          onClick={() => handleInstallTool(tool.id, tool.display_name)}
                        >
                          <BusyIcon busy={isBusy} icon={Download} size={13} />
                          {installLabel(tool)}
                        </button>
                        <MoreButton items={() => toolMenu(tool)} />
                      </span>
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </Panel>

      {workspace && (
        <Panel title="Workspace & servidor local" icon={Folder} flush>
          <div className="path-list">
            {paths.map((p) => (
              <div
                key={p.label}
                className="path-row hover-host"
                onContextMenu={(e) =>
                  menu.open(e, [{ label: "Copiar caminho", icon: Copy, onSelect: () => void copyToClipboard(p.value, onNotify, p.label) }])
                }
              >
                <p.icon size={14} className="text-accent shrink-0" aria-hidden />
                <span className="path-label">{p.label}</span>
                <code className="path-value" title={p.value}>
                  {p.value}
                </code>
                <span className="row-actions">
                  <button
                    type="button"
                    className="btn btn-ghost btn-icon btn-xs"
                    aria-label={`Copiar ${p.label}`}
                    title="Copiar"
                    onClick={() => void copyToClipboard(p.value, onNotify, p.label)}
                  >
                    <Copy aria-hidden />
                  </button>
                </span>
              </div>
            ))}
          </div>
        </Panel>
      )}
    </>
  );
};
