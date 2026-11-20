import React, { useCallback, useEffect, useState } from "react";
import {
  Activity,
  Award,
  Blocks,
  Bot,
  CircleCheck,
  Cloud,
  Copy,
  Cpu,
  Download,
  Folder,
  HardDrive,
  KeyRound,
  MemoryStick,
  Monitor,
  Palette,
  Play,
  RefreshCw,
  Server,
  Sparkles,
  Trash2,
  Wrench,
} from "lucide-react";
import { AicutActionError, getPublicServerUrl, runAction } from "../../lib/client";
import { formatBytes } from "../../lib/format";
import type {
  ActionSpec,
  ActivityEntry,
  ClipsConfig,
  HardwareProfile,
  JobRecord,
  ModuleInfo,
  NotifyFn,
  RouteId,
  ToolStatus,
  VisionModelStatus,
  WorkspaceConfig,
} from "../../lib/types";
import { copyToClipboard } from "../../lib/ui";
import type { SettingsTabId } from "../../app/navigation";
import { THEMES, type ThemeId } from "../../app/themes";
import {
  Badge,
  BusyIcon,
  Callout,
  ChoiceCards,
  Field,
  MoreButton,
  Panel,
  StatTile,
  Tabs,
  useContextMenu,
  type MenuItems,
} from "../../components";
import { JobsView } from "./JobsView";
import { ModulesView } from "./ModulesView";
import { AiConnectView } from "./AiConnectView";
import "./core.css";

interface SettingsViewProps {
  activeTab: SettingsTabId;
  onTabChange: (tab: SettingsTabId) => void;
  hardware: HardwareProfile | null;
  tools: ToolStatus[];
  workspace: WorkspaceConfig | null;
  theme: ThemeId;
  jobs: JobRecord[];
  modules: ModuleInfo[];
  actions: ActionSpec[];
  activity: ActivityEntry[];
  connected: boolean;
  onThemeChange: (theme: ThemeId) => void;
  onRefreshTools: () => Promise<void>;
  onRefreshJobs: () => void;
  onClearFinishedJobs: () => Promise<void>;
  onRefreshModules: () => void;
  onNavigate: (route: RouteId) => void;
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
  activeTab,
  onTabChange,
  hardware,
  tools,
  workspace,
  theme,
  jobs,
  modules,
  actions,
  activity,
  connected,
  onThemeChange,
  onRefreshTools,
  onRefreshJobs,
  onClearFinishedJobs,
  onRefreshModules,
  onNavigate,
  onNotify,
}) => {
  const menu = useContextMenu();

  // Tools state
  const [checking, setChecking] = useState(false);
  const [updatingAll, setUpdatingAll] = useState(false);
  const [busyTool, setBusyTool] = useState<string | null>(null);

  // AI & Keys state (single source of truth for Cortes Shorts, Enquadramento & Motion Studio)
  const [aiConfig, setAiConfig] = useState<ClipsConfig | null>(null);
  const [provider, setProvider] = useState<"gemini" | "local_free">("gemini");
  const [geminiKeyInput, setGeminiKeyInput] = useState("");
  const [geminiDirectorModel, setGeminiDirectorModel] = useState("auto");
  const [savingAi, setSavingAi] = useState(false);
  const [visionModels, setVisionModels] = useState<VisionModelStatus[]>([]);
  const [totalVisionBytes, setTotalVisionBytes] = useState(0);
  const [selectedLocalModel, setSelectedLocalModel] = useState("");
  const [installingLlama, setInstallingLlama] = useState(false);
  const [downloadingModel, setDownloadingModel] = useState<string | null>(null);
  const [uninstallingModel, setUninstallingModel] = useState<string | null>(null);

  const llamaTool = tools.find((t) => t.id === "llama-server");
  const primaryGpu = hardware?.gpus[0] ?? null;

  const loadAiConfig = useCallback(async () => {
    try {
      const [cfgRes, modelsRes] = await Promise.all([
        runAction<ClipsConfig>("clips.config_get"),
        runAction<{
          models: VisionModelStatus[];
          recommended_model_id: string;
          total_installed_bytes: number;
        }>("clips.models_list"),
      ]);
      setAiConfig(cfgRes.data);
      setProvider(cfgRes.data.default_provider === "local_free" ? "local_free" : "gemini");
      setGeminiDirectorModel(cfgRes.data.gemini_director_model || "auto");

      const mList = modelsRes.data.models ?? [];
      setVisionModels(mList);
      setTotalVisionBytes(modelsRes.data.total_installed_bytes ?? 0);
      setSelectedLocalModel((curr) => {
        if (curr && mList.some((m) => m.id === curr && m.installed)) return curr;
        const firstInst = mList.find((m) => m.installed);
        if (firstInst) return firstInst.id;
        return cfgRes.data.local_vision_model || modelsRes.data.recommended_model_id || "";
      });
    } catch {
      // ignore if offline
    }
  }, []);

  useEffect(() => {
    void loadAiConfig();
  }, [loadAiConfig]);

  async function handleSaveAiConfig(nextProvider = provider, nextLocalModel = selectedLocalModel) {
    setSavingAi(true);
    try {
      const payload: Record<string, unknown> = {
        default_provider: nextProvider,
        gemini_director_model: geminiDirectorModel,
        local_vision_model: nextLocalModel || undefined,
      };
      if (geminiKeyInput.trim()) payload.gemini_api_key = geminiKeyInput.trim();
      await runAction("clips.config_set", payload);
      setGeminiKeyInput("");
      await loadAiConfig();
      onNotify("success", "Configuração de IA salva", "Aplicada automaticamente em Cortes Shorts, Enquadramento e Motion Studio.");
    } catch (e) {
      const msg = e instanceof AicutActionError ? e.message : (e as Error).message;
      onNotify("error", "Erro ao salvar configuração de IA", msg);
    } finally {
      setSavingAi(false);
    }
  }

  async function handleInstallLlama() {
    setInstallingLlama(true);
    try {
      const res = await runAction("tools.install", { tool_id: "llama-server" });
      await onRefreshTools();
      onNotify("success", "llama.cpp server pronto!", res.summary);
    } catch (e) {
      onNotify("error", "Falha ao instalar llama-server", (e as Error).message);
    } finally {
      setInstallingLlama(false);
    }
  }

  async function handleDownloadVisionModel(modelId: string) {
    setDownloadingModel(modelId);
    try {
      const res = await runAction<{ job_id?: string }>("clips.model_download", { model_id: modelId, wait: false });
      onNotify("info", "Download do modelo iniciado", res.summary);
      setTimeout(() => void loadAiConfig(), 1500);
    } catch (e) {
      const msg = e instanceof AicutActionError ? e.message : (e as Error).message;
      onNotify("error", "Erro ao baixar modelo", msg);
    } finally {
      setDownloadingModel(null);
    }
  }

  async function handleDeleteVisionModel(modelId: string) {
    setUninstallingModel(modelId);
    try {
      const res = await runAction("clips.model_delete", { model_id: modelId });
      await loadAiConfig();
      onNotify("info", "Modelo removido", res.summary);
    } catch (e) {
      onNotify("error", "Erro ao remover modelo", (e as Error).message);
    } finally {
      setUninstallingModel(null);
    }
  }

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
    !tool.installed ? "Instalar melhor versão" : tool.update_available ? "Atualizar" : "Reinstalar";

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
        { label: "Importações", value: workspace.imports_dir, icon: Folder },
        { label: "Shorts, legendas & timelines", value: workspace.exports_dir, icon: Folder },
        { label: "Modelos de IA local", value: workspace.models_dir, icon: Cpu },
        { label: "Binários gerenciados", value: workspace.bin_dir, icon: Wrench },
        { label: "Endpoint local (MCP / REST / OpenAPI)", value: getPublicServerUrl(), icon: Server },
      ]
    : [];

  const activeJobsCount = jobs.filter((j) => j.status === "queued" || j.status === "running").length;
  const toolsNeedAttention = tools.filter((t) => !t.installed || t.update_available).length;

  return (
    <div className="stack">
      <Tabs
        ariaLabel="Categorias de Configurações"
        value={activeTab}
        onChange={onTabChange}
        items={[
          { id: "ai", label: "IA & Chaves", icon: Sparkles },
          { id: "tools", label: "Ferramentas", icon: Wrench, count: toolsNeedAttention || undefined },
          { id: "system", label: "Sistema", icon: Cpu },
          { id: "integrations", label: "Integrações", icon: Bot },
          { id: "modules", label: "Módulos", icon: Blocks, count: modules.length || undefined },
          { id: "jobs", label: "Tarefas", icon: Activity, count: activeJobsCount || undefined },
        ]}
      />

      {/* TAB 1: IA & CHAVES (Single source of truth for Gemini API & Local Vision) */}
      {activeTab === "ai" && (
        <Panel
          title="Motor de IA Visual & Chaves"
          icon={Sparkles}
          description="Configuração única compartilhada por Cortes Shorts, Enquadramento e Motion Studio."
          footer={
            <button
              type="button"
              className="btn btn-primary"
              disabled={savingAi}
              onClick={() => void handleSaveAiConfig()}
            >
              <BusyIcon busy={savingAi} icon={KeyRound} size={14} />
              Salvar configuração de IA
            </button>
          }
        >
          <ChoiceCards
            ariaLabel="Motor de IA padrão"
            value={provider}
            onChange={(next) => {
              setProvider(next);
              void handleSaveAiConfig(next, selectedLocalModel);
            }}
            options={[
              { id: "gemini", title: "Gemini API (Nuvem)", icon: Cloud, description: "Visão multimodal + áudio nativo, 0 MB em disco." },
              { id: "local_free", title: "Local grátis (GPU)", icon: Cpu, description: "Qwen2.5-VL 100% offline na sua placa de vídeo." },
            ]}
          />

          {provider === "gemini" ? (
            <div className="stack">
              <Field
                label="Chave Gemini API"
                htmlFor="cfg-gemini-key"
                aside={
                  aiConfig?.gemini_configured && (
                    <span className="text-success text-xs row gap-xs">
                      <CircleCheck size={12} aria-hidden /> configurada ({aiConfig.gemini_key_preview})
                    </span>
                  )
                }
              >
                <input
                  id="cfg-gemini-key"
                  type="password"
                  className="input"
                  value={geminiKeyInput}
                  onChange={(e) => setGeminiKeyInput(e.target.value)}
                  placeholder={
                    aiConfig?.gemini_configured
                      ? "cole uma nova chave apenas se quiser trocar…"
                      : "AIza… (grátis no Google AI Studio)"
                  }
                />
              </Field>

              <Field label="Modelo diretor Gemini" htmlFor="cfg-gemini-dir">
                <select
                  id="cfg-gemini-dir"
                  className="select"
                  value={geminiDirectorModel}
                  onChange={(e) => setGeminiDirectorModel(e.target.value)}
                >
                  <option value="auto">Automático (Melhor Gemini ativo na sua chave · Recomendado)</option>
                  <option value="gemini-3.8-pro">Gemini 3.8 Pro (Máxima inteligência visual e narrativa)</option>
                  <option value="gemini-3.8-flash">Gemini 3.8 Flash (Mais rápido e econômico)</option>
                </select>
              </Field>
            </div>
          ) : (
            <div className="stack">
              <div className="row row-between wrap text-xs secondary">
                <span className="row gap-xs">
                  <Cpu size={13} className="text-accent" aria-hidden />
                  {primaryGpu
                    ? `${primaryGpu.name}${primaryGpu.vram_mb ? ` (${Math.round(primaryGpu.vram_mb / 1024)} GB VRAM)` : ""}`
                    : "Modo CPU"}
                  {hardware?.cuda_version ? ` · CUDA ${hardware.cuda_version}` : ""}
                </span>
                <span className="row gap-xs">
                  <HardDrive size={13} aria-hidden /> {formatBytes(totalVisionBytes)} em disco
                </span>
              </div>

              <Callout
                tone={llamaTool?.installed ? "success" : "warning"}
                title={`Motor llama.cpp server · ${llamaTool?.installed ? "pronto" : "não instalado"}`}
                actions={
                  <button
                    type="button"
                    className={`btn btn-sm ${llamaTool?.installed ? "" : "btn-primary"}`}
                    disabled={installingLlama}
                    onClick={() => void handleInstallLlama()}
                  >
                    <BusyIcon busy={installingLlama} icon={Download} size={13} />
                    {llamaTool?.installed ? "Atualizar" : "Instalar auto"}
                  </button>
                }
              >
                {llamaTool?.recommended_build ?? "Detecta automaticamente CUDA / Vulkan / CPU"}
              </Callout>

              <div className="stack stack-sm">
                {visionModels.map((m) => {
                  const isActive = selectedLocalModel === m.id;
                  return (
                    <div key={m.id} className="inset stack stack-sm">
                      <div className="row row-between wrap gap-xs">
                        <div className="row wrap gap-xs">
                          <span className="strong text-sm">{m.display_name}</span>
                          {m.recommended && (
                            <Badge tone="accent" icon={Award}>
                              recomendado
                            </Badge>
                          )}
                          <Badge>{m.quality_tier}</Badge>
                        </div>
                        <span className="muted text-xs">{(m.size_mb / 1024).toFixed(1)} GB</span>
                      </div>
                      <div className="secondary text-xs">{m.quality_notes}</div>
                      <div className={`text-xs ${m.compatible ? "text-success" : "text-warning"}`}>{m.compatibility_reason}</div>
                      <div className="row gap-xs">
                        {m.installed ? (
                          <>
                            <button
                              type="button"
                              className={`btn btn-sm ${isActive ? "btn-primary" : ""}`}
                              onClick={() => {
                                setSelectedLocalModel(m.id);
                                void handleSaveAiConfig("local_free", m.id);
                              }}
                            >
                              {isActive ? <CircleCheck aria-hidden /> : <Play aria-hidden />}
                              {isActive ? "Ativo" : "Usar este"}
                            </button>
                            <button
                              type="button"
                              className="btn btn-sm"
                              disabled={uninstallingModel === m.id}
                              onClick={() => void handleDeleteVisionModel(m.id)}
                            >
                              <BusyIcon busy={uninstallingModel === m.id} icon={Trash2} size={13} />
                              Desinstalar
                            </button>
                          </>
                        ) : (
                          <button
                            type="button"
                            className="btn btn-primary btn-sm"
                            disabled={downloadingModel !== null}
                            onClick={() => void handleDownloadVisionModel(m.id)}
                          >
                            <BusyIcon busy={downloadingModel === m.id} icon={Download} size={13} />
                            Baixar ({(m.size_mb / 1024).toFixed(1)} GB)
                          </button>
                        )}
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>
          )}
        </Panel>
      )}

      {/* TAB 2: FERRAMENTAS */}
      {activeTab === "tools" && (
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
      )}

      {/* TAB 3: SISTEMA (Aparência + Hardware + Workspace) */}
      {activeTab === "system" && (
        <div className="stack">
          <Panel title="Aparência" icon={Palette} description="Esquemas de cor salvos localmente neste computador.">
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
                <StatTile icon={Monitor} label="Sistema" value={hardware.os_version} detail={`${hardware.arch} (${hardware.platform_tag})`} />
                <StatTile icon={Cpu} label="Processador" value={hardware.cpu_brand} detail={`${hardware.physical_cores} núcleos · ${hardware.logical_cores} threads`} />
                <StatTile icon={MemoryStick} label="Memória RAM" value={`${(hardware.total_memory_mb / 1024).toFixed(1)} GB`} detail={`${(hardware.available_memory_mb / 1024).toFixed(1)} GB livre`} />
                <StatTile
                  icon={HardDrive}
                  label="GPU / aceleração"
                  value={hardware.gpus.length > 0 ? hardware.gpus[0].name : "CPU integrada"}
                  detail={`${ACCEL_LABEL[hardware.recommended_accel] ?? hardware.recommended_accel}${hardware.cuda_version ? ` · CUDA ${hardware.cuda_version}` : ""}`}
                />
              </div>
              <p className="secondary text-sm">{hardware.recommendation_summary}</p>
            </Panel>
          )}

          {workspace && (
            <Panel title="Pastas do workspace" icon={Folder} flush>
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
                    <button
                      type="button"
                      className="btn btn-ghost btn-icon btn-sm row-actions"
                      aria-label={`Copiar ${p.label}`}
                      onClick={() => void copyToClipboard(p.value, onNotify, p.label)}
                    >
                      <Copy aria-hidden />
                    </button>
                  </div>
                ))}
              </div>
            </Panel>
          )}
        </div>
      )}

      {/* TAB 4: INTEGRAÇÕES (MCP / OpenAPI / CLI + Log de atividade) */}
      {activeTab === "integrations" && (
        <AiConnectView
          activity={activity}
          actionCount={actions.length}
          connected={connected}
          onNavigate={onNavigate}
          onNotify={onNotify}
        />
      )}

      {/* TAB 5: MÓDULOS & ACTION PLAYGROUND */}
      {activeTab === "modules" && (
        <ModulesView modules={modules} actions={actions} onRefresh={onRefreshModules} onNotify={onNotify} />
      )}

      {/* TAB 6: FILA DE TAREFAS */}
      {activeTab === "jobs" && (
        <JobsView jobs={jobs} onRefresh={onRefreshJobs} onClearFinished={onClearFinishedJobs} onNotify={onNotify} />
      )}
    </div>
  );
};
