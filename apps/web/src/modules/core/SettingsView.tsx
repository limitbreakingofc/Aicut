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
  amf_directml: "AMD Vulkan / VAAPI",
  qsv_openvino: "Intel QuickSync / Vulkan",
  metal: "Hardware GPU",
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


  // AI & Keys state (single source of truth for Cortes Shorts, Enquadramento & Motion Studio)
  const [aiConfig, setAiConfig] = useState<ClipsConfig | null>(null);
  const [provider, setProvider] = useState<"gemini" | "local_free">("gemini");
  const [geminiKeyInput, setGeminiKeyInput] = useState("");
  const [geminiDirectorModel, setGeminiDirectorModel] = useState("auto");
  const [savingAi, setSavingAi] = useState(false);
  const [visionModels, setVisionModels] = useState<VisionModelStatus[]>([]);
  const [selectedLocalModel, setSelectedLocalModel] = useState("");


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

  return (
    <div className="stack">
      <Tabs
        ariaLabel="Categorias de Configurações"
        value={activeTab}
        onChange={onTabChange}
        items={[
          { id: "ai", label: "IA & Chaves", icon: Sparkles },
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
            <Field label="Modelo visual instalado">
              <select className="select" value={selectedLocalModel} onChange={e => {setSelectedLocalModel(e.target.value); void handleSaveAiConfig("local_free", e.target.value);}}>
                <option value="">Automático</option>
                {visionModels.filter(m => m.installed).map(m => <option key={m.id} value={m.id}>{m.display_name}</option>)}
              </select>
              <button className="btn btn-sm" onClick={() => onNavigate("downloads")}>Gerenciar downloads</button>
            </Field>
          )}
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
