import React, { useEffect, useMemo, useState } from "react";
import {
  Award,
  CircleCheck,
  Cloud,
  Copy,
  Cpu,
  Download,
  Eye,
  Film,
  HardDrive,
  KeyRound,
  Lightbulb,
  Play,
  Scissors,
  Sparkles,
  Captions,
  Trash2,
  Zap,
} from "lucide-react";
import { AicutActionError, runAction } from "../../lib/client";
import { formatBytes, formatDuration, formatRelativeDate } from "../../lib/format";
import type {
  ClipsConfig,
  HardwareProfile,
  JobRecord,
  MediaAsset,
  MediaContextInfo,
  NotifyFn,
  RouteId,
  SmartClipPlan,
  SmartClipRun,
  ToolStatus,
  VisionModelStatus,
} from "../../lib/types";
import { copyToClipboard } from "../../lib/ui";
import { pageMeta } from "../../app/navigation";
import {
  ActiveJobsPanel,
  Badge,
  BusyIcon,
  Callout,
  Checkbox,
  ChoiceCards,
  EmptyState,
  Field,
  Meter,
  MoreButton,
  PageHeader,
  Panel,
  useContextMenu,
  type MenuItems,
} from "../../components";
import "./smart-clips.css";

interface SmartClipsViewProps {
  media: MediaAsset[];
  tools: ToolStatus[];
  jobs: JobRecord[];
  hardware: HardwareProfile | null;
  preselectedMediaId?: string | null;
  onNavigate: (route: RouteId) => void;
  onSelectMedia: (mediaId: string) => void;
  onNotify: NotifyFn;
  onRefreshTools: () => Promise<void>;
  onRefreshMedia: () => Promise<void>;
}

const DESCRIPTION_PRESETS = [
  {
    label: "Melhores momentos",
    text: "Encontre os momentos mais envolventes, visualmente marcantes e surpreendentes do vídeo ou live, com gancho forte nos primeiros 5 segundos e fechamento completo.",
  },
  {
    label: "Jogadas / clutches",
    text: "Procure jogadas épicas, clutches, viradas inesperadas, falhas engraçadas ou reações intensas na gameplay, mostrando a preparação, a jogada e a reação final.",
  },
  {
    label: "Explicações & dicas",
    text: "Encontre explicações didáticas, demonstrações na tela ou dicas valiosas que comecem com uma pergunta/problema claro e terminem com a solução completa.",
  },
  {
    label: "Humor / reações",
    text: "Encontre momentos engraçados, piadas, sustos, rage divertido ou interações memoráveis com expressões faciais fortes e punchline fechada.",
  },
];

type ShortsLayout = "template_split" | "gaming_split" | "fit_blur" | "center_crop" | "original";
type WebcamCorner = "bottom_right" | "bottom_left" | "top_right" | "top_left" | "center";

const SHORTS_LAYOUTS: Array<{ id: ShortsLayout; short: string; label: string; desc: string }> = [
  {
    id: "template_split",
    short: "Moldura SMOKING33",
    label: "Moldura Shorts (SMOKING33) · Câmera em cima + conteúdo em baixo",
    desc: "Aplica sua moldura 9:16 com borda dourada: webcam na janela superior e conteúdo/gameplay na janela inferior.",
  },
  {
    id: "gaming_split",
    short: "Split sem moldura",
    label: "Split 9:16 sem moldura · Câmera em cima + conteúdo em baixo",
    desc: "Webcam cheia na parte superior e gameplay/conteúdo na parte inferior, sem moldura.",
  },
  {
    id: "fit_blur",
    short: "Tela inteira + blur",
    label: "Shorts 9:16 · Tela inteira + blur",
    desc: "Mantém o vídeo 16:9 sem cortar bordas sobre fundo desfocado (ideal para tutoriais e tela única).",
  },
  {
    id: "center_crop",
    short: "Foco central",
    label: "Shorts 9:16 · Foco central",
    desc: "Enquadramento vertical cheio focado no centro (ideal para vlogs, podcasts e câmera principal).",
  },
  {
    id: "original",
    short: "Original 16:9",
    label: "Original 16:9 (horizontal)",
    desc: "Mantém a proporção original do vídeo sem crop vertical.",
  },
];

/** Tiny CSS-drawn preview of each 9:16 framing. */
function LayoutPreview({ id }: { id: ShortsLayout }) {
  return (
    <span className={`lp lp-${id}`} aria-hidden>
      <i />
      <i />
    </span>
  );
}

const clipTimecodes = (clip: SmartClipPlan) =>
  clip.segments.map((s) => `${formatDuration(s.start_sec)} → ${formatDuration(s.end_sec)}`).join("  +  ");

export const SmartClipsView: React.FC<SmartClipsViewProps> = ({
  media,
  tools,
  jobs,
  hardware,
  preselectedMediaId,
  onNavigate,
  onSelectMedia,
  onNotify,
  onRefreshTools,
  onRefreshMedia,
}) => {
  const menu = useContextMenu();
  const videoMedia = useMemo(() => media.filter((m) => m.kind === "video" || m.kind === "video_only"), [media]);

  const [selectedMediaId, setSelectedMediaId] = useState<string>(preselectedMediaId ?? videoMedia[0]?.id ?? "");
  const [provider, setProvider] = useState<"gemini" | "local_free">("gemini");
  const [description, setDescription] = useState<string>(DESCRIPTION_PRESETS[0].text);
  const [maxDurationSec, setMaxDurationSec] = useState<number>(180);
  const [maxClips, setMaxClips] = useState<number>(5);
  const [allowJumpCuts, setAllowJumpCuts] = useState<boolean>(false);

  // Provider configuration
  const [config, setConfig] = useState<ClipsConfig | null>(null);
  const [geminiKeyInput, setGeminiKeyInput] = useState<string>("");
  const [geminiDirectorModel, setGeminiDirectorModel] = useState<string>("gemini-2.5-pro");
  const [savingConfig, setSavingConfig] = useState<boolean>(false);

  // Local Free vision models
  const [visionModels, setVisionModels] = useState<VisionModelStatus[]>([]);
  const [totalVisionBytes, setTotalVisionBytes] = useState<number>(0);
  const [selectedLocalModel, setSelectedLocalModel] = useState<string>("");
  const [installingLlama, setInstallingLlama] = useState<boolean>(false);
  const [downloadingModel, setDownloadingModel] = useState<string | null>(null);
  const [uninstallingModel, setUninstallingModel] = useState<string | null>(null);

  // Runs & full-library video context
  const [runs, setRuns] = useState<SmartClipRun[]>([]);
  const [activeRunId, setActiveRunId] = useState<string | null>(null);
  const [mediaContext, setMediaContext] = useState<MediaContextInfo | null>(null);

  // Shorts rendering options
  const [layout, setLayout] = useState<ShortsLayout>("template_split");
  const [webcamCorner, setWebcamCorner] = useState<WebcamCorner>("bottom_right");
  const [contentFit, setContentFit] = useState<"fill" | "fit_blur">("fill");
  const [customTemplatePath, setCustomTemplatePath] = useState<string>("");
  const [burnSubtitles, setBurnSubtitles] = useState<boolean>(true);
  const [analyzing, setAnalyzing] = useState<boolean>(false);
  const [renderingKey, setRenderingKey] = useState<string | null>(null);

  const ffmpegTool = tools.find((t) => t.id === "ffmpeg");
  const llamaTool = tools.find((t) => t.id === "llama-server");

  useEffect(() => {
    if (preselectedMediaId) {
      setSelectedMediaId(preselectedMediaId);
    } else if (!selectedMediaId && videoMedia.length > 0) {
      setSelectedMediaId(videoMedia[0].id);
    }
  }, [preselectedMediaId, videoMedia.length]);

  const loadMediaContext = async (mediaId: string) => {
    if (!mediaId) {
      setMediaContext(null);
      return;
    }
    try {
      const res = await runAction<MediaContextInfo>("clips.media_context", { media_id: mediaId });
      setMediaContext(res.data);
    } catch {
      setMediaContext(null);
    }
  };

  const loadAll = async () => {
    try {
      const [cfgRes, modelsRes, runsRes] = await Promise.all([
        runAction<ClipsConfig>("clips.config_get"),
        runAction<{
          models: VisionModelStatus[];
          recommended_model_id: string;
          total_installed_bytes: number;
        }>("clips.models_list"),
        runAction<{ runs: SmartClipRun[] }>("clips.runs_list"),
      ]);

      setConfig(cfgRes.data);
      setCustomTemplatePath(cfgRes.data.template_path ?? "");
      if (cfgRes.data.default_provider === "local_free") {
        setProvider("local_free");
      } else {
        setProvider("gemini");
      }
      setGeminiDirectorModel(cfgRes.data.gemini_director_model || "gemini-2.5-pro");

      const mList = modelsRes.data.models ?? [];
      setVisionModels(mList);
      setTotalVisionBytes(modelsRes.data.total_installed_bytes ?? 0);
      setSelectedLocalModel((curr) => {
        if (curr && mList.some((m) => m.id === curr && m.installed)) return curr;
        const firstInst = mList.find((m) => m.installed);
        if (firstInst) return firstInst.id;
        return cfgRes.data.local_vision_model || modelsRes.data.recommended_model_id || "";
      });

      const rList = runsRes.data.runs ?? [];
      setRuns(rList);
      if (rList.length > 0 && !activeRunId) {
        setActiveRunId(rList[0].id);
      }
    } catch {
      // ignore if backend offline
    }
  };

  useEffect(() => {
    void loadAll();
  }, []);

  const clipJobKinds = ["clips.analyze", "clips.render", "clips.model_download"];
  const activeClipJobs = jobs.filter(
    (j) => clipJobKinds.includes(j.kind) && (j.status === "queued" || j.status === "running"),
  );
  // Transcriptions also matter: a finished transcript instantly links to this video's context
  const completedRelatedJobsCount = jobs.filter(
    (j) => (clipJobKinds.includes(j.kind) || j.kind === "transcribe.run") && j.status === "completed",
  ).length;

  useEffect(() => {
    if (selectedMediaId) {
      void loadMediaContext(selectedMediaId);
    } else {
      setMediaContext(null);
    }
  }, [selectedMediaId, media.length, completedRelatedJobsCount]);

  useEffect(() => {
    if (completedRelatedJobsCount > 0) {
      void loadAll();
      void onRefreshMedia();
    }
  }, [completedRelatedJobsCount]);

  const resolvedTranscript = mediaContext?.resolved_transcript ?? null;
  const relatedCount = Math.max(0, (mediaContext?.related_assets?.length ?? 1) - 1);

  const activeRun = useMemo(() => runs.find((r) => r.id === activeRunId) ?? runs[0] ?? null, [runs, activeRunId]);

  const saveGeminiSettings = async () => {
    setSavingConfig(true);
    try {
      const payload: Record<string, unknown> = {
        default_provider: provider,
        gemini_director_model: geminiDirectorModel,
      };
      if (geminiKeyInput.trim()) {
        payload.gemini_api_key = geminiKeyInput.trim();
      }
      const res = await runAction<ClipsConfig>("clips.config_set", payload);
      setConfig(res.data);
      setGeminiKeyInput("");
      onNotify("success", "Configuração salva", "Preferências da Gemini API atualizadas.");
    } catch (e) {
      onNotify("error", "Erro ao salvar", e instanceof AicutActionError ? e.message : "Falha ao salvar configuração.");
    } finally {
      setSavingConfig(false);
    }
  };

  const installLlamaServer = async () => {
    setInstallingLlama(true);
    try {
      await runAction("tools.install", { tool_id: "llama-server" });
      await onRefreshTools();
      onNotify("success", "llama.cpp instalado", "Runtime visual local pronto na melhor build para sua GPU/hardware.");
    } catch (e) {
      onNotify("error", "Erro ao instalar llama.cpp", e instanceof AicutActionError ? e.message : "Falha na instalação.");
    } finally {
      setInstallingLlama(false);
    }
  };

  const handleDownloadModel = async (modelId: string) => {
    setDownloadingModel(modelId);
    try {
      await runAction("clips.model_download", { model_id: modelId });
      setSelectedLocalModel(modelId);
      await runAction("clips.config_set", { local_vision_model: modelId }).catch(() => {});
      await loadAll();
      onNotify("success", "Modelo visual pronto", `Modelo ${modelId} baixado com sucesso.`);
    } catch (e) {
      onNotify("error", "Erro no download", e instanceof AicutActionError ? e.message : "Falha ao baixar modelo visual.");
    } finally {
      setDownloadingModel(null);
    }
  };

  const handleDeleteModel = async (modelId: string) => {
    setUninstallingModel(modelId);
    try {
      const res = await runAction<{ freed_bytes: number }>("clips.model_delete", { model_id: modelId });
      await loadAll();
      onNotify("info", "Modelo desinstalado", `Espaço liberado: ${formatBytes(res.data.freed_bytes ?? 0)}.`);
    } catch (e) {
      onNotify("error", "Erro ao desinstalar", e instanceof AicutActionError ? e.message : "Falha ao remover modelo.");
    } finally {
      setUninstallingModel(null);
    }
  };

  const handleStartAnalyze = async () => {
    if (!selectedMediaId) {
      onNotify("warning", "Selecione um vídeo", "Escolha um vídeo ou live da Biblioteca primeiro.");
      return;
    }
    if (!description.trim()) {
      onNotify("warning", "Descreva o que procurar", "Informe uma descrição editorial para guiar a IA visual.");
      return;
    }
    setAnalyzing(true);
    try {
      await runAction("clips.config_set", {
        default_provider: provider,
        gemini_director_model: geminiDirectorModel,
        local_vision_model: selectedLocalModel,
      }).catch(() => {});

      await runAction("clips.analyze", {
        media_id: selectedMediaId,
        description: description.trim(),
        provider,
        model_id: provider === "local_free" ? selectedLocalModel : geminiDirectorModel,
        max_duration_sec: maxDurationSec,
        min_duration_sec: 25,
        max_clips: maxClips,
        allow_jump_cuts: allowJumpCuts,
        wait: false,
      });
      onNotify(
        "info",
        "Análise visual iniciada",
        "A IA está assistindo ao vídeo para encontrar cortes coerentes de até 3 min. Acompanhe o progresso abaixo.",
      );
    } catch (e) {
      onNotify(
        "error",
        "Não foi possível iniciar",
        e instanceof AicutActionError ? e.message : "Verifique os pré-requisitos do motor escolhido.",
      );
    } finally {
      setAnalyzing(false);
    }
  };

  const handleRender = async (runId: string, clipId?: string) => {
    const key = clipId ?? `all:${runId}`;
    setRenderingKey(key);
    try {
      await runAction("clips.render", {
        run_id: runId,
        clip_id: clipId,
        layout,
        webcam_corner: webcamCorner,
        content_fit: contentFit,
        template_path: customTemplatePath.trim() ? customTemplatePath.trim() : undefined,
        burn_subtitles: burnSubtitles,
        wait: false,
      });
      onNotify(
        "info",
        "Renderização Shorts iniciada",
        "Os cortes serão salvos automaticamente na Biblioteca assim que o FFmpeg concluir.",
      );
    } catch (e) {
      onNotify("error", "Erro ao renderizar", e instanceof AicutActionError ? e.message : "Falha ao iniciar renderização.");
    } finally {
      setRenderingKey(null);
    }
  };

  const handleDeleteRun = async (runId: string) => {
    try {
      await runAction("clips.run_delete", { id: runId });
      if (activeRunId === runId) setActiveRunId(null);
      await loadAll();
    } catch {
      // ignore
    }
  };

  const saveTemplatePath = () => {
    void runAction("clips.config_set", {
      template_path: customTemplatePath.trim(),
    })
      .then(() =>
        onNotify(
          "success",
          "Moldura salva",
          customTemplatePath.trim() ? "Moldura PNG personalizada salva como padrão." : "Moldura padrão SMOKING33 restaurada.",
        ),
      )
      .catch(() => {});
  };

  const primaryGpu = hardware?.gpus?.[0];
  const currentLayout = SHORTS_LAYOUTS.find((l) => l.id === layout) ?? SHORTS_LAYOUTS[0];
  const isSplit = layout === "template_split" || layout === "gaming_split";

  /* ---------- Context menus ---------- */
  const clipMenu = (run: SmartClipRun, clip: SmartClipPlan, index: number): MenuItems => [
    { type: "label", label: `#${index + 1} · ${clip.title}` },
    {
      label: clip.rendered_media_id ? "Re-renderizar Short" : "Renderizar Short",
      icon: Scissors,
      disabled: renderingKey !== null,
      onSelect: () => void handleRender(run.id, clip.id),
    },
    clip.rendered_media_id && {
      label: "Abrir na Biblioteca",
      icon: Play,
      onSelect: () => onSelectMedia(clip.rendered_media_id!),
    },
    { type: "separator" },
    { label: "Copiar título", icon: Copy, onSelect: () => void copyToClipboard(clip.title, onNotify, "Título") },
    { label: "Copiar timecodes", icon: Copy, onSelect: () => void copyToClipboard(clipTimecodes(clip), onNotify, "Timecodes") },
    clip.hook_first_5s && {
      label: "Copiar gancho",
      icon: Copy,
      onSelect: () => void copyToClipboard(clip.hook_first_5s, onNotify, "Gancho"),
    },
  ];

  const visionMenu = (m: VisionModelStatus): MenuItems => [
    { type: "label", label: m.display_name },
    m.installed && {
      label: selectedLocalModel === m.id ? "Modelo ativo" : "Usar este modelo",
      icon: CircleCheck,
      disabled: selectedLocalModel === m.id,
      onSelect: () => setSelectedLocalModel(m.id),
    },
    !m.installed && {
      label: `Baixar (${(m.size_mb / 1024).toFixed(1)} GB)`,
      icon: Download,
      disabled: downloadingModel !== null,
      onSelect: () => void handleDownloadModel(m.id),
    },
    m.model_path && {
      label: "Copiar caminho do modelo",
      icon: Copy,
      onSelect: () => void copyToClipboard(m.model_path ?? "", onNotify, "Caminho do modelo"),
    },
    m.installed && { type: "separator" },
    m.installed && {
      label: "Desinstalar",
      icon: Trash2,
      danger: true,
      disabled: uninstallingModel === m.id,
      onSelect: () => void handleDeleteModel(m.id),
    },
  ];

  return (
    <>
      <PageHeader
        {...pageMeta("smart-clips")}
        actions={
          <button type="button" className="btn btn-sm" onClick={() => onNavigate("ingest")}>
            <Download aria-hidden /> Baixar YouTube / Twitch / PC
          </button>
        }
      />

      {!ffmpegTool?.installed && (
        <Callout
          tone="warning"
          title="FFmpeg necessário"
          actions={
            <button type="button" className="btn btn-primary btn-sm" onClick={() => onNavigate("settings")}>
              Instalar em Ferramentas
            </button>
          }
        >
          O FFmpeg precisa estar instalado para extrair frames e renderizar Shorts.
        </Callout>
      )}

      <ActiveJobsPanel jobs={activeClipJobs} title="Cortes IA em andamento" tone="accent" onViewAll={() => onNavigate("jobs")} />

      <div className="clips-config">
        {/* Step 1: video + editorial brief */}
        <Panel
          step={1}
          title="Vídeo & briefing"
          icon={Film}
          footer={
            <button
              type="button"
              className="btn btn-primary btn-block"
              disabled={analyzing || !selectedMediaId}
              onClick={handleStartAnalyze}
            >
              <BusyIcon busy={analyzing} icon={Sparkles} />
              Analisar vídeo e gerar planos de cortes
            </button>
          }
        >
          <Field label="Vídeo / live da Biblioteca" htmlFor="sc-media">
            {videoMedia.length === 0 ? (
              <Callout
                actions={
                  <button type="button" className="btn btn-sm" onClick={() => onNavigate("ingest")}>
                    Baixar & Importar
                  </button>
                }
              >
                Nenhum vídeo na Biblioteca ainda. Baixe do YouTube/Twitch ou importe um arquivo local.
              </Callout>
            ) : (
              <select
                id="sc-media"
                className="select"
                value={selectedMediaId}
                onChange={(e) => setSelectedMediaId(e.target.value)}
              >
                {videoMedia.map((m) => (
                  <option key={m.id} value={m.id}>
                    {m.title}
                    {m.metadata.duration_sec ? ` (${formatDuration(m.metadata.duration_sec)})` : ""}
                  </option>
                ))}
              </select>
            )}
          </Field>

          {selectedMediaId && (
            <Callout
              tone={resolvedTranscript ? "success" : "neutral"}
              icon={Captions}
              title={
                resolvedTranscript
                  ? `Transcrição vinculada automaticamente · ${resolvedTranscript.segment_count} falas · ${resolvedTranscript.language.toUpperCase()}`
                  : "Nenhuma transcrição encontrada na Biblioteca para este vídeo ainda"
              }
              actions={
                !resolvedTranscript && (
                  <button type="button" className="btn btn-sm" onClick={() => onNavigate("transcribe")}>
                    Transcrever
                  </button>
                )
              }
            >
              <span className="lineage">
                {resolvedTranscript && <span>origem: {resolvedTranscript.source}</span>}
                <span>video_id: {selectedMediaId.slice(0, 12)}</span>
                {mediaContext?.external_video_key && <span>{mediaContext.external_video_key}</span>}
                <span>{relatedCount} arquivo(s) vinculado(s) na Biblioteca</span>
              </span>
            </Callout>
          )}

          <Field label="O que a IA visual deve procurar nos cortes?" htmlFor="sc-desc">
            <div className="row wrap gap-xs">
              {DESCRIPTION_PRESETS.map((p) => (
                <button
                  key={p.label}
                  type="button"
                  className="chip chip-sm"
                  aria-pressed={description === p.text}
                  onClick={() => setDescription(p.text)}
                >
                  {p.label}
                </button>
              ))}
            </div>
            <textarea
              id="sc-desc"
              className="textarea"
              rows={4}
              value={description}
              onChange={(e) => setDescription(e.target.value)}
              placeholder="Ex.: Ache os momentos mais engraçados da live, jogadas absurdas com reação na webcam ou explicações completas que façam sentido do início ao fim..."
            />
          </Field>

          <div className="grid-2">
            <Field label="Duração máxima (Shorts até 3 min)" htmlFor="sc-maxdur">
              <select
                id="sc-maxdur"
                className="select"
                value={maxDurationSec}
                onChange={(e) => setMaxDurationSec(Number(e.target.value))}
              >
                <option value={60}>Até 1 minuto (60s)</option>
                <option value={90}>Até 1m30s (90s)</option>
                <option value={120}>Até 2 minutos (120s)</option>
                <option value={180}>Até 3 minutos (180s · Recomendado)</option>
              </select>
            </Field>
            <Field label="Quantidade máxima de cortes" htmlFor="sc-maxclips">
              <select
                id="sc-maxclips"
                className="select"
                value={maxClips}
                onChange={(e) => setMaxClips(Number(e.target.value))}
              >
                <option value={3}>Até 3 cortes</option>
                <option value={5}>Até 5 cortes</option>
                <option value={8}>Até 8 cortes</option>
                <option value={10}>Até 10 cortes</option>
              </select>
            </Field>
          </div>

          <Checkbox
            checked={allowJumpCuts}
            onChange={setAllowJumpCuts}
            label="Permitir jump-cuts internos leves"
            description={`Pula pausas longas e tempos mortos mantendo início–meio–fim em até ${maxDurationSec}s.`}
          />
        </Panel>

        {/* Step 2: visual AI engine */}
        <Panel step={2} title="Motor de IA visual" icon={Eye}>
          <ChoiceCards
            ariaLabel="Motor de IA visual"
            value={provider}
            onChange={setProvider}
            options={[
              { id: "gemini", title: "Gemini API", icon: Cloud, description: "Visão + áudio nativos de vídeo na nuvem." },
              { id: "local_free", title: "Local grátis", icon: Cpu, description: "Qwen2.5-VL offline na sua GPU." },
            ]}
          />

          {provider === "gemini" ? (
            <>
              <Callout tone="info" icon={Cloud}>
                O Gemini assiste diretamente aos blocos de vídeo (25 min com 3 min de sobreposição) usando{" "}
                <strong>Gemini 2.5 Flash</strong> na varredura e <strong>{geminiDirectorModel}</strong> na direção final. 0 MB de
                download local.
              </Callout>

              <Field
                label="Chave Gemini API"
                htmlFor="sc-key"
                aside={
                  config?.gemini_configured && (
                    <span className="text-success text-xs row gap-xs">
                      <CircleCheck size={12} aria-hidden /> configurada ({config.gemini_key_preview})
                    </span>
                  )
                }
              >
                <div className="row">
                  <input
                    id="sc-key"
                    type="password"
                    className="input"
                    value={geminiKeyInput}
                    onChange={(e) => setGeminiKeyInput(e.target.value)}
                    placeholder={
                      config?.gemini_configured
                        ? "cole uma nova chave apenas se quiser trocar…"
                        : "AIza… (obtenha grátis no Google AI Studio)"
                    }
                  />
                  <button type="button" className="btn btn-primary" disabled={savingConfig} onClick={saveGeminiSettings}>
                    <BusyIcon busy={savingConfig} icon={KeyRound} /> Salvar
                  </button>
                </div>
              </Field>

              <Field label="Modelo diretor (passagem B — coerência início–meio–fim)" htmlFor="sc-director">
                <select
                  id="sc-director"
                  className="select"
                  value={geminiDirectorModel}
                  onChange={(e) => setGeminiDirectorModel(e.target.value)}
                >
                  <option value="gemini-2.5-pro">Gemini 2.5 Pro (Máxima inteligência visual e narrativa)</option>
                  <option value="gemini-2.5-flash">Gemini 2.5 Flash (Mais rápido e econômico)</option>
                </select>
              </Field>
            </>
          ) : (
            <>
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
                    onClick={installLlamaServer}
                  >
                    <BusyIcon busy={installingLlama} icon={Download} size={13} />
                    {llamaTool?.installed ? "Atualizar" : "Instalar auto"}
                  </button>
                }
              >
                {llamaTool?.recommended_build ?? "Detecta automaticamente CUDA / Vulkan / CPU"}
              </Callout>

              <div className="vision-list">
                {visionModels.map((m) => {
                  const isActive = selectedLocalModel === m.id;
                  return (
                    <div
                      key={m.id}
                      className="vision-model hover-host"
                      data-active={isActive || undefined}
                      onContextMenu={(e) => menu.open(e, visionMenu(m))}
                    >
                      <div className="row row-between">
                        <div className="row wrap gap-xs" style={{ minWidth: 0 }}>
                          <span className="strong text-sm">{m.display_name}</span>
                          {m.recommended && (
                            <Badge tone="accent" icon={Award}>
                              recomendado
                            </Badge>
                          )}
                          <Badge>{m.quality_tier}</Badge>
                        </div>
                        <span className="row gap-xs">
                          <span className="muted text-xs">{(m.size_mb / 1024).toFixed(1)} GB</span>
                          <span className="row-actions">
                            <MoreButton items={() => visionMenu(m)} />
                          </span>
                        </span>
                      </div>
                      <div className="secondary text-xs">{m.quality_notes}</div>
                      <div className={`text-xs ${m.compatible ? "text-success" : "text-warning"}`}>{m.compatibility_reason}</div>
                      <div className="row gap-xs">
                        {m.installed ? (
                          <>
                            <button
                              type="button"
                              className={`btn btn-sm ${isActive ? "btn-primary" : ""}`}
                              onClick={() => setSelectedLocalModel(m.id)}
                            >
                              {isActive ? <CircleCheck aria-hidden /> : <Play aria-hidden />}
                              {isActive ? "Ativo" : "Usar este"}
                            </button>
                            <button
                              type="button"
                              className="btn btn-sm"
                              disabled={uninstallingModel === m.id}
                              onClick={() => handleDeleteModel(m.id)}
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
                            onClick={() => handleDownloadModel(m.id)}
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
            </>
          )}
        </Panel>
      </div>

      {/* Step 3: results & 9:16 rendering */}
      <Panel
        step={3}
        title="Cortes encontrados & renderização"
        icon={Scissors}
        actions={
          runs.length > 0 && (
            <>
              <select
                className="select select-sm select-auto run-select"
                aria-label="Análise"
                value={activeRun?.id ?? ""}
                onChange={(e) => setActiveRunId(e.target.value)}
              >
                {runs.map((r) => (
                  <option key={r.id} value={r.id}>
                    {r.media_title} · {r.clips.length} corte(s) · {r.provider} ({formatRelativeDate(r.created_at)})
                  </option>
                ))}
              </select>
              {activeRun && (
                <button
                  type="button"
                  className="btn btn-ghost btn-icon btn-sm"
                  onClick={() => handleDeleteRun(activeRun.id)}
                  aria-label="Excluir esta análise"
                  title="Excluir esta análise"
                >
                  <Trash2 aria-hidden />
                </button>
              )}
            </>
          )
        }
      >
        <div className="inset stack">
          <div className="row row-between wrap">
            <span className="section-label">Formato do Short</span>
            {activeRun && activeRun.clips.length > 0 && (
              <button
                type="button"
                className="btn btn-primary btn-sm"
                disabled={renderingKey !== null}
                onClick={() => handleRender(activeRun.id)}
              >
                <BusyIcon busy={renderingKey === `all:${activeRun.id}`} icon={Scissors} size={13} />
                Renderizar todos ({activeRun.clips.length})
              </button>
            )}
          </div>

          <ChoiceCards
            ariaLabel="Enquadramento Shorts"
            className="layout-picker"
            columns={5}
            value={layout}
            onChange={setLayout}
            options={SHORTS_LAYOUTS.map((l) => ({ id: l.id, title: l.short, preview: <LayoutPreview id={l.id} /> }))}
          />
          <p className="secondary text-xs">
            <strong className="text-accent">{currentLayout.label}</strong> — {currentLayout.desc}
          </p>

          {isSplit && (
            <div className="grid-2">
              <Field label="Origem da câmera (janela de cima)" htmlFor="sc-cam">
                <select
                  id="sc-cam"
                  className="select"
                  value={webcamCorner}
                  onChange={(e) => setWebcamCorner(e.target.value as WebcamCorner)}
                >
                  <option value="bottom_right">Canto inferior direito</option>
                  <option value="bottom_left">Canto inferior esquerdo</option>
                  <option value="top_right">Canto superior direito</option>
                  <option value="top_left">Canto superior esquerdo</option>
                  <option value="center">Centro (vídeo sem webcam no canto)</option>
                </select>
              </Field>
              <Field label="Conteúdo (janela de baixo)" htmlFor="sc-fit">
                <select
                  id="sc-fit"
                  className="select"
                  value={contentFit}
                  onChange={(e) => setContentFit(e.target.value as "fill" | "fit_blur")}
                >
                  <option value="fill">Preencher janela (sem bordas)</option>
                  <option value="fit_blur">Tela 16:9 inteira + blur</option>
                </select>
              </Field>
            </div>
          )}

          <div className="row wrap gap-md">
            <Checkbox
              checked={burnSubtitles}
              onChange={setBurnSubtitles}
              label="Queimar legendas karaokê sincronizadas (se transcrito)"
            />
            {layout === "template_split" && (
              <div className="row grow template-row">
                <input
                  type="text"
                  className="input input-sm"
                  aria-label="Caminho da moldura PNG"
                  value={customTemplatePath}
                  onChange={(e) => setCustomTemplatePath(e.target.value)}
                  placeholder="moldura padrão SMOKING33 (ou cole o caminho de outro PNG 9:16 transparente)"
                />
                <button type="button" className="btn btn-sm" onClick={saveTemplatePath}>
                  Salvar PNG
                </button>
              </div>
            )}
          </div>
        </div>

        {!activeRun || activeRun.clips.length === 0 ? (
          <EmptyState
            compact
            icon={Scissors}
            title="nenhuma análise concluída"
            description="Selecione um vídeo, descreva o que procurar e clique em Analisar vídeo."
          />
        ) : (
          <div className="stack">
            {activeRun.clips.map((clip, i) => (
              <article
                key={clip.id}
                className="clip-card hover-host"
                onContextMenu={(e) => menu.open(e, clipMenu(activeRun, clip, i))}
              >
                <header className="clip-head">
                  <span className="clip-index">#{String(i + 1).padStart(2, "0")}</span>
                  <div className="grow">
                    <div className="clip-title">{clip.title}</div>
                    <div className="clip-times">{clipTimecodes(clip)}</div>
                  </div>
                  <Badge tone="accent">{formatDuration(clip.total_duration_sec)}</Badge>
                  {clip.rendered_media_id && (
                    <button type="button" className="btn btn-sm" onClick={() => onSelectMedia(clip.rendered_media_id!)}>
                      <Play aria-hidden /> Abrir
                    </button>
                  )}
                  <button
                    type="button"
                    className="btn btn-primary btn-sm"
                    disabled={renderingKey !== null}
                    onClick={() => handleRender(activeRun.id, clip.id)}
                  >
                    <BusyIcon busy={renderingKey === clip.id} icon={Scissors} size={13} />
                    {clip.rendered_media_id ? "Re-renderizar" : "Renderizar"}
                  </button>
                  <MoreButton items={() => clipMenu(activeRun, clip, i)} />
                </header>

                <div className="grid-4 clip-scores">
                  <Meter label="Aderência" value={clip.scores.match_description} />
                  <Meter label="Visual" value={clip.scores.visual_interest} />
                  <Meter label="Coerência" value={clip.scores.narrative_coherence} />
                  <Meter label="Gancho" value={clip.scores.hook_strength} />
                </div>

                <div className="clip-notes">
                  {clip.visual_evidence && (
                    <p>
                      <Eye size={13} className="text-info" aria-hidden />
                      <span>
                        <span className="muted">O que a IA viu:</span> {clip.visual_evidence}
                      </span>
                    </p>
                  )}
                  {clip.hook_first_5s && (
                    <p>
                      <Zap size={13} className="text-warning" aria-hidden />
                      <span>
                        <span className="muted">Gancho (5s):</span> {clip.hook_first_5s}
                      </span>
                    </p>
                  )}
                  {clip.why_interesting && (
                    <p>
                      <Lightbulb size={13} className="text-magenta" aria-hidden />
                      <span>
                        <span className="muted">Por que funciona:</span> {clip.why_interesting}
                      </span>
                    </p>
                  )}
                </div>

                <div className="arc">
                  <div className="arc-step" data-step="1">
                    <span className="arc-label">1 · início</span>
                    {clip.narrative_arc.beginning || "—"}
                  </div>
                  <div className="arc-step" data-step="2">
                    <span className="arc-label">2 · meio</span>
                    {clip.narrative_arc.middle || "—"}
                  </div>
                  <div className="arc-step" data-step="3">
                    <span className="arc-label">3 · fim</span>
                    {clip.narrative_arc.end || "—"}
                  </div>
                </div>
              </article>
            ))}
          </div>
        )}
      </Panel>
    </>
  );
};
