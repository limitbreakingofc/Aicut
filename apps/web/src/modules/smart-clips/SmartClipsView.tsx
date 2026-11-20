import React, { useCallback, useEffect, useMemo, useState } from "react";
import {
  Captions,
  CircleCheck,
  Copy,
  Crop,
  Eye,
  Film,
  Layers,
  Lightbulb,
  Play,
  Scissors,
  Settings,
  Sparkles,
  Trash2,
  Zap,
} from "lucide-react";
import { AicutActionError, runAction } from "../../lib/client";
import { formatDuration, formatRelativeDate } from "../../lib/format";
import type {
  ClipsConfig,
  JobRecord,
  MediaAsset,
  MediaContextInfo,
  NotifyFn,
  RouteId,
  SmartClipPlan,
  SmartClipRun,
  ToolStatus,
} from "../../lib/types";
import { copyToClipboard } from "../../lib/ui";
import type { SettingsTabId, SmartClipsTabId } from "../../app/navigation";
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
  Panel,
  Tabs,
  useContextMenu,
  type MenuItems,
} from "../../components";
import { FramingStudioView } from "./FramingStudioView";
import "./smart-clips.css";

interface SmartClipsViewProps {
  activeTab: SmartClipsTabId;
  onTabChange: (tab: SmartClipsTabId) => void;
  media: MediaAsset[];
  tools: ToolStatus[];
  jobs: JobRecord[];
  preselectedMediaId?: string | null;
  onNavigate: (route: RouteId) => void;
  onOpenSettingsTab: (tab: SettingsTabId) => void;
  onSelectMedia: (mediaId: string) => void;
  onOpenInMotionStudio?: (seed: {
    runId: string;
    clipId: string;
    mediaId: string;
    startSec: number;
    endSec: number;
    title: string;
  }) => void;
  onNotify: NotifyFn;
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
type WebcamCorner = "auto_face" | "bottom_right" | "bottom_left" | "top_right" | "top_left" | "center";
type ContentFitMode = "smart_focus" | "fill" | "fit_blur";

const SHORTS_LAYOUTS: Array<{ id: ShortsLayout; short: string; label: string; desc: string }> = [
  {
    id: "template_split",
    short: "Moldura SMOKING33",
    label: "Moldura Shorts (SMOKING33) · Câmera em cima + conteúdo em baixo",
    desc: "Moldura 9:16 com borda dourada: webcam limpa em cima e IA Diretor no conteúdo em baixo.",
  },
  {
    id: "gaming_split",
    short: "Split limpo",
    label: "Split 9:16 sem moldura · Câmera em cima + conteúdo em baixo",
    desc: "Webcam sem bordas em cima e conteúdo dinâmico em baixo, sem moldura.",
  },
  {
    id: "fit_blur",
    short: "Tela inteira + blur",
    label: "Shorts 9:16 · Tela inteira + blur",
    desc: "Mantém o 16:9 completo sem cortar laterais sobre fundo desfocado.",
  },
  {
    id: "center_crop",
    short: "Foco central",
    label: "Shorts 9:16 · Foco central",
    desc: "Enquadramento vertical cheio focado no centro.",
  },
  {
    id: "original",
    short: "Original 16:9",
    label: "Original 16:9 (horizontal)",
    desc: "Mantém a proporção original do vídeo sem crop vertical.",
  },
];

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
  activeTab,
  onTabChange,
  media,
  tools,
  jobs,
  preselectedMediaId,
  onNavigate,
  onOpenSettingsTab,
  onSelectMedia,
  onOpenInMotionStudio,
  onNotify,
  onRefreshMedia,
}) => {
  const menu = useContextMenu();
  const videoMedia = useMemo(() => media.filter((m) => m.kind === "video" || m.kind === "video_only"), [media]);

  const [selectedMediaId, setSelectedMediaId] = useState<string>(preselectedMediaId ?? videoMedia[0]?.id ?? "");
  const [description, setDescription] = useState<string>(DESCRIPTION_PRESETS[0].text);
  const [maxDurationSec, setMaxDurationSec] = useState<number>(180);
  const [maxClips, setMaxClips] = useState<number>(5);
  const [allowJumpCuts, setAllowJumpCuts] = useState<boolean>(false);

  // Saved global AI config (edited exclusively in Configurações → IA & Chaves)
  const [config, setConfig] = useState<ClipsConfig | null>(null);
  const [runs, setRuns] = useState<SmartClipRun[]>([]);
  const [activeRunId, setActiveRunId] = useState<string | null>(null);
  const [mediaContext, setMediaContext] = useState<MediaContextInfo | null>(null);

  // 9:16 export format preferences (configured in the Enquadramento tab)
  const [layout, setLayout] = useState<ShortsLayout>("template_split");
  const [webcamCorner, setWebcamCorner] = useState<WebcamCorner>("auto_face");
  const [contentFit, setContentFit] = useState<ContentFitMode>("smart_focus");
  const [faceTracking, setFaceTracking] = useState<boolean>(true);
  const [faceZoom, setFaceZoom] = useState<number>(1.0);
  const [borderTrimPct, setBorderTrimPct] = useState<number>(0.045);
  const [customTemplatePath, setCustomTemplatePath] = useState<string>("");
  const [burnSubtitles, setBurnSubtitles] = useState<boolean>(true);

  const [analyzing, setAnalyzing] = useState<boolean>(false);
  const [renderingKey, setRenderingKey] = useState<string | null>(null);

  const ffmpegTool = tools.find((t) => t.id === "ffmpeg");

  useEffect(() => {
    if (preselectedMediaId) {
      setSelectedMediaId(preselectedMediaId);
    } else if (!selectedMediaId && videoMedia.length > 0) {
      setSelectedMediaId(videoMedia[0].id);
    }
  }, [preselectedMediaId, selectedMediaId, videoMedia]);

  const loadMediaContext = useCallback(async (mediaId: string) => {
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
  }, []);

  const loadAll = useCallback(async () => {
    try {
      const [cfgRes, runsRes] = await Promise.all([
        runAction<ClipsConfig>("clips.config_get"),
        runAction<{ runs: SmartClipRun[] }>("clips.runs_list"),
      ]);
      setConfig(cfgRes.data);
      setCustomTemplatePath(cfgRes.data.template_path ?? "");
      const rList = runsRes.data.runs ?? [];
      setRuns(rList);
      setActiveRunId((curr) => (curr && rList.some((r) => r.id === curr) ? curr : (rList[0]?.id ?? null)));
    } catch {
      // ignore if offline
    }
  }, []);

  useEffect(() => {
    void loadAll();
  }, [loadAll]);

  useEffect(() => {
    void loadMediaContext(selectedMediaId);
  }, [selectedMediaId, media.length, loadMediaContext]);

  const activeClipJobs = jobs.filter(
    (j) => (j.kind === "clips.analyze" || j.kind === "clips.render") && (j.status === "queued" || j.status === "running"),
  );
  const finishedCount = jobs.filter(
    (j) => (j.kind === "clips.analyze" || j.kind === "clips.render") && j.status === "completed",
  ).length;

  useEffect(() => {
    if (finishedCount > 0) {
      void loadAll();
      void onRefreshMedia();
    }
  }, [finishedCount, loadAll, onRefreshMedia]);

  const activeProvider = config?.default_provider === "local_free" ? "local_free" : "gemini";
  const providerReady = activeProvider === "gemini" ? Boolean(config?.gemini_configured) : true;
  const activeRun = runs.find((r) => r.id === activeRunId) ?? runs[0] ?? null;
  const resolvedTranscript = mediaContext?.resolved_transcript ?? null;
  const relatedCount = mediaContext?.related_assets.length ?? 0;
  const currentLayout = SHORTS_LAYOUTS.find((l) => l.id === layout) ?? SHORTS_LAYOUTS[0];
  const isSplit = layout === "template_split" || layout === "gaming_split";

  async function saveTemplatePath() {
    try {
      await runAction("clips.config_set", { template_path: customTemplatePath.trim() });
      await loadAll();
      onNotify(
        "success",
        "Moldura atualizada",
        customTemplatePath.trim() ? "Moldura PNG personalizada salva." : "Restaurada moldura padrão SMOKING33.",
      );
    } catch (e) {
      onNotify("error", "Erro ao salvar moldura", (e as Error).message);
    }
  }

  async function handleStartAnalyze() {
    if (!selectedMediaId) {
      onNotify("warning", "Selecione um vídeo", "Escolha um vídeo da Biblioteca para analisar.");
      return;
    }
    if (!description.trim()) {
      onNotify("warning", "Descreva os cortes", "Informe o que a IA deve procurar no vídeo.");
      return;
    }
    if (activeProvider === "gemini" && !config?.gemini_configured) {
      onNotify("warning", "Configure sua IA primeiro", "Adicione sua Gemini API Key em Configurações → IA & Chaves.");
      onOpenSettingsTab("ai");
      return;
    }
    setAnalyzing(true);
    try {
      const res = await runAction<{ job_id: string }>("clips.analyze", {
        media_id: selectedMediaId,
        description: description.trim(),
        provider: activeProvider,
        model_id: activeProvider === "local_free" ? config?.local_vision_model || undefined : undefined,
        max_duration_sec: maxDurationSec,
        max_clips: maxClips,
        allow_jump_cuts: allowJumpCuts,
        wait: false,
      });
      onNotify("success", "Análise iniciada", `Acompanhe o progresso (${res.data.job_id.slice(0, 8)}).`);
      onTabChange("results");
    } catch (e) {
      const msg = e instanceof AicutActionError ? e.message : (e as Error).message;
      onNotify("error", "Falha ao iniciar análise", msg);
    } finally {
      setAnalyzing(false);
    }
  }

  async function handleRender(runId: string, clipId?: string) {
    const key = clipId ?? `all:${runId}`;
    setRenderingKey(key);
    try {
      const res = await runAction<{ job_id: string }>("clips.render", {
        run_id: runId,
        clip_id: clipId,
        layout,
        webcam_corner: webcamCorner,
        content_fit: contentFit,
        face_tracking: faceTracking,
        face_zoom: faceZoom,
        border_trim_pct: borderTrimPct,
        template_path: layout === "template_split" && customTemplatePath.trim() ? customTemplatePath.trim() : undefined,
        burn_subtitles: burnSubtitles,
        wait: false,
      });
      onNotify("success", "Renderização iniciada", `Shorts serão salvos na Biblioteca (${res.data.job_id.slice(0, 8)}).`);
    } catch (e) {
      const msg = e instanceof AicutActionError ? e.message : (e as Error).message;
      onNotify("error", "Erro ao renderizar", msg);
    } finally {
      setRenderingKey(null);
    }
  }

  async function handleDeleteRun(runId: string) {
    try {
      await runAction("clips.run_delete", { run_id: runId });
      await loadAll();
      onNotify("info", "Análise removida", "Plano de cortes excluído do histórico.");
    } catch (e) {
      onNotify("error", "Erro ao remover análise", (e as Error).message);
    }
  }

  const clipMenu = (run: SmartClipRun, clip: SmartClipPlan, idx: number): MenuItems => [
    { type: "label", label: `#${String(idx + 1).padStart(2, "0")} · ${clip.title}` },
    {
      label: clip.rendered_media_id ? "Re-renderizar Short" : "Renderizar Short",
      icon: Scissors,
      disabled: renderingKey !== null,
      onSelect: () => void handleRender(run.id, clip.id),
    },
    onOpenInMotionStudio &&
      clip.segments.length > 0 && {
        label: "Editar no Motion Studio",
        icon: Layers,
        onSelect: () =>
          onOpenInMotionStudio({
            runId: run.id,
            clipId: clip.id,
            mediaId: run.media_id,
            startSec: clip.segments[0].start_sec,
            endSec: clip.segments[clip.segments.length - 1].end_sec,
            title: clip.title,
          }),
      },
    clip.rendered_media_id && {
      label: "Abrir Short na Biblioteca",
      icon: Play,
      onSelect: () => onSelectMedia(clip.rendered_media_id!),
    },
    { type: "separator" },
    { label: "Copiar título", icon: Copy, onSelect: () => void copyToClipboard(clip.title, onNotify, "Título") },
    { label: "Copiar timecodes", icon: Copy, onSelect: () => void copyToClipboard(clipTimecodes(clip), onNotify, "Timecodes") },
  ];

  const totalClipsCount = runs.reduce((acc, r) => acc + r.clips.length, 0);

  return (
    <div className="stack">
      <Tabs
        ariaLabel="Categorias de Cortes Shorts"
        value={activeTab}
        onChange={onTabChange}
        items={[
          { id: "generate", label: "Gerar Cortes", icon: Sparkles },
          { id: "results", label: "Resultados", icon: Scissors, count: totalClipsCount || undefined },
          { id: "framing", label: "Enquadramento", icon: Crop },
        ]}
      />

      {!ffmpegTool?.installed && (
        <Callout
          tone="warning"
          title="FFmpeg necessário"
          actions={
            <button type="button" className="btn btn-primary btn-sm" onClick={() => onOpenSettingsTab("tools")}>
              Instalar em Ferramentas
            </button>
          }
        >
          Instale o FFmpeg em Configurações → Ferramentas para analisar frames e renderizar Shorts.
        </Callout>
      )}

      <ActiveJobsPanel
        jobs={activeClipJobs}
        title="Cortes IA em andamento"
        tone="accent"
        onViewAll={() => onOpenSettingsTab("jobs")}
      />

      {/* TAB 1: GERAR CORTES (Briefing + Disparo — sem configurações duplicadas!) */}
      {activeTab === "generate" && (
        <Panel
          title="Vídeo & briefing editorial"
          icon={Film}
          actions={
            <button
              type="button"
              className="btn btn-ghost btn-sm"
              onClick={() => onOpenSettingsTab("ai")}
              title="Alterar motor de IA em Configurações → IA & Chaves"
            >
              <Settings size={13} aria-hidden />
              {activeProvider === "gemini"
                ? config?.gemini_configured
                  ? "Gemini API · pronto"
                  : "Configurar Gemini API"
                : "Local grátis · GPU"}
            </button>
          }
          footer={
            <button
              type="button"
              className="btn btn-primary btn-block"
              disabled={analyzing || !selectedMediaId}
              onClick={handleStartAnalyze}
            >
              <BusyIcon busy={analyzing} icon={Sparkles} />
              Analisar vídeo e gerar cortes
            </button>
          }
        >
          {!providerReady && (
            <Callout
              tone="warning"
              title="Chave Gemini API não configurada"
              actions={
                <button type="button" className="btn btn-primary btn-sm" onClick={() => onOpenSettingsTab("ai")}>
                  Configurar em IA & Chaves
                </button>
              }
            >
              Adicione sua chave Gemini ou selecione o modo Local Grátis em Configurações → IA & Chaves.
            </Callout>
          )}

          <Field label="Vídeo / live da Biblioteca" htmlFor="sc-media">
            {videoMedia.length === 0 ? (
              <Callout
                actions={
                  <button type="button" className="btn btn-sm" onClick={() => onNavigate("ingest")}>
                    Importar vídeo
                  </button>
                }
              >
                Nenhum vídeo na Biblioteca ainda. Baixe um link ou importe um arquivo local.
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
                  ? `Transcrição vinculada · ${resolvedTranscript.segment_count} falas · ${resolvedTranscript.language.toUpperCase()}`
                  : "Sem transcrição vinculada (opcional, mas melhora o corte)"
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
                <span>{relatedCount} arquivo(s) vinculado(s)</span>
              </span>
            </Callout>
          )}

          <Field label="O que a IA deve procurar nos cortes?" htmlFor="sc-desc">
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
              placeholder="Ex.: Ache os momentos mais engraçados, jogadas marcantes com reação na webcam ou explicações completas..."
            />
          </Field>

          <div className="grid-2">
            <Field label="Duração máxima por corte" htmlFor="sc-maxdur">
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
            <Field label="Quantidade máxima" htmlFor="sc-maxclips">
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
            description={`Pula pausas longas mantendo início–meio–fim em até ${maxDurationSec}s.`}
          />
        </Panel>
      )}

      {/* TAB 2: RESULTADOS & EXPORTAÇÃO */}
      {activeTab === "results" && (
        <Panel
          title="Cortes encontrados"
          icon={Scissors}
          actions={
            <div className="row wrap gap-xs">
              <button
                type="button"
                className="btn btn-sm"
                onClick={() => onTabChange("framing")}
                title="Ajustar formato 9:16 e áreas da webcam"
              >
                <Crop size={13} aria-hidden /> {currentLayout.short}
              </button>
              {runs.length > 0 && (
                <>
                  <select
                    className="select select-sm select-auto run-select"
                    aria-label="Análise"
                    value={activeRun?.id ?? ""}
                    onChange={(e) => setActiveRunId(e.target.value)}
                  >
                    {runs.map((r) => (
                      <option key={r.id} value={r.id}>
                        {r.media_title} · {r.clips.length} corte(s) ({formatRelativeDate(r.created_at)})
                      </option>
                    ))}
                  </select>
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
              )}
            </div>
          }
        >
          {!activeRun || activeRun.clips.length === 0 ? (
            <EmptyState
              compact
              icon={Scissors}
              title="nenhuma análise concluída"
              description="Vá na aba Gerar Cortes, escolha um vídeo e clique em Analisar vídeo."
              actions={
                <button type="button" className="btn btn-primary btn-sm" onClick={() => onTabChange("generate")}>
                  Ir para Gerar Cortes
                </button>
              }
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
                    {onOpenInMotionStudio && clip.segments.length > 0 && (
                      <button
                        type="button"
                        className="btn btn-sm"
                        title="Editar layouts dinâmicos e animações no Motion Studio"
                        onClick={() =>
                          onOpenInMotionStudio({
                            runId: activeRun.id,
                            clipId: clip.id,
                            mediaId: activeRun.media_id,
                            startSec: clip.segments[0].start_sec,
                            endSec: clip.segments[clip.segments.length - 1].end_sec,
                            title: clip.title,
                          })
                        }
                      >
                        <Layers size={13} aria-hidden /> Motion IA
                      </button>
                    )}
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
      )}

      {/* TAB 3: ENQUADRAMENTO (Fonte única de verdade para formato 9:16 + área da webcam & conteúdo) */}
      {activeTab === "framing" && (
        <div className="stack">
          <Panel title="Estilo de exportação 9:16" icon={CircleCheck}>
            <ChoiceCards
              ariaLabel="Formato do Short"
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
                    <option value="auto_face">Automático · IA detecta webcam, remove bordas e segue rosto</option>
                    <option value="bottom_right">Canto inferior direito</option>
                    <option value="bottom_left">Canto inferior esquerdo</option>
                    <option value="top_right">Canto superior direito</option>
                    <option value="top_left">Canto superior esquerdo</option>
                    <option value="center">Centro / Tela cheia</option>
                  </select>
                </Field>
                <Field label="Conteúdo (janela de baixo)" htmlFor="sc-fit">
                  <select
                    id="sc-fit"
                    className="select"
                    value={contentFit}
                    onChange={(e) => setContentFit(e.target.value as ContentFitMode)}
                  >
                    <option value="smart_focus">IA Diretor · Quem fala ou ação em destaque (Recomendado)</option>
                    <option value="fill">Centro fixo (sem bordas)</option>
                    <option value="fit_blur">Tela 16:9 inteira + blur</option>
                  </select>
                </Field>
              </div>
            )}

            {(isSplit || layout === "center_crop") && (
              <div className="grid-2">
                <Field label="Zoom na webcam (topo)" htmlFor="sc-zoom">
                  <select
                    id="sc-zoom"
                    className="select"
                    value={faceZoom}
                    onChange={(e) => setFaceZoom(Number(e.target.value))}
                    disabled={!faceTracking}
                  >
                    <option value={1.0}>Webcam inteira sem bordas · Sem zoom no rosto (Recomendado)</option>
                    <option value={1.15}>Zoom leve no rosto (1.15×)</option>
                    <option value={1.3}>Close no rosto (1.30×)</option>
                  </select>
                </Field>
                <Field label="Margem anti-borda (automático)" htmlFor="sc-trim">
                  <select
                    id="sc-trim"
                    className="select"
                    value={borderTrimPct}
                    onChange={(e) => setBorderTrimPct(Number(e.target.value))}
                    disabled={!faceTracking}
                  >
                    <option value={0.03}>Leve (3% — sem moldura)</option>
                    <option value={0.045}>Padrão (4.5% — bordas OBS)</option>
                    <option value={0.08}>Forte (8% — molduras grossas)</option>
                  </select>
                </Field>
              </div>
            )}

            <div className="row wrap gap-md">
              {(isSplit || layout === "center_crop") && (
                <Checkbox
                  checked={faceTracking}
                  onChange={setFaceTracking}
                  label="IA Cameraman na webcam (remove bordas e acompanha o rosto)"
                />
              )}
              <Checkbox
                checked={burnSubtitles}
                onChange={setBurnSubtitles}
                label="Queimar legendas karaokê sincronizadas"
              />
              {layout === "template_split" && (
                <div className="row grow template-row">
                  <input
                    type="text"
                    className="input input-sm"
                    aria-label="Caminho da moldura PNG"
                    value={customTemplatePath}
                    onChange={(e) => setCustomTemplatePath(e.target.value)}
                    placeholder="moldura padrão SMOKING33 (ou caminho de outro PNG 9:16 transparente)"
                  />
                  <button type="button" className="btn btn-sm" onClick={saveTemplatePath}>
                    Salvar PNG
                  </button>
                </div>
              )}
            </div>
          </Panel>

          <FramingStudioView
            media={media}
            tools={tools}
            preselectedMediaId={selectedMediaId}
            onNavigate={onNavigate}
            onNotify={onNotify}
          />
        </div>
      )}
    </div>
  );
};
