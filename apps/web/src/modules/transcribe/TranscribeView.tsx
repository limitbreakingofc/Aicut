import React, { useEffect, useMemo, useState } from "react";
import {
  ArrowRight,
  Award,
  Captions,
  Check,
  CircleCheck,
  Copy,
  Cpu,
  Download,
  FileCode2,
  FileText,
  HardDrive,
  Layers,
  ListVideo,
  Mic,
  Search,
  Sparkles,
  Trash2,
  TriangleAlert,
} from "lucide-react";
import { AicutActionError, runAction } from "../../lib/client";
import { formatBytes, formatDuration, formatRelativeDate } from "../../lib/format";
import type {
  ExportFormatInfo,
  HardwareProfile,
  JobRecord,
  MediaAsset,
  NotifyFn,
  RouteId,
  ToolStatus,
  TranscriptRecord,
  TranscriptSegment,
  WhisperModelStatus,
} from "../../lib/types";
import { copyToClipboard, normalizeSearch, storage } from "../../lib/ui";
import {
  ActiveJobsPanel,
  Badge,
  BusyIcon,
  Callout,
  Checkbox,
  ChoiceCards,
  EmptyState,
  Field,
  MoreButton,
  Panel,
  Switch,
  Tabs,
  useContextMenu,
  type MenuItems,
} from "../../components";
import "./transcribe.css";

interface TranscribeViewProps {
  media: MediaAsset[];
  tools: ToolStatus[];
  jobs: JobRecord[];
  hardware: HardwareProfile | null;
  preselectedMediaId?: string | null;
  onNavigate: (route: RouteId) => void;
  onNotify: NotifyFn;
  onRefreshTools: () => Promise<void>;
  onRefreshMedia: () => Promise<void>;
}

type TabId = "new" | "results" | "models";

const LANGUAGES = [
  { code: "auto", label: "Detectar idioma automaticamente" },
  { code: "pt", label: "Português (pt)" },
  { code: "en", label: "Inglês (en)" },
  { code: "es", label: "Espanhol (es)" },
  { code: "fr", label: "Francês (fr)" },
  { code: "de", label: "Alemão (de)" },
  { code: "it", label: "Italiano (it)" },
  { code: "ja", label: "Japonês (ja)" },
];

const FORMAT_GROUPS = [
  { category: "subtitle", label: "Legendas", icon: Captions },
  { category: "timeline", label: "Timelines", icon: FileCode2 },
  { category: "transcript", label: "Texto / IA", icon: FileText },
] as const;

const timecode = (seg: TranscriptSegment) => `${formatDuration(seg.start_ms / 1000)} → ${formatDuration(seg.end_ms / 1000)}`;

export const TranscribeView: React.FC<TranscribeViewProps> = ({
  media,
  tools,
  jobs,
  hardware,
  preselectedMediaId,
  onNavigate,
  onNotify,
  onRefreshTools,
  onRefreshMedia,
}) => {
  const menu = useContextMenu();
  const playableMedia = media.filter((m) => m.kind === "video" || m.kind === "video_only" || m.kind === "audio");

  const [tab, setTab] = useState<TabId>("new");
  const [selectedMediaId, setSelectedMediaId] = useState<string>(preselectedMediaId ?? playableMedia[0]?.id ?? "");
  const [models, setModels] = useState<WhisperModelStatus[]>([]);
  // Globally active model version used for transcription
  const [selectedModel, setSelectedModel] = useState<string>(() => storage.get("aicut_whisper_model") ?? "");

  const [formats, setFormats] = useState<ExportFormatInfo[]>([]);
  const [transcripts, setTranscripts] = useState<TranscriptRecord[]>([]);
  const [activeTranscriptId, setActiveTranscriptId] = useState<string | null>(null);

  const [language, setLanguage] = useState<string>("auto");
  const [translateToEnglish, setTranslateToEnglish] = useState(false);
  const [wordTimestamps, setWordTimestamps] = useState(true);
  const [maxSegmentChars, setMaxSegmentChars] = useState<string>("");
  const [initialPrompt, setInitialPrompt] = useState<string>("");
  const [selectedFormats, setSelectedFormats] = useState<string[]>(["srt", "fcpxml", "json"]);

  const [starting, setStarting] = useState(false);
  const [exportingFmt, setExportingFmt] = useState<string | null>(null);
  const [searchSeg, setSearchSeg] = useState("");

  const whisperTool = tools.find((t) => t.id === "whisper");
  const ffmpegTool = tools.find((t) => t.id === "ffmpeg");

  useEffect(() => {
    if (preselectedMediaId) {
      setSelectedMediaId(preselectedMediaId);
    } else if (!selectedMediaId && playableMedia.length > 0) {
      setSelectedMediaId(playableMedia[0].id);
    }
  }, [preselectedMediaId, playableMedia.length]);

  const chooseActiveModel = (modelId: string) => {
    setSelectedModel(modelId);
    storage.set("aicut_whisper_model", modelId);
    void runAction("system.set_setting", {
      key: "default_whisper_model",
      value: modelId,
    }).catch(() => {});
  };

  const loadAll = async () => {
    try {
      const [mRes, fRes, tRes] = await Promise.all([
        runAction<{
          models: WhisperModelStatus[];
          recommended_model_id: string;
          total_installed_bytes: number;
        }>("transcribe.models_list"),
        runAction<{ formats: ExportFormatInfo[] }>("transcribe.formats_list"),
        runAction<{ transcripts: TranscriptRecord[] }>("transcribe.list"),
      ]);
      const mList = mRes.data.models ?? [];
      setModels(mList);
      setFormats(fRes.data.formats ?? []);
      const tList = tRes.data.transcripts ?? [];
      setTranscripts(tList);
      if (tList.length > 0 && !activeTranscriptId) {
        setActiveTranscriptId(tList[0].id);
      }

      // Ensure active model points to a valid installed or recommended model
      setSelectedModel((curr) => {
        if (curr && mList.some((m) => m.id === curr && m.installed)) return curr;
        const firstInstalled = mList.find((m) => m.installed);
        if (firstInstalled) return firstInstalled.id;
        return mRes.data.recommended_model_id || mList[0]?.id || "large-v3-turbo-q5_0";
      });
    } catch {
      // ignore if offline
    }
  };

  useEffect(() => {
    void loadAll();
  }, []);

  const activeTranscribeJobs = jobs.filter(
    (j) =>
      (j.kind === "transcribe.run" || j.kind === "transcribe.model_download") &&
      (j.status === "queued" || j.status === "running"),
  );
  const completedTranscribeCount = jobs.filter((j) => j.kind === "transcribe.run" && j.status === "completed").length;

  useEffect(() => {
    if (completedTranscribeCount > 0) {
      void loadAll();
      void onRefreshMedia();
    }
  }, [completedTranscribeCount]);

  const installedModels = useMemo(() => models.filter((m) => m.installed), [models]);

  function toggleFormat(fmtId: string) {
    setSelectedFormats((prev) => (prev.includes(fmtId) ? prev.filter((x) => x !== fmtId) : [...prev, fmtId]));
  }

  async function handleStartTranscription(e: React.FormEvent) {
    e.preventDefault();
    if (!selectedMediaId) {
      onNotify("warning", "Selecione um vídeo ou áudio", "Escolha um arquivo da Biblioteca para transcrever.");
      return;
    }
    setStarting(true);
    try {
      const res = await runAction<{ job_id: string }>("transcribe.run", {
        media_id: selectedMediaId,
        model_id: selectedModel,
        language,
        translate_to_english: translateToEnglish,
        word_timestamps: wordTimestamps,
        max_segment_chars: maxSegmentChars.trim() ? Number(maxSegmentChars) : null,
        initial_prompt: initialPrompt.trim() || null,
        export_formats: selectedFormats,
        wait: false,
      });
      onNotify(
        "success",
        "Transcrição iniciada",
        `Job ${res.data.job_id.slice(0, 8)} usando ${chosenModelObj?.display_name ?? selectedModel}.`,
      );
    } catch (e) {
      const msg = e instanceof AicutActionError && e.detail.hint ? `${e.message} — ${e.detail.hint}` : (e as Error).message;
      onNotify("error", "Não foi possível iniciar transcrição", msg);
    } finally {
      setStarting(false);
    }
  }

  async function handleQuickExport(transcriptId: string, formatId: string) {
    setExportingFmt(`${transcriptId}:${formatId}`);
    try {
      const res = await runAction("transcribe.export", {
        transcript_id: transcriptId,
        formats: [formatId],
      });
      await onRefreshMedia();
      onNotify("success", "Salvo na Biblioteca!", res.summary);
    } catch (e) {
      onNotify("error", "Erro ao exportar formato", (e as Error).message);
    } finally {
      setExportingFmt(null);
    }
  }

  async function handleDeleteTranscript(id: string) {
    try {
      await runAction("transcribe.delete", { id });
      setTranscripts((prev) => prev.filter((t) => t.id !== id));
      if (activeTranscriptId === id) setActiveTranscriptId(null);
      onNotify("info", "Transcrição removida", "Arquivos já exportados na Biblioteca foram mantidos.");
    } catch (e) {
      onNotify("error", "Erro ao remover transcrição", (e as Error).message);
    }
  }

  const activeTranscript = transcripts.find((t) => t.id === activeTranscriptId) ?? transcripts[0] ?? null;
  const chosenModelObj = models.find((m) => m.id === selectedModel);

  const gpuSummary = hardware
    ? hardware.gpus.length > 0
      ? `${hardware.gpus[0].name}${hardware.gpus[0].vram_mb ? ` (${(hardware.gpus[0].vram_mb / 1024).toFixed(1)} GB VRAM)` : ""}${hardware.cuda_version ? ` · CUDA ${hardware.cuda_version}` : ""}`
      : `CPU ${hardware.cpu_brand} (${(hardware.total_memory_mb / 1024).toFixed(1)} GB RAM)`
    : "Detectando hardware...";

  const visibleSegments = useMemo(() => {
    if (!activeTranscript) return [];
    const q = normalizeSearch(searchSeg.trim());
    return q ? activeTranscript.segments.filter((s) => normalizeSearch(s.text).includes(q)) : activeTranscript.segments;
  }, [activeTranscript, searchSeg]);

  /* ---------- Context menus ---------- */
  const transcriptMenu = (t: TranscriptRecord): MenuItems => [
    { type: "label", label: t.media_title },
    { label: "Abrir transcrição", icon: FileText, onSelect: () => setActiveTranscriptId(t.id) },
    {
      label: "Copiar texto completo",
      icon: Copy,
      onSelect: () => void copyToClipboard(t.full_text || t.segments.map((s) => s.text).join(" "), onNotify, "Texto"),
    },
    { label: "Copiar ID", icon: Copy, onSelect: () => void copyToClipboard(t.id, onNotify, "ID da transcrição") },
    { type: "separator" },
    { type: "label", label: "Salvar na Biblioteca como" },
    ...formats.map((f) => ({
      label: f.name,
      icon: f.category === "subtitle" ? Captions : f.category === "timeline" ? FileCode2 : FileText,
      disabled: exportingFmt !== null,
      onSelect: () => void handleQuickExport(t.id, f.id),
    })),
    { type: "separator" },
    { label: "Excluir transcrição", icon: Trash2, danger: true, onSelect: () => void handleDeleteTranscript(t.id) },
  ];

  const segmentMenu = (seg: TranscriptSegment): MenuItems => [
    { type: "label", label: timecode(seg) },
    { label: "Copiar fala", icon: Copy, onSelect: () => void copyToClipboard(seg.text, onNotify, "Fala") },
    {
      label: "Copiar com timecode",
      icon: Copy,
      onSelect: () => void copyToClipboard(`[${timecode(seg)}] ${seg.text}`, onNotify, "Fala com timecode"),
    },
    {
      label: "Copiar início (segundos)",
      icon: Copy,
      onSelect: () => void copyToClipboard((seg.start_ms / 1000).toFixed(2), onNotify, "Início"),
    },
  ];

  const engineMissing = (!whisperTool?.installed || !ffmpegTool?.installed) && tools.length > 0;

  const [upgradingWhisper, setUpgradingWhisper] = useState(false);

  const activeDeviceBadge = useMemo(() => {
    if (!whisperTool?.installed) return null;
    const backend = whisperTool.installed_backend;
    const gpuName = hardware?.gpus?.[0]?.name;
    if (backend && backend !== "CPU") {
      return {
        tone: "success" as const,
        label: gpuName ? `GPU: ${gpuName} (${backend})` : `GPU (${backend})`,
      };
    }
    if (whisperTool.gpu_backend_missing) {
      return {
        tone: "warning" as const,
        label: "CPU (fallback — sem backend GPU)",
      };
    }
    return {
      tone: "neutral" as const,
      label: "CPU",
    };
  }, [whisperTool, hardware]);

  async function handleInstallGpuWhisper() {
    setUpgradingWhisper(true);
    try {
      await runAction("tools.install", { tool_id: "whisper", version: null });
      await onRefreshTools();
      onNotify("success", "whisper.cpp atualizado", "Motor de transcrição com suporte a GPU instalado.");
    } catch (e) {
      onNotify("error", "Falha ao atualizar whisper.cpp", (e as Error).message);
    } finally {
      setUpgradingWhisper(false);
    }
  }

  return (
    <>
      <Tabs
        ariaLabel="Seções de transcrição"
        value={tab}
        onChange={setTab}
        items={[
          { id: "new", label: "Transcrever", icon: Mic },
          { id: "results", label: "Transcrições", icon: ListVideo, count: transcripts.length },
        ]}
      />

      {engineMissing && (
        <Callout
          tone="warning"
          title="Motor de IA local pendente"
          actions={
            <>
              <button type="button" className="btn btn-sm" onClick={() => onNavigate("downloads")}>
                Ferramentas
              </button>
            </>
          }
        >
          {!whisperTool?.installed
            ? `whisper.cpp não instalado. Build ideal detectada: ${whisperTool?.recommended_build ?? "auto"}.`
            : "FFmpeg necessário para extrair o áudio do vídeo."}
        </Callout>
      )}

      {whisperTool?.gpu_backend_missing && (
        <Callout
          tone="warning"
          title="whisper-cli está rodando apenas na CPU (sem backend GPU)"
          actions={
            <>
              <button
                type="button"
                className="btn btn-sm btn-primary"
                disabled={upgradingWhisper}
                onClick={() => void handleInstallGpuWhisper()}
              >
                <BusyIcon busy={upgradingWhisper} icon={Download} />
                Atualizar whisper.cpp para GPU
              </button>
              <button type="button" className="btn btn-sm" onClick={() => onNavigate("downloads")}>
                Downloads
              </button>
            </>
          }
        >
          GPU detectada ({gpuSummary}), mas o binário atual em <code>bin/whisper/</code> não contém{" "}
          <code>libggml-cuda.so</code> nem <code>libggml-vulkan.so</code>. Clique para instalar o build com GPU.
        </Callout>
      )}

      {/* ================================================================ */}
      {/* NEW TRANSCRIPTION                                                */}
      {/* ================================================================ */}
      {tab === "new" && (
        <form onSubmit={handleStartTranscription} className="split">
          <div className="split-main">
            <Panel
              title="Transcrever vídeo / áudio"
              icon={Mic}
              actions={
                activeDeviceBadge ? (
                  <Badge tone={activeDeviceBadge.tone}>{activeDeviceBadge.label}</Badge>
                ) : undefined
              }
              footer={
                <>
                  <span className="secondary text-sm">
                    {chosenModelObj && !chosenModelObj.installed
                      ? "Baixe uma versão de modelo compatível na aba Modelos antes de iniciar."
                      : `${selectedFormats.length} formato(s) serão salvos automaticamente na Biblioteca.`}
                  </span>
                  <button
                    type="submit"
                    className="btn btn-primary"
                    disabled={starting || !selectedMediaId || !whisperTool?.installed || !chosenModelObj?.installed}
                  >
                    <BusyIcon busy={starting} icon={Sparkles} />
                    Transcrever com IA local
                  </button>
                </>
              }
            >
              <div className="panel-section">
                <div className="grid-2">
                  <Field label="Vídeo ou áudio da Biblioteca" htmlFor="tr-media">
                    <select
                      id="tr-media"
                      className="select"
                      value={selectedMediaId}
                      onChange={(e) => setSelectedMediaId(e.target.value)}
                    >
                      {playableMedia.length === 0 ? (
                        <option value="">Nenhum vídeo ou áudio na Biblioteca ainda</option>
                      ) : (
                        playableMedia.map((m) => (
                          <option key={m.id} value={m.id}>
                            {m.title} ({m.extension.toUpperCase()}
                            {m.metadata.duration_sec ? ` · ${formatDuration(m.metadata.duration_sec)}` : ""})
                          </option>
                        ))
                      )}
                    </select>
                  </Field>
                  <Field label="Idioma do áudio" htmlFor="tr-lang">
                    <select id="tr-lang" className="select" value={language} onChange={(e) => setLanguage(e.target.value)}>
                      {LANGUAGES.map((l) => (
                        <option key={l.code} value={l.code}>
                          {l.label}
                        </option>
                      ))}
                    </select>
                  </Field>
                </div>
              </div>

              <div className="panel-section">
                <div className="section-label">Opções</div>
                <div className="grid-2">
                  <Field label="Máx. caracteres por bloco (opcional)" htmlFor="tr-maxchars">
                    <input
                      id="tr-maxchars"
                      type="number"
                      className="input"
                      placeholder="ex: 42 (vazio = auto)"
                      value={maxSegmentChars}
                      onChange={(e) => setMaxSegmentChars(e.target.value)}
                    />
                  </Field>
                  <Field label="Prompt de contexto (nomes próprios / termos)" htmlFor="tr-prompt">
                    <input
                      id="tr-prompt"
                      className="input"
                      placeholder="ex: Aicut, DaVinci Resolve, Rust..."
                      value={initialPrompt}
                      onChange={(e) => setInitialPrompt(e.target.value)}
                    />
                  </Field>
                </div>
                <div className="row wrap gap-lg">
                  <Checkbox
                    checked={wordTimestamps}
                    onChange={setWordTimestamps}
                    label="Timestamps por palavra"
                    description="Ideal para karaokê e cortes IA"
                  />
                  <Checkbox checked={translateToEnglish} onChange={setTranslateToEnglish} label="Traduzir fala para inglês" />
                </div>
              </div>

              <div className="panel-section">
                <div className="section-label">Salvar automaticamente na Biblioteca ao terminar</div>
                <div className="format-groups">
                  {FORMAT_GROUPS.map((group) => {
                    const list = formats.filter((f) => f.category === group.category);
                    if (list.length === 0) return null;
                    return (
                      <div key={group.category} className="format-group">
                        <span className="format-group-label">
                          <group.icon size={13} aria-hidden /> {group.label}
                        </span>
                        <div className="row wrap gap-xs">
                          {list.map((f) => {
                            const on = selectedFormats.includes(f.id);
                            return (
                              <button
                                key={f.id}
                                type="button"
                                className="chip"
                                aria-pressed={on}
                                title={f.description}
                                onClick={() => toggleFormat(f.id)}
                              >
                                {on && <Check aria-hidden />}
                                {f.name}
                              </button>
                            );
                          })}
                        </div>
                      </div>
                    );
                  })}
                </div>
              </div>
            </Panel>
          </div>

          <aside className="split-aside">
            <Panel
              title="Modelo ativo"
              icon={Cpu}
              actions={
                <button type="button" className="btn btn-ghost btn-xs" onClick={() => onNavigate("downloads")}>
                  gerenciar <ArrowRight aria-hidden />
                </button>
              }
            >
              <div className="muted text-xs">{gpuSummary}</div>
              {installedModels.length === 0 ? (
                <Callout
                  tone="warning"
                  title="Nenhum modelo instalado"
                  actions={
                    <button type="button" className="btn btn-primary btn-sm" onClick={() => onNavigate("downloads")}>
                      Ver modelos
                    </button>
                  }
                >
                  Baixe um modelo recomendado para a sua GPU.
                </Callout>
              ) : (
                <ChoiceCards
                  ariaLabel="Modelo ativo"
                  columns={1}
                  value={selectedModel}
                  onChange={chooseActiveModel}
                  options={installedModels.map((m) => ({
                    id: m.id,
                    title: m.display_name,
                    description: `${m.quality_tier} · ${m.speed_tier} · ${m.size_mb} MB`,
                  }))}
                />
              )}
            </Panel>

            <ActiveJobsPanel jobs={activeTranscribeJobs} onViewAll={() => onNavigate("jobs")} />
          </aside>
        </form>
      )}

      {/* ================================================================ */}
      {/* SAVED TRANSCRIPTIONS + UNIVERSAL EXPORTER                         */}
      {/* ================================================================ */}
      {tab === "results" &&
        (transcripts.length === 0 ? (
          <div className="panel">
            <EmptyState
              icon={FileText}
              title="nenhuma transcrição ainda"
              description="Escolha um vídeo na aba Nova transcrição para transcrever 100% offline."
              actions={
                <button type="button" className="btn btn-primary" onClick={() => setTab("new")}>
                  Nova transcrição
                </button>
              }
            />
          </div>
        ) : (
          <div className="transcripts-layout">
            <Panel title="Arquivos" icon={ListVideo} flush>
              <div className="transcript-list" role="listbox" aria-label="Transcrições">
                {transcripts.map((t) => (
                  <div
                    key={t.id}
                    role="option"
                    tabIndex={0}
                    aria-selected={activeTranscript?.id === t.id}
                    className="transcript-item hover-host"
                    onClick={() => setActiveTranscriptId(t.id)}
                    onKeyDown={(e) => e.key === "Enter" && setActiveTranscriptId(t.id)}
                    onContextMenu={(e) => menu.open(e, transcriptMenu(t))}
                  >
                    <div className="row row-between">
                      <span className="truncate strong text-sm">{t.media_title}</span>
                      <span className="row-actions">
                        <MoreButton items={() => transcriptMenu(t)} />
                      </span>
                    </div>
                    <div className="muted text-xs">
                      {t.model_id} · {t.language.toUpperCase()} · {t.segments.length} blocos
                    </div>
                    <div className="muted text-xs">{formatRelativeDate(t.created_at)}</div>
                  </div>
                ))}
              </div>
            </Panel>

            {activeTranscript && (
              <Panel
                title={activeTranscript.media_title}
                icon={FileText}
                actions={
                  <>
                    <div className="input-wrap seg-search">
                      <Search aria-hidden />
                      <input
                        className="input input-sm"
                        placeholder="buscar fala…"
                        aria-label="Buscar fala na transcrição"
                        value={searchSeg}
                        onChange={(e) => setSearchSeg(e.target.value)}
                      />
                    </div>
                    <MoreButton items={() => transcriptMenu(activeTranscript)} />
                  </>
                }
              >
                <div className="row wrap gap-xs">
                  <Badge tone="accent">{activeTranscript.model_id}</Badge>
                  <Badge>{activeTranscript.language.toUpperCase()}</Badge>
                  <Badge>{activeTranscript.segments.length} blocos</Badge>
                  {activeTranscript.duration_sec !== undefined && <Badge>{formatDuration(activeTranscript.duration_sec)}</Badge>}
                  <Badge>{formatRelativeDate(activeTranscript.created_at)}</Badge>
                </div>

                <div className="stack stack-sm">
                  <div className="section-label">Salvar outro formato na Biblioteca (instantâneo)</div>
                  <div className="row wrap gap-xs">
                    {formats.map((f) => {
                      const key = `${activeTranscript.id}:${f.id}`;
                      return (
                        <button
                          key={f.id}
                          type="button"
                          className="chip chip-sm"
                          title={f.description}
                          disabled={exportingFmt === key}
                          onClick={() => handleQuickExport(activeTranscript.id, f.id)}
                        >
                          <BusyIcon busy={exportingFmt === key} icon={Download} size={12} />
                          {f.name}
                        </button>
                      );
                    })}
                  </div>
                </div>

                <div className="term segments" aria-label="Falas da transcrição">
                  {visibleSegments.length === 0 ? (
                    <div className="term-empty">nenhuma fala corresponde a “{searchSeg}”</div>
                  ) : (
                    visibleSegments.map((seg) => (
                      <div key={seg.index} className="term-line" onContextMenu={(e) => menu.open(e, segmentMenu(seg))}>
                        <span className="term-time">[{timecode(seg)}]</span>
                        <span>{seg.text}</span>
                      </div>
                    ))
                  )}
                </div>
              </Panel>
            )}
          </div>
        ))}

      {/* ================================================================ */}
      {/* GPU-CURATED MODEL & VERSION MANAGER                              */}
      {/* ================================================================ */}
    </>
  );
};
