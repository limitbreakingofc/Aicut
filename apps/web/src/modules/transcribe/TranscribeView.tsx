import React, { useEffect, useMemo, useState } from "react";
import {
  AlertTriangle,
  ArrowRight,
  Award,
  Check,
  CheckCircle2,
  Cpu,
  Download,
  FileCode2,
  FileText,
  Film,
  HardDrive,
  Loader2,
  Mic,
  Play,
  Sparkles,
  Subtitles,
  Trash2,
} from "lucide-react";
import { AicutActionError, runAction } from "../../lib/client";
import { formatBytes, formatDuration, formatRelativeDate } from "../../lib/format";
import type {
  ExportFormatInfo,
  HardwareProfile,
  JobRecord,
  MediaAsset,
  RouteId,
  ToolStatus,
  TranscriptRecord,
  WhisperModelStatus,
} from "../../lib/types";

interface TranscribeViewProps {
  media: MediaAsset[];
  tools: ToolStatus[];
  jobs: JobRecord[];
  hardware: HardwareProfile | null;
  preselectedMediaId?: string | null;
  onNavigate: (route: RouteId) => void;
  onNotify: (level: "info" | "success" | "warning" | "error", title: string, message: string) => void;
  onRefreshTools: () => Promise<void>;
  onRefreshMedia: () => Promise<void>;
}

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
  const playableMedia = media.filter(
    (m) => m.kind === "video" || m.kind === "video_only" || m.kind === "audio",
  );

  const [selectedMediaId, setSelectedMediaId] = useState<string>(
    preselectedMediaId ?? playableMedia[0]?.id ?? "",
  );
  const [models, setModels] = useState<WhisperModelStatus[]>([]);
  const [totalInstalledBytes, setTotalInstalledBytes] = useState<number>(0);
  const [showAllModels, setShowAllModels] = useState(false);

  // Per-family active version tab (e.g. family "large-v3-turbo" -> "large-v3-turbo-q8_0")
  const [familySelection, setFamilySelection] = useState<Record<string, string>>({});

  // Globally active model version used for transcription
  const [selectedModel, setSelectedModel] = useState<string>(() => {
    return localStorage.getItem("aicut_whisper_model") ?? "";
  });

  const [formats, setFormats] = useState<ExportFormatInfo[]>([]);
  const [transcripts, setTranscripts] = useState<TranscriptRecord[]>([]);
  const [activeTranscriptId, setActiveTranscriptId] = useState<string | null>(null);

  const [language, setLanguage] = useState<string>("auto");
  const [translateToEnglish, setTranslateToEnglish] = useState(false);
  const [wordTimestamps, setWordTimestamps] = useState(true);
  const [maxSegmentChars, setMaxSegmentChars] = useState<string>("");
  const [initialPrompt, setInitialPrompt] = useState<string>("");
  const [selectedFormats, setSelectedFormats] = useState<string[]>(["srt", "fcpxml", "json"]);

  const [downloadingModel, setDownloadingModel] = useState<string | null>(null);
  const [uninstallingModel, setUninstallingModel] = useState<string | null>(null);
  const [installingEngine, setInstallingEngine] = useState(false);
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
    localStorage.setItem("aicut_whisper_model", modelId);
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
      setTotalInstalledBytes(mRes.data.total_installed_bytes ?? 0);
      setFormats(fRes.data.formats ?? []);
      const tList = tRes.data.transcripts ?? [];
      setTranscripts(tList);
      if (tList.length > 0 && !activeTranscriptId) {
        setActiveTranscriptId(tList[0].id);
      }

      // Initialize per-family default version (prefer installed -> recommended -> highest quality compatible)
      setFamilySelection((prev) => {
        const next = { ...prev };
        const byFamily = new Map<string, WhisperModelStatus[]>();
        for (const m of mList) {
          const arr = byFamily.get(m.family_id) ?? [];
          arr.push(m);
          byFamily.set(m.family_id, arr);
        }
        for (const [famId, items] of byFamily.entries()) {
          if (!next[famId] || !items.some((x) => x.id === next[famId])) {
            const pick =
              items.find((x) => x.installed) ??
              items.find((x) => x.recommended) ??
              items.find((x) => x.compatible) ??
              items[0];
            if (pick) next[famId] = pick.id;
          }
        }
        return next;
      });

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
  const completedTranscribeCount = jobs.filter(
    (j) => j.kind === "transcribe.run" && j.status === "completed",
  ).length;

  useEffect(() => {
    if (completedTranscribeCount > 0) {
      void loadAll();
      void onRefreshMedia();
    }
  }, [completedTranscribeCount]);

  // Group visible models by family (default: ONLY curated hardware-compatible best models + any already installed!)
  const visibleFamilies = useMemo(() => {
    const visible = showAllModels
      ? models
      : models.filter((m) => m.curated_for_hardware || m.installed);

    const groups: Array<{
      familyId: string;
      familyName: string;
      versions: WhisperModelStatus[];
      maxQuality: number;
      hasRecommended: boolean;
      hasBestQuality: boolean;
    }> = [];

    const map = new Map<string, WhisperModelStatus[]>();
    for (const m of visible) {
      const list = map.get(m.family_id) ?? [];
      list.push(m);
      map.set(m.family_id, list);
    }

    for (const [familyId, versions] of map.entries()) {
      versions.sort((a, b) => b.quality_score - a.quality_score);
      groups.push({
        familyId,
        familyName: versions[0].family_name,
        versions,
        maxQuality: Math.max(...versions.map((v) => v.quality_score)),
        hasRecommended: versions.some((v) => v.recommended),
        hasBestQuality: versions.some((v) => v.best_quality_for_hardware),
      });
    }

    // Put recommended / highest quality families first
    groups.sort((a, b) => {
      if (a.hasRecommended !== b.hasRecommended) return a.hasRecommended ? -1 : 1;
      return b.maxQuality - a.maxQuality;
    });

    return groups;
  }, [models, showAllModels]);

  const installedModels = useMemo(() => models.filter((m) => m.installed), [models]);

  async function handleInstallWhisperEngine() {
    setInstallingEngine(true);
    try {
      const res = await runAction("tools.install", { tool_id: "whisper" });
      await onRefreshTools();
      onNotify("success", "whisper.cpp instalado!", res.summary);
    } catch (e) {
      onNotify("error", "Falha ao instalar whisper.cpp", (e as Error).message);
    } finally {
      setInstallingEngine(false);
    }
  }

  async function handleDownloadModel(model: WhisperModelStatus) {
    setDownloadingModel(model.id);
    try {
      await runAction("transcribe.model_download", { model_id: model.id });
      await loadAll();
      chooseActiveModel(model.id);
      onNotify("success", "Modelo instalado!", `${model.display_name} pronto e selecionado como ativo.`);
    } catch (e) {
      onNotify("error", "Erro ao baixar modelo", (e as Error).message);
    } finally {
      setDownloadingModel(null);
    }
  }

  async function handleUninstallModel(model: WhisperModelStatus) {
    setUninstallingModel(model.id);
    try {
      const res = await runAction<{
        model_id: string;
        deleted: boolean;
        freed_bytes: number;
      }>("transcribe.model_delete", { model_id: model.id });
      await loadAll();
      onNotify(
        "success",
        "Modelo desinstalado completamente",
        `${model.display_name} removido do disco (${formatBytes(res.data.freed_bytes)} liberados).`,
      );
    } catch (e) {
      onNotify("error", "Erro ao desinstalar modelo", (e as Error).message);
    } finally {
      setUninstallingModel(null);
    }
  }

  function toggleFormat(fmtId: string) {
    setSelectedFormats((prev) =>
      prev.includes(fmtId) ? prev.filter((x) => x !== fmtId) : [...prev, fmtId],
    );
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
      const msg =
        e instanceof AicutActionError && e.detail.hint
          ? `${e.message} — ${e.detail.hint}`
          : (e as Error).message;
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

  const subtitleFormats = formats.filter((f) => f.category === "subtitle");
  const timelineFormats = formats.filter((f) => f.category === "timeline");
  const transcriptFormats = formats.filter((f) => f.category === "transcript");

  const gpuSummary = hardware
    ? hardware.gpus.length > 0
      ? `${hardware.gpus[0].name}${hardware.gpus[0].vram_mb ? ` (${(hardware.gpus[0].vram_mb / 1024).toFixed(1)} GB VRAM)` : ""}${hardware.cuda_version ? ` · CUDA ${hardware.cuda_version}` : ""}`
      : `CPU ${hardware.cpu_brand} (${(hardware.total_memory_mb / 1024).toFixed(1)} GB RAM)`
    : "Detectando hardware...";

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 18 }}>
      <div className="page-header">
        <div>
          <h1 className="page-title">Transcrever com IA Local (Whisper)</h1>
          <p className="page-subtitle">
            Modelos filtrados automaticamente para sua GPU e hardware — troque versões/precisões comparando a qualidade final e desinstale modelos completamente quando quiser.
          </p>
        </div>
        <button type="button" className="btn" onClick={() => onNavigate("library")}>
          Ver Biblioteca <ArrowRight size={14} />
        </button>
      </div>

      {/* Engine status banner if whisper.cpp or ffmpeg is missing */}
      {(!whisperTool?.installed || !ffmpegTool?.installed) && tools.length > 0 && (
        <div
          className="panel"
          style={{
            display: "flex",
            alignItems: "center",
            justifyContent: "space-between",
            gap: 14,
            background: "var(--warning-soft)",
            borderColor: "rgba(251, 191, 36, 0.3)",
            padding: "12px 16px",
          }}
        >
          <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
            <AlertTriangle size={18} color="var(--warning)" />
            <div>
              <strong style={{ fontSize: 13.5 }}>Motor de IA Local pendente</strong>
              <div style={{ fontSize: 12.5, color: "var(--text-secondary)" }}>
                {!whisperTool?.installed
                  ? `whisper.cpp não instalado. Build ideal detectada: ${whisperTool?.recommended_build ?? "auto"}.`
                  : "FFmpeg necessário para extrair o áudio do vídeo."}
              </div>
            </div>
          </div>
          <div style={{ display: "flex", gap: 8 }}>
            {!whisperTool?.installed && (
              <button
                type="button"
                className="btn btn-primary btn-sm"
                disabled={installingEngine}
                onClick={handleInstallWhisperEngine}
              >
                {installingEngine ? <Loader2 size={14} /> : <Download size={14} />}
                Instalar whisper.cpp Agora
              </button>
            )}
            <button type="button" className="btn btn-sm" onClick={() => onNavigate("settings")}>
              Configurações
            </button>
          </div>
        </div>
      )}

      {/* ================================================================ */}
      {/* GPU-CURATED MODEL & VERSION MANAGER                              */}
      {/* ================================================================ */}
      <div className="panel" style={{ display: "flex", flexDirection: "column", gap: 14 }}>
        <div
          style={{
            display: "flex",
            flexWrap: "wrap",
            alignItems: "center",
            justifyContent: "space-between",
            gap: 10,
          }}
        >
          <div>
            <h2 className="panel-title">
              <Cpu size={16} color="var(--accent)" /> Modelos Compatíveis com seu Hardware
            </h2>
            <div style={{ fontSize: 12.5, color: "var(--text-secondary)", marginTop: 3 }}>
              {gpuSummary}
            </div>
          </div>

          <div style={{ display: "flex", flexWrap: "wrap", alignItems: "center", gap: 8 }}>
            <span className="pill">
              <HardDrive size={11} /> Instalados: {installedModels.length} ({formatBytes(totalInstalledBytes)})
            </span>
            <label className="check-row" style={{ fontSize: 12.5 }}>
              <input
                type="checkbox"
                checked={showAllModels}
                onChange={(e) => setShowAllModels(e.target.checked)}
              />
              Mostrar todos (incl. incompatíveis)
            </label>
          </div>
        </div>

        {/* Active installed models quick-switcher strip */}
        {installedModels.length > 0 && (
          <div
            style={{
              display: "flex",
              flexWrap: "wrap",
              alignItems: "center",
              gap: 8,
              padding: "10px 12px",
              borderRadius: "var(--radius-md)",
              background: "var(--bg-elevated)",
              border: "1px solid var(--border-subtle)",
            }}
          >
            <span style={{ fontSize: 12, fontWeight: 600, color: "var(--text-secondary)" }}>
              Modelo ativo para transcrever:
            </span>
            {installedModels.map((m) => (
              <button
                key={m.id}
                type="button"
                className={`btn btn-sm ${selectedModel === m.id ? "btn-primary" : ""}`}
                onClick={() => chooseActiveModel(m.id)}
              >
                {selectedModel === m.id && <Check size={12} />}
                {m.display_name}
              </button>
            ))}
          </div>
        )}

        {/* Family Cards with Interactive Version Switcher + Final Quality Comparison */}
        <div className="grid-2">
          {visibleFamilies.map((fam) => {
            const viewedId = familySelection[fam.familyId] ?? fam.versions[0]?.id;
            const current =
              fam.versions.find((v) => v.id === viewedId) ?? fam.versions[0];
            if (!current) return null;

            const isActiveForUse = selectedModel === current.id && current.installed;
            const isDownloading = downloadingModel === current.id;
            const isUninstalling = uninstallingModel === current.id;

            return (
              <div
                key={fam.familyId}
                style={{
                  padding: 15,
                  borderRadius: "var(--radius-md)",
                  background: isActiveForUse ? "var(--accent-soft)" : "var(--bg-elevated)",
                  border: `1px solid ${isActiveForUse ? "var(--accent-border)" : "var(--border-subtle)"}`,
                  display: "flex",
                  flexDirection: "column",
                  gap: 11,
                }}
              >
                {/* Family Header */}
                <div style={{ display: "flex", alignItems: "flex-start", justifyContent: "space-between", gap: 8 }}>
                  <div>
                    <strong style={{ fontSize: 14.5 }}>{fam.familyName}</strong>
                    <div style={{ fontSize: 12, color: "var(--text-secondary)", marginTop: 2 }}>
                      {current.description}
                    </div>
                  </div>
                  <div style={{ display: "flex", flexDirection: "column", alignItems: "flex-end", gap: 4 }}>
                    {current.best_quality_for_hardware && (
                      <span className="pill success">
                        <Award size={11} /> Maior Qualidade Final
                      </span>
                    )}
                    {current.recommended && (
                      <span className="pill accent">Recomendado p/ sua GPU</span>
                    )}
                  </div>
                </div>

                {/* Version / Quantization Switcher Pills */}
                <div style={{ display: "flex", flexDirection: "column", gap: 5 }}>
                  <span style={{ fontSize: 11.5, color: "var(--text-muted)", fontWeight: 600 }}>
                    Versão / Precisão ({fam.versions.length} disponível(is)):
                  </span>
                  <div style={{ display: "flex", flexWrap: "wrap", gap: 6 }}>
                    {fam.versions.map((ver) => {
                      const activeTab = ver.id === current.id;
                      return (
                        <button
                          key={ver.id}
                          type="button"
                          className={`btn btn-sm ${activeTab ? "btn-primary" : ""}`}
                          style={{ fontSize: 12 }}
                          onClick={() => {
                            setFamilySelection((prev) => ({
                              ...prev,
                              [fam.familyId]: ver.id,
                            }));
                            if (ver.installed) {
                              chooseActiveModel(ver.id);
                            }
                          }}
                        >
                          {ver.installed && (
                            <span
                              style={{
                                width: 6,
                                height: 6,
                                borderRadius: "50%",
                                background: activeTab ? "var(--accent-fg)" : "var(--success)",
                              }}
                            />
                          )}
                          {ver.version_tag}
                        </button>
                      );
                    })}
                  </div>
                </div>

                {/* Explicit Quality & Hardware Comparison Details */}
                <div
                  style={{
                    padding: "10px 12px",
                    borderRadius: "var(--radius-sm)",
                    background: "var(--bg-surface)",
                    border: "1px solid var(--border-subtle)",
                    display: "flex",
                    flexDirection: "column",
                    gap: 6,
                    fontSize: 12.5,
                  }}
                >
                  <div style={{ display: "flex", flexWrap: "wrap", gap: 6 }}>
                    <span className="pill accent">Qualidade: {current.quality_tier}</span>
                    <span className="pill">{current.speed_tier}</span>
                    <span className="pill">{current.size_mb} MB</span>
                    <span className="pill">
                      VRAM mín: {(current.min_vram_mb / 1024).toFixed(1)} GB
                    </span>
                  </div>
                  <div style={{ color: "var(--text-primary)", lineHeight: 1.45 }}>
                    <strong>Qualidade final:</strong> {current.quality_notes}
                  </div>
                  <div
                    style={{
                      fontSize: 12,
                      color: current.compatible ? "var(--success)" : "var(--danger)",
                    }}
                  >
                    {current.compatibility_reason}
                  </div>
                </div>

                {/* Actions: Download / Select Active / Complete Uninstall */}
                <div
                  style={{
                    display: "flex",
                    flexWrap: "wrap",
                    alignItems: "center",
                    justifyContent: "space-between",
                    gap: 8,
                    marginTop: "auto",
                    paddingTop: 4,
                  }}
                >
                  {current.installed ? (
                    <>
                      <button
                        type="button"
                        className={`btn btn-sm ${isActiveForUse ? "btn-primary" : ""}`}
                        onClick={() => chooseActiveModel(current.id)}
                      >
                        {isActiveForUse ? (
                          <>
                            <CheckCircle2 size={13} /> Em uso para transcrever
                          </>
                        ) : (
                          <>Usar esta Versão</>
                        )}
                      </button>

                      <button
                        type="button"
                        className="btn btn-danger btn-sm"
                        disabled={isUninstalling}
                        onClick={() => handleUninstallModel(current)}
                      >
                        {isUninstalling ? <Loader2 size={13} /> : <Trash2 size={13} />}
                        Desinstalar Completamente
                      </button>
                    </>
                  ) : (
                    <>
                      <span style={{ fontSize: 12, color: "var(--text-muted)" }}>
                        Não instalado no disco
                      </span>
                      <button
                        type="button"
                        className="btn btn-primary btn-sm"
                        disabled={isDownloading}
                        onClick={() => handleDownloadModel(current)}
                      >
                        {isDownloading ? <Loader2 size={13} /> : <Download size={13} />}
                        Baixar {current.version_tag.split(" ")[0]} ({current.size_mb} MB)
                      </button>
                    </>
                  )}
                </div>
              </div>
            );
          })}
        </div>
      </div>

      {/* ================================================================ */}
      {/* TRANSCRIPTION EXECUTION FORM                                     */}
      {/* ================================================================ */}
      <form onSubmit={handleStartTranscription} style={{ display: "flex", flexDirection: "column", gap: 16 }}>
        <div className="panel" style={{ display: "flex", flexDirection: "column", gap: 16 }}>
          <div className="panel-header" style={{ marginBottom: 0 }}>
            <h2 className="panel-title">
              <Mic size={16} color="var(--accent)" /> Transcrever Vídeo / Áudio da Biblioteca
            </h2>
            {chosenModelObj && (
              <span className={`pill ${chosenModelObj.installed ? "success" : "warning"}`}>
                {chosenModelObj.installed ? <CheckCircle2 size={11} /> : <AlertTriangle size={11} />}
                {chosenModelObj.display_name}
              </span>
            )}
          </div>

          <div className="grid-2">
            <div className="field">
              <label htmlFor="tr-media">Vídeo ou Áudio da Biblioteca</label>
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
            </div>

            <div className="field">
              <label htmlFor="tr-lang">Idioma do Áudio</label>
              <select
                id="tr-lang"
                className="select"
                value={language}
                onChange={(e) => setLanguage(e.target.value)}
              >
                {LANGUAGES.map((l) => (
                  <option key={l.code} value={l.code}>
                    {l.label}
                  </option>
                ))}
              </select>
            </div>
          </div>

          {/* Options row */}
          <div className="grid-3" style={{ alignItems: "end" }}>
            <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
              <label className="check-row">
                <input
                  type="checkbox"
                  checked={wordTimestamps}
                  onChange={(e) => setWordTimestamps(e.target.checked)}
                />
                Timestamps por palavra (ideal para karaokê e cortes IA)
              </label>
              <label className="check-row">
                <input
                  type="checkbox"
                  checked={translateToEnglish}
                  onChange={(e) => setTranslateToEnglish(e.target.checked)}
                />
                Traduzir fala para Inglês
              </label>
            </div>
            <div className="field">
              <label>Máx. caracteres por bloco (opcional)</label>
              <input
                type="number"
                className="input"
                placeholder="ex: 42 (vazio = auto)"
                value={maxSegmentChars}
                onChange={(e) => setMaxSegmentChars(e.target.value)}
              />
            </div>
            <div className="field">
              <label>Prompt de contexto (nomes próprios / termos)</label>
              <input
                className="input"
                placeholder="ex: Aicut, DaVinci Resolve, Rust..."
                value={initialPrompt}
                onChange={(e) => setInitialPrompt(e.target.value)}
              />
            </div>
          </div>

          {/* Auto-export formats into Library */}
          <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
            <div style={{ fontSize: 12.5, fontWeight: 600, color: "var(--text-secondary)" }}>
              Formatos para salvar automaticamente na Biblioteca ao terminar:
            </div>

            <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
              <div style={{ display: "flex", flexWrap: "wrap", gap: 6, alignItems: "center" }}>
                <span className="pill" style={{ minWidth: 84, justifyContent: "center" }}>
                  <Subtitles size={11} /> Legendas
                </span>
                {subtitleFormats.map((f) => (
                  <button
                    key={f.id}
                    type="button"
                    className={`btn btn-sm ${selectedFormats.includes(f.id) ? "btn-primary" : ""}`}
                    title={f.description}
                    onClick={() => toggleFormat(f.id)}
                  >
                    {selectedFormats.includes(f.id) && <Check size={12} />} {f.name}
                  </button>
                ))}
              </div>

              <div style={{ display: "flex", flexWrap: "wrap", gap: 6, alignItems: "center" }}>
                <span className="pill" style={{ minWidth: 84, justifyContent: "center" }}>
                  <FileCode2 size={11} /> Timelines
                </span>
                {timelineFormats.map((f) => (
                  <button
                    key={f.id}
                    type="button"
                    className={`btn btn-sm ${selectedFormats.includes(f.id) ? "btn-primary" : ""}`}
                    title={f.description}
                    onClick={() => toggleFormat(f.id)}
                  >
                    {selectedFormats.includes(f.id) && <Check size={12} />} {f.name}
                  </button>
                ))}
              </div>

              <div style={{ display: "flex", flexWrap: "wrap", gap: 6, alignItems: "center" }}>
                <span className="pill" style={{ minWidth: 84, justifyContent: "center" }}>
                  <FileText size={11} /> Texto/IA
                </span>
                {transcriptFormats.map((f) => (
                  <button
                    key={f.id}
                    type="button"
                    className={`btn btn-sm ${selectedFormats.includes(f.id) ? "btn-primary" : ""}`}
                    title={f.description}
                    onClick={() => toggleFormat(f.id)}
                  >
                    {selectedFormats.includes(f.id) && <Check size={12} />} {f.name}
                  </button>
                ))}
              </div>
            </div>
          </div>

          <div
            style={{
              display: "flex",
              alignItems: "center",
              justifyContent: "space-between",
              paddingTop: 12,
              borderTop: "1px solid var(--border-subtle)",
            }}
          >
            <div style={{ fontSize: 12.5, color: "var(--text-secondary)" }}>
              {chosenModelObj && !chosenModelObj.installed
                ? `Baixe uma versão de modelo compatível acima antes de iniciar.`
                : `${selectedFormats.length} formato(s) serão salvos automaticamente na Biblioteca.`}
            </div>
            <button
              type="submit"
              className="btn btn-primary"
              disabled={
                starting ||
                !selectedMediaId ||
                !whisperTool?.installed ||
                !chosenModelObj?.installed
              }
            >
              {starting ? <Loader2 size={15} /> : <Sparkles size={15} />}
              Transcrever com IA Local
            </button>
          </div>
        </div>
      </form>

      {/* Active Transcription / Model Download Jobs */}
      {activeTranscribeJobs.length > 0 && (
        <div className="panel" style={{ display: "flex", flexDirection: "column", gap: 10 }}>
          <h3 className="panel-title">Em andamento ({activeTranscribeJobs.length})</h3>
          {activeTranscribeJobs.map((j) => (
            <div key={j.id} style={{ display: "flex", flexDirection: "column", gap: 6 }}>
              <div style={{ display: "flex", justifyContent: "space-between", fontSize: 13 }}>
                <span>{j.title}</span>
                <span style={{ fontFamily: "var(--font-mono)", color: "var(--text-secondary)" }}>
                  {j.progress.percent.toFixed(0)}% · {j.progress.message ?? j.progress.stage}
                </span>
              </div>
              <div className="progress-track">
                <div className="progress-fill" style={{ width: `${Math.min(100, j.progress.percent)}%` }} />
              </div>
            </div>
          ))}
        </div>
      )}

      {/* Saved Transcriptions + On-Demand Universal Exporter */}
      <div className="panel" style={{ display: "flex", flexDirection: "column", gap: 14 }}>
        <div className="panel-header" style={{ marginBottom: 0 }}>
          <h2 className="panel-title">
            <Film size={16} color="var(--accent)" /> Transcrições Geradas ({transcripts.length})
          </h2>
          {activeTranscript && (
            <input
              className="input"
              style={{ maxWidth: 260 }}
              placeholder="Buscar fala na transcrição..."
              value={searchSeg}
              onChange={(e) => setSearchSeg(e.target.value)}
            />
          )}
        </div>

        {transcripts.length === 0 ? (
          <div style={{ color: "var(--text-secondary)", fontSize: 13 }}>
            Nenhuma transcrição gerada ainda. Selecione um vídeo acima para transcrever offline.
          </div>
        ) : (
          <div style={{ display: "grid", gridTemplateColumns: "280px 1fr", gap: 16 }}>
            <div style={{ display: "flex", flexDirection: "column", gap: 6, maxHeight: 420, overflowY: "auto" }}>
              {transcripts.map((t) => (
                <div
                  key={t.id}
                  onClick={() => setActiveTranscriptId(t.id)}
                  style={{
                    padding: "10px 12px",
                    borderRadius: "var(--radius-md)",
                    background: activeTranscript?.id === t.id ? "var(--accent-soft)" : "var(--bg-elevated)",
                    border: `1px solid ${activeTranscript?.id === t.id ? "var(--accent-border)" : "var(--border-subtle)"}`,
                    cursor: "pointer",
                    display: "flex",
                    flexDirection: "column",
                    gap: 4,
                  }}
                >
                  <div style={{ display: "flex", justifyContent: "space-between", gap: 6 }}>
                    <strong style={{ fontSize: 13, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
                      {t.media_title}
                    </strong>
                    <button
                      type="button"
                      className="btn btn-ghost btn-sm"
                      style={{ padding: 2 }}
                      onClick={(e) => {
                        e.stopPropagation();
                        void handleDeleteTranscript(t.id);
                      }}
                    >
                      <Trash2 size={12} />
                    </button>
                  </div>
                  <div style={{ fontSize: 11.5, color: "var(--text-muted)" }}>
                    {t.model_id} · {t.language.toUpperCase()} · {t.segments.length} blocos
                  </div>
                  <div style={{ fontSize: 11, color: "var(--text-muted)" }}>
                    {formatRelativeDate(t.created_at)}
                  </div>
                </div>
              ))}
            </div>

            {activeTranscript && (
              <div style={{ display: "flex", flexDirection: "column", gap: 12, minWidth: 0 }}>
                <div
                  style={{
                    padding: 12,
                    borderRadius: "var(--radius-md)",
                    background: "var(--bg-elevated)",
                    display: "flex",
                    flexDirection: "column",
                    gap: 8,
                  }}
                >
                  <div style={{ fontSize: 12, fontWeight: 600, color: "var(--text-secondary)" }}>
                    Salvar outro formato de legenda ou timeline na Biblioteca (instantâneo):
                  </div>
                  <div style={{ display: "flex", flexWrap: "wrap", gap: 6 }}>
                    {formats.map((f) => {
                      const key = `${activeTranscript.id}:${f.id}`;
                      return (
                        <button
                          key={f.id}
                          type="button"
                          className="btn btn-sm"
                          disabled={exportingFmt === key}
                          onClick={() => handleQuickExport(activeTranscript.id, f.id)}
                        >
                          {exportingFmt === key ? <Loader2 size={12} /> : <Download size={12} />}
                          {f.name}
                        </button>
                      );
                    })}
                  </div>
                </div>

                <div
                  style={{
                    maxHeight: 310,
                    overflowY: "auto",
                    display: "flex",
                    flexDirection: "column",
                    gap: 6,
                  }}
                >
                  {activeTranscript.segments
                    .filter(
                      (s) =>
                        !searchSeg.trim() ||
                        s.text.toLowerCase().includes(searchSeg.toLowerCase()),
                    )
                    .map((seg) => (
                      <div
                        key={seg.index}
                        style={{
                          display: "grid",
                          gridTemplateColumns: "110px 1fr",
                          gap: 10,
                          padding: "7px 10px",
                          borderRadius: "var(--radius-sm)",
                          background: "var(--bg-input)",
                          fontSize: 13,
                        }}
                      >
                        <span
                          style={{
                            fontFamily: "var(--font-mono)",
                            fontSize: 11.5,
                            color: "var(--accent)",
                            display: "inline-flex",
                            alignItems: "center",
                            gap: 4,
                          }}
                        >
                          <Play size={10} />
                          {formatDuration(seg.start_ms / 1000)} → {formatDuration(seg.end_ms / 1000)}
                        </span>
                        <span>{seg.text}</span>
                      </div>
                    ))}
                </div>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
};
