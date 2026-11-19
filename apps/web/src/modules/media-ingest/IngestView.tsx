import React, { useEffect, useState } from "react";
import {
  AlertTriangle,
  BookmarkPlus,
  ChevronDown,
  ChevronRight,
  Download,
  Film,
  FolderOpen,
  HardDrive,
  Link2,
  Loader2,
  Music,
  Scissors,
  Search,
  Shield,
  SlidersHorizontal,
  Subtitles,
  Upload,
  Video,
  VideoOff,
  ListVideo,
  ArrowRight,
  Trash2,
} from "lucide-react";
import { AicutActionError, isTauriRuntime, pickNativeFiles, runAction } from "../../lib/client";
import { formatBytes, formatDuration } from "../../lib/format";
import type { DownloadPreset, JobRecord, ProbeUrlResult, RouteId, ToolStatus } from "../../lib/types";

interface IngestViewProps {
  tools: ToolStatus[];
  jobs: JobRecord[];
  onNavigate: (route: RouteId) => void;
  onNotify: (level: "info" | "success" | "warning" | "error", title: string, message: string) => void;
  onMediaChanged: () => void;
}

const SPONSORBLOCK_OPTIONS = [
  { id: "sponsor", label: "Patrocinadores" },
  { id: "intro", label: "Introdução" },
  { id: "outro", label: "Encerramento" },
  { id: "selfpromo", label: "Autopromoção" },
  { id: "interaction", label: "Pedir Like/Inscrição" },
  { id: "preview", label: "Preview / Recap" },
];

export const IngestView: React.FC<IngestViewProps> = ({
  tools,
  jobs,
  onNavigate,
  onNotify,
  onMediaChanged,
}) => {
  // Active sub-panel (user can switch effortlessly or see both actions on Tab 1)
  const [section, setSection] = useState<"download" | "import">("download");

  // ---- yt-dlp Download state (ALL options visually configurable) ----
  const [url, setUrl] = useState("");
  const [probing, setProbing] = useState(false);
  const [probe, setProbe] = useState<ProbeUrlResult | null>(null);
  const [downloading, setDownloading] = useState(false);

  const [mode, setMode] = useState<"video_audio" | "video_only" | "audio_only">("video_audio");
  const [quality, setQuality] = useState("best");
  const [formatId, setFormatId] = useState("");
  const [videoCodec, setVideoCodec] = useState("auto");
  const [container, setContainer] = useState("mp4");
  const [audioQuality, setAudioQuality] = useState("best");

  // Subtitles & Metadata
  const [subsEnabled, setSubsEnabled] = useState(false);
  const [subsAuto, setSubsAuto] = useState(false);
  const [subsLangs, setSubsLangs] = useState("pt,en");
  const [subsEmbed, setSubsEmbed] = useState(true);
  const [subsFormat, setSubsFormat] = useState("srt");
  const [embedMetadata, setEmbedMetadata] = useState(true);
  const [embedThumbnail, setEmbedThumbnail] = useState(false);
  const [embedChapters, setEmbedChapters] = useState(true);
  const [splitChapters, setSplitChapters] = useState(false);

  // Clip & Playlist
  const [clipStart, setClipStart] = useState("");
  const [clipEnd, setClipEnd] = useState("");
  const [forceKeyframes, setForceKeyframes] = useState(false);
  const [playlistMode, setPlaylistMode] = useState<"single" | "playlist">("single");
  const [playlistItems, setPlaylistItems] = useState("");

  // Network, Cookies, SponsorBlock & Advanced
  const [concurrentFragments, setConcurrentFragments] = useState(4);
  const [rateLimit, setRateLimit] = useState("");
  const [proxy, setProxy] = useState("");
  const [impersonate, setImpersonate] = useState("");
  const [cookiesBrowser, setCookiesBrowser] = useState("");
  const [cookiesFile, setCookiesFile] = useState("");
  const [sponsorblock, setSponsorblock] = useState<string[]>([]);
  const [filenameTemplate, setFilenameTemplate] = useState("");
  const [customTitle, setCustomTitle] = useState("");
  const [downloadTags, setDownloadTags] = useState("");
  const [extraArgs, setExtraArgs] = useState("");

  // Accordions
  const [openClip, setOpenClip] = useState(false);
  const [openSubs, setOpenSubs] = useState(false);
  const [openPlaylist, setOpenPlaylist] = useState(false);
  const [openNetwork, setOpenNetwork] = useState(false);

  // Presets
  const [presets, setPresets] = useState<DownloadPreset[]>([]);
  const [savingPreset, setSavingPreset] = useState(false);
  const [presetName, setPresetName] = useState("");

  // ---- Local PC Import state ----
  const [importPaths, setImportPaths] = useState<string[]>([]);
  const [manualPathInput, setManualPathInput] = useState("");
  const [importMode, setImportMode] = useState<"reference" | "copy">("reference");
  const [importRecursive, setImportRecursive] = useState(false);
  const [skipDuplicates, setSkipDuplicates] = useState(true);
  const [importTags, setImportTags] = useState("");
  const [importing, setImporting] = useState(false);
  const [dragging, setDragging] = useState(false);
  const [lastImportedCount, setLastImportedCount] = useState<number | null>(null);

  const ytdlpTool = tools.find((t) => t.id === "yt-dlp");
  const ffmpegTool = tools.find((t) => t.id === "ffmpeg");
  const missingTools = (!ytdlpTool?.installed || !ffmpegTool?.installed) && tools.length > 0;

  useEffect(() => {
    void loadPresets();
  }, []);

  // Native Windows 11 Explorer Drag & Drop support via Tauri v2 global event API
  useEffect(() => {
    const tauriApi = (
      window as unknown as {
        __TAURI__?: {
          event?: {
            listen: (
              event: string,
              cb: (ev: { payload?: { paths?: string[] } }) => void,
            ) => Promise<() => void>;
          };
        };
      }
    ).__TAURI__;
    if (!tauriApi?.event?.listen) return;

    let unlisten: (() => void) | null = null;
    void tauriApi.event
      .listen("tauri://drag-drop", (ev) => {
        const dropped = ev.payload?.paths ?? [];
        if (dropped.length > 0) {
          setSection("import");
          setImportPaths((prev) => Array.from(new Set([...prev, ...dropped])));
          onNotify("info", "Arquivos recebidos", `${dropped.length} item(ns) pronto(s) para importar.`);
        }
      })
      .then((fn) => {
        unlisten = fn;
      });

    return () => {
      unlisten?.();
    };
  }, [onNotify]);

  // Adapt default container smoothly when switching between video and audio-only modes
  useEffect(() => {
    if (mode === "audio_only" && ["mp4", "mkv", "webm", "mov"].includes(container)) {
      setContainer("mp3");
    } else if (mode !== "audio_only" && ["mp3", "m4a", "wav", "flac", "opus"].includes(container)) {
      setContainer("mp4");
    }
  }, [mode]);

  async function loadPresets() {
    try {
      const res = await runAction<{ presets: DownloadPreset[] }>("media.presets_list");
      setPresets(res.data.presets ?? []);
    } catch {
      // ignore if offline
    }
  }

  function applyPreset(preset: DownloadPreset) {
    const c = preset.config;
    if (c.mode === "video_audio" || c.mode === "video_only" || c.mode === "audio_only") {
      setMode(c.mode);
    }
    if (typeof c.quality === "string") setQuality(c.quality);
    if (typeof c.video_codec === "string") setVideoCodec(c.video_codec);
    if (typeof c.container === "string") setContainer(c.container);
    if (typeof c.audio_quality === "string") setAudioQuality(c.audio_quality);
    if (typeof c.embed_metadata === "boolean") setEmbedMetadata(c.embed_metadata);
    if (typeof c.embed_thumbnail === "boolean") setEmbedThumbnail(c.embed_thumbnail);
    if (typeof c.embed_chapters === "boolean") setEmbedChapters(c.embed_chapters);
    if (typeof c.concurrent_fragments === "number") setConcurrentFragments(c.concurrent_fragments);
    setFormatId(typeof c.format_id === "string" ? c.format_id : "");
    onNotify("info", "Preset aplicado", preset.name);
  }

  async function handleSavePreset() {
    if (!presetName.trim()) return;
    setSavingPreset(true);
    try {
      await runAction("media.presets_save", {
        name: presetName.trim(),
        description: `${mode} · ${quality} · ${container}`,
        config: {
          mode,
          quality,
          format_id: formatId || null,
          video_codec: videoCodec,
          container,
          audio_quality: audioQuality,
          embed_metadata: embedMetadata,
          embed_thumbnail: embedThumbnail,
          embed_chapters: embedChapters,
          concurrent_fragments: concurrentFragments,
        },
      });
      setPresetName("");
      await loadPresets();
      onNotify("success", "Preset salvo", "Configuração salva para uso rápido e pela IA.");
    } catch (e) {
      onNotify("error", "Erro ao salvar preset", (e as Error).message);
    } finally {
      setSavingPreset(false);
    }
  }

  async function handleProbe() {
    if (!url.trim()) return;
    setProbing(true);
    setProbe(null);
    try {
      const res = await runAction<ProbeUrlResult>("media.probe_url", {
        url: url.trim(),
        cookies_from_browser: cookiesBrowser || null,
        cookies_file: cookiesFile.trim() || null,
        proxy: proxy.trim() || null,
        impersonate: impersonate || null,
      });
      setProbe(res.data);
      onNotify("success", "URL analisada", `${res.data.title} (${res.data.formats.length} formatos)`);
    } catch (e) {
      const msg = e instanceof AicutActionError && e.detail.hint ? `${e.message} — ${e.detail.hint}` : (e as Error).message;
      onNotify("error", "Falha ao analisar URL", msg);
    } finally {
      setProbing(false);
    }
  }

  async function handleStartDownload(e: React.FormEvent) {
    e.preventDefault();
    if (!url.trim()) {
      onNotify("warning", "URL obrigatória", "Cole o link do vídeo ou áudio antes de baixar.");
      return;
    }
    setDownloading(true);
    try {
      const payload: Record<string, unknown> = {
        url: url.trim(),
        mode,
        quality: formatId.trim() ? "custom" : quality,
        format_id: formatId.trim() || null,
        video_codec: videoCodec,
        container,
        audio_quality: mode === "audio_only" ? audioQuality : null,
        subtitles: {
          enabled: subsEnabled,
          auto_generated: subsAuto,
          languages: subsLangs
            .split(",")
            .map((s) => s.trim())
            .filter(Boolean),
          embed_in_file: subsEmbed,
          format: subsFormat,
        },
        embed_thumbnail: embedThumbnail,
        embed_metadata: embedMetadata,
        embed_chapters: embedChapters,
        split_chapters: splitChapters,
        clip_start: clipStart.trim() || null,
        clip_end: clipEnd.trim() || null,
        force_keyframes_at_cuts: forceKeyframes,
        playlist_mode: playlistMode,
        playlist_items: playlistItems.trim() || null,
        concurrent_fragments: concurrentFragments,
        rate_limit: rateLimit.trim() || null,
        proxy: proxy.trim() || null,
        impersonate: impersonate || null,
        cookies_from_browser: cookiesBrowser || null,
        cookies_file: cookiesFile.trim() || null,
        sponsorblock_remove: sponsorblock,
        filename_template: filenameTemplate.trim() || null,
        custom_title: customTitle.trim() || null,
        tags: downloadTags
          .split(",")
          .map((s) => s.trim())
          .filter(Boolean),
        extra_args: extraArgs
          .split(/\s+/)
          .map((s) => s.trim())
          .filter(Boolean),
        wait: false,
      };

      const res = await runAction<{ job_id: string }>("media.download", payload);
      onNotify(
        "success",
        "Download iniciado",
        `Job ${res.data.job_id.slice(0, 8)} em andamento — o arquivo irá automaticamente para a Biblioteca.`,
      );
    } catch (e) {
      const msg = e instanceof AicutActionError && e.detail.hint ? `${e.message} — ${e.detail.hint}` : (e as Error).message;
      onNotify("error", "Não foi possível iniciar o download", msg);
    } finally {
      setDownloading(false);
    }
  }

  async function handlePickNative(directory: boolean) {
    const paths = await pickNativeFiles(directory);
    if (paths && paths.length > 0) {
      setImportPaths((prev) => Array.from(new Set([...prev, ...paths])));
    } else if (!isTauriRuntime()) {
      onNotify(
        "info",
        "Modo navegador",
        "No modo web, cole os caminhos dos arquivos ou pastas abaixo. No app Windows 11 (Tauri), o seletor nativo abre automaticamente.",
      );
    }
  }

  function handleAddManualPaths() {
    const lines = manualPathInput
      .split(/\r?\n/)
      .map((l) => l.trim().replace(/^"(.*)"$/, "$1"))
      .filter(Boolean);
    if (lines.length === 0) return;
    setImportPaths((prev) => Array.from(new Set([...prev, ...lines])));
    setManualPathInput("");
  }

  async function handleImportLocal() {
    const combined = Array.from(
      new Set([
        ...importPaths,
        ...manualPathInput
          .split(/\r?\n/)
          .map((l) => l.trim().replace(/^"(.*)"$/, "$1"))
          .filter(Boolean),
      ]),
    );
    if (combined.length === 0) {
      onNotify("warning", "Nenhum arquivo selecionado", "Adicione arquivos ou pastas de vídeo/áudio para importar.");
      return;
    }

    setImporting(true);
    try {
      const res = await runAction<{
        imported: unknown[];
        skipped_duplicates: string[];
        errors: string[];
      }>("media.import_local", {
        paths: combined,
        mode: importMode,
        recursive: importRecursive,
        skip_duplicates: skipDuplicates,
        tags: importTags
          .split(",")
          .map((s) => s.trim())
          .filter(Boolean),
        wait: true,
      });
      setLastImportedCount(res.data.imported.length);
      setImportPaths([]);
      setManualPathInput("");
      onMediaChanged();
      onNotify("success", "Importação concluída", res.summary);
      if (res.data.errors.length > 0) {
        onNotify("warning", "Alguns itens não foram importados", res.data.errors[0]);
      }
    } catch (e) {
      onNotify("error", "Erro na importação", (e as Error).message);
    } finally {
      setImporting(false);
    }
  }

  const activeIngestJobs = jobs.filter(
    (j) =>
      (j.kind === "media.download" || j.kind === "media.import_local") &&
      (j.status === "queued" || j.status === "running"),
  );

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 18 }}>
      <div className="page-header">
        <div>
          <h1 className="page-title">Baixar & Importar</h1>
          <p className="page-subtitle">
            Baixe vídeos e áudios configurando todas as opções do yt-dlp ou importe arquivos do seu PC diretamente para a Biblioteca.
          </p>
        </div>
        <div className="segmented" role="tablist">
          <button
            type="button"
            className={`segmented-btn ${section === "download" ? "active" : ""}`}
            onClick={() => setSection("download")}
          >
            <Download size={15} /> Baixar (yt-dlp)
          </button>
          <button
            type="button"
            className={`segmented-btn ${section === "import" ? "active" : ""}`}
            onClick={() => setSection("import")}
          >
            <Upload size={15} /> Importar do PC
          </button>
        </div>
      </div>

      {/* Missing dependency helper banner */}
      {missingTools && section === "download" && (
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
              <strong style={{ fontSize: 13.5 }}>Ferramentas recomendadas pendentes</strong>
              <div style={{ fontSize: 12.5, color: "var(--text-secondary)" }}>
                {!ytdlpTool?.installed && "yt-dlp não instalado. "}
                {!ffmpegTool?.installed && "FFmpeg não instalado (necessário para unir vídeo+áudio e thumbnails)."}
              </div>
            </div>
          </div>
          <button
            type="button"
            className="btn btn-primary btn-sm"
            onClick={() => onNavigate("settings")}
          >
            Instalar em Configurações
          </button>
        </div>
      )}

      {/* ================================================================ */}
      {/* DOWNLOAD VIA YT-DLP                                              */}
      {/* ================================================================ */}
      {section === "download" && (
        <form onSubmit={handleStartDownload} style={{ display: "flex", flexDirection: "column", gap: 16 }}>
          {/* URL & Probe */}
          <div className="panel" style={{ display: "flex", flexDirection: "column", gap: 14 }}>
            <div style={{ display: "flex", gap: 10, alignItems: "flex-end" }}>
              <div className="field" style={{ flex: 1 }}>
                <label htmlFor="ytdlp-url">URL do vídeo, áudio ou playlist (YouTube, TikTok, Twitch, Instagram, +1000 sites)</label>
                <div style={{ position: "relative" }}>
                  <input
                    id="ytdlp-url"
                    type="url"
                    className="input"
                    placeholder="https://www.youtube.com/watch?v=..."
                    value={url}
                    onChange={(e) => setUrl(e.target.value)}
                    style={{ paddingLeft: 34 }}
                  />
                  <Link2
                    size={15}
                    style={{
                      position: "absolute",
                      left: 11,
                      top: "50%",
                      transform: "translateY(-50%)",
                      color: "var(--text-muted)",
                    }}
                  />
                </div>
              </div>
              <button
                type="button"
                className="btn"
                onClick={handleProbe}
                disabled={probing || !url.trim()}
              >
                {probing ? <Loader2 size={15} className="spin" /> : <Search size={15} />}
                Analisar URL
              </button>
              <button type="submit" className="btn btn-primary" disabled={downloading || !url.trim()}>
                {downloading ? <Loader2 size={15} /> : <Download size={15} />}
                Baixar Agora
              </button>
            </div>

            {/* Presets Strip */}
            {presets.length > 0 && (
              <div style={{ display: "flex", flexWrap: "wrap", alignItems: "center", gap: 8 }}>
                <span style={{ fontSize: 12, color: "var(--text-muted)", marginRight: 2 }}>Presets:</span>
                {presets.map((p) => (
                  <button
                    key={p.id}
                    type="button"
                    className="btn btn-sm"
                    onClick={() => applyPreset(p)}
                    title={p.description}
                  >
                    {p.name}
                  </button>
                ))}
              </div>
            )}

            {/* Probe Preview Card */}
            {probe && (
              <div
                style={{
                  display: "grid",
                  gridTemplateColumns: probe.thumbnail ? "200px 1fr" : "1fr",
                  gap: 16,
                  padding: 14,
                  borderRadius: "var(--radius-md)",
                  background: "var(--bg-elevated)",
                  border: "1px solid var(--border-subtle)",
                }}
              >
                {probe.thumbnail && (
                  <img
                    src={probe.thumbnail}
                    alt={probe.title}
                    style={{
                      width: "100%",
                      aspectRatio: "16/9",
                      objectFit: "cover",
                      borderRadius: "var(--radius-sm)",
                    }}
                  />
                )}
                <div style={{ display: "flex", flexDirection: "column", gap: 8, minWidth: 0 }}>
                  <div style={{ fontWeight: 600, fontSize: 15 }}>{probe.title}</div>
                  <div style={{ display: "flex", flexWrap: "wrap", gap: 6 }}>
                    {probe.uploader && <span className="pill">{probe.uploader}</span>}
                    {probe.extractor && <span className="pill accent">{probe.extractor}</span>}
                    {probe.duration_sec !== undefined && (
                      <span className="pill">{formatDuration(probe.duration_sec)}</span>
                    )}
                    {probe.chapters_count > 0 && (
                      <span className="pill">{probe.chapters_count} capítulos</span>
                    )}
                    {probe.available_subtitles.length > 0 && (
                      <span className="pill">{probe.available_subtitles.length} legendas</span>
                    )}
                  </div>
                  {probe.formats.length > 0 && (
                    <div className="field" style={{ marginTop: 4 }}>
                      <label htmlFor="exact-format">Selecionar stream exato detectado (opcional — sobrescreve qualidade)</label>
                      <select
                        id="exact-format"
                        className="select"
                        value={formatId}
                        onChange={(e) => setFormatId(e.target.value)}
                      >
                        <option value="">Automático (usar controles de Modo & Qualidade abaixo)</option>
                        {probe.formats.slice(0, 40).map((f) => (
                          <option key={f.format_id} value={f.format_id}>
                            [{f.format_id}] {f.kind} · {f.resolution ?? "áudio"} · {f.ext.toUpperCase()}
                            {f.fps ? ` · ${f.fps}fps` : ""}
                            {f.vcodec ? ` · ${f.vcodec}` : ""}
                            {f.acodec ? ` · ${f.acodec}` : ""}
                            {f.filesize_bytes ? ` · ${formatBytes(f.filesize_bytes)}` : ""}
                          </option>
                        ))}
                      </select>
                    </div>
                  )}
                </div>
              </div>
            )}
          </div>

          {/* Primary Visual Format Configuration */}
          <div className="panel" style={{ display: "flex", flexDirection: "column", gap: 16 }}>
            <div className="panel-header" style={{ marginBottom: 0 }}>
              <h2 className="panel-title">
                <SlidersHorizontal size={16} /> Modo, Qualidade & Formato
              </h2>
              <div className="segmented">
                <button
                  type="button"
                  className={`segmented-btn ${mode === "video_audio" ? "active" : ""}`}
                  onClick={() => setMode("video_audio")}
                >
                  <Video size={14} /> Vídeo + Áudio
                </button>
                <button
                  type="button"
                  className={`segmented-btn ${mode === "video_only" ? "active" : ""}`}
                  onClick={() => setMode("video_only")}
                >
                  <VideoOff size={14} /> Só Vídeo
                </button>
                <button
                  type="button"
                  className={`segmented-btn ${mode === "audio_only" ? "active" : ""}`}
                  onClick={() => setMode("audio_only")}
                >
                  <Music size={14} /> Só Áudio
                </button>
              </div>
            </div>

            <div className="grid-4">
              {mode !== "audio_only" ? (
                <div className="field">
                  <label htmlFor="dl-quality">Resolução / Qualidade</label>
                  <select
                    id="dl-quality"
                    className="select"
                    value={quality}
                    onChange={(e) => {
                      setQuality(e.target.value);
                      if (e.target.value !== "custom") setFormatId("");
                    }}
                  >
                    <option value="best">Melhor disponível</option>
                    <option value="2160p">4K (2160p)</option>
                    <option value="1440p">2K (1440p)</option>
                    <option value="1080p">Full HD (1080p)</option>
                    <option value="720p">HD (720p)</option>
                    <option value="480p">SD (480p)</option>
                    <option value="360p">360p</option>
                    <option value="worst">Menor tamanho</option>
                    <option value="custom">Personalizado (format_id)</option>
                  </select>
                </div>
              ) : (
                <div className="field">
                  <label htmlFor="dl-audio-q">Qualidade do Áudio</label>
                  <select
                    id="dl-audio-q"
                    className="select"
                    value={audioQuality}
                    onChange={(e) => setAudioQuality(e.target.value)}
                  >
                    <option value="best">Melhor disponível (0)</option>
                    <option value="320K">320 kbps (Alta)</option>
                    <option value="256K">256 kbps</option>
                    <option value="192K">192 kbps (Padrão)</option>
                    <option value="128K">128 kbps (Compacto)</option>
                  </select>
                </div>
              )}

              <div className="field">
                <label htmlFor="dl-container">Container / Formato Final</label>
                <select
                  id="dl-container"
                  className="select"
                  value={container}
                  onChange={(e) => setContainer(e.target.value)}
                >
                  <option value="auto">Automático (Original)</option>
                  {mode !== "audio_only" ? (
                    <>
                      <option value="mp4">MP4 (Universal)</option>
                      <option value="mkv">MKV (Sem perda de recursos)</option>
                      <option value="webm">WEBM</option>
                      <option value="mov">MOV (Editores)</option>
                    </>
                  ) : (
                    <>
                      <option value="mp3">MP3</option>
                      <option value="m4a">M4A (AAC)</option>
                      <option value="wav">WAV (Sem compressão)</option>
                      <option value="flac">FLAC (Lossless)</option>
                      <option value="opus">OPUS</option>
                    </>
                  )}
                </select>
              </div>

              {mode !== "audio_only" ? (
                <div className="field">
                  <label htmlFor="dl-codec">Codec de Vídeo Preferido</label>
                  <select
                    id="dl-codec"
                    className="select"
                    value={videoCodec}
                    onChange={(e) => setVideoCodec(e.target.value)}
                  >
                    <option value="auto">Automático</option>
                    <option value="h264">H.264 / AVC (Máx. compatibilidade)</option>
                    <option value="vp9">VP9 (YouTube HD/4K)</option>
                    <option value="av1">AV1 (Melhor compressão)</option>
                    <option value="hevc">H.265 / HEVC</option>
                  </select>
                </div>
              ) : (
                <div className="field">
                  <label htmlFor="dl-fmt-override">Format ID manual (opcional)</label>
                  <input
                    id="dl-fmt-override"
                    className="input"
                    placeholder="ex: 140 ou bestaudio"
                    value={formatId}
                    onChange={(e) => setFormatId(e.target.value)}
                  />
                </div>
              )}

              <div className="field">
                <label htmlFor="dl-fmt-id">Format Selector (-f) Manual</label>
                <input
                  id="dl-fmt-id"
                  className="input"
                  placeholder="ex: 137+140 (vazio = auto)"
                  value={formatId}
                  onChange={(e) => setFormatId(e.target.value)}
                />
              </div>
            </div>

            {/* Advanced Collapsible Sections for ALL yt-dlp Options */}
            <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
              {/* 1. Clip & Chapters */}
              <div className="accordion">
                <button
                  type="button"
                  className="accordion-trigger"
                  onClick={() => setOpenClip((v) => !v)}
                >
                  <span style={{ display: "inline-flex", alignItems: "center", gap: 8 }}>
                    <Scissors size={15} /> Recorte de Trecho & Capítulos
                  </span>
                  {openClip ? <ChevronDown size={15} /> : <ChevronRight size={15} />}
                </button>
                {openClip && (
                  <div className="accordion-body">
                    <div className="grid-2">
                      <div className="field">
                        <label>Início do trecho (--download-sections)</label>
                        <input
                          className="input"
                          placeholder="ex: 00:01:15 ou 75"
                          value={clipStart}
                          onChange={(e) => setClipStart(e.target.value)}
                        />
                      </div>
                      <div className="field">
                        <label>Fim do trecho</label>
                        <input
                          className="input"
                          placeholder="ex: 00:03:40 ou inf"
                          value={clipEnd}
                          onChange={(e) => setClipEnd(e.target.value)}
                        />
                      </div>
                    </div>
                    <div style={{ display: "flex", flexWrap: "wrap", gap: 18 }}>
                      <label className="check-row">
                        <input
                          type="checkbox"
                          checked={forceKeyframes}
                          onChange={(e) => setForceKeyframes(e.target.checked)}
                        />
                        Forçar keyframes exatos no corte
                      </label>
                      <label className="check-row">
                        <input
                          type="checkbox"
                          checked={embedChapters}
                          onChange={(e) => setEmbedChapters(e.target.checked)}
                        />
                        Embutir capítulos no arquivo
                      </label>
                      <label className="check-row">
                        <input
                          type="checkbox"
                          checked={splitChapters}
                          onChange={(e) => setSplitChapters(e.target.checked)}
                        />
                        Dividir download por capítulos
                      </label>
                    </div>
                  </div>
                )}
              </div>

              {/* 2. Subtitles & Metadata */}
              <div className="accordion">
                <button
                  type="button"
                  className="accordion-trigger"
                  onClick={() => setOpenSubs((v) => !v)}
                >
                  <span style={{ display: "inline-flex", alignItems: "center", gap: 8 }}>
                    <Subtitles size={15} /> Legendas, Capa & Metadados
                  </span>
                  {openSubs ? <ChevronDown size={15} /> : <ChevronRight size={15} />}
                </button>
                {openSubs && (
                  <div className="accordion-body">
                    <div style={{ display: "flex", flexWrap: "wrap", gap: 18 }}>
                      <label className="check-row">
                        <input
                          type="checkbox"
                          checked={subsEnabled}
                          onChange={(e) => setSubsEnabled(e.target.checked)}
                        />
                        Baixar legendas
                      </label>
                      <label className="check-row">
                        <input
                          type="checkbox"
                          checked={subsAuto}
                          disabled={!subsEnabled}
                          onChange={(e) => setSubsAuto(e.target.checked)}
                        />
                        Incluir legendas automáticas
                      </label>
                      <label className="check-row">
                        <input
                          type="checkbox"
                          checked={subsEmbed}
                          disabled={!subsEnabled}
                          onChange={(e) => setSubsEmbed(e.target.checked)}
                        />
                        Embutir legendas no vídeo
                      </label>
                      <label className="check-row">
                        <input
                          type="checkbox"
                          checked={embedMetadata}
                          onChange={(e) => setEmbedMetadata(e.target.checked)}
                        />
                        Embutir metadados
                      </label>
                      <label className="check-row">
                        <input
                          type="checkbox"
                          checked={embedThumbnail}
                          onChange={(e) => setEmbedThumbnail(e.target.checked)}
                        />
                        Embutir thumbnail / capa
                      </label>
                    </div>
                    {subsEnabled && (
                      <div className="grid-2">
                        <div className="field">
                          <label>Idiomas das legendas (separados por vírgula)</label>
                          <input
                            className="input"
                            placeholder="pt,en,es"
                            value={subsLangs}
                            onChange={(e) => setSubsLangs(e.target.value)}
                          />
                        </div>
                        <div className="field">
                          <label>Formato da legenda</label>
                          <select
                            className="select"
                            value={subsFormat}
                            onChange={(e) => setSubsFormat(e.target.value)}
                          >
                            <option value="srt">SRT</option>
                            <option value="vtt">VTT</option>
                            <option value="ass">ASS</option>
                            <option value="best">Melhor disponível</option>
                          </select>
                        </div>
                      </div>
                    )}
                  </div>
                )}
              </div>

              {/* 3. Playlist */}
              <div className="accordion">
                <button
                  type="button"
                  className="accordion-trigger"
                  onClick={() => setOpenPlaylist((v) => !v)}
                >
                  <span style={{ display: "inline-flex", alignItems: "center", gap: 8 }}>
                    <ListVideo size={15} /> Opções de Playlist
                  </span>
                  {openPlaylist ? <ChevronDown size={15} /> : <ChevronRight size={15} />}
                </button>
                {openPlaylist && (
                  <div className="accordion-body">
                    <div className="grid-2">
                      <div className="field">
                        <label>Comportamento para links de playlist</label>
                        <select
                          className="select"
                          value={playlistMode}
                          onChange={(e) => setPlaylistMode(e.target.value as "single" | "playlist")}
                        >
                          <option value="single">Apenas o vídeo atual (--no-playlist)</option>
                          <option value="playlist">Baixar playlist inteira (--yes-playlist)</option>
                        </select>
                      </div>
                      <div className="field">
                        <label>Itens da playlist (opcional)</label>
                        <input
                          className="input"
                          placeholder="ex: 1-5,8,10"
                          disabled={playlistMode !== "playlist"}
                          value={playlistItems}
                          onChange={(e) => setPlaylistItems(e.target.value)}
                        />
                      </div>
                    </div>
                  </div>
                )}
              </div>

              {/* 4. Network, Cookies, SponsorBlock & Custom */}
              <div className="accordion">
                <button
                  type="button"
                  className="accordion-trigger"
                  onClick={() => setOpenNetwork((v) => !v)}
                >
                  <span style={{ display: "inline-flex", alignItems: "center", gap: 8 }}>
                    <Shield size={15} /> Rede, Cookies, SponsorBlock & Avançado
                  </span>
                  {openNetwork ? <ChevronDown size={15} /> : <ChevronRight size={15} />}
                </button>
                {openNetwork && (
                  <div className="accordion-body">
                    <div className="grid-3">
                      <div className="field">
                        <label>Fragmentos paralelos: {concurrentFragments}x</label>
                        <input
                          type="range"
                          min={1}
                          max={16}
                          value={concurrentFragments}
                          onChange={(e) => setConcurrentFragments(Number(e.target.value))}
                        />
                      </div>
                      <div className="field">
                        <label>Limite de velocidade (--limit-rate)</label>
                        <input
                          className="input"
                          placeholder="ex: 5M ou 500K"
                          value={rateLimit}
                          onChange={(e) => setRateLimit(e.target.value)}
                        />
                      </div>
                      <div className="field">
                        <label>Proxy (opcional)</label>
                        <input
                          className="input"
                          placeholder="http://127.0.0.1:8080"
                          value={proxy}
                          onChange={(e) => setProxy(e.target.value)}
                        />
                      </div>
                    </div>

                    <div className="grid-3">
                      <div className="field">
                        <label>Cookies do navegador (vídeos privados/membros)</label>
                        <select
                          className="select"
                          value={cookiesBrowser}
                          onChange={(e) => setCookiesBrowser(e.target.value)}
                        >
                          <option value="">Não usar cookies</option>
                          <option value="edge">Microsoft Edge</option>
                          <option value="chrome">Google Chrome</option>
                          <option value="firefox">Mozilla Firefox</option>
                          <option value="brave">Brave</option>
                          <option value="opera">Opera</option>
                          <option value="vivaldi">Vivaldi</option>
                        </select>
                      </div>
                      <div className="field">
                        <label>Impersonate (--impersonate p/ Kick/Cloudflare)</label>
                        <select
                          className="select"
                          value={impersonate}
                          onChange={(e) => setImpersonate(e.target.value)}
                        >
                          <option value="">Automático</option>
                          <option value="chrome">Chrome</option>
                          <option value="edge">Edge</option>
                          <option value="firefox">Firefox</option>
                          <option value="safari">Safari</option>
                        </select>
                      </div>
                      <div className="field">
                        <label>Ou caminho do arquivo cookies.txt</label>
                        <input
                          className="input"
                          placeholder="C:\caminho\cookies.txt"
                          value={cookiesFile}
                          onChange={(e) => setCookiesFile(e.target.value)}
                        />
                      </div>
                    </div>

                    <div className="field">
                      <label>SponsorBlock — remover segmentos automaticamente</label>
                      <div style={{ display: "flex", flexWrap: "wrap", gap: 14 }}>
                        {SPONSORBLOCK_OPTIONS.map((opt) => (
                          <label key={opt.id} className="check-row">
                            <input
                              type="checkbox"
                              checked={sponsorblock.includes(opt.id)}
                              onChange={(e) => {
                                setSponsorblock((prev) =>
                                  e.target.checked
                                    ? [...prev, opt.id]
                                    : prev.filter((x) => x !== opt.id),
                                );
                              }}
                            />
                            {opt.label}
                          </label>
                        ))}
                      </div>
                    </div>

                    <div className="grid-3">
                      <div className="field">
                        <label>Título personalizado na Biblioteca</label>
                        <input
                          className="input"
                          placeholder="Opcional"
                          value={customTitle}
                          onChange={(e) => setCustomTitle(e.target.value)}
                        />
                      </div>
                      <div className="field">
                        <label>Tags (separadas por vírgula)</label>
                        <input
                          className="input"
                          placeholder="b-roll, podcast, referencia"
                          value={downloadTags}
                          onChange={(e) => setDownloadTags(e.target.value)}
                        />
                      </div>
                      <div className="field">
                        <label>Template de arquivo (-o)</label>
                        <input
                          className="input"
                          placeholder="%(title)s [%(id)s].%(ext)s"
                          value={filenameTemplate}
                          onChange={(e) => setFilenameTemplate(e.target.value)}
                        />
                      </div>
                    </div>

                    <div className="field">
                      <label>Flags extras do yt-dlp (opcional)</label>
                      <input
                        className="input"
                        placeholder="ex: --geo-bypass --retries 10"
                        value={extraArgs}
                        onChange={(e) => setExtraArgs(e.target.value)}
                      />
                    </div>
                  </div>
                )}
              </div>
            </div>

            {/* Save as Preset bar */}
            <div
              style={{
                display: "flex",
                alignItems: "center",
                justifyContent: "space-between",
                gap: 10,
                paddingTop: 10,
                borderTop: "1px solid var(--border-subtle)",
              }}
            >
              <div style={{ display: "flex", alignItems: "center", gap: 8, flex: 1, maxWidth: 380 }}>
                <input
                  className="input"
                  placeholder="Salvar configuração atual como preset..."
                  value={presetName}
                  onChange={(e) => setPresetName(e.target.value)}
                />
                <button
                  type="button"
                  className="btn btn-sm"
                  disabled={!presetName.trim() || savingPreset}
                  onClick={handleSavePreset}
                >
                  <BookmarkPlus size={14} /> Salvar Preset
                </button>
              </div>
              <button type="submit" className="btn btn-primary" disabled={downloading || !url.trim()}>
                {downloading ? <Loader2 size={15} /> : <Download size={15} />}
                Iniciar Download para Biblioteca
              </button>
            </div>
          </div>
        </form>
      )}

      {/* ================================================================ */}
      {/* IMPORT LOCAL PC VIDEOS & AUDIO                                   */}
      {/* ================================================================ */}
      {section === "import" && (
        <div className="panel" style={{ display: "flex", flexDirection: "column", gap: 16 }}>
          <div className="panel-header" style={{ marginBottom: 0 }}>
            <h2 className="panel-title">
              <HardDrive size={16} /> Importar Vídeos & Áudios do Computador
            </h2>
            <div className="segmented">
              <button
                type="button"
                className={`segmented-btn ${importMode === "reference" ? "active" : ""}`}
                onClick={() => setImportMode("reference")}
              >
                Referência (Instantâneo)
              </button>
              <button
                type="button"
                className={`segmented-btn ${importMode === "copy" ? "active" : ""}`}
                onClick={() => setImportMode("copy")}
              >
                Copiar para Workspace
              </button>
            </div>
          </div>

          <div
            className={`dropzone ${dragging ? "dragging" : ""}`}
            onDragOver={(e) => {
              e.preventDefault();
              setDragging(true);
            }}
            onDragLeave={() => setDragging(false)}
            onDrop={(e) => {
              e.preventDefault();
              setDragging(false);
              const droppedPaths: string[] = [];
              for (const file of Array.from(e.dataTransfer.files)) {
                const maybePath = (file as File & { path?: string }).path;
                if (maybePath) droppedPaths.push(maybePath);
              }
              if (droppedPaths.length > 0) {
                setImportPaths((prev) => Array.from(new Set([...prev, ...droppedPaths])));
              } else {
                onNotify(
                  "info",
                  "Use os botões ou cole o caminho",
                  "No app Windows 11 nativo, arrastar arquivos lê o caminho direto. No navegador, use o campo de caminhos abaixo.",
                );
              }
            }}
          >
            <Film size={28} color="var(--accent)" />
            <div style={{ fontWeight: 600 }}>Selecione vídeos, áudios ou pastas do seu PC</div>
            <div style={{ fontSize: 12.5, color: "var(--text-secondary)", maxWidth: 520 }}>
              Suporta todos os formatos do yt-dlp e FFmpeg: MP4, MKV, WEBM, MOV, AVI, FLV, TS, MP3, M4A, WAV, FLAC, OPUS, OGG, AAC.
            </div>
            <div style={{ display: "flex", gap: 10, marginTop: 6 }}>
              <button type="button" className="btn btn-primary" onClick={() => handlePickNative(false)}>
                <Upload size={15} /> Selecionar Arquivos
              </button>
              <button type="button" className="btn" onClick={() => handlePickNative(true)}>
                <FolderOpen size={15} /> Selecionar Pasta
              </button>
            </div>
          </div>

          {/* Direct path entry (works identically in Web, Windows, and Linux) */}
          <div className="field">
            <label htmlFor="manual-paths">Caminhos de arquivos ou pastas (um por linha)</label>
            <div style={{ display: "flex", gap: 8, alignItems: "flex-start" }}>
              <textarea
                id="manual-paths"
                className="textarea"
                placeholder={"C:\\Users\\Voce\\Videos\\clip.mp4\nC:\\Users\\Voce\\Musicas\\podcast.mp3"}
                value={manualPathInput}
                onChange={(e) => setManualPathInput(e.target.value)}
                rows={3}
              />
              <button type="button" className="btn" onClick={handleAddManualPaths}>
                Adicionar
              </button>
            </div>
          </div>

          {importPaths.length > 0 && (
            <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
              <div style={{ fontSize: 12.5, fontWeight: 600, color: "var(--text-secondary)" }}>
                Fila de importação ({importPaths.length})
              </div>
              {importPaths.map((p) => (
                <div
                  key={p}
                  style={{
                    display: "flex",
                    alignItems: "center",
                    justifyContent: "space-between",
                    padding: "7px 11px",
                    borderRadius: "var(--radius-sm)",
                    background: "var(--bg-elevated)",
                    fontFamily: "var(--font-mono)",
                    fontSize: 12,
                  }}
                >
                  <span style={{ overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
                    {p}
                  </span>
                  <button
                    type="button"
                    className="btn btn-ghost btn-sm"
                    onClick={() => setImportPaths((prev) => prev.filter((x) => x !== p))}
                  >
                    <Trash2 size={13} />
                  </button>
                </div>
              ))}
            </div>
          )}

          <div className="grid-2" style={{ alignItems: "center" }}>
            <div style={{ display: "flex", flexWrap: "wrap", gap: 16 }}>
              <label className="check-row">
                <input
                  type="checkbox"
                  checked={importRecursive}
                  onChange={(e) => setImportRecursive(e.target.checked)}
                />
                Incluir subpastas recursivamente
              </label>
              <label className="check-row">
                <input
                  type="checkbox"
                  checked={skipDuplicates}
                  onChange={(e) => setSkipDuplicates(e.target.checked)}
                />
                Ignorar duplicatas (fingerprint)
              </label>
            </div>
            <div className="field">
              <label>Tags iniciais (separadas por vírgula)</label>
              <input
                className="input"
                placeholder="raw, camera, audio"
                value={importTags}
                onChange={(e) => setImportTags(e.target.value)}
              />
            </div>
          </div>

          <div
            style={{
              display: "flex",
              alignItems: "center",
              justifyContent: "space-between",
              paddingTop: 10,
              borderTop: "1px solid var(--border-subtle)",
            }}
          >
            <div>
              {lastImportedCount !== null && (
                <button
                  type="button"
                  className="btn btn-sm"
                  onClick={() => onNavigate("library")}
                >
                  {lastImportedCount} arquivo(s) na Biblioteca <ArrowRight size={14} />
                </button>
              )}
            </div>
            <button
              type="button"
              className="btn btn-primary"
              disabled={importing || (importPaths.length === 0 && !manualPathInput.trim())}
              onClick={handleImportLocal}
            >
              {importing ? <Loader2 size={15} /> : <Upload size={15} />}
              Importar para Biblioteca
            </button>
          </div>
        </div>
      )}

      {/* Live Inline Queue Strip */}
      {activeIngestJobs.length > 0 && (
        <div className="panel" style={{ display: "flex", flexDirection: "column", gap: 10 }}>
          <div className="panel-header" style={{ marginBottom: 0 }}>
            <h3 className="panel-title">Em andamento agora ({activeIngestJobs.length})</h3>
            <button type="button" className="btn btn-ghost btn-sm" onClick={() => onNavigate("jobs")}>
              Ver Fila Completa <ArrowRight size={14} />
            </button>
          </div>
          {activeIngestJobs.map((j) => (
            <div key={j.id} style={{ display: "flex", flexDirection: "column", gap: 6 }}>
              <div style={{ display: "flex", justifyContent: "space-between", fontSize: 13 }}>
                <span>{j.title}</span>
                <span style={{ fontFamily: "var(--font-mono)", color: "var(--text-secondary)" }}>
                  {j.progress.percent.toFixed(0)}% · {j.progress.stage}
                </span>
              </div>
              <div className="progress-track">
                <div className="progress-fill" style={{ width: `${Math.min(100, j.progress.percent)}%` }} />
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
};
