import React, { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  ArrowRight,
  Bookmark,
  BookmarkPlus,
  CheckCircle2,
  CornerDownLeft,
  Download,
  FileVideo,
  FolderOpen,
  Link2,
  ListVideo,
  Music,
  RefreshCw,
  Scissors,
  Shield,
  SlidersHorizontal,
  Subtitles,
  Upload,
  Video,
  VideoOff,
  X,
} from "lucide-react";
import { AicutActionError, isTauriRuntime, pickNativeFiles, runAction } from "../../lib/client";
import { formatDuration } from "../../lib/format";
import type { DownloadPreset, JobRecord, NotifyFn, ProbeUrlResult, RouteId, ToolStatus } from "../../lib/types";
import { splitList } from "../../lib/ui";
import {
  ActiveJobsPanel,
  Badge,
  BusyIcon,
  Callout,
  Checkbox,
  Disclosure,
  Field,
  SegmentedControl,
} from "../../components";
import "./ingest.css";

interface IngestViewProps {
  tools: ToolStatus[];
  jobs: JobRecord[];
  onNavigate: (route: RouteId) => void;
  onNotify: NotifyFn;
  onMediaChanged: () => void;
}

type DownloadMode = "video_audio" | "audio_only" | "video_only";

const SPONSORBLOCK_OPTIONS = [
  { id: "sponsor", label: "Patrocinadores" },
  { id: "intro", label: "Intro" },
  { id: "outro", label: "Encerramento" },
  { id: "selfpromo", label: "Autopromoção" },
];

const parsePaths = (text: string) =>
  text
    .split(/\r?\n/)
    .map((l) => l.trim().replace(/^"(.*)"$/, "$1"))
    .filter(Boolean);

export const IngestView: React.FC<IngestViewProps> = ({ tools, jobs, onNavigate, onNotify, onMediaChanged }) => {
  // ---- Minimalist stage & drawer visibility ----
  const [url, setUrl] = useState("");
  const [drawerOpen, setDrawerOpen] = useState(false);
  const [probing, setProbing] = useState(false);
  const [probe, setProbe] = useState<ProbeUrlResult | null>(null);
  const [probeError, setProbeError] = useState<string | null>(null);
  const [downloading, setDownloading] = useState(false);

  // ---- Relevant download customization options ----
  const [mode, setMode] = useState<DownloadMode>("video_audio");
  const [quality, setQuality] = useState("best");
  const [formatId, setFormatId] = useState("");
  const [videoCodec, setVideoCodec] = useState("auto");
  const [container, setContainer] = useState("mp4");
  const [audioQuality, setAudioQuality] = useState("best");

  // Clip range
  const [clipStart, setClipStart] = useState("");
  const [clipEnd, setClipEnd] = useState("");
  const [forceKeyframes, setForceKeyframes] = useState(false);

  // Subtitles & metadata
  const [subsEnabled, setSubsEnabled] = useState(false);
  const [subsAuto, setSubsAuto] = useState(false);
  const [subsLangs, setSubsLangs] = useState("pt,en");
  const [subsEmbed, setSubsEmbed] = useState(true);
  const [embedThumbnail, setEmbedThumbnail] = useState(false);
  const [embedChapters, setEmbedChapters] = useState(true);

  // Playlist, cookies, SponsorBlock & personalization
  const [playlistMode, setPlaylistMode] = useState<"single" | "playlist">("single");
  const [playlistItems, setPlaylistItems] = useState("");
  const [cookiesBrowser, setCookiesBrowser] = useState("");
  const [sponsorblock, setSponsorblock] = useState<string[]>([]);
  const [customTitle, setCustomTitle] = useState("");
  const [concurrentFragments, setConcurrentFragments] = useState(4);

  // Presets
  const [presets, setPresets] = useState<DownloadPreset[]>([]);
  const [presetName, setPresetName] = useState("");
  const [savingPreset, setSavingPreset] = useState(false);

  // ---- Instant Drag & Drop / PC Import state ----
  const [dragging, setDragging] = useState(false);
  const [importing, setImporting] = useState(false);
  const [lastImportMsg, setLastImportMsg] = useState<string | null>(null);
  const [showWebPathBox, setShowWebPathBox] = useState(false);
  const [manualPathInput, setManualPathInput] = useState("");

  const urlInputRef = useRef<HTMLInputElement>(null);

  const ytdlpTool = tools.find((t) => t.id === "yt-dlp");
  const ffmpegTool = tools.find((t) => t.id === "ffmpeg");
  const missingTools = (!ytdlpTool?.installed || !ffmpegTool?.installed) && tools.length > 0;

  const activeIngestJobs = jobs.filter(
    (j) =>
      (j.kind === "media.download" || j.kind === "media.import_local") &&
      (j.status === "queued" || j.status === "running"),
  );

  // Adapt default container cleanly when switching between video and audio-only
  useEffect(() => {
    if (mode === "audio_only" && ["mp4", "mkv", "webm", "mov"].includes(container)) {
      setContainer("mp3");
    } else if (mode !== "audio_only" && ["mp3", "m4a", "wav", "flac", "opus"].includes(container)) {
      setContainer("mp4");
    }
  }, [mode, container]);

  const loadPresets = useCallback(async () => {
    try {
      const res = await runAction<{ presets: DownloadPreset[] }>("media.presets_list");
      setPresets(res.data.presets ?? []);
    } catch {
      // ignore if offline
    }
  }, []);

  useEffect(() => {
    void loadPresets();
  }, [loadPresets]);

  const importPathsDirectly = useCallback(
    async (paths: string[], recursive = false) => {
      const clean = Array.from(new Set(paths.map((p) => p.trim()).filter(Boolean)));
      if (clean.length === 0) return;
      setImporting(true);
      try {
        const res = await runAction<{
          imported: unknown[];
          skipped_duplicates: string[];
          errors: string[];
        }>("media.import_local", {
          paths: clean,
          mode: "reference",
          recursive,
          skip_duplicates: true,
          tags: [],
          wait: true,
        });
        const count = res.data.imported.length;
        onMediaChanged();
        if (count > 0) {
          setLastImportMsg(`${count} arquivo(s) importado(s) para a Biblioteca`);
          onNotify("success", "Importação concluída", res.summary);
        } else if (res.data.skipped_duplicates.length > 0) {
          setLastImportMsg("Arquivo já estava na Biblioteca");
          onNotify("info", "Já na Biblioteca", "Os arquivos selecionados já existem na Biblioteca.");
        }
        if (res.data.errors.length > 0) {
          onNotify("warning", "Alguns itens não foram importados", res.data.errors[0]);
        }
      } catch (e) {
        onNotify("error", "Erro na importação", (e as Error).message);
      } finally {
        setImporting(false);
      }
    },
    [onMediaChanged, onNotify],
  );

  // Native Linux Tauri v2 Drag & Drop across the entire dark canvas
  useEffect(() => {
    const tauriApi = (
      window as unknown as {
        __TAURI__?: {
          event?: {
            listen: (event: string, cb: (ev: { payload?: { paths?: string[] } }) => void) => Promise<() => void>;
          };
        };
      }
    ).__TAURI__;
    if (!tauriApi?.event?.listen) return;

    const unlisteners: Array<() => void> = [];
    void tauriApi.event
      .listen("tauri://drag-enter", () => setDragging(true))
      .then((fn) => unlisteners.push(fn));
    void tauriApi.event
      .listen("tauri://drag-leave", () => setDragging(false))
      .then((fn) => unlisteners.push(fn));
    void tauriApi.event
      .listen("tauri://drag-drop", (ev) => {
        setDragging(false);
        const dropped = ev.payload?.paths ?? [];
        if (dropped.length > 0) {
          void importPathsDirectly(dropped, false);
        }
      })
      .then((fn) => unlisteners.push(fn));

    return () => {
      unlisteners.forEach((u) => u());
    };
  }, [importPathsDirectly]);

  const handleBrowsePC = async (directory: boolean) => {
    const paths = await pickNativeFiles(directory);
    if (paths && paths.length > 0) {
      await importPathsDirectly(paths, directory);
    } else if (!isTauriRuntime()) {
      setShowWebPathBox(true);
    }
  };

  // Probe URL & open the right-side download settings panel
  const probeUrl = useCallback(
    async (rawUrl: string) => {
      const trimmed = rawUrl.trim();
      if (!trimmed) return;
      setDrawerOpen(true);
      setProbing(true);
      setProbe(null);
      setProbeError(null);
      setFormatId("");
      try {
        const res = await runAction<ProbeUrlResult>("media.probe_url", {
          url: trimmed,
          cookies_from_browser: cookiesBrowser || null,
          cookies_file: null,
          proxy: null,
          impersonate: null,
        });
        setProbe(res.data);
        if (res.data.is_playlist) {
          setPlaylistMode("playlist");
        } else {
          setPlaylistMode("single");
        }
      } catch (e) {
        const msg = e instanceof AicutActionError && e.detail.hint ? `${e.message} — ${e.detail.hint}` : (e as Error).message;
        setProbeError(msg);
      } finally {
        setProbing(false);
      }
    },
    [cookiesBrowser],
  );

  const handleUrlEnter = (e: React.FormEvent) => {
    e.preventDefault();
    if (!url.trim()) return;
    void probeUrl(url);
  };

  // Available resolutions detected from probed formats
  const availableHeights = useMemo(() => {
    if (!probe?.formats) return [];
    const set = new Set<number>();
    for (const f of probe.formats) {
      if (f.height && f.height >= 240) set.add(f.height);
    }
    return Array.from(set).sort((a, b) => b - a);
  }, [probe]);

  const applyPreset = (preset: DownloadPreset) => {
    const c = preset.config;
    if (c.mode === "video_audio" || c.mode === "video_only" || c.mode === "audio_only") setMode(c.mode);
    if (typeof c.quality === "string") setQuality(c.quality);
    if (typeof c.video_codec === "string") setVideoCodec(c.video_codec);
    if (typeof c.container === "string") setContainer(c.container);
    if (typeof c.audio_quality === "string") setAudioQuality(c.audio_quality);
    if (typeof c.embed_thumbnail === "boolean") setEmbedThumbnail(c.embed_thumbnail);
    if (typeof c.embed_chapters === "boolean") setEmbedChapters(c.embed_chapters);
    if (typeof c.concurrent_fragments === "number") setConcurrentFragments(c.concurrent_fragments);
    setFormatId(typeof c.format_id === "string" ? c.format_id : "");
    onNotify("info", "Preset aplicado", preset.name);
  };

  const handleSavePreset = async () => {
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
          embed_metadata: true,
          embed_thumbnail: embedThumbnail,
          embed_chapters: embedChapters,
          concurrent_fragments: concurrentFragments,
        },
      });
      setPresetName("");
      await loadPresets();
      onNotify("success", "Preset salvo", "Configuração salva para uso rápido.");
    } catch (e) {
      onNotify("error", "Erro ao salvar preset", (e as Error).message);
    } finally {
      setSavingPreset(false);
    }
  };

  const handleStartDownload = async () => {
    const trimmed = url.trim();
    if (!trimmed) return;
    setDownloading(true);
    try {
      const res = await runAction<{ job_id: string }>("media.download", {
        url: trimmed,
        mode,
        quality: formatId.trim() ? "custom" : quality,
        format_id: formatId.trim() || null,
        video_codec: videoCodec,
        container,
        audio_quality: mode === "audio_only" ? audioQuality : null,
        subtitles: {
          enabled: subsEnabled,
          auto_generated: subsAuto,
          languages: splitList(subsLangs),
          embed_in_file: subsEmbed,
          format: "srt",
        },
        embed_thumbnail: embedThumbnail,
        embed_metadata: true,
        embed_chapters: embedChapters,
        split_chapters: false,
        clip_start: clipStart.trim() || null,
        clip_end: clipEnd.trim() || null,
        force_keyframes_at_cuts: forceKeyframes,
        playlist_mode: playlistMode,
        playlist_items: playlistItems.trim() || null,
        concurrent_fragments: concurrentFragments,
        rate_limit: null,
        proxy: null,
        impersonate: null,
        cookies_from_browser: cookiesBrowser || null,
        cookies_file: null,
        sponsorblock_remove: sponsorblock,
        filename_template: null,
        custom_title: customTitle.trim() || null,
        tags: [],
        extra_args: [],
        wait: false,
      });
      onNotify(
        "success",
        "Download iniciado",
        `Job ${res.data.job_id.slice(0, 8)} em andamento — o arquivo irá direto para a Biblioteca.`,
      );
      setDrawerOpen(false);
      setUrl("");
      setProbe(null);
      setProbeError(null);
    } catch (e) {
      const msg = e instanceof AicutActionError && e.detail.hint ? `${e.message} — ${e.detail.hint}` : (e as Error).message;
      onNotify("error", "Não foi possível iniciar o download", msg);
    } finally {
      setDownloading(false);
    }
  };

  return (
    <div
      className="ingest-stage"
      data-dragging={dragging}
      onDragOver={(e) => {
        e.preventDefault();
        setDragging(true);
      }}
      onDragLeave={(e) => {
        if (e.currentTarget === e.target) setDragging(false);
      }}
      onDrop={(e) => {
        e.preventDefault();
        setDragging(false);
        const text = e.dataTransfer.getData("text/plain");
        if (text && /^https?:\/\//i.test(text.trim())) {
          setUrl(text.trim());
          void probeUrl(text.trim());
          return;
        }
        if (text && !isTauriRuntime()) {
          const parsed = parsePaths(text);
          if (parsed.length > 0) void importPathsDirectly(parsed, false);
        }
      }}
    >
      {dragging && (
        <div className="ingest-drop-banner">
          <Upload size={34} className="text-accent" aria-hidden />
          <div className="ingest-drop-banner-title">Solte os arquivos aqui para importar</div>
          <span className="muted text-xs">Vídeos, áudios e legendas vão direto para a Biblioteca</span>
        </div>
      )}

      {/* ================================================================ */}
      {/* MINIMALIST FULL-CANVAS DROP & URL HERO                           */}
      {/* ================================================================ */}
      <div className="ingest-center">
        {missingTools && (
          <div className="ingest-top-alerts">
            <Callout
              tone="warning"
              title="Ferramentas recomendadas pendentes"
              actions={
                <button type="button" className="btn btn-primary btn-sm" onClick={() => onNavigate("downloads")}>
                  Instalar em Configurações
                </button>
              }
            >
              {!ytdlpTool?.installed && "yt-dlp não instalado. "}
              {!ffmpegTool?.installed && "FFmpeg não instalado (necessário para unir vídeo+áudio e thumbnails)."}
            </Callout>
          </div>
        )}

        <div className="ingest-hero">
          {/* Centered tactile import icon */}
          <button
            type="button"
            className="ingest-icon-btn"
            disabled={importing}
            onClick={() => void handleBrowsePC(false)}
            aria-label="Importar arquivos do computador"
            title="Clique para procurar arquivos no PC ou arraste arquivos para esta área"
          >
            <BusyIcon busy={importing} icon={Download} size={48} />
          </button>

          <div className="ingest-hero-caption">
            <span className="ingest-hero-title">
              {importing ? "Importando para a Biblioteca…" : "Arraste arquivos aqui ou clique no ícone para importar"}
            </span>
            <div className="ingest-hero-sub">
              <span>MP4, MKV, MOV, MP3, WAV, SRT</span>
              <span>·</span>
              <button type="button" className="ingest-link-btn" onClick={() => void handleBrowsePC(true)}>
                <FolderOpen size={11} style={{ display: "inline", verticalAlign: "-1px", marginRight: 3 }} aria-hidden />
                importar pasta
              </button>
            </div>
          </div>

          {/* Minimalist URL bar right underneath the icon */}
          <form onSubmit={handleUrlEnter} className="ingest-url-form">
            <div className="ingest-url-bar">
              <Link2 size={16} className="ingest-url-icon" aria-hidden />
              <input
                ref={urlInputRef}
                type="url"
                className="ingest-url-input"
                placeholder="Cole ou digite um link para baixar (YouTube, Twitch, TikTok…)"
                aria-label="Link do vídeo ou áudio para baixar"
                value={url}
                onChange={(e) => setUrl(e.target.value)}
              />
              {url && (
                <button
                  type="button"
                  className="btn btn-ghost btn-icon btn-sm"
                  aria-label="Limpar link"
                  onClick={() => {
                    setUrl("");
                    setProbe(null);
                    setProbeError(null);
                    setDrawerOpen(false);
                    urlInputRef.current?.focus();
                  }}
                >
                  <X size={13} aria-hidden />
                </button>
              )}
              <button
                type="submit"
                className="ingest-url-submit"
                data-ready={url.trim().length > 0}
                disabled={!url.trim() || probing}
              >
                <BusyIcon busy={probing} icon={CornerDownLeft} size={12} />
                Enter
              </button>
            </div>
            <div className="ingest-url-hint">
              <span>Cole um link e pressione Enter para abrir as opções de download</span>
              <span>yt-dlp · +1000 sites</span>
            </div>
          </form>

          {/* Fallback manual path entry only when opened outside Tauri desktop */}
          {showWebPathBox && (
            <div className="ingest-web-import-box">
              <div className="row row-between">
                <span className="text-xs strong">Colar caminhos de arquivos ou pastas (modo navegador)</span>
                <button type="button" className="btn btn-ghost btn-icon btn-sm" onClick={() => setShowWebPathBox(false)}>
                  <X size={13} aria-hidden />
                </button>
              </div>
              <textarea
                className="textarea"
                rows={2}
                placeholder="C:\Videos\live.mp4 (um caminho por linha)"
                value={manualPathInput}
                onChange={(e) => setManualPathInput(e.target.value)}
              />
              <div className="row" style={{ justifyContent: "flex-end" }}>
                <button
                  type="button"
                  className="btn btn-primary btn-sm"
                  disabled={importing || !manualPathInput.trim()}
                  onClick={async () => {
                    await importPathsDirectly(parsePaths(manualPathInput), false);
                    setManualPathInput("");
                    setShowWebPathBox(false);
                  }}
                >
                  Importar para a Biblioteca
                </button>
              </div>
            </div>
          )}
        </div>

        {/* Non-intrusive bottom dock for active downloads or last import confirmation */}
        {(activeIngestJobs.length > 0 || lastImportMsg) && (
          <div className="ingest-bottom-dock">
            {activeIngestJobs.length > 0 && (
              <ActiveJobsPanel
                jobs={activeIngestJobs}
                title="Downloads em andamento"
                tone="accent"
                onViewAll={() => onNavigate("jobs")}
              />
            )}
            {lastImportMsg && (
              <div className="ingest-import-pill">
                <span className="row gap-xs text-success">
                  <CheckCircle2 size={14} aria-hidden />
                  {lastImportMsg}
                </span>
                <div className="row gap-xs">
                  <button type="button" className="btn btn-sm" onClick={() => onNavigate("library")}>
                    Abrir Biblioteca <ArrowRight size={12} aria-hidden />
                  </button>
                  <button
                    type="button"
                    className="btn btn-ghost btn-icon btn-sm"
                    aria-label="Fechar aviso"
                    onClick={() => setLastImportMsg(null)}
                  >
                    <X size={12} aria-hidden />
                  </button>
                </div>
              </div>
            )}
          </div>
        )}
      </div>

      {/* ================================================================ */}
      {/* RIGHT-SIDE DOWNLOAD CONFIGURATION PANEL (opens on URL + Enter)   */}
      {/* ================================================================ */}
      {drawerOpen && (
        <aside className="ingest-drawer" aria-label="Configurações do download">
          <header className="ingest-drawer-head">
            <div className="ingest-drawer-title">
              <Download size={14} className="text-accent" aria-hidden />
              <span>Configurar download</span>
            </div>
            <div className="row gap-xs">
              <button
                type="button"
                className="btn btn-ghost btn-icon btn-sm"
                title="Reanalisar link"
                disabled={probing || !url.trim()}
                onClick={() => void probeUrl(url)}
              >
                <BusyIcon busy={probing} icon={RefreshCw} size={13} />
              </button>
              <button
                type="button"
                className="btn btn-ghost btn-icon btn-sm"
                aria-label="Fechar configurações"
                onClick={() => setDrawerOpen(false)}
              >
                <X size={14} aria-hidden />
              </button>
            </div>
          </header>

          <div className="ingest-drawer-body">
            {/* Probed media preview */}
            {probing ? (
              <div className="ingest-media-card">
                <div className="ingest-media-thumb-empty">
                  <BusyIcon busy icon={FileVideo} size={20} />
                </div>
                <div className="ingest-media-meta">
                  <span className="ingest-media-title">Analisando link…</span>
                  <span className="ingest-media-sub">Obtendo título, duração e resoluções disponíveis</span>
                </div>
              </div>
            ) : probe ? (
              <div className="ingest-media-card">
                {probe.thumbnail ? (
                  <img src={probe.thumbnail} alt={probe.title} className="ingest-media-thumb" />
                ) : (
                  <div className="ingest-media-thumb-empty">
                    <FileVideo size={20} aria-hidden />
                  </div>
                )}
                <div className="ingest-media-meta">
                  <span className="ingest-media-title" title={probe.title}>
                    {probe.title}
                  </span>
                  <div className="ingest-media-sub">
                    {probe.uploader && <span>{probe.uploader}</span>}
                    {probe.duration_sec != null && <span>{formatDuration(probe.duration_sec)}</span>}
                    {probe.is_playlist && <Badge tone="accent">playlist</Badge>}
                  </div>
                </div>
              </div>
            ) : probeError ? (
              <Callout tone="warning" title="Prévia indisponível (você ainda pode baixar)">
                {probeError}
              </Callout>
            ) : null}

            {/* Saved quick presets */}
            {presets.length > 0 && (
              <div className="ingest-section-block">
                <span className="section-label">Presets rápidos</span>
                <div className="ingest-preset-chips">
                  {presets.map((p) => (
                    <button key={p.id} type="button" className="chip chip-sm" onClick={() => applyPreset(p)}>
                      <Bookmark size={11} aria-hidden /> {p.name}
                    </button>
                  ))}
                </div>
              </div>
            )}

            {/* Mode: Video+Audio / Audio Only / Video Only */}
            <Field label="Tipo de mídia">
              <SegmentedControl<DownloadMode>
                block
                size="sm"
                ariaLabel="Tipo de download"
                value={mode}
                onChange={setMode}
                options={[
                  { id: "video_audio", label: "Vídeo", icon: Video },
                  { id: "audio_only", label: "Áudio", icon: Music },
                  { id: "video_only", label: "Sem som", icon: VideoOff },
                ]}
              />
            </Field>

            {/* Quality & Container */}
            <div className="grid-2">
              {mode === "audio_only" ? (
                <Field label="Qualidade do áudio" htmlFor="ing-aq">
                  <select
                    id="ing-aq"
                    className="select select-sm"
                    value={audioQuality}
                    onChange={(e) => setAudioQuality(e.target.value)}
                  >
                    <option value="best">Melhor (320k / original)</option>
                    <option value="192k">Alta (192 kbps)</option>
                    <option value="128k">Padrão (128 kbps)</option>
                  </select>
                </Field>
              ) : (
                <Field label="Qualidade" htmlFor="ing-q">
                  <select
                    id="ing-q"
                    className="select select-sm"
                    value={quality}
                    onChange={(e) => {
                      setQuality(e.target.value);
                      setFormatId("");
                    }}
                  >
                    <option value="best">Melhor disponível</option>
                    <option value="2160p">4K (2160p){availableHeights.includes(2160) ? " ✓" : ""}</option>
                    <option value="1440p">2K (1440p){availableHeights.includes(1440) ? " ✓" : ""}</option>
                    <option value="1080p">Full HD (1080p){availableHeights.includes(1080) ? " ✓" : ""}</option>
                    <option value="720p">HD (720p){availableHeights.includes(720) ? " ✓" : ""}</option>
                    <option value="480p">SD (480p)</option>
                  </select>
                </Field>
              )}

              <Field label="Formato" htmlFor="ing-fmt">
                <select
                  id="ing-fmt"
                  className="select select-sm"
                  value={container}
                  onChange={(e) => setContainer(e.target.value)}
                >
                  {mode === "audio_only" ? (
                    <>
                      <option value="mp3">MP3</option>
                      <option value="wav">WAV (sem perda)</option>
                      <option value="m4a">M4A</option>
                      <option value="flac">FLAC</option>
                    </>
                  ) : (
                    <>
                      <option value="mp4">MP4 (compatível)</option>
                      <option value="mkv">MKV</option>
                      <option value="webm">WebM</option>
                    </>
                  )}
                </select>
              </Field>
            </div>

            <div className="ingest-divider" />

            {/* Optional Clip Range (great for long livestreams) */}
            <Disclosure
              title="Baixar apenas um trecho"
              icon={Scissors}
              summary={clipStart || clipEnd ? `${clipStart || "0:00"} → ${clipEnd || "fim"}` : "vídeo inteiro"}
              summaryActive={Boolean(clipStart || clipEnd)}
            >
              <div className="grid-2">
                <Field label="Início" htmlFor="ing-cs">
                  <input
                    id="ing-cs"
                    type="text"
                    className="input input-sm"
                    placeholder="00:00"
                    value={clipStart}
                    onChange={(e) => setClipStart(e.target.value)}
                  />
                </Field>
                <Field label="Fim" htmlFor="ing-ce">
                  <input
                    id="ing-ce"
                    type="text"
                    className="input input-sm"
                    placeholder="05:30"
                    value={clipEnd}
                    onChange={(e) => setClipEnd(e.target.value)}
                  />
                </Field>
              </div>
              <Checkbox
                checked={forceKeyframes}
                onChange={setForceKeyframes}
                label="Corte exato por frame (re-encode suave)"
              />
            </Disclosure>

            {/* Subtitles & Metadata */}
            <Disclosure
              title="Legendas & metadados"
              icon={Subtitles}
              summary={subsEnabled ? `legendas ${subsLangs}` : "padrão"}
              summaryActive={subsEnabled || embedThumbnail}
            >
              <Checkbox checked={subsEnabled} onChange={setSubsEnabled} label="Baixar legendas do vídeo" />
              {subsEnabled && (
                <div className="stack" style={{ gap: 8 }}>
                  <Field label="Idiomas (ex.: pt,en)" htmlFor="ing-sl">
                    <input
                      id="ing-sl"
                      type="text"
                      className="input input-sm"
                      value={subsLangs}
                      onChange={(e) => setSubsLangs(e.target.value)}
                    />
                  </Field>
                  <Checkbox checked={subsAuto} onChange={setSubsAuto} label="Incluir legendas automáticas" />
                  <Checkbox checked={subsEmbed} onChange={setSubsEmbed} label="Embutir legenda no arquivo" />
                </div>
              )}
              <Checkbox checked={embedThumbnail} onChange={setEmbedThumbnail} label="Embutir capa / miniatura" />
              <Checkbox checked={embedChapters} onChange={setEmbedChapters} label="Preservar capítulos" />
            </Disclosure>

            {/* Playlist options (auto-expanded if probed URL is a playlist) */}
            {(probe?.is_playlist || playlistMode === "playlist") && (
              <Disclosure
                title="Playlist"
                icon={ListVideo}
                defaultOpen={Boolean(probe?.is_playlist)}
                summary={playlistMode === "playlist" ? "baixar playlist" : "só 1 vídeo"}
                summaryActive={playlistMode === "playlist"}
              >
                <SegmentedControl<"single" | "playlist">
                  block
                  size="sm"
                  ariaLabel="Modo playlist"
                  value={playlistMode}
                  onChange={setPlaylistMode}
                  options={[
                    { id: "single", label: "Só este vídeo" },
                    { id: "playlist", label: "Playlist inteira" },
                  ]}
                />
                {playlistMode === "playlist" && (
                  <Field label="Itens específicos (opcional)" htmlFor="ing-pi">
                    <input
                      id="ing-pi"
                      type="text"
                      className="input input-sm"
                      placeholder="ex.: 1-5, 8"
                      value={playlistItems}
                      onChange={(e) => setPlaylistItems(e.target.value)}
                    />
                  </Field>
                )}
              </Disclosure>
            )}

            {/* Personalization, Browser Cookies & SponsorBlock */}
            <Disclosure
              title="Acesso, SponsorBlock & personalização"
              icon={SlidersHorizontal}
              summary={
                [cookiesBrowser && `cookies ${cookiesBrowser}`, sponsorblock.length > 0 && "sponsorblock", customTitle && "título"]
                  .filter(Boolean)
                  .join(" · ") || "opcional"
              }
              summaryActive={Boolean(cookiesBrowser || sponsorblock.length > 0 || customTitle || formatId)}
            >
              <Field label="Cookies do navegador (vídeos restritos / membros)" htmlFor="ing-ck">
                <select
                  id="ing-ck"
                  className="select select-sm"
                  value={cookiesBrowser}
                  onChange={(e) => setCookiesBrowser(e.target.value)}
                >
                  <option value="">Nenhum</option>
                  <option value="chrome">Google Chrome</option>
                  <option value="edge">Microsoft Edge</option>
                  <option value="firefox">Firefox</option>
                  <option value="brave">Brave</option>
                  <option value="opera">Opera</option>
                </select>
              </Field>

              <Field label="Título personalizado na Biblioteca" htmlFor="ing-title">
                <input
                  id="ing-title"
                  type="text"
                  className="input input-sm"
                  placeholder="Deixe vazio para usar o título original"
                  value={customTitle}
                  onChange={(e) => setCustomTitle(e.target.value)}
                />
              </Field>

              {mode !== "audio_only" && (
                <div className="grid-2">
                  <Field label="Codec de vídeo" htmlFor="ing-vc">
                    <select
                      id="ing-vc"
                      className="select select-sm"
                      value={videoCodec}
                      onChange={(e) => setVideoCodec(e.target.value)}
                    >
                      <option value="auto">Automático</option>
                      <option value="h264">H.264 (máxima compat.)</option>
                      <option value="vp9">VP9</option>
                      <option value="av1">AV1</option>
                    </select>
                  </Field>
                  <Field label="Conexões paralelas" htmlFor="ing-cf">
                    <select
                      id="ing-cf"
                      className="select select-sm"
                      value={concurrentFragments}
                      onChange={(e) => setConcurrentFragments(Number(e.target.value))}
                    >
                      <option value={1}>1× (estável)</option>
                      <option value={4}>4× (rápido · padrão)</option>
                      <option value={8}>8× (máximo)</option>
                    </select>
                  </Field>
                </div>
              )}

              <Field label="Remover trechos (SponsorBlock)">
                <div className="row wrap gap-xs">
                  {SPONSORBLOCK_OPTIONS.map((opt) => {
                    const active = sponsorblock.includes(opt.id);
                    return (
                      <button
                        key={opt.id}
                        type="button"
                        className="chip chip-sm"
                        aria-pressed={active}
                        onClick={() =>
                          setSponsorblock((prev) =>
                            prev.includes(opt.id) ? prev.filter((x) => x !== opt.id) : [...prev, opt.id],
                          )
                        }
                      >
                        <Shield size={10} aria-hidden /> {opt.label}
                      </button>
                    );
                  })}
                </div>
              </Field>

              {/* Save current config as preset */}
              <div className="row gap-xs">
                <input
                  type="text"
                  className="input input-sm grow"
                  placeholder="Salvar configuração como preset…"
                  value={presetName}
                  onChange={(e) => setPresetName(e.target.value)}
                />
                <button
                  type="button"
                  className="btn btn-sm"
                  disabled={savingPreset || !presetName.trim()}
                  onClick={() => void handleSavePreset()}
                >
                  <BusyIcon busy={savingPreset} icon={BookmarkPlus} size={12} />
                  Salvar
                </button>
              </div>
            </Disclosure>
          </div>

          <footer className="ingest-drawer-footer">
            <button
              type="button"
              className="btn btn-primary btn-block"
              disabled={downloading || !url.trim()}
              onClick={() => void handleStartDownload()}
            >
              <BusyIcon busy={downloading} icon={Download} size={15} />
              {downloading ? "Iniciando download…" : "Baixar para a Biblioteca"}
            </button>
          </footer>
        </aside>
      )}
    </div>
  );
};
