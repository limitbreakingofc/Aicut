import React, { useEffect, useMemo, useState } from "react";
import {
  ArrowRight,
  Bookmark,
  BookmarkPlus,
  Copy,
  Download,
  FileVideo,
  FolderOpen,
  HardDrive,
  Link2,
  ListVideo,
  Music,
  Scissors,
  Search,
  Settings2,
  Shield,
  SlidersHorizontal,
  Subtitles,
  Terminal,
  Trash2,
  Upload,
  Video,
  VideoOff,
  X,
} from "lucide-react";
import { AicutActionError, isTauriRuntime, pickNativeFiles, runAction } from "../../lib/client";
import { formatBytes, formatDuration } from "../../lib/format";
import type { DownloadPreset, JobRecord, NotifyFn, ProbeUrlResult, ProbedFormat, RouteId, ToolStatus } from "../../lib/types";
import { copyToClipboard, splitList } from "../../lib/ui";
import { pageMeta } from "../../app/navigation";
import {
  ActiveJobsPanel,
  Badge,
  BusyIcon,
  Callout,
  Checkbox,
  ChoiceCards,
  CodeBlock,
  Disclosure,
  EmptyState,
  Field,
  MoreButton,
  PageHeader,
  Panel,
  SegmentedControl,
  Tabs,
  useContextMenu,
  type MenuItems,
} from "../../components";
import "./ingest.css";

interface IngestViewProps {
  tools: ToolStatus[];
  jobs: JobRecord[];
  onNavigate: (route: RouteId) => void;
  onNotify: NotifyFn;
  onMediaChanged: () => void;
}

type DownloadMode = "video_audio" | "video_only" | "audio_only";

const SPONSORBLOCK_OPTIONS = [
  { id: "sponsor", label: "Patrocinadores" },
  { id: "intro", label: "Introdução" },
  { id: "outro", label: "Encerramento" },
  { id: "selfpromo", label: "Autopromoção" },
  { id: "interaction", label: "Pedir Like/Inscrição" },
  { id: "preview", label: "Preview / Recap" },
];

const FORMAT_KIND_LABEL: Record<ProbedFormat["kind"], string> = {
  video_audio: "vídeo+áudio",
  video_only: "só vídeo",
  audio_only: "só áudio",
};

const parsePaths = (text: string) =>
  text
    .split(/\r?\n/)
    .map((l) => l.trim().replace(/^"(.*)"$/, "$1"))
    .filter(Boolean);

export const IngestView: React.FC<IngestViewProps> = ({ tools, jobs, onNavigate, onNotify, onMediaChanged }) => {
  const menu = useContextMenu();

  // Active sub-panel
  const [section, setSection] = useState<"download" | "import">("download");

  // ---- yt-dlp Download state (ALL options visually configurable) ----
  const [url, setUrl] = useState("");
  const [probing, setProbing] = useState(false);
  const [probe, setProbe] = useState<ProbeUrlResult | null>(null);
  const [downloading, setDownloading] = useState(false);

  const [mode, setMode] = useState<DownloadMode>("video_audio");
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

  // Collapsible sections (state survives switching tabs)
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
            listen: (event: string, cb: (ev: { payload?: { paths?: string[] } }) => void) => Promise<() => void>;
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

  /** Exact payload sent to `media.download` (also shown as a CLI command). */
  const downloadPayload = useMemo<Record<string, unknown>>(
    () => ({
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
        languages: splitList(subsLangs),
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
      tags: splitList(downloadTags),
      extra_args: extraArgs
        .split(/\s+/)
        .map((s) => s.trim())
        .filter(Boolean),
      wait: false,
    }),
    [
      url, mode, quality, formatId, videoCodec, container, audioQuality, subsEnabled, subsAuto, subsLangs,
      subsEmbed, subsFormat, embedThumbnail, embedMetadata, embedChapters, splitChapters, clipStart, clipEnd,
      forceKeyframes, playlistMode, playlistItems, concurrentFragments, rateLimit, proxy, impersonate,
      cookiesBrowser, cookiesFile, sponsorblock, filenameTemplate, customTitle, downloadTags, extraArgs,
    ],
  );

  const cliCommand = `aicut run media.download --json '${JSON.stringify(downloadPayload)}'`;

  async function handleStartDownload(e: React.FormEvent) {
    e.preventDefault();
    if (!url.trim()) {
      onNotify("warning", "URL obrigatória", "Cole o link do vídeo ou áudio antes de baixar.");
      return;
    }
    setDownloading(true);
    try {
      const res = await runAction<{ job_id: string }>("media.download", downloadPayload);
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
    const lines = parsePaths(manualPathInput);
    if (lines.length === 0) return;
    setImportPaths((prev) => Array.from(new Set([...prev, ...lines])));
    setManualPathInput("");
  }

  async function handleImportLocal() {
    const combined = Array.from(new Set([...importPaths, ...parsePaths(manualPathInput)]));
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
        tags: splitList(importTags),
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

  /* ---------- Context menus ---------- */
  const formatMenu = (f: ProbedFormat): MenuItems => [
    { type: "label", label: `Formato ${f.format_id} · ${FORMAT_KIND_LABEL[f.kind]}` },
    { label: `Usar este formato (-f ${f.format_id})`, icon: Download, onSelect: () => setFormatId(f.format_id) },
    f.kind === "video_only" && {
      label: `Combinar com melhor áudio (${f.format_id}+bestaudio)`,
      icon: Music,
      onSelect: () => setFormatId(`${f.format_id}+bestaudio`),
    },
    formatId && { label: "Voltar ao automático", icon: X, onSelect: () => setFormatId("") },
    { type: "separator" },
    { label: "Copiar format_id", icon: Copy, onSelect: () => void copyToClipboard(f.format_id, onNotify, "format_id") },
  ];

  const presetMenu = (p: DownloadPreset): MenuItems => [
    { type: "label", label: p.name },
    { label: "Aplicar preset", icon: Bookmark, onSelect: () => applyPreset(p) },
    {
      label: "Copiar configuração (JSON)",
      icon: Copy,
      onSelect: () => void copyToClipboard(JSON.stringify(p.config, null, 2), onNotify, "Configuração do preset"),
    },
  ];

  const pathMenu = (p: string): MenuItems => [
    { label: "Copiar caminho", icon: Copy, onSelect: () => void copyToClipboard(p, onNotify, "Caminho") },
    { type: "separator" },
    {
      label: "Remover da fila",
      icon: Trash2,
      danger: true,
      onSelect: () => setImportPaths((prev) => prev.filter((x) => x !== p)),
    },
  ];

  /* ---------- Disclosure summaries ---------- */
  const clipActive = Boolean(clipStart || clipEnd || forceKeyframes || splitChapters);
  const clipSummary =
    clipStart || clipEnd ? `${clipStart || "início"} → ${clipEnd || "fim"}` : "vídeo inteiro";

  const subsActive = subsEnabled || embedThumbnail;
  const subsSummary = [
    subsEnabled ? `legendas ${subsLangs || "—"} · ${subsFormat}` : "sem legendas",
    embedThumbnail && "capa",
    embedMetadata && "metadados",
  ]
    .filter(Boolean)
    .join(" · ");

  const playlistActive = playlistMode === "playlist";
  const playlistSummary = playlistActive ? `playlist ${playlistItems ? `[${playlistItems}]` : "inteira"}` : "só o vídeo atual";

  const networkParts = [
    `${concurrentFragments}× fragmentos`,
    rateLimit && `limite ${rateLimit}`,
    proxy && "proxy",
    cookiesBrowser && `cookies ${cookiesBrowser}`,
    cookiesFile && "cookies.txt",
    impersonate && `impersonate ${impersonate}`,
    sponsorblock.length > 0 && `sponsorblock ${sponsorblock.length}`,
    extraArgs.trim() && "flags extras",
  ].filter(Boolean);
  const networkActive = networkParts.length > 1 || concurrentFragments !== 4 || Boolean(customTitle || downloadTags || filenameTemplate);

  return (
    <>
      <PageHeader
        {...pageMeta("ingest")}
        actions={
          <button type="button" className="btn btn-sm" onClick={() => onNavigate("library")}>
            Biblioteca <ArrowRight aria-hidden />
          </button>
        }
      />

      <Tabs
        ariaLabel="Origem da mídia"
        value={section}
        onChange={setSection}
        items={[
          { id: "download", label: "Baixar da web (yt-dlp)", icon: Download },
          { id: "import", label: "Importar do PC", icon: Upload, count: importPaths.length || undefined },
        ]}
      />

      {missingTools && section === "download" && (
        <Callout
          tone="warning"
          title="Ferramentas recomendadas pendentes"
          actions={
            <button type="button" className="btn btn-primary btn-sm" onClick={() => onNavigate("settings")}>
              Instalar em Configurações
            </button>
          }
        >
          {!ytdlpTool?.installed && "yt-dlp não instalado. "}
          {!ffmpegTool?.installed && "FFmpeg não instalado (necessário para unir vídeo+áudio e thumbnails)."}
        </Callout>
      )}

      {/* ================================================================ */}
      {/* DOWNLOAD VIA YT-DLP                                              */}
      {/* ================================================================ */}
      {section === "download" && (
        <form onSubmit={handleStartDownload} className="split">
          <div className="split-main">
            <Panel title="Origem" icon={Link2} actions={<Badge>yt-dlp · +1000 sites</Badge>}>
              <div className="url-row">
                <div className="input-wrap grow">
                  <span className="input-prompt" aria-hidden>
                    $
                  </span>
                  <input
                    id="ytdlp-url"
                    type="url"
                    className="input input-lg"
                    placeholder="cole a URL: youtube, tiktok, twitch, instagram, playlist…"
                    aria-label="URL do vídeo, áudio ou playlist"
                    value={url}
                    onChange={(e) => setUrl(e.target.value)}
                  />
                </div>
                <button
                  type="button"
                  className="btn btn-lg"
                  onClick={handleProbe}
                  disabled={probing || !url.trim()}
                >
                  <BusyIcon busy={probing} icon={Search} />
                  Analisar
                </button>
                <button type="submit" className="btn btn-primary btn-lg" disabled={downloading || !url.trim()}>
                  <BusyIcon busy={downloading} icon={Download} />
                  Baixar agora
                </button>
              </div>

              {probe && (
                <div className="probe-card">
                  {probe.thumbnail ? (
                    <img className="probe-thumb" src={probe.thumbnail} alt="" />
                  ) : (
                    <div className="probe-thumb probe-thumb-empty" aria-hidden>
                      <FileVideo size={22} />
                    </div>
                  )}
                  <div className="stack stack-sm" style={{ minWidth: 0 }}>
                    <div className="strong">{probe.title}</div>
                    <div className="row wrap gap-xs">
                      {probe.extractor && <Badge tone="accent">{probe.extractor}</Badge>}
                      {probe.uploader && <Badge>{probe.uploader}</Badge>}
                      {probe.duration_sec !== undefined && <Badge>{formatDuration(probe.duration_sec)}</Badge>}
                      {probe.chapters_count > 0 && <Badge>{probe.chapters_count} capítulos</Badge>}
                      {probe.available_subtitles.length > 0 && (
                        <Badge title={probe.available_subtitles.join(", ")}>
                          legendas: {probe.available_subtitles.slice(0, 6).join(", ")}
                          {probe.available_subtitles.length > 6 ? "…" : ""}
                        </Badge>
                      )}
                      <Badge>{probe.formats.length} formatos</Badge>
                    </div>
                  </div>
                </div>
              )}
            </Panel>

            {probe && probe.formats.length > 0 && (
              <Panel
                title="Formatos detectados"
                icon={ListVideo}
                flush
                actions={
                  formatId ? (
                    <>
                      <Badge tone="accent">-f {formatId}</Badge>
                      <button type="button" className="btn btn-ghost btn-xs" onClick={() => setFormatId("")}>
                        <X aria-hidden /> automático
                      </button>
                    </>
                  ) : (
                    <span className="muted text-xs">clique numa linha para fixar o stream (-f)</span>
                  )
                }
              >
                <div className="table-wrap formats-table">
                  <table className="table">
                    <thead>
                      <tr>
                        <th>id</th>
                        <th>ext</th>
                        <th>tipo</th>
                        <th>resolução</th>
                        <th className="num">fps</th>
                        <th>vcodec</th>
                        <th>acodec</th>
                        <th className="num">tbr</th>
                        <th className="num">tamanho</th>
                        <th className="col-fit" aria-label="Ações" />
                      </tr>
                    </thead>
                    <tbody>
                      {probe.formats.map((f) => (
                        <tr
                          key={f.format_id}
                          data-clickable
                          aria-selected={formatId === f.format_id || formatId.startsWith(`${f.format_id}+`)}
                          onClick={() => setFormatId((curr) => (curr === f.format_id ? "" : f.format_id))}
                          onContextMenu={(e) => menu.open(e, formatMenu(f))}
                        >
                          <td className="text-accent strong">{f.format_id}</td>
                          <td>{f.ext}</td>
                          <td>
                            <span className={f.kind === "audio_only" ? "text-magenta" : f.kind === "video_only" ? "text-info" : ""}>
                              {FORMAT_KIND_LABEL[f.kind]}
                            </span>
                          </td>
                          <td>{f.resolution ?? "—"}</td>
                          <td className="num">{f.fps ?? "—"}</td>
                          <td className="muted">{f.vcodec ?? "—"}</td>
                          <td className="muted">{f.acodec ?? "—"}</td>
                          <td className="num muted">{f.tbr_kbps ? `${Math.round(f.tbr_kbps)}k` : "—"}</td>
                          <td className="num">{f.filesize_bytes ? formatBytes(f.filesize_bytes) : "—"}</td>
                          <td className="col-fit">
                            <span className="row-actions">
                              <MoreButton items={() => formatMenu(f)} />
                            </span>
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </Panel>
            )}

            <Panel
              title="Modo, qualidade & formato"
              icon={SlidersHorizontal}
              actions={
                <SegmentedControl
                  ariaLabel="Modo de download"
                  size="sm"
                  value={mode}
                  onChange={setMode}
                  options={[
                    { id: "video_audio", label: "Vídeo + Áudio", icon: Video },
                    { id: "video_only", label: "Só Vídeo", icon: VideoOff },
                    { id: "audio_only", label: "Só Áudio", icon: Music },
                  ]}
                />
              }
            >
              <div className={mode === "audio_only" ? "grid-3" : "grid-4"}>
                {mode !== "audio_only" ? (
                  <Field label="Resolução / qualidade" htmlFor="dl-quality">
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
                  </Field>
                ) : (
                  <Field label="Qualidade do áudio" htmlFor="dl-audio-q">
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
                  </Field>
                )}

                <Field label="Container / formato final" htmlFor="dl-container">
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
                </Field>

                {mode !== "audio_only" && (
                  <Field label="Codec de vídeo preferido" htmlFor="dl-codec">
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
                  </Field>
                )}

                <Field label="Format selector (-f)" htmlFor="dl-fmt-id">
                  <input
                    id="dl-fmt-id"
                    className="input"
                    placeholder={mode === "audio_only" ? "ex: 140 ou bestaudio" : "ex: 137+140 (vazio = auto)"}
                    value={formatId}
                    onChange={(e) => setFormatId(e.target.value)}
                  />
                </Field>
              </div>
            </Panel>

            <Panel title="Opções avançadas" icon={Settings2}>
              <div className="disclosure-group">
                <Disclosure
                  title="Recorte de trecho & capítulos"
                  icon={Scissors}
                  summary={clipSummary}
                  summaryActive={clipActive}
                  open={openClip}
                  onOpenChange={setOpenClip}
                >
                  <div className="grid-2">
                    <Field label="Início do trecho (--download-sections)" htmlFor="clip-start">
                      <input
                        id="clip-start"
                        className="input"
                        placeholder="ex: 00:01:15 ou 75"
                        value={clipStart}
                        onChange={(e) => setClipStart(e.target.value)}
                      />
                    </Field>
                    <Field label="Fim do trecho" htmlFor="clip-end">
                      <input
                        id="clip-end"
                        className="input"
                        placeholder="ex: 00:03:40 ou inf"
                        value={clipEnd}
                        onChange={(e) => setClipEnd(e.target.value)}
                      />
                    </Field>
                  </div>
                  <div className="row wrap gap-lg">
                    <Checkbox checked={forceKeyframes} onChange={setForceKeyframes} label="Forçar keyframes exatos no corte" />
                    <Checkbox checked={embedChapters} onChange={setEmbedChapters} label="Embutir capítulos no arquivo" />
                    <Checkbox checked={splitChapters} onChange={setSplitChapters} label="Dividir download por capítulos" />
                  </div>
                </Disclosure>

                <Disclosure
                  title="Legendas, capa & metadados"
                  icon={Subtitles}
                  summary={subsSummary}
                  summaryActive={subsActive}
                  open={openSubs}
                  onOpenChange={setOpenSubs}
                >
                  <div className="row wrap gap-lg">
                    <Checkbox checked={subsEnabled} onChange={setSubsEnabled} label="Baixar legendas" />
                    <Checkbox
                      checked={subsAuto}
                      disabled={!subsEnabled}
                      onChange={setSubsAuto}
                      label="Incluir legendas automáticas"
                    />
                    <Checkbox
                      checked={subsEmbed}
                      disabled={!subsEnabled}
                      onChange={setSubsEmbed}
                      label="Embutir legendas no vídeo"
                    />
                    <Checkbox checked={embedMetadata} onChange={setEmbedMetadata} label="Embutir metadados" />
                    <Checkbox checked={embedThumbnail} onChange={setEmbedThumbnail} label="Embutir thumbnail / capa" />
                  </div>
                  {subsEnabled && (
                    <div className="grid-2">
                      <Field label="Idiomas das legendas (separados por vírgula)" htmlFor="subs-langs">
                        <input
                          id="subs-langs"
                          className="input"
                          placeholder="pt,en,es"
                          value={subsLangs}
                          onChange={(e) => setSubsLangs(e.target.value)}
                        />
                      </Field>
                      <Field label="Formato da legenda" htmlFor="subs-format">
                        <select
                          id="subs-format"
                          className="select"
                          value={subsFormat}
                          onChange={(e) => setSubsFormat(e.target.value)}
                        >
                          <option value="srt">SRT</option>
                          <option value="vtt">VTT</option>
                          <option value="ass">ASS</option>
                          <option value="best">Melhor disponível</option>
                        </select>
                      </Field>
                    </div>
                  )}
                </Disclosure>

                <Disclosure
                  title="Playlist"
                  icon={ListVideo}
                  summary={playlistSummary}
                  summaryActive={playlistActive}
                  open={openPlaylist}
                  onOpenChange={setOpenPlaylist}
                >
                  <div className="grid-2">
                    <Field label="Comportamento para links de playlist" htmlFor="pl-mode">
                      <select
                        id="pl-mode"
                        className="select"
                        value={playlistMode}
                        onChange={(e) => setPlaylistMode(e.target.value as "single" | "playlist")}
                      >
                        <option value="single">Apenas o vídeo atual (--no-playlist)</option>
                        <option value="playlist">Baixar playlist inteira (--yes-playlist)</option>
                      </select>
                    </Field>
                    <Field label="Itens da playlist (opcional)" htmlFor="pl-items">
                      <input
                        id="pl-items"
                        className="input"
                        placeholder="ex: 1-5,8,10"
                        disabled={playlistMode !== "playlist"}
                        value={playlistItems}
                        onChange={(e) => setPlaylistItems(e.target.value)}
                      />
                    </Field>
                  </div>
                </Disclosure>

                <Disclosure
                  title="Rede, cookies, SponsorBlock & avançado"
                  icon={Shield}
                  summary={networkParts.join(" · ")}
                  summaryActive={networkActive}
                  open={openNetwork}
                  onOpenChange={setOpenNetwork}
                >
                  <div className="grid-3">
                    <Field label={`Fragmentos paralelos: ${concurrentFragments}×`} htmlFor="net-frag">
                      <input
                        id="net-frag"
                        type="range"
                        min={1}
                        max={16}
                        value={concurrentFragments}
                        onChange={(e) => setConcurrentFragments(Number(e.target.value))}
                      />
                    </Field>
                    <Field label="Limite de velocidade (--limit-rate)" htmlFor="net-rate">
                      <input
                        id="net-rate"
                        className="input"
                        placeholder="ex: 5M ou 500K"
                        value={rateLimit}
                        onChange={(e) => setRateLimit(e.target.value)}
                      />
                    </Field>
                    <Field label="Proxy (opcional)" htmlFor="net-proxy">
                      <input
                        id="net-proxy"
                        className="input"
                        placeholder="http://127.0.0.1:8080"
                        value={proxy}
                        onChange={(e) => setProxy(e.target.value)}
                      />
                    </Field>
                  </div>

                  <div className="grid-3">
                    <Field label="Cookies do navegador (privados/membros)" htmlFor="net-cookies">
                      <select
                        id="net-cookies"
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
                    </Field>
                    <Field label="Impersonate (Kick/Cloudflare)" htmlFor="net-imp">
                      <select
                        id="net-imp"
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
                    </Field>
                    <Field label="Ou caminho do arquivo cookies.txt" htmlFor="net-cookiefile">
                      <input
                        id="net-cookiefile"
                        className="input"
                        placeholder="C:\caminho\cookies.txt"
                        value={cookiesFile}
                        onChange={(e) => setCookiesFile(e.target.value)}
                      />
                    </Field>
                  </div>

                  <Field label="SponsorBlock — remover segmentos automaticamente">
                    <div className="row wrap gap-xs">
                      {SPONSORBLOCK_OPTIONS.map((opt) => {
                        const on = sponsorblock.includes(opt.id);
                        return (
                          <button
                            key={opt.id}
                            type="button"
                            className="chip"
                            aria-pressed={on}
                            onClick={() =>
                              setSponsorblock((prev) => (on ? prev.filter((x) => x !== opt.id) : [...prev, opt.id]))
                            }
                          >
                            {opt.label}
                          </button>
                        );
                      })}
                    </div>
                  </Field>

                  <div className="grid-3">
                    <Field label="Título personalizado na Biblioteca" htmlFor="adv-title">
                      <input
                        id="adv-title"
                        className="input"
                        placeholder="Opcional"
                        value={customTitle}
                        onChange={(e) => setCustomTitle(e.target.value)}
                      />
                    </Field>
                    <Field label="Tags (separadas por vírgula)" htmlFor="adv-tags">
                      <input
                        id="adv-tags"
                        className="input"
                        placeholder="b-roll, podcast, referencia"
                        value={downloadTags}
                        onChange={(e) => setDownloadTags(e.target.value)}
                      />
                    </Field>
                    <Field label="Template de arquivo (-o)" htmlFor="adv-tpl">
                      <input
                        id="adv-tpl"
                        className="input"
                        placeholder="%(title)s [%(id)s].%(ext)s"
                        value={filenameTemplate}
                        onChange={(e) => setFilenameTemplate(e.target.value)}
                      />
                    </Field>
                  </div>

                  <Field label="Flags extras do yt-dlp (opcional)" htmlFor="adv-extra">
                    <input
                      id="adv-extra"
                      className="input"
                      placeholder="ex: --geo-bypass --retries 10"
                      value={extraArgs}
                      onChange={(e) => setExtraArgs(e.target.value)}
                    />
                  </Field>
                </Disclosure>
              </div>
            </Panel>
          </div>

          <aside className="split-aside">
            <Panel
              title="Comando"
              icon={Terminal}
              actions={
                <button
                  type="button"
                  className="btn btn-ghost btn-xs"
                  onClick={() => void copyToClipboard(cliCommand, onNotify, "Comando CLI")}
                  title="Copiar como comando da CLI aicut"
                >
                  <Copy aria-hidden /> copiar
                </button>
              }
              footer={
                <button
                  type="submit"
                  className="btn btn-primary btn-block"
                  disabled={downloading || !url.trim()}
                >
                  <BusyIcon busy={downloading} icon={Download} />
                  Baixar para a Biblioteca
                </button>
              }
            >
              <p className="muted text-xs">
                Mesmo payload enviado ao core — cole em qualquer IA ou terminal para repetir este download.
              </p>
              <CodeBlock code={`$ aicut run media.download --json\n${JSON.stringify(downloadPayload, null, 2)}`} maxHeight={260} />
            </Panel>

            <Panel title="Presets" icon={Bookmark} flush={presets.length > 0}>
              {presets.length > 0 && (
                <div className="preset-list">
                  {presets.map((p) => (
                    <button
                      key={p.id}
                      type="button"
                      className="preset-row hover-host"
                      title={`Aplicar: ${p.description}`}
                      onClick={() => applyPreset(p)}
                      onContextMenu={(e) => menu.open(e, presetMenu(p))}
                    >
                      <Bookmark size={13} aria-hidden className="text-accent" />
                      <span className="grow">
                        <span className="preset-name">{p.name}</span>
                        <span className="preset-desc">{p.description}</span>
                      </span>
                    </button>
                  ))}
                </div>
              )}
              <div className="preset-save">
                <input
                  className="input input-sm"
                  aria-label="Nome do novo preset"
                  placeholder="salvar configuração atual como…"
                  value={presetName}
                  onChange={(e) => setPresetName(e.target.value)}
                  onKeyDown={(e) => {
                    if (e.key === "Enter") {
                      e.preventDefault();
                      void handleSavePreset();
                    }
                  }}
                />
                <button
                  type="button"
                  className="btn btn-sm"
                  disabled={!presetName.trim() || savingPreset}
                  onClick={handleSavePreset}
                >
                  <BusyIcon busy={savingPreset} icon={BookmarkPlus} size={13} /> Salvar
                </button>
              </div>
            </Panel>

            <ActiveJobsPanel jobs={activeIngestJobs} onViewAll={() => onNavigate("jobs")} />
          </aside>
        </form>
      )}

      {/* ================================================================ */}
      {/* IMPORT LOCAL PC VIDEOS & AUDIO                                   */}
      {/* ================================================================ */}
      {section === "import" && (
        <div className="split">
          <div className="split-main">
            <Panel title="Arquivos do computador" icon={HardDrive}>
              <div
                className="dropzone"
                data-dragging={dragging || undefined}
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
                <div className="dropzone-icon" aria-hidden>
                  <Upload />
                </div>
                <div className="dropzone-title">Arraste vídeos, áudios ou pastas para cá</div>
                <div className="dropzone-desc">
                  MP4, MKV, WEBM, MOV, AVI, FLV, TS, MP3, M4A, WAV, FLAC, OPUS, OGG, AAC — todos os formatos do yt-dlp e FFmpeg.
                </div>
                <div className="row" style={{ marginTop: 8 }}>
                  <button type="button" className="btn btn-primary" onClick={() => handlePickNative(false)}>
                    <Upload aria-hidden /> Selecionar arquivos
                  </button>
                  <button type="button" className="btn" onClick={() => handlePickNative(true)}>
                    <FolderOpen aria-hidden /> Selecionar pasta
                  </button>
                </div>
              </div>

              <Field
                label="Ou cole caminhos de arquivos/pastas (um por linha)"
                htmlFor="manual-paths"
                hint="Aspas são removidas automaticamente — cole direto do Explorer (Ctrl+Shift+C)."
              >
                <div className="row row-top">
                  <textarea
                    id="manual-paths"
                    className="textarea"
                    placeholder={"C:\\Users\\Voce\\Videos\\clip.mp4\nC:\\Users\\Voce\\Musicas\\podcast.mp3"}
                    value={manualPathInput}
                    onChange={(e) => setManualPathInput(e.target.value)}
                    rows={3}
                  />
                  <button type="button" className="btn" onClick={handleAddManualPaths} disabled={!manualPathInput.trim()}>
                    Adicionar
                  </button>
                </div>
              </Field>
            </Panel>

            <Panel
              title={
                <>
                  Fila de importação <Badge tone={importPaths.length ? "accent" : undefined}>{importPaths.length}</Badge>
                </>
              }
              icon={ListVideo}
              flush={importPaths.length > 0}
              actions={
                importPaths.length > 0 && (
                  <button type="button" className="btn btn-ghost btn-xs" onClick={() => setImportPaths([])}>
                    <Trash2 aria-hidden /> limpar
                  </button>
                )
              }
            >
              {importPaths.length === 0 ? (
                <EmptyState
                  compact
                  icon={ListVideo}
                  title="fila vazia"
                  description="Selecione arquivos, arraste do Explorer ou cole caminhos acima."
                />
              ) : (
                <div className="table-wrap">
                  <table className="table">
                    <thead>
                      <tr>
                        <th className="col-fit">#</th>
                        <th>caminho</th>
                        <th className="col-fit">tipo</th>
                        <th className="col-fit" aria-label="Ações" />
                      </tr>
                    </thead>
                    <tbody>
                      {importPaths.map((p, i) => {
                        const ext = /\.([a-z0-9]{2,5})$/i.exec(p)?.[1]?.toLowerCase();
                        return (
                          <tr key={p} onContextMenu={(e) => menu.open(e, pathMenu(p))}>
                            <td className="col-fit muted tabular">{String(i + 1).padStart(2, "0")}</td>
                            <td className="cell-title" title={p}>
                              {p}
                            </td>
                            <td className="col-fit muted">{ext ?? "pasta"}</td>
                            <td className="col-fit">
                              <span className="row-actions">
                                <button
                                  type="button"
                                  className="btn btn-ghost btn-icon btn-xs"
                                  aria-label="Remover da fila"
                                  title="Remover da fila"
                                  onClick={() => setImportPaths((prev) => prev.filter((x) => x !== p))}
                                >
                                  <X aria-hidden />
                                </button>
                                <MoreButton items={() => pathMenu(p)} />
                              </span>
                            </td>
                          </tr>
                        );
                      })}
                    </tbody>
                  </table>
                </div>
              )}
            </Panel>
          </div>

          <aside className="split-aside">
            <Panel
              title="Como importar"
              icon={Settings2}
              footer={
                <div className="stack stack-sm w-full">
                  <button
                    type="button"
                    className="btn btn-primary btn-block"
                    disabled={importing || (importPaths.length === 0 && !manualPathInput.trim())}
                    onClick={handleImportLocal}
                  >
                    <BusyIcon busy={importing} icon={Upload} />
                    Importar para a Biblioteca
                  </button>
                  {lastImportedCount !== null && (
                    <button type="button" className="btn btn-ghost btn-sm btn-block" onClick={() => onNavigate("library")}>
                      {lastImportedCount} arquivo(s) na Biblioteca <ArrowRight aria-hidden />
                    </button>
                  )}
                </div>
              }
            >
              <ChoiceCards
                ariaLabel="Modo de importação"
                columns={1}
                value={importMode}
                onChange={setImportMode}
                options={[
                  {
                    id: "reference",
                    title: "Referência (instantâneo)",
                    description: "Usa o arquivo onde ele está — nada é copiado.",
                  },
                  {
                    id: "copy",
                    title: "Copiar para o workspace",
                    description: "Duplica para a pasta de importações do Aicut.",
                  },
                ]}
              />
              <div className="stack stack-sm">
                <Checkbox checked={importRecursive} onChange={setImportRecursive} label="Incluir subpastas recursivamente" />
                <Checkbox checked={skipDuplicates} onChange={setSkipDuplicates} label="Ignorar duplicatas (fingerprint)" />
              </div>
              <Field label="Tags iniciais (separadas por vírgula)" htmlFor="import-tags">
                <input
                  id="import-tags"
                  className="input"
                  placeholder="raw, camera, audio"
                  value={importTags}
                  onChange={(e) => setImportTags(e.target.value)}
                />
              </Field>
            </Panel>

            <ActiveJobsPanel jobs={activeIngestJobs} onViewAll={() => onNavigate("jobs")} />
          </aside>
        </div>
      )}
    </>
  );
};
