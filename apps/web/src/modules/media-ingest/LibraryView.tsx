import React, { useEffect, useState } from "react";
import {
  Copy,
  Download,
  ExternalLink,
  FileCode2,
  FileText,
  Film,
  HardDrive,
  Mic,
  Music,
  Play,
  RefreshCw,
  Save,
  Scissors,
  Search,
  Sparkles,
  Subtitles,
  Trash2,
  Video,
  VideoOff,
  X,
} from "lucide-react";
import {
  fetchMediaText,
  mediaStreamUrl,
  mediaTextUrl,
  mediaThumbnailUrl,
  runAction,
} from "../../lib/client";
import { formatBytes, formatDuration, formatRelativeDate } from "../../lib/format";
import type { MediaAsset, MediaKind, RouteId } from "../../lib/types";

interface LibraryViewProps {
  media: MediaAsset[];
  selectedMediaId: string | null;
  searchQuery: string;
  filterKind: string | null;
  onSelectMedia: (id: string | null) => void;
  onSearchChange: (q: string) => void;
  onFilterKindChange: (kind: string | null) => void;
  onRefresh: () => void;
  onNavigate: (route: RouteId) => void;
  onTranscribeMedia: (mediaId: string) => void;
  onSmartClipMedia: (mediaId: string) => void;
  onNotify: (level: "info" | "success" | "warning" | "error", title: string, message: string) => void;
}

const isTextKind = (kind: MediaKind) =>
  kind === "subtitle" || kind === "timeline" || kind === "transcript";

export const LibraryView: React.FC<LibraryViewProps> = ({
  media,
  selectedMediaId,
  searchQuery,
  filterKind,
  onSelectMedia,
  onSearchChange,
  onFilterKindChange,
  onRefresh,
  onNavigate,
  onTranscribeMedia,
  onSmartClipMedia,
  onNotify,
}) => {
  const [sourceFilter, setSourceFilter] = useState<"all" | "download" | "local" | "generated">("all");
  const [sortBy, setSortBy] = useState("newest");

  const selectedAsset = media.find((m) => m.id === selectedMediaId) ?? null;
  const [editTitle, setEditTitle] = useState("");
  const [editTags, setEditTags] = useState("");
  const [editNotes, setEditNotes] = useState("");
  const [deleteFileOnDisk, setDeleteFileOnDisk] = useState(false);
  const [saving, setSaving] = useState(false);
  const [textPreview, setTextPreview] = useState<string | null>(null);
  const [loadingText, setLoadingText] = useState(false);

  useEffect(() => {
    if (selectedAsset) {
      setEditTitle(selectedAsset.title);
      setEditTags(selectedAsset.tags.join(", "));
      setEditNotes(selectedAsset.notes);
      setDeleteFileOnDisk(false);
      setTextPreview(null);

      if (isTextKind(selectedAsset.kind)) {
        setLoadingText(true);
        void fetchMediaText(selectedAsset.id)
          .then((txt) => setTextPreview(txt))
          .catch(() => setTextPreview("Não foi possível carregar o preview do arquivo."))
          .finally(() => setLoadingText(false));
      }
    }
  }, [selectedAsset?.id]);

  const filtered = media
    .filter((item) => {
      if (filterKind && item.kind !== filterKind) return false;
      if (sourceFilter !== "all" && item.source_type !== sourceFilter) return false;
      if (searchQuery.trim()) {
        const q = searchQuery.toLowerCase();
        const hit =
          item.title.toLowerCase().includes(q) ||
          item.file_name.toLowerCase().includes(q) ||
          (item.source_url ?? "").toLowerCase().includes(q) ||
          item.tags.some((t) => t.toLowerCase().includes(q));
        if (!hit) return false;
      }
      return true;
    })
    .sort((a, b) => {
      switch (sortBy) {
        case "oldest":
          return a.created_at.localeCompare(b.created_at);
        case "title":
          return a.title.localeCompare(b.title);
        case "size":
          return b.file_size_bytes - a.file_size_bytes;
        case "duration":
          return (b.metadata.duration_sec ?? 0) - (a.metadata.duration_sec ?? 0);
        default:
          return b.created_at.localeCompare(a.created_at);
      }
    });

  async function handleSaveEdits() {
    if (!selectedAsset) return;
    setSaving(true);
    try {
      await runAction("media.update", {
        id: selectedAsset.id,
        title: editTitle.trim() || selectedAsset.title,
        tags: editTags
          .split(",")
          .map((t) => t.trim())
          .filter(Boolean),
        notes: editNotes,
      });
      onRefresh();
      onNotify("success", "Alterações salvas", "Metadados atualizados na Biblioteca.");
    } catch (e) {
      onNotify("error", "Erro ao atualizar", (e as Error).message);
    } finally {
      setSaving(false);
    }
  }

  async function handleDelete() {
    if (!selectedAsset) return;
    try {
      await runAction("media.delete", {
        id: selectedAsset.id,
        delete_file: deleteFileOnDisk,
      });
      onSelectMedia(null);
      onRefresh();
      onNotify(
        "success",
        "Mídia removida",
        deleteFileOnDisk
          ? "Removida da Biblioteca e apagada do disco."
          : "Removida da Biblioteca (arquivo mantido no disco).",
      );
    } catch (e) {
      onNotify("error", "Erro ao remover", (e as Error).message);
    }
  }

  async function handleCopyText() {
    if (!textPreview) return;
    try {
      await navigator.clipboard.writeText(textPreview);
      onNotify("success", "Copiado!", "Conteúdo copiado para a área de transferência.");
    } catch {
      onNotify("warning", "Selecione o texto manualmente", "Não foi possível acessar a área de transferência.");
    }
  }

  function kindBadge(kind: MediaKind) {
    switch (kind) {
      case "audio":
        return (
          <span className="media-kind-badge">
            <Music size={11} /> Áudio
          </span>
        );
      case "video_only":
        return (
          <span className="media-kind-badge">
            <VideoOff size={11} /> Só Vídeo
          </span>
        );
      case "subtitle":
        return (
          <span className="media-kind-badge">
            <Subtitles size={11} /> Legenda
          </span>
        );
      case "timeline":
        return (
          <span className="media-kind-badge">
            <FileCode2 size={11} /> Timeline
          </span>
        );
      case "transcript":
        return (
          <span className="media-kind-badge">
            <FileText size={11} /> Transcrição
          </span>
        );
      default:
        return (
          <span className="media-kind-badge">
            <Video size={11} /> Vídeo
          </span>
        );
    }
  }

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 18 }}>
      <div className="page-header">
        <div>
          <h1 className="page-title">Biblioteca</h1>
          <p className="page-subtitle">
            Vídeos e áudios baixados via yt-dlp ou importados do PC + legendas, timelines e transcrições geradas pela IA local ({media.length} item(ns)).
          </p>
        </div>
        <div style={{ display: "flex", gap: 8 }}>
          <button type="button" className="btn" onClick={onRefresh}>
            <RefreshCw size={14} /> Atualizar
          </button>
          <button type="button" className="btn btn-primary" onClick={() => onNavigate("ingest")}>
            <Download size={14} /> Baixar / Importar
          </button>
        </div>
      </div>

      {/* Filter & Search Bar */}
      <div
        className="panel"
        style={{
          display: "flex",
          flexWrap: "wrap",
          alignItems: "center",
          justifyContent: "space-between",
          gap: 12,
          padding: "12px 14px",
        }}
      >
        <div style={{ position: "relative", flex: "1 1 220px", maxWidth: 340 }}>
          <input
            className="input"
            placeholder="Buscar por título, arquivo, tag ou URL..."
            value={searchQuery}
            onChange={(e) => onSearchChange(e.target.value)}
            style={{ paddingLeft: 32 }}
          />
          <Search
            size={14}
            style={{
              position: "absolute",
              left: 10,
              top: "50%",
              transform: "translateY(-50%)",
              color: "var(--text-muted)",
            }}
          />
        </div>

        <div style={{ display: "flex", flexWrap: "wrap", gap: 6, alignItems: "center" }}>
          {[
            { id: null, label: "Todos" },
            { id: "video", label: "Vídeo" },
            { id: "video_only", label: "Só Vídeo" },
            { id: "audio", label: "Áudio" },
            { id: "subtitle", label: "Legendas" },
            { id: "timeline", label: "Timelines" },
            { id: "transcript", label: "Transcrições" },
          ].map((opt) => (
            <button
              key={String(opt.id)}
              type="button"
              className={`btn btn-sm ${filterKind === opt.id ? "btn-primary" : ""}`}
              onClick={() => onFilterKindChange(opt.id)}
            >
              {opt.label}
            </button>
          ))}

          <select
            className="select"
            style={{ width: "auto" }}
            value={sourceFilter}
            onChange={(e) =>
              setSourceFilter(e.target.value as "all" | "download" | "local" | "generated")
            }
          >
            <option value="all">Todas as origens</option>
            <option value="download">Baixados (yt-dlp)</option>
            <option value="local">Importados (PC)</option>
            <option value="generated">Gerados por IA</option>
          </select>

          <select
            className="select"
            style={{ width: "auto" }}
            value={sortBy}
            onChange={(e) => setSortBy(e.target.value)}
          >
            <option value="newest">Mais recentes</option>
            <option value="oldest">Mais antigos</option>
            <option value="title">Título A–Z</option>
            <option value="duration">Maior duração</option>
            <option value="size">Maior tamanho</option>
          </select>
        </div>
      </div>

      {/* Empty State or Media Grid */}
      {filtered.length === 0 ? (
        <div className="panel" style={{ textAlign: "center", padding: "48px 24px" }}>
          <Film size={36} color="var(--text-muted)" style={{ marginBottom: 10 }} />
          <div style={{ fontWeight: 600, fontSize: 16 }}>
            {media.length === 0 ? "Sua Biblioteca está vazia" : "Nenhum resultado para o filtro atual"}
          </div>
          <p style={{ color: "var(--text-secondary)", margin: "6px auto 16px", maxWidth: 440 }}>
            {media.length === 0
              ? "Baixe um vídeo com yt-dlp ou importe arquivos do PC na primeira aba — e transcreva na aba Transcrever IA para gerar legendas e timelines."
              : "Tente limpar a busca ou trocar os filtros de tipo e origem."}
          </p>
          {media.length === 0 && (
            <button type="button" className="btn btn-primary" onClick={() => onNavigate("ingest")}>
              Ir para Baixar & Importar
            </button>
          )}
        </div>
      ) : (
        <div className="media-grid">
          {filtered.map((item) => (
            <article
              key={item.id}
              className={`media-card ${selectedMediaId === item.id ? "selected" : ""}`}
              onClick={() => onSelectMedia(item.id)}
            >
              <div className="media-thumb">
                {item.thumbnail_path && !isTextKind(item.kind) ? (
                  <img src={mediaThumbnailUrl(item.id)} alt={item.title} loading="lazy" />
                ) : item.kind === "audio" ? (
                  <div style={{ display: "flex", flexDirection: "column", alignItems: "center", gap: 6, color: "var(--accent)" }}>
                    <Music size={34} />
                    <span style={{ fontSize: 11.5, fontFamily: "var(--font-mono)", color: "var(--text-secondary)" }}>
                      {item.extension.toUpperCase()}
                    </span>
                  </div>
                ) : item.kind === "subtitle" ? (
                  <div style={{ display: "flex", flexDirection: "column", alignItems: "center", gap: 6, color: "var(--accent)" }}>
                    <Subtitles size={34} />
                    <span style={{ fontSize: 11.5, fontFamily: "var(--font-mono)", color: "var(--text-secondary)" }}>
                      .{item.extension.toUpperCase()}
                    </span>
                  </div>
                ) : item.kind === "timeline" ? (
                  <div style={{ display: "flex", flexDirection: "column", alignItems: "center", gap: 6, color: "var(--success)" }}>
                    <FileCode2 size={34} />
                    <span style={{ fontSize: 11.5, fontFamily: "var(--font-mono)", color: "var(--text-secondary)" }}>
                      .{item.extension.toUpperCase()}
                    </span>
                  </div>
                ) : item.kind === "transcript" ? (
                  <div style={{ display: "flex", flexDirection: "column", alignItems: "center", gap: 6, color: "var(--accent)" }}>
                    <FileText size={34} />
                    <span style={{ fontSize: 11.5, fontFamily: "var(--font-mono)", color: "var(--text-secondary)" }}>
                      .{item.extension.toUpperCase()}
                    </span>
                  </div>
                ) : (
                  <Film size={32} color="var(--text-muted)" />
                )}
                {kindBadge(item.kind)}
                {item.metadata.duration_sec !== undefined && (
                  <span className="media-duration">{formatDuration(item.metadata.duration_sec)}</span>
                )}
              </div>

              <div className="media-card-body">
                <h3 className="media-card-title" title={item.title}>
                  {item.title}
                </h3>
                <div style={{ display: "flex", flexWrap: "wrap", gap: 5 }}>
                  <span className={`pill ${item.source_type === "generated" ? "accent" : ""}`}>
                    {item.source_type === "download" ? (
                      <Download size={10} />
                    ) : item.source_type === "generated" ? (
                      <Sparkles size={10} />
                    ) : (
                      <HardDrive size={10} />
                    )}
                    {item.source_type === "download"
                      ? "yt-dlp"
                      : item.source_type === "generated"
                        ? "IA Local"
                        : "PC"}
                  </span>
                  {item.metadata.height && <span className="pill accent">{item.metadata.height}p</span>}
                  <span className="pill">{item.extension.toUpperCase()}</span>
                  <span className="pill">{formatBytes(item.file_size_bytes)}</span>
                </div>
                {item.tags.length > 0 && (
                  <div style={{ display: "flex", flexWrap: "wrap", gap: 4 }}>
                    {item.tags.slice(0, 4).map((t) => (
                      <span key={t} style={{ fontSize: 11, color: "var(--text-muted)" }}>
                        #{t}
                      </span>
                    ))}
                  </div>
                )}
              </div>
            </article>
          ))}
        </div>
      )}

      {/* Media / Subtitle / Timeline Inspector Modal */}
      {selectedAsset && (
        <div className="modal-backdrop" onClick={() => onSelectMedia(null)}>
          <div className="modal-card" onClick={(e) => e.stopPropagation()}>
            <div
              style={{
                display: "flex",
                alignItems: "center",
                justifyContent: "space-between",
                padding: "14px 18px",
                borderBottom: "1px solid var(--border-subtle)",
              }}
            >
              <div style={{ display: "flex", alignItems: "center", gap: 10, minWidth: 0 }}>
                {isTextKind(selectedAsset.kind) ? (
                  <FileCode2 size={16} color="var(--accent)" />
                ) : (
                  <Play size={16} color="var(--accent)" />
                )}
                <strong style={{ overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
                  {selectedAsset.title}
                </strong>
              </div>
              <button type="button" className="btn btn-ghost btn-sm" onClick={() => onSelectMedia(null)}>
                <X size={16} />
              </button>
            </div>

            <div
              style={{
                overflowY: "auto",
                padding: 18,
                display: "grid",
                gridTemplateColumns: "1.25fr 1fr",
                gap: 20,
              }}
            >
              {/* Left: Player OR Text/Timeline Code Preview */}
              <div style={{ display: "flex", flexDirection: "column", gap: 14, minWidth: 0 }}>
                {isTextKind(selectedAsset.kind) ? (
                  <>
                    <div style={{ display: "flex", gap: 8, justifyContent: "space-between", alignItems: "center" }}>
                      <span className="pill accent">
                        {selectedAsset.metadata.format_name ?? selectedAsset.extension.toUpperCase()}
                      </span>
                      <div style={{ display: "flex", gap: 6 }}>
                        <button type="button" className="btn btn-sm" onClick={handleCopyText}>
                          <Copy size={13} /> Copiar
                        </button>
                        <a
                          href={mediaTextUrl(selectedAsset.id)}
                          download={selectedAsset.file_name}
                          className="btn btn-sm btn-primary"
                          style={{ textDecoration: "none" }}
                        >
                          <Download size={13} /> Baixar Arquivo
                        </a>
                      </div>
                    </div>
                    <pre className="code-block" style={{ maxHeight: 340, minHeight: 220 }}>
                      {loadingText ? "Carregando conteúdo..." : textPreview ?? ""}
                    </pre>
                  </>
                ) : selectedAsset.kind === "audio" ? (
                  <div
                    style={{
                      padding: 28,
                      borderRadius: "var(--radius-md)",
                      background: "var(--bg-input)",
                      border: "1px solid var(--border-subtle)",
                      display: "flex",
                      flexDirection: "column",
                      alignItems: "center",
                      gap: 16,
                    }}
                  >
                    <Music size={44} color="var(--accent)" />
                    <audio controls src={mediaStreamUrl(selectedAsset.id)} style={{ width: "100%" }} />
                  </div>
                ) : (
                  <video
                    controls
                    src={mediaStreamUrl(selectedAsset.id)}
                    poster={selectedAsset.thumbnail_path ? mediaThumbnailUrl(selectedAsset.id) : undefined}
                    style={{
                      width: "100%",
                      maxHeight: 360,
                      borderRadius: "var(--radius-md)",
                      background: "#000",
                    }}
                  />
                )}

                {!isTextKind(selectedAsset.kind) && (
                  <div style={{ display: "flex", gap: 8, flexWrap: "wrap" }}>
                    <button
                      type="button"
                      className="btn btn-primary"
                      style={{ flex: 1 }}
                      onClick={() => {
                        const id = selectedAsset.id;
                        onSelectMedia(null);
                        onTranscribeMedia(id);
                      }}
                    >
                      <Mic size={15} /> Transcrever IA Local
                    </button>
                    {(selectedAsset.kind === "video" || selectedAsset.kind === "video_only") && (
                      <button
                        type="button"
                        className="btn btn-primary"
                        style={{ flex: 1 }}
                        onClick={() => {
                          const id = selectedAsset.id;
                          onSelectMedia(null);
                          onSmartClipMedia(id);
                        }}
                      >
                        <Scissors size={15} /> Cortes Shorts IA (até 3 min)
                      </button>
                    )}
                  </div>
                )}

                {selectedAsset.parent_media_id && (
                  <button
                    type="button"
                    className="btn btn-sm"
                    onClick={() => onSelectMedia(selectedAsset.parent_media_id ?? null)}
                  >
                    <Film size={14} /> Abrir vídeo/áudio de origem
                  </button>
                )}

                <div
                  style={{
                    display: "grid",
                    gridTemplateColumns: "repeat(2, 1fr)",
                    gap: 8,
                    padding: 12,
                    borderRadius: "var(--radius-md)",
                    background: "var(--bg-elevated)",
                    fontSize: 12.5,
                  }}
                >
                  <div>
                    <span style={{ color: "var(--text-muted)" }}>Tamanho: </span>
                    {formatBytes(selectedAsset.file_size_bytes)}
                  </div>
                  {selectedAsset.metadata.duration_sec !== undefined && (
                    <div>
                      <span style={{ color: "var(--text-muted)" }}>Duração: </span>
                      {formatDuration(selectedAsset.metadata.duration_sec)}
                    </div>
                  )}
                  {selectedAsset.metadata.width && selectedAsset.metadata.height && (
                    <div>
                      <span style={{ color: "var(--text-muted)" }}>Resolução: </span>
                      {selectedAsset.metadata.width}×{selectedAsset.metadata.height}
                      {selectedAsset.metadata.fps ? ` @ ${selectedAsset.metadata.fps}fps` : ""}
                    </div>
                  )}
                  {selectedAsset.metadata.model_id && (
                    <div>
                      <span style={{ color: "var(--text-muted)" }}>Modelo IA: </span>
                      {selectedAsset.metadata.model_id}
                    </div>
                  )}
                  {selectedAsset.metadata.language && (
                    <div>
                      <span style={{ color: "var(--text-muted)" }}>Idioma: </span>
                      {selectedAsset.metadata.language.toUpperCase()}
                    </div>
                  )}
                  <div>
                    <span style={{ color: "var(--text-muted)" }}>Criado: </span>
                    {formatRelativeDate(selectedAsset.created_at)}
                  </div>
                </div>

                <div
                  style={{
                    fontSize: 12,
                    fontFamily: "var(--font-mono)",
                    color: "var(--text-secondary)",
                    background: "var(--bg-input)",
                    padding: "8px 10px",
                    borderRadius: "var(--radius-sm)",
                    wordBreak: "break-all",
                  }}
                >
                  {selectedAsset.file_path}
                </div>

                {selectedAsset.source_url && (
                  <a
                    href={selectedAsset.source_url}
                    target="_blank"
                    rel="noreferrer"
                    style={{
                      display: "inline-flex",
                      alignItems: "center",
                      gap: 6,
                      fontSize: 12.5,
                      color: "var(--accent)",
                      textDecoration: "none",
                    }}
                  >
                    <ExternalLink size={13} /> Abrir URL de origem
                  </a>
                )}
              </div>

              {/* Right: Metadata Editor & Safe Delete */}
              <div style={{ display: "flex", flexDirection: "column", gap: 14 }}>
                <div className="field">
                  <label>Título</label>
                  <input
                    className="input"
                    value={editTitle}
                    onChange={(e) => setEditTitle(e.target.value)}
                  />
                </div>
                <div className="field">
                  <label>Tags (separadas por vírgula)</label>
                  <input
                    className="input"
                    value={editTags}
                    onChange={(e) => setEditTags(e.target.value)}
                  />
                </div>
                <div className="field">
                  <label>Notas (visíveis também para a IA)</label>
                  <textarea
                    className="textarea"
                    rows={4}
                    value={editNotes}
                    onChange={(e) => setEditNotes(e.target.value)}
                    style={{ fontFamily: "var(--font-sans)" }}
                  />
                </div>

                <button
                  type="button"
                  className="btn btn-primary"
                  disabled={saving}
                  onClick={handleSaveEdits}
                >
                  <Save size={14} /> Salvar Metadados
                </button>

                <div
                  style={{
                    marginTop: "auto",
                    paddingTop: 14,
                    borderTop: "1px solid var(--border-subtle)",
                    display: "flex",
                    flexDirection: "column",
                    gap: 10,
                  }}
                >
                  <label className="check-row">
                    <input
                      type="checkbox"
                      checked={deleteFileOnDisk}
                      onChange={(e) => setDeleteFileOnDisk(e.target.checked)}
                    />
                    Também apagar o arquivo físico do disco
                  </label>
                  <button type="button" className="btn btn-danger" onClick={handleDelete}>
                    <Trash2 size={14} /> Remover da Biblioteca
                  </button>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
