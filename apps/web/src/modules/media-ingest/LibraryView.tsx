import React, { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  ArrowDown,
  ArrowUp,
  Captions,
  CheckSquare,
  Copy,
  Download,
  ExternalLink,
  FileCode2,
  FileText,
  Film,
  FolderInput,
  HardDrive,
  LayoutGrid,
  List,
  Mic,
  Music,
  RefreshCw,
  RotateCcw,
  Save,
  Scissors,
  Search,
  Sparkles,
  Square,
  Trash2,
  Video,
  VideoOff,
  X,
  type LucideIcon,
} from "lucide-react";
import { fetchMediaText, mediaStreamUrl, mediaTextUrl, mediaThumbnailUrl, runAction } from "../../lib/client";
import { formatBytes, formatDuration, formatRelativeDate } from "../../lib/format";
import type { MediaAsset, MediaKind, MediaSourceType, NotifyFn, RouteId } from "../../lib/types";
import { copyToClipboard, normalizeSearch, splitList, storage } from "../../lib/ui";
import {
  Badge,
  BusyIcon,
  Checkbox,
  CodeBlock,
  EmptyState,
  Field,
  KeyValueGrid,
  MoreButton,
  ResizeHandle,
  SegmentedControl,
  Tabs,
  usePersistentWidth,
  useContextMenu,
  type MenuItems,
  type Tone,
} from "../../components";
import "./library.css";

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
  onNotify: NotifyFn;
}

const isTextKind = (kind: MediaKind) => kind === "subtitle" || kind === "timeline" || kind === "transcript";
const isPlayable = (kind: MediaKind) => kind === "video" || kind === "video_only" || kind === "audio";
const isVideo = (kind: MediaKind) => kind === "video" || kind === "video_only";

const KIND_META: Record<MediaKind, { label: string; icon: LucideIcon; tone: Tone }> = {
  video: { label: "vídeo", icon: Video, tone: "accent" },
  video_only: { label: "só vídeo", icon: VideoOff, tone: "accent" },
  audio: { label: "áudio", icon: Music, tone: "magenta" },
  subtitle: { label: "legenda", icon: Captions, tone: "info" },
  timeline: { label: "timeline", icon: FileCode2, tone: "success" },
  transcript: { label: "transcrição", icon: FileText, tone: "warning" },
};

const SOURCE_META: Record<MediaSourceType, { label: string; icon: LucideIcon }> = {
  download: { label: "yt-dlp", icon: Download },
  local: { label: "pc", icon: HardDrive },
  generated: { label: "ia local", icon: Sparkles },
};

const KIND_FILTERS: Array<{ id: MediaKind | null; label: string }> = [
  { id: null, label: "Todos" },
  { id: "video", label: "Vídeo" },
  { id: "video_only", label: "Só vídeo" },
  { id: "audio", label: "Áudio" },
  { id: "subtitle", label: "Legendas" },
  { id: "timeline", label: "Timelines" },
  { id: "transcript", label: "Transcrições" },
];

type SortKey = "newest" | "oldest" | "title" | "duration" | "size";
type ViewMode = "library" | "trash";

function isTypingTarget(target: EventTarget | null) {
  return (
    target instanceof HTMLElement &&
    (target.isContentEditable || ["INPUT", "TEXTAREA", "SELECT"].includes(target.tagName))
  );
}

/** Native checkbox with support for the "some selected" (indeterminate) state. */
function SelectBox({
  checked,
  indeterminate = false,
  label,
  onToggle,
}: {
  checked: boolean;
  indeterminate?: boolean;
  label: string;
  onToggle: () => void;
}) {
  return (
    <input
      type="checkbox"
      className="checkbox"
      aria-label={label}
      title={label}
      checked={checked}
      ref={(el) => {
        if (el) el.indeterminate = indeterminate;
      }}
      onClick={(e) => e.stopPropagation()}
      onChange={onToggle}
    />
  );
}

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
  const menu = useContextMenu();
  const searchRef = useRef<HTMLInputElement>(null);
  const inspectorRef = useRef<HTMLElement>(null);

  // Library vs. Trash, multi-selection & bulk actions
  const [viewMode, setViewMode] = useState<ViewMode>("library");
  const [trashedItems, setTrashedItems] = useState<MediaAsset[]>([]);
  const [trashTotalBytes, setTrashTotalBytes] = useState(0);
  const [selectedIds, setSelectedIds] = useState<Set<string>>(new Set());
  const [bulkBusy, setBulkBusy] = useState(false);

  const [sourceFilter, setSourceFilter] = useState<"all" | MediaSourceType>("all");
  const [sortBy, setSortBy] = useState<SortKey>("newest");
  const [view, setView] = useState<"list" | "grid">(() => (storage.get("aicut_library_view") === "grid" ? "grid" : "list"));
  const [inspectorWidth, setInspectorWidth] = usePersistentWidth("aicut_inspector_w", 440, 340, 760);

  const loadTrash = useCallback(async () => {
    try {
      const res = await runAction<{
        total: number;
        total_size_bytes: number;
        items: MediaAsset[];
      }>("media.trash_list");
      setTrashedItems(res.data.items ?? []);
      setTrashTotalBytes(res.data.total_size_bytes ?? 0);
    } catch {
      // ignore if offline
    }
  }, []);

  useEffect(() => {
    void loadTrash();
  }, [loadTrash, media.length]);

  const isTrash = viewMode === "trash";
  const activeList = isTrash ? trashedItems : media;
  const selectedAsset = activeList.find((m) => m.id === selectedMediaId) ?? null;

  const [editTitle, setEditTitle] = useState("");
  const [editTags, setEditTags] = useState("");
  const [editNotes, setEditNotes] = useState("");
  const [deleteFileOnDisk, setDeleteFileOnDisk] = useState(false);
  const [saving, setSaving] = useState(false);
  const [textPreview, setTextPreview] = useState<string | null>(null);
  const [loadingText, setLoadingText] = useState(false);

  useEffect(() => {
    storage.set("aicut_library_view", view);
  }, [view]);

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

  const kindCounts = useMemo(() => {
    const counts = new Map<string, number>();
    for (const m of activeList) counts.set(m.kind, (counts.get(m.kind) ?? 0) + 1);
    return counts;
  }, [activeList]);

  const filtered = useMemo(() => {
    const q = normalizeSearch(searchQuery.trim());
    return activeList
      .filter((item) => {
        if (filterKind && item.kind !== filterKind) return false;
        if (sourceFilter !== "all" && item.source_type !== sourceFilter) return false;
        if (q) {
          const hit =
            normalizeSearch(item.title).includes(q) ||
            normalizeSearch(item.file_name).includes(q) ||
            normalizeSearch(item.source_url ?? "").includes(q) ||
            item.tags.some((t) => normalizeSearch(t).includes(q));
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
  }, [activeList, filterKind, sourceFilter, searchQuery, sortBy]);

  const totalBytes = useMemo(() => filtered.reduce((sum, m) => sum + m.file_size_bytes, 0), [filtered]);
  const selectedBytes = useMemo(
    () => activeList.filter((m) => selectedIds.has(m.id)).reduce((sum, m) => sum + m.file_size_bytes, 0),
    [activeList, selectedIds],
  );

  /* ---------- Selection ---------- */
  const allFilteredSelected = filtered.length > 0 && filtered.every((item) => selectedIds.has(item.id));
  const someFilteredSelected = !allFilteredSelected && filtered.some((item) => selectedIds.has(item.id));

  function toggleSelectItem(id: string) {
    setSelectedIds((prev) => {
      const next = new Set(prev);
      if (next.has(id)) {
        next.delete(id);
      } else {
        next.add(id);
      }
      return next;
    });
  }

  function toggleSelectAllFiltered() {
    if (allFilteredSelected) {
      setSelectedIds(new Set());
    } else {
      setSelectedIds(new Set(filtered.map((i) => i.id)));
    }
  }

  function switchViewMode(nextMode: ViewMode) {
    setViewMode(nextMode);
    setSelectedIds(new Set());
    onSelectMedia(null);
    if (nextMode === "trash") {
      void loadTrash();
    }
  }

  /* ---------- Mutations ---------- */
  async function handleBulkDelete(deleteFile: boolean) {
    const ids = Array.from(selectedIds);
    if (ids.length === 0) return;
    setBulkBusy(true);
    try {
      const res = await runAction<{
        processed_ids: string[];
        files_deleted: number;
        freed_bytes: number;
      }>("media.bulk_delete", {
        ids,
        delete_file: deleteFile,
      });
      setSelectedIds(new Set());
      onRefresh();
      await loadTrash();
      onNotify(
        "success",
        deleteFile ? "Itens apagados permanentemente" : "Movidos para a Lixeira",
        deleteFile
          ? `${res.data.processed_ids.length} item(ns) apagado(s) (${formatBytes(res.data.freed_bytes ?? 0)} liberados no disco).`
          : `${res.data.processed_ids.length} item(ns) movido(s) para a Lixeira (arquivos preservados no disco).`,
      );
    } catch (e) {
      onNotify("error", "Erro na remoção em lote", (e as Error).message);
    } finally {
      setBulkBusy(false);
    }
  }

  async function handleRestoreIds(ids: string[]) {
    if (ids.length === 0) return;
    setBulkBusy(true);
    try {
      const res = await runAction<{ restored_ids: string[] }>("media.restore", { ids });
      setSelectedIds(new Set());
      onRefresh();
      await loadTrash();
      onNotify("success", "Restaurado da Lixeira", `${res.data.restored_ids.length} item(ns) restaurado(s) para a Biblioteca.`);
    } catch (e) {
      onNotify("error", "Erro ao restaurar", (e as Error).message);
    } finally {
      setBulkBusy(false);
    }
  }

  async function handlePurgeTrash(ids?: string[], deleteFiles = true) {
    setBulkBusy(true);
    try {
      const res = await runAction<{
        purged_count: number;
        files_deleted: number;
        freed_bytes: number;
      }>("media.trash_empty", {
        ids: ids && ids.length > 0 ? ids : undefined,
        delete_files: deleteFiles,
      });
      setSelectedIds(new Set());
      await loadTrash();
      onNotify(
        "info",
        "Lixeira limpa",
        `${res.data.purged_count} item(ns) excluído(s) permanentemente (${formatBytes(res.data.freed_bytes ?? 0)} liberados no disco).`,
      );
    } catch (e) {
      onNotify("error", "Erro ao limpar Lixeira", (e as Error).message);
    } finally {
      setBulkBusy(false);
    }
  }

  async function handleSaveEdits() {
    if (!selectedAsset) return;
    setSaving(true);
    try {
      await runAction("media.update", {
        id: selectedAsset.id,
        title: editTitle.trim() || selectedAsset.title,
        tags: splitList(editTags),
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

  /** Single item: soft delete to the Trash, or permanent deletion from disk. */
  async function deleteAsset(asset: MediaAsset, deleteFile: boolean) {
    try {
      await runAction("media.delete", { id: asset.id, delete_file: deleteFile });
      if (selectedMediaId === asset.id) onSelectMedia(null);
      onRefresh();
      await loadTrash();
      onNotify(
        "success",
        deleteFile ? "Mídia apagada permanentemente" : "Movida para a Lixeira",
        deleteFile
          ? "Removida da Biblioteca e apagada do disco."
          : "Movida para a Lixeira do Aicut (você pode restaurar ou apagar definitivamente por lá).",
      );
    } catch (e) {
      onNotify("error", "Erro ao remover", (e as Error).message);
    }
  }

  const transcribe = (id: string) => {
    onSelectMedia(null);
    onTranscribeMedia(id);
  };
  const smartClip = (id: string) => {
    onSelectMedia(null);
    onSmartClipMedia(id);
  };

  /* ---------- Keyboard: / ↑ ↓ Esc Space Delete Ctrl+A ---------- */
  const live = useRef({ filtered, selectedIds, selectedMediaId, selectedAsset, isTrash });
  live.current = { filtered, selectedIds, selectedMediaId, selectedAsset, isTrash };
  const actions = useRef({ toggleSelectAllFiltered, toggleSelectItem, handleBulkDelete, deleteAsset });
  actions.current = { toggleSelectAllFiltered, toggleSelectItem, handleBulkDelete, deleteAsset };

  useEffect(() => {
    const onKeyDown = (e: KeyboardEvent) => {
      if (e.defaultPrevented || e.altKey || isTypingTarget(e.target)) return;
      if (document.querySelector(".palette, .modal, .ctx-menu")) return;
      const s = live.current;
      const mod = e.ctrlKey || e.metaKey;

      if (mod && e.key.toLowerCase() === "a") {
        e.preventDefault();
        actions.current.toggleSelectAllFiltered();
        return;
      }
      if (mod) return;

      if (e.key === "/") {
        e.preventDefault();
        searchRef.current?.focus();
        searchRef.current?.select();
      } else if ((e.key === "ArrowDown" || e.key === "ArrowUp") && !s.isTrash && s.filtered.length > 0) {
        e.preventDefault();
        const list = s.filtered;
        const idx = list.findIndex((m) => m.id === s.selectedMediaId);
        const step = e.key === "ArrowDown" ? 1 : -1;
        const next = idx === -1 ? (step === 1 ? 0 : list.length - 1) : Math.min(list.length - 1, Math.max(0, idx + step));
        onSelectMedia(list[next].id);
        document.querySelector(`[data-media-id="${CSS.escape(list[next].id)}"]`)?.scrollIntoView({ block: "nearest" });
      } else if (e.key === " " && s.selectedMediaId && !s.isTrash) {
        e.preventDefault();
        actions.current.toggleSelectItem(s.selectedMediaId);
      } else if (e.key === "Delete" && !s.isTrash) {
        if (s.selectedIds.size > 0) {
          e.preventDefault();
          void actions.current.handleBulkDelete(false);
        } else if (s.selectedAsset) {
          e.preventDefault();
          void actions.current.deleteAsset(s.selectedAsset, false);
        }
      } else if (e.key === "Escape") {
        if (s.selectedMediaId) onSelectMedia(null);
        else if (s.selectedIds.size > 0) setSelectedIds(new Set());
      }
    };
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [onSelectMedia]);

  // In stacked (narrow) layouts the inspector sits above the list — bring it into view
  useEffect(() => {
    if (selectedMediaId) inspectorRef.current?.scrollIntoView({ block: "nearest" });
  }, [selectedMediaId]);

  /* ---------- Context menus ---------- */
  const selectionToggle = (item: MediaAsset) =>
    selectedIds.has(item.id)
      ? { label: "Desmarcar", icon: Square, shortcut: isTrash ? undefined : "Espaço", onSelect: () => toggleSelectItem(item.id) }
      : { label: "Selecionar", icon: CheckSquare, shortcut: isTrash ? undefined : "Espaço", onSelect: () => toggleSelectItem(item.id) };

  const copyItems = (item: MediaAsset): MenuItems => [
    { label: "Copiar caminho do arquivo", icon: Copy, onSelect: () => void copyToClipboard(item.file_path, onNotify, "Caminho") },
    { label: "Copiar título", icon: Copy, onSelect: () => void copyToClipboard(item.title, onNotify, "Título") },
    item.source_url && {
      label: "Copiar URL de origem",
      icon: Copy,
      onSelect: () => void copyToClipboard(item.source_url ?? "", onNotify, "URL"),
    },
    { label: "Copiar ID", icon: Copy, onSelect: () => void copyToClipboard(item.id, onNotify, "ID da mídia") },
  ];

  const libraryMenu = (item: MediaAsset): MenuItems => {
    const kind = KIND_META[item.kind];
    return [
      { type: "label", label: `${kind.label} · ${item.extension.toUpperCase()} · ${formatBytes(item.file_size_bytes)}` },
      { label: "Abrir no inspetor", icon: kind.icon, onSelect: () => onSelectMedia(item.id) },
      isPlayable(item.kind) && { label: "Transcrever com IA local", icon: Mic, onSelect: () => transcribe(item.id) },
      isVideo(item.kind) && { label: "Cortes Shorts com IA", icon: Scissors, onSelect: () => smartClip(item.id) },
      item.parent_media_id && {
        label: "Abrir mídia de origem",
        icon: FolderInput,
        onSelect: () => onSelectMedia(item.parent_media_id ?? null),
      },
      item.source_url && {
        label: "Abrir URL de origem",
        icon: ExternalLink,
        onSelect: () => window.open(item.source_url, "_blank", "noopener"),
      },
      isTextKind(item.kind) && {
        label: "Baixar arquivo",
        icon: Download,
        onSelect: () => {
          const a = document.createElement("a");
          a.href = mediaTextUrl(item.id);
          a.download = item.file_name;
          a.click();
        },
      },
      { type: "separator" },
      selectionToggle(item),
      {
        label: `Mostrar só "${kind.label}"`,
        icon: kind.icon,
        disabled: filterKind === item.kind,
        onSelect: () => onFilterKindChange(item.kind),
      },
      { type: "separator" },
      ...copyItems(item),
      { type: "separator" },
      { label: "Mover para a Lixeira", icon: Trash2, shortcut: "Del", onSelect: () => void deleteAsset(item, false) },
    ];
  };

  const trashMenu = (item: MediaAsset): MenuItems => [
    { type: "label", label: `na lixeira · ${formatBytes(item.file_size_bytes)}` },
    {
      label: "Restaurar para a Biblioteca",
      icon: RotateCcw,
      disabled: bulkBusy,
      onSelect: () => void handleRestoreIds([item.id]),
    },
    selectionToggle(item),
    { type: "separator" },
    ...copyItems(item),
    { type: "separator" },
    {
      label: "Excluir permanentemente (disco)",
      icon: Trash2,
      danger: true,
      disabled: bulkBusy,
      onSelect: () => void handlePurgeTrash([item.id], true),
    },
  ];

  const bulkMenu = (): MenuItems => {
    const ids = Array.from(selectedIds);
    const header = { type: "label" as const, label: `${ids.length} selecionado(s) · ${formatBytes(selectedBytes)}` };
    const clear = { label: "Limpar seleção", icon: Square, shortcut: "Esc", onSelect: () => setSelectedIds(new Set()) };
    return isTrash
      ? [
          header,
          { label: `Restaurar ${ids.length} item(ns)`, icon: RotateCcw, disabled: bulkBusy, onSelect: () => void handleRestoreIds(ids) },
          clear,
          { type: "separator" },
          {
            label: `Excluir ${ids.length} permanentemente (disco)`,
            icon: Trash2,
            danger: true,
            disabled: bulkBusy,
            onSelect: () => void handlePurgeTrash(ids, true),
          },
        ]
      : [
          header,
          {
            label: `Mover ${ids.length} para a Lixeira`,
            icon: Trash2,
            shortcut: "Del",
            disabled: bulkBusy,
            onSelect: () => void handleBulkDelete(false),
          },
          clear,
          { type: "separator" },
          {
            label: `Apagar ${ids.length} permanentemente (disco)`,
            icon: Trash2,
            danger: true,
            disabled: bulkBusy,
            onSelect: () => void handleBulkDelete(true),
          },
        ];
  };

  /** Right click on a selected item acts on the whole selection, like a file manager. */
  const itemMenu = (item: MediaAsset): MenuItems =>
    selectedIds.size > 1 && selectedIds.has(item.id) ? bulkMenu() : isTrash ? trashMenu(item) : libraryMenu(item);

  const onItemClick = (e: React.MouseEvent, item: MediaAsset) => {
    if (isTrash || e.ctrlKey || e.metaKey) {
      toggleSelectItem(item.id);
    } else {
      onSelectMedia(item.id);
    }
  };

  /* ---------- Small renderers ---------- */
  const sortHeader = (label: string, key: SortKey, alt?: SortKey) => {
    const active = sortBy === key || sortBy === alt;
    const Arrow = sortBy === "oldest" || sortBy === "title" ? ArrowUp : ArrowDown;
    return (
      <button
        type="button"
        className="th-sort"
        data-active={active || undefined}
        onClick={() => setSortBy(alt && sortBy === key ? alt : key)}
        title={`Ordenar por ${label}`}
      >
        {label}
        {active && <Arrow size={11} aria-hidden />}
      </button>
    );
  };

  const thumb = (item: MediaAsset, size: "sm" | "lg") => {
    const kind = KIND_META[item.kind];
    const Icon = kind.icon;
    if (item.thumbnail_path && !isTextKind(item.kind)) {
      return <img src={mediaThumbnailUrl(item.id)} alt="" loading="lazy" />;
    }
    return (
      <span className="thumb-placeholder" data-tone={kind.tone}>
        <Icon size={size === "sm" ? 14 : 26} aria-hidden />
        {size === "lg" && <span className="thumb-ext">.{item.extension}</span>}
      </span>
    );
  };

  const sourceBadge = (item: MediaAsset) => {
    const src = SOURCE_META[item.source_type];
    return (
      <span className={item.source_type === "generated" ? "src text-magenta" : "src"}>
        <src.icon size={11} aria-hidden /> {src.label}
      </span>
    );
  };

  const trashRowActions = (item: MediaAsset) => (
    <>
      <button
        type="button"
        className="btn btn-ghost btn-icon btn-xs"
        aria-label="Restaurar"
        title="Restaurar para a Biblioteca"
        disabled={bulkBusy}
        onClick={(e) => {
          e.stopPropagation();
          void handleRestoreIds([item.id]);
        }}
      >
        <RotateCcw aria-hidden />
      </button>
      <button
        type="button"
        className="btn btn-ghost btn-icon btn-xs text-danger"
        aria-label="Excluir permanentemente"
        title="Excluir permanentemente do disco"
        disabled={bulkBusy}
        onClick={(e) => {
          e.stopPropagation();
          void handlePurgeTrash([item.id], true);
        }}
      >
        <Trash2 aria-hidden />
      </button>
    </>
  );

  return (
    <>
      <div className="row row-between wrap gap-sm">
        <Tabs
          ariaLabel="Biblioteca ou Lixeira"
          value={viewMode}
          onChange={switchViewMode}
          items={[
            { id: "library", label: "Biblioteca", icon: Film, count: media.length },
            { id: "trash", label: "Lixeira", icon: Trash2, count: trashedItems.length },
          ]}
        />
        <div className="row gap-xs">
          <button
            type="button"
            className="btn btn-sm"
            onClick={() => {
              onRefresh();
              void loadTrash();
            }}
          >
            <RefreshCw aria-hidden /> Atualizar
          </button>
          {isTrash && trashedItems.length > 0 && (
            <button
              type="button"
              className="btn btn-danger btn-sm"
              disabled={bulkBusy}
              onClick={() => handlePurgeTrash(undefined, true)}
            >
              <Trash2 aria-hidden /> Esvaziar Lixeira ({formatBytes(trashTotalBytes)})
            </button>
          )}
        </div>
      </div>

      {/* Toolbar */}
      <div className="toolbar library-toolbar">
        <div className="input-wrap library-search">
          <Search aria-hidden />
          <input
            ref={searchRef}
            className="input input-sm"
            placeholder={isTrash ? "buscar na lixeira…" : "buscar título, arquivo, tag ou URL…"}
            aria-label="Buscar na Biblioteca"
            aria-keyshortcuts="/"
            value={searchQuery}
            onChange={(e) => onSearchChange(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Escape") {
                e.currentTarget.blur();
                if (searchQuery) onSearchChange("");
              }
            }}
          />
          {!searchQuery && <kbd className="kbd search-kbd">/</kbd>}
        </div>

        <div className="row wrap gap-xs" role="group" aria-label="Filtrar por tipo">
          {KIND_FILTERS.map((opt) => {
            const count = opt.id ? (kindCounts.get(opt.id) ?? 0) : activeList.length;
            return (
              <button
                key={String(opt.id)}
                type="button"
                className="chip chip-sm"
                aria-pressed={filterKind === opt.id}
                onClick={() => onFilterKindChange(opt.id)}
              >
                {opt.label}
                <span className="chip-count">{count}</span>
              </button>
            );
          })}
        </div>

        <div className="row ml-auto">
          {view === "grid" && (
            <button
              type="button"
              className="chip chip-sm"
              aria-pressed={allFilteredSelected}
              disabled={filtered.length === 0}
              onClick={toggleSelectAllFiltered}
              title="Selecionar tudo (Ctrl+A)"
            >
              {allFilteredSelected ? <CheckSquare aria-hidden /> : <Square aria-hidden />}
              tudo ({filtered.length})
            </button>
          )}
          <select
            className="select select-sm select-auto"
            aria-label="Filtrar por origem"
            value={sourceFilter}
            onChange={(e) => setSourceFilter(e.target.value as "all" | MediaSourceType)}
          >
            <option value="all">Todas as origens</option>
            <option value="download">Baixados (yt-dlp)</option>
            <option value="local">Importados (PC)</option>
            <option value="generated">Gerados por IA</option>
          </select>
          <select
            className="select select-sm select-auto"
            aria-label="Ordenar"
            value={sortBy}
            onChange={(e) => setSortBy(e.target.value as SortKey)}
          >
            <option value="newest">Mais recentes</option>
            <option value="oldest">Mais antigos</option>
            <option value="title">Título A–Z</option>
            <option value="duration">Maior duração</option>
            <option value="size">Maior tamanho</option>
          </select>
          <SegmentedControl
            ariaLabel="Modo de exibição"
            size="sm"
            value={view}
            onChange={setView}
            options={[
              { id: "list", label: "", icon: List, title: "Lista" },
              { id: "grid", label: "", icon: LayoutGrid, title: "Grade" },
            ]}
          />
        </div>
      </div>

      {/* Bulk selection bar */}
      {selectedIds.size > 0 && (
        <div className="selection-bar" role="region" aria-label="Ações em lote">
          <SelectBox
            checked={allFilteredSelected}
            indeterminate={someFilteredSelected}
            label={allFilteredSelected ? "Desmarcar todos" : "Selecionar tudo"}
            onToggle={toggleSelectAllFiltered}
          />
          <span className="strong">{selectedIds.size} selecionado(s)</span>
          <span className="muted">{formatBytes(selectedBytes)}</span>
          <span className="toolbar-sep" />
          {isTrash ? (
            <>
              <button
                type="button"
                className="btn btn-primary btn-sm"
                disabled={bulkBusy}
                onClick={() => handleRestoreIds(Array.from(selectedIds))}
              >
                <RotateCcw aria-hidden /> Restaurar ({selectedIds.size})
              </button>
              <button
                type="button"
                className="btn btn-danger btn-sm"
                disabled={bulkBusy}
                onClick={() => handlePurgeTrash(Array.from(selectedIds), true)}
              >
                <Trash2 aria-hidden /> Excluir permanente (disco)
              </button>
            </>
          ) : (
            <>
              <button type="button" className="btn btn-sm" disabled={bulkBusy} onClick={() => handleBulkDelete(false)}>
                <Trash2 aria-hidden /> Mover p/ Lixeira ({selectedIds.size})
              </button>
              <button type="button" className="btn btn-danger btn-sm" disabled={bulkBusy} onClick={() => handleBulkDelete(true)}>
                <Trash2 aria-hidden /> Apagar permanente (disco)
              </button>
            </>
          )}
          <button type="button" className="btn btn-ghost btn-sm ml-auto" onClick={() => setSelectedIds(new Set())}>
            <X aria-hidden /> Limpar seleção
          </button>
        </div>
      )}

      <div
        className="library-layout"
        data-inspector={selectedAsset ? "" : undefined}
        style={{ "--inspector-w": `${inspectorWidth}px` } as React.CSSProperties}
      >
        <div className="library-results">
          {filtered.length === 0 ? (
            <div className="panel">
              <EmptyState
                icon={isTrash ? Trash2 : Film}
                title={
                  isTrash
                    ? trashedItems.length === 0
                      ? "a lixeira está vazia"
                      : "nenhum item na lixeira para o filtro atual"
                    : media.length === 0
                      ? "biblioteca vazia"
                      : "nenhum resultado"
                }
                description={
                  isTrash
                    ? "Itens removidos sem apagar do disco aparecem aqui para você restaurar ou excluir definitivamente quando quiser."
                    : media.length === 0
                      ? "Baixe um vídeo com yt-dlp ou importe arquivos do PC — depois transcreva para gerar legendas e timelines."
                      : "Limpe a busca ou troque os filtros de tipo e origem."
                }
                actions={
                  !isTrash && media.length === 0 ? (
                    <button type="button" className="btn btn-primary" onClick={() => onNavigate("ingest")}>
                      Ir para Baixar & Importar
                    </button>
                  ) : activeList.length > 0 ? (
                    <button
                      type="button"
                      className="btn"
                      onClick={() => {
                        onSearchChange("");
                        onFilterKindChange(null);
                        setSourceFilter("all");
                      }}
                    >
                      Limpar filtros
                    </button>
                  ) : undefined
                }
              />
            </div>
          ) : view === "list" ? (
            <div className="table-wrap">
              <table className="table media-table">
                <thead>
                  <tr>
                    <th className="col-fit">
                      <SelectBox
                        checked={allFilteredSelected}
                        indeterminate={someFilteredSelected}
                        label={allFilteredSelected ? "Desmarcar todos" : `Selecionar tudo (${filtered.length})`}
                        onToggle={toggleSelectAllFiltered}
                      />
                    </th>
                    <th className="col-fit" aria-label="Miniatura" />
                    <th>{sortHeader("título", "title")}</th>
                    <th className="col-fit c-kind">tipo</th>
                    <th className="col-fit num">{sortHeader("duração", "duration")}</th>
                    <th className="col-fit c-res">resolução</th>
                    <th className="col-fit c-ext">ext</th>
                    <th className="col-fit num">{sortHeader("tamanho", "size")}</th>
                    <th className="col-fit c-source">origem</th>
                    <th className="col-fit c-created">{isTrash ? "removido" : sortHeader("criado", "newest", "oldest")}</th>
                    <th className="col-fit" aria-label="Ações" />
                  </tr>
                </thead>
                <tbody>
                  {filtered.map((item) => {
                    const kind = KIND_META[item.kind];
                    const checked = selectedIds.has(item.id);
                    return (
                      <tr
                        key={item.id}
                        data-media-id={item.id}
                        data-clickable
                        data-checked={checked || undefined}
                        aria-selected={selectedMediaId === item.id || checked}
                        onClick={(e) => onItemClick(e, item)}
                        onContextMenu={(e) => menu.open(e, itemMenu(item))}
                      >
                        <td className="col-fit">
                          <SelectBox
                            checked={checked}
                            label={checked ? "Desmarcar" : "Selecionar"}
                            onToggle={() => toggleSelectItem(item.id)}
                          />
                        </td>
                        <td className="col-fit">
                          <span className="thumb-sm">{thumb(item, "sm")}</span>
                        </td>
                        <td className="cell-title" title={item.title}>
                          <span>{item.title}</span>
                          {item.tags.length > 0 && (
                            <span className="row-tags">
                              {item.tags.slice(0, 3).map((t) => (
                                <span key={t} className="tag">
                                  #{t}
                                </span>
                              ))}
                            </span>
                          )}
                        </td>
                        <td className="col-fit c-kind">
                          <Badge tone={kind.tone}>{kind.label}</Badge>
                        </td>
                        <td className="col-fit num">
                          {item.metadata.duration_sec !== undefined ? formatDuration(item.metadata.duration_sec) : "—"}
                        </td>
                        <td className="col-fit muted c-res">{item.metadata.height ? `${item.metadata.height}p` : "—"}</td>
                        <td className="col-fit muted c-ext">{item.extension}</td>
                        <td className="col-fit num">{formatBytes(item.file_size_bytes)}</td>
                        <td className="col-fit c-source">{sourceBadge(item)}</td>
                        <td className="col-fit muted c-created">
                          {formatRelativeDate(isTrash && item.deleted_at ? item.deleted_at : item.created_at)}
                        </td>
                        <td className="col-fit">
                          <span className="row-actions">
                            {isTrash && trashRowActions(item)}
                            <MoreButton items={() => itemMenu(item)} />
                          </span>
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          ) : (
            <div className="media-grid">
              {filtered.map((item) => {
                const kind = KIND_META[item.kind];
                const checked = selectedIds.has(item.id);
                return (
                  <article
                    key={item.id}
                    data-media-id={item.id}
                    className="media-card hover-host"
                    data-selected={selectedMediaId === item.id || checked || undefined}
                    data-checked={checked || undefined}
                    tabIndex={0}
                    onClick={(e) => onItemClick(e, item)}
                    onKeyDown={(e) => {
                      if (e.key === "Enter") onSelectMedia(item.id);
                    }}
                    onContextMenu={(e) => menu.open(e, itemMenu(item))}
                  >
                    <div className="media-thumb">
                      {thumb(item, "lg")}
                      <span className="card-check" data-visible={checked || selectedIds.size > 0 || undefined}>
                        <SelectBox
                          checked={checked}
                          label={checked ? "Desmarcar" : "Selecionar"}
                          onToggle={() => toggleSelectItem(item.id)}
                        />
                      </span>
                      <span className="media-overlay media-kind">
                        <kind.icon size={11} aria-hidden /> {kind.label}
                      </span>
                      {item.metadata.duration_sec !== undefined && (
                        <span className="media-overlay media-duration">{formatDuration(item.metadata.duration_sec)}</span>
                      )}
                      <span className="media-card-more row-actions">
                        <MoreButton items={() => itemMenu(item)} />
                      </span>
                    </div>
                    <div className="media-card-body">
                      <div className="media-card-title clamp-2" title={item.title}>
                        {item.title}
                      </div>
                      <div className="media-card-meta">
                        {sourceBadge(item)}
                        {item.metadata.height ? <span>{item.metadata.height}p</span> : null}
                        <span>{item.extension}</span>
                        <span>{formatBytes(item.file_size_bytes)}</span>
                      </div>
                      {isTrash && (
                        <div className="row gap-xs" onClick={(e) => e.stopPropagation()}>
                          <button
                            type="button"
                            className="btn btn-primary btn-xs grow"
                            disabled={bulkBusy}
                            onClick={() => handleRestoreIds([item.id])}
                          >
                            <RotateCcw aria-hidden /> Restaurar
                          </button>
                          <button
                            type="button"
                            className="btn btn-danger btn-xs"
                            disabled={bulkBusy}
                            title="Excluir permanentemente do disco"
                            aria-label="Excluir permanentemente do disco"
                            onClick={() => handlePurgeTrash([item.id], true)}
                          >
                            <Trash2 aria-hidden />
                          </button>
                        </div>
                      )}
                    </div>
                  </article>
                );
              })}
            </div>
          )}

          <div className="library-summary">
            <span>
              {filtered.length} de {activeList.length} item(ns) · {formatBytes(totalBytes)}
              {!isTrash && trashedItems.length > 0 && ` · lixeira: ${trashedItems.length} (${formatBytes(trashTotalBytes)})`}
            </span>
            {isTrash ? (
              <span className="muted">
                clique = selecionar · <kbd className="kbd">Ctrl</kbd>+<kbd className="kbd">A</kbd> tudo · botão direito = restaurar / excluir
              </span>
            ) : (
              <span className="muted">
                <kbd className="kbd">↑</kbd> <kbd className="kbd">↓</kbd> navegar · <kbd className="kbd">Ctrl</kbd>+clique /{" "}
                <kbd className="kbd">Espaço</kbd> selecionar · <kbd className="kbd">Del</kbd> lixeira · botão direito = ações
              </span>
            )}
          </div>
        </div>

        {/* Docked inspector */}
        {selectedAsset && (
          <aside
            ref={inspectorRef}
            className="inspector"
            aria-label={`Inspetor: ${selectedAsset.title}`}
            onContextMenu={(e) => {
              if ((e.target as HTMLElement).closest("input, textarea, video, audio")) return;
              menu.open(e, isTrash ? trashMenu(selectedAsset) : libraryMenu(selectedAsset));
            }}
          >
            <ResizeHandle
              width={inspectorWidth}
              min={340}
              max={760}
              resetTo={440}
              onChange={setInspectorWidth}
              label="Redimensionar inspetor"
            />
            <header className="inspector-header">
              <span className="icon-tile icon-tile-sm" data-tone={KIND_META[selectedAsset.kind].tone} aria-hidden>
                {React.createElement(KIND_META[selectedAsset.kind].icon)}
              </span>
              <div className="grow">
                <div className="inspector-title truncate" title={selectedAsset.title}>
                  {selectedAsset.title}
                </div>
                <div className="inspector-sub truncate">{selectedAsset.file_name}</div>
              </div>
              <MoreButton items={() => (isTrash ? trashMenu(selectedAsset) : libraryMenu(selectedAsset))} />
              <button
                type="button"
                className="btn btn-ghost btn-icon btn-xs"
                onClick={() => onSelectMedia(null)}
                aria-label="Fechar inspetor"
                title="Fechar (Esc)"
              >
                <X aria-hidden />
              </button>
            </header>

            <div className="inspector-body">
              {isTextKind(selectedAsset.kind) ? (
                <div className="stack stack-sm">
                  <div className="row row-between">
                    <Badge tone={KIND_META[selectedAsset.kind].tone}>
                      {selectedAsset.metadata.format_name ?? selectedAsset.extension.toUpperCase()}
                    </Badge>
                    <div className="row gap-xs">
                      <button
                        type="button"
                        className="btn btn-sm"
                        disabled={!textPreview}
                        onClick={() => textPreview && void copyToClipboard(textPreview, onNotify, "Conteúdo")}
                      >
                        <Copy aria-hidden /> Copiar
                      </button>
                      <a href={mediaTextUrl(selectedAsset.id)} download={selectedAsset.file_name} className="btn btn-sm btn-primary">
                        <Download aria-hidden /> Baixar arquivo
                      </a>
                    </div>
                  </div>
                  <CodeBlock code={loadingText ? "carregando conteúdo…" : (textPreview ?? "")} minHeight={200} maxHeight={360} />
                </div>
              ) : selectedAsset.kind === "audio" ? (
                <div className="audio-stage">
                  <Music size={30} aria-hidden />
                  <audio controls src={mediaStreamUrl(selectedAsset.id)} />
                </div>
              ) : (
                <video
                  className="inspector-video"
                  controls
                  src={mediaStreamUrl(selectedAsset.id)}
                  poster={selectedAsset.thumbnail_path ? mediaThumbnailUrl(selectedAsset.id) : undefined}
                />
              )}

              {!isTrash && (isPlayable(selectedAsset.kind) || selectedAsset.parent_media_id || selectedAsset.source_url) && (
                <div className="row wrap gap-xs">
                  {isPlayable(selectedAsset.kind) && (
                    <button type="button" className="btn btn-primary btn-sm" onClick={() => transcribe(selectedAsset.id)}>
                      <Mic aria-hidden /> Transcrever IA
                    </button>
                  )}
                  {isVideo(selectedAsset.kind) && (
                    <button type="button" className="btn btn-primary btn-sm" onClick={() => smartClip(selectedAsset.id)}>
                      <Scissors aria-hidden /> Cortes Shorts IA
                    </button>
                  )}
                  {selectedAsset.parent_media_id && (
                    <button
                      type="button"
                      className="btn btn-sm"
                      onClick={() => onSelectMedia(selectedAsset.parent_media_id ?? null)}
                    >
                      <FolderInput aria-hidden /> Mídia de origem
                    </button>
                  )}
                  {selectedAsset.source_url && (
                    <a className="btn btn-sm" href={selectedAsset.source_url} target="_blank" rel="noreferrer">
                      <ExternalLink aria-hidden /> URL de origem
                    </a>
                  )}
                </div>
              )}

              <section className="stack stack-sm">
                <div className="section-label">Detalhes</div>
                <KeyValueGrid
                  items={[
                    { label: "Tipo", value: KIND_META[selectedAsset.kind].label },
                    { label: "Origem", value: SOURCE_META[selectedAsset.source_type].label },
                    { label: "Tamanho", value: formatBytes(selectedAsset.file_size_bytes) },
                    selectedAsset.metadata.duration_sec !== undefined && {
                      label: "Duração",
                      value: formatDuration(selectedAsset.metadata.duration_sec),
                    },
                    selectedAsset.metadata.width &&
                      selectedAsset.metadata.height && {
                        label: "Resolução",
                        value: `${selectedAsset.metadata.width}×${selectedAsset.metadata.height}${
                          selectedAsset.metadata.fps ? ` @ ${selectedAsset.metadata.fps}fps` : ""
                        }`,
                      },
                    (selectedAsset.metadata.video_codec || selectedAsset.metadata.audio_codec) && {
                      label: "Codecs",
                      value: [selectedAsset.metadata.video_codec, selectedAsset.metadata.audio_codec].filter(Boolean).join(" / "),
                    },
                    selectedAsset.metadata.model_id && { label: "Modelo IA", value: selectedAsset.metadata.model_id },
                    selectedAsset.metadata.language && { label: "Idioma", value: selectedAsset.metadata.language.toUpperCase() },
                    { label: "Criado", value: formatRelativeDate(selectedAsset.created_at) },
                    selectedAsset.deleted_at && { label: "Na lixeira desde", value: formatRelativeDate(selectedAsset.deleted_at) },
                  ]}
                />
                <div className="row row-top gap-xs">
                  <code className="path grow">{selectedAsset.file_path}</code>
                  <button
                    type="button"
                    className="btn btn-ghost btn-icon btn-sm"
                    aria-label="Copiar caminho"
                    title="Copiar caminho"
                    onClick={() => void copyToClipboard(selectedAsset.file_path, onNotify, "Caminho")}
                  >
                    <Copy aria-hidden />
                  </button>
                </div>
              </section>

              {isTrash ? (
                <section className="stack stack-sm danger-zone">
                  <div className="section-label">Na lixeira</div>
                  <div className="row wrap gap-xs">
                    <button
                      type="button"
                      className="btn btn-primary btn-sm"
                      disabled={bulkBusy}
                      onClick={() => handleRestoreIds([selectedAsset.id])}
                    >
                      <RotateCcw aria-hidden /> Restaurar
                    </button>
                    <button
                      type="button"
                      className="btn btn-danger btn-sm"
                      disabled={bulkBusy}
                      onClick={() => handlePurgeTrash([selectedAsset.id], true)}
                    >
                      <Trash2 aria-hidden /> Excluir permanentemente (disco)
                    </button>
                  </div>
                </section>
              ) : (
                <>
                  <section className="stack stack-sm">
                    <div className="section-label">Metadados</div>
                    <Field label="Título" htmlFor="insp-title">
                      <input id="insp-title" className="input" value={editTitle} onChange={(e) => setEditTitle(e.target.value)} />
                    </Field>
                    <Field label="Tags (separadas por vírgula)" htmlFor="insp-tags">
                      <input id="insp-tags" className="input" value={editTags} onChange={(e) => setEditTags(e.target.value)} />
                    </Field>
                    <Field label="Notas (visíveis também para a IA)" htmlFor="insp-notes">
                      <textarea
                        id="insp-notes"
                        className="textarea"
                        rows={3}
                        value={editNotes}
                        onChange={(e) => setEditNotes(e.target.value)}
                      />
                    </Field>
                    <button type="button" className="btn btn-primary btn-sm" disabled={saving} onClick={handleSaveEdits}>
                      <BusyIcon busy={saving} icon={Save} size={13} /> Salvar metadados
                    </button>
                  </section>

                  <section className="stack stack-sm danger-zone">
                    <div className="section-label text-danger">Remover</div>
                    <Checkbox
                      checked={deleteFileOnDisk}
                      onChange={setDeleteFileOnDisk}
                      label="Apagar permanentemente do disco"
                      description="Se desmarcado, o item vai para a Lixeira e pode ser restaurado."
                    />
                    <button
                      type="button"
                      className={`btn btn-sm ${deleteFileOnDisk ? "btn-danger" : ""}`}
                      onClick={() => void deleteAsset(selectedAsset, deleteFileOnDisk)}
                    >
                      <Trash2 aria-hidden />
                      {deleteFileOnDisk ? "Apagar permanentemente (disco)" : "Mover para a Lixeira"}
                    </button>
                  </section>
                </>
              )}
            </div>
          </aside>
        )}
      </div>
    </>
  );
};
