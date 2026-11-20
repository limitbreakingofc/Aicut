import { useCallback, useEffect, useMemo, useRef, useState, type MouseEvent as ReactMouseEvent } from "react";
import { Command, Copy, Keyboard, PanelLeft, Palette, RefreshCw, Trash2 } from "lucide-react";
import {
  createEventSocket,
  fetchActions,
  fetchActivity,
  fetchModules,
  getPublicServerUrl,
  runAction,
} from "./lib/client";
import type {
  ActionSpec,
  ActivityEntry,
  HardwareProfile,
  JobRecord,
  MediaAsset,
  ModuleInfo,
  NotifyLevel,
  RouteId,
  ToolStatus,
  UiState,
  WorkspaceConfig,
} from "./lib/types";
import { copyToClipboard, storage, useMediaQuery } from "./lib/ui";
import { ContextMenuProvider, useContextMenu } from "./components";
import { ROUTES, ROUTE_MAP, routeDigit, routeForDigit } from "./app/navigation";
import { THEMES, isThemeId, themeLabel, type ThemeId } from "./app/themes";
import { Sidebar, type NavCount } from "./app/Sidebar";
import { Topbar } from "./app/Topbar";
import { StatusBar } from "./app/StatusBar";
import { CommandPalette, type PaletteCommand } from "./app/CommandPalette";
import { ShortcutsDialog } from "./app/ShortcutsDialog";
import { Toaster, type ToastItem } from "./app/Toaster";
import { IngestView } from "./modules/media-ingest/IngestView";
import { LibraryView } from "./modules/media-ingest/LibraryView";
import { TranscribeView } from "./modules/transcribe/TranscribeView";
import { SmartClipsView } from "./modules/smart-clips/SmartClipsView";
import { FramingStudioView } from "./modules/smart-clips/FramingStudioView";
import { MotionStudioView, type MotionSeed } from "./modules/motion-studio/MotionStudioView";
import { JobsView } from "./modules/core/JobsView";
import { ModulesView } from "./modules/core/ModulesView";
import { AiConnectView } from "./modules/core/AiConnectView";
import { SettingsView } from "./modules/core/SettingsView";

/** True when a keystroke belongs to a text field (shortcuts must not hijack typing). */
function isTypingTarget(target: EventTarget | null): boolean {
  if (!(target instanceof HTMLElement)) return false;
  if (target.isContentEditable) return true;
  if (target instanceof HTMLTextAreaElement || target instanceof HTMLSelectElement) return true;
  if (target instanceof HTMLInputElement) {
    return !["checkbox", "radio", "button", "submit", "range", "color", "file"].includes(target.type);
  }
  return false;
}

export function App() {
  return (
    <ContextMenuProvider>
      <AicutApp />
    </ContextMenuProvider>
  );
}

function AicutApp() {
  const ctxMenu = useContextMenu();
  const [route, setRoute] = useState<RouteId>("ingest");
  const [connected, setConnected] = useState(false);
  const [theme, setTheme] = useState<ThemeId>(() => {
    const saved = storage.get("aicut_theme");
    return isThemeId(saved) ? saved : "dark";
  });
  const [sidebarCollapsed, setSidebarCollapsed] = useState(() => storage.get("aicut_sidebar") === "collapsed");
  const sidebarLocked = useMediaQuery("(max-width: 860px)");

  const [media, setMedia] = useState<MediaAsset[]>([]);
  const [selectedMediaId, setSelectedMediaId] = useState<string | null>(null);
  const [transcribeTargetId, setTranscribeTargetId] = useState<string | null>(null);
  const [smartClipsTargetId, setSmartClipsTargetId] = useState<string | null>(null);
  const [motionSeed, setMotionSeed] = useState<MotionSeed | null>(null);
  const [searchQuery, setSearchQuery] = useState("");
  const [filterKind, setFilterKind] = useState<string | null>(null);

  const [jobs, setJobs] = useState<JobRecord[]>([]);
  const [modules, setModules] = useState<ModuleInfo[]>([]);
  const [actions, setActions] = useState<ActionSpec[]>([]);
  const [activity, setActivity] = useState<ActivityEntry[]>([]);
  const [hardware, setHardware] = useState<HardwareProfile | null>(null);
  const [tools, setTools] = useState<ToolStatus[]>([]);
  const [workspace, setWorkspace] = useState<WorkspaceConfig | null>(null);
  const [toasts, setToasts] = useState<ToastItem[]>([]);
  const [paletteOpen, setPaletteOpen] = useState(false);
  const [shortcutsOpen, setShortcutsOpen] = useState(false);

  const socketRef = useRef<{ sendUiSync: (state: UiState) => void; close: () => void } | null>(null);
  const scrollRef = useRef<HTMLElement>(null);

  const pushToast = useCallback((level: NotifyLevel, title: string, message: string) => {
    const id = `${Date.now()}-${Math.random().toString(36).slice(2, 7)}`;
    setToasts((prev) => [...prev.slice(-4), { id, level, title, message }]);
    window.setTimeout(() => {
      setToasts((prev) => prev.filter((t) => t.id !== id));
    }, 4800);
  }, []);

  const copy = useCallback((text: string, label: string) => void copyToClipboard(text, pushToast, label), [pushToast]);

  useEffect(() => {
    document.documentElement.setAttribute("data-theme", theme);
    storage.set("aicut_theme", theme);
  }, [theme]);

  useEffect(() => {
    storage.set("aicut_sidebar", sidebarCollapsed ? "collapsed" : "expanded");
  }, [sidebarCollapsed]);

  const refreshMedia = useCallback(async () => {
    try {
      const res = await runAction<{ items: MediaAsset[] }>("media.list");
      setMedia(res.data.items ?? []);
    } catch {
      // ignore if offline
    }
  }, []);

  const refreshJobs = useCallback(async () => {
    try {
      const res = await runAction<{ jobs: JobRecord[] }>("jobs.list", { only_active: false });
      setJobs(res.data.jobs ?? []);
    } catch {
      // ignore
    }
  }, []);

  const refreshTools = useCallback(async () => {
    try {
      const res = await runAction<{ hardware: HardwareProfile; tools: ToolStatus[] }>("tools.list");
      setHardware(res.data.hardware);
      setTools(res.data.tools ?? []);
    } catch {
      // ignore
    }
  }, []);

  const refreshModulesAndActions = useCallback(async () => {
    try {
      const [mods, acts, actLog] = await Promise.all([fetchModules(), fetchActions(), fetchActivity()]);
      setModules(mods);
      setActions(acts);
      setActivity(actLog);
    } catch {
      // ignore
    }
  }, []);

  const bootstrapAll = useCallback(async () => {
    try {
      const info = await runAction<{ workspace: WorkspaceConfig; hardware: HardwareProfile }>("system.info");
      setWorkspace(info.data.workspace);
      setHardware(info.data.hardware);
    } catch {
      // ignore
    }
    await Promise.all([refreshMedia(), refreshJobs(), refreshTools(), refreshModulesAndActions()]);
  }, [refreshMedia, refreshJobs, refreshTools, refreshModulesAndActions]);

  const clearFinishedJobs = useCallback(async () => {
    try {
      const res = await runAction<{ removed_count: number }>("jobs.clear_finished");
      pushToast("info", "Fila limpa", `${res.data.removed_count} tarefa(s) finalizada(s) removida(s).`);
      void refreshJobs();
    } catch (e) {
      pushToast("error", "Erro ao limpar fila", (e as Error).message);
    }
  }, [pushToast, refreshJobs]);

  useEffect(() => {
    void bootstrapAll();

    const sock = createEventSocket(
      (msg: unknown) => {
        const frame = msg as Record<string, unknown>;
        if (frame.type === "snapshot") {
          if (Array.isArray(frame.jobs)) setJobs(frame.jobs as JobRecord[]);
          if (Array.isArray(frame.activity)) setActivity(frame.activity as ActivityEntry[]);
          return;
        }
        const ev = frame.event as { type?: string; payload?: unknown } | undefined;
        if (!ev?.type) return;

        switch (ev.type) {
          case "job_created":
          case "job_updated": {
            const job = ev.payload as JobRecord;
            setJobs((prev) => {
              const idx = prev.findIndex((j) => j.id === job.id);
              if (idx >= 0) {
                const copyList = [...prev];
                copyList[idx] = job;
                return copyList;
              }
              return [job, ...prev];
            });
            break;
          }
          case "media_created": {
            const asset = ev.payload as MediaAsset;
            setMedia((prev) => [asset, ...prev.filter((m) => m.id !== asset.id)]);
            break;
          }
          case "media_updated": {
            const asset = ev.payload as MediaAsset;
            setMedia((prev) => prev.map((m) => (m.id === asset.id ? asset : m)));
            break;
          }
          case "media_deleted": {
            const { id } = ev.payload as { id: string };
            setMedia((prev) => prev.filter((m) => m.id !== id));
            setSelectedMediaId((curr) => (curr === id ? null : curr));
            break;
          }
          case "ui_state_changed": {
            const st = ev.payload as UiState;
            if (st.active_route) setRoute(st.active_route);
            setSelectedMediaId(st.selected_media_id ?? null);
            setSearchQuery(st.search_query ?? "");
            setFilterKind(st.active_filter_kind ?? null);
            break;
          }
          case "ui_notification": {
            const n = ev.payload as { level: NotifyLevel; title: string; message: string };
            pushToast(n.level, n.title, n.message);
            break;
          }
          case "tool_status_changed": {
            const st = ev.payload as ToolStatus;
            setTools((prev) => prev.map((t) => (t.id === st.id ? st : t)));
            break;
          }
          case "module_state_changed": {
            void refreshModulesAndActions();
            break;
          }
          case "action_executed": {
            const entry = ev.payload as ActivityEntry;
            setActivity((prev) => [entry, ...prev].slice(0, 100));
            break;
          }
        }
      },
      (isUp) => {
        setConnected(isUp);
        if (isUp) {
          void bootstrapAll();
        }
      },
    );

    socketRef.current = sock;
    return () => sock.close();
  }, [bootstrapAll, pushToast, refreshModulesAndActions]);

  const syncUi = useCallback(
    (next: Partial<UiState>) => {
      socketRef.current?.sendUiSync({
        active_route: next.active_route ?? route,
        selected_media_id: next.selected_media_id !== undefined ? next.selected_media_id : selectedMediaId,
        search_query: next.search_query ?? searchQuery,
        active_filter_kind: next.active_filter_kind !== undefined ? next.active_filter_kind : filterKind,
        player: {
          media_id: selectedMediaId,
          playing: false,
          current_time_sec: 0,
        },
      });
    },
    [route, selectedMediaId, searchQuery, filterKind],
  );

  const navigate = useCallback(
    (nextRoute: RouteId) => {
      setRoute(nextRoute);
      syncUi({ active_route: nextRoute });
    },
    [syncUi],
  );

  const selectMedia = useCallback(
    (id: string | null) => {
      setSelectedMediaId(id);
      syncUi({ selected_media_id: id });
    },
    [syncUi],
  );

  const toggleSidebar = useCallback(() => setSidebarCollapsed((v) => !v), []);
  const openPalette = useCallback(() => setPaletteOpen(true), []);
  const openShortcuts = useCallback(() => setShortcutsOpen(true), []);

  // Every view starts at the top
  useEffect(() => {
    scrollRef.current?.scrollTo({ top: 0 });
  }, [route]);

  // Global keyboard shortcuts
  const navigateRef = useRef(navigate);
  navigateRef.current = navigate;
  useEffect(() => {
    const onKeyDown = (e: KeyboardEvent) => {
      if (e.defaultPrevented) return;
      const mod = e.ctrlKey || e.metaKey;
      const key = e.key.toLowerCase();
      if (mod && !e.shiftKey && key === "k") {
        e.preventDefault();
        setPaletteOpen((v) => !v);
      } else if (mod && e.shiftKey && key === "p") {
        e.preventDefault();
        setPaletteOpen(true);
      } else if (mod && !e.shiftKey && key === "b") {
        e.preventDefault();
        setSidebarCollapsed((v) => !v);
      } else if (e.altKey && !mod && /^Digit[1-9]$/.test(e.code)) {
        const target = routeForDigit(Number(e.code.slice(5)));
        if (target) {
          e.preventDefault();
          setPaletteOpen(false);
          navigateRef.current(target);
        }
      } else if (e.key === "?" && !mod && !e.altKey && !isTypingTarget(e.target)) {
        e.preventDefault();
        setShortcutsOpen(true);
      } else if (e.key === "Escape") {
        setPaletteOpen(false);
      }
    };
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, []);

  const activeJobsCount = jobs.filter((j) => j.status === "queued" || j.status === "running").length;
  const toolsNeedAttention = tools.filter((t) => !t.installed || t.update_available).length;

  const navCounts: Partial<Record<RouteId, NavCount>> = {
    library: { value: media.length, title: `${media.length} item(ns) na Biblioteca` },
    jobs: { value: activeJobsCount, tone: "accent", title: `${activeJobsCount} tarefa(s) em andamento` },
    settings: { value: toolsNeedAttention, tone: "warning", title: "Ferramentas para instalar ou atualizar" },
  };

  const paletteCommands = useMemo<PaletteCommand[]>(
    () => [
      ...ROUTES.map((r) => ({
        id: `go:${r.id}`,
        label: r.label,
        description: r.description,
        icon: r.icon,
        group: "Ir para",
        shortcut: `Alt+${routeDigit(r.id)}`,
        keywords: r.keywords,
        run: () => navigate(r.id),
      })),
      {
        id: "ui:sidebar",
        label: sidebarCollapsed ? "Expandir menu lateral" : "Recolher menu lateral",
        icon: PanelLeft,
        group: "Interface",
        shortcut: "Ctrl+B",
        keywords: "sidebar barra lateral",
        run: toggleSidebar,
      },
      {
        id: "ui:shortcuts",
        label: "Mostrar atalhos de teclado",
        icon: Keyboard,
        group: "Interface",
        shortcut: "?",
        keywords: "teclado ajuda help",
        run: openShortcuts,
      },
      ...THEMES.map((t) => ({
        id: `theme:${t.id}`,
        label: `Esquema de cores: ${t.label}${t.id === theme ? " (atual)" : ""}`,
        description: t.description,
        icon: Palette,
        group: "Interface",
        keywords: "tema cores aparência escuro claro",
        run: () => setTheme(t.id),
      })),
      {
        id: "data:refresh",
        label: "Recarregar dados do core",
        description: "Biblioteca, fila, ferramentas, módulos e atividade",
        icon: RefreshCw,
        group: "Dados",
        keywords: "atualizar sincronizar reload",
        run: () => void bootstrapAll(),
      },
      {
        id: "jobs:clear",
        label: "Limpar tarefas finalizadas",
        icon: Trash2,
        group: "Dados",
        keywords: "fila jobs limpar",
        run: () => void clearFinishedJobs(),
      },
    ],
    [navigate, sidebarCollapsed, toggleSidebar, openShortcuts, theme, bootstrapAll, clearFinishedJobs],
  );

  // Right click on empty areas: app-level menu (editable fields keep the native menu)
  const onBackgroundContextMenu = (e: ReactMouseEvent) => {
    const target = e.target as HTMLElement;
    if (target.closest("input, textarea, select, video, audio, [contenteditable='true']")) return;
    const selection = window.getSelection()?.toString().trim() ?? "";
    ctxMenu.open(e, [
      selection && { label: "Copiar seleção", icon: Copy, shortcut: "Ctrl+C", onSelect: () => copy(selection, "Seleção") },
      selection && { type: "separator" },
      { label: "Paleta de comandos", icon: Command, shortcut: "Ctrl+K", onSelect: openPalette },
      {
        label: sidebarCollapsed ? "Expandir menu lateral" : "Recolher menu lateral",
        icon: PanelLeft,
        shortcut: "Ctrl+B",
        onSelect: toggleSidebar,
      },
      { label: "Atalhos de teclado", icon: Keyboard, shortcut: "?", onSelect: openShortcuts },
      { type: "separator" },
      { type: "label", label: `Esquema de cores · ${themeLabel(theme)}` },
      ...THEMES.map((t) => ({ label: t.label, checked: t.id === theme, onSelect: () => setTheme(t.id) })),
      { type: "separator" },
      { label: "Recarregar dados", icon: RefreshCw, onSelect: () => void bootstrapAll() },
    ]);
  };

  const meta = ROUTE_MAP[route];

  return (
    <div
      className="app-shell"
      data-sidebar={sidebarCollapsed ? "collapsed" : "expanded"}
      onContextMenu={onBackgroundContextMenu}
    >
      <Topbar
        route={route}
        sidebarCollapsed={sidebarCollapsed}
        sidebarLocked={sidebarLocked}
        theme={theme}
        onToggleSidebar={toggleSidebar}
        onOpenPalette={openPalette}
        onOpenShortcuts={openShortcuts}
        onThemeChange={setTheme}
      />

      <Sidebar
        route={route}
        collapsed={sidebarCollapsed || sidebarLocked}
        counts={navCounts}
        onNavigate={navigate}
        menuFor={(id) =>
          id === "jobs"
            ? [{ label: "Limpar finalizadas", icon: Trash2, onSelect: () => void clearFinishedJobs() }]
            : id === "library"
              ? [{ label: "Recarregar Biblioteca", icon: RefreshCw, onSelect: () => void refreshMedia() }]
              : id === "settings"
                ? [{ label: "Recarregar ferramentas", icon: RefreshCw, onSelect: () => void refreshTools() }]
                : []
        }
      />

      <main className="page-scroll" ref={scrollRef} aria-label={meta.title}>
        <div className="page">
          {route === "ingest" && (
            <IngestView
              tools={tools}
              jobs={jobs}
              onNavigate={navigate}
              onNotify={pushToast}
              onMediaChanged={refreshMedia}
            />
          )}

          {route === "library" && (
            <LibraryView
              media={media}
              selectedMediaId={selectedMediaId}
              searchQuery={searchQuery}
              filterKind={filterKind}
              onSelectMedia={selectMedia}
              onSearchChange={(q) => {
                setSearchQuery(q);
                syncUi({ search_query: q });
              }}
              onFilterKindChange={(k) => {
                setFilterKind(k);
                syncUi({ active_filter_kind: k });
              }}
              onRefresh={refreshMedia}
              onNavigate={navigate}
              onTranscribeMedia={(mediaId) => {
                setTranscribeTargetId(mediaId);
                navigate("transcribe");
              }}
              onSmartClipMedia={(mediaId) => {
                setSmartClipsTargetId(mediaId);
                navigate("smart-clips");
              }}
              onNotify={pushToast}
            />
          )}

          {route === "transcribe" && (
            <TranscribeView
              media={media}
              tools={tools}
              jobs={jobs}
              hardware={hardware}
              preselectedMediaId={transcribeTargetId}
              onNavigate={navigate}
              onNotify={pushToast}
              onRefreshTools={refreshTools}
              onRefreshMedia={refreshMedia}
            />
          )}

          {route === "smart-clips" && (
            <SmartClipsView
              media={media}
              tools={tools}
              jobs={jobs}
              hardware={hardware}
              preselectedMediaId={smartClipsTargetId}
              onNavigate={navigate}
              onSelectMedia={(mediaId) => {
                setSelectedMediaId(mediaId);
                navigate("library");
              }}
              onOpenInMotionStudio={(seed) => {
                setMotionSeed(seed);
                navigate("motion-studio");
              }}
              onNotify={pushToast}
              onRefreshTools={refreshTools}
              onRefreshMedia={refreshMedia}
            />
          )}

          {route === "framing" && (
            <FramingStudioView
              media={media}
              tools={tools}
              preselectedMediaId={smartClipsTargetId ?? selectedMediaId}
              onNavigate={navigate}
              onNotify={pushToast}
            />
          )}

          {route === "motion-studio" && (
            <MotionStudioView
              media={media}
              tools={tools}
              jobs={jobs}
              seed={motionSeed}
              onNavigate={navigate}
              onSelectMedia={(mediaId) => {
                setSelectedMediaId(mediaId);
                navigate("library");
              }}
              onNotify={pushToast}
              onRefreshMedia={refreshMedia}
            />
          )}

          {route === "jobs" && (
            <JobsView jobs={jobs} onRefresh={refreshJobs} onClearFinished={clearFinishedJobs} onNotify={pushToast} />
          )}

          {route === "modules" && (
            <ModulesView
              modules={modules}
              actions={actions}
              onRefresh={refreshModulesAndActions}
              onNotify={pushToast}
            />
          )}

          {route === "ai-connect" && (
            <AiConnectView
              activity={activity}
              actionCount={actions.length}
              connected={connected}
              onNavigate={navigate}
              onNotify={pushToast}
            />
          )}

          {route === "settings" && (
            <SettingsView
              hardware={hardware}
              tools={tools}
              workspace={workspace}
              theme={theme}
              onThemeChange={setTheme}
              onRefreshTools={refreshTools}
              onNotify={pushToast}
            />
          )}
        </div>
      </main>

      <StatusBar
        connected={connected}
        serverUrl={getPublicServerUrl()}
        activeJobs={activeJobsCount}
        mediaCount={media.length}
        toolsNeedAttention={toolsNeedAttention}
        hardware={hardware}
        onNavigate={navigate}
        onOpenShortcuts={openShortcuts}
        onOpenPalette={openPalette}
        onClearFinished={() => void clearFinishedJobs()}
        onRefreshTools={() => void refreshTools()}
        onCopy={copy}
      />

      <CommandPalette open={paletteOpen} commands={paletteCommands} onClose={() => setPaletteOpen(false)} />
      <ShortcutsDialog open={shortcutsOpen} onClose={() => setShortcutsOpen(false)} />
      <Toaster toasts={toasts} onDismiss={(id) => setToasts((prev) => prev.filter((t) => t.id !== id))} />
    </div>
  );
}

export default App;
