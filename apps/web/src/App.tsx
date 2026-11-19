import React, { useCallback, useEffect, useRef, useState } from "react";
import {
  Activity,
  Blocks,
  Bot,
  Command,
  Download,
  Film,
  Mic,
  Settings,
  X,
} from "lucide-react";
import {
  createEventSocket,
  fetchActions,
  fetchActivity,
  fetchModules,
  runAction,
} from "./lib/client";
import type {
  ActionSpec,
  ActivityEntry,
  HardwareProfile,
  JobRecord,
  MediaAsset,
  ModuleInfo,
  RouteId,
  ToolStatus,
  UiState,
  WorkspaceConfig,
} from "./lib/types";
import { IngestView } from "./modules/media-ingest/IngestView";
import { LibraryView } from "./modules/media-ingest/LibraryView";
import { TranscribeView } from "./modules/transcribe/TranscribeView";
import { JobsView } from "./modules/core/JobsView";
import { ModulesView } from "./modules/core/ModulesView";
import { AiConnectView } from "./modules/core/AiConnectView";
import { SettingsView } from "./modules/core/SettingsView";

interface ToastItem {
  id: string;
  level: "info" | "success" | "warning" | "error";
  title: string;
  message: string;
}

export function App() {
  const [route, setRoute] = useState<RouteId>("ingest");
  const [connected, setConnected] = useState(false);
  const [theme, setTheme] = useState<"dark" | "light">(() => {
    const saved = localStorage.getItem("aicut_theme");
    return saved === "light" ? "light" : "dark";
  });

  const [media, setMedia] = useState<MediaAsset[]>([]);
  const [selectedMediaId, setSelectedMediaId] = useState<string | null>(null);
  const [transcribeTargetId, setTranscribeTargetId] = useState<string | null>(null);
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
  const [paletteQuery, setPaletteQuery] = useState("");

  const socketRef = useRef<{ sendUiSync: (state: UiState) => void; close: () => void } | null>(null);

  const pushToast = useCallback(
    (level: "info" | "success" | "warning" | "error", title: string, message: string) => {
      const id = `${Date.now()}-${Math.random().toString(36).slice(2, 7)}`;
      setToasts((prev) => [...prev.slice(-4), { id, level, title, message }]);
      window.setTimeout(() => {
        setToasts((prev) => prev.filter((t) => t.id !== id));
      }, 4800);
    },
    [],
  );

  useEffect(() => {
    document.documentElement.setAttribute("data-theme", theme);
    localStorage.setItem("aicut_theme", theme);
  }, [theme]);

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
      const [mods, acts, actLog] = await Promise.all([
        fetchModules(),
        fetchActions(),
        fetchActivity(),
      ]);
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
                const copy = [...prev];
                copy[idx] = job;
                return copy;
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
            const n = ev.payload as {
              level: "info" | "success" | "warning" | "error";
              title: string;
              message: string;
            };
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
        selected_media_id:
          next.selected_media_id !== undefined ? next.selected_media_id : selectedMediaId,
        search_query: next.search_query ?? searchQuery,
        active_filter_kind:
          next.active_filter_kind !== undefined ? next.active_filter_kind : filterKind,
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

  useEffect(() => {
    const onKeyDown = (e: KeyboardEvent) => {
      if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === "k") {
        e.preventDefault();
        setPaletteOpen((v) => !v);
        setPaletteQuery("");
      } else if (e.key === "Escape") {
        setPaletteOpen(false);
      }
    };
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, []);

  const activeJobsCount = jobs.filter((j) => j.status === "queued" || j.status === "running").length;
  const toolsNeedAttention = tools.filter((t) => !t.installed || t.update_available).length;

  const navItems: Array<{
    id: RouteId;
    label: string;
    icon: React.ReactNode;
    badge?: number;
    badgeAccent?: boolean;
  }> = [
    { id: "ingest", label: "Baixar & Importar", icon: <Download size={16} /> },
    { id: "library", label: "Biblioteca", icon: <Film size={16} />, badge: media.length },
    { id: "transcribe", label: "Transcrever IA", icon: <Mic size={16} /> },
    {
      id: "jobs",
      label: "Fila de Tarefas",
      icon: <Activity size={16} />,
      badge: activeJobsCount > 0 ? activeJobsCount : undefined,
      badgeAccent: true,
    },
    { id: "modules", label: "Módulos", icon: <Blocks size={16} /> },
    { id: "ai-connect", label: "Conectar IA", icon: <Bot size={16} /> },
  ];

  const paletteCommands: Array<{ label: string; hint: string; run: () => void }> = [
    { label: "Ir para Baixar & Importar (yt-dlp / PC)", hint: "Aba 1", run: () => navigate("ingest") },
    { label: "Ir para Biblioteca de Mídia", hint: "Aba 2", run: () => navigate("library") },
    { label: "Ir para Transcrever com IA Local (Whisper)", hint: "Aba 3", run: () => navigate("transcribe") },
    { label: "Ir para Fila de Tarefas", hint: "Aba 4", run: () => navigate("jobs") },
    { label: "Ir para Módulos & Action Playground", hint: "Aba 5", run: () => navigate("modules") },
    { label: "Ir para Conectar IA (MCP / OpenAPI / CLI)", hint: "Aba 6", run: () => navigate("ai-connect") },
    { label: "Ir para Configurações (Hardware & Ferramentas)", hint: "Menu", run: () => navigate("settings") },
    {
      label: "Alternar Tema Escuro / Claro",
      hint: "Aparência",
      run: () => setTheme((t) => (t === "dark" ? "light" : "dark")),
    },
  ].filter(
    (c) =>
      !paletteQuery.trim() ||
      c.label.toLowerCase().includes(paletteQuery.toLowerCase()) ||
      c.hint.toLowerCase().includes(paletteQuery.toLowerCase()),
  );

  return (
    <div className="app-shell">
      <header className="titlebar" data-tauri-drag-region>
        <div className="titlebar-brand">
          <span className="brand-mark">A</span>
          <span>Aicut</span>
          <span style={{ color: "var(--text-muted)", fontWeight: 400, fontSize: 12 }}>
            v0.2.0 · Modular AI Studio
          </span>
        </div>

        <div className="titlebar-meta">
          <button
            type="button"
            className="status-chip"
            style={{ cursor: "pointer" }}
            onClick={() => setPaletteOpen(true)}
            title="Abrir Paleta de Comandos (Ctrl+K)"
          >
            <Command size={12} /> Comandos <kbd style={{ opacity: 0.7 }}>Ctrl+K</kbd>
          </button>
          <button
            type="button"
            className="status-chip"
            style={{ cursor: "pointer" }}
            onClick={() => navigate("ai-connect")}
          >
            <span className={`status-dot ${connected ? "" : "offline"}`} />
            {connected ? "IA Pronta · MCP :43210" : "Reconectando Core..."}
          </button>
        </div>
      </header>

      <div className="app-body">
        <aside className="sidebar">
          <nav className="nav-group" aria-label="Navegação Principal">
            <div className="nav-label">Estúdio</div>
            {navItems.map((item) => (
              <button
                key={item.id}
                type="button"
                className={`nav-item ${route === item.id ? "active" : ""}`}
                onClick={() => navigate(item.id)}
              >
                {item.icon}
                <span className="nav-text">{item.label}</span>
                {item.badge !== undefined && (
                  <span className={`nav-badge ${item.badgeAccent ? "accent" : ""}`}>
                    {item.badge}
                  </span>
                )}
              </button>
            ))}
          </nav>

          <div className="sidebar-footer">
            <button
              type="button"
              className={`nav-item ${route === "settings" ? "active" : ""}`}
              onClick={() => navigate("settings")}
            >
              <Settings size={16} />
              <span className="nav-text">Configurações</span>
              {toolsNeedAttention > 0 && (
                <span className="nav-badge accent" title="Ferramentas para instalar ou atualizar">
                  {toolsNeedAttention}
                </span>
              )}
            </button>
          </div>
        </aside>

        <main className="main-viewport">
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
              onSelectMedia={(id) => {
                setSelectedMediaId(id);
                syncUi({ selected_media_id: id });
              }}
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
              onNotify={pushToast}
            />
          )}

          {route === "transcribe" && (
            <TranscribeView
              media={media}
              tools={tools}
              jobs={jobs}
              preselectedMediaId={transcribeTargetId}
              onNavigate={navigate}
              onNotify={pushToast}
              onRefreshTools={refreshTools}
              onRefreshMedia={refreshMedia}
            />
          )}

          {route === "jobs" && (
            <JobsView jobs={jobs} onRefresh={refreshJobs} onNotify={pushToast} />
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
              onNotify={pushToast}
            />
          )}

          {route === "settings" && (
            <SettingsView
              hardware={hardware}
              tools={tools}
              workspace={workspace}
              theme={theme}
              onToggleTheme={() => setTheme((t) => (t === "dark" ? "light" : "dark"))}
              onRefreshTools={refreshTools}
              onNotify={pushToast}
            />
          )}
        </main>
      </div>

      {paletteOpen && (
        <div className="modal-backdrop" onClick={() => setPaletteOpen(false)}>
          <div
            className="modal-card"
            style={{ width: "min(520px, 92vw)", padding: 14, gap: 10 }}
            onClick={(e) => e.stopPropagation()}
          >
            <input
              autoFocus
              className="input"
              placeholder="Ir para aba ou comando..."
              value={paletteQuery}
              onChange={(e) => setPaletteQuery(e.target.value)}
            />
            <div style={{ display: "flex", flexDirection: "column", gap: 4 }}>
              {paletteCommands.map((cmd) => (
                <button
                  key={cmd.label}
                  type="button"
                  className="nav-item"
                  onClick={() => {
                    cmd.run();
                    setPaletteOpen(false);
                  }}
                >
                  <span>{cmd.label}</span>
                  <span className="nav-badge">{cmd.hint}</span>
                </button>
              ))}
            </div>
          </div>
        </div>
      )}

      {toasts.length > 0 && (
        <div className="toast-stack" aria-live="polite">
          {toasts.map((t) => (
            <div key={t.id} className="toast">
              <div style={{ flex: 1, minWidth: 0 }}>
                <div style={{ fontWeight: 600, fontSize: 13 }}>{t.title}</div>
                <div style={{ fontSize: 12.5, color: "var(--text-secondary)", marginTop: 2 }}>
                  {t.message}
                </div>
              </div>
              <button
                type="button"
                className="btn btn-ghost btn-sm"
                style={{ padding: 4 }}
                onClick={() => setToasts((prev) => prev.filter((x) => x.id !== t.id))}
              >
                <X size={13} />
              </button>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

export default App;
