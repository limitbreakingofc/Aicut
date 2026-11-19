import type { ActionOutput, ActionSpec, ActivityEntry, AicutError, ModuleInfo, UiState } from "./types";

const DEFAULT_API_BASE = "http://127.0.0.1:43210";

declare global {
  interface Window {
    __TAURI_INTERNALS__?: {
      invoke?: <T = unknown>(cmd: string, args?: Record<string, unknown>) => Promise<T>;
    };
  }
}

export function isTauriRuntime(): boolean {
  return typeof window !== "undefined" && Boolean(window.__TAURI_INTERNALS__);
}

export function getApiBase(): string {
  if (typeof window !== "undefined") {
    const stored = localStorage.getItem("aicut_api_base");
    if (stored) return stored.replace(/\/$/, "");
    // In Tauri desktop, embedded server runs on 127.0.0.1:43210; in Vite dev proxy, relative "" works too
    if (isTauriRuntime()) return DEFAULT_API_BASE;
    if (window.location.port === "5173") return "";
  }
  return DEFAULT_API_BASE;
}

export function getPublicServerUrl(): string {
  const base = getApiBase();
  return base || DEFAULT_API_BASE;
}

export function getWsUrl(): string {
  const base = getPublicServerUrl();
  return base.replace(/^http/, "ws") + "/api/v1/events";
}

export function mediaStreamUrl(id: string): string {
  return `${getPublicServerUrl()}/api/v1/media/${encodeURIComponent(id)}/stream`;
}

export function mediaThumbnailUrl(id: string): string {
  return `${getPublicServerUrl()}/api/v1/media/${encodeURIComponent(id)}/thumbnail`;
}

export class AicutActionError extends Error {
  public readonly detail: AicutError;
  constructor(detail: AicutError) {
    super(detail.message);
    this.name = "AicutActionError";
    this.detail = detail;
  }
}

export async function runAction<T = unknown>(
  actionId: string,
  input: Record<string, unknown> = {},
): Promise<ActionOutput<T>> {
  const res = await fetch(`${getApiBase()}/api/v1/actions/${encodeURIComponent(actionId)}`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      "X-Aicut-Actor": "human",
    },
    body: JSON.stringify(input),
  });

  const json = await res.json().catch(() => null);
  if (!res.ok || !json?.ok) {
    const err: AicutError = json?.error ?? {
      code: `HTTP_${res.status}`,
      message: `Falha ao executar ${actionId} (HTTP ${res.status})`,
      recoverable: false,
    };
    throw new AicutActionError(err);
  }
  return json.output as ActionOutput<T>;
}

export async function fetchModules(): Promise<ModuleInfo[]> {
  const res = await fetch(`${getApiBase()}/api/v1/modules`);
  const json = await res.json();
  return (json.modules ?? []) as ModuleInfo[];
}

export async function fetchActions(): Promise<ActionSpec[]> {
  const res = await fetch(`${getApiBase()}/api/v1/actions`);
  const json = await res.json();
  return (json.actions ?? []) as ActionSpec[];
}

export async function fetchActivity(): Promise<ActivityEntry[]> {
  const res = await fetch(`${getApiBase()}/api/v1/activity`);
  const json = await res.json();
  return (json.entries ?? []) as ActivityEntry[];
}

export async function pickNativeFiles(directory: boolean): Promise<string[] | null> {
  if (!isTauriRuntime() || !window.__TAURI_INTERNALS__?.invoke) {
    return null;
  }
  try {
    const selected = await window.__TAURI_INTERNALS__.invoke<string[] | null>("pick_media_paths", {
      directory,
    });
    return selected ?? [];
  } catch {
    return null;
  }
}

export function createEventSocket(
  onMessage: (payload: unknown) => void,
  onStatusChange: (connected: boolean) => void,
): { sendUiSync: (state: UiState) => void; close: () => void } {
  let ws: WebSocket | null = null;
  let closed = false;
  let retryTimer: number | null = null;

  const connect = () => {
    if (closed) return;
    try {
      ws = new WebSocket(getWsUrl());
      ws.onopen = () => onStatusChange(true);
      ws.onclose = () => {
        onStatusChange(false);
        if (!closed) {
          retryTimer = window.setTimeout(connect, 2000);
        }
      };
      ws.onerror = () => {
        onStatusChange(false);
      };
      ws.onmessage = (ev) => {
        try {
          onMessage(JSON.parse(ev.data));
        } catch {
          // ignore malformed frames
        }
      };
    } catch {
      onStatusChange(false);
      if (!closed) {
        retryTimer = window.setTimeout(connect, 2500);
      }
    }
  };

  connect();

  return {
    sendUiSync: (state: UiState) => {
      if (ws && ws.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify({ type: "ui_sync", state }));
      }
    },
    close: () => {
      closed = true;
      if (retryTimer !== null) window.clearTimeout(retryTimer);
      ws?.close();
    },
  };
}
