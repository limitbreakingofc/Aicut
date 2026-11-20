import { Download, Film, Layers, Mic, Scissors, Settings, type LucideIcon } from "lucide-react";
import type { RouteId } from "../lib/types";

/** Single source of truth for app navigation: 5 studio workspaces + Settings hub pinned at bottom. */

export const APP_VERSION = "0.10.0";

export type PrimaryRouteId = "ingest" | "library" | "transcribe" | "smart-clips" | "motion-studio" | "settings";
export type SmartClipsTabId = "generate" | "results" | "framing";
export type SettingsTabId = "ai" | "tools" | "system" | "integrations" | "modules" | "jobs";

export interface RouteMeta {
  id: PrimaryRouteId;
  /** Short label for sidebar, topbar and command palette. */
  label: string;
  /** Page H1 / accessibility label. */
  title: string;
  /** Short subtitle and palette description. */
  description: string;
  icon: LucideIcon;
  /** Rendered alone at the bottom of the sidebar. */
  footer?: boolean;
  /** Extra search terms for the command palette. */
  keywords?: string;
}

export const ROUTES: RouteMeta[] = [
  {
    id: "ingest",
    label: "Importar",
    title: "Importar & Baixar",
    description: "Arraste arquivos do PC ou cole um link para baixar direto na Biblioteca.",
    icon: Download,
    keywords: "yt-dlp youtube twitch tiktok download url importar pc arquivos pasta",
  },
  {
    id: "library",
    label: "Biblioteca",
    title: "Biblioteca",
    description: "Vídeos, áudios, legendas, Shorts gerados e Lixeira.",
    icon: Film,
    keywords: "mídia vídeos áudios legendas timelines transcrições player lixeira",
  },
  {
    id: "transcribe",
    label: "Transcrever",
    title: "Transcrever IA",
    description: "Whisper local em abas: transcrever, resultados e modelos.",
    icon: Mic,
    keywords: "whisper legenda srt vtt transcrição modelo gpu fcpxml",
  },
  {
    id: "smart-clips",
    label: "Cortes Shorts",
    title: "Cortes Shorts IA",
    description: "Gerar cortes verticais 9:16, revisar resultados e calibrar enquadramento da webcam.",
    icon: Scissors,
    keywords: "shorts reels tiktok cortes clips enquadramento webcam bordas moldura 9:16",
  },
  {
    id: "motion-studio",
    label: "Motion Studio",
    title: "Motion Studio",
    description: "Editora IA conversacional + preview 9:16 ao vivo com animações HyperFrames.",
    icon: Layers,
    keywords: "motion studio remotion hyperframes layout dinamico chat ia preview",
  },
  {
    id: "settings",
    label: "Configurações",
    title: "Configurações",
    description: "IA & chaves, ferramentas, sistema, integrações MCP, módulos e fila de tarefas.",
    icon: Settings,
    footer: true,
    keywords: "tema escuro claro hardware gpu ferramentas ffmpeg yt-dlp gemini api key mcp modulos tarefas",
  },
];

export const ROUTE_MAP = Object.fromEntries(ROUTES.map((r) => [r.id, r])) as Record<PrimaryRouteId, RouteMeta>;

/** Maps any RouteId (including legacy MCP/deep-link routes) into its primary workspace + tab. */
export function resolveRouteTarget(id: RouteId): {
  route: PrimaryRouteId;
  clipsTab?: SmartClipsTabId;
  settingsTab?: SettingsTabId;
} {
  switch (id) {
    case "framing":
      return { route: "smart-clips", clipsTab: "framing" };
    case "jobs":
      return { route: "settings", settingsTab: "jobs" };
    case "modules":
      return { route: "settings", settingsTab: "modules" };
    case "ai-connect":
      return { route: "settings", settingsTab: "integrations" };
    case "ingest":
    case "library":
    case "transcribe":
    case "smart-clips":
    case "motion-studio":
    case "settings":
      return { route: id };
    default:
      return { route: "ingest" };
  }
}

/** Alt+1 … Alt+6 jump straight to each workspace (registry order). */
export function routeDigit(id: PrimaryRouteId): number {
  return ROUTES.findIndex((r) => r.id === id) + 1;
}

export function routeForDigit(digit: number): PrimaryRouteId | null {
  return ROUTES[digit - 1]?.id ?? null;
}

/** Standard PageHeader props for primary routes. */
export function pageMeta(id: PrimaryRouteId) {
  const meta = ROUTE_MAP[id];
  return { icon: meta.icon, title: meta.title, description: meta.description };
}
