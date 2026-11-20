import { Activity, Blocks, Bot, Download, Film, Mic, Scissors, Settings, type LucideIcon } from "lucide-react";
import type { RouteId } from "../lib/types";

/** Single source of truth for app navigation: sidebar, breadcrumb, page headers and command palette. */

export const APP_VERSION = "0.5.0";

export type NavGroupId = "studio" | "system";

export interface RouteMeta {
  id: RouteId;
  /** Short label for sidebar, breadcrumb and palette. */
  label: string;
  /** Page H1. */
  title: string;
  /** Page subtitle and palette description. */
  description: string;
  icon: LucideIcon;
  group: NavGroupId;
  /** Position in the creation workflow (Estúdio pages only). */
  step?: number;
  /** Rendered in the sidebar footer instead of its group. */
  footer?: boolean;
  /** Extra search terms for the command palette. */
  keywords?: string;
}

export const NAV_GROUPS: Array<{ id: NavGroupId; label: string }> = [
  { id: "studio", label: "Estúdio" },
  { id: "system", label: "Sistema" },
];

export const ROUTES: RouteMeta[] = [
  {
    id: "ingest",
    label: "Baixar & Importar",
    title: "Baixar & Importar",
    description:
      "Baixe vídeos e áudios com todas as opções do yt-dlp ou importe arquivos do seu PC direto para a Biblioteca.",
    icon: Download,
    group: "studio",
    step: 1,
    keywords: "yt-dlp youtube twitch tiktok download url importar pc arquivos pasta",
  },
  {
    id: "library",
    label: "Biblioteca",
    title: "Biblioteca",
    description:
      "Tudo o que você baixou ou importou, mais legendas, timelines e transcrições geradas pela IA local.",
    icon: Film,
    group: "studio",
    step: 2,
    keywords: "mídia vídeos áudios legendas timelines transcrições player tags notas",
  },
  {
    id: "transcribe",
    label: "Transcrever IA",
    title: "Transcrever com IA local",
    description:
      "Whisper 100% offline com modelos escolhidos para a sua GPU — legendas, timelines e textos salvos direto na Biblioteca.",
    icon: Mic,
    group: "studio",
    step: 3,
    keywords: "whisper legenda srt vtt transcrição modelo gpu fcpxml",
  },
  {
    id: "smart-clips",
    label: "Cortes Shorts IA",
    title: "Cortes Shorts com IA",
    description:
      "A IA assiste ao vídeo, segue o seu briefing e cria cortes de até 3 min com início, meio e fim — via Gemini API ou 100% local e grátis (Qwen2.5-VL) na sua GPU.",
    icon: Scissors,
    group: "studio",
    step: 4,
    keywords: "shorts reels tiktok cortes clips gemini qwen 9:16 vertical renderizar",
  },
  {
    id: "jobs",
    label: "Fila de Tarefas",
    title: "Fila de tarefas",
    description:
      "Acompanhe em tempo real downloads, importações, transcrições e renderizações disparados por você ou por qualquer IA.",
    icon: Activity,
    group: "system",
    keywords: "jobs progresso fila tarefas logs cancelar",
  },
  {
    id: "modules",
    label: "Módulos",
    title: "Módulos & Actions",
    description:
      "Arquitetura plugável: cada módulo registra suas Actions, ferramentas e JSON Schemas compartilhados entre você e as IAs.",
    icon: Blocks,
    group: "system",
    keywords: "módulos plugins actions playground schema json",
  },
  {
    id: "ai-connect",
    label: "Conectar IA",
    title: "Conectar qualquer IA",
    description:
      "Paridade total entre humanos e IAs via MCP (stdio + HTTP/SSE), OpenAPI 3.1 e CLI com entrada e saída JSON.",
    icon: Bot,
    group: "system",
    keywords: "mcp openapi claude cursor chatgpt cli agentes atividade",
  },
  {
    id: "settings",
    label: "Configurações",
    title: "Configurações",
    description:
      "Aparência, hardware detectado, ferramentas instaladas na melhor build para o seu PC e pastas do workspace.",
    icon: Settings,
    group: "system",
    footer: true,
    keywords: "tema escuro claro hardware gpu ferramentas ffmpeg yt-dlp whisper workspace pastas",
  },
];

export const ROUTE_MAP = Object.fromEntries(ROUTES.map((r) => [r.id, r])) as Record<RouteId, RouteMeta>;

/** Alt+1 … Alt+8 jump straight to each view (registry order). */
export function routeDigit(id: RouteId): number {
  return ROUTES.findIndex((r) => r.id === id) + 1;
}

export function routeForDigit(digit: number): RouteId | null {
  return ROUTES[digit - 1]?.id ?? null;
}

export const WORKFLOW_STEPS = ROUTES.filter((r) => r.step !== undefined).length;

export function groupLabel(group: NavGroupId): string {
  return NAV_GROUPS.find((g) => g.id === group)?.label ?? "";
}

/** Path-like slug used in the top bar and page eyebrow (e.g. "estudio/biblioteca"). */
export function routePath(id: RouteId): string {
  const meta = ROUTE_MAP[id];
  const slug = (value: string) =>
    value
      .normalize("NFD")
      .replace(/[\u0300-\u036f]/g, "")
      .toLowerCase()
      .replace(/&/g, "e")
      .replace(/[^a-z0-9]+/g, "-")
      .replace(/^-|-$/g, "");
  return `${slug(groupLabel(meta.group))}/${slug(meta.label)}`;
}

/** Small caption above page titles: "etapa 2/4 · estúdio" or the group name. */
export function eyebrowFor(id: RouteId): string {
  const meta = ROUTE_MAP[id];
  const group = groupLabel(meta.group).toLowerCase();
  return meta.step ? `etapa ${meta.step}/${WORKFLOW_STEPS} · ${group}` : group;
}

/** Standard PageHeader props for a route (title, icon, eyebrow, description). */
export function pageMeta(id: RouteId) {
  const meta = ROUTE_MAP[id];
  return { icon: meta.icon, title: meta.title, description: meta.description, eyebrow: eyebrowFor(id) };
}
