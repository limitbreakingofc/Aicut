import React, { useState } from "react";
import { Bot, CircleCheck, CircleX, Copy, Cpu, Globe, Radio, Terminal, User } from "lucide-react";
import { getPublicServerUrl } from "../../lib/client";
import { formatClock, formatRelativeDate } from "../../lib/format";
import type { ActivityEntry, NotifyFn, RouteId } from "../../lib/types";
import { copyToClipboard } from "../../lib/ui";
import { Badge, CodeBlock, EmptyState, Panel, SegmentedControl, useContextMenu, type MenuItems } from "../../components";
import "./core.css";

interface AiConnectViewProps {
  activity: ActivityEntry[];
  actionCount: number;
  connected: boolean;
  onNavigate: (route: RouteId) => void;
  onNotify: NotifyFn;
}

type ActorFilter = "all" | "ai" | "human";

export const AiConnectView: React.FC<AiConnectViewProps> = ({ activity, actionCount, connected, onNavigate, onNotify }) => {
  const menu = useContextMenu();
  const baseUrl = getPublicServerUrl();
  const [actorFilter, setActorFilter] = useState<ActorFilter>("all");

  const claudeConfig = JSON.stringify(
    {
      mcpServers: {
        aicut: {
          command: "aicut",
          args: ["mcp"],
        },
      },
    },
    null,
    2,
  );

  const cursorHttpConfig = JSON.stringify(
    {
      mcpServers: {
        aicut: {
          url: `${baseUrl}/mcp`,
        },
      },
    },
    null,
    2,
  );

  const openApiInfo = `OpenAPI 3.1:     ${baseUrl}/openapi.json\nOpenAI Tools:    ${baseUrl}/api/v1/tools/openai\nAnthropic Tools: ${baseUrl}/api/v1/tools/anthropic`;

  const cliExample = `aicut actions --schema\naicut run media.probe_url --json '{"url":"https://youtu.be/..."}'\naicut run media.download --json '{"url":"https://youtu.be/...","mode":"video_audio","quality":"1080p","container":"mp4","wait":true}'`;

  const copy = (label: string, text: string) => void copyToClipboard(text, onNotify, label);

  const integrations = [
    {
      title: "Claude Desktop & clientes MCP (stdio)",
      icon: Bot,
      description: "Conecta automaticamente à janela aberta do Aicut (você vê a IA operando ao vivo!) ou roda em segundo plano se o app estiver fechado.",
      code: claudeConfig,
      label: "Configuração Claude Desktop",
    },
    {
      title: "Cursor, VS Code, Windsurf & Antigravity (MCP HTTP)",
      icon: Cpu,
      description: `Streamable HTTP em ${baseUrl}/mcp e SSE legado em ${baseUrl}/mcp/sse.`,
      code: cursorHttpConfig,
      label: "Configuração MCP HTTP",
    },
    {
      title: "ChatGPT, Open WebUI, n8n & LangChain (OpenAPI 3.1)",
      icon: Globe,
      description: "Importe o schema OpenAPI dinâmico ou exporte ferramentas prontas para os SDKs da OpenAI/Anthropic.",
      code: openApiInfo,
      label: "URL OpenAPI 3.1",
      copyText: `${baseUrl}/openapi.json`,
    },
    {
      title: "Agentes de terminal (Claude Code, Codex, Gemini CLI)",
      icon: Terminal,
      description: "Toda funcionalidade aceita entrada e saída JSON estruturada via binário aicut.",
      code: cliExample,
      label: "Comandos CLI",
    },
  ];

  const visible = activity.filter((e) => actorFilter === "all" || (actorFilter === "ai" ? e.actor === "ai" : e.actor !== "ai"));

  const entryMenu = (entry: ActivityEntry): MenuItems => [
    { type: "label", label: `${entry.action_id} · ${formatRelativeDate(entry.timestamp)}` },
    { label: "Copiar action id", icon: Copy, onSelect: () => copy("Action id", entry.action_id) },
    { label: "Copiar resumo", icon: Copy, onSelect: () => copy("Resumo", entry.summary) },
    {
      label: "Copiar linha do log",
      icon: Copy,
      onSelect: () =>
        copy(
          "Linha do log",
          `${entry.timestamp} ${entry.actor} ${entry.action_id} ${entry.ok ? "ok" : "erro"} ${entry.duration_ms}ms ${entry.summary}`,
        ),
    },
    { type: "separator" },
    { label: "Abrir Módulos & Actions", icon: Terminal, onSelect: () => onNavigate("modules") },
  ];

  return (
    <div className="stack">
      <div className="row row-between wrap gap-xs">
        <span className="secondary text-xs">
          {actionCount} actions expostas via MCP (stdio + HTTP/SSE), OpenAPI 3.1 e CLI JSON.
        </span>
        <Badge tone={connected ? "success" : "danger"} dot>
          {connected ? `servidor local ativo · ${baseUrl}` : "core offline — reconectando"}
        </Badge>
      </div>

      <div className="grid-2 integrations">
        {integrations.map((it) => (
          <Panel
            key={it.title}
            title={it.title}
            icon={it.icon}
            description={it.description}
            onContextMenu={(e) =>
              menu.open(e, [
                { label: `Copiar: ${it.label}`, icon: Copy, onSelect: () => copy(it.label, it.copyText ?? it.code) },
                it.copyText && { label: "Copiar bloco completo", icon: Copy, onSelect: () => copy("Bloco", it.code) },
              ])
            }
          >
            <CodeBlock code={it.code} onCopy={() => copy(it.label, it.copyText ?? it.code)} copyLabel="copiar" />
          </Panel>
        ))}
      </div>

      <Panel
        title={
          <>
            Atividade em tempo real <Badge>{activity.length}</Badge>
          </>
        }
        icon={Radio}
        actions={
          <SegmentedControl
            ariaLabel="Filtrar por ator"
            size="sm"
            value={actorFilter}
            onChange={setActorFilter}
            options={[
              { id: "all", label: "todos" },
              { id: "ai", label: "ia", icon: Bot },
              { id: "human", label: "humano", icon: User },
            ]}
          />
        }
        flush
      >
        {visible.length === 0 ? (
          <EmptyState
            compact
            icon={Radio}
            title="aguardando eventos"
            description="Assim que você ou uma IA executar qualquer ação no Aicut, ela aparece aqui ao vivo."
          />
        ) : (
          <div className="activity-log" role="log" aria-live="polite">
            {visible.map((entry) => (
              <div key={entry.id} className="activity-line" onContextMenu={(e) => menu.open(e, entryMenu(entry))}>
                <span className="term-time" title={formatRelativeDate(entry.timestamp)}>
                  {formatClock(entry.timestamp)}
                </span>
                <span className={entry.actor === "ai" ? "actor text-magenta" : "actor"}>
                  {entry.actor === "ai" ? <Bot size={12} aria-hidden /> : <User size={12} aria-hidden />}
                  {entry.actor === "ai" ? "ia" : entry.actor === "system" ? "sistema" : "humano"}
                </span>
                <span className={entry.ok ? "text-success" : "text-danger"} title={entry.ok ? "ok" : "erro"}>
                  {entry.ok ? <CircleCheck size={13} aria-hidden /> : <CircleX size={13} aria-hidden />}
                </span>
                <code className="activity-action">{entry.action_id}</code>
                <span className="activity-summary" title={entry.summary}>
                  {entry.summary}
                </span>
                <span className="muted tabular activity-ms">{entry.duration_ms}ms</span>
              </div>
            ))}
          </div>
        )}
      </Panel>
    </div>
  );
};
