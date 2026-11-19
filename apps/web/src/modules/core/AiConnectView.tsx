import React from "react";
import { Bot, Check, Copy, Cpu, Globe, Terminal, User } from "lucide-react";
import { getPublicServerUrl } from "../../lib/client";
import { formatRelativeDate } from "../../lib/format";
import type { ActivityEntry } from "../../lib/types";

interface AiConnectViewProps {
  activity: ActivityEntry[];
  actionCount: number;
  onNotify: (level: "info" | "success" | "warning" | "error", title: string, message: string) => void;
}

export const AiConnectView: React.FC<AiConnectViewProps> = ({
  activity,
  actionCount,
  onNotify,
}) => {
  const baseUrl = getPublicServerUrl();

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

  const cliExample = `aicut actions --schema\naicut run media.probe_url --json '{"url":"https://youtu.be/..."}'\naicut run media.download --json '{"url":"https://youtu.be/...","mode":"video_audio","quality":"1080p","container":"mp4","wait":true}'`;

  async function copyText(label: string, text: string) {
    try {
      await navigator.clipboard.writeText(text);
      onNotify("success", "Copiado!", `${label} copiado para a área de transferência.`);
    } catch {
      onNotify("warning", "Não foi possível copiar automaticamente", "Selecione o bloco de código manualmente.");
    }
  }

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 18 }}>
      <div className="page-header">
        <div>
          <h1 className="page-title">Conectar Qualquer Chat de IA</h1>
          <p className="page-subtitle">
            O Aicut expõe {actionCount} Actions com paridade total entre humanos e IAs via MCP (stdio + HTTP/SSE), OpenAPI 3.1 e CLI JSON.
          </p>
        </div>
        <span className="pill success">
          <Check size={12} /> Servidor Local Ativo: {baseUrl}
        </span>
      </div>

      <div className="grid-2">
        {/* Claude Desktop / MCP stdio */}
        <div className="panel" style={{ display: "flex", flexDirection: "column", gap: 10 }}>
          <div className="panel-header" style={{ marginBottom: 0 }}>
            <h2 className="panel-title">
              <Bot size={16} color="var(--accent)" /> Claude Desktop & Clientes MCP (stdio)
            </h2>
            <button
              type="button"
              className="btn btn-sm"
              onClick={() => copyText("Configuração Claude Desktop", claudeConfig)}
            >
              <Copy size={13} /> Copiar
            </button>
          </div>
          <p style={{ margin: 0, fontSize: 12.5, color: "var(--text-secondary)" }}>
            Conecta automaticamente à janela aberta do Aicut (você vê a IA operando ao vivo!) ou roda em segundo plano se o app estiver fechado.
          </p>
          <pre className="code-block">{claudeConfig}</pre>
        </div>

        {/* Cursor / Windsurf / VS Code / Antigravity HTTP MCP */}
        <div className="panel" style={{ display: "flex", flexDirection: "column", gap: 10 }}>
          <div className="panel-header" style={{ marginBottom: 0 }}>
            <h2 className="panel-title">
              <Cpu size={16} color="var(--accent)" /> Cursor, VS Code, Windsurf & Antigravity (MCP HTTP)
            </h2>
            <button
              type="button"
              className="btn btn-sm"
              onClick={() => copyText("Configuração MCP HTTP", cursorHttpConfig)}
            >
              <Copy size={13} /> Copiar
            </button>
          </div>
          <p style={{ margin: 0, fontSize: 12.5, color: "var(--text-secondary)" }}>
            Streamable HTTP em <code>{baseUrl}/mcp</code> e SSE legado em <code>{baseUrl}/mcp/sse</code>.
          </p>
          <pre className="code-block">{cursorHttpConfig}</pre>
        </div>

        {/* OpenAPI / ChatGPT / Open WebUI / n8n */}
        <div className="panel" style={{ display: "flex", flexDirection: "column", gap: 10 }}>
          <div className="panel-header" style={{ marginBottom: 0 }}>
            <h2 className="panel-title">
              <Globe size={16} color="var(--accent)" /> ChatGPT, Open WebUI, n8n & LangChain (OpenAPI 3.1)
            </h2>
            <button
              type="button"
              className="btn btn-sm"
              onClick={() => copyText("URL OpenAPI 3.1", `${baseUrl}/openapi.json`)}
            >
              <Copy size={13} /> Copiar URL
            </button>
          </div>
          <p style={{ margin: 0, fontSize: 12.5, color: "var(--text-secondary)" }}>
            Importe o schema OpenAPI dinâmico ou exporte ferramentas prontas para OpenAI/Anthropic SDKs:
          </p>
          <pre className="code-block">
            {`OpenAPI 3.1:     ${baseUrl}/openapi.json\nOpenAI Tools:    ${baseUrl}/api/v1/tools/openai\nAnthropic Tools: ${baseUrl}/api/v1/tools/anthropic`}
          </pre>
        </div>

        {/* CLI / Terminal Agents */}
        <div className="panel" style={{ display: "flex", flexDirection: "column", gap: 10 }}>
          <div className="panel-header" style={{ marginBottom: 0 }}>
            <h2 className="panel-title">
              <Terminal size={16} color="var(--accent)" /> Agentes de Terminal (Claude Code, Codex, Gemini CLI)
            </h2>
            <button
              type="button"
              className="btn btn-sm"
              onClick={() => copyText("Comandos CLI", cliExample)}
            >
              <Copy size={13} /> Copiar
            </button>
          </div>
          <p style={{ margin: 0, fontSize: 12.5, color: "var(--text-secondary)" }}>
            Toda funcionalidade aceita entrada e saída JSON estruturada via binário <code>aicut</code>:
          </p>
          <pre className="code-block">{cliExample}</pre>
        </div>
      </div>

      {/* Live AI & Human Activity Feed */}
      <div className="panel" style={{ display: "flex", flexDirection: "column", gap: 10 }}>
        <div className="panel-header" style={{ marginBottom: 0 }}>
          <h2 className="panel-title">Feed de Atividade em Tempo Real (IA & Humano)</h2>
          <span className="pill">{activity.length} evento(s)</span>
        </div>
        {activity.length === 0 ? (
          <div style={{ color: "var(--text-secondary)", fontSize: 13 }}>
            Assim que você ou uma IA executar qualquer ação no Aicut, ela aparecerá aqui ao vivo.
          </div>
        ) : (
          <div style={{ display: "flex", flexDirection: "column", gap: 6, maxHeight: 280, overflowY: "auto" }}>
            {activity.map((entry) => (
              <div
                key={entry.id}
                style={{
                  display: "flex",
                  alignItems: "center",
                  justifyContent: "space-between",
                  gap: 10,
                  padding: "8px 10px",
                  borderRadius: "var(--radius-sm)",
                  background: "var(--bg-elevated)",
                  fontSize: 12.5,
                }}
              >
                <div style={{ display: "flex", alignItems: "center", gap: 8, minWidth: 0 }}>
                  <span className={`pill ${entry.actor === "ai" ? "accent" : ""}`}>
                    {entry.actor === "ai" ? <Bot size={11} /> : <User size={11} />}
                    {entry.actor === "ai" ? "IA" : "Humano"}
                  </span>
                  <code style={{ fontFamily: "var(--font-mono)", fontSize: 12 }}>{entry.action_id}</code>
                  <span style={{ color: "var(--text-secondary)", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
                    {entry.summary}
                  </span>
                </div>
                <span style={{ color: "var(--text-muted)", fontFamily: "var(--font-mono)", fontSize: 11.5, whiteSpace: "nowrap" }}>
                  {entry.duration_ms}ms · {formatRelativeDate(entry.timestamp)}
                </span>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
};
