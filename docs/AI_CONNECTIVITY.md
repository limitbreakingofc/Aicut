# Conectando Qualquer Chat de IA ao Aicut

Com o Aicut aberto no Fedora (ou via `aicut serve`), o servidor local fica disponível em `http://127.0.0.1:43210`.

## 1. Claude Desktop (MCP stdio)

Adicione ao arquivo de configuração MCP do cliente:

```json
{
  "mcpServers": {
    "aicut": {
      "command": "aicut",
      "args": ["mcp"]
    }
  }
}
```

> Se o aplicativo Aicut estiver aberto no Fedora, `aicut mcp` conecta-se automaticamente à janela aberta para você acompanhar tudo ao vivo na tela. Se estiver fechado, executa de forma autônoma sobre o mesmo banco SQLite e workspace.

## 2. Cursor, VS Code, Windsurf e Antigravity (MCP Streamable HTTP)

```json
{
  "mcpServers": {
    "aicut": {
      "url": "http://127.0.0.1:43210/mcp"
    }
  }
}
```

Clientes MCP SSE legados podem conectar em `http://127.0.0.1:43210/mcp/sse`.

## 3. ChatGPT, Open WebUI, n8n e LangChain (OpenAPI 3.1)

- **OpenAPI 3.1 Spec**: `http://127.0.0.1:43210/openapi.json`
- **OpenAI Tools Array**: `http://127.0.0.1:43210/api/v1/tools/openai`
- **Anthropic Tools Array**: `http://127.0.0.1:43210/api/v1/tools/anthropic`

## 4. Agentes de Terminal (Claude Code, Codex CLI, Gemini CLI)

```bash
aicut doctor
aicut actions --schema
aicut run media.probe_url --json '{"url":"https://www.youtube.com/watch?v=..."}'
aicut run media.download --json '{"url":"https://www.youtube.com/watch?v=...","mode":"video_only","quality":"1080p","container":"mp4","wait":true}'
aicut run media.import_local --json '{"paths":["/home/usuario/Vídeos/meu_video.mp4"],"mode":"reference","wait":true}'
```
