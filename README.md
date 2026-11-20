# Aicut — Estúdio Modular AI-First para Windows 11

**Aicut** é um aplicativo desktop para **Windows 11** com **core em Rust**, interface moderna e fluida em **Tauri v2 + React/TypeScript**, arquitetura **100% modular** e controle completo para que **qualquer chat ou agente de IA** opere o app com a mesma paridade que um humano.

## Principais Recursos (v0.1.0)

- **Aba 1 — Baixar & Importar**:
  - **yt-dlp completo e visual**: analise URLs (`Probe`), escolha entre **Vídeo + Áudio**, **Só Vídeo** ou **Só Áudio**, qualidade/resolução ou stream exato (`format_id`), codec, container, qualidade de áudio, legendas (manuais, automáticas, embutidas), recorte de trecho (`--download-sections`), capítulos, playlists, cookies do navegador, SponsorBlock, fragmentos concorrentes, limite de velocidade, proxy e **presets salváveis**.
  - **Importação do PC**: importe vídeos e áudios locais (arquivo único, múltiplos arquivos ou pastas inteiras) em modo **Referência** (instantâneo) ou **Cópia para Workspace**.
- **Aba 2 — Biblioteca**:
  - Unifica tudo o que foi baixado via `yt-dlp` e todos os vídeos e áudios importados do PC em todos os formatos suportados pelo `yt-dlp`/FFmpeg.
  - Player integrado de vídeo e áudio com streaming HTTP Range, metadados técnicos (`ffprobe`), busca, filtros, tags e notas.
- **Configurações — Hardware & Atualizador Extensível de Ferramentas**:
  - Detecta automaticamente CPU, arquitetura, RAM e GPU (NVIDIA/AMD/Intel) para baixar e atualizar as **melhores versões e builds** do `yt-dlp`, `FFmpeg`, `FFprobe` e futuras ferramentas declaradas por novos módulos.
- **Qualquer Chat de IA com Paridade Humana**:
  - **MCP (Model Context Protocol)**: Streamable HTTP (`http://127.0.0.1:43210/mcp`), SSE (`/mcp/sse`) e `stdio` (`aicut mcp` — com ponte automática para a janela aberta!).
  - **OpenAPI 3.1 + REST + WebSocket**: `http://127.0.0.1:43210/openapi.json`, `/api/v1/tools/openai`, `/api/v1/tools/anthropic`.
  - **CLI estruturada**: `aicut run <action> --json '{...}'`.
  - **Sincronização bidirecional de UI**: a IA vê em qual aba/vídeo você está e pode navegar, abrir vídeos na tela e notificar você ao vivo.

## Interface (v0.4.5)

- **Estilo terminal moderno, em largura total**: fonte mono, painéis com bordas finas, tabelas densas e barra de status inferior clicável (core MCP, tarefas, mídias, ferramentas, GPU).
- **Menus de botão direito** em mídias, seleções em lote, tarefas, transcrições, cortes, modelos, ferramentas e módulos (também via botões `⋯` e `Shift+F10`).
- **Atalhos**: `Ctrl+K` paleta · `Alt+1…8` telas · `Ctrl+B` menu lateral · `?` lista completa · Biblioteca: `/` busca, `↑ ↓`, `Espaço`, `Ctrl+A`, `Del`.
- **Esquemas de cor**: Noite, Fósforo, Âmbar e Claro (Configurações → Aparência ou botão de paleta no topo).

## Arquitetura do Monorepo

- `crates/aicut-core` — Microkernel modular, `AicutModule` trait, `ActionRegistry` (JSON Schema), `EventBus`, `JobManager`, `HardwareProfiler`, `ToolManager`, SQLite `MediaRepository`, `UiStateStore`.
- `crates/modules/media-ingest` — Módulo #1 (`yt-dlp`, importação local, `ffprobe`, thumbnails, Biblioteca, presets).
- `crates/aicut-runtime` — Kernel padrão + módulos built-in (`system`, `tools`, `jobs`, `ui`).
- `crates/aicut-mcp` — Servidor nativo MCP JSON-RPC 2.0 (`stdio` + HTTP/SSE).
- `crates/aicut-server` — Daemon local Axum (REST, OpenAPI 3.1, WebSocket, Media Streaming, MCP).
- `crates/aicut-cli` — Binário unificado `aicut`.
- `apps/web` — Interface estilo terminal moderno (React 19 + TypeScript + Vite): shell em `src/app`, kit de componentes em `src/components`, design tokens em `src/styles` e uma pasta por módulo em `src/modules`.
- `apps/desktop` — Shell nativo Windows 11 (Tauri v2).

## Desenvolvimento Local

```bash
# Testes do backend Rust
cargo test --workspace

# Rodar daemon local (porta 43210)
cargo run -p aicut-cli -- serve

# Rodar interface web em modo dev (outro terminal)
pnpm install
pnpm dev:web

# Build do app Windows 11 (em máquina Windows 11)
pnpm tauri:build
```

Veja também:
- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)
- [docs/AI_CONNECTIVITY.md](docs/AI_CONNECTIVITY.md)
