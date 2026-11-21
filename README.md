# Aicut — Estúdio Modular AI-First para Fedora Linux

**Aicut** é um aplicativo desktop nativo para **Fedora Linux 44 e 45 (x86_64)**, com aceleração opcional **NVIDIA CUDA / NVENC / Vulkan**, com **core em Rust**, interface moderna em **Tauri v2 + React/TypeScript**, arquitetura **100% modular** e paridade total para qualquer chat ou agente de IA operar via **MCP / OpenAPI / CLI**.

## Principais Recursos

- **Importar & Baixar**: `yt-dlp` visual completo (presets, cortes por trecho, legendas, SponsorBlock, cookies) + importação local em lote.
- **Biblioteca**: player integrado com streaming HTTP Range, proxy H.264/NVENC automático para preview suave no WebKitGTK, metadados (`ffprobe`), busca, tags e lixeira.
- **Transcrever IA**: `whisper.cpp` local com modelos GGML selecionados por VRAM/CUDA/CPU e exportação `.srt`, `.vtt`, `.json` e FCPXML.
- **Cortes Shorts IA**: detecção viral com Gemini ou IA local na GPU (`Qwen2.5-VL` via `llama-server`), rastreamento facial ONNX (UltraFace), calibração visual de enquadramento 9:16 e renderização via **NVIDIA NVENC**.
- **Motion Studio**: editora IA conversacional + detector pixel a pixel de conteúdo pausado/estático + animações vetoriais HyperFrames renderizadas via Chromium CDP em RGBA transparente.
- **Downloads — Ferramentas e modelos locais**: detecta distro Linux, GPU NVIDIA (`nvidia-smi`), CUDA Runtime, Vulkan ICD e gerencia `ffmpeg`, `yt-dlp`, `whisper.cpp` e `llama.cpp`.
- **Paridade Humana + IA**: MCP (`stdio` + Streamable HTTP/SSE em `127.0.0.1:43210/mcp`), OpenAPI 3.1 (`/openapi.json`) e CLI JSON (`aicut run ...`).

Na aba **Downloads**, filtre os itens instalados ou escolha uma categoria. Para ferramentas, selecione uma tag compatível ou informe uma tag do projeto. Para modelos de voz e visão, escolha a variante no catálogo. O botão **Abrir pasta** mostra os arquivos gerenciados pelo seu usuário. A desinstalação de ferramentas remove a cópia do Aicut. O aplicativo volta a detectar executáveis presentes no PATH do Fedora.

## Instalação no Fedora Workstation

Baixe o RPM da [página de releases](https://github.com/limitbreakingofc/Aicut/releases)
e execute na pasta do arquivo:

```bash
sudo dnf install ./Aicut-*.x86_64.rpm
```

O RPM instala o aplicativo desktop. Para usar a CLI e o MCP via terminal, baixe também
`aicut-cli-fedora-x86_64` e instale-o em um diretório no seu `PATH`:

```bash
install -Dm755 aicut-cli-fedora-x86_64 ~/.local/bin/aicut
```

O Motion Studio usa Chromium para renderizar animações. Instale-o com
`sudo dnf install chromium`. A aceleração NVIDIA requer o driver e, conforme o
recurso, um runtime CUDA ou Vulkan. O pacote padrão usa `ffmpeg-free`; para
recursos que exigem H.264 ou NVENC, instale o FFmpeg pelo gerenciador de ferramentas do Aicut ou use um FFmpeg do sistema que inclua esses encoders.

## Arquitetura do Monorepo

- `crates/aicut-core` — Microkernel modular, `AicutModule`, `ActionRegistry`, `EventBus`, `JobManager`, `HardwareProfiler` (Linux/NVIDIA), `ToolManager`, SQLite `MediaRepository`, `UiStateStore`.
- `crates/modules/*` — Módulos `media-ingest`, `transcribe`, `smart-clips` e `motion-studio`.
- `crates/aicut-runtime` — Kernel padrão + módulos built-in.
- `crates/aicut-mcp` — Servidor nativo MCP JSON-RPC 2.0 (`stdio` + HTTP/SSE).
- `crates/aicut-server` — Daemon local Axum (REST, OpenAPI 3.1, WebSocket, Media Streaming + NVENC preview proxy, MCP).
- `crates/aicut-cli` — Binário unificado `aicut`.
- `apps/web` — Interface React 19 + TypeScript + Vite.
- `apps/desktop` — Shell nativo Linux (Tauri v2 + proteção automática Wayland/NVIDIA WebKitGTK).

## Desenvolvimento no Fedora

Instale Rust estável, Node.js 22 e pnpm 10, além das dependências do sistema:

```bash
sudo dnf install gcc gcc-c++ make pkgconf-pkg-config git curl file \
  webkit2gtk4.1-devel gtk3-devel libappindicator-gtk3-devel \
  openssl-devel librsvg2-devel libxdo-devel rpm-build \
  ffmpeg-free gstreamer1-plugins-base gstreamer1-plugins-good
pnpm install --frozen-lockfile
cargo test --workspace
pnpm tauri:dev
```

Para gerar o RPM localmente, execute `pnpm tauri:build`. O pacote fica em
`apps/desktop/src-tauri/target/release/bundle/rpm/`. Para desenvolver só a
interface, execute `cargo run -p aicut-cli -- serve` e `pnpm dev:web` em terminais separados.

Veja também:
- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)
- [docs/AI_CONNECTIVITY.md](docs/AI_CONNECTIVITY.md)
