# Aicut — Estúdio Modular AI-First para Linux (Arch / CachyOS & NVIDIA)

**Aicut** é um aplicativo desktop nativo para **Linux** (com foco em **Arch Linux / CachyOS** e aceleração **NVIDIA CUDA / NVENC / Vulkan**), com **core em Rust**, interface moderna em **Tauri v2 + React/TypeScript**, arquitetura **100% modular** e paridade total para qualquer chat ou agente de IA operar via **MCP / OpenAPI / CLI**.

## Principais Recursos

- **Importar & Baixar**: `yt-dlp` visual completo (presets, cortes por trecho, legendas, SponsorBlock, cookies) + importação local em lote.
- **Biblioteca**: player integrado com streaming HTTP Range, proxy H.264/NVENC automático para preview suave no WebKitGTK, metadados (`ffprobe`), busca, tags e lixeira.
- **Transcrever IA**: `whisper.cpp` local com modelos GGML selecionados por VRAM/CUDA/CPU e exportação `.srt`, `.vtt`, `.json` e FCPXML.
- **Cortes Shorts IA**: detecção viral com Gemini ou IA local na GPU (`Qwen2.5-VL` via `llama-server`), rastreamento facial ONNX (UltraFace), calibração visual de enquadramento 9:16 e renderização via **NVIDIA NVENC**.
- **Motion Studio**: editora IA conversacional + detector pixel a pixel de conteúdo pausado/estático + animações vetoriais HyperFrames renderizadas via Chromium CDP em RGBA transparente.
- **Configurações — Hardware NVIDIA & Gerenciador de Ferramentas**: detecta distro Linux, GPU NVIDIA (`nvidia-smi`), CUDA Runtime (`/opt/cuda`), Vulkan ICD e gerencia `ffmpeg`, `yt-dlp`, `whisper.cpp` e `llama.cpp`.
- **Paridade Humana + IA**: MCP (`stdio` + Streamable HTTP/SSE em `127.0.0.1:43210/mcp`), OpenAPI 3.1 (`/openapi.json`) e CLI JSON (`aicut run ...`).

## Instalação no CachyOS / Arch Linux

### Opção 1 — AppImage (sem instalação)
```bash
chmod +x Aicut_*_amd64.AppImage
./Aicut_*_amd64.AppImage
```
*(No Arch/CachyOS, se `fuse2` não estiver instalado, rode com `--appimage-extract-and-run` ou instale `sudo pacman -S fuse2`.)*

### Opção 2 — Pacote nativo Arch / CachyOS (`PKGBUILD`)
```bash
cd packaging/arch
makepkg -si
```

### Dependências recomendadas para NVIDIA máxima no CachyOS / Arch
```bash
sudo pacman -S ffmpeg yt-dlp chromium nvidia-utils vulkan-icd-loader
# Opcional para backend CUDA nativo no llama.cpp / whisper.cpp:
sudo pacman -S cuda
```

## Arquitetura do Monorepo

- `crates/aicut-core` — Microkernel modular, `AicutModule`, `ActionRegistry`, `EventBus`, `JobManager`, `HardwareProfiler` (Linux/NVIDIA), `ToolManager`, SQLite `MediaRepository`, `UiStateStore`.
- `crates/modules/*` — Módulos `media-ingest`, `transcribe`, `smart-clips` e `motion-studio`.
- `crates/aicut-runtime` — Kernel padrão + módulos built-in.
- `crates/aicut-mcp` — Servidor nativo MCP JSON-RPC 2.0 (`stdio` + HTTP/SSE).
- `crates/aicut-server` — Daemon local Axum (REST, OpenAPI 3.1, WebSocket, Media Streaming + NVENC preview proxy, MCP).
- `crates/aicut-cli` — Binário unificado `aicut`.
- `apps/web` — Interface React 19 + TypeScript + Vite.
- `apps/desktop` — Shell nativo Linux (Tauri v2 + proteção automática Wayland/NVIDIA WebKitGTK).
- `packaging/arch/PKGBUILD` — Receita nativa para Arch Linux / CachyOS.

## Desenvolvimento Local (Linux)

```bash
cargo test --workspace
cargo run -p aicut-cli -- serve
pnpm install
pnpm dev:web
NO_STRIP=true pnpm tauri:build
```

Veja também:
- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)
- [docs/AI_CONNECTIVITY.md](docs/AI_CONNECTIVITY.md)
