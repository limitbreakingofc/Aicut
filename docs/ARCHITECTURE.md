# Arquitetura Modular do Aicut

O Aicut segue uma arquitetura **Microkernel + Módulos Plugáveis + Barramento Único de Actions**.

## 1. Como criar um novo módulo

Todo módulo implementa a trait `AicutModule` definida em `crates/aicut-core/src/module.rs`:

1. Crie um novo crate em `crates/modules/<seu-modulo>/` seguindo `crates/modules/media-ingest` como modelo.
2. Implemente `AicutModule`:
   - `manifest()`: metadados, versão e capacidades.
   - `declared_tools()`: ferramentas externas necessárias (ex: Whisper, ONNX, FFmpeg) — entram automaticamente no menu **Configurações → Ferramentas** com detecção de hardware e atualizador!
   - `actions()`: lista de `TypedAction<Input, Output, _>` — cada Action gera automaticamente:
     - JSON Schema (`schemars`)
     - Endpoint REST (`/api/v1/actions/<id>`)
     - Operação no `/openapi.json`
     - Tool no servidor **MCP** (`tools/list` e `tools/call`)
     - Definição para OpenAI & Anthropic SDKs
     - Suporte imediato no **Action Playground** da interface e na **CLI** (`aicut run`).
   - `health()`: diagnóstico de saúde exibido na aba **Módulos**.
3. Registre o módulo em `crates/aicut-runtime/src/lib.rs` dentro de `bootstrap_kernel`.

## 2. Paridade Humano ↔ IA

- Tanto o clique humano na interface quanto uma chamada de IA passam por `AicutKernel::dispatch(action_id, json_input, actor)`.
- Todas as mutações emitem eventos pelo `EventBus` (`tokio::sync::broadcast`), entregues em tempo real via WebSocket (`/api/v1/events`).
- O estado visual (`UiStateStore`) sincroniza rota ativa, mídia selecionada e busca entre o app desktop e qualquer agente conectado.
