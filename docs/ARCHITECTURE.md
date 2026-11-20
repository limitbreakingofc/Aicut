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

## Frontend (`apps/web`)

```
src/
├── App.tsx              # estado global, WebSocket de eventos, atalhos e roteamento das telas
├── app/                 # shell: navigation.ts (registro único de rotas), Topbar, Sidebar,
│                        # StatusBar, CommandPalette, ShortcutsDialog, Toaster, themes.ts
├── components/          # kit de UI: Panel, PageHeader, Tabs, Disclosure, ChoiceCards,
│                        # ProgressBar, Callout, Modal, ResizeHandle, ContextMenu…
├── styles/              # tokens.css (cores/tipografia/esquemas), base, layout, components
├── lib/                 # client (REST/WS), types, format, ui (cx, clipboard, storage)
└── modules/<módulo>/    # uma tela por módulo + CSS da própria tela
```

- **Nova tela**: registre a rota em `app/navigation.ts` (rótulo, ícone, grupo, descrição) e renderize a view em `App.tsx` — sidebar, breadcrumb, paleta de comandos e atalho `Alt+N` são gerados automaticamente.
- **Menus de botão direito**: dentro de qualquer view, `const menu = useContextMenu()` e `onContextMenu={(e) => menu.open(e, itens)}`; use `<MoreButton items={() => itens} />` para o botão `⋯` equivalente.
- **Cores e esquemas**: todos os componentes usam apenas os tokens de `styles/tokens.css`; um novo esquema é um bloco `[data-theme="…"]` nesse arquivo + uma entrada em `app/themes.ts`.

