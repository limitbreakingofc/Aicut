use aicut_core::{
    download_rnnoise_model, enhance_media_audio_to_library, installed_rnnoise_model,
    load_audio_master_config, rnnoise_model_path, save_audio_master_config, ActionAnnotations,
    ActionHandler, ActionSpec, ActivityEntry, AicutError, AicutEvent, AicutKernel, AicutModule,
    AicutResult, AudioEnhanceResult, AudioMasterConfig, EmptyInput, HardwareProfile, JobRecord,
    ModuleInfo, ModuleManifest, NotificationLevel, ToolStatus, TypedAction, UiState,
    WorkspaceConfig,
};
use aicut_module_media_ingest::MediaIngestModule;
use aicut_module_motion_studio::MotionStudioModule;
use aicut_module_smart_clips::SmartClipsModule;
use aicut_module_transcribe::TranscribeModule;
use async_trait::async_trait;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::sync::Arc;

/// Bootstraps the default AicutKernel with all core modules and built-in feature modules.
pub async fn bootstrap_kernel(workspace: WorkspaceConfig) -> AicutResult<Arc<AicutKernel>> {
    let kernel = AicutKernel::new(workspace)?;
    kernel.register_module(Arc::new(SystemModule)).await?;
    kernel.register_module(Arc::new(ToolsModule)).await?;
    kernel.register_module(Arc::new(JobsModule)).await?;
    kernel.register_module(Arc::new(UiModule)).await?;
    kernel.register_module(MediaIngestModule::new()).await?;
    kernel.register_module(TranscribeModule::new()).await?;
    kernel.register_module(SmartClipsModule::new()).await?;
    kernel.register_module(MotionStudioModule::new()).await?;
    kernel.register_module(Arc::new(AudioMasterModule)).await?;
    Ok(kernel)
}

// ============================================================================
// 1. SystemModule (info, hardware, modules, settings, activity)
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct SystemInfoOutput {
    pub app_name: String,
    pub version: String,
    pub workspace: WorkspaceConfig,
    pub hardware: HardwareProfile,
    pub media_count: usize,
    pub active_jobs: usize,
    pub enabled_modules: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct ListModulesOutput {
    pub modules: Vec<ModuleInfo>,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct SetModuleEnabledInput {
    pub module_id: String,
    pub enabled: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct SetModuleEnabledOutput {
    pub module_id: String,
    pub enabled: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct DescribeActionInput {
    pub action_id: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct UpdateSettingInput {
    pub key: String,
    pub value: Value,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct SettingsMapOutput {
    pub settings: serde_json::Map<String, Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct ActivityInput {
    #[serde(default = "default_activity_limit")]
    pub limit: usize,
}

fn default_activity_limit() -> usize {
    50
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct ActivityOutput {
    pub entries: Vec<ActivityEntry>,
}

pub struct SystemModule;

#[async_trait]
impl AicutModule for SystemModule {
    fn manifest(&self) -> ModuleManifest {
        ModuleManifest {
            id: "system".to_string(),
            name: "Núcleo do Sistema".to_string(),
            version: env!("CARGO_PKG_VERSION").to_string(),
            description: "Informações do aplicativo, perfil de hardware detectado, gestão de módulos, configurações e histórico de atividade.".to_string(),
            category: "core".to_string(),
            core: true,
            capabilities: vec!["hardware-profiling".into(), "module-management".into(), "settings".into()],
        }
    }

    fn actions(&self) -> Vec<Arc<dyn ActionHandler>> {
        vec![
            Arc::new(TypedAction::<EmptyInput, SystemInfoOutput, _>::new(
                "system.info",
                "system",
                "Informações do Sistema",
                "Get Aicut version, workspace paths, detected hardware profile, media count, and active jobs summary.",
                ActionAnnotations { read_only: true, idempotent: true, ..Default::default() },
                |ctx, _| {
                    Box::pin(async move {
                        let modules = ctx.kernel.list_modules().await;
                        let enabled_modules = modules.iter().filter(|m| m.enabled).count();
                        let media_count = ctx.kernel.repo.count_media().unwrap_or(0);
                        let active_jobs = ctx.kernel.jobs.list(true).len();
                        let out = SystemInfoOutput {
                            app_name: "Aicut".to_string(),
                            version: env!("CARGO_PKG_VERSION").to_string(),
                            workspace: ctx.kernel.workspace.clone(),
                            hardware: ctx.kernel.hardware.clone(),
                            media_count,
                            active_jobs,
                            enabled_modules,
                        };
                        let summary = format!(
                            "Aicut v{} ({}) — {} media item(s), {} active job(s)",
                            out.version, out.hardware.platform_tag, media_count, active_jobs
                        );
                        Ok((out, summary, vec![]))
                    })
                },
            )),
            Arc::new(TypedAction::<EmptyInput, HardwareProfile, _>::new(
                "system.hardware_info",
                "system",
                "Perfil de Hardware Detectado",
                "Inspect detected OS, CPU architecture, cores, memory, GPUs (NVIDIA/AMD/Intel/Apple), and recommended hardware acceleration backend used for selecting optimal tool builds.",
                ActionAnnotations { read_only: true, idempotent: true, ..Default::default() },
                |ctx, _| {
                    Box::pin(async move {
                        let hw = ctx.kernel.hardware.clone();
                        let summary = hw.recommendation_summary.clone();
                        Ok((hw, summary, vec![]))
                    })
                },
            )),
            Arc::new(TypedAction::<EmptyInput, ListModulesOutput, _>::new(
                "system.list_modules",
                "system",
                "Listar Módulos",
                "List all installed Aicut modules, their health, enabled state, actions, and managed tools.",
                ActionAnnotations { read_only: true, idempotent: true, ..Default::default() },
                |ctx, _| {
                    Box::pin(async move {
                        let modules = ctx.kernel.list_modules().await;
                        let summary = format!("{} module(s) registered", modules.len());
                        Ok((ListModulesOutput { modules }, summary, vec![]))
                    })
                },
            )),
            Arc::new(TypedAction::<SetModuleEnabledInput, SetModuleEnabledOutput, _>::new(
                "system.set_module_enabled",
                "system",
                "Ativar/Desativar Módulo",
                "Enable or disable a non-core Aicut module dynamically.",
                ActionAnnotations { idempotent: true, ..Default::default() },
                |ctx, input| {
                    Box::pin(async move {
                        let enabled = ctx.kernel.set_module_enabled(&input.module_id, input.enabled)?;
                        let summary = format!(
                            "Module '{}' is now {}",
                            input.module_id,
                            if enabled { "enabled" } else { "disabled" }
                        );
                        Ok((
                            SetModuleEnabledOutput {
                                module_id: input.module_id,
                                enabled,
                            },
                            summary,
                            vec![],
                        ))
                    })
                },
            )),
            Arc::new(TypedAction::<DescribeActionInput, ActionSpec, _>::new(
                "system.describe_action",
                "system",
                "Descrever Action",
                "Return the full specification and JSON Schema for a specific Aicut action.",
                ActionAnnotations { read_only: true, idempotent: true, ..Default::default() },
                |ctx, input| {
                    Box::pin(async move {
                        let spec = ctx.kernel.get_action_spec(&input.action_id).ok_or_else(|| {
                            AicutError::not_found(format!("Action '{}' not found", input.action_id))
                        })?;
                        let summary = format!("Schema for action '{}'", spec.id);
                        Ok((spec, summary, vec![]))
                    })
                },
            )),
            Arc::new(TypedAction::<EmptyInput, SettingsMapOutput, _>::new(
                "system.get_settings",
                "system",
                "Ler Configurações",
                "Return all persisted user settings.",
                ActionAnnotations { read_only: true, idempotent: true, ..Default::default() },
                |ctx, _| {
                    Box::pin(async move {
                        let settings = ctx.kernel.repo.list_settings()?;
                        Ok((SettingsMapOutput { settings }, "Loaded user settings".into(), vec![]))
                    })
                },
            )),
            Arc::new(TypedAction::<UpdateSettingInput, SettingsMapOutput, _>::new(
                "system.set_setting",
                "system",
                "Salvar Configuração",
                "Persist a key/value user setting (e.g., theme, default_import_mode).",
                ActionAnnotations { idempotent: true, ..Default::default() },
                |ctx, input| {
                    Box::pin(async move {
                        ctx.kernel.repo.set_setting(&input.key, &input.value)?;
                        let settings = ctx.kernel.repo.list_settings()?;
                        Ok((
                            SettingsMapOutput { settings },
                            format!("Saved setting '{}'", input.key),
                            vec![],
                        ))
                    })
                },
            )),
            Arc::new(TypedAction::<ActivityInput, ActivityOutput, _>::new(
                "system.recent_activity",
                "system",
                "Atividade Recente",
                "List recent actions executed by humans and AI agents.",
                ActionAnnotations { read_only: true, idempotent: true, ..Default::default() },
                |ctx, input| {
                    Box::pin(async move {
                        let entries = ctx.kernel.events.recent_activity(input.limit.min(200));
                        let summary = format!("Returned {} activity entry(ies)", entries.len());
                        Ok((ActivityOutput { entries }, summary, vec![]))
                    })
                },
            )),
        ]
    }
}

// ============================================================================
// 2. ToolsModule (Hardware-aware Tool Installer & Updater)
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct ToolsListOutput {
    pub hardware: HardwareProfile,
    pub tools: Vec<ToolStatus>,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct ToolInstallInput {
    pub tool_id: String,
}

pub struct ToolsModule;

#[async_trait]
impl AicutModule for ToolsModule {
    fn manifest(&self) -> ModuleManifest {
        ModuleManifest {
            id: "tools".to_string(),
            name: "Gerenciador de Ferramentas".to_string(),
            version: env!("CARGO_PKG_VERSION").to_string(),
            description: "Detecta o hardware e instala/atualiza automaticamente as melhores versões de yt-dlp, FFmpeg e futuras ferramentas de módulos.".to_string(),
            category: "core".to_string(),
            core: true,
            capabilities: vec!["tool-install".into(), "tool-update".into(), "hardware-selection".into()],
        }
    }

    fn actions(&self) -> Vec<Arc<dyn ActionHandler>> {
        vec![
            Arc::new(TypedAction::<EmptyInput, ToolsListOutput, _>::new(
                "tools.list",
                "tools",
                "Status das Ferramentas",
                "List all external tools declared by installed modules (yt-dlp, ffmpeg, ffprobe, and future tools) along with installed versions, paths, and hardware-matched build recommendations.",
                ActionAnnotations { read_only: true, idempotent: true, ..Default::default() },
                |ctx, _| {
                    Box::pin(async move {
                        let tools = ctx.kernel.tools.list_statuses().await;
                        let installed = tools.iter().filter(|t| t.installed).count();
                        let summary = format!("{installed}/{} tool(s) installed", tools.len());
                        Ok((
                            ToolsListOutput {
                                hardware: ctx.kernel.hardware.clone(),
                                tools,
                            },
                            summary,
                            vec![],
                        ))
                    })
                },
            )),
            Arc::new(TypedAction::<EmptyInput, ToolsListOutput, _>::new(
                "tools.check_updates",
                "tools",
                "Verificar Atualizações de Ferramentas",
                "Query official release channels to check if newer versions of yt-dlp, FFmpeg, or other managed tools are available for this hardware.",
                ActionAnnotations { read_only: true, ..Default::default() },
                |ctx, _| {
                    Box::pin(async move {
                        let tools = ctx.kernel.tools.check_updates().await?;
                        let updates = tools.iter().filter(|t| t.update_available).count();
                        let summary = format!("Checked updates: {updates} tool(s) can be installed or updated");
                        Ok((
                            ToolsListOutput {
                                hardware: ctx.kernel.hardware.clone(),
                                tools,
                            },
                            summary,
                            vec![],
                        ))
                    })
                },
            )),
            Arc::new(TypedAction::<ToolInstallInput, ToolStatus, _>::new(
                "tools.install",
                "tools",
                "Instalar ou Atualizar Ferramenta",
                "Automatically detect hardware and install or update a specific tool (e.g. 'yt-dlp' or 'ffmpeg') with the optimal build.",
                ActionAnnotations { long_running: true, ..Default::default() },
                |ctx, input| {
                    Box::pin(async move {
                        let status = ctx.kernel.tools.install_or_update(&input.tool_id, ctx.actor).await?;
                        let summary = format!(
                            "Installed/updated {} ({})",
                            status.display_name,
                            status.installed_version.as_deref().unwrap_or("ready")
                        );
                        Ok((status, summary, vec![]))
                    })
                },
            )),
            Arc::new(TypedAction::<EmptyInput, ToolsListOutput, _>::new(
                "tools.update_all",
                "tools",
                "Atualizar Todas as Ferramentas",
                "Install missing tools and update all manageable tools (yt-dlp, ffmpeg, etc.) to the best hardware-matched releases.",
                ActionAnnotations { long_running: true, ..Default::default() },
                |ctx, _| {
                    Box::pin(async move {
                        let current = ctx.kernel.tools.list_statuses().await;
                        for tool in current {
                            // Skip companion binaries like ffprobe (updated together with ffmpeg)
                            if tool.id == "ffprobe" {
                                continue;
                            }
                            let _ = ctx.kernel.tools.install_or_update(&tool.id, ctx.actor).await;
                        }
                        let tools = ctx.kernel.tools.list_statuses().await;
                        let summary = "Completed update sweep for all managed tools".to_string();
                        Ok((
                            ToolsListOutput {
                                hardware: ctx.kernel.hardware.clone(),
                                tools,
                            },
                            summary,
                            vec![],
                        ))
                    })
                },
            )),
        ]
    }
}

// ============================================================================
// 3. JobsModule
// ============================================================================

#[derive(Debug, Clone, Default, Serialize, Deserialize, JsonSchema)]
pub struct JobsListInput {
    #[serde(default)]
    pub only_active: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct JobsListOutput {
    pub jobs: Vec<JobRecord>,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct JobIdInput {
    pub job_id: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct JobsClearOutput {
    pub removed_count: usize,
}

pub struct JobsModule;

#[async_trait]
impl AicutModule for JobsModule {
    fn manifest(&self) -> ModuleManifest {
        ModuleManifest {
            id: "jobs".to_string(),
            name: "Fila de Tarefas".to_string(),
            version: env!("CARGO_PKG_VERSION").to_string(),
            description: "Monitora progresso em tempo real, logs e cancelamento de downloads, importações e instalações.".to_string(),
            category: "core".to_string(),
            core: true,
            capabilities: vec!["job-queue".into(), "cancellation".into()],
        }
    }

    fn actions(&self) -> Vec<Arc<dyn ActionHandler>> {
        vec![
            Arc::new(TypedAction::<JobsListInput, JobsListOutput, _>::new(
                "jobs.list",
                "jobs",
                "Listar Tarefas",
                "List active and finished background jobs (downloads, imports, tool installs).",
                ActionAnnotations { read_only: true, idempotent: true, ..Default::default() },
                |ctx, input| {
                    Box::pin(async move {
                        let jobs = ctx.kernel.jobs.list(input.only_active);
                        let summary = format!("{} job(s) listed", jobs.len());
                        Ok((JobsListOutput { jobs }, summary, vec![]))
                    })
                },
            )),
            Arc::new(TypedAction::<JobIdInput, JobRecord, _>::new(
                "jobs.get",
                "jobs",
                "Detalhes da Tarefa",
                "Get live progress, speed, ETA, logs, and result of a specific job.",
                ActionAnnotations { read_only: true, idempotent: true, ..Default::default() },
                |ctx, input| {
                    Box::pin(async move {
                        let job = ctx.kernel.jobs.get(&input.job_id).ok_or_else(|| {
                            AicutError::not_found(format!("Job '{}' not found", input.job_id))
                        })?;
                        let summary = format!("Job '{}' is {:?} ({:.0}%)", job.title, job.status, job.progress.percent);
                        Ok((job, summary, vec![]))
                    })
                },
            )),
            Arc::new(TypedAction::<JobIdInput, JobRecord, _>::new(
                "jobs.cancel",
                "jobs",
                "Cancelar Tarefa",
                "Cancel an active background job immediately.",
                ActionAnnotations { destructive: true, ..Default::default() },
                |ctx, input| {
                    Box::pin(async move {
                        let job = ctx.kernel.jobs.cancel(&input.job_id)?;
                        let summary = format!("Cancelled job '{}'", job.id);
                        Ok((job, summary, vec![]))
                    })
                },
            )),
            Arc::new(TypedAction::<EmptyInput, JobsClearOutput, _>::new(
                "jobs.clear_finished",
                "jobs",
                "Limpar Tarefas Concluídas",
                "Remove completed, failed, and cancelled jobs from the queue history.",
                ActionAnnotations { idempotent: true, ..Default::default() },
                |ctx, _| {
                    Box::pin(async move {
                        let removed_count = ctx.kernel.jobs.clear_finished();
                        let summary = format!("Cleared {removed_count} finished job(s)");
                        Ok((JobsClearOutput { removed_count }, summary, vec![]))
                    })
                },
            )),
        ]
    }
}

// ============================================================================
// 4. UiModule (Bidirectional Human <-> AI UI control)
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct UiNavigateInput {
    /// Target tab: "ingest" | "library" | "jobs" | "modules" | "ai-connect" | "settings"
    pub route: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct UiSelectMediaInput {
    pub media_id: Option<String>,
    /// When true (default), also navigates the desktop UI to the Library tab.
    #[serde(default = "default_true")]
    pub open_library: bool,
}

fn default_true() -> bool {
    true
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct UiSetSearchInput {
    pub query: String,
    #[serde(default)]
    pub filter_kind: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct UiNotifyInput {
    pub title: String,
    pub message: String,
    #[serde(default = "default_level")]
    pub level: NotificationLevel,
}

fn default_level() -> NotificationLevel {
    NotificationLevel::Info
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct UiNotifyOutput {
    pub sent: bool,
}

pub struct UiModule;

#[async_trait]
impl AicutModule for UiModule {
    fn manifest(&self) -> ModuleManifest {
        ModuleManifest {
            id: "ui".to_string(),
            name: "Ponte de Interface (AI ↔ UI)".to_string(),
            version: env!("CARGO_PKG_VERSION").to_string(),
            description: "Permite que qualquer chat de IA leia a tela atual, navegue abas, selecione vídeos na Biblioteca e envie notificações visuais ao usuário.".to_string(),
            category: "core".to_string(),
            core: true,
            capabilities: vec!["ui-awareness".into(), "ui-navigation".into(), "notifications".into()],
        }
    }

    fn actions(&self) -> Vec<Arc<dyn ActionHandler>> {
        vec![
            Arc::new(TypedAction::<EmptyInput, UiState, _>::new(
                "ui.get_state",
                "ui",
                "Ler Estado Atual da Tela",
                "Inspect what the human user currently sees in the Aicut desktop window (active tab, selected media item, search filter, player state).",
                ActionAnnotations { read_only: true, idempotent: true, ..Default::default() },
                |ctx, _| {
                    Box::pin(async move {
                        let st = ctx.kernel.ui.get();
                        let summary = format!(
                            "Active tab: '{}', selected media: {:?}",
                            st.active_route, st.selected_media_id
                        );
                        Ok((st, summary, vec![]))
                    })
                },
            )),
            Arc::new(TypedAction::<UiNavigateInput, UiState, _>::new(
                "ui.navigate",
                "ui",
                "Navegar para Aba",
                "Switch the visible tab in the Aicut desktop window ('ingest', 'library', 'transcribe', 'smart-clips', 'framing', 'motion-studio', 'jobs', 'modules', 'ai-connect', 'settings').",
                ActionAnnotations { idempotent: true, ..Default::default() },
                |ctx, input| {
                    Box::pin(async move {
                        let allowed = ["ingest", "library", "transcribe", "smart-clips", "framing", "motion-studio", "audio-master", "jobs", "modules", "ai-connect", "settings"];
                        if !allowed.contains(&input.route.as_str()) {
                            return Err(AicutError::invalid_input(format!(
                                "Unknown route '{}'. Valid routes: {}",
                                input.route,
                                allowed.join(", ")
                            )));
                        }
                        let st = ctx.kernel.ui.update(|s| s.active_route = input.route.clone());
                        let summary = format!("Navigated UI to '{}'", st.active_route);
                        Ok((st, summary, vec![]))
                    })
                },
            )),
            Arc::new(TypedAction::<UiSelectMediaInput, UiState, _>::new(
                "ui.select_media",
                "ui",
                "Selecionar Mídia na Tela",
                "Select and open a specific media item in the Library preview player so the user can see it immediately.",
                ActionAnnotations { idempotent: true, ..Default::default() },
                |ctx, input| {
                    Box::pin(async move {
                        let st = ctx.kernel.ui.update(|s| {
                            s.selected_media_id = input.media_id.clone();
                            if input.open_library && input.media_id.is_some() {
                                s.active_route = "library".to_string();
                            }
                        });
                        let summary = format!("Selected media {:?} in UI", st.selected_media_id);
                        Ok((st, summary, vec![]))
                    })
                },
            )),
            Arc::new(TypedAction::<UiSetSearchInput, UiState, _>::new(
                "ui.set_search",
                "ui",
                "Filtrar Biblioteca na Tela",
                "Update the Library search box and optional media kind filter visible on screen.",
                ActionAnnotations { idempotent: true, ..Default::default() },
                |ctx, input| {
                    Box::pin(async move {
                        let st = ctx.kernel.ui.update(|s| {
                            s.search_query = input.query;
                            s.active_filter_kind = input.filter_kind;
                            s.active_route = "library".to_string();
                        });
                        let summary = format!("Updated Library search filter to '{}'", st.search_query);
                        Ok((st, summary, vec![]))
                    })
                },
            )),
            Arc::new(TypedAction::<UiNotifyInput, UiNotifyOutput, _>::new(
                "ui.notify",
                "ui",
                "Enviar Notificação na Tela",
                "Display a real-time toast notification inside the Aicut desktop UI.",
                ActionAnnotations::default(),
                |ctx, input| {
                    Box::pin(async move {
                        ctx.kernel.events.publish(AicutEvent::UiNotification {
                            level: input.level,
                            title: input.title.clone(),
                            message: input.message.clone(),
                            actor: ctx.actor,
                        });
                        let summary = format!("Displayed notification '{}'", input.title);
                        Ok((UiNotifyOutput { sent: true }, summary, vec![]))
                    })
                },
            )),
        ]
    }
}

// ============================================================================
// 5. AudioMasterModule (YouTube -14 LUFS auto-mastering, RNNoise neural denoise)
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct AudioConfigOutput {
    pub config: AudioMasterConfig,
    pub rnnoise_installed: bool,
    pub rnnoise_path: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct AudioConfigSetInput {
    #[serde(default)]
    pub preset: Option<String>,
    #[serde(default)]
    pub denoise: Option<String>,
    #[serde(default)]
    pub voice_eq: Option<bool>,
    #[serde(default)]
    pub auto_compressor: Option<bool>,
    #[serde(default)]
    pub target_lufs: Option<f64>,
    #[serde(default)]
    pub true_peak_db: Option<f64>,
    #[serde(default)]
    pub lra: Option<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct AudioEnhanceInput {
    pub media_id: String,
    #[serde(default)]
    pub preset: Option<String>,
    #[serde(default)]
    pub wait: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct AudioEnhanceJobOutput {
    pub job_id: String,
    pub status: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub result: Option<AudioEnhanceResult>,
}

pub struct AudioMasterModule;

#[async_trait]
impl AicutModule for AudioMasterModule {
    fn manifest(&self) -> ModuleManifest {
        ModuleManifest {
            id: "audio-master".to_string(),
            name: "Áudio IA (Masterização YouTube)".to_string(),
            version: env!("CARGO_PKG_VERSION").to_string(),
            description: "Masterização automática de áudio para YouTube (-14 LUFS / -1.0 dBTP), redução neural de ruído (RNNoise), equalização de voz e compressão dinâmica — aplicada automaticamente nos Cortes Shorts, no Motion Studio e em mídias da Biblioteca.".to_string(),
            category: "audio".to_string(),
            core: false,
            capabilities: vec![
                "youtube-loudness-14lufs".into(),
                "rnnoise-neural-denoise".into(),
                "auto-voice-compression".into(),
            ],
        }
    }

    fn actions(&self) -> Vec<Arc<dyn ActionHandler>> {
        vec![
            Arc::new(TypedAction::<EmptyInput, AudioConfigOutput, _>::new(
                "audio.config.get",
                "audio-master",
                "Obter Configuração de Masterização de Áudio",
                "Get the active automatic YouTube audio mastering preset (-14 LUFS), noise reduction mode, voice EQ, compression settings, and RNNoise neural model status.",
                ActionAnnotations { read_only: true, idempotent: true, ..Default::default() },
                |ctx, _| {
                    Box::pin(async move {
                        let config = load_audio_master_config(&ctx.kernel);
                        let rnnoise_installed =
                            installed_rnnoise_model(&ctx.kernel.workspace.models_dir).is_some();
                        let rnnoise_path =
                            rnnoise_model_path(&ctx.kernel.workspace.models_dir)
                                .to_string_lossy()
                                .to_string();
                        let summary = format!(
                            "Audio preset '{}' ({:.1} LUFS, RNNoise: {})",
                            config.preset, config.target_lufs, rnnoise_installed
                        );
                        Ok((
                            AudioConfigOutput {
                                config,
                                rnnoise_installed,
                                rnnoise_path,
                            },
                            summary,
                            vec![],
                        ))
                    })
                },
            )),
            Arc::new(TypedAction::<AudioConfigSetInput, AudioConfigOutput, _>::new(
                "audio.config.set",
                "audio-master",
                "Salvar Configuração de Masterização de Áudio",
                "Update the automatic audio mastering preset (youtube_auto, reels_tiktok, podcast_voice, off) and voice processing options applied automatically to all Cortes Shorts and Motion Studio renders.",
                ActionAnnotations { idempotent: true, ..Default::default() },
                |ctx, input| {
                    Box::pin(async move {
                        let mut cfg = if let Some(ref p) = input.preset {
                            AudioMasterConfig::from_preset(p)
                        } else {
                            load_audio_master_config(&ctx.kernel)
                        };
                        if let Some(d) = input.denoise {
                            cfg.denoise = d;
                        }
                        if let Some(eq) = input.voice_eq {
                            cfg.voice_eq = eq;
                        }
                        if let Some(comp) = input.auto_compressor {
                            cfg.auto_compressor = comp;
                        }
                        if let Some(lufs) = input.target_lufs {
                            cfg.target_lufs = lufs;
                        }
                        if let Some(tp) = input.true_peak_db {
                            cfg.true_peak_db = tp;
                        }
                        if let Some(lra) = input.lra {
                            cfg.lra = lra;
                        }
                        let saved = save_audio_master_config(&ctx.kernel, &cfg)?;
                        let rnnoise_installed =
                            installed_rnnoise_model(&ctx.kernel.workspace.models_dir).is_some();
                        let rnnoise_path =
                            rnnoise_model_path(&ctx.kernel.workspace.models_dir)
                                .to_string_lossy()
                                .to_string();
                        let summary = format!(
                            "Saved audio mastering preset '{}' ({:.1} LUFS / {:.1} dBTP)",
                            saved.preset, saved.target_lufs, saved.true_peak_db
                        );
                        Ok((
                            AudioConfigOutput {
                                config: saved,
                                rnnoise_installed,
                                rnnoise_path,
                            },
                            summary,
                            vec![],
                        ))
                    })
                },
            )),
            Arc::new(TypedAction::<EmptyInput, AudioConfigOutput, _>::new(
                "audio.install_rnnoise",
                "audio-master",
                "Instalar Modelo Neural RNNoise (Grátis · 85 KB)",
                "Download the open-source RNNoise neural speech denoising model (bd.rnnn) for AI background noise removal inside FFmpeg.",
                ActionAnnotations { idempotent: true, ..Default::default() },
                |ctx, _| {
                    Box::pin(async move {
                        let path = download_rnnoise_model(&ctx.kernel).await?;
                        let config = load_audio_master_config(&ctx.kernel);
                        let summary = format!("Installed RNNoise neural model at {}", path.display());
                        Ok((
                            AudioConfigOutput {
                                config,
                                rnnoise_installed: true,
                                rnnoise_path: path.to_string_lossy().to_string(),
                            },
                            summary,
                            vec![],
                        ))
                    })
                },
            )),
            Arc::new(TypedAction::<AudioEnhanceInput, AudioEnhanceJobOutput, _>::new(
                "audio.enhance",
                "audio-master",
                "Masterizar Áudio de Mídia da Biblioteca (YouTube -14 LUFS)",
                "Run automatic AI audio mastering (2-pass EBU R128 loudness normalization, RNNoise/spectral denoise, voice EQ & auto compression) on any Library video or audio without re-encoding video.",
                ActionAnnotations { long_running: true, ..Default::default() },
                |ctx, input| {
                    Box::pin(async move {
                        let source = ctx
                            .kernel
                            .repo
                            .get_media(&input.media_id)?
                            .ok_or_else(|| {
                                AicutError::not_found(format!(
                                    "Mídia '{}' não encontrada",
                                    input.media_id
                                ))
                            })?;
                        let (job, cancel) = ctx.kernel.jobs.create_job(
                            "audio.enhance",
                            format!("Masterizar áudio de '{}'", source.title),
                            "audio-master",
                            ctx.actor,
                        );
                        let job_id = job.id.clone();
                        let kernel = ctx.kernel.clone();
                        let task_job_id = job_id.clone();
                        let media_id = input.media_id.clone();
                        let override_cfg = input
                            .preset
                            .as_deref()
                            .map(AudioMasterConfig::from_preset);

                        let handle = tokio::spawn(async move {
                            kernel.jobs.mark_running(&task_job_id, "mastering_audio");
                            if cancel.is_cancelled() {
                                kernel.jobs.mark_cancelled(&task_job_id);
                                return Err(AicutError::cancelled("Masterização cancelada."));
                            }
                            match enhance_media_audio_to_library(
                                &kernel,
                                &media_id,
                                override_cfg,
                                Some(&task_job_id),
                            )
                            .await
                            {
                                Ok(res) => {
                                    kernel.jobs.complete(
                                        &task_job_id,
                                        serde_json::to_value(&res).unwrap_or(Value::Null),
                                    );
                                    Ok(res)
                                }
                                Err(err) => {
                                    kernel.jobs.fail(&task_job_id, err.clone());
                                    Err(err)
                                }
                            }
                        });

                        if input.wait {
                            let res = handle
                                .await
                                .map_err(|e| AicutError::internal(format!("Audio task panicked: {e}")))??;
                            let summary = format!(
                                "Mastered '{}' ({:.1} LUFS)",
                                res.asset.title, res.target_lufs
                            );
                            Ok((
                                AudioEnhanceJobOutput {
                                    job_id,
                                    status: "completed".to_string(),
                                    result: Some(res),
                                },
                                summary,
                                vec![],
                            ))
                        } else {
                            Ok((
                                AudioEnhanceJobOutput {
                                    job_id: job_id.clone(),
                                    status: "queued".to_string(),
                                    result: None,
                                },
                                format!("Started automatic audio mastering job {job_id}"),
                                vec![],
                            ))
                        }
                    })
                },
            )),
        ]
    }
}
