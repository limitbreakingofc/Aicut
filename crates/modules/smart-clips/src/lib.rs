pub mod engine;
pub mod frames;
pub mod gemini;
pub mod local_vlm;
pub mod models;
pub mod postprocess;
pub mod prompts;
pub mod renderer;

use aicut_core::{
    ActionAnnotations, ActionHandler, AicutError, AicutEvent, AicutKernel, AicutModule, EmptyInput,
    ModuleHealth, ModuleManifest, SmartClipRun, ToolDefinition, ToolInstallStrategy, TypedAction,
};
use async_trait::async_trait;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use std::sync::Arc;

use crate::engine::{
    resolve_gemini_api_key, start_clip_analysis, start_clip_render, AnalyzeClipsRequest,
    AnalyzeClipsResponse, RenderClipRequest, RenderClipResponse, SETTING_DEFAULT_PROVIDER,
    SETTING_GEMINI_API_KEY, SETTING_GEMINI_DIRECTOR_MODEL, SETTING_GEMINI_SCAN_MODEL,
    SETTING_LOCAL_VISION_MODEL,
};
use crate::models::{
    delete_vision_model, download_vision_model, list_vision_models, recommended_vision_model_id,
    VisionModelStatus, VisionModelUninstallResult,
};

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct VisionModelsListOutput {
    pub models: Vec<VisionModelStatus>,
    pub recommended_model_id: String,
    pub total_installed_bytes: u64,
    pub installed_count: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct VisionModelIdInput {
    pub model_id: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct ClipsConfigOutput {
    pub gemini_configured: bool,
    pub gemini_key_preview: Option<String>,
    pub default_provider: String,
    pub gemini_scan_model: String,
    pub gemini_director_model: String,
    pub local_vision_model: String,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, JsonSchema)]
pub struct ClipsConfigUpdateInput {
    #[serde(default)]
    pub gemini_api_key: Option<String>,
    #[serde(default)]
    pub default_provider: Option<String>,
    #[serde(default)]
    pub gemini_scan_model: Option<String>,
    #[serde(default)]
    pub gemini_director_model: Option<String>,
    #[serde(default)]
    pub local_vision_model: Option<String>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, JsonSchema)]
pub struct ClipRunsListInput {
    #[serde(default)]
    pub media_id: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct ClipRunsListOutput {
    pub runs: Vec<SmartClipRun>,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct ClipRunIdInput {
    pub id: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct ClipRunDeleteOutput {
    pub id: String,
    pub deleted: bool,
}

fn read_clips_config(kernel: &AicutKernel) -> ClipsConfigOutput {
    let key = resolve_gemini_api_key(kernel, None);
    let preview = key.as_deref().map(|k| {
        if k.len() > 8 {
            format!("{}...{}", &k[..4], &k[k.len() - 4..])
        } else {
            "****".to_string()
        }
    });
    let get_str = |k: &str, default: &str| -> String {
        kernel
            .repo
            .get_setting(k)
            .ok()
            .flatten()
            .and_then(|v| v.as_str().map(|s| s.to_string()))
            .filter(|s| !s.trim().is_empty())
            .unwrap_or_else(|| default.to_string())
    };

    ClipsConfigOutput {
        gemini_configured: key.is_some(),
        gemini_key_preview: preview,
        default_provider: get_str(SETTING_DEFAULT_PROVIDER, "gemini"),
        gemini_scan_model: get_str(SETTING_GEMINI_SCAN_MODEL, "gemini-2.5-flash"),
        gemini_director_model: get_str(SETTING_GEMINI_DIRECTOR_MODEL, "gemini-2.5-pro"),
        local_vision_model: get_str(
            SETTING_LOCAL_VISION_MODEL,
            recommended_vision_model_id(&kernel.hardware),
        ),
    }
}

pub struct SmartClipsModule;

impl SmartClipsModule {
    pub fn new() -> Arc<Self> {
        Arc::new(Self)
    }
}

#[async_trait]
impl AicutModule for SmartClipsModule {
    fn manifest(&self) -> ModuleManifest {
        ModuleManifest {
            id: "smart-clips".to_string(),
            name: "Cortes Shorts IA (Visual + Narrativa)".to_string(),
            version: env!("CARGO_PKG_VERSION").to_string(),
            description: "Assiste vídeos e lives usando visão multimodal (Gemini 2.5 Flash/Pro na nuvem ou Qwen2.5-VL 100% local grátis via llama.cpp), segue sua descrição editorial e gera cortes Shorts coerentes (início, meio e fim) de até 3 minutos em 9:16 com legendas karaokê.".to_string(),
            category: "ai".to_string(),
            core: false,
            capabilities: vec![
                "multimodal-video-analysis".into(),
                "gemini-video-vision".into(),
                "local-vlm-qwen25vl".into(),
                "shorts-auto-reframe".into(),
                "narrative-coherence-3min".into(),
            ],
        }
    }

    fn declared_tools(&self) -> Vec<ToolDefinition> {
        vec![ToolDefinition {
            id: "llama-server".to_string(),
            display_name: "llama.cpp Server (IA Visual Local)".to_string(),
            description: "Runtime nativo multimodal (CUDA NVIDIA + cudart / Vulkan AMD-Intel / Metal / CPU) para rodar modelos visuais GGUF como Qwen2.5-VL 100% offline dentro do Aicut.".to_string(),
            owner_module: "smart-clips".to_string(),
            category: "ai".to_string(),
            binary_name: "llama-server".to_string(),
            version_args: vec!["--version".to_string()],
            strategy: ToolInstallStrategy::LlamaCppOfficialRelease,
        }]
    }

    async fn health(&self, kernel: &Arc<AicutKernel>) -> ModuleHealth {
        let has_ffmpeg = kernel.tools.resolve_tool("ffmpeg").is_ok();
        let has_gemini = resolve_gemini_api_key(kernel, None).is_some();
        let has_llama = kernel.tools.resolve_tool("llama-server").is_ok();
        let has_local_model = list_vision_models(kernel).iter().any(|m| m.installed);

        if !has_ffmpeg {
            return ModuleHealth::degraded("FFmpeg não instalado — instale em Configurações > Ferramentas");
        }
        if has_gemini && has_llama && has_local_model {
            ModuleHealth::healthy("Gemini API + IA Visual Local (Qwen2.5-VL) prontos")
        } else if has_gemini {
            ModuleHealth::healthy("Gemini API pronta (modo Local Free opcional disponível)")
        } else if has_llama && has_local_model {
            ModuleHealth::healthy("IA Visual Local Free (llama.cpp + Qwen2.5-VL) pronta")
        } else {
            ModuleHealth::degraded(
                "Configure sua Gemini API Key ou instale llama.cpp + um modelo Qwen2.5-VL na aba Cortes IA",
            )
        }
    }

    fn actions(&self) -> Vec<Arc<dyn ActionHandler>> {
        vec![
            Arc::new(TypedAction::<EmptyInput, VisionModelsListOutput, _>::new(
                "clips.models_list",
                "smart-clips",
                "Listar Modelos Visuais Locais (Compatíveis com GPU)",
                "List curated local multimodal GGUF vision models (Qwen2.5-VL 7B/3B) evaluated against the user's GPU VRAM and system RAM, with quality scores and compatibility reasons.",
                ActionAnnotations { read_only: true, idempotent: true, ..Default::default() },
                |ctx, _| {
                    Box::pin(async move {
                        let models = list_vision_models(&ctx.kernel);
                        let installed_count = models.iter().filter(|m| m.installed).count();
                        let total_installed_bytes: u64 =
                            models.iter().filter_map(|m| m.actual_size_bytes).sum();
                        let recommended_model_id =
                            recommended_vision_model_id(&ctx.kernel.hardware).to_string();
                        let summary = format!(
                            "{installed_count}/{} local vision model(s) installed (recommended: {recommended_model_id})",
                            models.len()
                        );
                        Ok((
                            VisionModelsListOutput {
                                models,
                                recommended_model_id,
                                total_installed_bytes,
                                installed_count,
                            },
                            summary,
                            vec![],
                        ))
                    })
                },
            )),
            Arc::new(TypedAction::<VisionModelIdInput, VisionModelStatus, _>::new(
                "clips.model_download",
                "smart-clips",
                "Baixar Modelo Visual Local (GGUF + mmproj)",
                "Download a curated Qwen2.5-VL GGUF model and its multimodal vision projector (mmproj) from Hugging Face for 100% free offline video analysis.",
                ActionAnnotations { long_running: true, ..Default::default() },
                |ctx, input| {
                    Box::pin(async move {
                        let status =
                            download_vision_model(ctx.kernel.clone(), &input.model_id, ctx.actor)
                                .await?;
                        let summary = format!("Downloaded local vision model '{}'", status.display_name);
                        Ok((status, summary, vec![]))
                    })
                },
            )),
            Arc::new(TypedAction::<VisionModelIdInput, VisionModelUninstallResult, _>::new(
                "clips.model_delete",
                "smart-clips",
                "Desinstalar Modelo Visual Local",
                "Completely remove a downloaded local vision model (GGUF + mmproj) from disk and report freed space.",
                ActionAnnotations { destructive: true, ..Default::default() },
                |ctx, input| {
                    Box::pin(async move {
                        let res = delete_vision_model(&ctx.kernel, &input.model_id)?;
                        let mb = res.freed_bytes as f64 / (1024.0 * 1024.0);
                        let summary = if res.deleted {
                            format!("Removed vision model '{}' ({mb:.1} MB freed)", res.model_id)
                        } else {
                            format!("Vision model '{}' was not installed", res.model_id)
                        };
                        Ok((res, summary, vec![]))
                    })
                },
            )),
            Arc::new(TypedAction::<EmptyInput, ClipsConfigOutput, _>::new(
                "clips.config_get",
                "smart-clips",
                "Ler Configuração de Cortes IA",
                "Read saved Smart Clips provider settings (whether Gemini API key is configured, default provider, and active models).",
                ActionAnnotations { read_only: true, idempotent: true, ..Default::default() },
                |ctx, _| {
                    Box::pin(async move {
                        let cfg = read_clips_config(&ctx.kernel);
                        Ok((cfg, "Loaded Smart Clips configuration".into(), vec![]))
                    })
                },
            )),
            Arc::new(TypedAction::<ClipsConfigUpdateInput, ClipsConfigOutput, _>::new(
                "clips.config_set",
                "smart-clips",
                "Salvar Configuração de Cortes IA",
                "Save Gemini API key, default provider ('gemini' or 'local_free'), and preferred scan/director/local vision models.",
                ActionAnnotations { idempotent: true, ..Default::default() },
                |ctx, input| {
                    Box::pin(async move {
                        if let Some(ref k) = input.gemini_api_key {
                            ctx.kernel
                                .repo
                                .set_setting(SETTING_GEMINI_API_KEY, &serde_json::json!(k.trim()))?;
                        }
                        if let Some(ref p) = input.default_provider {
                            ctx.kernel
                                .repo
                                .set_setting(SETTING_DEFAULT_PROVIDER, &serde_json::json!(p.trim()))?;
                        }
                        if let Some(ref m) = input.gemini_scan_model {
                            ctx.kernel
                                .repo
                                .set_setting(SETTING_GEMINI_SCAN_MODEL, &serde_json::json!(m.trim()))?;
                        }
                        if let Some(ref m) = input.gemini_director_model {
                            ctx.kernel.repo.set_setting(
                                SETTING_GEMINI_DIRECTOR_MODEL,
                                &serde_json::json!(m.trim()),
                            )?;
                        }
                        if let Some(ref m) = input.local_vision_model {
                            ctx.kernel.repo.set_setting(
                                SETTING_LOCAL_VISION_MODEL,
                                &serde_json::json!(m.trim()),
                            )?;
                        }
                        let cfg = read_clips_config(&ctx.kernel);
                        Ok((cfg, "Saved Smart Clips configuration".into(), vec![]))
                    })
                },
            )),
            Arc::new(TypedAction::<AnalyzeClipsRequest, AnalyzeClipsResponse, _>::new(
                "clips.analyze",
                "smart-clips",
                "Analisar Vídeo/Live e Gerar Planos de Cortes Shorts (até 3 min)",
                "Visually analyze a video or livestream using Gemini (native video+audio) or local free Qwen2.5-VL + user editorial description to find coherent Beginning-Middle-End Shorts clips up to 3 minutes.",
                ActionAnnotations { long_running: true, ..Default::default() },
                |ctx, req| {
                    Box::pin(async move {
                        let res = start_clip_analysis(ctx.kernel.clone(), req, ctx.actor).await?;
                        let summary = if let Some(ref run) = res.run {
                            format!(
                                "Found {} coherent clip(s) for '{}' via {}",
                                run.clips.len(),
                                run.media_title,
                                run.provider
                            )
                        } else {
                            format!("Started visual Smart Clips analysis job {}", res.job_id)
                        };
                        Ok((
                            res,
                            summary,
                            vec!["Render any suggested Short into 9:16 with clips.render".into()],
                        ))
                    })
                },
            )),
            Arc::new(TypedAction::<ClipRunsListInput, ClipRunsListOutput, _>::new(
                "clips.runs_list",
                "smart-clips",
                "Listar Análises de Cortes IA",
                "List saved Smart Clips analysis runs (optionally filtered by source media_id).",
                ActionAnnotations { read_only: true, idempotent: true, ..Default::default() },
                |ctx, input| {
                    Box::pin(async move {
                        let runs = ctx.kernel.repo.list_clip_runs(input.media_id.as_deref())?;
                        let summary = format!("Found {} Smart Clips run(s)", runs.len());
                        Ok((ClipRunsListOutput { runs }, summary, vec![]))
                    })
                },
            )),
            Arc::new(TypedAction::<ClipRunIdInput, SmartClipRun, _>::new(
                "clips.run_get",
                "smart-clips",
                "Detalhes da Análise de Cortes",
                "Get full details and all coherent clip suggestions for a specific Smart Clips run ID.",
                ActionAnnotations { read_only: true, idempotent: true, ..Default::default() },
                |ctx, input| {
                    Box::pin(async move {
                        let run = ctx
                            .kernel
                            .repo
                            .get_clip_run(&input.id)?
                            .ok_or_else(|| {
                                AicutError::not_found(format!("Clip run '{}' not found", input.id))
                            })?;
                        let summary = format!(
                            "Loaded {} clip suggestion(s) for '{}'",
                            run.clips.len(),
                            run.media_title
                        );
                        Ok((run, summary, vec![]))
                    })
                },
            )),
            Arc::new(TypedAction::<ClipRunIdInput, ClipRunDeleteOutput, _>::new(
                "clips.run_delete",
                "smart-clips",
                "Excluir Análise de Cortes",
                "Delete a saved Smart Clips analysis run by ID (already rendered videos in Library remain untouched).",
                ActionAnnotations { destructive: true, ..Default::default() },
                |ctx, input| {
                    Box::pin(async move {
                        let deleted = ctx.kernel.repo.delete_clip_run(&input.id)?;
                        if deleted {
                            ctx.kernel
                                .events
                                .publish(AicutEvent::SmartClipRunDeleted { id: input.id.clone() });
                        }
                        let summary = format!("Deleted Smart Clips run '{}': {deleted}", input.id);
                        Ok((
                            ClipRunDeleteOutput {
                                id: input.id,
                                deleted,
                            },
                            summary,
                            vec![],
                        ))
                    })
                },
            )),
            Arc::new(TypedAction::<RenderClipRequest, RenderClipResponse, _>::new(
                "clips.render",
                "smart-clips",
                "Renderizar Cortes Shorts 9:16 para a Biblioteca",
                "Render one or all suggested clips from a Smart Clips run into hardware-accelerated MP4 Shorts (9:16 fit_blur/center_crop/gaming_split or 16:9 original) with optional burned-in karaoke subtitles, saving them straight to the Library.",
                ActionAnnotations { long_running: true, ..Default::default() },
                |ctx, req| {
                    Box::pin(async move {
                        let res = start_clip_render(ctx.kernel.clone(), req, ctx.actor).await?;
                        let summary = if !res.rendered_assets.is_empty() {
                            format!(
                                "Rendered {} Short(s) directly to Library",
                                res.rendered_assets.len()
                            )
                        } else {
                            format!("Started Shorts render job {}", res.job_id)
                        };
                        Ok((
                            res,
                            summary,
                            vec!["View rendered Shorts in Library with ui.navigate {\"route\":\"library\"}".into()],
                        ))
                    })
                },
            )),
        ]
    }
}
