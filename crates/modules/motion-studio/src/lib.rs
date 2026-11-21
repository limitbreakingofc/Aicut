pub mod director;
pub mod hyperframes_render;
pub mod renderer;
pub mod schema;

use aicut_core::{
    ActionAnnotations, ActionHandler, AicutError, AicutKernel, AicutModule, JobProgress,
    MediaAsset, ModuleHealth, ModuleManifest, TypedAction,
};
use async_trait::async_trait;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use std::sync::Arc;

use crate::director::{edit_composition_via_chat, plan_composition_with_ai};
use crate::hyperframes_render::build_live_hyperframes_document;
use crate::renderer::render_composition_to_library;
use crate::schema::{
    compile_to_remotion_props, delete_composition, get_composition, load_all_compositions,
    save_composition, AicutComposition,
};

#[derive(Debug, Clone, Default, Serialize, Deserialize, JsonSchema)]
pub struct MotionListInput {
    #[serde(default)]
    pub media_id: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct MotionListOutput {
    pub compositions: Vec<AicutComposition>,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct MotionIdInput {
    pub id: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct MotionDeleteOutput {
    pub id: String,
    pub deleted: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct MotionPlanInput {
    pub media_id: String,
    #[serde(default)]
    pub clip_start_sec: Option<f64>,
    #[serde(default)]
    pub clip_end_sec: Option<f64>,
    #[serde(default)]
    pub title: Option<String>,
    #[serde(default)]
    pub goal: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct MotionPlanOutput {
    pub composition: AicutComposition,
    pub assistant_reply: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct MotionSaveInput {
    pub composition: AicutComposition,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct MotionChatEditInput {
    pub id: String,
    pub message: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct MotionExportCodeInput {
    pub id: String,
    /// "hyperframes" | "remotion" | "json"
    #[serde(default = "default_export_target")]
    pub target: String,
}

fn default_export_target() -> String {
    "hyperframes".to_string()
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct MotionExportCodeOutput {
    pub id: String,
    pub target: String,
    pub language: String,
    pub code: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct MotionRenderInput {
    pub id: String,
    #[serde(default)]
    pub wait: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct MotionRenderOutput {
    pub job_id: String,
    pub status: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub asset: Option<MediaAsset>,
}

pub struct MotionStudioModule;

impl MotionStudioModule {
    pub fn new() -> Arc<Self> {
        Arc::new(Self)
    }
}

#[async_trait]
impl AicutModule for MotionStudioModule {
    fn manifest(&self) -> ModuleManifest {
        ModuleManifest {
            id: "motion-studio".to_string(),
            name: "Motion Studio (Layout Dinâmico em Código IA)".to_string(),
            version: env!("CARGO_PKG_VERSION").to_string(),
            description: "Sistema de composição em código motion controlável por IA via chat e MCP: alterna layouts cena a cena (SMOKING33, Tela Cheia + Webcam PiP, Foco Câmera, Zoom Callout), exporta para HyperFrames (HTML/CSS/GSAP) e Remotion (React/TSX Props) e renderiza em GPU NVENC.".to_string(),
            category: "ai".to_string(),
            core: false,
            capabilities: vec![
                "dynamic-multi-scene-layout".into(),
                "ai-chat-motion-director".into(),
                "hyperframes-html-export".into(),
                "remotion-props-export".into(),
                "nvenc-hybrid-render".into(),
            ],
        }
    }

    fn declared_tools(&self) -> Vec<aicut_core::ToolDefinition> {
        vec![aicut_core::ToolDefinition {
            id: "chromium".to_string(),
            display_name: "Chrome Headless Shell (Motion Studio)".to_string(),
            description: "Motor Chromium headless oficial do Google (portátil) para renderizar animações HTML/CSS/SVG do Motion Studio com canal alfa RGBA.".to_string(),
            owner_module: "motion-studio".to_string(),
            category: "multimedia".to_string(),
            binary_name: "chrome-headless-shell".to_string(),
            version_args: vec!["--version".to_string()],
            strategy: aicut_core::ToolInstallStrategy::ChromeHeadlessShellOfficial,
        }]
    }

    async fn health(&self, kernel: &Arc<AicutKernel>) -> ModuleHealth {
        if kernel.tools.resolve_tool("ffmpeg").is_ok() {
            ModuleHealth::healthy("Motion Studio pronto (Cenas Dinâmicas + Chat IA + NVENC)")
        } else {
            ModuleHealth::degraded("FFmpeg não instalado — instale em Configurações > Ferramentas")
        }
    }

    fn actions(&self) -> Vec<Arc<dyn ActionHandler>> {
        vec![
            Arc::new(TypedAction::<MotionListInput, MotionListOutput, _>::new(
                "motion.list",
                "motion-studio",
                "Listar Composições Motion",
                "List saved dynamic multi-scene AicutComposition projects (optionally filtered by media_id).",
                ActionAnnotations { read_only: true, idempotent: true, ..Default::default() },
                |ctx, input| {
                    Box::pin(async move {
                        let mut compositions = load_all_compositions(&ctx.kernel);
                        if let Some(ref mid) = input.media_id {
                            compositions.retain(|c| c.media_id == *mid);
                        }
                        let summary = format!("Found {} Motion composition(s)", compositions.len());
                        Ok((MotionListOutput { compositions }, summary, vec![]))
                    })
                },
            )),
            Arc::new(TypedAction::<MotionIdInput, AicutComposition, _>::new(
                "motion.get",
                "motion-studio",
                "Carregar Composição Motion",
                "Fetch a full AicutComposition scene graph by ID.",
                ActionAnnotations { read_only: true, idempotent: true, ..Default::default() },
                |ctx, input| {
                    Box::pin(async move {
                        let comp = get_composition(&ctx.kernel, &input.id).ok_or_else(|| {
                            AicutError::not_found(format!("Motion composition '{}' not found", input.id))
                        })?;
                        let summary = format!("Loaded '{}' ({} scenes)", comp.title, comp.scenes.len());
                        Ok((comp, summary, vec![]))
                    })
                },
            )),
            Arc::new(TypedAction::<MotionPlanInput, MotionPlanOutput, _>::new(
                "motion.plan",
                "motion-studio",
                "Gerar Roteiro Dinâmico de Layout com IA",
                "Create a dynamic multi-scene AicutComposition where AI switches layouts scene by scene (SMOKING33, fullscreen content + webcam PiP, camera focus, callout zoom) using saved Framing Studio areas.",
                ActionAnnotations::default(),
                |ctx, input| {
                    Box::pin(async move {
                        let (composition, assistant_reply) = plan_composition_with_ai(
                            &ctx.kernel,
                            &input.media_id,
                            input.clip_start_sec,
                            input.clip_end_sec,
                            input.title.as_deref(),
                            input.goal.as_deref(),
                        )
                        .await?;
                        let summary = format!(
                            "Planned '{}' with {} dynamic scene(s)",
                            composition.title,
                            composition.scenes.len()
                        );
                        Ok((
                            MotionPlanOutput {
                                composition,
                                assistant_reply,
                            },
                            summary,
                            vec!["Refine any scene via chat with motion.chat_edit or render with motion.render".into()],
                        ))
                    })
                },
            )),
            Arc::new(TypedAction::<MotionSaveInput, AicutComposition, _>::new(
                "motion.save",
                "motion-studio",
                "Salvar Composição Motion",
                "Normalize and persist edits to an AicutComposition scene graph.",
                ActionAnnotations { idempotent: true, ..Default::default() },
                |ctx, input| {
                    Box::pin(async move {
                        let mut comp = input.composition;
                        comp.normalize();
                        save_composition(&ctx.kernel, &comp)?;
                        let summary = format!("Saved '{}' ({} scenes)", comp.title, comp.scenes.len());
                        Ok((comp, summary, vec![]))
                    })
                },
            )),
            Arc::new(TypedAction::<MotionIdInput, MotionDeleteOutput, _>::new(
                "motion.delete",
                "motion-studio",
                "Excluir Composição Motion",
                "Delete a saved Motion Studio composition by ID.",
                ActionAnnotations { destructive: true, ..Default::default() },
                |ctx, input| {
                    Box::pin(async move {
                        let deleted = delete_composition(&ctx.kernel, &input.id)?;
                        Ok((
                            MotionDeleteOutput {
                                id: input.id,
                                deleted,
                            },
                            format!("Deleted composition: {deleted}"),
                            vec![],
                        ))
                    })
                },
            )),
            Arc::new(TypedAction::<MotionChatEditInput, MotionPlanOutput, _>::new(
                "motion.chat_edit",
                "motion-studio",
                "Editar Layout Dinâmico via Chat com IA",
                "Send a natural-language instruction (e.g. 'between 0:12 and 0:20 show fullscreen content with webcam in top-right corner') to let the AI live-edit the composition scenes.",
                ActionAnnotations::default(),
                |ctx, input| {
                    Box::pin(async move {
                        let comp = get_composition(&ctx.kernel, &input.id).ok_or_else(|| {
                            AicutError::not_found(format!("Motion composition '{}' not found", input.id))
                        })?;
                        let (composition, assistant_reply) =
                            edit_composition_via_chat(&ctx.kernel, comp, &input.message).await?;
                        let summary = format!(
                            "Updated '{}' ({} scenes) — {}",
                            composition.title,
                            composition.scenes.len(),
                            assistant_reply
                        );
                        Ok((
                            MotionPlanOutput {
                                composition,
                                assistant_reply,
                            },
                            summary,
                            vec![],
                        ))
                    })
                },
            )),
            Arc::new(TypedAction::<MotionExportCodeInput, MotionExportCodeOutput, _>::new(
                "motion.export_code",
                "motion-studio",
                "Exportar Código Motion (HyperFrames HTML / Remotion Props / JSON)",
                "Compile an AicutComposition into ready-to-use HyperFrames HTML+CSS+GSAP code, Remotion inputProps JSON, or canonical Aicut Scene Graph JSON.",
                ActionAnnotations { read_only: true, idempotent: true, ..Default::default() },
                |ctx, input| {
                    Box::pin(async move {
                        let comp = get_composition(&ctx.kernel, &input.id).ok_or_else(|| {
                            AicutError::not_found(format!("Motion composition '{}' not found", input.id))
                        })?;
                        let (target, language, code) = match input.target.to_lowercase().as_str() {
                            "remotion" => (
                                "remotion".to_string(),
                                "json".to_string(),
                                serde_json::to_string_pretty(&compile_to_remotion_props(&comp))
                                    .unwrap_or_default(),
                            ),
                            "json" => (
                                "json".to_string(),
                                "json".to_string(),
                                serde_json::to_string_pretty(&comp).unwrap_or_default(),
                            ),
                            _ => (
                                "hyperframes".to_string(),
                                "html".to_string(),
                                build_live_hyperframes_document(&comp),
                            ),
                        };
                        let summary = format!("Exported '{}' as {target} motion code", comp.title);
                        Ok((
                            MotionExportCodeOutput {
                                id: comp.id,
                                target,
                                language,
                                code,
                            },
                            summary,
                            vec![],
                        ))
                    })
                },
            )),
            Arc::new(TypedAction::<MotionRenderInput, MotionRenderOutput, _>::new(
                "motion.render",
                "motion-studio",
                "Renderizar Short Dinâmico (NVENC) para a Biblioteca",
                "Render all dynamic scenes, layouts, PiP webcams, callout zooms, kinetic headlines and karaoke subtitles into a 9:16 MP4 Short on GPU (NVENC/AMF/QSV) and add it to the Library.",
                ActionAnnotations { long_running: true, ..Default::default() },
                |ctx, input| {
                    Box::pin(async move {
                        let comp = get_composition(&ctx.kernel, &input.id).ok_or_else(|| {
                            AicutError::not_found(format!("Motion composition '{}' not found", input.id))
                        })?;
                        let (job, cancel) = ctx.kernel.jobs.create_job(
                            "motion.render",
                            format!("Renderizar Motion Short '{}'", comp.title),
                            "motion-studio",
                            ctx.actor,
                        );
                        let job_id = job.id.clone();
                        let kernel = ctx.kernel.clone();
                        let task_job_id = job_id.clone();
                        let mut comp_for_task = comp.clone();

                        let handle = tokio::spawn(async move {
                            kernel.jobs.mark_running(&task_job_id, "rendering_motion");
                            kernel.jobs.update_progress(
                                &task_job_id,
                                JobProgress {
                                    percent: 20.0,
                                    stage: "rendering_motion".into(),
                                    message: Some(format!(
                                        "Renderizando {} cenas dinâmicas de '{}' em GPU...",
                                        comp_for_task.scenes.len(),
                                        comp_for_task.title
                                    )),
                                    ..Default::default()
                                },
                            );
                            if cancel.is_cancelled() {
                                kernel.jobs.mark_cancelled(&task_job_id);
                                return Err(AicutError::cancelled("Render cancelado."));
                            }
                            match render_composition_to_library(&kernel, &comp_for_task, Some(&task_job_id)).await {
                                Ok(asset) => {
                                    comp_for_task.rendered_media_id = Some(asset.id.clone());
                                    let _ = save_composition(&kernel, &comp_for_task);
                                    kernel.jobs.complete(
                                        &task_job_id,
                                        serde_json::to_value(&asset).unwrap_or(serde_json::Value::Null),
                                    );
                                    Ok(asset)
                                }
                                Err(err) => {
                                    kernel.jobs.fail(&task_job_id, err.clone());
                                    Err(err)
                                }
                            }
                        });

                        if input.wait {
                            let asset = handle
                                .await
                                .map_err(|e| AicutError::internal(format!("Render task panicked: {e}")))??;
                            let summary = format!("Rendered Motion Short '{}' to Library", asset.title);
                            Ok((
                                MotionRenderOutput {
                                    job_id,
                                    status: "completed".to_string(),
                                    asset: Some(asset),
                                },
                                summary,
                                vec![],
                            ))
                        } else {
                            Ok((
                                MotionRenderOutput {
                                    job_id: job_id.clone(),
                                    status: "queued".to_string(),
                                    asset: None,
                                },
                                format!("Started Motion Studio render job {job_id}"),
                                vec![],
                            ))
                        }
                    })
                },
            )),
        ]
    }
}
