pub mod engine;
pub mod exporters;
pub mod models;

use aicut_core::{
    ActionAnnotations, ActionHandler, AicutError, AicutEvent, AicutKernel, AicutModule, EmptyInput,
    MediaAsset, ModuleHealth, ModuleManifest, ToolDefinition, ToolInstallStrategy,
    TranscriptRecord, TypedAction,
};
use async_trait::async_trait;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use std::sync::Arc;

use crate::engine::{start_transcription, TranscribeRequest, TranscribeResponse};
use crate::exporters::{export_and_save_to_library, ExportFormat, ExportFormatInfo};
use crate::models::{delete_model, download_model, list_models, WhisperModelStatus};

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct ModelsListOutput {
    pub models: Vec<WhisperModelStatus>,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct ModelIdInput {
    pub model_id: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct ModelDeleteOutput {
    pub model_id: String,
    pub deleted: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct FormatsListOutput {
    pub formats: Vec<ExportFormatInfo>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, JsonSchema)]
pub struct TranscriptsListInput {
    #[serde(default)]
    pub media_id: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct TranscriptsListOutput {
    pub transcripts: Vec<TranscriptRecord>,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct TranscriptIdInput {
    pub id: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct TranscriptDeleteOutput {
    pub id: String,
    pub deleted: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct ExportTranscriptInput {
    pub transcript_id: String,
    pub formats: Vec<ExportFormat>,
    #[serde(default)]
    pub fps: Option<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct ExportTranscriptOutput {
    pub transcript_id: String,
    pub exported_assets: Vec<MediaAsset>,
}

pub struct TranscribeModule;

impl TranscribeModule {
    pub fn new() -> Arc<Self> {
        Arc::new(Self)
    }
}

#[async_trait]
impl AicutModule for TranscribeModule {
    fn manifest(&self) -> ModuleManifest {
        ModuleManifest {
            id: "transcribe".to_string(),
            name: "Transcrição Local com IA (Whisper)".to_string(),
            version: env!("CARGO_PKG_VERSION").to_string(),
            description: "Transcreve vídeos e áudios 100% offline com whisper.cpp + Whisper Large v3 Turbo e exporta qualquer formato de legenda ou timeline (SRT, VTT, ASS, FCPXML, Premiere XML, EDL, OTIO, CSV, JSON) direto para a Biblioteca.".to_string(),
            category: "ai".to_string(),
            core: false,
            capabilities: vec![
                "local-speech-to-text".into(),
                "whisper-cpp".into(),
                "word-timestamps".into(),
                "subtitle-export".into(),
                "timeline-export".into(),
            ],
        }
    }

    fn declared_tools(&self) -> Vec<ToolDefinition> {
        vec![ToolDefinition {
            id: "whisper".to_string(),
            display_name: "whisper.cpp (IA Local)".to_string(),
            description: "Motor nativo de transcrição offline (CUDA NVIDIA / OpenBLAS / ARM64) para rodar Whisper Large v3 Turbo e gerar legendas e timelines.".to_string(),
            owner_module: "transcribe".to_string(),
            category: "ai".to_string(),
            binary_name: "whisper-cli".to_string(),
            version_args: vec!["--help".to_string()],
            strategy: ToolInstallStrategy::WhisperCppOfficialRelease,
        }]
    }

    async fn health(&self, kernel: &Arc<AicutKernel>) -> ModuleHealth {
        let has_whisper = kernel.tools.resolve_tool("whisper").is_ok();
        let has_model = list_models(kernel).iter().any(|m| m.installed);
        match (has_whisper, has_model) {
            (true, true) => ModuleHealth::healthy("whisper.cpp e modelo prontos"),
            (true, false) => ModuleHealth::degraded("whisper.cpp pronto — baixe um modelo na aba Transcrever IA"),
            (false, _) => ModuleHealth::degraded("whisper.cpp não instalado — instale em Configurações > Ferramentas"),
        }
    }

    fn actions(&self) -> Vec<Arc<dyn ActionHandler>> {
        vec![
            Arc::new(TypedAction::<EmptyInput, ModelsListOutput, _>::new(
                "transcribe.models_list",
                "transcribe",
                "Listar Modelos Whisper",
                "List available local Whisper GGML models (e.g. large-v3-turbo-q5_0, large-v3-turbo, small, medium, large-v3), installed status, and hardware recommendation.",
                ActionAnnotations { read_only: true, idempotent: true, ..Default::default() },
                |ctx, _| {
                    Box::pin(async move {
                        let models = list_models(&ctx.kernel);
                        let installed = models.iter().filter(|m| m.installed).count();
                        let summary = format!("{installed}/{} Whisper model(s) installed", models.len());
                        Ok((ModelsListOutput { models }, summary, vec![]))
                    })
                },
            )),
            Arc::new(TypedAction::<ModelIdInput, WhisperModelStatus, _>::new(
                "transcribe.model_download",
                "transcribe",
                "Baixar Modelo Whisper",
                "Download a Whisper GGML model (recommended: 'large-v3-turbo-q5_0' or 'large-v3-turbo') locally for offline transcription.",
                ActionAnnotations { long_running: true, ..Default::default() },
                |ctx, input| {
                    Box::pin(async move {
                        let status = download_model(ctx.kernel.clone(), &input.model_id, ctx.actor).await?;
                        let summary = format!("Downloaded Whisper model '{}'", status.display_name);
                        Ok((status, summary, vec![]))
                    })
                },
            )),
            Arc::new(TypedAction::<ModelIdInput, ModelDeleteOutput, _>::new(
                "transcribe.model_delete",
                "transcribe",
                "Remover Modelo Whisper",
                "Delete a downloaded local Whisper model file to free disk space.",
                ActionAnnotations { destructive: true, ..Default::default() },
                |ctx, input| {
                    Box::pin(async move {
                        let deleted = delete_model(&ctx.kernel, &input.model_id)?;
                        let summary = format!("Deleted model '{}': {deleted}", input.model_id);
                        Ok((ModelDeleteOutput { model_id: input.model_id, deleted }, summary, vec![]))
                    })
                },
            )),
            Arc::new(TypedAction::<EmptyInput, FormatsListOutput, _>::new(
                "transcribe.formats_list",
                "transcribe",
                "Listar Formatos de Legenda & Timeline",
                "List all 13 supported subtitle, transcript, and pro video editor timeline export formats (SRT, VTT, ASS, ASS Karaoke, LRC, FCPXML, Premiere XML, EDL, OTIO, CSV Markers, TXT, JSON).",
                ActionAnnotations { read_only: true, idempotent: true, ..Default::default() },
                |_ctx, _| {
                    Box::pin(async move {
                        let formats = ExportFormat::all_info();
                        let summary = format!("{} export format(s) available", formats.len());
                        Ok((FormatsListOutput { formats }, summary, vec![]))
                    })
                },
            )),
            Arc::new(TypedAction::<TranscribeRequest, TranscribeResponse, _>::new(
                "transcribe.run",
                "transcribe",
                "Transcrever Vídeo/Áudio com IA Local",
                "Transcribe any video or audio asset from the Aicut Library offline using whisper.cpp + Whisper Large v3 Turbo, and automatically save chosen subtitle/timeline formats (SRT, VTT, ASS, FCPXML, Premiere XML, EDL, OTIO, CSV, JSON) into the Library.",
                ActionAnnotations { long_running: true, ..Default::default() },
                |ctx, req| {
                    Box::pin(async move {
                        let res = start_transcription(ctx.kernel.clone(), req, ctx.actor).await?;
                        let summary = if let Some(ref tr) = res.transcript {
                            format!(
                                "Transcribed '{}' ({} segments, {} file(s) saved to Library)",
                                tr.media_title,
                                tr.segments.len(),
                                res.exported_assets.len()
                            )
                        } else {
                            format!("Started local AI transcription job {}", res.job_id)
                        };
                        Ok((
                            res,
                            summary,
                            vec!["Export additional formats anytime via transcribe.export".into()],
                        ))
                    })
                },
            )),
            Arc::new(TypedAction::<TranscriptsListInput, TranscriptsListOutput, _>::new(
                "transcribe.list",
                "transcribe",
                "Listar Transcrições",
                "List saved AI transcriptions (optionally filtered by source media_id).",
                ActionAnnotations { read_only: true, idempotent: true, ..Default::default() },
                |ctx, input| {
                    Box::pin(async move {
                        let transcripts = ctx.kernel.repo.list_transcripts(input.media_id.as_deref())?;
                        let summary = format!("Found {} transcript(s)", transcripts.len());
                        Ok((TranscriptsListOutput { transcripts }, summary, vec![]))
                    })
                },
            )),
            Arc::new(TypedAction::<TranscriptIdInput, TranscriptRecord, _>::new(
                "transcribe.get",
                "transcribe",
                "Ler Transcrição Completa",
                "Get the complete transcript text, segments, and word-level millisecond timestamps by transcript ID.",
                ActionAnnotations { read_only: true, idempotent: true, ..Default::default() },
                |ctx, input| {
                    Box::pin(async move {
                        let rec = ctx
                            .kernel
                            .repo
                            .get_transcript(&input.id)?
                            .ok_or_else(|| AicutError::not_found(format!("Transcript '{}' not found", input.id)))?;
                        let summary = format!("Loaded transcript for '{}' ({} segments)", rec.media_title, rec.segments.len());
                        Ok((rec, summary, vec![]))
                    })
                },
            )),
            Arc::new(TypedAction::<ExportTranscriptInput, ExportTranscriptOutput, _>::new(
                "transcribe.export",
                "transcribe",
                "Exportar Legenda ou Timeline para a Biblioteca",
                "Convert an existing saved transcription into one or more subtitle or timeline formats (srt, vtt, ass, ass_karaoke, lrc, txt, txt_timestamps, json, fcpxml, premiere_xml, edl, otio, csv_markers) and save them directly to the Aicut Library.",
                ActionAnnotations { idempotent: true, ..Default::default() },
                |ctx, input| {
                    Box::pin(async move {
                        if input.formats.is_empty() {
                            return Err(AicutError::invalid_input("Provide at least one export format."));
                        }
                        let rec = ctx
                            .kernel
                            .repo
                            .get_transcript(&input.transcript_id)?
                            .ok_or_else(|| {
                                AicutError::not_found(format!("Transcript '{}' not found", input.transcript_id))
                            })?;
                        let source = ctx.kernel.repo.get_media(&rec.media_id)?;
                        let mut exported_assets = Vec::new();
                        for fmt in input.formats {
                            let asset = export_and_save_to_library(
                                &ctx.kernel,
                                &rec,
                                source.as_ref(),
                                fmt,
                                input.fps,
                            )?;
                            exported_assets.push(asset);
                        }
                        let summary = format!(
                            "Exported {} file(s) from '{}' into the Library",
                            exported_assets.len(),
                            rec.media_title
                        );
                        Ok((
                            ExportTranscriptOutput {
                                transcript_id: input.transcript_id,
                                exported_assets,
                            },
                            summary,
                            vec!["View generated files in Library with ui.navigate {\"route\":\"library\"}".into()],
                        ))
                    })
                },
            )),
            Arc::new(TypedAction::<TranscriptIdInput, TranscriptDeleteOutput, _>::new(
                "transcribe.delete",
                "transcribe",
                "Excluir Transcrição",
                "Delete a saved transcription record by ID (generated files already in Library remain untouched).",
                ActionAnnotations { destructive: true, ..Default::default() },
                |ctx, input| {
                    Box::pin(async move {
                        let deleted = ctx.kernel.repo.delete_transcript(&input.id)?;
                        if deleted {
                            ctx.kernel
                                .events
                                .publish(AicutEvent::TranscriptDeleted { id: input.id.clone() });
                        }
                        let summary = format!("Deleted transcript '{}': {deleted}", input.id);
                        Ok((TranscriptDeleteOutput { id: input.id, deleted }, summary, vec![]))
                    })
                },
            )),
        ]
    }
}
