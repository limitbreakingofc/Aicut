use aicut_core::{
    ActorKind, AicutError, AicutEvent, AicutKernel, AicutResult, JobProgress, MediaAsset,
    ResolvedTranscript, SmartClipRun, TranscriptRecord,
};
use chrono::Utc;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use std::sync::Arc;
use tokio_util::sync::CancellationToken;
use uuid::Uuid;

use crate::framing::{resolve_effective_framing, ContentRegionMode, NormalizedRect};
use crate::frames::probe_duration_sec;
use crate::gemini::{analyze_with_gemini, GeminiConfig};
use crate::local_vlm::analyze_with_local_vlm;
use crate::models::resolve_any_installed_vision_model;
use crate::postprocess::validate_and_finalize_clips;
use crate::renderer::{render_clip_to_library, ContentFit, ShortsLayout, WebcamCorner};

pub const SETTING_GEMINI_API_KEY: &str = "smart_clips_gemini_api_key";
pub const SETTING_DEFAULT_PROVIDER: &str = "smart_clips_default_provider";
pub const SETTING_GEMINI_SCAN_MODEL: &str = "smart_clips_gemini_scan_model";
pub const SETTING_GEMINI_DIRECTOR_MODEL: &str = "smart_clips_gemini_director_model";
pub const SETTING_LOCAL_VISION_MODEL: &str = "smart_clips_local_vision_model";
pub const SETTING_TEMPLATE_PATH: &str = "smart_clips_template_path";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum ClipProvider {
    /// Cloud: Google Gemini (native video + audio understanding)
    Gemini,
    /// 100% Local Free: embedded llama-server + Qwen2.5-VL GGUF on user's GPU/CPU
    LocalFree,
}

impl Default for ClipProvider {
    fn default() -> Self {
        Self::Gemini
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct AnalyzeClipsRequest {
    /// ID of the video/livestream asset in the Aicut Library.
    pub media_id: String,
    /// Editorial description of what interesting moments/topics/reactions to find.
    pub description: String,
    /// AI provider: "gemini" (native video+audio) or "local_free" (offline llama-server + Qwen2.5-VL).
    #[serde(default)]
    pub provider: ClipProvider,
    /// Optional explicit model override (e.g. "qwen2.5-vl-7b-q4_k_m" for local_free, or "gemini-2.5-pro" for Gemini director).
    #[serde(default)]
    pub model_id: Option<String>,
    /// Optional Gemini API key override (falls back to saved Aicut setting or GEMINI_API_KEY env var).
    #[serde(default)]
    pub gemini_api_key: Option<String>,
    /// Optional existing transcript ID to anchor word-level timestamps (automatically picks latest transcript for this media if omitted).
    #[serde(default)]
    pub transcript_id: Option<String>,
    /// Hard maximum duration per clip in seconds (default 180s = 3 minutes for Shorts).
    #[serde(default = "default_max_duration")]
    pub max_duration_sec: f64,
    /// Minimum duration per clip in seconds (default 25s).
    #[serde(default = "default_min_duration")]
    pub min_duration_sec: f64,
    /// Maximum number of clips to return (default 5).
    #[serde(default = "default_max_clips")]
    pub max_clips: usize,
    /// Allow light internal jump-cuts to skip dead air/pauses while preserving Beginning-Middle-End.
    #[serde(default)]
    pub allow_jump_cuts: bool,
    /// Wait synchronously for analysis to complete (ideal for CLI/MCP).
    #[serde(default)]
    pub wait: bool,
}

fn default_max_duration() -> f64 {
    180.0
}

fn default_min_duration() -> f64 {
    25.0
}

fn default_max_clips() -> usize {
    5
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct AnalyzeClipsResponse {
    pub job_id: String,
    pub status: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub run: Option<SmartClipRun>,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct RenderClipRequest {
    pub run_id: String,
    /// Specific clip ID inside the run to render (if omitted, renders ALL clips in the run).
    #[serde(default)]
    pub clip_id: Option<String>,
    #[serde(default)]
    pub layout: ShortsLayout,
    #[serde(default)]
    pub webcam_corner: WebcamCorner,
    #[serde(default)]
    pub content_fit: ContentFit,
    /// Optional custom 9:16 PNG template path (uses built-in SMOKING33 template if omitted).
    #[serde(default)]
    pub template_path: Option<String>,
    /// Enable native AI face detection + zero-border webcam edge detection + smooth camera tracking.
    #[serde(default = "default_true")]
    pub face_tracking: bool,
    /// Framing zoom factor on the face (0.85 = wider, 1.0 = natural cameraman, 1.2 = closer).
    #[serde(default = "default_face_zoom")]
    pub face_zoom: f32,
    /// Inner border trim fraction to eliminate OBS borders/gameplay bleed (default 0.045 = 4.5%).
    #[serde(default = "default_border_trim")]
    pub border_trim_pct: f32,
    /// Optional explicit normalized webcam area (0..1). Overrides saved Framing Studio profile if set.
    #[serde(default)]
    pub webcam_rect: Option<NormalizedRect>,
    /// Optional explicit normalized content area (0..1). Overrides saved Framing Studio profile if set.
    #[serde(default)]
    pub content_rect: Option<NormalizedRect>,
    /// Automatically apply the precision framing saved in the Enquadramento tab (default true).
    #[serde(default = "default_true")]
    pub use_saved_framing: bool,
    #[serde(default = "default_true")]
    pub burn_subtitles: bool,
    #[serde(default)]
    pub wait: bool,
}

fn default_true() -> bool {
    true
}

fn default_face_zoom() -> f32 {
    1.0
}

fn default_border_trim() -> f32 {
    0.045
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct RenderClipResponse {
    pub job_id: String,
    pub status: String,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub rendered_assets: Vec<MediaAsset>,
}

pub fn resolve_gemini_api_key(kernel: &AicutKernel, override_key: Option<&str>) -> Option<String> {
    if let Some(k) = override_key.map(str::trim).filter(|s| !s.is_empty()) {
        return Some(k.to_string());
    }
    if let Ok(Some(val)) = kernel.repo.get_setting(SETTING_GEMINI_API_KEY) {
        if let Some(s) = val.as_str().map(str::trim).filter(|s| !s.is_empty()) {
            return Some(s.to_string());
        }
    }
    std::env::var("GEMINI_API_KEY")
        .ok()
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
}

pub async fn start_clip_analysis(
    kernel: Arc<AicutKernel>,
    req: AnalyzeClipsRequest,
    actor: ActorKind,
) -> AicutResult<AnalyzeClipsResponse> {
    if req.description.trim().is_empty() {
        return Err(AicutError::invalid_input(
            "Informe uma descricao do que a IA deve procurar nos cortes.",
        ));
    }

    let source = kernel
        .repo
        .get_media(&req.media_id)?
        .ok_or_else(|| AicutError::not_found(format!("Media '{}' not found in Library", req.media_id)))?;

    let ffmpeg_bin = kernel.tools.resolve_tool("ffmpeg")?;
    let ffprobe_bin = kernel.tools.resolve_tool("ffprobe")?;

    // Validate provider dependencies upfront so user gets immediate actionable feedback
    let resolved_provider = match req.provider {
        ClipProvider::Gemini => {
            let api_key = resolve_gemini_api_key(&kernel, req.gemini_api_key.as_deref()).ok_or_else(|| {
                AicutError::dependency_missing(
                    "gemini_api_key",
                    "Chave da Gemini API nao configurada. Salve sua GEMINI_API_KEY na aba Cortes IA ou alterne para o modo Local Free.",
                )
            })?;
            let scan_model = kernel
                .repo
                .get_setting(SETTING_GEMINI_SCAN_MODEL)
                .ok()
                .flatten()
                .and_then(|v| v.as_str().map(|s| s.to_string()))
                .filter(|s| !s.trim().is_empty() && !s.starts_with("gemini-2."))
                .unwrap_or_else(|| "gemini-3.8-flash".to_string());
            let director_model = req
                .model_id
                .clone()
                .filter(|s| !s.trim().is_empty() && !s.starts_with("gemini-2."))
                .or_else(|| {
                    kernel
                        .repo
                        .get_setting(SETTING_GEMINI_DIRECTOR_MODEL)
                        .ok()
                        .flatten()
                        .and_then(|v| v.as_str().map(|s| s.to_string()))
                        .filter(|s| !s.trim().is_empty() && !s.starts_with("gemini-2."))
                })
                .unwrap_or_else(|| "auto".to_string());
            ResolvedProvider::Gemini(GeminiConfig {
                api_key,
                scan_model,
                director_model,
            })
        }
        ClipProvider::LocalFree => {
            let llama_bin = kernel.tools.resolve_tool("llama-server")?;
            let preferred_id = req
                .model_id
                .clone()
                .filter(|s| !s.trim().is_empty())
                .or_else(|| {
                    kernel
                        .repo
                        .get_setting(SETTING_LOCAL_VISION_MODEL)
                        .ok()
                        .flatten()
                        .and_then(|v| v.as_str().map(|s| s.to_string()))
                        .filter(|s| !s.trim().is_empty())
                });
            let (model_id, model_path, mmproj_path) =
                resolve_any_installed_vision_model(&kernel, preferred_id.as_deref())?;
            ResolvedProvider::Local {
                llama_bin,
                model_id,
                model_path,
                mmproj_path,
            }
        }
    };

    // Full-library transcript lookup: explicit ID -> exact media_id -> related sibling/parent ID ->
    // external video ID ([fPQDxLwScrA]) -> Library JSON/SRT/VTT file -> sidecar file on disk!
    let resolved_transcript: Option<ResolvedTranscript> = if let Some(ref tid) = req.transcript_id {
        kernel
            .repo
            .get_transcript(tid)?
            .map(|t| ResolvedTranscript {
                transcript: t,
                source: "explicit_id".to_string(),
                source_asset_id: Some(source.id.clone()),
                source_file_path: None,
            })
    } else {
        kernel.repo.find_best_transcript_for_media(&source.id)?
    };
    let related_assets = kernel.repo.find_related_media(&source.id).unwrap_or_default();

    let provider_label = match &resolved_provider {
        ResolvedProvider::Gemini(cfg) => format!("Gemini ({})", cfg.director_model),
        ResolvedProvider::Local { model_id, .. } => format!("Local Free ({model_id})"),
    };

    let (job, cancel) = kernel.jobs.create_job(
        "clips.analyze",
        format!("Cortes IA '{}' ({provider_label})", source.title),
        "smart-clips",
        actor,
    );
    let job_id = job.id.clone();

    let wait = req.wait;
    let handle = tokio::spawn(execute_analysis_task(
        kernel.clone(),
        job_id.clone(),
        source,
        ffmpeg_bin,
        ffprobe_bin,
        resolved_provider,
        resolved_transcript,
        related_assets,
        req,
        cancel,
    ));

    if wait {
        let run = handle
            .await
            .map_err(|e| AicutError::internal(format!("Analysis task panicked: {e}")))??;
        Ok(AnalyzeClipsResponse {
            job_id,
            status: "completed".to_string(),
            run: Some(run),
        })
    } else {
        Ok(AnalyzeClipsResponse {
            job_id,
            status: "queued".to_string(),
            run: None,
        })
    }
}

enum ResolvedProvider {
    Gemini(GeminiConfig),
    Local {
        llama_bin: PathBuf,
        model_id: String,
        model_path: PathBuf,
        mmproj_path: PathBuf,
    },
}

#[allow(clippy::too_many_arguments)]
async fn execute_analysis_task(
    kernel: Arc<AicutKernel>,
    job_id: String,
    source: MediaAsset,
    ffmpeg_bin: PathBuf,
    ffprobe_bin: PathBuf,
    provider: ResolvedProvider,
    resolved_transcript: Option<ResolvedTranscript>,
    related_assets: Vec<MediaAsset>,
    req: AnalyzeClipsRequest,
    cancel: CancellationToken,
) -> AicutResult<SmartClipRun> {
    kernel.jobs.mark_running(&job_id, "probing_video");
    let tmp_dir = std::env::temp_dir().join(format!("aicut-clips-{job_id}"));
    std::fs::create_dir_all(&tmp_dir)?;

    let res: AicutResult<SmartClipRun> = async {
        let video_path = PathBuf::from(&source.file_path);
        if !video_path.is_file() {
            return Err(AicutError::not_found(format!(
                "Arquivo de video nao encontrado no disco: {}",
                source.file_path
            )));
        }

        let duration_sec = match source.metadata.duration_sec.filter(|&d| d > 1.0) {
            Some(d) => d,
            None => probe_duration_sec(&ffprobe_bin, &video_path).await?,
        };

        let max_dur = req.max_duration_sec.clamp(20.0, 180.0);
        let min_dur = req.min_duration_sec.clamp(10.0, max_dur);
        let max_clips = req.max_clips.clamp(1, 15);
        let transcript: Option<TranscriptRecord> =
            resolved_transcript.as_ref().map(|rt| rt.transcript.clone());

        let tr_note = match &resolved_transcript {
            Some(rt) => format!(
                "vinculada ({} falas, origem: {}, id: {})",
                rt.transcript.segments.len(),
                rt.source,
                rt.transcript.id
            ),
            None => "nao encontrada na Biblioteca (analise puramente visual/audio)".to_string(),
        };

        kernel.jobs.append_log(
            &job_id,
            format!(
                "Analisando '{}' [video_id: {}] ({duration_sec:.1}s) | max {max_dur:.0}s | arquivos vinculados na Biblioteca: {} | transcricao: {tr_note}",
                source.title,
                source.id,
                related_assets.len()
            ),
        );

        // Enrich editorial brief with Library lineage metadata (video ID, URL, already generated clips)
        let mut enriched_description = req.description.trim().to_string();
        let existing_shorts: Vec<String> = related_assets
            .iter()
            .filter(|a| a.id != source.id && a.tags.iter().any(|t| t == "short" || t == "ai-clip"))
            .map(|a| a.title.clone())
            .take(8)
            .collect();
        if !existing_shorts.is_empty() {
            enriched_description.push_str(&format!(
                "\n[CONTEXTO DA BIBLIOTECA — video_id: {} | Shorts ja gerados anteriormente para este video (evite repetir exatamente os mesmos momentos): {}]",
                source.id,
                existing_shorts.join("; ")
            ));
        }

        let (provider_name, model_id, raw_clips) = match provider {
            ResolvedProvider::Gemini(ref cfg) => {
                let clips = analyze_with_gemini(
                    &kernel,
                    &job_id,
                    &ffmpeg_bin,
                    &video_path,
                    duration_sec,
                    transcript.as_ref(),
                    &enriched_description,
                    max_dur,
                    min_dur,
                    max_clips,
                    req.allow_jump_cuts,
                    cfg,
                    &tmp_dir,
                    &cancel,
                )
                .await?;
                ("gemini".to_string(), cfg.director_model.clone(), clips)
            }
            ResolvedProvider::Local {
                ref llama_bin,
                ref model_id,
                ref model_path,
                ref mmproj_path,
            } => {
                let clips = analyze_with_local_vlm(
                    &kernel,
                    &job_id,
                    &ffmpeg_bin,
                    llama_bin,
                    model_path,
                    mmproj_path,
                    &video_path,
                    duration_sec,
                    transcript.as_ref(),
                    &enriched_description,
                    max_dur,
                    min_dur,
                    max_clips,
                    req.allow_jump_cuts,
                    &tmp_dir,
                    &cancel,
                )
                .await?;
                ("local_free".to_string(), model_id.clone(), clips)
            }
        };

        kernel.jobs.update_progress(
            &job_id,
            JobProgress {
                percent: 93.0,
                stage: "validating_and_snapping".into(),
                message: Some("Validando teto de 3 minutos, coerencia e alinhando cortes nas palavras...".into()),
                ..Default::default()
            },
        );

        let clips = validate_and_finalize_clips(
            raw_clips,
            transcript.as_ref(),
            duration_sec,
            min_dur,
            max_dur,
            max_clips,
            req.allow_jump_cuts,
        );

        let run = SmartClipRun {
            id: Uuid::new_v4().to_string(),
            media_id: source.id.clone(),
            media_title: source.title.clone(),
            provider: provider_name,
            model_id,
            description: req.description.trim().to_string(),
            max_duration_sec: max_dur,
            clips,
            created_at: Utc::now(),
        };

        kernel.repo.insert_clip_run(&run)?;
        kernel
            .events
            .publish(AicutEvent::SmartClipRunCreated(run.clone()));
        Ok(run)
    }
    .await;

    let _ = std::fs::remove_dir_all(&tmp_dir);

    match res {
        Ok(run) => {
            kernel.jobs.complete(
                &job_id,
                serde_json::to_value(&run).unwrap_or(serde_json::Value::Null),
            );
            Ok(run)
        }
        Err(err) => {
            if cancel.is_cancelled() {
                kernel.jobs.mark_cancelled(&job_id);
            } else {
                kernel.jobs.fail(&job_id, err.clone());
            }
            Err(err)
        }
    }
}

pub async fn start_clip_render(
    kernel: Arc<AicutKernel>,
    req: RenderClipRequest,
    actor: ActorKind,
) -> AicutResult<RenderClipResponse> {
    let run = kernel
        .repo
        .get_clip_run(&req.run_id)?
        .ok_or_else(|| AicutError::not_found(format!("Clip run '{}' not found", req.run_id)))?;

    let source = kernel
        .repo
        .get_media(&run.media_id)?
        .ok_or_else(|| AicutError::not_found(format!("Source media '{}' not found", run.media_id)))?;

    let ffmpeg_bin = kernel.tools.resolve_tool("ffmpeg")?;
    let ffprobe_bin = kernel.tools.resolve_tool("ffprobe")?;
    let transcript = kernel
        .repo
        .find_best_transcript_for_media(&source.id)?
        .map(|rt| rt.transcript);

    let target_indices: Vec<usize> = if let Some(ref cid) = req.clip_id {
        let idx = run
            .clips
            .iter()
            .position(|c| c.id == *cid)
            .ok_or_else(|| AicutError::not_found(format!("Clip '{cid}' not found in run")))?;
        vec![idx]
    } else {
        (0..run.clips.len()).collect()
    };

    if target_indices.is_empty() {
        return Err(AicutError::invalid_input("Nenhum corte disponivel para renderizar."));
    }

    let (job, cancel) = kernel.jobs.create_job(
        "clips.render",
        format!("Renderizar {} Short(s) de '{}'", target_indices.len(), source.title),
        "smart-clips",
        actor,
    );
    let job_id = job.id.clone();

    let effective_template_path = req
        .template_path
        .clone()
        .filter(|s| !s.trim().is_empty())
        .or_else(|| {
            kernel
                .repo
                .get_setting(SETTING_TEMPLATE_PATH)
                .ok()
                .flatten()
                .and_then(|v| v.as_str().map(|s| s.to_string()))
                .filter(|s| !s.trim().is_empty())
        });

    let saved_framing = if req.use_saved_framing {
        Some(resolve_effective_framing(&kernel, Some(&source.id)).0)
    } else {
        None
    };
    let effective_webcam_rect = req
        .webcam_rect
        .or_else(|| saved_framing.as_ref().and_then(|f| f.webcam_rect));
    let effective_content_rect = req
        .content_rect
        .or_else(|| saved_framing.as_ref().and_then(|f| f.content_rect));
    let effective_webcam_lock_static = saved_framing
        .as_ref()
        .map(|f| f.webcam_lock_static)
        .unwrap_or(false);
    let effective_content_mode = saved_framing
        .as_ref()
        .map(|f| f.content_mode)
        .unwrap_or(ContentRegionMode::AiDirectorInRegion);

    let wait = req.wait;
    let task_kernel = kernel.clone();
    let task_job_id = job_id.clone();
    let handle = tokio::spawn(async move {
        task_kernel.jobs.mark_running(&task_job_id, "rendering_shorts");
        let mut updated_run = run;
        let mut rendered = Vec::new();
        let total = target_indices.len().max(1);

        let res: AicutResult<Vec<MediaAsset>> = async {
            for (step, &idx) in target_indices.iter().enumerate() {
                if cancel.is_cancelled() {
                    return Err(AicutError::cancelled("Renderizacao de Shorts cancelada."));
                }
                let clip = updated_run.clips[idx].clone();
                let pct = (step as f64 / total as f64) * 95.0;
                task_kernel.jobs.update_progress(
                    &task_job_id,
                    JobProgress {
                        percent: pct,
                        stage: if req.face_tracking || req.content_fit == ContentFit::SmartFocus {
                            "ai_framing_and_rendering".into()
                        } else {
                            "rendering_shorts".into()
                        },
                        message: Some(format!(
                            "IA enquadrando câmera + foco em quem fala/ação e renderizando ({}/{total}): '{}' ({:.0}s)...",
                            step + 1,
                            clip.title,
                            clip.total_duration_sec
                        )),
                        ..Default::default()
                    },
                );

                let asset = render_clip_to_library(
                    &task_kernel,
                    &ffmpeg_bin,
                    &ffprobe_bin,
                    &source,
                    &clip,
                    transcript.as_ref(),
                    req.layout,
                    req.webcam_corner,
                    req.content_fit,
                    effective_template_path.as_deref(),
                    req.face_tracking,
                    req.face_zoom,
                    req.border_trim_pct,
                    effective_webcam_rect,
                    effective_webcam_lock_static,
                    effective_content_rect,
                    effective_content_mode,
                    req.burn_subtitles,
                )
                .await?;
                updated_run.clips[idx].rendered_media_id = Some(asset.id.clone());
                rendered.push(asset);
            }
            task_kernel.repo.insert_clip_run(&updated_run)?;
            task_kernel
                .events
                .publish(AicutEvent::SmartClipRunCreated(updated_run));
            Ok(rendered)
        }
        .await;

        match res {
            Ok(assets) => {
                task_kernel.jobs.complete(
                    &task_job_id,
                    serde_json::to_value(&assets).unwrap_or(serde_json::Value::Null),
                );
                Ok(assets)
            }
            Err(err) => {
                if cancel.is_cancelled() {
                    task_kernel.jobs.mark_cancelled(&task_job_id);
                } else {
                    task_kernel.jobs.fail(&task_job_id, err.clone());
                }
                Err(err)
            }
        }
    });

    if wait {
        let rendered_assets = handle
            .await
            .map_err(|e| AicutError::internal(format!("Render task panicked: {e}")))??;
        Ok(RenderClipResponse {
            job_id,
            status: "completed".to_string(),
            rendered_assets,
        })
    } else {
        Ok(RenderClipResponse {
            job_id,
            status: "queued".to_string(),
            rendered_assets: Vec::new(),
        })
    }
}
