use aicut_core::{
    configure_tokio_command, extract_external_video_key, AicutError, AicutKernel, AicutResult,
};
use base64::Engine as _;
use chrono::Utc;
use image::{imageops, RgbImage};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::io::Cursor;
use std::path::Path;
use tokio::process::Command;
use uuid::Uuid;

use crate::engine::{resolve_gemini_api_key, SETTING_GEMINI_DIRECTOR_MODEL, SETTING_GEMINI_SCAN_MODEL};
use crate::face_track::{
    detect_pip_webcam_zero_border, installed_face_model, select_dominant_face_cluster,
    FaceBox, FaceDetector, PixelRect,
};
use crate::frames::probe_duration_sec;
use crate::gemini::{generate_with_inline_image, resolve_active_gemini_models};
use crate::prompts::extract_json_object;

pub const SETTING_FRAMING_STORE: &str = "smart_clips_framing_store";

/// Resolution-independent normalized rectangle in `[0.0, 1.0]` coordinates.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct NormalizedRect {
    pub x: f64,
    pub y: f64,
    pub w: f64,
    pub h: f64,
}

impl NormalizedRect {
    pub fn clamped(self) -> Self {
        let w = self.w.clamp(0.05, 1.0);
        let h = self.h.clamp(0.05, 1.0);
        let x = self.x.clamp(0.0, (1.0 - w).max(0.0));
        let y = self.y.clamp(0.0, (1.0 - h).max(0.0));
        Self { x, y, w, h }
    }

    pub fn to_pixel_rect(self) -> PixelRect {
        let c = self.clamped();
        PixelRect {
            x: c.x,
            y: c.y,
            w: c.w,
            h: c.h,
        }
    }

    pub fn from_pixel_rect(r: PixelRect) -> Self {
        Self {
            x: r.x,
            y: r.y,
            w: r.w,
            h: r.h,
        }
        .clamped()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum ContentRegionMode {
    /// AI Director tracks active speaker or peak visual action strictly inside the saved content area.
    AiDirectorInRegion,
    /// Shows the exact saved content area cleanly with zero panning.
    FixedRegion,
    /// Keeps 100% of the saved content area visible over a soft blurred background.
    FitBlurRegion,
}

impl Default for ContentRegionMode {
    fn default() -> Self {
        Self::AiDirectorInRegion
    }
}

/// Precision saved webcam + content framing profile configured in the Enquadramento tab.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct FramingProfile {
    #[serde(default = "default_profile_id")]
    pub id: String,
    #[serde(default = "default_profile_name")]
    pub name: String,
    /// Precision clean webcam area (no borders, normalized 0..1). None = automatic AI detection.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub webcam_rect: Option<NormalizedRect>,
    /// Optional precision content region (normalized 0..1). None = full 16:9 content minus webcam.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub content_rect: Option<NormalizedRect>,
    /// When true and `webcam_rect` is set, displays the exact saved webcam area cleanly with zero zoom/pan.
    #[serde(default = "default_true")]
    pub webcam_lock_static: bool,
    #[serde(default)]
    pub content_mode: ContentRegionMode,
    /// Safety inset used when webcam is auto-detected (default 4.5%).
    #[serde(default = "default_border_trim")]
    pub border_trim_pct: f32,
    /// Natural webcam zoom factor (1.0 = full clean webcam without face zoom).
    #[serde(default = "default_face_zoom")]
    pub face_zoom: f32,
    #[serde(default = "default_now_iso")]
    pub updated_at: String,
}

impl Default for FramingProfile {
    fn default() -> Self {
        Self {
            id: "default".to_string(),
            name: "Padrão".to_string(),
            webcam_rect: None,
            content_rect: None,
            webcam_lock_static: true,
            content_mode: ContentRegionMode::AiDirectorInRegion,
            border_trim_pct: 0.045,
            face_zoom: 1.0,
            updated_at: default_now_iso(),
        }
    }
}

fn default_profile_id() -> String {
    "default".to_string()
}
fn default_profile_name() -> String {
    "Padrão".to_string()
}
fn default_true() -> bool {
    true
}
fn default_border_trim() -> f32 {
    0.045
}
fn default_face_zoom() -> f32 {
    1.0
}
fn default_now_iso() -> String {
    Utc::now().to_rfc3339()
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, JsonSchema)]
pub struct FramingStore {
    #[serde(default)]
    pub default_profile: FramingProfile,
    #[serde(default)]
    pub presets: Vec<FramingProfile>,
    #[serde(default)]
    pub per_media: HashMap<String, FramingProfile>,
}

pub fn load_framing_store(kernel: &AicutKernel) -> FramingStore {
    kernel
        .repo
        .get_setting(SETTING_FRAMING_STORE)
        .ok()
        .flatten()
        .and_then(|v| serde_json::from_value::<FramingStore>(v).ok())
        .unwrap_or_default()
}

pub fn save_framing_store(kernel: &AicutKernel, store: &FramingStore) -> AicutResult<()> {
    let val = serde_json::to_value(store)
        .map_err(|e| AicutError::internal(format!("Failed serializing framing store: {e}")))?;
    kernel.repo.set_setting(SETTING_FRAMING_STORE, &val)
}

/// Resolves the effective `FramingProfile` for a media asset:
/// 1. Exact `media_id` override
/// 2. External video key override (e.g. `yt:fPQDxLwScrA`)
/// 3. Global `default_profile`
pub fn resolve_effective_framing(
    kernel: &AicutKernel,
    media_id: Option<&str>,
) -> (FramingProfile, bool) {
    let store = load_framing_store(kernel);
    if let Some(mid) = media_id.filter(|s| !s.trim().is_empty()) {
        if let Some(p) = store.per_media.get(mid) {
            return (p.clone(), true);
        }
        if let Ok(Some(asset)) = kernel.repo.get_media(mid) {
            if let Some(ext_key) = extract_external_video_key(
                asset.source_url.as_deref(),
                &asset.title,
                &asset.file_name,
            ) {
                if let Some(p) = store.per_media.get(&ext_key) {
                    return (p.clone(), true);
                }
            }
        }
    }
    (store.default_profile, false)
}

// ---- Action Input / Output Schemas ----

#[derive(Debug, Clone, Default, Serialize, Deserialize, JsonSchema)]
pub struct FramingGetInput {
    #[serde(default)]
    pub media_id: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct FramingGetOutput {
    pub active_profile: FramingProfile,
    pub has_media_override: bool,
    pub default_profile: FramingProfile,
    pub presets: Vec<FramingProfile>,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct FramingSaveInput {
    /// Save scope: "default" (global for all Shorts), "media" (specific video), or "preset" (named preset).
    #[serde(default = "default_scope")]
    pub scope: String,
    #[serde(default)]
    pub media_id: Option<String>,
    pub profile: FramingProfile,
}

fn default_scope() -> String {
    "default".to_string()
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct FramingDeleteInput {
    #[serde(default)]
    pub preset_id: Option<String>,
    #[serde(default)]
    pub media_id: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct FramingPreviewInput {
    pub media_id: String,
    #[serde(default)]
    pub timestamp_sec: Option<f64>,
    #[serde(default)]
    pub auto_detect: bool,
    #[serde(default = "default_border_trim")]
    pub border_trim_pct: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct FramingPreviewOutput {
    pub media_id: String,
    pub timestamp_sec: f64,
    pub duration_sec: f64,
    pub width: u32,
    pub height: u32,
    pub image_data_url: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub detected_webcam_rect: Option<NormalizedRect>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub detected_content_rect: Option<NormalizedRect>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub detected_face_rect: Option<NormalizedRect>,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct FramingReviewInput {
    pub media_id: String,
    #[serde(default)]
    pub timestamp_sec: Option<f64>,
    pub webcam_rect: NormalizedRect,
    #[serde(default)]
    pub content_rect: Option<NormalizedRect>,
    /// Use configured multimodal Vision LLM (Gemini) in addition to local UltraFace + Border QA.
    #[serde(default = "default_true")]
    pub use_vlm: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct FramingReviewOutput {
    pub approved: bool,
    /// Overall framing quality score from 0 to 100.
    pub score: u8,
    /// "otimo" | "ajustar" | "problema"
    pub verdict: String,
    pub reviewer_engine: String,
    pub has_border_leak: bool,
    pub border_sides: Vec<String>,
    /// "ideal" | "muito_perto" | "muito_longe" | "sem_rosto"
    pub zoom_assessment: String,
    pub head_or_face_clipped: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub face_height_pct: Option<f32>,
    pub summary: String,
    pub suggestions: Vec<String>,
    /// Exact cropped webcam image inspected by the AI (`data:image/jpeg;base64,...`).
    pub cropped_webcam_preview_data_url: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub suggested_webcam_rect: Option<NormalizedRect>,
}

pub fn save_framing_profile(
    kernel: &AicutKernel,
    input: FramingSaveInput,
) -> AicutResult<FramingGetOutput> {
    let mut store = load_framing_store(kernel);
    let mut profile = input.profile;
    profile.webcam_rect = profile.webcam_rect.map(NormalizedRect::clamped);
    profile.content_rect = profile.content_rect.map(NormalizedRect::clamped);
    profile.face_zoom = profile.face_zoom.clamp(0.85, 1.45);
    profile.border_trim_pct = profile.border_trim_pct.clamp(0.01, 0.12);
    profile.updated_at = Utc::now().to_rfc3339();

    match input.scope.to_lowercase().as_str() {
        "media" => {
            let mid = input.media_id.clone().ok_or_else(|| {
                AicutError::invalid_input("media_id is required when scope is 'media'.")
            })?;
            profile.id = format!("media:{mid}");
            store.per_media.insert(mid.clone(), profile.clone());
            // Also index by external video key if available so related assets inherit it
            if let Ok(Some(asset)) = kernel.repo.get_media(&mid) {
                if let Some(ext_key) = extract_external_video_key(
                    asset.source_url.as_deref(),
                    &asset.title,
                    &asset.file_name,
                ) {
                    store.per_media.insert(ext_key, profile);
                }
            }
        }
        "preset" => {
            if profile.id.trim().is_empty() || profile.id == "default" {
                profile.id = Uuid::new_v4().to_string();
            }
            if profile.name.trim().is_empty() {
                profile.name = "Preset personalizado".to_string();
            }
            if let Some(existing) = store.presets.iter_mut().find(|p| p.id == profile.id) {
                *existing = profile;
            } else {
                store.presets.push(profile);
            }
        }
        _ => {
            profile.id = "default".to_string();
            if profile.name.trim().is_empty() {
                profile.name = "Padrão".to_string();
            }
            store.default_profile = profile;
        }
    }

    save_framing_store(kernel, &store)?;
    let (active_profile, has_media_override) =
        resolve_effective_framing(kernel, input.media_id.as_deref());
    Ok(FramingGetOutput {
        active_profile,
        has_media_override,
        default_profile: store.default_profile,
        presets: store.presets,
    })
}

pub fn delete_framing_entry(
    kernel: &AicutKernel,
    input: FramingDeleteInput,
) -> AicutResult<FramingGetOutput> {
    let mut store = load_framing_store(kernel);
    if let Some(ref pid) = input.preset_id {
        store.presets.retain(|p| p.id != *pid);
    }
    if let Some(ref mid) = input.media_id {
        store.per_media.remove(mid);
        if let Ok(Some(asset)) = kernel.repo.get_media(mid) {
            if let Some(ext_key) = extract_external_video_key(
                asset.source_url.as_deref(),
                &asset.title,
                &asset.file_name,
            ) {
                store.per_media.remove(&ext_key);
            }
        }
    }
    save_framing_store(kernel, &store)?;
    let (active_profile, has_media_override) =
        resolve_effective_framing(kernel, input.media_id.as_deref());
    Ok(FramingGetOutput {
        active_profile,
        has_media_override,
        default_profile: store.default_profile,
        presets: store.presets,
    })
}

/// Extracts a high-clarity preview frame for the Framing Studio canvas and optionally runs
/// automatic UltraFace + 4-phase Zero-Border OBS webcam detection across sampled frames.
pub async fn extract_framing_preview(
    kernel: &AicutKernel,
    input: FramingPreviewInput,
) -> AicutResult<FramingPreviewOutput> {
    let source = kernel
        .repo
        .get_media(&input.media_id)?
        .ok_or_else(|| AicutError::not_found(format!("Media '{}' not found", input.media_id)))?;
    let ffmpeg_bin = kernel.tools.resolve_tool("ffmpeg")?;
    let ffprobe_bin = kernel.tools.resolve_tool("ffprobe")?;

    let duration_sec = match source.metadata.duration_sec {
        Some(d) if d > 0.5 => d,
        _ => probe_duration_sec(&ffprobe_bin, Path::new(&source.file_path))
            .await
            .unwrap_or(60.0),
    };
    let ts = input
        .timestamp_sec
        .unwrap_or_else(|| (duration_sec * 0.20).min(30.0))
        .clamp(0.0, (duration_sec - 0.2).max(0.0));

    let tmp_dir = std::env::temp_dir().join(format!("aicut-frameprev-{}", Uuid::new_v4()));
    std::fs::create_dir_all(&tmp_dir)?;
    let frame_path = tmp_dir.join("preview.jpg");

    extract_single_jpeg(
        &ffmpeg_bin,
        Path::new(&source.file_path),
        ts,
        1280,
        &frame_path,
    )
    .await?;

    let jpeg_bytes = std::fs::read(&frame_path)?;
    let (width, height) = image::load_from_memory(&jpeg_bytes)
        .map(|im| (im.width(), im.height()))
        .unwrap_or((
            source.metadata.width.unwrap_or(1280),
            source.metadata.height.unwrap_or(720),
        ));
    let b64 = base64::engine::general_purpose::STANDARD.encode(&jpeg_bytes);
    let image_data_url = format!("data:image/jpeg;base64,{b64}");

    let mut detected_webcam_rect = None;
    let mut detected_content_rect = None;
    let mut detected_face_rect = None;

    if input.auto_detect {
        let sampled_paths = extract_evenly_spaced_frames(
            &ffmpeg_bin,
            Path::new(&source.file_path),
            duration_sec,
            ts,
            10,
            &tmp_dir,
        )
        .await;
        if let (Ok(model_path), false) = (
            installed_face_model(kernel),
            sampled_paths.is_empty(),
        ) {
            let trim = input.border_trim_pct.clamp(0.01, 0.12) as f64;
            if let Ok(Ok((cam_r, cnt_r, face_r))) = tokio::task::spawn_blocking(move || -> AicutResult<(Option<NormalizedRect>, Option<NormalizedRect>, Option<NormalizedRect>)> {
                let det = FaceDetector::load(&model_path)?;
                let loaded: Vec<(f64, RgbImage)> = sampled_paths
                    .iter()
                    .enumerate()
                    .filter_map(|(i, p)| image::open(p).ok().map(|im| (i as f64, im.to_rgb8())))
                    .collect();
                if loaded.is_empty() {
                    return Ok((None, None, None));
                }
                let mut faces = Vec::new();
                for (_, rgb) in &loaded {
                    if let Ok(found) = det.detect(rgb, 0.58) {
                        faces.extend(found);
                    }
                }
                let Some(anchor) = select_dominant_face_cluster(&faces) else {
                    return Ok((None, None, None));
                };
                let face_rect = NormalizedRect {
                    x: anchor.x1 as f64,
                    y: anchor.y1 as f64,
                    w: anchor.w() as f64,
                    h: anchor.h() as f64,
                }
                .clamped();
                let pip = detect_pip_webcam_zero_border(&loaded, anchor, trim, false);
                let cam_rect = if pip.walls_detected >= 2 {
                    NormalizedRect::from_pixel_rect(pip.safe_rect)
                } else {
                    // Fallback clean window around detected face if no clear OBS border
                    NormalizedRect {
                        x: (anchor.cx() as f64 - 0.17).clamp(0.0, 0.66),
                        y: (anchor.cy() as f64 - 0.17).clamp(0.0, 0.66),
                        w: 0.34,
                        h: 0.34,
                    }
                    .clamped()
                };
                let content_rect = suggest_clean_content_rect_avoiding_webcam(cam_rect);
                Ok((Some(cam_rect), Some(content_rect), Some(face_rect)))
            })
            .await
            {
                detected_webcam_rect = cam_r;
                detected_content_rect = cnt_r;
                detected_face_rect = face_r;
            }
        }
    }

    let _ = std::fs::remove_dir_all(&tmp_dir);

    Ok(FramingPreviewOutput {
        media_id: source.id,
        timestamp_sec: ts,
        duration_sec,
        width,
        height,
        image_data_url,
        detected_webcam_rect,
        detected_content_rect,
        detected_face_rect,
    })
}

/// Reviews the user's webcam framing using a Dual-Layer AI Pipeline:
/// - Layer 1 (Instant Native Rust): UltraFace ONNX checks face zoom ratio (flags "muito_perto"
///   if camera is glued to face), head/face clipping, and 4-edge OBS border leaks across 3 timestamps.
/// - Layer 2 (Multimodal Vision LLM): Sends the exact cropped 1020:582 webcam preview JPEG to
///   Gemini Vision (when configured) for human-like visual verification and 1-click auto-fix!
pub async fn review_webcam_framing(
    kernel: &AicutKernel,
    input: FramingReviewInput,
) -> AicutResult<FramingReviewOutput> {
    let source = kernel
        .repo
        .get_media(&input.media_id)?
        .ok_or_else(|| AicutError::not_found(format!("Media '{}' not found", input.media_id)))?;
    let ffmpeg_bin = kernel.tools.resolve_tool("ffmpeg")?;
    let ffprobe_bin = kernel.tools.resolve_tool("ffprobe")?;

    let duration_sec = match source.metadata.duration_sec {
        Some(d) if d > 0.5 => d,
        _ => probe_duration_sec(&ffprobe_bin, Path::new(&source.file_path))
            .await
            .unwrap_or(60.0),
    };
    let ts = input
        .timestamp_sec
        .unwrap_or_else(|| (duration_sec * 0.20).min(30.0))
        .clamp(0.0, (duration_sec - 0.2).max(0.0));
    let rect = input.webcam_rect.clamped();

    let tmp_dir = std::env::temp_dir().join(format!("aicut-framerev-{}", Uuid::new_v4()));
    std::fs::create_dir_all(&tmp_dir)?;

    let sample_ts = vec![
        ts,
        (ts + 12.0).min((duration_sec - 0.5).max(0.0)),
        (ts + 28.0).min((duration_sec - 0.5).max(0.0)),
    ];
    let mut sample_paths = Vec::new();
    for (i, &t_s) in sample_ts.iter().enumerate() {
        let p = tmp_dir.join(format!("rev_{i}.jpg"));
        if extract_single_jpeg(&ffmpeg_bin, Path::new(&source.file_path), t_s, 1280, &p)
            .await
            .is_ok()
        {
            sample_paths.push(p);
        }
    }
    if sample_paths.is_empty() {
        let _ = std::fs::remove_dir_all(&tmp_dir);
        return Err(AicutError::internal(
            "Nao foi possivel extrair frames do video para revisar o enquadramento.",
        ));
    }

    let model_path = installed_face_model(kernel).ok();
    let local_qa = tokio::task::spawn_blocking(move || -> AicutResult<LocalWebcamQa> {
        let imgs: Vec<RgbImage> = sample_paths
            .iter()
            .filter_map(|p| image::open(p).ok().map(|im| im.to_rgb8()))
            .collect();
        if imgs.is_empty() {
            return Err(AicutError::internal("Failed decoding review frames"));
        }

        // Crop exact webcam box matching SMOKING33 top slot aspect ratio (1020:582 = 1.7525)
        let cropped_imgs: Vec<RgbImage> = imgs
            .iter()
            .map(|im| crop_rect_for_top_slot(im, rect, 640, 366))
            .collect();

        let preview_jpeg = encode_rgb_to_jpeg(&cropped_imgs[0])?;

        // Check 4 outer edges for static/solid OBS border leaks or sharp border lines
        let border_sides = detect_crop_edge_border_leaks(&cropped_imgs);

        // Run UltraFace ONNX on cropped webcam previews
        let detector = model_path
            .as_deref()
            .and_then(|p| FaceDetector::load(p).ok());
        let mut faces: Vec<FaceBox> = Vec::new();
        if let Some(det) = detector {
            for ci in &cropped_imgs {
                if let Ok(found) = det.detect(ci, 0.52) {
                    if let Some(best) = found.into_iter().max_by(|a, b| {
                        (a.h() * a.score)
                            .partial_cmp(&(b.h() * b.score))
                            .unwrap_or(std::cmp::Ordering::Equal)
                    }) {
                        faces.push(best);
                    }
                }
            }
        }

        let avg_face = if !faces.is_empty() {
            let n = faces.len() as f32;
            Some(FaceBox {
                x1: faces.iter().map(|f| f.x1).sum::<f32>() / n,
                y1: faces.iter().map(|f| f.y1).sum::<f32>() / n,
                x2: faces.iter().map(|f| f.x2).sum::<f32>() / n,
                y2: faces.iter().map(|f| f.y2).sum::<f32>() / n,
                score: faces.iter().map(|f| f.score).sum::<f32>() / n,
            })
        } else {
            None
        };

        Ok(evaluate_local_qa(rect, avg_face, border_sides, preview_jpeg))
    })
    .await
    .map_err(|e| AicutError::internal(format!("Framing QA task failed: {e}")))??;

    let _ = std::fs::remove_dir_all(&tmp_dir);

    let b64_crop = base64::engine::general_purpose::STANDARD.encode(&local_qa.preview_jpeg);
    let cropped_data_url = format!("data:image/jpeg;base64,{b64_crop}");

    // Layer 2: Multimodal Vision LLM review via Gemini Vision if API key is configured
    if input.use_vlm {
        if let Some(api_key) = resolve_gemini_api_key(kernel, None) {
            let scan_pref = kernel
                .repo
                .get_setting(SETTING_GEMINI_SCAN_MODEL)
                .ok()
                .flatten()
                .and_then(|v| v.as_str().map(|s| s.to_string()))
                .unwrap_or_else(|| "auto".to_string());
            let dir_pref = kernel
                .repo
                .get_setting(SETTING_GEMINI_DIRECTOR_MODEL)
                .ok()
                .flatten()
                .and_then(|v| v.as_str().map(|s| s.to_string()))
                .unwrap_or_else(|| "auto".to_string());
            let http = kernel.tools.http_client();
            let (flash_model, _) =
                resolve_active_gemini_models(http, &api_key, &scan_pref, &dir_pref).await;

            let prompt = build_vlm_framing_review_prompt(local_qa.face_height_pct);
            if let Ok(raw_text) =
                generate_with_inline_image(http, &api_key, &flash_model, &b64_crop, &prompt).await
            {
                if let Ok(parsed) = parse_vlm_framing_review(&raw_text) {
                    let mut border_sides = parsed.border_sides.clone();
                    for s in &local_qa.border_sides {
                        if !border_sides.iter().any(|x| x.eq_ignore_ascii_case(s)) {
                            border_sides.push(s.clone());
                        }
                    }
                    let has_border_leak = parsed.has_border_leak || !border_sides.is_empty();
                    let suggested = apply_vlm_adjustments(
                        rect,
                        &parsed.recommended_adjustments,
                        &border_sides,
                        &parsed.zoom_assessment,
                    );
                    let score = parsed.score.min(100);
                    let approved = parsed.approved && !has_border_leak && score >= 78;
                    let verdict = if approved {
                        "otimo".to_string()
                    } else if score >= 55 {
                        "ajustar".to_string()
                    } else {
                        "problema".to_string()
                    };

                    return Ok(FramingReviewOutput {
                        approved,
                        score,
                        verdict,
                        reviewer_engine: format!("Gemini Vision ({flash_model}) + UltraFace QA"),
                        has_border_leak,
                        border_sides,
                        zoom_assessment: parsed.zoom_assessment,
                        head_or_face_clipped: parsed.head_or_face_clipped || local_qa.head_clipped,
                        face_height_pct: local_qa.face_height_pct,
                        summary: parsed.summary,
                        suggestions: parsed.suggestions,
                        cropped_webcam_preview_data_url: cropped_data_url,
                        suggested_webcam_rect: suggested,
                    });
                }
            }
        }
    }

    // Fallback cleanly to Layer 1 Native Rust Vision QA (instant, offline, free)
    Ok(FramingReviewOutput {
        approved: local_qa.approved,
        score: local_qa.score,
        verdict: local_qa.verdict,
        reviewer_engine: "IA Visual Local (UltraFace ONNX + Detector de Bordas)".to_string(),
        has_border_leak: !local_qa.border_sides.is_empty(),
        border_sides: local_qa.border_sides,
        zoom_assessment: local_qa.zoom_assessment,
        head_or_face_clipped: local_qa.head_clipped,
        face_height_pct: local_qa.face_height_pct,
        summary: local_qa.summary,
        suggestions: local_qa.suggestions,
        cropped_webcam_preview_data_url: cropped_data_url,
        suggested_webcam_rect: local_qa.suggested_rect,
    })
}

struct LocalWebcamQa {
    approved: bool,
    score: u8,
    verdict: String,
    border_sides: Vec<String>,
    zoom_assessment: String,
    head_clipped: bool,
    face_height_pct: Option<f32>,
    summary: String,
    suggestions: Vec<String>,
    suggested_rect: Option<NormalizedRect>,
    preview_jpeg: Vec<u8>,
}

fn evaluate_local_qa(
    rect: NormalizedRect,
    face: Option<FaceBox>,
    border_sides: Vec<String>,
    preview_jpeg: Vec<u8>,
) -> LocalWebcamQa {
    let mut score: i32 = 95;
    let mut suggestions = Vec::new();
    let mut head_clipped = false;
    let mut face_height_pct = None;
    let mut adj = rect;
    let mut changed = false;

    let zoom_assessment = if let Some(f) = face {
        let fh_pct = f.h() * 100.0;
        face_height_pct = Some((fh_pct * 10.0).round() / 10.0);
        if f.y1 < 0.04 {
            head_clipped = true;
            score -= 18;
            suggestions.push("O topo da cabeça está muito próximo ou cortado na borda superior — suba um pouco a área da webcam.".to_string());
            let lift = rect.h * 0.08;
            adj.y = (adj.y - lift).max(0.0);
            adj.h = (adj.h + lift).min(1.0 - adj.y);
            changed = true;
        }
        if f.x1 < 0.04 || f.x2 > 0.96 {
            head_clipped = true;
            score -= 15;
            suggestions.push("O rosto está encostado na lateral — centralize ou amplie ligeiramente a largura da área.".to_string());
            changed = true;
        }

        if fh_pct > 46.0 {
            score -= 22;
            suggestions.push(format!(
                "Zoom muito alto (rosto ocupa {fh_pct:.0}% da altura) — abra mais a caixa da webcam para mostrar rosto e ombros de forma natural."
            ));
            let expand_w = rect.w * 0.14;
            let expand_h = rect.h * 0.14;
            adj.x = (adj.x - expand_w * 0.5).max(0.0);
            adj.y = (adj.y - expand_h * 0.5).max(0.0);
            adj.w = (adj.w + expand_w).min(1.0 - adj.x);
            adj.h = (adj.h + expand_h).min(1.0 - adj.y);
            changed = true;
            "muito_perto".to_string()
        } else if fh_pct < 12.0 {
            score -= 14;
            suggestions.push(format!(
                "Rosto pequeno na webcam ({fh_pct:.0}% da altura) — feche um pouco a área ao redor da webcam."
            ));
            "muito_longe".to_string()
        } else {
            "ideal".to_string()
        }
    } else {
        score -= 35;
        suggestions.push("Nenhum rosto foi detectado dentro da caixa selecionada — arraste a caixa verde para cima da sua webcam.".to_string());
        "sem_rosto".to_string()
    };

    if !border_sides.is_empty() {
        score -= (border_sides.len() as i32) * 16;
        let sides_pt = border_sides
            .iter()
            .map(|s| match s.as_str() {
                "top" => "topo",
                "bottom" => "baixo",
                "left" => "esquerda",
                "right" => "direita",
                other => other,
            })
            .collect::<Vec<_>>()
            .join(", ");
        suggestions.push(format!(
            "Possível borda/moldura do OBS detectada em: {sides_pt}. Encolha ligeiramente esses lados para dentro."
        ));
        for s in &border_sides {
            match s.as_str() {
                "left" => {
                    let d = rect.w * 0.035;
                    adj.x += d;
                    adj.w = (adj.w - d).max(0.08);
                }
                "right" => {
                    let d = rect.w * 0.035;
                    adj.w = (adj.w - d).max(0.08);
                }
                "top" => {
                    let d = rect.h * 0.035;
                    adj.y += d;
                    adj.h = (adj.h - d).max(0.08);
                }
                "bottom" => {
                    let d = rect.h * 0.035;
                    adj.h = (adj.h - d).max(0.08);
                }
                _ => {}
            }
        }
        changed = true;
    }

    let score_u8 = score.clamp(0, 100) as u8;
    let approved = score_u8 >= 80 && border_sides.is_empty() && !head_clipped && zoom_assessment == "ideal";
    let verdict = if approved {
        "otimo".to_string()
    } else if score_u8 >= 55 {
        "ajustar".to_string()
    } else {
        "problema".to_string()
    };

    let summary = if approved {
        "Enquadramento aprovado! Webcam limpa sem bordas aparentes, rosto bem posicionado e distância natural (sem zoom colado).".to_string()
    } else if !border_sides.is_empty() {
        "A IA detectou resquício de borda ou moldura do OBS nas extremidades da webcam. Clique em 'Aplicar ajuste sugerido' para limpar.".to_string()
    } else if zoom_assessment == "muito_perto" {
        "A área da webcam está muito fechada (zoom colado no rosto). Recomenda-se abrir mais a caixa para mostrar a webcam de forma natural.".to_string()
    } else if zoom_assessment == "sem_rosto" {
        "Não foi possível confirmar o rosto na área selecionada. Ajuste a caixa verde sobre o interior da webcam.".to_string()
    } else {
        "Enquadramento razoável, mas pequenos ajustes podem melhorar o respiro da cabeça e a limpeza das bordas.".to_string()
    };

    LocalWebcamQa {
        approved,
        score: score_u8,
        verdict,
        border_sides,
        zoom_assessment,
        head_clipped,
        face_height_pct,
        summary,
        suggestions,
        suggested_rect: if changed { Some(adj.clamped()) } else { None },
        preview_jpeg,
    }
}

/// Checks the 4 outer edges (top/bottom/left/right 3px strips) of cropped webcam previews across
/// multiple timestamps to catch static solid OBS borders, black bars, or sharp border lines.
pub fn detect_crop_edge_border_leaks(crops: &[RgbImage]) -> Vec<String> {
    if crops.is_empty() {
        return Vec::new();
    }
    let w = crops[0].width() as usize;
    let h = crops[0].height() as usize;
    if w < 32 || h < 32 {
        return Vec::new();
    }

    let rgb_diff = |a: &image::Rgb<u8>, b: &image::Rgb<u8>| -> f64 {
        ((a[0] as i32 - b[0] as i32).abs()
            + (a[1] as i32 - b[1] as i32).abs()
            + (a[2] as i32 - b[2] as i32).abs()) as f64
            / 3.0
    };

    let img = &crops[0];
    let has_multi = crops.len() >= 2;
    let img2 = if has_multi { &crops[1] } else { img };
    let mut leaked = Vec::new();

    // Helper for horizontal edge strip (top or bottom)
    let check_h_edge = |edge_y: usize, inner_y: usize| -> bool {
        let x0 = w / 8;
        let x1 = w - w / 8;
        let mut uniform_sum = 0.0;
        let mut jump_cnt = 0usize;
        let mut t_diff_sum = 0.0;
        let mut cnt = 0usize;
        for x in x0..x1.saturating_sub(1) {
            let p = img.get_pixel(x as u32, edge_y as u32);
            let pn = img.get_pixel((x + 1) as u32, edge_y as u32);
            let pin = img.get_pixel(x as u32, inner_y as u32);
            uniform_sum += rgb_diff(p, pn);
            if rgb_diff(p, pin) >= 18.0 {
                jump_cnt += 1;
            }
            if has_multi && img2.width() as usize == w && img2.height() as usize == h {
                t_diff_sum += rgb_diff(p, img2.get_pixel(x as u32, edge_y as u32));
            }
            cnt += 1;
        }
        if cnt == 0 {
            return false;
        }
        let uniformity = uniform_sum / cnt as f64;
        let jump_frac = jump_cnt as f64 / cnt as f64;
        let t_mad = t_diff_sum / cnt as f64;
        (uniformity < 4.0 && jump_frac >= 0.45 && (!has_multi || t_mad < 2.5))
            || (jump_frac >= 0.65 && uniformity < 8.0)
    };

    // Helper for vertical edge strip (left or right)
    let check_v_edge = |edge_x: usize, inner_x: usize| -> bool {
        let y0 = h / 8;
        let y1 = h - h / 8;
        let mut uniform_sum = 0.0;
        let mut jump_cnt = 0usize;
        let mut t_diff_sum = 0.0;
        let mut cnt = 0usize;
        for y in y0..y1.saturating_sub(1) {
            let p = img.get_pixel(edge_x as u32, y as u32);
            let pn = img.get_pixel(edge_x as u32, (y + 1) as u32);
            let pin = img.get_pixel(inner_x as u32, y as u32);
            uniform_sum += rgb_diff(p, pn);
            if rgb_diff(p, pin) >= 18.0 {
                jump_cnt += 1;
            }
            if has_multi && img2.width() as usize == w && img2.height() as usize == h {
                t_diff_sum += rgb_diff(p, img2.get_pixel(edge_x as u32, y as u32));
            }
            cnt += 1;
        }
        if cnt == 0 {
            return false;
        }
        let uniformity = uniform_sum / cnt as f64;
        let jump_frac = jump_cnt as f64 / cnt as f64;
        let t_mad = t_diff_sum / cnt as f64;
        (uniformity < 4.0 && jump_frac >= 0.45 && (!has_multi || t_mad < 2.5))
            || (jump_frac >= 0.65 && uniformity < 8.0)
    };

    if check_h_edge(1, 10.min(h - 1)) {
        leaked.push("top".to_string());
    }
    if check_h_edge(h - 2, h.saturating_sub(11)) {
        leaked.push("bottom".to_string());
    }
    if check_v_edge(1, 10.min(w - 1)) {
        leaked.push("left".to_string());
    }
    if check_v_edge(w - 2, w.saturating_sub(11)) {
        leaked.push("right".to_string());
    }
    leaked
}

#[derive(Debug, Deserialize)]
struct VlmFramingReviewJson {
    #[serde(default)]
    approved: bool,
    #[serde(default = "default_vlm_score")]
    score: u8,
    #[serde(default)]
    has_border_leak: bool,
    #[serde(default)]
    border_sides: Vec<String>,
    #[serde(default = "default_zoom_str")]
    zoom_assessment: String,
    #[serde(default)]
    head_or_face_clipped: bool,
    #[serde(default)]
    summary: String,
    #[serde(default)]
    suggestions: Vec<String>,
    #[serde(default)]
    recommended_adjustments: VlmAdjustments,
}

#[derive(Debug, Default, Deserialize)]
struct VlmAdjustments {
    #[serde(default)]
    inset_top_pct: f64,
    #[serde(default)]
    inset_bottom_pct: f64,
    #[serde(default)]
    inset_left_pct: f64,
    #[serde(default)]
    inset_right_pct: f64,
    #[serde(default)]
    expand_pct: f64,
}

fn default_vlm_score() -> u8 {
    75
}
fn default_zoom_str() -> String {
    "ideal".to_string()
}

fn build_vlm_framing_review_prompt(face_height_pct: Option<f32>) -> String {
    let hint = face_height_pct
        .map(|p| format!("Medicao auxiliar: o rosto ocupa cerca de {p:.0}% da altura desta janela."))
        .unwrap_or_default();
    format!(
        r#"Você é um diretor de vídeo especialista em Shorts/Reels revisando o recorte da WEBCAM que vai na janela superior de um Short 9:16.
Analise visualmente esta imagem recortada da webcam com rigor:
1. BORDA / MOLDURA DO OBS: existe QUALQUER pedaço de borda colorida, neon, linha reta de moldura, canto arredondado, sombra de overlay, barra de nome ou vazamento da gameplay nas extremidades (top, bottom, left, right)?
2. ZOOM / DISTÂNCIA: o enquadramento parece uma webcam natural (mostrando rosto + ombros/setup confortavelmente = "ideal"), ou está com zoom muito alto / colado no rosto ("muito_perto"), ou muito longe ("muito_longe")?
3. CORTE DE CABEÇA / ROSTO: o topo da cabeça ou parte do rosto está cortado?
{hint}

Responda APENAS JSON válido neste formato exato:
{{
  "approved": true,
  "score": 92,
  "has_border_leak": false,
  "border_sides": [],
  "zoom_assessment": "ideal",
  "head_or_face_clipped": false,
  "summary": "Resumo objetivo em português do Brasil (1-2 frases).",
  "suggestions": ["Sugestão prática curta em português, se houver."],
  "recommended_adjustments": {{
    "inset_top_pct": 0.0,
    "inset_bottom_pct": 0.0,
    "inset_left_pct": 0.0,
    "inset_right_pct": 0.0,
    "expand_pct": 0.0
  }}
}}
Regras para recommended_adjustments: use valores entre 0.0 e 12.0 (em porcentagem relativa à caixa atual). Se houver borda em algum lado, informe inset_*_pct (ex: 3.5). Se estiver muito perto/colado no rosto, informe expand_pct (ex: 10.0)."#
    )
}

fn parse_vlm_framing_review(raw: &str) -> AicutResult<VlmFramingReviewJson> {
    let json_str = extract_json_object(raw).unwrap_or(raw);
    serde_json::from_str::<VlmFramingReviewJson>(json_str)
        .map_err(|e| AicutError::internal(format!("Failed parsing VLM framing review JSON: {e}")))
}

fn apply_vlm_adjustments(
    rect: NormalizedRect,
    adj: &VlmAdjustments,
    border_sides: &[String],
    zoom_assessment: &str,
) -> Option<NormalizedRect> {
    let mut r = rect;
    let mut touched = false;

    let mut in_t = adj.inset_top_pct.clamp(0.0, 15.0) / 100.0;
    let mut in_b = adj.inset_bottom_pct.clamp(0.0, 15.0) / 100.0;
    let mut in_l = adj.inset_left_pct.clamp(0.0, 15.0) / 100.0;
    let mut in_r = adj.inset_right_pct.clamp(0.0, 15.0) / 100.0;

    for s in border_sides {
        match s.to_lowercase().as_str() {
            "top" if in_t < 0.025 => in_t = 0.03,
            "bottom" if in_b < 0.025 => in_b = 0.03,
            "left" if in_l < 0.025 => in_l = 0.03,
            "right" if in_r < 0.025 => in_r = 0.03,
            _ => {}
        }
    }

    if in_t > 0.002 || in_b > 0.002 || in_l > 0.002 || in_r > 0.002 {
        let dx_l = r.w * in_l;
        let dx_r = r.w * in_r;
        let dy_t = r.h * in_t;
        let dy_b = r.h * in_b;
        r.x += dx_l;
        r.y += dy_t;
        r.w = (r.w - dx_l - dx_r).max(0.08);
        r.h = (r.h - dy_t - dy_b).max(0.08);
        touched = true;
    }

    let mut exp = adj.expand_pct.clamp(0.0, 18.0) / 100.0;
    if zoom_assessment == "muito_perto" && exp < 0.05 && border_sides.is_empty() {
        exp = 0.10;
    }
    if exp > 0.005 {
        let dw = r.w * exp;
        let dh = r.h * exp;
        r.x = (r.x - dw * 0.5).max(0.0);
        r.y = (r.y - dh * 0.5).max(0.0);
        r.w = (r.w + dw).min(1.0 - r.x);
        r.h = (r.h + dh).min(1.0 - r.y);
        touched = true;
    }

    if touched {
        Some(r.clamped())
    } else {
        None
    }
}

fn suggest_clean_content_rect_avoiding_webcam(cam: NormalizedRect) -> NormalizedRect {
    // Default full clean 16:9 area; if webcam is strictly on left or right edge, keep full
    // screen so IA Diretor has maximum room (it already masks out `webcam_rect`).
    let _ = cam;
    NormalizedRect {
        x: 0.0,
        y: 0.0,
        w: 1.0,
        h: 1.0,
    }
}

fn crop_rect_for_top_slot(img: &RgbImage, rect: NormalizedRect, out_w: u32, out_h: u32) -> RgbImage {
    let fw = img.width() as f64;
    let fh = img.height() as f64;
    let rx = (rect.x * fw).floor().clamp(0.0, (fw - 4.0).max(0.0));
    let ry = (rect.y * fh).floor().clamp(0.0, (fh - 4.0).max(0.0));
    let rw = (rect.w * fw).round().clamp(4.0, (fw - rx).max(4.0));
    let rh = (rect.h * fh).round().clamp(4.0, (fh - ry).max(4.0));

    let target_aspect = out_w as f64 / out_h.max(1) as f64;
    let (cw, ch) = if rw / rh.max(1.0) >= target_aspect {
        (rh * target_aspect, rh)
    } else {
        (rw, rw / target_aspect)
    };
    let cx = (rx + (rw - cw) * 0.5).round().clamp(0.0, (fw - cw).max(0.0)) as u32;
    let cy = (ry + (rh - ch) * 0.5).round().clamp(0.0, (fh - ch).max(0.0)) as u32;
    let cw_u = (cw.round() as u32).clamp(4, img.width().saturating_sub(cx).max(1));
    let ch_u = (ch.round() as u32).clamp(4, img.height().saturating_sub(cy).max(1));

    let cropped = imageops::crop_imm(img, cx, cy, cw_u, ch_u).to_image();
    imageops::resize(&cropped, out_w, out_h, imageops::FilterType::Lanczos3)
}

fn encode_rgb_to_jpeg(img: &RgbImage) -> AicutResult<Vec<u8>> {
    let mut buf = Cursor::new(Vec::new());
    image::DynamicImage::ImageRgb8(img.clone())
        .write_to(&mut buf, image::ImageFormat::Jpeg)
        .map_err(|e| AicutError::internal(format!("Failed encoding review preview JPEG: {e}")))?;
    Ok(buf.into_inner())
}

async fn extract_single_jpeg(
    ffmpeg_bin: &Path,
    video_path: &Path,
    timestamp_sec: f64,
    max_width: u32,
    out_path: &Path,
) -> AicutResult<()> {
    let mut cmd = Command::new(ffmpeg_bin);
    configure_tokio_command(&mut cmd);
    let vf = format!("scale='min({max_width},iw)':-2:flags=lanczos");
    let out = cmd
        .args([
            "-y",
            "-ss",
            &format!("{timestamp_sec:.3}"),
            "-i",
        ])
        .arg(video_path)
        .args(["-frames:v", "1", "-vf", &vf, "-q:v", "4"])
        .arg(out_path)
        .output()
        .await
        .map_err(|e| AicutError::internal(format!("FFmpeg framing preview failed: {e}")))?;

    if !out.status.success() || !out_path.is_file() {
        let stderr = String::from_utf8_lossy(&out.stderr);
        return Err(AicutError::internal(format!(
            "FFmpeg failed extracting preview frame: {stderr}"
        )));
    }
    Ok(())
}

async fn extract_evenly_spaced_frames(
    ffmpeg_bin: &Path,
    video_path: &Path,
    duration_sec: f64,
    anchor_ts: f64,
    count: usize,
    out_dir: &Path,
) -> Vec<std::path::PathBuf> {
    let mut paths = Vec::new();
    let span = duration_sec.min(180.0).max(5.0);
    let start = (anchor_ts - span * 0.25).max(0.5);
    let end = (start + span).min((duration_sec - 0.5).max(start + 0.5));
    let n = count.max(2);
    for i in 0..n {
        let frac = i as f64 / (n - 1) as f64;
        let ts = start + (end - start) * frac;
        let p = out_dir.join(format!("auto_{i:02}.jpg"));
        if extract_single_jpeg(ffmpeg_bin, video_path, ts, 640, &p)
            .await
            .is_ok()
        {
            paths.push(p);
        }
    }
    paths
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn crop_edge_leak_detector_flags_solid_obs_border_on_cropped_webcam() {
        let mut crops = Vec::new();
        for i in 0..3usize {
            let mut img = RgbImage::from_pixel(320, 180, image::Rgb([55, 50, 62]));
            // Natural textured webcam interior
            for y in 0..180u32 {
                for x in 0..320u32 {
                    let jitter = ((i * 5 + (x as usize % 7)) % 14) as u8;
                    img.put_pixel(
                        x,
                        y,
                        image::Rgb([
                            60u8.saturating_add((x / 5) as u8).saturating_add(jitter),
                            52u8.saturating_add((y / 4) as u8),
                            68u8.saturating_add(jitter),
                        ]),
                    );
                }
            }
            // Leaked static bright gold OBS border on LEFT (x=0..4) and BOTTOM (y=176..180)
            for y in 0..180u32 {
                for x in 0..5u32 {
                    img.put_pixel(x, y, image::Rgb([245, 200, 20]));
                }
            }
            for y in 175..180u32 {
                for x in 0..320u32 {
                    img.put_pixel(x, y, image::Rgb([245, 200, 20]));
                }
            }
            crops.push(img);
        }

        let leaked = detect_crop_edge_border_leaks(&crops);
        assert!(leaked.contains(&"left".to_string()), "expected left leak, got {leaked:?}");
        assert!(leaked.contains(&"bottom".to_string()), "expected bottom leak, got {leaked:?}");
        assert!(!leaked.contains(&"top".to_string()));
        assert!(!leaked.contains(&"right".to_string()));
    }

    #[test]
    fn local_qa_flags_tight_face_zoom_and_suggests_wider_natural_rect() {
        let rect = NormalizedRect { x: 0.72, y: 0.68, w: 0.20, h: 0.22 };
        let huge_face = FaceBox { x1: 0.30, y1: 0.15, x2: 0.70, y2: 0.75, score: 0.95 }; // 60% height!
        let qa = evaluate_local_qa(rect, Some(huge_face), vec![], vec![]);
        assert_eq!(qa.zoom_assessment, "muito_perto");
        assert!(!qa.approved);
        let sug = qa.suggested_rect.expect("expected suggested wider rect");
        assert!(sug.w > rect.w && sug.h > rect.h);
    }

    #[test]
    fn local_qa_approves_clean_natural_webcam_framing() {
        let rect = NormalizedRect { x: 0.68, y: 0.62, w: 0.30, h: 0.34 };
        let natural_face = FaceBox { x1: 0.38, y1: 0.22, x2: 0.62, y2: 0.52, score: 0.96 }; // 30% height
        let qa = evaluate_local_qa(rect, Some(natural_face), vec![], vec![]);
        assert_eq!(qa.zoom_assessment, "ideal");
        assert!(qa.approved);
        assert!(qa.score >= 90);
    }
}
