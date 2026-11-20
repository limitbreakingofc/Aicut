use aicut_core::{
    configure_tokio_command, prepare_clip_audio_master_chain, AccelBackend, AicutError, AicutEvent,
    AicutKernel, AicutResult, MediaAsset, MediaKind, MediaMetadata, MediaSourceType, SmartClipPlan,
    SmartClipSegment, TranscriptRecord,
};
use chrono::Utc;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use std::path::Path;
use tokio::process::Command;
use uuid::Uuid;

use crate::face_track::{
    compute_dual_tracks, CameraSlotSpec, CameraTrackPlan, ContentSlotSpec,
};
use crate::framing::{ContentRegionMode, NormalizedRect};
use crate::template::{detect_template_geometry_from_path, resolve_template_png, TemplateGeometry};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum ShortsLayout {
    /// 9:16 vertical (1080x1920): custom/built-in branded PNG template with camera on TOP and content on BOTTOM.
    TemplateSplit,
    /// 9:16 vertical (1080x1920): borderless split with webcam on TOP and gameplay/content on BOTTOM.
    GamingSplit,
    /// 9:16 vertical (1080x1920): keeps full 16:9 frame sharp in center over blurred background.
    FitBlur,
    /// 9:16 vertical (1080x1920): full-bleed center crop ideal for vlogs, talks, and facecams.
    CenterCrop,
    /// Keeps original source aspect ratio (e.g. 16:9 horizontal clip).
    Original,
}

impl Default for ShortsLayout {
    fn default() -> Self {
        Self::TemplateSplit
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum WebcamCorner {
    /// AI automatically detects the webcam/face anywhere on screen, removes OBS borders, and tracks smoothly.
    AutoFace,
    BottomRight,
    BottomLeft,
    TopRight,
    TopLeft,
    Center,
}

impl Default for WebcamCorner {
    fn default() -> Self {
        Self::AutoFace
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum ContentFit {
    /// AI Director dynamically frames the active speaker (mouth/face tracking) or peak visual action/highlight on the bottom window.
    SmartFocus,
    /// Fills the entire bottom content box cleanly (static center crop, no black bars).
    Fill,
    /// Keeps 100% of the 16:9 screen visible inside the bottom content box over a soft blurred fill.
    FitBlur,
}

impl Default for ContentFit {
    fn default() -> Self {
        Self::SmartFocus
    }
}

#[allow(clippy::too_many_arguments)]
pub async fn render_clip_to_library(
    kernel: &AicutKernel,
    ffmpeg_bin: &Path,
    ffprobe_bin: &Path,
    source: &MediaAsset,
    clip: &SmartClipPlan,
    transcript: Option<&TranscriptRecord>,
    layout: ShortsLayout,
    webcam_corner: WebcamCorner,
    content_fit: ContentFit,
    template_path: Option<&str>,
    face_tracking: bool,
    face_zoom: f32,
    border_trim_pct: f32,
    custom_webcam_rect: Option<NormalizedRect>,
    webcam_lock_static: bool,
    custom_content_rect: Option<NormalizedRect>,
    content_region_mode: ContentRegionMode,
    burn_subtitles: bool,
) -> AicutResult<MediaAsset> {
    if clip.segments.is_empty() {
        return Err(AicutError::invalid_input("Clip has no valid segments to render."));
    }

    let clips_dir = kernel.workspace.exports_dir.join("Shorts");
    std::fs::create_dir_all(&clips_dir)?;

    let safe_slug = sanitize_filename(&clip.title);
    let short_id = &Uuid::new_v4().to_string()[..8];
    let file_name = format!("{safe_slug}-{short_id}.mp4");
    let out_path = clips_dir.join(&file_name);

    // Optional ASS karaoke subtitles re-mapped to clip-relative 00:00 timeline
    let tmp_ass = if burn_subtitles {
        if let Some(tr) = transcript {
            let ass_content = build_clip_ass_subtitles(tr, &clip.segments, layout);
            if !ass_content.is_empty() {
                let p = std::env::temp_dir().join(format!("aicut-short-sub-{short_id}.ass"));
                std::fs::write(&p, ass_content)?;
                Some(p)
            } else {
                None
            }
        } else {
            None
        }
    } else {
        None
    };

    let resolved_template = if layout == ShortsLayout::TemplateSplit {
        let tpl_path = resolve_template_png(kernel, template_path)?;
        let geom = detect_template_geometry_from_path(&tpl_path);
        Some((tpl_path, geom))
    } else {
        None
    };

    // Effective content fit respects a saved ContentRegionMode when custom_content_rect is set
    let effective_content_fit = if custom_content_rect.is_some() {
        match content_region_mode {
            ContentRegionMode::FitBlurRegion => ContentFit::FitBlur,
            ContentRegionMode::FixedRegion | ContentRegionMode::AiDirectorInRegion => {
                ContentFit::SmartFocus
            }
        }
    } else {
        content_fit
    };

    // Run unified single-pass AI tracking for BOTH top webcam (user-saved rect OR IA Cameraman)
    // and bottom content (user-saved rect OR IA Diretor: active speaker / peak visual action)
    let is_split_or_center = matches!(
        layout,
        ShortsLayout::TemplateSplit | ShortsLayout::GamingSplit | ShortsLayout::CenterCrop
    );
    let needs_face_track = is_split_or_center && (face_tracking || custom_webcam_rect.is_some());
    let needs_content_track = (matches!(
        layout,
        ShortsLayout::TemplateSplit | ShortsLayout::GamingSplit
    ) && (effective_content_fit == ContentFit::SmartFocus || custom_content_rect.is_some()))
        || (layout == ShortsLayout::CenterCrop && face_tracking);

    let mut tmp_cam_cmd_path: Option<std::path::PathBuf> = None;
    let mut tmp_content_cmd_path: Option<std::path::PathBuf> = None;
    let (camera_track_plan, content_track_plan): (Option<CameraTrackPlan>, Option<CameraTrackPlan>) =
        if needs_face_track || needs_content_track {
            let (src_w, src_h) = resolve_video_dimensions(ffprobe_bin, source).await;
            let cam_slot = if needs_face_track {
                let (slot_w, slot_h) = match layout {
                    ShortsLayout::TemplateSplit => {
                        let tc = resolved_template
                            .as_ref()
                            .map(|(_, g)| g.top_camera)
                            .unwrap_or_else(|| TemplateGeometry::default().top_camera);
                        (tc.w, tc.h)
                    }
                    ShortsLayout::GamingSplit => (1080, 730),
                    ShortsLayout::CenterCrop => (1080, 1920),
                    _ => (1080, 730),
                };
                Some(CameraSlotSpec {
                    target_w: slot_w,
                    target_h: slot_h,
                    corner_hint: webcam_corner,
                    face_zoom,
                    border_trim_pct,
                    custom_rect: custom_webcam_rect.map(|r| r.to_pixel_rect()),
                    lock_static: webcam_lock_static || !face_tracking,
                })
            } else {
                None
            };
            let content_slot = if needs_content_track {
                let (slot_w, slot_h) = match layout {
                    ShortsLayout::TemplateSplit => {
                        let bc = resolved_template
                            .as_ref()
                            .map(|(_, g)| g.bottom_content)
                            .unwrap_or_else(|| TemplateGeometry::default().bottom_content);
                        (bc.w, bc.h)
                    }
                    ShortsLayout::GamingSplit => (1080, 1190),
                    ShortsLayout::CenterCrop => (1080, 1920),
                    _ => (1080, 1190),
                };
                Some(ContentSlotSpec {
                    target_w: slot_w,
                    target_h: slot_h,
                    custom_rect: custom_content_rect.map(|r| r.to_pixel_rect()),
                    lock_static: custom_content_rect.is_some()
                        && matches!(
                            content_region_mode,
                            ContentRegionMode::FixedRegion | ContentRegionMode::FitBlurRegion
                        ),
                })
            } else {
                None
            };

            if let Ok(dual) = compute_dual_tracks(
                kernel,
                ffmpeg_bin,
                Path::new(&source.file_path),
                src_w,
                src_h,
                &clip.segments,
                cam_slot,
                content_slot,
            )
            .await
            {
                if let Some(ref cam) = dual.camera {
                    if cam.keyframes.len() > 1 {
                        let p = std::env::temp_dir().join(format!("aicut-camtrack-{short_id}.cmd"));
                        if cam.write_sendcmd_file(&p).is_ok() {
                            tmp_cam_cmd_path = Some(p);
                        }
                    }
                }
                if let Some(ref cnt) = dual.content {
                    if cnt.keyframes.len() > 1 {
                        let p =
                            std::env::temp_dir().join(format!("aicut-contenttrack-{short_id}.cmd"));
                        if cnt.write_sendcmd_file(&p).is_ok() {
                            tmp_content_cmd_path = Some(p);
                        }
                    }
                }
                (dual.camera, dual.content)
            } else {
                (None, None)
            }
        } else {
            (None, None)
        };

    // Pass 1 automatic YouTube audio mastering analysis (-14 LUFS / -1.0 dBTP + denoise + EQ + compressor)
    let span_start = clip.segments.first().map(|s| s.start_sec);
    let span_dur = clip.segments.first().map(|s| (s.end_sec - s.start_sec).max(0.5));
    let audio_master_chain = prepare_clip_audio_master_chain(
        kernel,
        ffmpeg_bin,
        Path::new(&source.file_path),
        span_start,
        span_dur,
    )
    .await;

    let hw_encoder = select_hw_h264_encoder(kernel.hardware.recommended_accel);
    let mut render_res = run_ffmpeg_render(
        ffmpeg_bin,
        Path::new(&source.file_path),
        &clip.segments,
        layout,
        webcam_corner,
        effective_content_fit,
        resolved_template
            .as_ref()
            .map(|(p, g)| (p.as_path(), *g)),
        camera_track_plan
            .as_ref()
            .map(|p| (p, tmp_cam_cmd_path.as_deref())),
        content_track_plan
            .as_ref()
            .map(|p| (p, tmp_content_cmd_path.as_deref())),
        tmp_ass.as_deref(),
        audio_master_chain.as_deref(),
        hw_encoder,
        &out_path,
    )
    .await;

    // Fallback 1: software libx264 keeping full YouTube mastered audio
    if render_res.is_err() && hw_encoder != "libx264" {
        render_res = run_ffmpeg_render(
            ffmpeg_bin,
            Path::new(&source.file_path),
            &clip.segments,
            layout,
            webcam_corner,
            effective_content_fit,
            resolved_template
                .as_ref()
                .map(|(p, g)| (p.as_path(), *g)),
            camera_track_plan
                .as_ref()
                .map(|p| (p, tmp_cam_cmd_path.as_deref())),
            content_track_plan
                .as_ref()
                .map(|p| (p, tmp_content_cmd_path.as_deref())),
            tmp_ass.as_deref(),
            audio_master_chain.as_deref(),
            "libx264",
            &out_path,
        )
        .await;
    }

    // Fallback 2: if a denoise filter isn't present in host FFmpeg, apply pure YouTube loudnorm
    if render_res.is_err() && audio_master_chain.is_some() {
        let simple_loudnorm = "loudnorm=I=-14.0:TP=-1.0:LRA=11.0,aresample=48000";
        render_res = run_ffmpeg_render(
            ffmpeg_bin,
            Path::new(&source.file_path),
            &clip.segments,
            layout,
            webcam_corner,
            effective_content_fit,
            resolved_template
                .as_ref()
                .map(|(p, g)| (p.as_path(), *g)),
            camera_track_plan
                .as_ref()
                .map(|p| (p, tmp_cam_cmd_path.as_deref())),
            content_track_plan
                .as_ref()
                .map(|p| (p, tmp_content_cmd_path.as_deref())),
            tmp_ass.as_deref(),
            Some(simple_loudnorm),
            "libx264",
            &out_path,
        )
        .await;
    }

    render_res?;

    if let Some(ref p) = tmp_ass {
        let _ = std::fs::remove_file(p);
    }
    if let Some(ref p) = tmp_cam_cmd_path {
        let _ = std::fs::remove_file(p);
    }
    if let Some(ref p) = tmp_content_cmd_path {
        let _ = std::fs::remove_file(p);
    }

    // Generate thumbnail for Library card
    let asset_id = Uuid::new_v4().to_string();
    let thumb_path = kernel.workspace.thumbnails_dir.join(format!("{asset_id}.jpg"));
    let thumb_ok = generate_thumbnail(ffmpeg_bin, &out_path, &thumb_path).await;

    let file_size_bytes = std::fs::metadata(&out_path).map(|m| m.len()).unwrap_or(0);
    let (width, height) = match layout {
        ShortsLayout::Original => (source.metadata.width, source.metadata.height),
        _ => (Some(1080), Some(1920)),
    };

    let notes = format!(
        "Hook (5s): {}\nVisual: {}\nInicio: {}\nMeio: {}\nFim: {}",
        clip.hook_first_5s,
        clip.visual_evidence,
        clip.narrative_arc.beginning,
        clip.narrative_arc.middle,
        clip.narrative_arc.end
    );

    let now = Utc::now();
    let asset = MediaAsset {
        id: asset_id,
        title: format!("{} [Short]", clip.title),
        kind: MediaKind::Video,
        source_type: MediaSourceType::Generated,
        source_url: source.source_url.clone(),
        parent_media_id: Some(source.id.clone()),
        file_path: out_path.to_string_lossy().to_string(),
        file_name,
        extension: "mp4".to_string(),
        file_size_bytes,
        fingerprint: None,
        thumbnail_path: if thumb_ok {
            Some(thumb_path.to_string_lossy().to_string())
        } else {
            None
        },
        metadata: MediaMetadata {
            duration_sec: Some(clip.total_duration_sec),
            width,
            height,
            fps: source.metadata.fps.or(Some(30.0)),
            video_codec: Some("h264".to_string()),
            audio_codec: Some("aac".to_string()),
            format_name: Some("mp4".to_string()),
            transcript_id: transcript.map(|t| t.id.clone()),
            ..Default::default()
        },
        tags: vec![
            "short".to_string(),
            "ai-clip".to_string(),
            format!("{:?}", layout).to_lowercase(),
            format!("video_id:{}", source.id),
        ],
        notes,
        created_at: now,
        updated_at: now,
        deleted_at: None,
    };

    kernel.repo.insert_media(&asset)?;
    kernel.events.publish(AicutEvent::MediaCreated(asset.clone()));
    Ok(asset)
}

pub fn select_hw_h264_encoder(accel: AccelBackend) -> &'static str {
    match accel {
        AccelBackend::CudaNvenc => "h264_nvenc",
        AccelBackend::QsvOpenvino => "h264_qsv",
        AccelBackend::AmfDirectml | AccelBackend::Metal | AccelBackend::CpuOnly => "libx264",
    }
}

async fn resolve_video_dimensions(ffprobe_bin: &Path, source: &MediaAsset) -> (u32, u32) {
    if let (Some(w), Some(h)) = (source.metadata.width, source.metadata.height) {
        if w >= 64 && h >= 64 {
            return (w, h);
        }
    }
    let mut cmd = Command::new(ffprobe_bin);
    configure_tokio_command(&mut cmd);
    if let Ok(out) = cmd
        .args([
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height",
            "-of",
            "csv=s=x:p=0",
        ])
        .arg(&source.file_path)
        .output()
        .await
    {
        if out.status.success() {
            let txt = String::from_utf8_lossy(&out.stdout);
            if let Some((ws, hs)) = txt.trim().split_once('x') {
                if let (Ok(w), Ok(h)) = (ws.parse::<u32>(), hs.parse::<u32>()) {
                    if w >= 64 && h >= 64 {
                        return (w, h);
                    }
                }
            }
        }
    }
    (1920, 1080)
}

fn webcam_crop_expr(corner: WebcamCorner) -> &'static str {
    match corner {
        WebcamCorner::AutoFace | WebcamCorner::BottomRight => {
            "crop=iw*0.30:ih*0.34:iw*0.68:ih*0.64"
        }
        WebcamCorner::BottomLeft => "crop=iw*0.30:ih*0.34:iw*0.02:ih*0.64",
        WebcamCorner::TopRight => "crop=iw*0.30:ih*0.34:iw*0.68:ih*0.02",
        WebcamCorner::TopLeft => "crop=iw*0.30:ih*0.34:iw*0.02:ih*0.02",
        WebcamCorner::Center => "crop=iw*0.60:ih*0.75:iw*0.20:ih*0.08",
    }
}

fn build_tracked_crop_chain(
    input_label: &str,
    output_label: &str,
    target_w: u32,
    target_h: u32,
    plan: &CameraTrackPlan,
    cmd_path_opt: Option<&Path>,
) -> String {
    if let Some(cmd_path) = cmd_path_opt {
        let escaped = escape_ffmpeg_filter_path(cmd_path);
        let filter_name = if plan.filter_target.is_empty() {
            "crop@cam"
        } else {
            plan.filter_target.as_str()
        };
        return format!(
            "{input_label}sendcmd=f='{escaped}',{filter_name}=w={}:h={}:x={}:y={},scale={target_w}:{target_h}:flags=lanczos,setsar=1{output_label}",
            plan.crop_w, plan.crop_h, plan.init_x, plan.init_y
        );
    }
    format!(
        "{input_label}crop=w={}:h={}:x={}:y={},scale={target_w}:{target_h}:flags=lanczos,setsar=1{output_label}",
        plan.crop_w, plan.crop_h, plan.init_x, plan.init_y
    )
}

fn build_camera_filter_chain(
    input_label: &str,
    output_label: &str,
    target_w: u32,
    target_h: u32,
    webcam_corner: WebcamCorner,
    camera_track: Option<(&CameraTrackPlan, Option<&Path>)>,
) -> String {
    if let Some((plan, cmd_path_opt)) = camera_track {
        return build_tracked_crop_chain(
            input_label,
            output_label,
            target_w,
            target_h,
            plan,
            cmd_path_opt,
        );
    }
    let cam_crop = webcam_crop_expr(webcam_corner);
    format!(
        "{input_label}{cam_crop},scale={target_w}:{target_h}:force_original_aspect_ratio=increase,crop={target_w}:{target_h},setsar=1{output_label}"
    )
}

fn push_content_filter_chain(
    filter_parts: &mut Vec<String>,
    input_label: &str,
    output_label: &str,
    prefix: &str,
    target_w: u32,
    target_h: u32,
    content_fit: ContentFit,
    content_track: Option<(&CameraTrackPlan, Option<&Path>)>,
) {
    match content_fit {
        ContentFit::SmartFocus => {
            if let Some((plan, cmd_path_opt)) = content_track {
                filter_parts.push(build_tracked_crop_chain(
                    input_label,
                    output_label,
                    target_w,
                    target_h,
                    plan,
                    cmd_path_opt,
                ));
            } else {
                filter_parts.push(format!(
                    "{input_label}scale={target_w}:{target_h}:force_original_aspect_ratio=increase,crop={target_w}:{target_h},setsar=1{output_label}"
                ));
            }
        }
        ContentFit::Fill => {
            filter_parts.push(format!(
                "{input_label}scale={target_w}:{target_h}:force_original_aspect_ratio=increase,crop={target_w}:{target_h},setsar=1{output_label}"
            ));
        }
        ContentFit::FitBlur => {
            filter_parts.push(format!(
                "{input_label}split=2[{prefix}_bg_in][{prefix}_fg_in]"
            ));
            filter_parts.push(format!(
                "[{prefix}_bg_in]scale={target_w}:{target_h}:force_original_aspect_ratio=increase,crop={target_w}:{target_h},boxblur=18:4[{prefix}_bg]"
            ));
            filter_parts.push(format!(
                "[{prefix}_fg_in]scale={target_w}:{target_h}:force_original_aspect_ratio=decrease[{prefix}_fg]"
            ));
            filter_parts.push(format!(
                "[{prefix}_bg][{prefix}_fg]overlay=(W-w)/2:(H-h)/2,setsar=1{output_label}"
            ));
        }
    }
}

#[allow(clippy::too_many_arguments)]
async fn run_ffmpeg_render(
    ffmpeg_bin: &Path,
    input_path: &Path,
    segments: &[SmartClipSegment],
    layout: ShortsLayout,
    webcam_corner: WebcamCorner,
    content_fit: ContentFit,
    template: Option<(&Path, TemplateGeometry)>,
    camera_track: Option<(&CameraTrackPlan, Option<&Path>)>,
    content_track: Option<(&CameraTrackPlan, Option<&Path>)>,
    ass_path: Option<&Path>,
    audio_master_chain: Option<&str>,
    encoder: &str,
    out_path: &Path,
) -> AicutResult<()> {
    let mut cmd = Command::new(ffmpeg_bin);
    configure_tokio_command(&mut cmd);

    let mut filter_parts: Vec<String> = Vec::new();
    let has_master = audio_master_chain
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .is_some();

    if segments.len() == 1 {
        let seg = &segments[0];
        let dur = (seg.end_sec - seg.start_sec).max(0.5);
        cmd.args([
            "-y",
            "-ss",
            &format!("{:.3}", seg.start_sec),
            "-t",
            &format!("{dur:.3}"),
            "-i",
        ])
        .arg(input_path);
        filter_parts.push("[0:v]setpts=PTS-STARTPTS[vbase]".to_string());
        if let Some(af) = audio_master_chain.map(str::trim).filter(|s| !s.is_empty()) {
            filter_parts.push(format!("[0:a]asetpts=PTS-STARTPTS,{af}[abase]"));
        } else {
            filter_parts.push("[0:a]asetpts=PTS-STARTPTS[abase]".to_string());
        }
    } else {
        cmd.args(["-y", "-i"]).arg(input_path);
        let mut concat_inputs = String::new();
        for (i, seg) in segments.iter().enumerate() {
            let dur = (seg.end_sec - seg.start_sec).max(0.2);
            let fade_out_start = (dur - 0.03).max(0.0);
            filter_parts.push(format!(
                "[0:v]trim=start={:.3}:end={:.3},setpts=PTS-STARTPTS[v{i}]",
                seg.start_sec, seg.end_sec
            ));
            filter_parts.push(format!(
                "[0:a]atrim=start={:.3}:end={:.3},asetpts=PTS-STARTPTS,afade=t=in:st=0:d=0.025,afade=t=out:st={fade_out_start:.3}:d=0.025[a{i}]",
                seg.start_sec, seg.end_sec
            ));
            concat_inputs.push_str(&format!("[v{i}][a{i}]"));
        }
        if has_master {
            let af = audio_master_chain.unwrap().trim();
            filter_parts.push(format!(
                "{concat_inputs}concat=n={}:v=1:a=1[vbase][araw]",
                segments.len()
            ));
            filter_parts.push(format!("[araw]{af}[abase]"));
        } else {
            filter_parts.push(format!(
                "{concat_inputs}concat=n={}:v=1:a=1[vbase][abase]",
                segments.len()
            ));
        }
    }

    // Add PNG template as input 1 when using TemplateSplit
    if let Some((tpl_path, _)) = template {
        cmd.arg("-i").arg(tpl_path);
    }

    // Apply layout transform onto [vbase] -> [vlayout]
    match layout {
        ShortsLayout::TemplateSplit => {
            let geom = template.map(|(_, g)| g).unwrap_or_default();
            let tc = geom.top_camera;
            let bc = geom.bottom_content;

            filter_parts.push("[vbase]split=3[canvas_src][cam_src][content_src]".to_string());
            // Derive canvas from video stream so FPS & EOF always match source cleanly
            filter_parts.push(
                "[canvas_src]scale=1080:1920,drawbox=t=fill:c=black,setsar=1[canvas]".to_string(),
            );
            // Top box: CAMERA (AI Face Tracked + Zero-Border Safe Box when enabled)
            filter_parts.push(build_camera_filter_chain(
                "[cam_src]",
                "[cam_box]",
                tc.w,
                tc.h,
                webcam_corner,
                camera_track,
            ));
            // Bottom box: CONTENT (AI Director framing active speaker / peak action when SmartFocus)
            push_content_filter_chain(
                &mut filter_parts,
                "[content_src]",
                "[content_box]",
                "cb",
                bc.w,
                bc.h,
                content_fit,
                content_track,
            );
            // Stack camera on top slot, content on bottom slot, and PNG template overlay on top
            filter_parts.push(format!("[canvas][cam_box]overlay={}:{}[tmp1]", tc.x, tc.y));
            filter_parts.push(format!("[tmp1][content_box]overlay={}:{}[tmp2]", bc.x, bc.y));
            filter_parts.push("[1:v]scale=1080:1920,format=rgba[tpl]".to_string());
            filter_parts.push("[tmp2][tpl]overlay=0:0:format=auto,setsar=1[vlayout]".to_string());
        }
        ShortsLayout::Original => {
            filter_parts.push("[vbase]scale=trunc(iw/2)*2:trunc(ih/2)*2[vlayout]".to_string());
        }
        ShortsLayout::CenterCrop => {
            if let Some((plan, cmd_path_opt)) = camera_track.or(content_track) {
                filter_parts.push(build_tracked_crop_chain(
                    "[vbase]",
                    "[vlayout]",
                    1080,
                    1920,
                    plan,
                    cmd_path_opt,
                ));
            } else {
                filter_parts.push(
                    "[vbase]scale=1080:1920:force_original_aspect_ratio=increase,crop=1080:1920,setsar=1[vlayout]"
                        .to_string(),
                );
            }
        }
        ShortsLayout::FitBlur => {
            filter_parts.push("[vbase]split=2[bg_in][fg_in]".to_string());
            filter_parts.push(
                "[bg_in]scale=1080:1920:force_original_aspect_ratio=increase,crop=1080:1920,boxblur=24:6[bg]"
                    .to_string(),
            );
            filter_parts.push(
                "[fg_in]scale=1080:1920:force_original_aspect_ratio=decrease[fg]".to_string(),
            );
            filter_parts.push("[bg][fg]overlay=(W-w)/2:(H-h)/2,setsar=1[vlayout]".to_string());
        }
        ShortsLayout::GamingSplit => {
            // Camera on TOP (1080x730, AI Face Tracked + Zero-Border) + Content on BOTTOM (1080x1190, AI Director)
            filter_parts.push("[vbase]split=2[cam_in][bot_in]".to_string());
            filter_parts.push(build_camera_filter_chain(
                "[cam_in]",
                "[cam]",
                1080,
                730,
                webcam_corner,
                camera_track,
            ));
            push_content_filter_chain(
                &mut filter_parts,
                "[bot_in]",
                "[bot]",
                "gb",
                1080,
                1190,
                content_fit,
                content_track,
            );
            filter_parts.push("[cam][bot]vstack=inputs=2,setsar=1[vlayout]".to_string());
        }
    }

    // Optional burned-in Shorts karaoke subtitles
    let final_v_label = if let Some(ass) = ass_path {
        let escaped = escape_ffmpeg_filter_path(ass);
        filter_parts.push(format!("[vlayout]ass='{escaped}'[vout]"));
        "[vout]"
    } else {
        "[vlayout]"
    };

    let filter_complex = filter_parts.join(";");
    cmd.args(["-filter_complex", &filter_complex, "-map", final_v_label, "-map", "[abase]"]);

    // Encoder flags tuned per backend
    match encoder {
        "h264_nvenc" => {
            cmd.args(["-c:v", "h264_nvenc", "-preset", "p5", "-cq", "20", "-pix_fmt", "yuv420p"]);
        }
        "h264_amf" => {
            cmd.args(["-c:v", "h264_amf", "-quality", "quality", "-pix_fmt", "yuv420p"]);
        }
        "h264_qsv" => {
            cmd.args(["-c:v", "h264_qsv", "-global_quality", "20", "-pix_fmt", "yuv420p"]);
        }
        "h264_videotoolbox" => {
            cmd.args(["-c:v", "h264_videotoolbox", "-q:v", "65", "-pix_fmt", "yuv420p"]);
        }
        _ => {
            cmd.args(["-c:v", "libx264", "-preset", "fast", "-crf", "20", "-pix_fmt", "yuv420p"]);
        }
    }

    cmd.args([
        "-c:a",
        "aac",
        "-b:a",
        "160k",
        "-ar",
        "48000",
        "-movflags",
        "+faststart",
    ])
    .arg(out_path);

    let out = cmd
        .output()
        .await
        .map_err(|e| AicutError::internal(format!("Failed running FFmpeg render: {e}")))?;

    if !out.status.success() || !out_path.is_file() {
        let stderr = String::from_utf8_lossy(&out.stderr);
        return Err(AicutError::new(
            "SHORT_RENDER_FAILED",
            format!("FFmpeg ({encoder}) falhou ao renderizar corte: {stderr}"),
        ));
    }
    Ok(())
}

async fn generate_thumbnail(ffmpeg_bin: &Path, video_path: &Path, thumb_path: &Path) -> bool {
    if let Some(parent) = thumb_path.parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    let mut cmd = Command::new(ffmpeg_bin);
    configure_tokio_command(&mut cmd);
    cmd.args(["-y", "-ss", "1.0", "-i"])
        .arg(video_path)
        .args(["-frames:v", "1", "-vf", "scale=360:-2", "-q:v", "4"])
        .arg(thumb_path)
        .output()
        .await
        .map(|o| o.status.success() && thumb_path.is_file())
        .unwrap_or(false)
}

/// Escapes a filesystem path for FFmpeg's `ass='...'` filter on Windows and Unix.
pub fn escape_ffmpeg_filter_path(path: &Path) -> String {
    let normalized = path.to_string_lossy().replace('\\', "/");
    normalized
        .replace(':', "\\:")
        .replace('\'', "\\'")
}

/// Maps source transcript segments/words onto the clip's relative 00:00 timeline and emits
/// Shorts-safe ASS karaoke subtitles.
pub fn build_clip_ass_subtitles(
    transcript: &TranscriptRecord,
    segments: &[SmartClipSegment],
    layout: ShortsLayout,
) -> String {
    let (play_x, play_y, font_size, margin_v) = match layout {
        ShortsLayout::Original => (1920, 1080, 56, 70),
        _ => (1080, 1920, 72, 340),
    };

    let mut dialogues: Vec<String> = Vec::new();
    let mut clip_offset = 0.0;

    for seg in segments {
        let seg_dur = (seg.end_sec - seg.start_sec).max(0.0);
        for tseg in &transcript.segments {
            let ts = tseg.start_ms as f64 / 1000.0;
            let te = tseg.end_ms as f64 / 1000.0;
            if te <= seg.start_sec || ts >= seg.end_sec {
                continue;
            }
            let rel_s = (ts.max(seg.start_sec) - seg.start_sec) + clip_offset;
            let rel_e = (te.min(seg.end_sec) - seg.start_sec) + clip_offset;
            if rel_e - rel_s < 0.15 {
                continue;
            }
            let text = if !tseg.words.is_empty() {
                let mut line = String::new();
                for w in &tseg.words {
                    let ws = w.start_ms as f64 / 1000.0;
                    let we = w.end_ms as f64 / 1000.0;
                    if we <= seg.start_sec || ws >= seg.end_sec {
                        continue;
                    }
                    let cs = (((we.min(seg.end_sec) - ws.max(seg.start_sec)).max(0.05)) * 100.0)
                        .round() as u32;
                    let clean = w.text.trim().replace('{', "(").replace('}', ")");
                    if !clean.is_empty() {
                        line.push_str(&format!("{{\\kf{cs}}}{clean} "));
                    }
                }
                line.trim().to_string()
            } else {
                tseg.text.trim().replace('\n', " ")
            };

            if !text.is_empty() {
                dialogues.push(format!(
                    "Dialogue: 0,{},{},Shorts,,0,0,0,,{}",
                    format_ass_time(rel_s),
                    format_ass_time(rel_e),
                    text
                ));
            }
        }
        clip_offset += seg_dur;
    }

    if dialogues.is_empty() {
        return String::new();
    }

    format!(
        "[Script Info]\nScriptType: v4.00+\nPlayResX: {play_x}\nPlayResY: {play_y}\nWrapStyle: 1\n\n[V4+ Styles]\nFormat: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding\nStyle: Shorts,Arial,{font_size},&H0000F2FF,&H00FFFFFF,&H00000000,&H80000000,-1,0,0,0,100,100,0,0,1,5,2,2,60,60,{margin_v},1\n\n[Events]\nFormat: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n{}\n",
        dialogues.join("\n")
    )
}

fn format_ass_time(sec: f64) -> String {
    let total_cs = (sec.max(0.0) * 100.0).round() as u64;
    let cs = total_cs % 100;
    let total_s = total_cs / 100;
    let s = total_s % 60;
    let m = (total_s / 60) % 60;
    let h = total_s / 3600;
    format!("{h}:{m:02}:{s:02}.{cs:02}")
}

fn sanitize_filename(raw: &str) -> String {
    let cleaned: String = raw
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '-' || c == '_' {
                c
            } else if c.is_whitespace() {
                '-'
            } else {
                '_'
            }
        })
        .collect();
    let trimmed = cleaned.trim_matches(|c| c == '-' || c == '_');
    let truncated: String = trimmed.chars().take(48).collect();
    if truncated.is_empty() {
        "aicut-short".to_string()
    } else {
        truncated
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    #[test]
    fn escapes_windows_ass_filter_path() {
        let p = PathBuf::from(r"C:\Users\limit\AppData\Local\Temp\sub.ass");
        assert_eq!(
            escape_ffmpeg_filter_path(&p),
            "C\\:/Users/limit/AppData/Local/Temp/sub.ass"
        );
    }
}
