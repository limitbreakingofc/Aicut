use aicut_core::{
    configure_tokio_command, AicutError, AicutEvent, AicutKernel, AicutResult, MediaAsset,
    MediaKind, MediaMetadata, MediaSourceType, SmartClipSegment, TranscriptRecord,
};
use aicut_module_smart_clips::framing::NormalizedRect;
use aicut_module_smart_clips::renderer::{
    build_clip_ass_subtitles, escape_ffmpeg_filter_path, select_hw_h264_encoder, ShortsLayout,
};
use aicut_module_smart_clips::template::{
    detect_template_geometry_from_path, resolve_template_png, TemplateGeometry,
};
use chrono::Utc;
use std::path::Path;
use tokio::process::Command;
use uuid::Uuid;

use crate::schema::{AicutComposition, MotionScene, PipCorner, SceneLayoutPreset};

/// Renders a full dynamic multi-scene `AicutComposition` via FFmpeg NVENC (with automatic
/// libx264 fallback) and saves the resulting 9:16 MP4 directly to the Aicut Library.
pub async fn render_composition_to_library(
    kernel: &AicutKernel,
    comp: &AicutComposition,
) -> AicutResult<MediaAsset> {
    let source = kernel
        .repo
        .get_media(&comp.media_id)?
        .ok_or_else(|| AicutError::not_found(format!("Media '{}' not found", comp.media_id)))?;
    let ffmpeg_bin = kernel.tools.resolve_tool("ffmpeg")?;

    let mut normalized = comp.clone();
    normalized.normalize();

    let clips_dir = kernel.workspace.exports_dir.join("Shorts");
    std::fs::create_dir_all(&clips_dir)?;
    let slug = sanitize_filename(&normalized.title);
    let short_id = &Uuid::new_v4().to_string()[..8];
    let file_name = format!("{slug}-motion-{short_id}.mp4");
    let out_path = clips_dir.join(&file_name);

    let needs_template = normalized
        .scenes
        .iter()
        .any(|s| s.layout == SceneLayoutPreset::SplitBranded);
    let resolved_template = if needs_template {
        let p = resolve_template_png(kernel, normalized.template_path.as_deref())?;
        let g = detect_template_geometry_from_path(&p);
        Some((p, g))
    } else {
        None
    };

    let transcript = if normalized.burn_subtitles {
        kernel
            .repo
            .find_best_transcript_for_media(&source.id)
            .ok()
            .flatten()
            .map(|rt| rt.transcript)
    } else {
        None
    };

    // Unified kinetic ASS track: animated scene headlines + badges + callout labels + karaoke subs
    let ass_content = build_motion_ass_track(&normalized, transcript.as_ref());
    let tmp_ass = if !ass_content.is_empty() {
        let p = std::env::temp_dir().join(format!("aicut-motion-ovl-{short_id}.ass"));
        std::fs::write(&p, ass_content)?;
        Some(p)
    } else {
        None
    };

    let hw_encoder = select_hw_h264_encoder(kernel.hardware.recommended_accel);
    let first = run_ffmpeg_motion_render(
        &ffmpeg_bin,
        Path::new(&source.file_path),
        &normalized,
        resolved_template.as_ref().map(|(p, g)| (p.as_path(), *g)),
        tmp_ass.as_deref(),
        hw_encoder,
        &out_path,
    )
    .await;

    if first.is_err() && hw_encoder != "libx264" {
        run_ffmpeg_motion_render(
            &ffmpeg_bin,
            Path::new(&source.file_path),
            &normalized,
            resolved_template.as_ref().map(|(p, g)| (p.as_path(), *g)),
            tmp_ass.as_deref(),
            "libx264",
            &out_path,
        )
        .await?;
    } else {
        first?;
    }

    if let Some(ref p) = tmp_ass {
        let _ = std::fs::remove_file(p);
    }

    // Thumbnail for Library card
    let asset_id = Uuid::new_v4().to_string();
    let thumb_path = kernel.workspace.thumbnails_dir.join(format!("{asset_id}.jpg"));
    let thumb_ok = generate_thumb(&ffmpeg_bin, &out_path, &thumb_path).await;
    let file_size_bytes = std::fs::metadata(&out_path).map(|m| m.len()).unwrap_or(0);

    let now = Utc::now();
    let asset = MediaAsset {
        id: asset_id,
        title: format!("{} [Motion Short]", normalized.title),
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
            duration_sec: Some(normalized.duration_sec),
            width: Some(normalized.width),
            height: Some(normalized.height),
            fps: Some(normalized.fps as f64),
            video_codec: Some("h264".to_string()),
            audio_codec: Some("aac".to_string()),
            format_name: Some("mp4".to_string()),
            transcript_id: transcript.map(|t| t.id),
            ..Default::default()
        },
        tags: vec![
            "short".to_string(),
            "motion-studio".to_string(),
            format!("video_id:{}", source.id),
            format!("scenes:{}", normalized.scenes.len()),
        ],
        notes: format!(
            "Motion Studio ({} cenas dinâmicas) · {:.1}s–{:.1}s",
            normalized.scenes.len(),
            normalized.clip_start_sec,
            normalized.clip_end_sec
        ),
        created_at: now,
        updated_at: now,
        deleted_at: None,
    };

    kernel.repo.insert_media(&asset)?;
    kernel.events.publish(AicutEvent::MediaCreated(asset.clone()));
    Ok(asset)
}

/// Builds the complete multi-scene FFmpeg filtergraph string for `comp`.
pub fn build_motion_filtergraph(
    comp: &AicutComposition,
    template_geom: Option<TemplateGeometry>,
    ass_path: Option<&Path>,
) -> String {
    let mut parts: Vec<String> = Vec::new();
    let fps = comp.fps.max(1);
    let n = comp.scenes.len();

    parts.push(format!("[0:v]setpts=PTS-STARTPTS,fps={fps}[vbase]"));
    parts.push("[0:a]asetpts=PTS-STARTPTS[abase]".to_string());

    if n > 1 {
        let outs: String = (0..n).map(|i| format!("[vs{i}]")).collect();
        parts.push(format!("[vbase]split={n}{outs}"));
    }

    let branded_count = comp
        .scenes
        .iter()
        .filter(|s| s.layout == SceneLayoutPreset::SplitBranded)
        .count();
    if branded_count > 0 {
        if branded_count == 1 {
            parts.push("[1:v]scale=1080:1920,format=rgba[tpl0]".to_string());
        } else {
            let touts: String = (0..branded_count).map(|i| format!("[tpl{i}]")).collect();
            parts.push(format!("[1:v]scale=1080:1920,format=rgba,split={branded_count}{touts}"));
        }
    }

    let geom = template_geom.unwrap_or_default();
    let mut tpl_idx = 0usize;

    for (i, sc) in comp.scenes.iter().enumerate() {
        let in_lbl = if n == 1 {
            "[vbase]".to_string()
        } else {
            format!("[vs{i}]")
        };
        parts.push(format!(
            "{in_lbl}trim=start={:.3}:end={:.3},setpts=PTS-STARTPTS[raw{i}]",
            sc.start_sec, sc.end_sec
        ));

        let cam_crop = norm_rect_crop_expr(comp.webcam_rect);
        let content_crop = build_content_crop_expr(comp.content_rect, sc);

        match sc.layout {
            SceneLayoutPreset::SplitBranded => {
                let tc = geom.top_camera;
                let bc = geom.bottom_content;
                let t_lbl = format!("[tpl{tpl_idx}]");
                tpl_idx += 1;
                parts.push(format!("[raw{i}]split=3[cv{i}][cm{i}][ct{i}]"));
                parts.push(format!(
                    "[cv{i}]scale=1080:1920,drawbox=t=fill:c=black,setsar=1[bg{i}]"
                ));
                parts.push(format!(
                    "[cm{i}]{cam_crop},scale={}:{}:force_original_aspect_ratio=increase:flags=lanczos,crop={}:{},setsar=1[cam{i}]",
                    tc.w, tc.h, tc.w, tc.h
                ));
                push_content_box(&mut parts, i, &content_crop, bc.w, bc.h, sc.content_focus.fit_blur);
                parts.push(format!("[bg{i}][cam{i}]overlay={}:{}[o1_{i}]", tc.x, tc.y));
                parts.push(format!("[o1_{i}][cnt{i}]overlay={}:{}[o2_{i}]", bc.x, bc.y));
                parts.push(format!(
                    "[o2_{i}]{t_lbl}overlay=0:0:format=auto,setsar=1,format=yuv420p[sc{i}]"
                ));
            }
            SceneLayoutPreset::SplitClean => {
                parts.push(format!("[raw{i}]split=2[cm{i}][ct{i}]"));
                parts.push(format!(
                    "[cm{i}]{cam_crop},scale=1080:730:force_original_aspect_ratio=increase:flags=lanczos,crop=1080:730,setsar=1[cam{i}]"
                ));
                push_content_box(&mut parts, i, &content_crop, 1080, 1190, sc.content_focus.fit_blur);
                parts.push(format!(
                    "[cam{i}][cnt{i}]vstack=inputs=2,setsar=1,format=yuv420p[sc{i}]"
                ));
            }
            SceneLayoutPreset::CameraHeroPip => {
                parts.push(format!("[raw{i}]split=2[cm{i}][ct{i}]"));
                parts.push(format!(
                    "[cm{i}]{cam_crop},scale=1080:1260:force_original_aspect_ratio=increase:flags=lanczos,crop=1080:1260,setsar=1[cam{i}]"
                ));
                push_content_box(&mut parts, i, &content_crop, 1080, 660, sc.content_focus.fit_blur);
                parts.push(format!(
                    "[cam{i}][cnt{i}]vstack=inputs=2,setsar=1,format=yuv420p[sc{i}]"
                ));
            }
            SceneLayoutPreset::CameraFocus => {
                parts.push(format!(
                    "[raw{i}]{cam_crop},scale=1080:1920:force_original_aspect_ratio=increase:flags=lanczos,crop=1080:1920,setsar=1,format=yuv420p[sc{i}]"
                ));
            }
            SceneLayoutPreset::ContentFullscreen => {
                parts.push(format!("[raw{i}]null[ct{i}]"));
                push_content_box(&mut parts, i, &content_crop, 1080, 1920, sc.content_focus.fit_blur);
                parts.push(format!("[cnt{i}]setsar=1,format=yuv420p[sc{i}]"));
            }
            SceneLayoutPreset::ContentHeroPip | SceneLayoutPreset::CalloutZoom => {
                let pip_w = (((1080.0 * sc.pip.width_pct).round() as u32) & !1).clamp(220, 520);
                let pip_h = (((pip_w as f64 / 1.75).round() as u32) & !1).clamp(130, 320);
                let border_hex = clean_ffmpeg_hex(&sc.pip.border_hex);
                let (ox, oy) = match sc.pip.corner {
                    PipCorner::TopRight => ("W-w-36", "110"),
                    PipCorner::TopLeft => ("36", "110"),
                    PipCorner::BottomRight => ("W-w-36", "H-h-280"),
                    PipCorner::BottomLeft => ("36", "H-h-280"),
                };
                if sc.pip.visible {
                    parts.push(format!("[raw{i}]split=2[ct{i}][cm{i}]"));
                    push_content_box(&mut parts, i, &content_crop, 1080, 1920, sc.content_focus.fit_blur);
                    let cnt_with_ring = if sc.layout == SceneLayoutPreset::CalloutZoom {
                        let ring_hex = clean_ffmpeg_hex(&sc.overlay.accent_hex);
                        parts.push(format!(
                            "[cnt{i}]drawbox=x=iw*0.26:y=ih*0.34:w=iw*0.48:h=ih*0.26:color={ring_hex}@0.92:t=8[cntr{i}]"
                        ));
                        format!("[cntr{i}]")
                    } else {
                        format!("[cnt{i}]")
                    };
                    parts.push(format!(
                        "[cm{i}]{cam_crop},scale={pip_w}:{pip_h}:force_original_aspect_ratio=increase:flags=lanczos,crop={pip_w}:{pip_h},pad={}:{}:6:6:color={border_hex},setsar=1[pip{i}]",
                        pip_w + 12,
                        pip_h + 12
                    ));
                    parts.push(format!(
                        "{cnt_with_ring}[pip{i}]overlay={ox}:{oy},setsar=1,format=yuv420p[sc{i}]"
                    ));
                } else {
                    parts.push(format!("[raw{i}]null[ct{i}]"));
                    push_content_box(&mut parts, i, &content_crop, 1080, 1920, sc.content_focus.fit_blur);
                    parts.push(format!("[cnt{i}]setsar=1,format=yuv420p[sc{i}]"));
                }
            }
        }
    }

    if n == 1 {
        parts.push("[sc0]null[vlayout]".to_string());
    } else {
        let sc_ins: String = (0..n).map(|i| format!("[sc{i}]")).collect();
        parts.push(format!("{sc_ins}concat=n={n}:v=1:a=0[vlayout]"));
    }

    if let Some(ass) = ass_path {
        let escaped = escape_ffmpeg_filter_path(ass);
        parts.push(format!("[vlayout]ass='{escaped}'[vout]"));
    } else {
        parts.push("[vlayout]null[vout]".to_string());
    }

    parts.join(";")
}

fn push_content_box(
    parts: &mut Vec<String>,
    i: usize,
    content_crop: &str,
    w: u32,
    h: u32,
    fit_blur: bool,
) {
    if fit_blur {
        parts.push(format!("[ct{i}]{content_crop},split=2[cb_bg_i{i}][cb_fg_i{i}]"));
        parts.push(format!(
            "[cb_bg_i{i}]scale={w}:{h}:force_original_aspect_ratio=increase,crop={w}:{h},boxblur=20:5[cb_bg{i}]"
        ));
        parts.push(format!(
            "[cb_fg_i{i}]scale={w}:{h}:force_original_aspect_ratio=decrease:flags=lanczos[cb_fg{i}]"
        ));
        parts.push(format!(
            "[cb_bg{i}][cb_fg{i}]overlay=(W-w)/2:(H-h)/2,setsar=1[cnt{i}]"
        ));
    } else {
        parts.push(format!(
            "[ct{i}]{content_crop},scale={w}:{h}:force_original_aspect_ratio=increase:flags=lanczos,crop={w}:{h},setsar=1[cnt{i}]"
        ));
    }
}

fn norm_rect_crop_expr(r: NormalizedRect) -> String {
    let c = r.clamped();
    format!(
        "crop=w=iw*{:.4}:h=ih*{:.4}:x=iw*{:.4}:y=ih*{:.4}",
        c.w, c.h, c.x, c.y
    )
}

fn build_content_crop_expr(base_rect: Option<NormalizedRect>, sc: &MotionScene) -> String {
    let base = base_rect
        .map(NormalizedRect::clamped)
        .unwrap_or(NormalizedRect {
            x: 0.0,
            y: 0.0,
            w: 1.0,
            h: 1.0,
        });
    let zoom = sc.content_focus.zoom.clamp(1.0, 3.0);
    if zoom <= 1.02 {
        return norm_rect_crop_expr(base);
    }
    let zw = (base.w / zoom).clamp(0.15, base.w);
    let zh = (base.h / zoom).clamp(0.15, base.h);
    let fx = base.x + base.w * sc.content_focus.focus_x.clamp(0.05, 0.95);
    let fy = base.y + base.h * sc.content_focus.focus_y.clamp(0.05, 0.95);
    let x = (fx - zw * 0.5).clamp(base.x, (base.x + base.w - zw).max(base.x));
    let y = (fy - zh * 0.5).clamp(base.y, (base.y + base.h - zh).max(base.y));
    norm_rect_crop_expr(NormalizedRect { x, y, w: zw, h: zh })
}

fn clean_ffmpeg_hex(hex: &str) -> String {
    let h = hex.trim().trim_start_matches('#');
    if h.len() == 6 && h.chars().all(|c| c.is_ascii_hexdigit()) {
        format!("0x{h}")
    } else {
        "0xEAB308".to_string()
    }
}

/// Generates unified kinetic ASS overlay track (animated scene headlines + callout badges + karaoke subtitles).
pub fn build_motion_ass_track(comp: &AicutComposition, transcript: Option<&TranscriptRecord>) -> String {
    let mut events: Vec<String> = Vec::new();

    // 1. Scene kinetic headlines, badges, and callouts
    for sc in &comp.scenes {
        let s_ts = format_ass_time(sc.start_sec);
        let e_ts = format_ass_time(sc.end_sec);
        if let Some(ref hl) = sc.overlay.headline.as_deref().map(str::trim).filter(|s| !s.is_empty()) {
            let safe = hl.replace('{', "(").replace('}', ")").replace('\n', " ");
            events.push(format!(
                "Dialogue: 2,{s_ts},{e_ts},Headline,,0,0,0,,{{\\fad(160,160)\\fscx85\\fscy85\\t(0,220,\\fscx100\\fscy100)}}{safe}"
            ));
        }
        if let Some(ref badge) = sc.overlay.badge.as_deref().map(str::trim).filter(|s| !s.is_empty()) {
            let safe = badge.replace('{', "(").replace('}', ")").to_uppercase();
            events.push(format!(
                "Dialogue: 2,{s_ts},{e_ts},Badge,,0,0,0,,{{\\fad(120,120)}}[ {safe} ]"
            ));
        }
        if let Some(ref callout) = sc.overlay.callout_text.as_deref().map(str::trim).filter(|s| !s.is_empty()) {
            let safe = callout.replace('{', "(").replace('}', ")");
            events.push(format!(
                "Dialogue: 2,{s_ts},{e_ts},Callout,,0,0,0,,{{\\fad(150,150)\\pos(540,620)}}{safe}"
            ));
        }
    }

    // 2. Append word-synced karaoke subtitles from transcript when enabled
    if comp.burn_subtitles {
        if let Some(tr) = transcript {
            let seg = vec![SmartClipSegment {
                start_sec: comp.clip_start_sec,
                end_sec: comp.clip_end_sec,
                role: "main".to_string(),
            }];
            let sub_ass = build_clip_ass_subtitles(tr, &seg, ShortsLayout::TemplateSplit);
            for line in sub_ass.lines() {
                if line.starts_with("Dialogue:") {
                    events.push(line.to_string());
                }
            }
        }
    }

    if events.is_empty() {
        return String::new();
    }

    format!(
        "[Script Info]\nScriptType: v4.00+\nPlayResX: 1080\nPlayResY: 1920\nWrapStyle: 1\n\n[V4+ Styles]\nFormat: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding\nStyle: Shorts,Arial,72,&H0000F2FF,&H00FFFFFF,&H00000000,&H80000000,-1,0,0,0,100,100,0,0,1,5,2,2,60,60,340,1\nStyle: Headline,Arial,74,&H0008B3EA,&H00FFFFFF,&H00000000,&H90000000,-1,0,0,0,100,100,1,0,1,6,3,8,50,50,120,1\nStyle: Badge,Arial,40,&H00FFFFFF,&H00FFFFFF,&H00000000,&HB008B3EA,-1,0,0,0,100,100,2,0,3,8,0,7,48,48,54,1\nStyle: Callout,Arial,54,&H0008B3EA,&H00FFFFFF,&H00000000,&HA0000000,-1,0,0,0,100,100,0,0,1,5,2,5,40,40,40,1\n\n[Events]\nFormat: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n{}\n",
        events.join("\n")
    )
}

async fn run_ffmpeg_motion_render(
    ffmpeg_bin: &Path,
    input_path: &Path,
    comp: &AicutComposition,
    template: Option<(&Path, TemplateGeometry)>,
    ass_path: Option<&Path>,
    encoder: &str,
    out_path: &Path,
) -> AicutResult<()> {
    let mut cmd = Command::new(ffmpeg_bin);
    configure_tokio_command(&mut cmd);

    cmd.args([
        "-y",
        "-ss",
        &format!("{:.3}", comp.clip_start_sec),
        "-t",
        &format!("{:.3}", comp.duration_sec),
        "-i",
    ])
    .arg(input_path);

    if let Some((tpl_path, _)) = template {
        cmd.arg("-i").arg(tpl_path);
    }

    let filter = build_motion_filtergraph(comp, template.map(|(_, g)| g), ass_path);
    cmd.args(["-filter_complex", &filter, "-map", "[vout]", "-map", "[abase]"]);

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
        .map_err(|e| AicutError::internal(format!("Failed running FFmpeg Motion render: {e}")))?;

    if !out.status.success() || !out_path.is_file() {
        let stderr = String::from_utf8_lossy(&out.stderr);
        return Err(AicutError::new(
            "MOTION_RENDER_FAILED",
            format!("FFmpeg ({encoder}) falhou ao renderizar Motion Short: {stderr}"),
        ));
    }
    Ok(())
}

async fn generate_thumb(ffmpeg_bin: &Path, video_path: &Path, thumb_path: &Path) -> bool {
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
        "aicut-motion".to_string()
    } else {
        truncated
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::schema::{ContentFocusConfig, PipConfig, SceneOverlay, SceneTransition};

    #[test]
    fn builds_valid_multiscene_filtergraph_and_kinetic_ass() {
        let comp = AicutComposition {
            id: "c1".into(),
            title: "Short Dinâmico".into(),
            media_id: "m1".into(),
            clip_start_sec: 0.0,
            clip_end_sec: 20.0,
            duration_sec: 20.0,
            width: 1080,
            height: 1920,
            fps: 30,
            webcam_rect: NormalizedRect { x: 0.70, y: 0.64, w: 0.28, h: 0.33 },
            content_rect: None,
            burn_subtitles: false,
            template_path: None,
            engine_preference: "hybrid_nvenc".into(),
            scenes: vec![
                MotionScene {
                    id: "s1".into(),
                    title: "Gancho".into(),
                    start_sec: 0.0,
                    end_sec: 4.0,
                    layout: SceneLayoutPreset::CameraFocus,
                    transition_in: SceneTransition::Cut,
                    pip: PipConfig::default(),
                    content_focus: ContentFocusConfig::default(),
                    overlay: SceneOverlay {
                        headline: Some("INACREDITÁVEL!".into()),
                        ..Default::default()
                    },
                    rationale: String::new(),
                    custom_code: None,
                },
                MotionScene {
                    id: "s2".into(),
                    title: "SMOKING33".into(),
                    start_sec: 4.0,
                    end_sec: 12.0,
                    layout: SceneLayoutPreset::SplitBranded,
                    transition_in: SceneTransition::Fade,
                    pip: PipConfig::default(),
                    content_focus: ContentFocusConfig::default(),
                    overlay: SceneOverlay::default(),
                    rationale: String::new(),
                    custom_code: None,
                },
                MotionScene {
                    id: "s3".into(),
                    title: "Destaque PiP".into(),
                    start_sec: 12.0,
                    end_sec: 20.0,
                    layout: SceneLayoutPreset::ContentHeroPip,
                    transition_in: SceneTransition::Fade,
                    pip: PipConfig::default(),
                    content_focus: ContentFocusConfig::default(),
                    overlay: SceneOverlay {
                        badge: Some("DESTAQUE".into()),
                        ..Default::default()
                    },
                    rationale: String::new(),
                    custom_code: None,
                },
            ],
            chat_history: vec![],
            rendered_media_id: None,
            updated_at: Utc::now().to_rfc3339(),
        };

        let fg = build_motion_filtergraph(&comp, Some(TemplateGeometry::default()), None);
        assert!(fg.contains("[vbase]split=3[vs0][vs1][vs2]"));
        assert!(fg.contains("concat=n=3:v=1:a=0[vlayout]"));
        assert!(fg.contains("[tpl0]overlay=0:0:format=auto"));

        let ass = build_motion_ass_track(&comp, None);
        assert!(ass.contains("INACREDITÁVEL!"));
        assert!(ass.contains("[ DESTAQUE ]"));
    }
}
