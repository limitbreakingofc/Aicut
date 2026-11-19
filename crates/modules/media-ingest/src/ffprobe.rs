use aicut_core::{MediaKind, MediaMetadata};
use serde_json::Value;
use std::path::{Path, PathBuf};
use tokio::process::Command;

use crate::formats::infer_kind_from_extension;

#[derive(Debug, Clone)]
pub struct ProbeOutcome {
    pub kind: MediaKind,
    pub metadata: MediaMetadata,
}

/// Runs `ffprobe` to extract technical video/audio metadata and classify Video vs VideoOnly vs Audio.
pub async fn probe_file(
    ffprobe_path: Option<&Path>,
    media_path: &Path,
    ext: &str,
) -> ProbeOutcome {
    let fallback_kind = infer_kind_from_extension(ext);
    let Some(ffprobe) = ffprobe_path else {
        return ProbeOutcome {
            kind: fallback_kind,
            metadata: MediaMetadata::default(),
        };
    };

    let output = match Command::new(ffprobe)
        .args([
            "-v",
            "quiet",
            "-print_format",
            "json",
            "-show_format",
            "-show_streams",
        ])
        .arg(media_path)
        .output()
        .await
    {
        Ok(o) if o.status.success() => o,
        _ => {
            return ProbeOutcome {
                kind: fallback_kind,
                metadata: MediaMetadata::default(),
            };
        }
    };

    let json: Value = match serde_json::from_slice(&output.stdout) {
        Ok(v) => v,
        Err(_) => {
            return ProbeOutcome {
                kind: fallback_kind,
                metadata: MediaMetadata::default(),
            };
        }
    };

    let mut meta = MediaMetadata::default();
    let mut has_video = false;
    let mut has_audio = false;
    let mut has_subtitles = false;

    if let Some(format) = json.get("format") {
        meta.duration_sec = format
            .get("duration")
            .and_then(|v| v.as_str())
            .and_then(|s| s.parse::<f64>().ok());
        meta.bitrate_bps = format
            .get("bit_rate")
            .and_then(|v| v.as_str())
            .and_then(|s| s.parse::<u64>().ok());
        meta.format_name = format
            .get("format_name")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string());
    }

    if let Some(streams) = json.get("streams").and_then(|v| v.as_array()) {
        for s in streams {
            let codec_type = s.get("codec_type").and_then(|v| v.as_str()).unwrap_or("");
            match codec_type {
                "video" => {
                    // Ignore attached album-art images (e.g. mjpeg cover in mp3) as real video streams
                    let is_attached_pic = s
                        .get("disposition")
                        .and_then(|d| d.get("attached_pic"))
                        .and_then(|v| v.as_i64())
                        .unwrap_or(0)
                        == 1;
                    if is_attached_pic {
                        continue;
                    }
                    has_video = true;
                    if meta.video_codec.is_none() {
                        meta.video_codec = s
                            .get("codec_name")
                            .and_then(|v| v.as_str())
                            .map(|v| v.to_string());
                        meta.width = s.get("width").and_then(|v| v.as_u64()).map(|v| v as u32);
                        meta.height = s.get("height").and_then(|v| v.as_u64()).map(|v| v as u32);
                        meta.fps = s
                            .get("avg_frame_rate")
                            .or_else(|| s.get("r_frame_rate"))
                            .and_then(|v| v.as_str())
                            .and_then(parse_fraction);
                    }
                }
                "audio" => {
                    has_audio = true;
                    if meta.audio_codec.is_none() {
                        meta.audio_codec = s
                            .get("codec_name")
                            .and_then(|v| v.as_str())
                            .map(|v| v.to_string());
                        meta.audio_channels =
                            s.get("channels").and_then(|v| v.as_u64()).map(|v| v as u32);
                        meta.sample_rate = s
                            .get("sample_rate")
                            .and_then(|v| v.as_str())
                            .and_then(|sr| sr.parse::<u32>().ok());
                    }
                }
                "subtitle" => {
                    has_subtitles = true;
                }
                _ => {}
            }
        }
    }

    meta.has_subtitles = Some(has_subtitles);

    let kind = match (has_video, has_audio) {
        (true, true) => MediaKind::Video,
        (true, false) => MediaKind::VideoOnly,
        (false, true) => MediaKind::Audio,
        (false, false) => fallback_kind,
    };

    ProbeOutcome { kind, metadata: meta }
}

/// Generates a JPEG thumbnail for video assets using `ffmpeg`.
pub async fn generate_thumbnail(
    ffmpeg_path: Option<&Path>,
    media_path: &Path,
    thumbnails_dir: &Path,
    media_id: &str,
    duration_sec: Option<f64>,
) -> Option<PathBuf> {
    let ffmpeg = ffmpeg_path?;
    let _ = std::fs::create_dir_all(thumbnails_dir);
    let out_path = thumbnails_dir.join(format!("{media_id}.jpg"));

    let seek_sec = duration_sec
        .map(|d| if d > 10.0 { d * 0.1 } else { (d * 0.5).max(0.0) })
        .unwrap_or(1.0);

    let status = Command::new(ffmpeg)
        .args([
            "-y",
            "-ss",
            &format!("{seek_sec:.2}"),
            "-i",
        ])
        .arg(media_path)
        .args([
            "-frames:v",
            "1",
            "-vf",
            "scale=640:-2",
            "-q:v",
            "3",
        ])
        .arg(&out_path)
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .status()
        .await
        .ok()?;

    if status.success() && out_path.is_file() {
        Some(out_path)
    } else {
        None
    }
}

fn parse_fraction(raw: &str) -> Option<f64> {
    if let Some((num_s, den_s)) = raw.split_once('/') {
        let num: f64 = num_s.parse().ok()?;
        let den: f64 = den_s.parse().ok()?;
        if den > 0.0 && num > 0.0 {
            return Some((num / den * 100.0).round() / 100.0);
        }
        None
    } else {
        raw.parse::<f64>().ok().filter(|v| *v > 0.0)
    }
}
