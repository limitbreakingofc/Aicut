use aicut_core::{configure_tokio_command, AicutError, AicutResult, TranscriptRecord};
use std::path::{Path, PathBuf};
use tokio::process::Command;

#[derive(Debug, Clone, Copy)]
pub struct TimeWindow {
    pub start_sec: f64,
    pub end_sec: f64,
}

impl TimeWindow {
    pub fn duration_sec(&self) -> f64 {
        (self.end_sec - self.start_sec).max(0.0)
    }
}

/// Splits a video or long livestream into overlapping windows (e.g. 25 min with 3 min overlap)
/// so a coherent 3-minute story crossing a boundary appears intact in at least one window.
pub fn plan_windows(total_duration_sec: f64, window_sec: f64, overlap_sec: f64) -> Vec<TimeWindow> {
    if total_duration_sec <= 0.0 {
        return Vec::new();
    }
    let w = window_sec.max(60.0);
    let ov = overlap_sec.clamp(0.0, w * 0.5);
    if total_duration_sec <= w {
        return vec![TimeWindow {
            start_sec: 0.0,
            end_sec: total_duration_sec,
        }];
    }

    let step = (w - ov).max(30.0);
    let mut windows = Vec::new();
    let mut start = 0.0;
    while start < total_duration_sec {
        let end = (start + w).min(total_duration_sec);
        windows.push(TimeWindow {
            start_sec: start,
            end_sec: end,
        });
        if end >= total_duration_sec - 1.0 {
            break;
        }
        start += step;
    }
    windows
}

/// Queries exact media duration in seconds via `ffprobe`.
pub async fn probe_duration_sec(ffprobe_bin: &Path, input_path: &Path) -> AicutResult<f64> {
    let mut cmd = Command::new(ffprobe_bin);
    configure_tokio_command(&mut cmd);
    let out = cmd
        .args([
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
        ])
        .arg(input_path)
        .output()
        .await
        .map_err(|e| AicutError::internal(format!("Failed running ffprobe: {e}")))?;

    if !out.status.success() {
        let stderr = String::from_utf8_lossy(&out.stderr);
        return Err(AicutError::internal(format!(
            "ffprobe failed reading duration: {stderr}"
        )));
    }
    let text = String::from_utf8_lossy(&out.stdout);
    text.trim()
        .parse::<f64>()
        .ok()
        .filter(|d| d.is_finite() && *d > 0.0)
        .ok_or_else(|| AicutError::internal("Could not parse valid duration from ffprobe output"))
}

/// Builds a lightweight MP4 proxy (low-res, low-fps, mono audio) for native multimodal upload to Gemini.
#[allow(clippy::too_many_arguments)]
pub async fn build_video_proxy_chunk(
    ffmpeg_bin: &Path,
    input_path: &Path,
    start_sec: f64,
    duration_sec: f64,
    target_height: u32,
    fps: f64,
    out_mp4: &Path,
) -> AicutResult<()> {
    let vf = format!("fps={fps:.3},scale=-2:{target_height}:flags=fast_bilinear");
    let mut cmd = Command::new(ffmpeg_bin);
    configure_tokio_command(&mut cmd);
    let out = cmd
        .args(["-y", "-ss", &format!("{start_sec:.3}"), "-i"])
        .arg(input_path)
        .args([
            "-t",
            &format!("{duration_sec:.3}"),
            "-vf",
            &vf,
            "-c:v",
            "libx264",
            "-preset",
            "ultrafast",
            "-crf",
            "30",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-ac",
            "1",
            "-ar",
            "16000",
            "-b:a",
            "48k",
            "-movflags",
            "+faststart",
        ])
        .arg(out_mp4)
        .output()
        .await
        .map_err(|e| AicutError::internal(format!("Failed running FFmpeg proxy: {e}")))?;

    if !out.status.success() || !out_mp4.is_file() {
        let stderr = String::from_utf8_lossy(&out.stderr);
        return Err(AicutError::internal(format!(
            "FFmpeg proxy generation failed: {stderr}"
        )));
    }
    Ok(())
}

/// Extracts compact 3x3 storyboard contact sheets (JPEG) for fast visual scanning on local VLMs.
/// Returns `(sheet_start_sec, sheet_end_sec, jpeg_path)` tuples in chronological order.
pub async fn extract_storyboard_sheets(
    ffmpeg_bin: &Path,
    input_path: &Path,
    start_sec: f64,
    duration_sec: f64,
    out_dir: &Path,
    max_sheets: usize,
) -> AicutResult<Vec<(f64, f64, PathBuf)>> {
    std::fs::create_dir_all(out_dir)?;
    let tiles_per_sheet = 9usize; // 3x3
    let target_frames = (max_sheets.max(1) * tiles_per_sheet) as f64;
    let fps = (target_frames / duration_sec.max(1.0)).clamp(0.03, 1.0);
    let vf = format!(
        "fps={fps:.5},scale=320:180:force_original_aspect_ratio=decrease,pad=320:180:(ow-iw)/2:(oh-ih)/2,setsar=1,tile=3x3"
    );
    let pattern = out_dir.join("sheet_%03d.jpg");

    let mut cmd = Command::new(ffmpeg_bin);
    configure_tokio_command(&mut cmd);
    let out = cmd
        .args(["-y", "-ss", &format!("{start_sec:.3}"), "-i"])
        .arg(input_path)
        .args([
            "-t",
            &format!("{duration_sec:.3}"),
            "-vf",
            &vf,
            "-frames:v",
            &max_sheets.max(1).to_string(),
            "-q:v",
            "5",
            "-fps_mode",
            "vfr",
        ])
        .arg(&pattern)
        .output()
        .await
        .map_err(|e| AicutError::internal(format!("Failed extracting storyboards: {e}")))?;

    if !out.status.success() {
        let stderr = String::from_utf8_lossy(&out.stderr);
        return Err(AicutError::internal(format!(
            "FFmpeg storyboard extraction failed: {stderr}"
        )));
    }

    let mut sheets: Vec<PathBuf> = std::fs::read_dir(out_dir)?
        .flatten()
        .map(|e| e.path())
        .filter(|p| {
            p.file_name()
                .and_then(|n| n.to_str())
                .map(|s| s.starts_with("sheet_") && s.ends_with(".jpg"))
                .unwrap_or(false)
        })
        .collect();
    sheets.sort();

    let count = sheets.len().max(1);
    let span_per_sheet = duration_sec / count as f64;
    let result = sheets
        .into_iter()
        .enumerate()
        .map(|(i, p)| {
            let s = start_sec + i as f64 * span_per_sheet;
            let e = (s + span_per_sheet).min(start_sec + duration_sec);
            (s, e, p)
        })
        .collect();
    Ok(result)
}

/// Extracts individual chronological JPEG frames for surgical candidate review (Pass B on local VLM).
pub async fn extract_candidate_frames(
    ffmpeg_bin: &Path,
    input_path: &Path,
    start_sec: f64,
    duration_sec: f64,
    out_dir: &Path,
    max_frames: usize,
) -> AicutResult<Vec<(f64, PathBuf)>> {
    std::fs::create_dir_all(out_dir)?;
    let fps = (max_frames.max(2) as f64 / duration_sec.max(1.0)).clamp(0.05, 1.0);
    let vf = format!("fps={fps:.5},scale=512:-2");
    let pattern = out_dir.join("frame_%03d.jpg");

    let mut cmd = Command::new(ffmpeg_bin);
    configure_tokio_command(&mut cmd);
    let out = cmd
        .args(["-y", "-ss", &format!("{start_sec:.3}"), "-i"])
        .arg(input_path)
        .args([
            "-t",
            &format!("{duration_sec:.3}"),
            "-vf",
            &vf,
            "-frames:v",
            &max_frames.max(2).to_string(),
            "-q:v",
            "5",
        ])
        .arg(&pattern)
        .output()
        .await
        .map_err(|e| AicutError::internal(format!("Failed extracting candidate frames: {e}")))?;

    if !out.status.success() {
        let stderr = String::from_utf8_lossy(&out.stderr);
        return Err(AicutError::internal(format!(
            "FFmpeg candidate frame extraction failed: {stderr}"
        )));
    }

    let mut frames: Vec<PathBuf> = std::fs::read_dir(out_dir)?
        .flatten()
        .map(|e| e.path())
        .filter(|p| {
            p.file_name()
                .and_then(|n| n.to_str())
                .map(|s| s.starts_with("frame_") && s.ends_with(".jpg"))
                .unwrap_or(false)
        })
        .collect();
    frames.sort();

    let n = frames.len().max(1);
    let step = duration_sec / n as f64;
    Ok(frames
        .into_iter()
        .enumerate()
        .map(|(i, p)| (start_sec + i as f64 * step, p))
        .collect())
}

/// Formats synchronized transcript lines with absolute second timestamps for an analysis window.
pub fn format_transcript_window(
    transcript: Option<&TranscriptRecord>,
    start_sec: f64,
    end_sec: f64,
    max_chars: usize,
) -> String {
    let Some(tr) = transcript else {
        return "(Sem transcricao de fala disponivel — analise principalmente pelos elementos visuais e audio do video.)".to_string();
    };

    let mut out = String::new();
    for seg in &tr.segments {
        let s = seg.start_ms as f64 / 1000.0;
        let e = seg.end_ms as f64 / 1000.0;
        if e < start_sec || s > end_sec {
            continue;
        }
        let text = seg.text.trim();
        if text.is_empty() {
            continue;
        }
        let line = format!("[{s:.1}s - {e:.1}s] {text}\n");
        if out.len() + line.len() > max_chars {
            out.push_str("...(transcricao truncada pelo limite de contexto)\n");
            break;
        }
        out.push_str(&line);
    }

    if out.is_empty() {
        "(Sem falas transcritas neste intervalo — foque na acao visual e reacoes.)".to_string()
    } else {
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn plans_single_window_for_short_videos() {
        let w = plan_windows(600.0, 1500.0, 180.0);
        assert_eq!(w.len(), 1);
        assert_eq!(w[0].start_sec, 0.0);
        assert_eq!(w[0].end_sec, 600.0);
    }

    #[test]
    fn plans_overlapping_windows_for_long_livestreams() {
        // 2-hour live (7200s) with 25-min windows (1500s) and 3-min overlap (180s)
        let w = plan_windows(7200.0, 1500.0, 180.0);
        assert!(w.len() >= 5);
        assert_eq!(w[0].start_sec, 0.0);
        assert_eq!(w[0].end_sec, 1500.0);
        // Second window overlaps first by 180s
        assert_eq!(w[1].start_sec, 1320.0);
        assert_eq!(w.last().unwrap().end_sec, 7200.0);
    }
}
