use aicut_core::{
    configure_tokio_command, ActorKind, AicutError, AicutEvent, AicutKernel, AicutResult,
    JobProgress, MediaAsset, MediaKind, MediaSourceType,
};
use chrono::Utc;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::sync::Arc;
use tokio::io::{AsyncBufReadExt, BufReader};
use tokio::process::Command;
use tokio_util::sync::CancellationToken;
use uuid::Uuid;

use crate::ffprobe::{generate_thumbnail, probe_file};
use crate::formats::{compute_fast_fingerprint, extension_lower};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema, Default)]
#[serde(rename_all = "snake_case")]
pub enum DownloadMode {
    #[default]
    VideoAudio,
    VideoOnly,
    AudioOnly,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema, Default)]
#[serde(rename_all = "snake_case")]
pub enum QualityPreset {
    #[default]
    Best,
    #[serde(rename = "2160p")]
    P2160,
    #[serde(rename = "1440p")]
    P1440,
    #[serde(rename = "1080p")]
    P1080,
    #[serde(rename = "720p")]
    P720,
    #[serde(rename = "480p")]
    P480,
    #[serde(rename = "360p")]
    P360,
    Worst,
    Custom,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema, Default)]
#[serde(rename_all = "snake_case")]
pub enum VideoCodecPref {
    #[default]
    Auto,
    H264,
    Vp9,
    Av1,
    Hevc,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema, Default)]
#[serde(rename_all = "snake_case")]
pub enum ContainerFormat {
    #[default]
    Auto,
    Mp4,
    Mkv,
    Webm,
    Mov,
    Mp3,
    M4a,
    Wav,
    Flac,
    Opus,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema, Default)]
#[serde(rename_all = "snake_case")]
pub enum PlaylistMode {
    #[default]
    Single,
    Playlist,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum BrowserCookieSource {
    Chrome,
    Edge,
    Firefox,
    Brave,
    Opera,
    Vivaldi,
    Chromium,
}

impl BrowserCookieSource {
    pub fn as_ytdlp_arg(&self) -> &'static str {
        match self {
            Self::Chrome => "chrome",
            Self::Edge => "edge",
            Self::Firefox => "firefox",
            Self::Brave => "brave",
            Self::Opera => "opera",
            Self::Vivaldi => "vivaldi",
            Self::Chromium => "chromium",
        }
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, JsonSchema)]
pub struct SubtitleOptions {
    #[serde(default)]
    pub enabled: bool,
    #[serde(default)]
    pub auto_generated: bool,
    #[serde(default)]
    pub languages: Vec<String>,
    #[serde(default)]
    pub embed_in_file: bool,
    #[serde(default)]
    pub format: Option<String>,
}

/// Complete, strongly-typed yt-dlp download configuration usable both from the visual UI and any AI chat.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct DownloadRequest {
    pub url: String,
    #[serde(default)]
    pub mode: DownloadMode,
    #[serde(default)]
    pub quality: QualityPreset,
    #[serde(default)]
    pub format_id: Option<String>,
    #[serde(default)]
    pub video_codec: VideoCodecPref,
    #[serde(default)]
    pub container: ContainerFormat,
    #[serde(default)]
    pub audio_quality: Option<String>,
    #[serde(default)]
    pub subtitles: SubtitleOptions,
    #[serde(default)]
    pub embed_thumbnail: bool,
    #[serde(default = "default_true")]
    pub embed_metadata: bool,
    #[serde(default)]
    pub embed_chapters: bool,
    #[serde(default)]
    pub split_chapters: bool,
    #[serde(default)]
    pub clip_start: Option<String>,
    #[serde(default)]
    pub clip_end: Option<String>,
    #[serde(default)]
    pub force_keyframes_at_cuts: bool,
    #[serde(default)]
    pub playlist_mode: PlaylistMode,
    #[serde(default)]
    pub playlist_items: Option<String>,
    #[serde(default)]
    pub concurrent_fragments: Option<u8>,
    #[serde(default)]
    pub rate_limit: Option<String>,
    #[serde(default)]
    pub proxy: Option<String>,
    #[serde(default)]
    pub impersonate: Option<String>,
    #[serde(default)]
    pub cookies_from_browser: Option<BrowserCookieSource>,
    #[serde(default)]
    pub cookies_file: Option<String>,
    #[serde(default)]
    pub sponsorblock_remove: Vec<String>,
    #[serde(default)]
    pub filename_template: Option<String>,
    #[serde(default)]
    pub custom_title: Option<String>,
    #[serde(default)]
    pub tags: Vec<String>,
    #[serde(default)]
    pub extra_args: Vec<String>,
    #[serde(default)]
    pub wait: bool,
}

fn default_true() -> bool {
    true
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct ProbeUrlRequest {
    pub url: String,
    #[serde(default)]
    pub cookies_from_browser: Option<BrowserCookieSource>,
    #[serde(default)]
    pub cookies_file: Option<String>,
    #[serde(default)]
    pub proxy: Option<String>,
    #[serde(default)]
    pub impersonate: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct ProbedFormat {
    pub format_id: String,
    pub ext: String,
    pub kind: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub resolution: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub height: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub fps: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub vcodec: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub acodec: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub filesize_bytes: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tbr_kbps: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub note: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct ProbeUrlResult {
    pub url: String,
    pub title: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub uploader: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub extractor: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub duration_sec: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub thumbnail: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    pub is_playlist: bool,
    pub playlist_count: usize,
    pub chapters_count: usize,
    pub available_subtitles: Vec<String>,
    pub formats: Vec<ProbedFormat>,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct DownloadResponse {
    pub job_id: String,
    pub status: String,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub assets: Vec<MediaAsset>,
}

/// Builds the exact yt-dlp CLI argument vector from `DownloadRequest` safely (no shell).
/// IMPORTANT: Uses `--no-simulate`, `--progress`, and `--print-to-file` (NEVER `--print` alone,
/// because `--print` implies `--simulate` and `--quiet` in yt-dlp!).
pub fn build_ytdlp_args(
    req: &DownloadRequest,
    output_dir: &Path,
    ffmpeg_path: Option<&Path>,
    manifest_file: Option<&Path>,
) -> Vec<String> {
    let mut args = vec![
        "--no-simulate".to_string(),
        "--progress".to_string(),
        "--newline".to_string(),
        "--progress-template".to_string(),
        "download:AICUT_PROGRESS:%(progress._percent_str)s|%(progress.downloaded_bytes)s|%(progress.total_bytes)s|%(progress.speed)s|%(progress.eta)s".to_string(),
    ];

    if let Some(mfile) = manifest_file {
        args.push("--print-to-file".to_string());
        args.push("after_move:%(filepath)s".to_string());
        args.push(mfile.to_string_lossy().to_string());
    }

    if let Some(ffmpeg) = ffmpeg_path {
        if let Some(parent) = ffmpeg.parent() {
            args.push("--ffmpeg-location".to_string());
            args.push(parent.to_string_lossy().to_string());
        }
    }

    // Playlist mode
    match req.playlist_mode {
        PlaylistMode::Single => args.push("--no-playlist".to_string()),
        PlaylistMode::Playlist => {
            args.push("--yes-playlist".to_string());
            if let Some(ref items) = req.playlist_items {
                if !items.trim().is_empty() {
                    args.push("--playlist-items".to_string());
                    args.push(items.trim().to_string());
                }
            }
        }
    }

    // Format selection
    let format_expr = build_format_selector(req);
    args.push("-f".to_string());
    args.push(format_expr);

    // Post-processing / Container & Audio extraction
    match req.mode {
        DownloadMode::AudioOnly => {
            args.push("-x".to_string());
            let audio_fmt = match req.container {
                ContainerFormat::Mp3 => Some("mp3"),
                ContainerFormat::M4a => Some("m4a"),
                ContainerFormat::Wav => Some("wav"),
                ContainerFormat::Flac => Some("flac"),
                ContainerFormat::Opus => Some("opus"),
                _ => None,
            };
            if let Some(af) = audio_fmt {
                args.push("--audio-format".to_string());
                args.push(af.to_string());
            }
            if let Some(ref aq) = req.audio_quality {
                let q = aq.trim();
                if !q.is_empty() && q != "best" {
                    args.push("--audio-quality".to_string());
                    args.push(q.to_string());
                } else {
                    args.push("--audio-quality".to_string());
                    args.push("0".to_string());
                }
            }
        }
        DownloadMode::VideoAudio | DownloadMode::VideoOnly => {
            let merge_fmt = match req.container {
                ContainerFormat::Mp4 => Some("mp4"),
                ContainerFormat::Mkv => Some("mkv"),
                ContainerFormat::Webm => Some("webm"),
                ContainerFormat::Mov => Some("mov"),
                _ => None,
            };
            if let Some(mf) = merge_fmt {
                args.push("--merge-output-format".to_string());
                args.push(mf.to_string());
                if req.mode == DownloadMode::VideoOnly {
                    args.push("--remux-video".to_string());
                    args.push(mf.to_string());
                }
            }
        }
    }

    // Subtitles
    if req.subtitles.enabled {
        args.push("--write-subs".to_string());
        if req.subtitles.auto_generated {
            args.push("--write-auto-subs".to_string());
        }
        if !req.subtitles.languages.is_empty() {
            args.push("--sub-langs".to_string());
            args.push(req.subtitles.languages.join(","));
        }
        if let Some(ref fmt) = req.subtitles.format {
            if !fmt.trim().is_empty() {
                args.push("--sub-format".to_string());
                args.push(fmt.trim().to_string());
            }
        }
        if req.subtitles.embed_in_file {
            args.push("--embed-subs".to_string());
        }
    }

    // Metadata / Thumbnails / Chapters
    if req.embed_metadata {
        args.push("--embed-metadata".to_string());
    }
    if req.embed_thumbnail {
        args.push("--embed-thumbnail".to_string());
    }
    if req.embed_chapters {
        args.push("--embed-chapters".to_string());
    }
    if req.split_chapters {
        args.push("--split-chapters".to_string());
    }

    // Clip / Section cutting (--download-sections)
    let start = req
        .clip_start
        .as_deref()
        .map(str::trim)
        .filter(|s| !s.is_empty());
    let end = req
        .clip_end
        .as_deref()
        .map(str::trim)
        .filter(|s| !s.is_empty());
    if start.is_some() || end.is_some() {
        let s = start.unwrap_or("0");
        let e = end.unwrap_or("inf");
        args.push("--download-sections".to_string());
        args.push(format!("*{s}-{e}"));
        if req.force_keyframes_at_cuts {
            args.push("--force-keyframes-at-cuts".to_string());
        }
    }

    // Network, Concurrency, Impersonation, Cookies & SponsorBlock
    if let Some(cf) = req.concurrent_fragments {
        let clamped = cf.clamp(1, 16);
        args.push("--concurrent-fragments".to_string());
        args.push(clamped.to_string());
    }
    if let Some(ref rl) = req.rate_limit {
        if !rl.trim().is_empty() {
            args.push("--limit-rate".to_string());
            args.push(rl.trim().to_string());
        }
    }
    if let Some(ref proxy) = req.proxy {
        if !proxy.trim().is_empty() {
            args.push("--proxy".to_string());
            args.push(proxy.trim().to_string());
        }
    }
    if let Some(ref imp) = req.impersonate {
        if !imp.trim().is_empty() {
            args.push("--impersonate".to_string());
            args.push(imp.trim().to_string());
        }
    }
    if let Some(browser) = req.cookies_from_browser {
        args.push("--cookies-from-browser".to_string());
        args.push(browser.as_ytdlp_arg().to_string());
    } else if let Some(ref cfile) = req.cookies_file {
        if !cfile.trim().is_empty() {
            args.push("--cookies".to_string());
            args.push(cfile.trim().to_string());
        }
    }
    if !req.sponsorblock_remove.is_empty() {
        args.push("--sponsorblock-remove".to_string());
        args.push(req.sponsorblock_remove.join(","));
    }

    // Output directory & filename template
    let template = req
        .filename_template
        .as_deref()
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .unwrap_or("%(title).120B [%(id)s].%(ext)s");
    let out_pattern = output_dir.join(template);
    args.push("-o".to_string());
    args.push(out_pattern.to_string_lossy().to_string());

    // Extra user/AI flags
    for extra in &req.extra_args {
        let trimmed = extra.trim();
        if !trimmed.is_empty() {
            args.push(trimmed.to_string());
        }
    }

    args.push(req.url.trim().to_string());
    args
}

pub fn build_format_selector(req: &DownloadRequest) -> String {
    if let Some(ref fid) = req.format_id {
        if !fid.trim().is_empty() {
            return fid.trim().to_string();
        }
    }

    if req.mode == DownloadMode::AudioOnly {
        return "bestaudio/best".to_string();
    }

    let height_filter = match req.quality {
        QualityPreset::P2160 => "[height<=2160]",
        QualityPreset::P1440 => "[height<=1440]",
        QualityPreset::P1080 => "[height<=1080]",
        QualityPreset::P720 => "[height<=720]",
        QualityPreset::P480 => "[height<=480]",
        QualityPreset::P360 => "[height<=360]",
        QualityPreset::Worst => {
            return if req.mode == DownloadMode::VideoOnly {
                "worstvideo".to_string()
            } else {
                "worstvideo+worstaudio/worst".to_string()
            };
        }
        QualityPreset::Best | QualityPreset::Custom => "",
    };

    let codec_filter = match req.video_codec {
        VideoCodecPref::Auto => "",
        VideoCodecPref::H264 => "[vcodec^=avc]",
        VideoCodecPref::Vp9 => "[vcodec^=vp9]",
        VideoCodecPref::Av1 => "[vcodec^=av01]",
        VideoCodecPref::Hevc => "[vcodec^=hev]",
    };

    match req.mode {
        DownloadMode::VideoOnly => {
            if codec_filter.is_empty() {
                format!("bestvideo{height_filter}/bestvideo")
            } else {
                format!("bestvideo{height_filter}{codec_filter}/bestvideo{height_filter}/bestvideo")
            }
        }
        DownloadMode::VideoAudio => {
            if codec_filter.is_empty() {
                format!("bestvideo{height_filter}+bestaudio/best{height_filter}/best")
            } else {
                format!(
                    "bestvideo{height_filter}{codec_filter}+bestaudio/bestvideo{height_filter}+bestaudio/best{height_filter}/best"
                )
            }
        }
        DownloadMode::AudioOnly => "bestaudio/best".to_string(),
    }
}

pub async fn probe_url(kernel: &Arc<AicutKernel>, req: &ProbeUrlRequest) -> AicutResult<ProbeUrlResult> {
    if req.url.trim().is_empty() {
        return Err(AicutError::invalid_input("URL cannot be empty."));
    }
    let ytdlp_path = kernel.tools.resolve_tool("yt-dlp")?;

    let mut args = vec![
        "--dump-single-json".to_string(),
        "--no-warnings".to_string(),
        "--no-playlist".to_string(),
        "--skip-download".to_string(),
    ];
    if let Some(ref imp) = req.impersonate {
        if !imp.trim().is_empty() {
            args.push("--impersonate".to_string());
            args.push(imp.trim().to_string());
        }
    }
    if let Some(browser) = req.cookies_from_browser {
        args.push("--cookies-from-browser".to_string());
        args.push(browser.as_ytdlp_arg().to_string());
    } else if let Some(ref cfile) = req.cookies_file {
        if !cfile.trim().is_empty() {
            args.push("--cookies".to_string());
            args.push(cfile.trim().to_string());
        }
    }
    if let Some(ref proxy) = req.proxy {
        if !proxy.trim().is_empty() {
            args.push("--proxy".to_string());
            args.push(proxy.trim().to_string());
        }
    }
    args.push(req.url.trim().to_string());

    let mut cmd = Command::new(&ytdlp_path);
    configure_tokio_command(&mut cmd);
    let output = cmd
        .env("PYTHONIOENCODING", "utf-8")
        .args(&args)
        .output()
        .await
        .map_err(|e| AicutError::internal(format!("Failed to execute yt-dlp: {e}")))?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr).trim().to_string();
        return Err(
            AicutError::new("YTDLP_PROBE_FAILED", format!("yt-dlp probe failed: {stderr}"))
                .recoverable(
                    "Verify the URL is valid and accessible, or enable browser cookies / impersonate (e.g. 'chrome') for Cloudflare-protected sites like Kick.",
                ),
        );
    }

    let json: Value = serde_json::from_slice(&output.stdout)?;
    Ok(parse_probe_json(req.url.trim(), &json))
}

pub fn parse_probe_json(url: &str, json: &Value) -> ProbeUrlResult {
    let title = json
        .get("title")
        .and_then(|v| v.as_str())
        .unwrap_or("Untitled")
        .to_string();
    let uploader = json
        .get("uploader")
        .or_else(|| json.get("channel"))
        .and_then(|v| v.as_str())
        .map(|s| s.to_string());
    let extractor = json
        .get("extractor_key")
        .or_else(|| json.get("extractor"))
        .and_then(|v| v.as_str())
        .map(|s| s.to_string());
    let duration_sec = json.get("duration").and_then(|v| v.as_f64());
    let thumbnail = json
        .get("thumbnail")
        .and_then(|v| v.as_str())
        .map(|s| s.to_string());
    let description = json
        .get("description")
        .and_then(|v| v.as_str())
        .map(|s| s.chars().take(400).collect());

    let is_playlist = json.get("_type").and_then(|v| v.as_str()) == Some("playlist");
    let playlist_count = json
        .get("entries")
        .and_then(|v| v.as_array())
        .map(|a| a.len())
        .unwrap_or(if is_playlist { 0 } else { 1 });

    let chapters_count = json
        .get("chapters")
        .and_then(|v| v.as_array())
        .map(|a| a.len())
        .unwrap_or(0);

    let mut available_subtitles = Vec::new();
    if let Some(subs) = json.get("subtitles").and_then(|v| v.as_object()) {
        for lang in subs.keys() {
            available_subtitles.push(lang.clone());
        }
    }
    available_subtitles.sort();

    let mut formats = Vec::new();
    if let Some(fmts) = json.get("formats").and_then(|v| v.as_array()) {
        for f in fmts {
            let Some(format_id) = f.get("format_id").and_then(|v| v.as_str()) else {
                continue;
            };
            let ext = f
                .get("ext")
                .and_then(|v| v.as_str())
                .unwrap_or("unknown")
                .to_string();
            let vcodec = f
                .get("vcodec")
                .and_then(|v| v.as_str())
                .filter(|s| *s != "none")
                .map(|s| s.to_string());
            let acodec = f
                .get("acodec")
                .and_then(|v| v.as_str())
                .filter(|s| *s != "none")
                .map(|s| s.to_string());

            let kind = match (&vcodec, &acodec) {
                (Some(_), Some(_)) => "video_audio",
                (Some(_), None) => "video_only",
                (None, Some(_)) => "audio_only",
                (None, None) => continue,
            }
            .to_string();

            let height = f.get("height").and_then(|v| v.as_u64()).map(|v| v as u32);
            let width = f.get("width").and_then(|v| v.as_u64()).map(|v| v as u32);
            let resolution = match (width, height) {
                (Some(w), Some(h)) => Some(format!("{w}x{h}")),
                _ => f
                    .get("resolution")
                    .and_then(|v| v.as_str())
                    .map(|s| s.to_string()),
            };

            formats.push(ProbedFormat {
                format_id: format_id.to_string(),
                ext,
                kind,
                resolution,
                height,
                fps: f.get("fps").and_then(|v| v.as_f64()),
                vcodec,
                acodec,
                filesize_bytes: f
                    .get("filesize")
                    .or_else(|| f.get("filesize_approx"))
                    .and_then(|v| v.as_u64()),
                tbr_kbps: f.get("tbr").and_then(|v| v.as_f64()),
                note: f
                    .get("format_note")
                    .and_then(|v| v.as_str())
                    .map(|s| s.to_string()),
            });
        }
    }

    formats.sort_by(|a, b| {
        b.height
            .unwrap_or(0)
            .cmp(&a.height.unwrap_or(0))
            .then_with(|| {
                b.tbr_kbps
                    .unwrap_or(0.0)
                    .partial_cmp(&a.tbr_kbps.unwrap_or(0.0))
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
    });

    ProbeUrlResult {
        url: url.to_string(),
        title,
        uploader,
        extractor,
        duration_sec,
        thumbnail,
        description,
        is_playlist,
        playlist_count,
        chapters_count,
        available_subtitles,
        formats,
    }
}

pub fn parse_progress_line(line: &str) -> Option<JobProgress> {
    let rest = line.strip_prefix("AICUT_PROGRESS:")?;
    let parts: Vec<&str> = rest.split('|').collect();
    if parts.len() < 5 {
        return None;
    }

    let percent = parts[0]
        .trim()
        .trim_end_matches('%')
        .parse::<f64>()
        .unwrap_or(0.0)
        .clamp(0.0, 100.0);
    let downloaded_bytes = parts[1].trim().parse::<u64>().ok();
    let total_bytes = parts[2].trim().parse::<u64>().ok();
    let speed_bytes_per_sec = parts[3].trim().parse::<f64>().ok();
    let eta_seconds = parts[4].trim().parse::<f64>().ok();

    Some(JobProgress {
        percent,
        stage: "downloading".to_string(),
        downloaded_bytes,
        total_bytes,
        speed_bytes_per_sec,
        eta_seconds,
        message: None,
    })
}

pub async fn start_download(
    kernel: Arc<AicutKernel>,
    req: DownloadRequest,
    actor: ActorKind,
) -> AicutResult<DownloadResponse> {
    if req.url.trim().is_empty() {
        return Err(AicutError::invalid_input("Download URL cannot be empty."));
    }
    let ytdlp_path = kernel.tools.resolve_tool("yt-dlp")?;
    let ffmpeg_path = kernel.tools.resolve_tool("ffmpeg").ok();
    let ffprobe_path = kernel.tools.resolve_tool("ffprobe").ok();

    let title = req
        .custom_title
        .clone()
        .filter(|s| !s.trim().is_empty())
        .unwrap_or_else(|| format!("Download {}", req.url.trim()));

    let (job, cancel) = kernel
        .jobs
        .create_job("media.download", title, "media-ingest", actor);
    let job_id = job.id.clone();

    let wait = req.wait;
    let handle = tokio::spawn(execute_download_task(
        kernel.clone(),
        job_id.clone(),
        ytdlp_path,
        ffmpeg_path,
        ffprobe_path,
        req,
        cancel,
    ));

    if wait {
        let assets = handle
            .await
            .map_err(|e| AicutError::internal(format!("Download task panicked: {e}")))??;
        Ok(DownloadResponse {
            job_id,
            status: "completed".to_string(),
            assets,
        })
    } else {
        Ok(DownloadResponse {
            job_id,
            status: "queued".to_string(),
            assets: Vec::new(),
        })
    }
}

async fn execute_download_task(
    kernel: Arc<AicutKernel>,
    job_id: String,
    ytdlp_path: PathBuf,
    ffmpeg_path: Option<PathBuf>,
    ffprobe_path: Option<PathBuf>,
    req: DownloadRequest,
    cancel: CancellationToken,
) -> AicutResult<Vec<MediaAsset>> {
    kernel.jobs.mark_running(&job_id, "initializing");
    let _ = std::fs::create_dir_all(&kernel.workspace.downloads_dir);

    let manifest_path = std::env::temp_dir().join(format!("aicut-ytdlp-{job_id}.txt"));
    let _ = std::fs::remove_file(&manifest_path);

    let args = build_ytdlp_args(
        &req,
        &kernel.workspace.downloads_dir,
        ffmpeg_path.as_deref(),
        Some(&manifest_path),
    );
    kernel
        .jobs
        .append_log(&job_id, format!("Executando yt-dlp ({} args)", args.len()));

    let mut cmd = Command::new(&ytdlp_path);
    configure_tokio_command(&mut cmd);
    let mut child = cmd
        .env("PYTHONIOENCODING", "utf-8")
        .args(&args)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|e| {
            let err = AicutError::internal(format!("Failed to launch yt-dlp: {e}"));
            kernel.jobs.fail(&job_id, err.clone());
            err
        })?;

    let stdout = child.stdout.take().expect("piped stdout");
    let stderr = child.stderr.take().expect("piped stderr");

    let mut stdout_lines = BufReader::new(stdout).lines();
    let mut stderr_lines = BufReader::new(stderr).lines();

    let mut stdout_done = false;
    let mut stderr_done = false;
    let mut stderr_buf: Vec<String> = Vec::new();
    let mut fallback_paths: Vec<PathBuf> = Vec::new();

    while !(stdout_done && stderr_done) {
        tokio::select! {
            _ = cancel.cancelled() => {
                let _ = child.kill().await;
                let _ = std::fs::remove_file(&manifest_path);
                kernel.jobs.mark_cancelled(&job_id);
                return Err(AicutError::cancelled("Download cancelled by user/AI."));
            }
            line = stdout_lines.next_line(), if !stdout_done => {
                match line {
                    Ok(Some(l)) => {
                        let trimmed = l.trim();
                        if let Some(prog) = parse_progress_line(trimmed) {
                            kernel.jobs.update_progress(&job_id, prog);
                        } else if !trimmed.is_empty() {
                            if let Some(dest) = extract_destination_path(trimmed) {
                                if !fallback_paths.contains(&dest) {
                                    fallback_paths.push(dest);
                                }
                            }
                            if trimmed.contains("[Merger]") || trimmed.contains("[ExtractAudio]") || trimmed.contains("[Fixup") {
                                kernel.jobs.update_progress(&job_id, JobProgress {
                                    percent: 94.0,
                                    stage: "post_processing".to_string(),
                                    message: Some(trimmed.to_string()),
                                    ..Default::default()
                                });
                            }
                            kernel.jobs.append_log(&job_id, trimmed);
                        }
                    }
                    Ok(None) | Err(_) => stdout_done = true,
                }
            }
            err_line = stderr_lines.next_line(), if !stderr_done => {
                match err_line {
                    Ok(Some(el)) => {
                        let trimmed = el.trim();
                        if !trimmed.is_empty() {
                            stderr_buf.push(trimmed.to_string());
                            kernel.jobs.append_log(&job_id, format!("stderr: {trimmed}"));
                        }
                    }
                    Ok(None) | Err(_) => stderr_done = true,
                }
            }
        }
    }

    let status = child
        .wait()
        .await
        .map_err(|e| AicutError::internal(format!("Failed awaiting yt-dlp: {e}")))?;

    let mut final_files: Vec<PathBuf> = Vec::new();
    if let Ok(content) = std::fs::read_to_string(&manifest_path) {
        for line in content.lines() {
            let trimmed = line.trim().trim_start_matches('\u{feff}');
            if !trimmed.is_empty() {
                let p = PathBuf::from(trimmed);
                if p.is_file() && !final_files.contains(&p) {
                    final_files.push(p);
                }
            }
        }
    }
    let _ = std::fs::remove_file(&manifest_path);

    if final_files.is_empty() {
        for p in fallback_paths {
            if p.is_file() && !final_files.contains(&p) {
                final_files.push(p);
            }
        }
    }

    if !status.success() {
        let msg = if stderr_buf.is_empty() {
            format!("yt-dlp exited with code {:?}", status.code())
        } else {
            stderr_buf.join(" | ")
        };
        let err = AicutError::new("YTDLP_DOWNLOAD_FAILED", msg).recoverable(
            "Verifique a URL, atualize o yt-dlp em Configurações -> Ferramentas, ou ative Cookies/Impersonate (Chrome) para sites protegidos como Kick.",
        );
        kernel.jobs.fail(&job_id, err.clone());
        return Err(err);
    }

    kernel.jobs.update_progress(
        &job_id,
        JobProgress {
            percent: 96.0,
            stage: "indexing_library".to_string(),
            message: Some("Extraindo metadados e adicionando à Biblioteca...".to_string()),
            ..Default::default()
        },
    );

    let mut indexed_assets = Vec::new();
    for path in final_files {
        if !path.is_file() {
            continue;
        }
        let asset = index_downloaded_file(
            &kernel,
            &path,
            &req,
            ffmpeg_path.as_deref(),
            ffprobe_path.as_deref(),
        )
        .await?;
        indexed_assets.push(asset);
    }

    if indexed_assets.is_empty() {
        let err = AicutError::internal(
            "yt-dlp concluiu, mas nenhum arquivo final foi encontrado na pasta de Downloads.",
        );
        kernel.jobs.fail(&job_id, err.clone());
        return Err(err);
    }

    kernel.jobs.complete(
        &job_id,
        serde_json::json!({
            "count": indexed_assets.len(),
            "assets": indexed_assets,
        }),
    );

    Ok(indexed_assets)
}

fn extract_destination_path(line: &str) -> Option<PathBuf> {
    if let Some(rest) = line.strip_prefix("[Merger] Merging formats into \"") {
        let path = rest.trim_end_matches('"');
        if !path.is_empty() {
            return Some(PathBuf::from(path));
        }
    }
    if let Some(rest) = line.strip_prefix("[ExtractAudio] Destination: ") {
        let path = rest.trim();
        if !path.is_empty() {
            return Some(PathBuf::from(path));
        }
    }
    if let Some(rest) = line.strip_prefix("[download] Destination: ") {
        let path = rest.trim();
        if !path.is_empty() {
            return Some(PathBuf::from(path));
        }
    }
    if let Some(rest) = line.strip_prefix("[download] ") {
        if let Some(path) = rest.strip_suffix(" has already been downloaded") {
            let p = path.trim();
            if !p.is_empty() {
                return Some(PathBuf::from(p));
            }
        }
    }
    None
}

async fn index_downloaded_file(
    kernel: &Arc<AicutKernel>,
    path: &Path,
    req: &DownloadRequest,
    ffmpeg_path: Option<&Path>,
    ffprobe_path: Option<&Path>,
) -> AicutResult<MediaAsset> {
    let id = Uuid::new_v4().to_string();
    let ext = extension_lower(path);
    let file_name = path
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("media")
        .to_string();
    let default_title = path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or(&file_name)
        .to_string();
    let title = req
        .custom_title
        .clone()
        .filter(|s| !s.trim().is_empty())
        .unwrap_or(default_title);

    let file_size_bytes = std::fs::metadata(path).map(|m| m.len()).unwrap_or(0);
    let fingerprint = compute_fast_fingerprint(path).ok();
    let probe = probe_file(ffprobe_path, path, &ext).await;
    let kind = match req.mode {
        DownloadMode::AudioOnly => MediaKind::Audio,
        DownloadMode::VideoOnly => MediaKind::VideoOnly,
        DownloadMode::VideoAudio => probe.kind,
    };

    let thumbnail_path = if kind != MediaKind::Audio {
        generate_thumbnail(
            ffmpeg_path,
            path,
            &kernel.workspace.thumbnails_dir,
            &id,
            probe.metadata.duration_sec,
        )
        .await
        .map(|p| p.to_string_lossy().to_string())
    } else {
        None
    };

    let now = Utc::now();
    let asset = MediaAsset {
        id,
        title,
        kind,
        source_type: MediaSourceType::Download,
        source_url: Some(req.url.trim().to_string()),
        parent_media_id: None,
        file_path: path.to_string_lossy().to_string(),
        file_name,
        extension: ext,
        file_size_bytes,
        fingerprint,
        thumbnail_path,
        metadata: probe.metadata,
        tags: req.tags.clone(),
        notes: String::new(),
        created_at: now,
        updated_at: now,
    };

    kernel.repo.insert_media(&asset)?;
    kernel.events.publish(AicutEvent::MediaCreated(asset.clone()));
    Ok(asset)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn base_req() -> DownloadRequest {
        DownloadRequest {
            url: "https://www.youtube.com/watch?v=dQw4w9WgXcQ".into(),
            mode: DownloadMode::VideoAudio,
            quality: QualityPreset::P1080,
            format_id: None,
            video_codec: VideoCodecPref::H264,
            container: ContainerFormat::Mp4,
            audio_quality: None,
            subtitles: SubtitleOptions::default(),
            embed_thumbnail: false,
            embed_metadata: true,
            embed_chapters: true,
            split_chapters: false,
            clip_start: None,
            clip_end: None,
            force_keyframes_at_cuts: false,
            playlist_mode: PlaylistMode::Single,
            playlist_items: None,
            concurrent_fragments: Some(4),
            rate_limit: Some("10M".into()),
            proxy: None,
            impersonate: Some("chrome".into()),
            cookies_from_browser: Some(BrowserCookieSource::Edge),
            cookies_file: None,
            sponsorblock_remove: vec!["sponsor".into(), "intro".into()],
            filename_template: None,
            custom_title: None,
            tags: vec![],
            extra_args: vec![],
            wait: false,
        }
    }

    #[test]
    fn builds_full_ytdlp_arguments_without_simulate_bug() {
        let req = base_req();
        let manifest = Path::new("/tmp/manifest.txt");
        let args = build_ytdlp_args(&req, Path::new("/tmp/dl"), None, Some(manifest));
        // Critical regression guard: NEVER use `--print` (which implies --simulate & --quiet)
        assert!(!args.contains(&"--print".to_string()));
        assert!(args.contains(&"--no-simulate".to_string()));
        assert!(args.contains(&"--progress".to_string()));
        assert!(args.contains(&"--print-to-file".to_string()));
        assert!(args.contains(&"--impersonate".to_string()));
        assert!(args.contains(&"chrome".to_string()));
        assert!(args.contains(&"--no-playlist".to_string()));
        assert!(args.contains(&"--merge-output-format".to_string()));
        assert!(args.contains(&"mp4".to_string()));
        assert!(args.contains(&"--embed-chapters".to_string()));
        assert!(args.contains(&"--concurrent-fragments".to_string()));
        assert!(args.contains(&"4".to_string()));
        assert!(args.contains(&"--cookies-from-browser".to_string()));
        assert!(args.contains(&"edge".to_string()));
        assert!(args.contains(&"--sponsorblock-remove".to_string()));
    }

    #[test]
    fn builds_video_only_and_audio_only_selectors() {
        let mut vo = base_req();
        vo.mode = DownloadMode::VideoOnly;
        vo.quality = QualityPreset::P2160;
        vo.video_codec = VideoCodecPref::Auto;
        assert_eq!(build_format_selector(&vo), "bestvideo[height<=2160]/bestvideo");

        let mut ao = base_req();
        ao.mode = DownloadMode::AudioOnly;
        ao.container = ContainerFormat::Mp3;
        ao.audio_quality = Some("320K".into());
        let args = build_ytdlp_args(&ao, Path::new("/tmp/dl"), None, None);
        assert!(args.contains(&"-x".to_string()));
        assert!(args.contains(&"--audio-format".to_string()));
        assert!(args.contains(&"mp3".to_string()));
        assert!(args.contains(&"--audio-quality".to_string()));
        assert!(args.contains(&"320K".to_string()));
    }

    #[test]
    fn builds_clip_section_and_subtitles() {
        let mut req = base_req();
        req.clip_start = Some("00:00:30".into());
        req.clip_end = Some("00:01:45".into());
        req.force_keyframes_at_cuts = true;
        req.subtitles = SubtitleOptions {
            enabled: true,
            auto_generated: true,
            languages: vec!["pt".into(), "en".into()],
            embed_in_file: true,
            format: Some("srt".into()),
        };
        let args = build_ytdlp_args(&req, Path::new("/tmp/dl"), None, None);
        assert!(args.contains(&"--download-sections".to_string()));
        assert!(args.contains(&"*00:00:30-00:01:45".to_string()));
        assert!(args.contains(&"--force-keyframes-at-cuts".to_string()));
        assert!(args.contains(&"--write-subs".to_string()));
        assert!(args.contains(&"--write-auto-subs".to_string()));
        assert!(args.contains(&"--embed-subs".to_string()));
        assert!(args.contains(&"pt,en".to_string()));
    }

    #[test]
    fn parses_progress_and_probe_payloads() {
        let prog = parse_progress_line("AICUT_PROGRESS: 42.5%|1048576|2467237|524288.0|3.5").unwrap();
        assert!((prog.percent - 42.5).abs() < 0.01);
        assert_eq!(prog.downloaded_bytes, Some(1048576));
        assert_eq!(prog.total_bytes, Some(2467237));

        let sample = serde_json::json!({
            "title": "Demo Video",
            "uploader": "Aicut Lab",
            "extractor_key": "Youtube",
            "duration": 125.0,
            "thumbnail": "https://example.com/thumb.jpg",
            "formats": [
                {"format_id": "137", "ext": "mp4", "vcodec": "avc1", "acodec": "none", "width": 1920, "height": 1080, "fps": 60.0, "tbr": 4200.0},
                {"format_id": "140", "ext": "m4a", "vcodec": "none", "acodec": "mp4a", "tbr": 128.0}
            ]
        });
        let parsed = parse_probe_json("https://youtu.be/demo", &sample);
        assert_eq!(parsed.title, "Demo Video");
        assert_eq!(parsed.formats.len(), 2);
        assert_eq!(parsed.formats[0].kind, "video_only");
        assert_eq!(parsed.formats[1].kind, "audio_only");
    }
}
