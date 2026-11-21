use chrono::Utc;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::path::{Path, PathBuf};
use tokio::process::Command;
use uuid::Uuid;

use crate::error::{AicutError, AicutResult};
use crate::events::AicutEvent;
use crate::jobs::JobProgress;
use crate::kernel::AicutKernel;
use crate::media::{MediaAsset, MediaKind, MediaMetadata, MediaSourceType};
use crate::process::configure_tokio_command;

pub const SETTING_AUDIO_MASTER_CONFIG: &str = "audio.master_config";

/// Open-source RNNoise neural speech denoising model (`beguiling-drafter`, broadcast voice).
pub const RNNOISE_MODEL_URL: &str =
    "https://raw.githubusercontent.com/GregorR/rnnoise-models/master/beguiling-drafter-2018-08-30/bd.rnnn";

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct AudioMasterConfig {
    /// Active mastering preset: "youtube_auto" | "reels_tiktok" | "podcast_voice" | "off"
    #[serde(default = "default_preset")]
    pub preset: String,
    /// Noise suppression mode: "auto" (RNNoise neural when installed, else adaptive spectral) | "neural_rnnoise" | "spectral" | "off"
    #[serde(default = "default_denoise")]
    pub denoise: String,
    /// Apply gentle broadcast voice clarity & warmth EQ.
    #[serde(default = "default_true")]
    pub voice_eq: bool,
    /// Apply automatic vocal compressor + brickwall limiter (levels whispers vs screams).
    #[serde(default = "default_true")]
    pub auto_compressor: bool,
    /// Target integrated loudness in LUFS (YouTube standard = -14.0 LUFS).
    #[serde(default = "default_target_lufs")]
    pub target_lufs: f64,
    /// Maximum true peak ceiling in dBTP (YouTube standard = -1.0 dBTP).
    #[serde(default = "default_true_peak")]
    pub true_peak_db: f64,
    /// Target Loudness Range (LRA) in LU (YouTube = 11.0, Reels/TikTok = 7.0).
    #[serde(default = "default_lra")]
    pub lra: f64,
}

fn default_preset() -> String {
    "youtube_auto".to_string()
}

fn default_denoise() -> String {
    "auto".to_string()
}

fn default_true() -> bool {
    true
}

fn default_target_lufs() -> f64 {
    -14.0
}

fn default_true_peak() -> f64 {
    -1.0
}

fn default_lra() -> f64 {
    11.0
}

impl Default for AudioMasterConfig {
    fn default() -> Self {
        Self::from_preset("youtube_auto")
    }
}

impl AudioMasterConfig {
    pub fn from_preset(preset: &str) -> Self {
        match preset.trim().to_lowercase().as_str() {
            "reels_tiktok" => Self {
                preset: "reels_tiktok".to_string(),
                denoise: "auto".to_string(),
                voice_eq: true,
                auto_compressor: true,
                target_lufs: -14.0,
                true_peak_db: -1.0,
                lra: 7.0,
            },
            "podcast_voice" => Self {
                preset: "podcast_voice".to_string(),
                denoise: "auto".to_string(),
                voice_eq: true,
                auto_compressor: true,
                target_lufs: -16.0,
                true_peak_db: -1.5,
                lra: 9.0,
            },
            "off" | "raw" | "none" => Self {
                preset: "off".to_string(),
                denoise: "off".to_string(),
                voice_eq: false,
                auto_compressor: false,
                target_lufs: -14.0,
                true_peak_db: -1.0,
                lra: 11.0,
            },
            _ => Self {
                preset: "youtube_auto".to_string(),
                denoise: "auto".to_string(),
                voice_eq: true,
                auto_compressor: true,
                target_lufs: -14.0,
                true_peak_db: -1.0,
                lra: 11.0,
            },
        }
    }

    pub fn is_enabled(&self) -> bool {
        !matches!(
            self.preset.trim().to_lowercase().as_str(),
            "off" | "raw" | "none"
        )
    }

    pub fn sanitized(mut self) -> Self {
        self.target_lufs = self.target_lufs.clamp(-24.0, -9.0);
        self.true_peak_db = self.true_peak_db.clamp(-3.0, -0.1);
        self.lra = self.lra.clamp(4.0, 18.0);
        if !matches!(
            self.preset.as_str(),
            "youtube_auto" | "reels_tiktok" | "podcast_voice" | "off"
        ) {
            self.preset = "youtube_auto".to_string();
        }
        if !matches!(
            self.denoise.as_str(),
            "auto" | "neural_rnnoise" | "spectral" | "off"
        ) {
            self.denoise = "auto".to_string();
        }
        self
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct LoudnormStats {
    pub input_i: f64,
    pub input_tp: f64,
    pub input_lra: f64,
    pub input_thresh: f64,
    pub target_offset: f64,
}

/// Loads the active automatic audio mastering configuration (defaults to YouTube -14 LUFS auto).
pub fn load_audio_master_config(kernel: &AicutKernel) -> AudioMasterConfig {
    kernel
        .repo
        .get_setting(SETTING_AUDIO_MASTER_CONFIG)
        .ok()
        .flatten()
        .and_then(|v| serde_json::from_value::<AudioMasterConfig>(v).ok())
        .map(AudioMasterConfig::sanitized)
        .unwrap_or_default()
}

/// Persists the automatic audio mastering configuration.
pub fn save_audio_master_config(
    kernel: &AicutKernel,
    cfg: &AudioMasterConfig,
) -> AicutResult<AudioMasterConfig> {
    let clean = cfg.clone().sanitized();
    let val = serde_json::to_value(&clean)
        .map_err(|e| AicutError::internal(format!("Failed serializing audio master config: {e}")))?;
    kernel.repo.set_setting(SETTING_AUDIO_MASTER_CONFIG, &val)?;
    Ok(clean)
}

/// Returns the local path where the optional RNNoise neural speech model (`bd.rnnn`) is stored.
pub fn rnnoise_model_path(models_dir: &Path) -> PathBuf {
    models_dir.join("audio").join("bd.rnnn")
}

/// Returns the RNNoise model path only if it is already downloaded and non-empty on disk.
pub fn installed_rnnoise_model(models_dir: &Path) -> Option<PathBuf> {
    let p = rnnoise_model_path(models_dir);
    let ok = std::fs::metadata(&p)
        .map(|m| m.is_file() && m.len() > 10_000)
        .unwrap_or(false);
    if ok {
        Some(p)
    } else {
        None
    }
}

/// Downloads the lightweight (~85 KB) open-source RNNoise neural speech model for FFmpeg `arnndn`.
pub async fn download_rnnoise_model(kernel: &AicutKernel) -> AicutResult<PathBuf> {
    let _directory = crate::workspace::lock_downloads(&kernel.workspace.models_dir)?;
    let dest = rnnoise_model_path(&kernel.workspace.models_dir);
    if let Some(parent) = dest.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let resp = kernel
        .tools
        .http_client()
        .get(RNNOISE_MODEL_URL)
        .send()
        .await
        .map_err(|e| AicutError::internal(format!("Falha ao baixar modelo RNNoise: {e}")))?;
    if !resp.status().is_success() {
        return Err(AicutError::internal(format!(
            "Download do modelo RNNoise retornou HTTP {}",
            resp.status()
        )));
    }
    let bytes = resp
        .bytes()
        .await
        .map_err(|e| AicutError::internal(format!("Erro ao ler modelo RNNoise: {e}")))?;
    if bytes.len() < 10_000 {
        return Err(AicutError::internal(
            "Arquivo do modelo RNNoise veio incompleto.",
        ));
    }
    let stage = crate::workspace::download_stage(&kernel.workspace.models_dir)?;
    let tmp = stage.path().join("model");
    std::fs::write(&tmp, &bytes)?;
    std::fs::rename(&tmp, &dest)?;
    Ok(dest)
}

fn escape_ffmpeg_filter_value(path: &Path) -> String {
    path.to_string_lossy()
        .replace('\\', "/")
        .replace(':', "\\:")
        .replace('\'', "\\'")
}

/// Builds the pre-loudnorm audio polish stages (highpass, neural/spectral denoise, voice EQ,
/// automatic vocal compressor & brickwall limiter).
pub fn build_pre_loudnorm_stages(
    cfg: &AudioMasterConfig,
    rnnoise_path: Option<&Path>,
) -> Vec<String> {
    let mut stages: Vec<String> = Vec::new();
    if !cfg.is_enabled() {
        return stages;
    }

    // 1. Cut subsonic mic rumble, AC hum & wind plosives cleanly
    stages.push("highpass=f=75".to_string());

    // 2. Automatic noise reduction (RNNoise neural speech model when installed, else adaptive spectral FFT)
    match cfg.denoise.as_str() {
        "off" => {}
        "spectral" => {
            stages.push("afftdn=nr=10:nf=-25:tn=1".to_string());
        }
        _ => {
            if let Some(model_p) = rnnoise_path {
                let escaped = escape_ffmpeg_filter_value(model_p);
                stages.push(format!("arnndn=m='{escaped}':mix=0.85"));
            } else {
                stages.push("afftdn=nr=10:nf=-25:tn=1".to_string());
            }
        }
    }

    // 3. Broadcast voice clarity & warmth EQ
    if cfg.voice_eq {
        stages.push("equalizer=f=180:t=q:w=1.2:g=1.4".to_string());
        stages.push("equalizer=f=320:t=q:w=1.4:g=-1.8".to_string());
        stages.push("equalizer=f=3200:t=q:w=1.5:g=2.0".to_string());
        stages.push("equalizer=f=8500:t=q:w=1.2:g=0.9".to_string());
    }

    // 4. Automatic vocal dynamics leveling (keeps whispers clear and tames streamer screams)
    if cfg.auto_compressor {
        let comp = match cfg.preset.as_str() {
            "reels_tiktok" => {
                "acompressor=threshold=-21dB:ratio=3.5:attack=6:release=100:makeup=2.5"
            }
            "podcast_voice" => {
                "acompressor=threshold=-19dB:ratio=2.4:attack=10:release=150:makeup=1.5"
            }
            _ => "acompressor=threshold=-20dB:ratio=3.0:attack=8:release=120:makeup=2.0",
        };
        stages.push(comp.to_string());
        stages.push("alimiter=limit=-1.2dB:attack=5:release=50".to_string());
    }

    stages
}

/// Builds the complete comma-separated FFmpeg audio filter chain ready to append after `[abase]`
/// (returns `None` when mastering preset is `"off"`).
pub fn build_mastered_audio_filter_chain(
    cfg: &AudioMasterConfig,
    rnnoise_path: Option<&Path>,
    measured: Option<&LoudnormStats>,
) -> Option<String> {
    if !cfg.is_enabled() {
        return None;
    }
    let mut stages = build_pre_loudnorm_stages(cfg, rnnoise_path);

    let loudnorm = if let Some(m) = measured {
        format!(
            "loudnorm=I={:.1}:TP={:.1}:LRA={:.1}:measured_I={:.2}:measured_TP={:.2}:measured_LRA={:.2}:measured_thresh={:.2}:offset={:.2}:linear=true:print_format=summary",
            cfg.target_lufs,
            cfg.true_peak_db,
            cfg.lra,
            m.input_i,
            m.input_tp,
            m.input_lra,
            m.input_thresh,
            m.target_offset
        )
    } else {
        format!(
            "loudnorm=I={:.1}:TP={:.1}:LRA={:.1}",
            cfg.target_lufs, cfg.true_peak_db, cfg.lra
        )
    };
    stages.push(loudnorm);
    // FFmpeg's loudnorm internally upsamples to 192 kHz; cleanly resample to 48 kHz broadcast rate.
    stages.push("aresample=48000".to_string());
    Some(stages.join(","))
}

/// Parses FFmpeg's `loudnorm=print_format=json` stderr summary block.
pub fn parse_loudnorm_json_from_stderr(stderr: &str) -> Option<LoudnormStats> {
    // Locate the last `{ ... }` JSON object containing `"input_i"`
    let key_idx = stderr.rfind("\"input_i\"")?;
    let start = stderr[..key_idx].rfind('{')?;
    let end = stderr[key_idx..].find('}')? + key_idx;
    let slice = &stderr[start..=end];
    let val: Value = serde_json::from_str(slice).ok()?;

    let parse_f = |k: &str| -> Option<f64> {
        let s = val.get(k)?.as_str()?;
        let f = s.trim().parse::<f64>().ok()?;
        if f.is_finite() {
            Some(f)
        } else {
            None
        }
    };

    let input_i = parse_f("input_i")?;
    let input_tp = parse_f("input_tp")?;
    let input_lra = parse_f("input_lra")?;
    let input_thresh = parse_f("input_thresh")?;
    let target_offset = parse_f("target_offset").unwrap_or(0.0);

    // Reject completely silent clips (< -65 LUFS) so linear loudnorm doesn't boost silence noise floor
    if input_i < -65.0 {
        return None;
    }

    Some(LoudnormStats {
        input_i,
        input_tp,
        input_lra,
        input_thresh,
        target_offset,
    })
}

/// Runs fast Pass 1 EBU R128 loudness analysis on the target clip window (or whole file) so Pass 2
/// applies linear broadcast-grade YouTube normalization (`-14 LUFS`, `-1.0 dBTP`).
pub async fn measure_audio_loudness_pass1(
    ffmpeg_bin: &Path,
    input_path: &Path,
    start_sec: Option<f64>,
    duration_sec: Option<f64>,
    cfg: &AudioMasterConfig,
    rnnoise_path: Option<&Path>,
) -> Option<LoudnormStats> {
    if !cfg.is_enabled() {
        return None;
    }
    let mut stages = build_pre_loudnorm_stages(cfg, rnnoise_path);
    stages.push(format!(
        "loudnorm=I={:.1}:TP={:.1}:LRA={:.1}:print_format=json",
        cfg.target_lufs, cfg.true_peak_db, cfg.lra
    ));
    let af = stages.join(",");

    let mut cmd = Command::new(ffmpeg_bin);
    configure_tokio_command(&mut cmd);
    cmd.arg("-hide_banner");
    if let Some(s) = start_sec {
        cmd.args(["-ss", &format!("{:.3}", s.max(0.0))]);
    }
    if let Some(d) = duration_sec {
        cmd.args(["-t", &format!("{:.3}", d.max(0.5))]);
    }
    cmd.arg("-i")
        .arg(input_path)
        .args(["-vn", "-af", &af, "-f", "null", "-"]);

    let out = cmd.output().await.ok()?;
    let stderr = String::from_utf8_lossy(&out.stderr);
    parse_loudnorm_json_from_stderr(&stderr)
}

/// Convenience helper for renderers (Cortes Shorts & Motion Studio): loads active user preset,
/// performs fast Pass 1 loudness measurement on the exact clip span, and returns the ready-to-use
/// FFmpeg audio filter chain (or `None` when disabled).
pub async fn prepare_clip_audio_master_chain(
    kernel: &AicutKernel,
    ffmpeg_bin: &Path,
    input_path: &Path,
    start_sec: Option<f64>,
    duration_sec: Option<f64>,
) -> Option<String> {
    let cfg = load_audio_master_config(kernel);
    if !cfg.is_enabled() {
        return None;
    }
    let rnnoise = installed_rnnoise_model(&kernel.workspace.models_dir);
    let measured = measure_audio_loudness_pass1(
        ffmpeg_bin,
        input_path,
        start_sec,
        duration_sec,
        &cfg,
        rnnoise.as_deref(),
    )
    .await;
    build_mastered_audio_filter_chain(&cfg, rnnoise.as_deref(), measured.as_ref())
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct AudioEnhanceResult {
    pub asset: MediaAsset,
    pub preset_used: String,
    pub rnnoise_used: bool,
    pub before_lufs: Option<f64>,
    pub before_true_peak_db: Option<f64>,
    pub target_lufs: f64,
    pub target_true_peak_db: f64,
}

fn sanitize_audio_slug(raw: &str) -> String {
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
    let short: String = trimmed.chars().take(48).collect();
    if short.is_empty() {
        "aicut-audio".to_string()
    } else {
        short
    }
}

/// Applies automatic YouTube/broadcast AI audio mastering (2-pass EBU R128 loudness normalization,
/// RNNoise neural or adaptive spectral denoising, broadcast voice EQ, vocal compressor & limiter)
/// to any Library video (lossless `-c:v copy`!) or audio asset and saves the mastered version to
/// the Library.
pub async fn enhance_media_audio_to_library(
    kernel: &AicutKernel,
    media_id: &str,
    override_cfg: Option<AudioMasterConfig>,
    job_id: Option<&str>,
) -> AicutResult<AudioEnhanceResult> {
    let source = kernel
        .repo
        .get_media(media_id)?
        .ok_or_else(|| AicutError::not_found(format!("Mídia '{}' não encontrada", media_id)))?;
    let ffmpeg_bin = kernel.tools.resolve_tool("ffmpeg")?;

    let mut cfg = override_cfg
        .unwrap_or_else(|| load_audio_master_config(kernel))
        .sanitized();
    if !cfg.is_enabled() {
        cfg = AudioMasterConfig::from_preset("youtube_auto");
    }

    let rnnoise_path = installed_rnnoise_model(&kernel.workspace.models_dir);
    let rnnoise_used = rnnoise_path.is_some() && !matches!(cfg.denoise.as_str(), "off" | "spectral");

    if let Some(jid) = job_id {
        kernel.jobs.update_progress(
            jid,
            JobProgress {
                percent: 20.0,
                stage: "measuring_loudness".into(),
                message: Some("Medindo volume original (LUFS / True Peak)...".into()),
                ..Default::default()
            },
        );
    }

    let measured = measure_audio_loudness_pass1(
        &ffmpeg_bin,
        Path::new(&source.file_path),
        None,
        None,
        &cfg,
        rnnoise_path.as_deref(),
    )
    .await;

    let af_chain = build_mastered_audio_filter_chain(&cfg, rnnoise_path.as_deref(), measured.as_ref())
        .unwrap_or_else(|| {
            format!(
                "loudnorm=I={:.1}:TP={:.1}:LRA={:.1},aresample=48000",
                cfg.target_lufs, cfg.true_peak_db, cfg.lra
            )
        });

    if let Some(jid) = job_id {
        let meas_txt = measured
            .as_ref()
            .map(|m| {
                format!(
                    "Áudio original: {:.1} LUFS (pico {:.1} dBTP) → alvo {:.1} LUFS / {:.1} dBTP{}",
                    m.input_i,
                    m.input_tp,
                    cfg.target_lufs,
                    cfg.true_peak_db,
                    if rnnoise_used { " + RNNoise neural" } else { "" }
                )
            })
            .unwrap_or_else(|| {
                format!(
                    "Aplicando masterização automática → {:.1} LUFS / {:.1} dBTP...",
                    cfg.target_lufs, cfg.true_peak_db
                )
            });
        kernel.jobs.append_log(jid, meas_txt);
        kernel.jobs.update_progress(
            jid,
            JobProgress {
                percent: 58.0,
                stage: "mastering_audio".into(),
                message: Some("Aplicando tratamento de áudio e normalização YouTube...".into()),
                ..Default::default()
            },
        );
    }

    let out_dir = kernel.workspace.exports_dir.join("AudioMaster");
    std::fs::create_dir_all(&out_dir)?;
    let slug = sanitize_audio_slug(&source.title);
    let short_id = &Uuid::new_v4().to_string()[..8];
    let is_video = source.kind == MediaKind::Video;
    let ext = if is_video { "mp4" } else { "m4a" };
    let file_name = format!("{slug}-audio-{short_id}.{ext}");
    let out_path = out_dir.join(&file_name);

    let run_pass = |chain: &str| {
        let mut cmd = Command::new(&ffmpeg_bin);
        configure_tokio_command(&mut cmd);
        cmd.args(["-y", "-i"]).arg(&source.file_path);
        if is_video {
            cmd.args(["-map", "0:v:0", "-map", "0:a:0?", "-c:v", "copy"]);
        } else {
            cmd.args(["-vn"]);
        }
        cmd.args([
            "-af",
            chain,
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            "-ar",
            "48000",
            "-movflags",
            "+faststart",
        ])
        .arg(&out_path);
        cmd
    };

    let mut out = run_pass(&af_chain)
        .output()
        .await
        .map_err(|e| AicutError::internal(format!("Falha ao executar FFmpeg Audio Master: {e}")))?;

    if !out.status.success() || !out_path.is_file() {
        let fallback_chain = format!(
            "loudnorm=I={:.1}:TP={:.1}:LRA={:.1},aresample=48000",
            cfg.target_lufs, cfg.true_peak_db, cfg.lra
        );
        out = run_pass(&fallback_chain)
            .output()
            .await
            .map_err(|e| AicutError::internal(format!("Falha no fallback de masterização: {e}")))?;
        if !out.status.success() || !out_path.is_file() {
            let stderr = String::from_utf8_lossy(&out.stderr);
            return Err(AicutError::internal(format!(
                "FFmpeg falhou ao masterizar áudio: {stderr}"
            )));
        }
    }

    let file_size_bytes = std::fs::metadata(&out_path).map(|m| m.len()).unwrap_or(0);
    let now = Utc::now();
    let asset = MediaAsset {
        id: Uuid::new_v4().to_string(),
        title: format!("{} [Áudio Master]", source.title),
        kind: source.kind.clone(),
        source_type: MediaSourceType::Generated,
        source_url: source.source_url.clone(),
        parent_media_id: Some(source.id.clone()),
        file_path: out_path.to_string_lossy().to_string(),
        file_name,
        extension: ext.to_string(),
        file_size_bytes,
        fingerprint: None,
        thumbnail_path: source.thumbnail_path.clone(),
        metadata: MediaMetadata {
            audio_codec: Some("aac".to_string()),
            format_name: Some(ext.to_string()),
            ..source.metadata.clone()
        },
        tags: vec![
            "audio-master".to_string(),
            format!("preset:{}", cfg.preset),
            format!("lufs:{:.0}", cfg.target_lufs),
        ],
        notes: format!(
            "Masterizado automaticamente ({} · alvo {:.1} LUFS / {:.1} dBTP{})",
            cfg.preset,
            cfg.target_lufs,
            cfg.true_peak_db,
            if rnnoise_used { " · RNNoise Neural" } else { "" }
        ),
        created_at: now,
        updated_at: now,
        deleted_at: None,
    };

    kernel.repo.insert_media(&asset)?;
    kernel.events.publish(AicutEvent::MediaCreated(asset.clone()));

    Ok(AudioEnhanceResult {
        asset,
        preset_used: cfg.preset,
        rnnoise_used,
        before_lufs: measured.as_ref().map(|m| m.input_i),
        before_true_peak_db: measured.as_ref().map(|m| m.input_tp),
        target_lufs: cfg.target_lufs,
        target_true_peak_db: cfg.true_peak_db,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn builds_youtube_14_lufs_two_pass_and_single_pass_chains() {
        let cfg = AudioMasterConfig::from_preset("youtube_auto");
        let stats = LoudnormStats {
            input_i: -21.4,
            input_tp: -3.2,
            input_lra: 12.1,
            input_thresh: -32.0,
            target_offset: 0.4,
        };
        let chain = build_mastered_audio_filter_chain(&cfg, None, Some(&stats)).unwrap();
        assert!(chain.contains("highpass=f=75"));
        assert!(chain.contains("afftdn="));
        assert!(chain.contains("acompressor="));
        assert!(chain.contains("loudnorm=I=-14.0:TP=-1.0:LRA=11.0:measured_I=-21.40"));
        assert!(chain.contains("linear=true"));
        assert!(chain.ends_with("aresample=48000"));

        let rn = Path::new("/home/user/.local/share/Aicut/models/audio/bd.rnnn");
        let neural_chain = build_mastered_audio_filter_chain(&cfg, Some(rn), None).unwrap();
        assert!(neural_chain.contains("arnndn=m='/home/user/.local/share/Aicut/models/audio/bd.rnnn'"));

        let off = AudioMasterConfig::from_preset("off");
        assert!(build_mastered_audio_filter_chain(&off, None, None).is_none());
    }

    #[test]
    fn parses_ffmpeg_loudnorm_json_from_stderr() {
        let stderr = r#"
[Parsed_loudnorm_4 @ 0x55a]
{
	"input_i" : "-22.84",
	"input_tp" : "-2.15",
	"input_lra" : "9.80",
	"input_thresh" : "-33.40",
	"output_i" : "-14.00",
	"output_tp" : "-1.00",
	"output_lra" : "8.10",
	"output_thresh" : "-24.50",
	"normalization_type" : "dynamic",
	"target_offset" : "0.12"
}
"#;
        let parsed = parse_loudnorm_json_from_stderr(stderr).unwrap();
        assert!((parsed.input_i - (-22.84)).abs() < 0.01);
        assert!((parsed.input_tp - (-2.15)).abs() < 0.01);
        assert!((parsed.target_offset - 0.12).abs() < 0.01);
    }
}
