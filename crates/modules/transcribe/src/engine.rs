use aicut_core::{
    configure_linux_gpu_env_tokio, configure_tokio_command, ActorKind, AicutError, AicutEvent,
    AicutKernel, AicutResult, JobProgress, MediaAsset, TranscriptRecord, TranscriptSegment,
    TranscriptWord,
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

use crate::exporters::{export_and_save_to_library, ExportFormat};
use crate::models::resolve_model_path;

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct TranscribeRequest {
    /// ID of the video or audio asset in the Aicut Library to transcribe.
    pub media_id: String,
    /// Whisper GGML model ID (e.g. "large-v3-turbo-q5_0", "large-v3-turbo", "small").
    #[serde(default = "default_model_id")]
    pub model_id: String,
    /// Language code ("auto", "pt", "en", "es", "fr", "de", "ja", etc.).
    #[serde(default = "default_language")]
    pub language: String,
    /// Translate speech to English while transcribing.
    #[serde(default)]
    pub translate_to_english: bool,
    /// Enable word-level timestamps & tighter subtitle splits.
    #[serde(default = "default_true")]
    pub word_timestamps: bool,
    /// Optional max characters per subtitle segment (e.g. 42 for Netflix/Reels style).
    #[serde(default)]
    pub max_segment_chars: Option<u32>,
    /// Optional initial prompt with proper nouns, brand names, or vocabulary hints.
    #[serde(default)]
    pub initial_prompt: Option<String>,
    /// Formats to automatically generate and save into the Library when transcription finishes.
    #[serde(default = "default_export_formats")]
    pub export_formats: Vec<ExportFormat>,
    /// Wait synchronously for transcription to finish (recommended for AI chats / CLI).
    #[serde(default)]
    pub wait: bool,
}

fn default_model_id() -> String {
    "large-v3-turbo-q5_0".to_string()
}

fn default_language() -> String {
    "auto".to_string()
}

fn default_true() -> bool {
    true
}

fn default_export_formats() -> Vec<ExportFormat> {
    vec![ExportFormat::Srt]
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct TranscribeResponse {
    pub job_id: String,
    pub status: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub transcript: Option<TranscriptRecord>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub exported_assets: Vec<MediaAsset>,
}

pub async fn start_transcription(
    kernel: Arc<AicutKernel>,
    req: TranscribeRequest,
    actor: ActorKind,
) -> AicutResult<TranscribeResponse> {
    let source = kernel
        .repo
        .get_media(&req.media_id)?
        .ok_or_else(|| AicutError::not_found(format!("Media '{}' not found in Library", req.media_id)))?;

    let whisper_bin = kernel.tools.resolve_tool("whisper")?;
    let ffmpeg_bin = kernel.tools.resolve_tool("ffmpeg")?;
    let model_path = resolve_model_path(&kernel, &req.model_id)?;

    let (job, cancel) = kernel.jobs.create_job(
        "transcribe.run",
        format!("Transcrever '{}' ({})", source.title, req.model_id),
        "transcribe",
        actor,
    );
    let job_id = job.id.clone();

    let wait = req.wait;
    let handle = tokio::spawn(execute_transcription_task(
        kernel.clone(),
        job_id.clone(),
        source,
        whisper_bin,
        ffmpeg_bin,
        model_path,
        req,
        cancel,
    ));

    if wait {
        let (transcript, exported_assets) = handle
            .await
            .map_err(|e| AicutError::internal(format!("Transcription task panicked: {e}")))??;
        Ok(TranscribeResponse {
            job_id,
            status: "completed".to_string(),
            transcript: Some(transcript),
            exported_assets,
        })
    } else {
        Ok(TranscribeResponse {
            job_id,
            status: "queued".to_string(),
            transcript: None,
            exported_assets: Vec::new(),
        })
    }
}

async fn execute_transcription_task(
    kernel: Arc<AicutKernel>,
    job_id: String,
    source: MediaAsset,
    whisper_bin: PathBuf,
    ffmpeg_bin: PathBuf,
    model_path: PathBuf,
    req: TranscribeRequest,
    cancel: CancellationToken,
) -> AicutResult<(TranscriptRecord, Vec<MediaAsset>)> {
    kernel.jobs.mark_running(&job_id, "extracting_audio");
    kernel.jobs.update_progress(
        &job_id,
        JobProgress {
            percent: 5.0,
            stage: "extracting_audio".into(),
            message: Some("Extraindo áudio WAV 16kHz via FFmpeg...".into()),
            ..Default::default()
        },
    );

    let tmp_dir = std::env::temp_dir().join(format!("aicut-whisper-{job_id}"));
    std::fs::create_dir_all(&tmp_dir)?;
    let wav_path = tmp_dir.join("input_16k.wav");
    let out_prefix = tmp_dir.join("transcript");
    let out_json = tmp_dir.join("transcript.json");

    let run_res: AicutResult<(TranscriptRecord, Vec<MediaAsset>)> = async {
        // 1. Convert any video/audio file into 16kHz mono PCM WAV for whisper.cpp
        let mut ff_cmd = Command::new(&ffmpeg_bin);
        configure_tokio_command(&mut ff_cmd);
        let ff_out = ff_cmd
            .args(["-y", "-i"])
            .arg(&source.file_path)
            .args(["-vn", "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le"])
            .arg(&wav_path)
            .output()
            .await
            .map_err(|e| AicutError::internal(format!("Failed running FFmpeg: {e}")))?;

        if !ff_out.status.success() || !wav_path.is_file() {
            let err_txt = String::from_utf8_lossy(&ff_out.stderr).trim().to_string();
            return Err(AicutError::new(
                "FFMPEG_AUDIO_EXTRACT_FAILED",
                format!("Could not extract audio track from '{}': {err_txt}", source.title),
            ));
        }

        if cancel.is_cancelled() {
            return Err(AicutError::cancelled("Transcription cancelled."));
        }

        // 2. Run whisper-cli with full JSON output (-ojf) and live progress (-pp)
        let threads = kernel.hardware.logical_cores.clamp(1, 16);
        let mut args: Vec<String> = vec![
            "-m".into(),
            model_path.to_string_lossy().to_string(),
            "-f".into(),
            wav_path.to_string_lossy().to_string(),
            "-l".into(),
            req.language.trim().to_string(),
            "-t".into(),
            threads.to_string(),
            "-ojf".into(),
            "-pp".into(),
            "-of".into(),
            out_prefix.to_string_lossy().to_string(),
        ];

        if req.translate_to_english {
            args.push("-tr".into());
        }
        if req.word_timestamps {
            args.push("-sow".into());
        }
        if let Some(ml) = req.max_segment_chars.filter(|&v| v > 0) {
            args.push("-ml".into());
            args.push(ml.to_string());
        }
        if let Some(ref prompt) = req.initial_prompt {
            if !prompt.trim().is_empty() {
                args.push("--prompt".into());
                args.push(prompt.trim().to_string());
            }
        }

        kernel.jobs.update_progress(
            &job_id,
            JobProgress {
                percent: 12.0,
                stage: "transcribing".into(),
                message: Some(format!("Transcrevendo com {}...", req.model_id)),
                ..Default::default()
            },
        );
        kernel
            .jobs
            .append_log(&job_id, format!("Executando whisper-cli ({})", whisper_bin.display()));

        let mut cmd = Command::new(&whisper_bin);
        configure_linux_gpu_env_tokio(&mut cmd, &whisper_bin);
        let mut child = cmd
            .args(&args)
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .map_err(|e| AicutError::internal(format!("Failed launching whisper-cli: {e}")))?;

        let stdout = child.stdout.take().expect("piped stdout");
        let stderr = child.stderr.take().expect("piped stderr");
        let mut stdout_lines = BufReader::new(stdout).lines();
        let mut stderr_lines = BufReader::new(stderr).lines();
        let mut stdout_done = false;
        let mut stderr_done = false;
        let mut stderr_buf = Vec::new();

        while !(stdout_done && stderr_done) {
            tokio::select! {
                _ = cancel.cancelled() => {
                    let _ = child.kill().await;
                    return Err(AicutError::cancelled("Transcription cancelled by user/AI."));
                }
                line = stdout_lines.next_line(), if !stdout_done => {
                    match line {
                        Ok(Some(l)) => {
                            let trimmed = l.trim();
                            if !trimmed.is_empty() {
                                if let Some(pct) = parse_whisper_progress(trimmed) {
                                    kernel.jobs.update_progress(&job_id, JobProgress {
                                        percent: 12.0 + pct * 0.80,
                                        stage: "transcribing".into(),
                                        message: Some(format!("Transcrevendo áudio ({pct:.0}%)...")),
                                        ..Default::default()
                                    });
                                } else {
                                    kernel.jobs.append_log(&job_id, trimmed);
                                }
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
                                if let Some(pct) = parse_whisper_progress(trimmed) {
                                    kernel.jobs.update_progress(&job_id, JobProgress {
                                        percent: 12.0 + pct * 0.80,
                                        stage: "transcribing".into(),
                                        message: Some(format!("Transcrevendo áudio ({pct:.0}%)...")),
                                        ..Default::default()
                                    });
                                } else {
                                    stderr_buf.push(trimmed.to_string());
                                    kernel.jobs.append_log(&job_id, trimmed);
                                }
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
            .map_err(|e| AicutError::internal(format!("Failed awaiting whisper-cli: {e}")))?;

        if !status.success() || !out_json.is_file() {
            let msg = if stderr_buf.is_empty() {
                format!("whisper-cli exited with code {:?}", status.code())
            } else {
                stderr_buf
                    .into_iter()
                    .rev()
                    .take(6)
                    .collect::<Vec<_>>()
                    .into_iter()
                    .rev()
                    .collect::<Vec<_>>()
                    .join(" | ")
            };
            return Err(AicutError::new("WHISPER_FAILED", msg).recoverable(
                "Verify whisper.cpp and the selected model are installed properly in Settings -> Tools.",
            ));
        }

        // 3. Parse whisper.cpp JSON output into canonical TranscriptRecord
        let raw = std::fs::read_to_string(&out_json)?;
        let parsed_json: Value = serde_json::from_str(&raw)?;
        let record = parse_whisper_json(&parsed_json, &source, &req.model_id, &req.language);

        kernel.repo.insert_transcript(&record)?;
        kernel
            .events
            .publish(AicutEvent::TranscriptCreated(record.clone()));

        // 4. Auto-export requested subtitle/timeline formats directly into the Library
        kernel.jobs.update_progress(
            &job_id,
            JobProgress {
                percent: 95.0,
                stage: "exporting".into(),
                message: Some(format!(
                    "Salvando {} formato(s) na Biblioteca...",
                    req.export_formats.len()
                )),
                ..Default::default()
            },
        );

        let mut exported_assets = Vec::new();
        for fmt in &req.export_formats {
            let asset = export_and_save_to_library(&kernel, &record, Some(&source), *fmt, None)?;
            exported_assets.push(asset);
        }

        Ok((record, exported_assets))
    }
    .await;

    let _ = std::fs::remove_dir_all(&tmp_dir);

    match run_res {
        Ok((record, exported)) => {
            kernel.jobs.complete(
                &job_id,
                serde_json::json!({
                    "transcript_id": record.id,
                    "segments": record.segments.len(),
                    "exported_count": exported.len(),
                }),
            );
            Ok((record, exported))
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

pub fn parse_whisper_progress(line: &str) -> Option<f64> {
    let idx = line.find("progress =")?;
    let rest = &line[idx + "progress =".len()..];
    let num_str = rest.trim().split('%').next()?.trim();
    num_str.parse::<f64>().ok().map(|v| v.clamp(0.0, 100.0))
}

pub fn parse_whisper_json(
    json: &Value,
    source: &MediaAsset,
    model_id: &str,
    requested_lang: &str,
) -> TranscriptRecord {
    let detected_lang = json
        .get("result")
        .and_then(|r| r.get("language"))
        .and_then(|v| v.as_str())
        .filter(|s| !s.is_empty())
        .unwrap_or(requested_lang)
        .to_string();

    let mut segments = Vec::new();
    let mut full_parts = Vec::new();

    if let Some(arr) = json.get("transcription").and_then(|v| v.as_array()) {
        for (idx, item) in arr.iter().enumerate() {
            let text = item
                .get("text")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .trim()
                .to_string();
            if text.is_empty() {
                continue;
            }
            let start_ms = item
                .get("offsets")
                .and_then(|o| o.get("from"))
                .and_then(|v| v.as_u64())
                .unwrap_or(0);
            let end_ms = item
                .get("offsets")
                .and_then(|o| o.get("to"))
                .and_then(|v| v.as_u64())
                .unwrap_or(start_ms + 500);

            let words = extract_words_from_tokens(item);
            full_parts.push(text.clone());
            segments.push(TranscriptSegment {
                index: idx,
                start_ms,
                end_ms,
                text,
                words,
            });
        }
    }

    let duration_sec = source
        .metadata
        .duration_sec
        .or_else(|| segments.last().map(|s| s.end_ms as f64 / 1000.0));

    TranscriptRecord {
        id: Uuid::new_v4().to_string(),
        media_id: source.id.clone(),
        media_title: source.title.clone(),
        model_id: model_id.to_string(),
        language: detected_lang,
        duration_sec,
        full_text: full_parts.join(" "),
        segments,
        created_at: Utc::now(),
    }
}

/// Reconstructs clean word-level timestamps from whisper.cpp's subword `tokens` array.
fn extract_words_from_tokens(seg: &Value) -> Vec<TranscriptWord> {
    let Some(tokens) = seg.get("tokens").and_then(|v| v.as_array()) else {
        return Vec::new();
    };
    let mut words: Vec<TranscriptWord> = Vec::new();
    for tok in tokens {
        let raw = tok.get("text").and_then(|v| v.as_str()).unwrap_or("");
        if raw.is_empty() || (raw.starts_with('[') && raw.ends_with(']')) {
            continue;
        }
        let start_ms = tok
            .get("offsets")
            .and_then(|o| o.get("from"))
            .and_then(|v| v.as_u64())
            .unwrap_or(0);
        let end_ms = tok
            .get("offsets")
            .and_then(|o| o.get("to"))
            .and_then(|v| v.as_u64())
            .unwrap_or(start_ms);
        let prob = tok.get("p").and_then(|v| v.as_f64()).map(|p| p as f32);

        let starts_new = raw.starts_with(' ') || words.is_empty();
        let piece = raw.trim();
        if piece.is_empty() {
            continue;
        }

        if starts_new {
            words.push(TranscriptWord {
                start_ms,
                end_ms: end_ms.max(start_ms),
                text: piece.to_string(),
                probability: prob,
            });
        } else if let Some(last) = words.last_mut() {
            last.text.push_str(piece);
            last.end_ms = last.end_ms.max(end_ms);
        }
    }
    words
}

pub fn check_binary_present(path: &Path) -> bool {
    path.is_file()
}
