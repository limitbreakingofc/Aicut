use aicut_core::{
    configure_std_command, AccelBackend, AicutError, AicutKernel, AicutResult, JobProgress,
    TranscriptRecord,
};
use base64::Engine;
use serde_json::Value;
use std::net::TcpListener;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use tokio_util::sync::CancellationToken;

use crate::frames::{
    extract_candidate_frames, extract_storyboard_sheets, format_transcript_window, plan_windows,
    TimeWindow,
};
use crate::prompts::{
    build_pass_a_prompt, build_pass_b_prompt, extract_json_object, CandidateWindow, PassAResponse,
    PassBResponse, RawClipPlan,
};

/// RAII guard guaranteeing `llama-server` is always terminated and GPU VRAM is freed
/// when analysis finishes, fails, or gets cancelled.
struct LlamaServerGuard {
    child: Child,
    base_url: String,
}

impl Drop for LlamaServerGuard {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

impl LlamaServerGuard {
    fn spawn(
        llama_bin: &Path,
        model_path: &Path,
        mmproj_path: &Path,
        accel: AccelBackend,
        logical_cores: usize,
    ) -> AicutResult<Self> {
        let listener = TcpListener::bind("127.0.0.1:0")
            .map_err(|e| AicutError::internal(format!("Failed finding free port: {e}")))?;
        let port = listener
            .local_addr()
            .map_err(|e| AicutError::internal(format!("Failed reading local port: {e}")))?
            .port();
        drop(listener);

        let gpu_layers = match accel {
            AccelBackend::CpuOnly => "0",
            _ => "99",
        };
        let threads = logical_cores.clamp(2, 12).to_string();

        let mut cmd = Command::new(llama_bin);
        configure_std_command(&mut cmd);
        if let Some(parent) = llama_bin.parent() {
            cmd.current_dir(parent);
        }
        cmd.args([
            "-m",
            &model_path.to_string_lossy(),
            "--mmproj",
            &mmproj_path.to_string_lossy(),
            "--host",
            "127.0.0.1",
            "--port",
            &port.to_string(),
            "-ngl",
            gpu_layers,
            "-c",
            "8192",
            "-t",
            &threads,
            "--parallel",
            "1",
        ])
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null());

        let child = cmd.spawn().map_err(|e| {
            AicutError::internal(format!(
                "Falha ao iniciar llama-server ({}): {e}",
                llama_bin.display()
            ))
        })?;

        Ok(Self {
            child,
            base_url: format!("http://127.0.0.1:{port}"),
        })
    }

    async fn wait_ready(
        &mut self,
        http: &reqwest::Client,
        cancel: &CancellationToken,
    ) -> AicutResult<()> {
        let health_url = format!("{}/health", self.base_url);
        for _ in 0..240 {
            if cancel.is_cancelled() {
                return Err(AicutError::cancelled("Inicializacao do llama-server cancelada."));
            }
            if let Ok(Some(exit)) = self.child.try_wait() {
                return Err(AicutError::new(
                    "LLAMA_SERVER_EXITED",
                    format!(
                        "llama-server encerrou inesperadamente ao carregar o modelo (codigo {:?}). Verifique a VRAM disponivel ou tente uma versao menor (ex.: 3B Q4_K_M).",
                        exit.code()
                    ),
                ));
            }
            if let Ok(resp) = http.get(&health_url).send().await {
                if resp.status().is_success() {
                    return Ok(());
                }
            }
            tokio::time::sleep(std::time::Duration::from_millis(500)).await;
        }
        Err(AicutError::internal(
            "Tempo esgotado aguardando llama-server carregar o modelo visual.",
        ))
    }
}

#[allow(clippy::too_many_arguments)]
pub async fn analyze_with_local_vlm(
    kernel: &AicutKernel,
    job_id: &str,
    ffmpeg_bin: &Path,
    llama_bin: &Path,
    model_path: &Path,
    mmproj_path: &Path,
    video_path: &Path,
    duration_sec: f64,
    transcript: Option<&TranscriptRecord>,
    user_description: &str,
    max_clip_sec: f64,
    min_clip_sec: f64,
    max_clips: usize,
    allow_jump_cuts: bool,
    tmp_dir: &Path,
    cancel: &CancellationToken,
) -> AicutResult<Vec<RawClipPlan>> {
    kernel.jobs.update_progress(
        job_id,
        JobProgress {
            percent: 12.0,
            stage: "starting_local_vlm".into(),
            message: Some("Carregando IA visual local (llama-server + Qwen2.5-VL) na GPU...".into()),
            ..Default::default()
        },
    );

    let mut server = LlamaServerGuard::spawn(
        llama_bin,
        model_path,
        mmproj_path,
        kernel.hardware.recommended_accel,
        kernel.hardware.logical_cores,
    )?;

    let http = kernel.tools.http_client();
    server.wait_ready(http, cancel).await?;

    // Smaller 15-minute windows with 3-minute overlap keep local 8k context sharp & fast
    let windows = plan_windows(duration_sec, 900.0, 180.0);
    if windows.is_empty() {
        return Ok(Vec::new());
    }

    let mut all_candidates: Vec<CandidateWindow> = Vec::new();
    let total_windows = windows.len();

    // ---- PASS A: Storyboard Visual Scan (3x3 Contact Sheets) ----
    for (idx, win) in windows.iter().enumerate() {
        if cancel.is_cancelled() {
            return Err(AicutError::cancelled("Analise visual local cancelada."));
        }
        let pct = 18.0 + (idx as f64 / total_windows as f64) * 38.0;
        kernel.jobs.update_progress(
            job_id,
            JobProgress {
                percent: pct,
                stage: "local_vlm_scan".into(),
                message: Some(format!(
                    "Passagem A ({}/{total_windows}): analisando storyboards visuais localmente...",
                    idx + 1
                )),
                ..Default::default()
            },
        );

        let sheet_dir = tmp_dir.join(format!("sheets_{idx}"));
        let sheets = extract_storyboard_sheets(
            ffmpeg_bin,
            video_path,
            win.start_sec,
            win.duration_sec(),
            &sheet_dir,
            4,
        )
        .await?;

        let image_paths: Vec<PathBuf> = sheets.iter().map(|(_, _, p)| p.clone()).collect();
        let excerpt = format_transcript_window(transcript, win.start_sec, win.end_sec, 10_000);
        let prompt = build_pass_a_prompt(
            user_description,
            win.start_sec,
            win.end_sec,
            max_clip_sec,
            (max_clips + 2).min(6),
            &excerpt,
            true,
        );

        let raw_text = chat_multimodal_json(http, &server.base_url, &image_paths, &prompt).await?;
        if let Some(json_str) = extract_json_object(&raw_text) {
            if let Ok(parsed) = serde_json::from_str::<PassAResponse>(json_str) {
                for mut cand in parsed.candidates {
                    if win.start_sec > 30.0 && cand.end_sec <= win.duration_sec() + 10.0 {
                        cand.start_sec += win.start_sec;
                        cand.end_sec += win.start_sec;
                    }
                    all_candidates.push(cand);
                }
            }
        }
    }

    if all_candidates.is_empty() {
        return Ok(Vec::new());
    }

    all_candidates.sort_by(|a, b| b.score.cmp(&a.score));
    let refine_limit = (max_clips + 2).min(6);
    let mut top_candidates: Vec<CandidateWindow> = Vec::new();
    for cand in all_candidates {
        let overlaps = top_candidates.iter().any(|p| {
            let inter = (p.end_sec.min(cand.end_sec) - p.start_sec.max(cand.start_sec)).max(0.0);
            let union = (p.end_sec.max(cand.end_sec) - p.start_sec.min(cand.start_sec)).max(0.001);
            inter / union > 0.35
        });
        if !overlaps {
            top_candidates.push(cand);
            if top_candidates.len() >= refine_limit {
                break;
            }
        }
    }

    // ---- PASS B: Frame-by-Frame Surgical Refinement ----
    let mut final_raw_clips: Vec<RawClipPlan> = Vec::new();
    let total_cands = top_candidates.len().max(1);

    for (idx, cand) in top_candidates.iter().enumerate() {
        if cancel.is_cancelled() {
            return Err(AicutError::cancelled("Refinamento local cancelado."));
        }
        let pct = 58.0 + (idx as f64 / total_cands as f64) * 32.0;
        kernel.jobs.update_progress(
            job_id,
            JobProgress {
                percent: pct,
                stage: "local_vlm_director".into(),
                message: Some(format!(
                    "Passagem B ({}/{total_cands}): refinando corte coerente inicio-meio-fim...",
                    idx + 1
                )),
                ..Default::default()
            },
        );

        let win = TimeWindow {
            start_sec: (cand.start_sec - 15.0).max(0.0),
            end_sec: (cand.end_sec + 15.0).min(duration_sec),
        };
        if win.duration_sec() < 10.0 {
            continue;
        }

        let frames_dir = tmp_dir.join(format!("frames_{idx}"));
        let frames = extract_candidate_frames(
            ffmpeg_bin,
            video_path,
            win.start_sec,
            win.duration_sec(),
            &frames_dir,
            8,
        )
        .await?;

        let frame_paths: Vec<PathBuf> = frames.iter().map(|(_, p)| p.clone()).collect();
        let frame_times: Vec<String> = frames
            .iter()
            .enumerate()
            .map(|(i, (t, _))| format!("Frame {}: {t:.1}s", i + 1))
            .collect();
        let detailed_tr = format_transcript_window(transcript, win.start_sec, win.end_sec, 9_000);
        let hint = format!(
            "Titulo sugerido: {} | Intervalo aproximado: {:.1}s-{:.1}s | Gancho: {} | Visual: {} | Tempos dos frames enviados: {}",
            cand.title,
            cand.start_sec,
            cand.end_sec,
            cand.hook_reason,
            cand.visual_evidence,
            frame_times.join(", ")
        );
        let prompt = build_pass_b_prompt(
            user_description,
            win.start_sec,
            win.end_sec,
            max_clip_sec,
            min_clip_sec,
            allow_jump_cuts,
            &hint,
            &detailed_tr,
        );

        if let Ok(raw_text) =
            chat_multimodal_json(http, &server.base_url, &frame_paths, &prompt).await
        {
            if let Some(json_str) = extract_json_object(&raw_text) {
                if let Ok(parsed) = serde_json::from_str::<PassBResponse>(json_str) {
                    for mut clip in parsed.clips {
                        for seg in &mut clip.segments {
                            if win.start_sec > 25.0 && seg.end_sec <= win.duration_sec() + 10.0 {
                                seg.start_sec += win.start_sec;
                                seg.end_sec += win.start_sec;
                            }
                        }
                        final_raw_clips.push(clip);
                    }
                }
            }
        }
    }

    // `server` drops here automatically -> kills llama-server and frees 100% of VRAM
    Ok(final_raw_clips)
}

async fn chat_multimodal_json(
    http: &reqwest::Client,
    base_url: &str,
    image_paths: &[PathBuf],
    prompt: &str,
) -> AicutResult<String> {
    let mut parts: Vec<Value> = Vec::with_capacity(image_paths.len() + 1);
    for img_path in image_paths {
        let bytes = tokio::fs::read(img_path).await?;
        let b64 = base64::engine::general_purpose::STANDARD.encode(&bytes);
        parts.push(serde_json::json!({
            "type": "image_url",
            "image_url": {
                "url": format!("data:image/jpeg;base64,{b64}")
            }
        }));
    }
    parts.push(serde_json::json!({
        "type": "text",
        "text": prompt
    }));

    let body = serde_json::json!({
        "messages": [
            {
                "role": "system",
                "content": "Voce e um editor de video multimodal especialista em Shorts coerentes. Responda SEMPRE com JSON valido apenas."
            },
            {
                "role": "user",
                "content": parts
            }
        ],
        "temperature": 0.25,
        "max_tokens": 2048
    });

    let url = format!("{base_url}/v1/chat/completions");
    let resp = http
        .post(&url)
        .json(&body)
        .send()
        .await
        .map_err(|e| AicutError::internal(format!("Local llama-server request failed: {e}")))?;

    if !resp.status().is_success() {
        let status = resp.status();
        let err_body = resp.text().await.unwrap_or_default();
        return Err(AicutError::new(
            "LOCAL_VLM_ERROR",
            format!("llama-server retornou HTTP {status}: {err_body}"),
        ));
    }

    let val: Value = resp
        .json()
        .await
        .map_err(|e| AicutError::internal(format!("Invalid llama-server JSON response: {e}")))?;

    let text = val
        .pointer("/choices/0/message/content")
        .and_then(|v| v.as_str())
        .ok_or_else(|| AicutError::internal("llama-server returned empty response"))?;

    Ok(text.to_string())
}
