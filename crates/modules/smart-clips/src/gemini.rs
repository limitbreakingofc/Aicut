use aicut_core::{AicutError, AicutKernel, AicutResult, JobProgress, TranscriptRecord};
use serde_json::Value;
use std::path::Path;
use tokio_util::sync::CancellationToken;

use crate::frames::{
    build_video_proxy_chunk, format_transcript_window, plan_windows, TimeWindow,
};
use crate::prompts::{
    build_pass_a_prompt, build_pass_b_prompt, extract_json_object, CandidateWindow, PassAResponse,
    PassBResponse, RawClipPlan,
};

#[derive(Debug, Clone)]
pub struct GeminiConfig {
    pub api_key: String,
    pub scan_model: String,
    pub director_model: String,
}

struct UploadedGeminiFile {
    name: String,
    uri: String,
}

#[allow(clippy::too_many_arguments)]
pub async fn analyze_with_gemini(
    kernel: &AicutKernel,
    job_id: &str,
    ffmpeg_bin: &Path,
    video_path: &Path,
    duration_sec: f64,
    transcript: Option<&TranscriptRecord>,
    user_description: &str,
    max_clip_sec: f64,
    min_clip_sec: f64,
    max_clips: usize,
    allow_jump_cuts: bool,
    cfg: &GeminiConfig,
    tmp_dir: &Path,
    cancel: &CancellationToken,
) -> AicutResult<Vec<RawClipPlan>> {
    let http = kernel.tools.http_client();
    // 25-minute windows with 3-minute overlap so 3-minute stories never get split across chunks
    let windows = plan_windows(duration_sec, 1500.0, 180.0);
    if windows.is_empty() {
        return Ok(Vec::new());
    }

    let mut all_candidates: Vec<CandidateWindow> = Vec::new();
    let total_windows = windows.len();

    // ---- PASS A: Native Video+Audio Discovery with Gemini Flash ----
    for (idx, win) in windows.iter().enumerate() {
        if cancel.is_cancelled() {
            return Err(AicutError::cancelled("Analise Gemini cancelada."));
        }
        let pct_base = 15.0 + (idx as f64 / total_windows as f64) * 40.0;
        kernel.jobs.update_progress(
            job_id,
            JobProgress {
                percent: pct_base,
                stage: "gemini_scan".into(),
                message: Some(format!(
                    "Passagem A ({}/{total_windows}): gerando proxy leve e assistindo via {}...",
                    idx + 1,
                    cfg.scan_model
                )),
                ..Default::default()
            },
        );

        let proxy_path = tmp_dir.join(format!("gemini_win_{idx}.mp4"));
        // Use 360p 0.5fps for very long videos (>1h), 480p 1fps for normal videos
        let (height, fps) = if duration_sec > 3600.0 {
            (360, 0.5)
        } else {
            (480, 1.0)
        };
        build_video_proxy_chunk(
            ffmpeg_bin,
            video_path,
            win.start_sec,
            win.duration_sec(),
            height,
            fps,
            &proxy_path,
        )
        .await?;

        let uploaded = upload_video_to_gemini(
            http,
            &cfg.api_key,
            &proxy_path,
            &format!("aicut-win-{idx}.mp4"),
            cancel,
        )
        .await?;

        let excerpt = format_transcript_window(transcript, win.start_sec, win.end_sec, 28_000);
        let prompt = build_pass_a_prompt(
            user_description,
            win.start_sec,
            win.end_sec,
            max_clip_sec,
            (max_clips + 3).min(10),
            &excerpt,
            false,
        );

        let gen_res = generate_with_video_file(
            http,
            &cfg.api_key,
            &cfg.scan_model,
            &uploaded.uri,
            &prompt,
        )
        .await;

        // Always clean up temporary file from Gemini Files API and disk
        let _ = delete_gemini_file(http, &cfg.api_key, &uploaded.name).await;
        let _ = std::fs::remove_file(&proxy_path);

        let raw_text = gen_res?;
        if let Some(json_str) = extract_json_object(&raw_text) {
            if let Ok(parsed) = serde_json::from_str::<PassAResponse>(json_str) {
                for mut cand in parsed.candidates {
                    // Normalize relative timestamps if model returned window-relative times on chunk > 0
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

    // Sort candidates by score descending and keep top (max_clips + 2) for surgical Pass B
    all_candidates.sort_by(|a, b| b.score.cmp(&a.score));
    let refine_limit = (max_clips + 2).min(8);
    let top_candidates: Vec<CandidateWindow> = select_distinct_candidates(all_candidates, refine_limit);

    // ---- PASS B: High-Fidelity Shorts Director with Gemini Pro ----
    let mut final_raw_clips: Vec<RawClipPlan> = Vec::new();
    let total_cands = top_candidates.len().max(1);

    for (idx, cand) in top_candidates.iter().enumerate() {
        if cancel.is_cancelled() {
            return Err(AicutError::cancelled("Refinamento Gemini cancelado."));
        }
        let pct = 58.0 + (idx as f64 / total_cands as f64) * 32.0;
        kernel.jobs.update_progress(
            job_id,
            JobProgress {
                percent: pct,
                stage: "gemini_director".into(),
                message: Some(format!(
                    "Passagem B ({}/{total_cands}): refinando arco inicio-meio-fim com {}...",
                    idx + 1,
                    cfg.director_model
                )),
                ..Default::default()
            },
        );

        let pad_start = (cand.start_sec - 20.0).max(0.0);
        let pad_end = (cand.end_sec + 20.0).min(duration_sec);
        let win = TimeWindow {
            start_sec: pad_start,
            end_sec: pad_end,
        };
        if win.duration_sec() < 10.0 {
            continue;
        }

        let refine_proxy = tmp_dir.join(format!("gemini_refine_{idx}.mp4"));
        build_video_proxy_chunk(
            ffmpeg_bin,
            video_path,
            win.start_sec,
            win.duration_sec(),
            720,
            2.0,
            &refine_proxy,
        )
        .await?;

        let uploaded = upload_video_to_gemini(
            http,
            &cfg.api_key,
            &refine_proxy,
            &format!("aicut-refine-{idx}.mp4"),
            cancel,
        )
        .await?;

        let detailed_tr = format_transcript_window(transcript, win.start_sec, win.end_sec, 18_000);
        let hint = format!(
            "Titulo sugerido: {} | Intervalo aproximado: {:.1}s-{:.1}s | Gancho: {} | Visual: {}",
            cand.title, cand.start_sec, cand.end_sec, cand.hook_reason, cand.visual_evidence
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

        let gen_res = generate_with_video_file(
            http,
            &cfg.api_key,
            &cfg.director_model,
            &uploaded.uri,
            &prompt,
        )
        .await;

        let _ = delete_gemini_file(http, &cfg.api_key, &uploaded.name).await;
        let _ = std::fs::remove_file(&refine_proxy);

        if let Ok(raw_text) = gen_res {
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

    Ok(final_raw_clips)
}

fn select_distinct_candidates(
    sorted: Vec<CandidateWindow>,
    limit: usize,
) -> Vec<CandidateWindow> {
    let mut picked: Vec<CandidateWindow> = Vec::new();
    for cand in sorted {
        let overlaps = picked.iter().any(|p| {
            let inter = (p.end_sec.min(cand.end_sec) - p.start_sec.max(cand.start_sec)).max(0.0);
            let union = (p.end_sec.max(cand.end_sec) - p.start_sec.min(cand.start_sec)).max(0.001);
            inter / union > 0.35
        });
        if !overlaps {
            picked.push(cand);
            if picked.len() >= limit {
                break;
            }
        }
    }
    picked
}

async fn upload_video_to_gemini(
    http: &reqwest::Client,
    api_key: &str,
    video_path: &Path,
    display_name: &str,
    cancel: &CancellationToken,
) -> AicutResult<UploadedGeminiFile> {
    let bytes = tokio::fs::read(video_path).await?;
    let total_len = bytes.len();

    // Step 1: initiate resumable upload
    let start_url = format!(
        "https://generativelanguage.googleapis.com/upload/v1beta/files?key={api_key}"
    );
    let init_resp = http
        .post(&start_url)
        .header("X-Goog-Upload-Protocol", "resumable")
        .header("X-Goog-Upload-Command", "start")
        .header("X-Goog-Upload-Header-Content-Length", total_len.to_string())
        .header("X-Goog-Upload-Header-Content-Type", "video/mp4")
        .json(&serde_json::json!({
            "file": { "display_name": display_name }
        }))
        .send()
        .await
        .map_err(|e| AicutError::internal(format!("Gemini upload init failed: {e}")))?;

    if !init_resp.status().is_success() {
        let status = init_resp.status();
        let body = init_resp.text().await.unwrap_or_default();
        return Err(AicutError::new(
            "GEMINI_UPLOAD_FAILED",
            format!("Gemini Files API retornou HTTP {status}: {body}"),
        ));
    }

    let upload_url = init_resp
        .headers()
        .get("x-goog-upload-url")
        .and_then(|v| v.to_str().ok())
        .ok_or_else(|| AicutError::internal("Missing x-goog-upload-url header from Gemini"))?
        .to_string();

    // Step 2: upload video payload
    let fin_resp = http
        .post(&upload_url)
        .header("Content-Length", total_len.to_string())
        .header("X-Goog-Upload-Offset", "0")
        .header("X-Goog-Upload-Command", "upload, finalize")
        .body(bytes)
        .send()
        .await
        .map_err(|e| AicutError::internal(format!("Gemini upload payload failed: {e}")))?;

    if !fin_resp.status().is_success() {
        let status = fin_resp.status();
        let body = fin_resp.text().await.unwrap_or_default();
        return Err(AicutError::new(
            "GEMINI_UPLOAD_FAILED",
            format!("Gemini Files API upload falhou (HTTP {status}): {body}"),
        ));
    }

    let payload: Value = fin_resp
        .json()
        .await
        .map_err(|e| AicutError::internal(format!("Invalid Gemini file response JSON: {e}")))?;

    let file_obj = payload.get("file").unwrap_or(&payload);
    let name = file_obj
        .get("name")
        .and_then(|v| v.as_str())
        .ok_or_else(|| AicutError::internal("Gemini file response missing 'name'"))?
        .to_string();
    let uri = file_obj
        .get("uri")
        .and_then(|v| v.as_str())
        .ok_or_else(|| AicutError::internal("Gemini file response missing 'uri'"))?
        .to_string();

    // Step 3: poll until state == ACTIVE
    let poll_url = format!("https://generativelanguage.googleapis.com/v1beta/{name}?key={api_key}");
    for _ in 0..90 {
        if cancel.is_cancelled() {
            let _ = delete_gemini_file(http, api_key, &name).await;
            return Err(AicutError::cancelled("Upload Gemini cancelado."));
        }
        let check = http.get(&poll_url).send().await;
        if let Ok(resp) = check {
            if let Ok(j) = resp.json::<Value>().await {
                let state = j.get("state").and_then(|v| v.as_str()).unwrap_or("");
                if state == "ACTIVE" {
                    return Ok(UploadedGeminiFile { name, uri });
                }
                if state == "FAILED" {
                    return Err(AicutError::internal(
                        "Gemini falhou ao processar o video enviado.",
                    ));
                }
            }
        }
        tokio::time::sleep(std::time::Duration::from_secs(2)).await;
    }

    Err(AicutError::internal(
        "Tempo esgotado aguardando Gemini processar o video proxy.",
    ))
}

async fn generate_with_video_file(
    http: &reqwest::Client,
    api_key: &str,
    model: &str,
    file_uri: &str,
    prompt: &str,
) -> AicutResult<String> {
    let url = format!(
        "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
    );
    let body = serde_json::json!({
        "contents": [{
            "role": "user",
            "parts": [
                {
                    "file_data": {
                        "mime_type": "video/mp4",
                        "file_uri": file_uri
                    }
                },
                { "text": prompt }
            ]
        }],
        "generationConfig": {
            "temperature": 0.25,
            "responseMimeType": "application/json"
        }
    });

    let resp = http
        .post(&url)
        .json(&body)
        .send()
        .await
        .map_err(|e| AicutError::internal(format!("Gemini generateContent failed: {e}")))?;

    if !resp.status().is_success() {
        let status = resp.status();
        let text = resp.text().await.unwrap_or_default();
        return Err(AicutError::new(
            "GEMINI_API_ERROR",
            format!("Gemini API ({model}) retornou HTTP {status}: {text}"),
        ));
    }

    let val: Value = resp
        .json()
        .await
        .map_err(|e| AicutError::internal(format!("Invalid Gemini response JSON: {e}")))?;

    let text = val
        .pointer("/candidates/0/content/parts/0/text")
        .and_then(|v| v.as_str())
        .ok_or_else(|| AicutError::internal("Gemini returned empty response text"))?;

    Ok(text.to_string())
}

async fn delete_gemini_file(http: &reqwest::Client, api_key: &str, name: &str) -> AicutResult<()> {
    let url = format!("https://generativelanguage.googleapis.com/v1beta/{name}?key={api_key}");
    let _ = http.delete(&url).send().await;
    Ok(())
}
