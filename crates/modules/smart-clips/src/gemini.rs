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
    // Dynamically resolve active Gemini models available to this API key (auto-upgrading deprecated 2.5 models)
    let (scan_model, director_model) =
        resolve_active_gemini_models(http, &cfg.api_key, &cfg.scan_model, &cfg.director_model)
            .await;
    kernel.jobs.append_log(
        job_id,
        format!("Modelos Gemini ativos resolvidos: varredura={scan_model} | diretor={director_model}"),
    );

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
                    "Passagem A ({}/{total_windows}): gerando proxy leve e assistindo via {scan_model}...",
                    idx + 1,
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
            &scan_model,
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
                    "Passagem B ({}/{total_cands}): refinando arco inicio-meio-fim com {director_model}...",
                    idx + 1,
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

        let gen_res = match generate_with_video_file(
            http,
            &cfg.api_key,
            &director_model,
            &uploaded.uri,
            &prompt,
        )
        .await
        {
            Ok(txt) => Ok(txt),
            Err(_) if director_model != scan_model => {
                // Safe fallback to verified scan_model (e.g. gemini-3.8-flash) if Pro model is unavailable
                generate_with_video_file(
                    http,
                    &cfg.api_key,
                    &scan_model,
                    &uploaded.uri,
                    &prompt,
                )
                .await
            }
            Err(e) => Err(e),
        };

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

/// Sends an inline JPEG image + prompt to an active Gemini model and returns raw JSON text.
pub async fn generate_with_inline_image(
    http: &reqwest::Client,
    api_key: &str,
    model: &str,
    jpeg_b64: &str,
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
                    "inline_data": {
                        "mime_type": "image/jpeg",
                        "data": jpeg_b64
                    }
                },
                { "text": prompt }
            ]
        }],
        "generationConfig": {
            "temperature": 0.25,
            "responseMimeType": "application/json",
            "maxOutputTokens": 6144
        }
    });

    let resp = http
        .post(&url)
        .json(&body)
        .send()
        .await
        .map_err(|e| AicutError::internal(format!("Gemini framing review request failed: {e}")))?;

    if !resp.status().is_success() {
        let status = resp.status();
        let err_txt = resp.text().await.unwrap_or_default();
        return Err(AicutError::new(
            "GEMINI_API_ERROR",
            format!("Gemini API ({model}) retornou HTTP {status}: {err_txt}"),
        ));
    }

    let val: Value = resp
        .json()
        .await
        .map_err(|e| AicutError::internal(format!("Failed parsing Gemini framing review JSON: {e}")))?;
    let text = val
        .get("candidates")
        .and_then(|v| v.as_array())
        .and_then(|arr| arr.first())
        .and_then(|c| c.get("content"))
        .and_then(|c| c.get("parts"))
        .and_then(|p| p.as_array())
        .and_then(|arr| arr.iter().find_map(|part| part.get("text").and_then(|t| t.as_str())))
        .ok_or_else(|| AicutError::internal("Gemini framing review returned empty text"))?;
    Ok(text.to_string())
}

/// Sends a text prompt to an active Gemini model requesting strict JSON output.
pub async fn generate_json_text(
    http: &reqwest::Client,
    api_key: &str,
    model: &str,
    prompt: &str,
) -> AicutResult<String> {
    let url = format!(
        "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
    );
    let body = serde_json::json!({
        "contents": [{
            "role": "user",
            "parts": [{ "text": prompt }]
        }],
        "generationConfig": {
            "temperature": 0.25,
            "responseMimeType": "application/json",
            "maxOutputTokens": 4096
        }
    });

    let resp = http
        .post(&url)
        .json(&body)
        .send()
        .await
        .map_err(|e| AicutError::internal(format!("Gemini JSON request failed: {e}")))?;

    if !resp.status().is_success() {
        let status = resp.status();
        let err_txt = resp.text().await.unwrap_or_default();
        return Err(AicutError::new(
            "GEMINI_API_ERROR",
            format!("Gemini API ({model}) retornou HTTP {status}: {err_txt}"),
        ));
    }

    let val: Value = resp
        .json()
        .await
        .map_err(|e| AicutError::internal(format!("Failed parsing Gemini JSON response: {e}")))?;
    let text = val
        .get("candidates")
        .and_then(|v| v.as_array())
        .and_then(|arr| arr.first())
        .and_then(|c| c.get("content"))
        .and_then(|c| c.get("parts"))
        .and_then(|p| p.as_array())
        .and_then(|arr| arr.iter().find_map(|part| part.get("text").and_then(|t| t.as_str())))
        .ok_or_else(|| AicutError::internal("Gemini returned empty response text"))?;
    Ok(text.to_string())
}

/// Queries the user's Gemini API key for currently available `generateContent` models so deprecated
/// model IDs (like `gemini-2.5-flash` / `gemini-2.5-pro`) automatically upgrade to the newest
/// compatible Gemini Flash & Pro models (e.g. `gemini-3.8-flash` / `gemini-3.8-pro`).
pub async fn resolve_active_gemini_models(
    http: &reqwest::Client,
    api_key: &str,
    requested_scan: &str,
    requested_director: &str,
) -> (String, String) {
    let available = fetch_available_gemini_models(http, api_key).await;

    let normalize = |req: &str| -> String {
        let trimmed = req.trim().trim_start_matches("models/");
        if trimmed.is_empty()
            || trimmed.eq_ignore_ascii_case("auto")
            || trimmed.starts_with("gemini-2.")
            || trimmed.starts_with("gemini-1.")
        {
            "auto".to_string()
        } else {
            trimmed.to_string()
        }
    };

    let req_scan = normalize(requested_scan);
    let req_dir = normalize(requested_director);

    if available.is_empty() {
        let fallback_scan = if req_scan == "auto" {
            "gemini-3.8-flash".to_string()
        } else {
            req_scan
        };
        let fallback_dir = if req_dir == "auto" {
            "gemini-3.8-pro".to_string()
        } else {
            req_dir
        };
        return (fallback_scan, fallback_dir);
    }

    let scan = if req_scan != "auto" && available.iter().any(|m| m == &req_scan) {
        req_scan
    } else {
        pick_best_gemini_model(&available, true)
            .unwrap_or_else(|| "gemini-3.8-flash".to_string())
    };

    let want_flash_dir = req_dir.contains("flash");
    let director = if req_dir != "auto" && available.iter().any(|m| m == &req_dir) {
        req_dir
    } else if want_flash_dir {
        scan.clone()
    } else {
        pick_best_gemini_model(&available, false).unwrap_or_else(|| scan.clone())
    };

    (scan, director)
}

async fn fetch_available_gemini_models(http: &reqwest::Client, api_key: &str) -> Vec<String> {
    let url = format!(
        "https://generativelanguage.googleapis.com/v1beta/models?pageSize=100&key={api_key}"
    );
    let Ok(resp) = http.get(&url).send().await else {
        return Vec::new();
    };
    if !resp.status().is_success() {
        return Vec::new();
    }
    let Ok(val) = resp.json::<Value>().await else {
        return Vec::new();
    };
    let Some(models) = val.get("models").and_then(|v| v.as_array()) else {
        return Vec::new();
    };

    models
        .iter()
        .filter_map(|m| {
            let methods = m.get("supportedGenerationMethods")?.as_array()?;
            let can_gen = methods
                .iter()
                .any(|x| x.as_str() == Some("generateContent"));
            if !can_gen {
                return None;
            }
            let raw_name = m.get("name")?.as_str()?;
            let id = raw_name.trim_start_matches("models/");
            let lower = id.to_lowercase();
            if !lower.starts_with("gemini-") {
                return None;
            }
            if lower.contains("embedding")
                || lower.contains("imagen")
                || lower.contains("veo")
                || lower.contains("tts")
                || lower.contains("audio")
                || lower.contains("image")
                || lower.contains("robotics")
                || lower.contains("computer-use")
                || lower.contains("aqa")
                || lower.contains("lite")
            {
                return None;
            }
            Some(id.to_string())
        })
        .collect()
}

/// Ranks available Gemini model IDs and selects the highest-version Flash (or Pro) model.
pub fn pick_best_gemini_model(models: &[String], prefer_flash: bool) -> Option<String> {
    let score = |id: &str| -> (u8, u32, u32, u8) {
        let lower = id.to_lowercase();
        let tier_match = if prefer_flash {
            if lower.contains("flash") { 2 } else { 1 }
        } else if lower.contains("pro") {
            2
        } else if lower.contains("flash") {
            1
        } else {
            0
        };
        let (maj, min) = parse_gemini_version(&lower);
        let stability = if lower.contains("exp") {
            0
        } else if lower.contains("preview") || lower.contains("latest") {
            1
        } else {
            2
        };
        (tier_match, maj, min, stability)
    };

    models
        .iter()
        .max_by_key(|m| score(m))
        .cloned()
}

fn parse_gemini_version(id: &str) -> (u32, u32) {
    let rest = id.trim_start_matches("gemini-");
    let token = rest.split('-').next().unwrap_or("");
    let mut parts = token.split('.');
    let major = parts.next().and_then(|s| s.parse::<u32>().ok()).unwrap_or(0);
    let minor = parts.next().and_then(|s| s.parse::<u32>().ok()).unwrap_or(0);
    (major, minor)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn picks_newest_stable_gemini_flash_and_pro() {
        let list = vec![
            "gemini-2.5-flash".to_string(),
            "gemini-3.0-pro-preview".to_string(),
            "gemini-3.8-flash".to_string(),
            "gemini-3.8-pro".to_string(),
        ];
        assert_eq!(
            pick_best_gemini_model(&list, true).as_deref(),
            Some("gemini-3.8-flash")
        );
        assert_eq!(
            pick_best_gemini_model(&list, false).as_deref(),
            Some("gemini-3.8-pro")
        );
    }

    #[test]
    fn falls_back_to_flash_when_pro_unavailable() {
        let list = vec!["gemini-3.8-flash".to_string(), "gemini-3.5-flash".to_string()];
        assert_eq!(
            pick_best_gemini_model(&list, false).as_deref(),
            Some("gemini-3.8-flash")
        );
    }
}
