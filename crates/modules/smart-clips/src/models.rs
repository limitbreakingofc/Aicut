use aicut_core::{
    AccelBackend, ActorKind, AicutError, AicutKernel, AicutResult, HardwareProfile, JobProgress,
};
use futures_util::StreamExt;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::Arc;

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct VisionModelSpec {
    pub id: String,
    pub family_id: String,
    pub family_name: String,
    pub version_tag: String,
    pub display_name: String,
    pub hf_repo: String,
    pub model_filename: String,
    pub mmproj_filename: String,
    pub size_mb: u64,
    pub min_vram_mb: u64,
    pub min_ram_mb: u64,
    pub quality_score: f32,
    pub quality_tier: String,
    pub speed_tier: String,
    pub quality_notes: String,
    pub description: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct VisionModelStatus {
    pub id: String,
    pub family_id: String,
    pub family_name: String,
    pub version_tag: String,
    pub display_name: String,
    pub hf_repo: String,
    pub model_filename: String,
    pub mmproj_filename: String,
    pub size_mb: u64,
    pub min_vram_mb: u64,
    pub min_ram_mb: u64,
    pub quality_score: f32,
    pub quality_tier: String,
    pub speed_tier: String,
    pub quality_notes: String,
    pub description: String,
    pub installed: bool,
    pub compatible: bool,
    pub compatibility_reason: String,
    pub curated_for_hardware: bool,
    pub recommended: bool,
    pub best_quality_for_hardware: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model_path: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub mmproj_path: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub actual_size_bytes: Option<u64>,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct VisionModelUninstallResult {
    pub model_id: String,
    pub deleted: bool,
    pub freed_bytes: u64,
    pub removed_files: Vec<String>,
}

/// Curated official `ggml-org` multimodal Vision-Language GGUF models for local video analysis.
pub fn catalog() -> Vec<VisionModelSpec> {
    vec![
        // ---- Family 1: Qwen2.5-VL 7B Instruct (Best Local Visual + Narrative Intelligence) ----
        VisionModelSpec {
            id: "qwen2.5-vl-7b-q8_0".into(),
            family_id: "qwen2.5-vl-7b".into(),
            family_name: "Qwen2.5-VL 7B Instruct".into(),
            version_tag: "Q8_0 (8-bit Máxima Fidelidade)".into(),
            display_name: "Qwen2.5-VL 7B · Q8_0".into(),
            hf_repo: "ggml-org/Qwen2.5-VL-7B-Instruct-GGUF".into(),
            model_filename: "Qwen2.5-VL-7B-Instruct-Q8_0.gguf".into(),
            mmproj_filename: "mmproj-Qwen2.5-VL-7B-Instruct-Q8_0.gguf".into(),
            size_mb: 8800,
            min_vram_mb: 11500,
            min_ram_mb: 16384,
            quality_score: 9.8,
            quality_tier: "9.8/10 · Máxima Local".into(),
            speed_tier: "GPU 12 GB+ VRAM".into(),
            quality_notes: "Maior fidelidade visual e narrativa local (8-bit quase sem perdas). Ideal para GPUs de 12–24 GB VRAM (ex.: RTX 3060 12 GB / 4070+ / 4090).".into(),
            description: "Entende cenas complexas, expressões faciais, gameplay, textos na tela e estrutura início–meio–fim 100% offline.".into(),
        },
        VisionModelSpec {
            id: "qwen2.5-vl-7b-q4_k_m".into(),
            family_id: "qwen2.5-vl-7b".into(),
            family_name: "Qwen2.5-VL 7B Instruct".into(),
            version_tag: "Q4_K_M (4-bit Equilibrado)".into(),
            display_name: "Qwen2.5-VL 7B · Q4_K_M".into(),
            hf_repo: "ggml-org/Qwen2.5-VL-7B-Instruct-GGUF".into(),
            model_filename: "Qwen2.5-VL-7B-Instruct-Q4_K_M.gguf".into(),
            mmproj_filename: "mmproj-Qwen2.5-VL-7B-Instruct-Q8_0.gguf".into(),
            size_mb: 5380,
            min_vram_mb: 7200,
            min_ram_mb: 12288,
            quality_score: 9.4,
            quality_tier: "9.4/10 · Sweet Spot GPU".into(),
            speed_tier: "Rápido (GPU 8–12 GB)".into(),
            quality_notes: "Melhor equilíbrio entre inteligência visual/narrativa e folga de VRAM para contextos com múltiplos frames (perfeito para RTX 3060 12 GB e 4060 8 GB).".into(),
            description: "Modelo visual 7B completo em 4-bit com projetor Q8_0: excelente precisão para achar momentos fortes em lives e vídeos.".into(),
        },
        // ---- Family 2: Qwen2.5-VL 3B Instruct (Compact / Mid GPU / CPU Fallback) ----
        VisionModelSpec {
            id: "qwen2.5-vl-3b-q8_0".into(),
            family_id: "qwen2.5-vl-3b".into(),
            family_name: "Qwen2.5-VL 3B Instruct".into(),
            version_tag: "Q8_0 (8-bit Compacto)".into(),
            display_name: "Qwen2.5-VL 3B · Q8_0".into(),
            hf_repo: "ggml-org/Qwen2.5-VL-3B-Instruct-GGUF".into(),
            model_filename: "Qwen2.5-VL-3B-Instruct-Q8_0.gguf".into(),
            mmproj_filename: "mmproj-Qwen2.5-VL-3B-Instruct-Q8_0.gguf".into(),
            size_mb: 3900,
            min_vram_mb: 5000,
            min_ram_mb: 8192,
            quality_score: 8.6,
            quality_tier: "8.6/10 · Muito Boa".into(),
            speed_tier: "Muito Rápido (GPU 6 GB+)".into(),
            quality_notes: "Ótima visão computacional para GPUs de 6 GB VRAM; mais rápido que o 7B com boa coerência narrativa em PT-BR.".into(),
            description: "Versão 3B em 8-bit: alta nitidez visual para placas intermediárias.".into(),
        },
        VisionModelSpec {
            id: "qwen2.5-vl-3b-q4_k_m".into(),
            family_id: "qwen2.5-vl-3b".into(),
            family_name: "Qwen2.5-VL 3B Instruct".into(),
            version_tag: "Q4_K_M (4-bit Ultraleve)".into(),
            display_name: "Qwen2.5-VL 3B · Q4_K_M".into(),
            hf_repo: "ggml-org/Qwen2.5-VL-3B-Instruct-GGUF".into(),
            model_filename: "Qwen2.5-VL-3B-Instruct-Q4_K_M.gguf".into(),
            mmproj_filename: "mmproj-Qwen2.5-VL-3B-Instruct-Q8_0.gguf".into(),
            size_mb: 2500,
            min_vram_mb: 3200,
            min_ram_mb: 8192,
            quality_score: 8.1,
            quality_tier: "8.1/10 · Leve / Universal".into(),
            speed_tier: "Rápido (GPU 4 GB / CPU)".into(),
            quality_notes: "Opção mais leve para notebooks, GPUs de 4 GB ou modo CPU. Para máxima inteligência narrativa em hardware modesto, prefira o modo Gemini API.".into(),
            description: "Roda offline em praticamente qualquer PC com 8 GB+ de RAM.".into(),
        },
    ]
}

pub fn vision_models_dir(kernel: &AicutKernel) -> PathBuf {
    kernel.workspace.models_dir.join("vision")
}

pub fn model_dir_for(kernel: &AicutKernel, model_id: &str) -> PathBuf {
    vision_models_dir(kernel).join(model_id)
}

pub fn find_spec(model_id: &str) -> Option<VisionModelSpec> {
    catalog().into_iter().find(|m| m.id == model_id)
}

pub fn resolve_installed_vision_model(
    kernel: &AicutKernel,
    model_id: &str,
) -> AicutResult<(PathBuf, PathBuf)> {
    let spec = find_spec(model_id).ok_or_else(|| {
        AicutError::not_found(format!("Unknown local vision model '{model_id}'"))
    })?;
    let dir = model_dir_for(kernel, &spec.id);
    let model_path = dir.join(&spec.model_filename);
    // Accept either exact expected mmproj filename or any mmproj*.gguf saved in that model dir
    let mmproj_path = find_mmproj_in_dir(&dir, &spec.mmproj_filename);

    if model_path.is_file() {
        if let Some(proj) = mmproj_path {
            return Ok((model_path, proj));
        }
    }

    Err(AicutError::dependency_missing(
        &format!("vision_model:{model_id}"),
        format!(
            "Modelo visual '{}' ainda não foi baixado. Baixe-o em Configurações → IA & Chaves (modo Local grátis).",
            spec.display_name
        ),
    ))
}

/// Resolves the preferred local vision model if installed, otherwise automatically falls back
/// to ANY installed Qwen2.5-VL model on disk so users never hit a false "not installed" error
/// when they downloaded a different quantization/size than the default recommendation.
pub fn resolve_any_installed_vision_model(
    kernel: &AicutKernel,
    preferred_model_id: Option<&str>,
) -> AicutResult<(String, PathBuf, PathBuf)> {
    if let Some(pref) = preferred_model_id.map(str::trim).filter(|s| !s.is_empty()) {
        if let Ok((mp, proj)) = resolve_installed_vision_model(kernel, pref) {
            return Ok((pref.to_string(), mp, proj));
        }
    }
    let rec = recommended_vision_model_id(&kernel.hardware);
    if let Ok((mp, proj)) = resolve_installed_vision_model(kernel, rec) {
        return Ok((rec.to_string(), mp, proj));
    }
    for spec in catalog() {
        if let Ok((mp, proj)) = resolve_installed_vision_model(kernel, &spec.id) {
            return Ok((spec.id, mp, proj));
        }
    }
    Err(AicutError::dependency_missing(
        "vision_model",
        "Nenhum modelo visual local (Qwen2.5-VL) está baixado. Baixe um modelo em Configurações → IA & Chaves (modo Local grátis).",
    ))
}

fn find_mmproj_in_dir(dir: &Path, preferred: &str) -> Option<PathBuf> {
    let exact = dir.join(preferred);
    if exact.is_file() {
        return Some(exact);
    }
    let entries = std::fs::read_dir(dir).ok()?;
    for entry in entries.flatten() {
        let p = entry.path();
        if p.is_file() {
            if let Some(name) = p.file_name().and_then(|n| n.to_str()) {
                let lower = name.to_lowercase();
                if lower.starts_with("mmproj") && lower.ends_with(".gguf") {
                    return Some(p);
                }
            }
        }
    }
    None
}

fn best_gpu_vram_mb(hw: &HardwareProfile) -> Option<u64> {
    hw.gpus.iter().filter_map(|g| g.vram_mb).max()
}

pub fn evaluate_compatibility(spec: &VisionModelSpec, hw: &HardwareProfile) -> (bool, String) {
    let has_dedicated_gpu = !matches!(hw.recommended_accel, AccelBackend::CpuOnly);
    let gpu_vram = best_gpu_vram_mb(hw);

    if has_dedicated_gpu {
        if let Some(vram) = gpu_vram {
            if vram >= spec.min_vram_mb {
                return (
                    true,
                    format!(
                        "Compatível com sua GPU ({vram} MB VRAM >= {} MB necessários).",
                        spec.min_vram_mb
                    ),
                );
            }
            if hw.total_memory_mb >= spec.min_ram_mb + 4096 {
                return (
                    true,
                    format!(
                        "VRAM limitada ({vram} MB < {} MB): rodará em modo híbrido GPU+RAM (mais lento).",
                        spec.min_vram_mb
                    ),
                );
            }
            return (
                false,
                format!(
                    "Exige {} MB de VRAM (sua GPU tem {vram} MB).",
                    spec.min_vram_mb
                ),
            );
        }
        if hw.total_memory_mb >= spec.min_ram_mb {
            return (
                true,
                format!("Compatível com memória do sistema ({} MB RAM).", hw.total_memory_mb),
            );
        }
    }

    if hw.total_memory_mb >= spec.min_ram_mb {
        (
            true,
            format!(
                "Compatível em modo CPU ({} MB RAM >= {} MB). Para maior velocidade, use GPU ou Gemini API.",
                hw.total_memory_mb, spec.min_ram_mb
            ),
        )
    } else {
        (
            false,
            format!(
                "Requer pelo menos {} MB de RAM (sistema possui {} MB).",
                spec.min_ram_mb, hw.total_memory_mb
            ),
        )
    }
}

/// Automatically picks the highest-quality compatible local vision model for the host GPU/RAM.
/// On an RTX 3060 12 GB (~12288 MB VRAM), prefers `qwen2.5-vl-7b-q4_k_m` (sweet spot leaving generous
/// VRAM headroom for multi-frame context KV cache) while also allowing `qwen2.5-vl-7b-q8_0` as max quality.
pub fn recommended_vision_model_id(hw: &HardwareProfile) -> &'static str {
    let gpu_vram = best_gpu_vram_mb(hw).unwrap_or(0);
    let has_gpu = !matches!(hw.recommended_accel, AccelBackend::CpuOnly);

    if has_gpu {
        if gpu_vram >= 15000 {
            return "qwen2.5-vl-7b-q8_0";
        }
        if gpu_vram >= 7500 {
            return "qwen2.5-vl-7b-q4_k_m";
        }
        if gpu_vram >= 5200 {
            return "qwen2.5-vl-3b-q8_0";
        }
    }
    "qwen2.5-vl-3b-q4_k_m"
}

pub fn list_vision_models(kernel: &AicutKernel) -> Vec<VisionModelStatus> {
    let hw = &kernel.hardware;
    let rec_id = recommended_vision_model_id(hw);
    let specs = catalog();

    let mut evaluated: Vec<(VisionModelSpec, bool, String, bool, Option<PathBuf>, Option<PathBuf>, Option<u64>)> = specs
        .into_iter()
        .map(|spec| {
            let (compatible, reason) = evaluate_compatibility(&spec, hw);
            let dir = model_dir_for(kernel, &spec.id);
            let mp = dir.join(&spec.model_filename);
            let mm = find_mmproj_in_dir(&dir, &spec.mmproj_filename);
            let installed = mp.is_file() && mm.is_some();
            let actual_bytes = if installed {
                let s1 = std::fs::metadata(&mp).map(|m| m.len()).unwrap_or(0);
                let s2 = mm
                    .as_ref()
                    .and_then(|p| std::fs::metadata(p).ok())
                    .map(|m| m.len())
                    .unwrap_or(0);
                Some(s1 + s2)
            } else {
                None
            };
            (
                spec,
                compatible,
                reason,
                installed,
                if installed { Some(mp) } else { None },
                if installed { mm } else { None },
                actual_bytes,
            )
        })
        .collect();

    let best_quality_id = evaluated
        .iter()
        .filter(|(_, compat, _, _, _, _, _)| *compat)
        .max_by(|a, b| {
            a.0.quality_score
                .partial_cmp(&b.0.quality_score)
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .map(|(s, ..)| s.id.clone());

    evaluated
        .drain(..)
        .map(|(spec, compatible, compatibility_reason, installed, mp, mm, actual_size_bytes)| {
            let recommended = spec.id == rec_id;
            let best_quality_for_hardware = best_quality_id.as_deref() == Some(spec.id.as_str());
            let curated_for_hardware = compatible || installed;
            VisionModelStatus {
                id: spec.id,
                family_id: spec.family_id,
                family_name: spec.family_name,
                version_tag: spec.version_tag,
                display_name: spec.display_name,
                hf_repo: spec.hf_repo,
                model_filename: spec.model_filename,
                mmproj_filename: spec.mmproj_filename,
                size_mb: spec.size_mb,
                min_vram_mb: spec.min_vram_mb,
                min_ram_mb: spec.min_ram_mb,
                quality_score: spec.quality_score,
                quality_tier: spec.quality_tier,
                speed_tier: spec.speed_tier,
                quality_notes: spec.quality_notes,
                description: spec.description,
                installed,
                compatible,
                compatibility_reason,
                curated_for_hardware,
                recommended,
                best_quality_for_hardware,
                model_path: mp.map(|p| p.to_string_lossy().to_string()),
                mmproj_path: mm.map(|p| p.to_string_lossy().to_string()),
                actual_size_bytes,
            }
        })
        .collect()
}

/// Resolves exact download URLs for backbone GGUF and multimodal projector (`mmproj`) on Hugging Face,
/// querying the HF tree API first so minor filename changes (e.g. `Q8_0` vs `f16` mmproj) never 404.
async fn resolve_hf_pair_urls(
    http: &reqwest::Client,
    spec: &VisionModelSpec,
) -> (String, String, String) {
    let default_model_url = format!(
        "https://huggingface.co/{}/resolve/main/{}",
        spec.hf_repo, spec.model_filename
    );
    let default_mmproj_url = format!(
        "https://huggingface.co/{}/resolve/main/{}",
        spec.hf_repo, spec.mmproj_filename
    );

    let api_url = format!("https://huggingface.co/api/models/{}/tree/main", spec.hf_repo);
    let Ok(resp) = http.get(&api_url).send().await else {
        return (default_model_url, default_mmproj_url, spec.mmproj_filename.clone());
    };
    if !resp.status().is_success() {
        return (default_model_url, default_mmproj_url, spec.mmproj_filename.clone());
    }
    let Ok(entries) = resp.json::<Vec<serde_json::Value>>().await else {
        return (default_model_url, default_mmproj_url, spec.mmproj_filename.clone());
    };

    let files: Vec<String> = entries
        .iter()
        .filter_map(|e| e.get("path").and_then(|v| v.as_str()).map(|s| s.to_string()))
        .collect();

    let model_file = files
        .iter()
        .find(|f| f.eq_ignore_ascii_case(&spec.model_filename))
        .cloned()
        .unwrap_or_else(|| spec.model_filename.clone());

    let mmproj_file = files
        .iter()
        .find(|f| f.eq_ignore_ascii_case(&spec.mmproj_filename))
        .or_else(|| {
            files.iter().find(|f| {
                let l = f.to_lowercase();
                l.starts_with("mmproj") && l.contains("q8") && l.ends_with(".gguf")
            })
        })
        .or_else(|| {
            files.iter().find(|f| {
                let l = f.to_lowercase();
                l.starts_with("mmproj") && l.ends_with(".gguf")
            })
        })
        .cloned()
        .unwrap_or_else(|| spec.mmproj_filename.clone());

    let model_url = format!("https://huggingface.co/{}/resolve/main/{model_file}", spec.hf_repo);
    let mmproj_url = format!("https://huggingface.co/{}/resolve/main/{mmproj_file}", spec.hf_repo);
    (model_url, mmproj_url, mmproj_file)
}

pub async fn download_vision_model(
    kernel: Arc<AicutKernel>,
    model_id: &str,
    actor: ActorKind,
) -> AicutResult<VisionModelStatus> {
    let spec = find_spec(model_id).ok_or_else(|| {
        AicutError::not_found(format!("Unknown local vision model '{model_id}'"))
    })?;

    let dir = model_dir_for(&kernel, &spec.id);
    let _directory = aicut_core::workspace::lock_downloads(&kernel.workspace.models_dir)?;
    std::fs::create_dir_all(vision_models_dir(&kernel))?;
    let stage = aicut_core::workspace::download_stage(&kernel.workspace.models_dir)?;
    let payload = stage.path().join("model");
    std::fs::create_dir(&payload)?;

    let (job, cancel) = kernel.jobs.create_job(
        "clips.model_download",
        format!("Baixar IA Visual {}", spec.display_name),
        "smart-clips",
        actor,
    );
    kernel.jobs.mark_running(&job.id, "resolving");

    let outcome: AicutResult<VisionModelStatus> = async {
        let http = kernel.tools.http_client();
        let (model_url, mmproj_url, resolved_mmproj_name) = resolve_hf_pair_urls(http, &spec).await;

        let model_dst = payload.join(&spec.model_filename);
        let mmproj_dst = payload.join(&resolved_mmproj_name);

        if !model_dst.is_file() {
            kernel.jobs.append_log(
                &job.id,
                format!("Baixando pesos principais ({}) de {model_url}", spec.model_filename),
            );
            download_with_progress(
                &kernel,
                &job.id,
                &model_url,
                &model_dst,
                0.0,
                82.0,
                "downloading_model",
                &cancel,
            )
            .await?;
        }

        if !mmproj_dst.is_file() {
            kernel.jobs.append_log(
                &job.id,
                format!("Baixando projetor visual ({resolved_mmproj_name}) de {mmproj_url}"),
            );
            download_with_progress(
                &kernel,
                &job.id,
                &mmproj_url,
                &mmproj_dst,
                82.0,
                98.0,
                "downloading_mmproj",
                &cancel,
            )
            .await?;
        }

        aicut_core::workspace::publish_directory(&payload, &dir)?;
        list_vision_models(&kernel)
            .into_iter()
            .find(|m| m.id == spec.id)
            .ok_or_else(|| AicutError::internal("Vision model missing from catalog after download"))
    }
    .await;

    match outcome {
        Ok(status) => {
            kernel.jobs.complete(
                &job.id,
                serde_json::to_value(&status).unwrap_or(serde_json::Value::Null),
            );
            Ok(status)
        }
        Err(err) => {
            if cancel.is_cancelled() {
                kernel.jobs.mark_cancelled(&job.id);
            } else {
                kernel.jobs.fail(&job.id, err.clone());
            }
            Err(err)
        }
    }
}

#[allow(clippy::too_many_arguments)]
async fn download_with_progress(
    kernel: &AicutKernel,
    job_id: &str,
    url: &str,
    dest: &Path,
    pct_start: f64,
    pct_end: f64,
    stage: &str,
    cancel: &tokio_util::sync::CancellationToken,
) -> AicutResult<()> {
    let resp = kernel
        .tools
        .http_client()
        .get(url)
        .send()
        .await
        .map_err(|e| AicutError::internal(format!("Vision model download failed: {e}")))?;

    if !resp.status().is_success() {
        return Err(AicutError::internal(format!(
            "Download falhou com HTTP {} para {url}",
            resp.status()
        )));
    }

    let total = resp.content_length();
    let part_path = dest.with_extension("gguf.part");
    let mut file = std::fs::File::create(&part_path)?;
    let mut stream = resp.bytes_stream();
    let mut downloaded: u64 = 0;
    let started = std::time::Instant::now();
    let mut last_emit = std::time::Instant::now();

    while let Some(chunk_res) = tokio::select! {
        _ = cancel.cancelled() => {
            let _ = std::fs::remove_file(&part_path);
            return Err(AicutError::cancelled("Download do modelo visual cancelado."));
        }
        next = stream.next() => next,
    } {
        let chunk = chunk_res
            .map_err(|e| AicutError::internal(format!("Erro de stream no download: {e}")))?;
        file.write_all(&chunk)?;
        downloaded += chunk.len() as u64;

        if last_emit.elapsed().as_millis() >= 250 {
            let elapsed = started.elapsed().as_secs_f64().max(0.001);
            let speed = downloaded as f64 / elapsed;
            let frac = total
                .map(|t| (downloaded as f64 / t.max(1) as f64).clamp(0.0, 1.0))
                .unwrap_or(0.5);
            let percent = pct_start + frac * (pct_end - pct_start);
            let eta = total.map(|t| t.saturating_sub(downloaded) as f64 / speed.max(1.0));
            kernel.jobs.update_progress(
                job_id,
                JobProgress {
                    percent,
                    stage: stage.to_string(),
                    downloaded_bytes: Some(downloaded),
                    total_bytes: total,
                    speed_bytes_per_sec: Some(speed),
                    eta_seconds: eta,
                    message: None,
                },
            );
            last_emit = std::time::Instant::now();
        }
    }

    file.flush()?;
    drop(file);
    if dest.exists() {
        let _ = std::fs::remove_file(dest);
    }
    std::fs::rename(&part_path, dest)?;
    Ok(())
}

pub fn delete_vision_model(
    kernel: &AicutKernel,
    model_id: &str,
) -> AicutResult<VisionModelUninstallResult> {
    let _directory = aicut_core::workspace::lock_downloads(&kernel.workspace.models_dir)?;
    let _spec = find_spec(model_id).ok_or_else(|| {
        AicutError::not_found(format!("Unknown local vision model '{model_id}'"))
    })?;
    let dir = model_dir_for(kernel, model_id);
    kernel.repo.clear_selected_model(crate::engine::SETTING_LOCAL_VISION_MODEL, model_id)?;
    if !dir.exists() {
        aicut_core::workspace::remove_empty_category(&vision_models_dir(kernel))?;
        return Ok(VisionModelUninstallResult {
            model_id: model_id.to_string(),
            deleted: false,
            freed_bytes: 0,
            removed_files: Vec::new(),
        });
    }

    let mut freed_bytes: u64 = 0;
    let mut removed_files = Vec::new();
    if let Ok(entries) = std::fs::read_dir(&dir) {
        for entry in entries.flatten() {
            let p = entry.path();
            if p.is_file() {
                if let Ok(meta) = std::fs::metadata(&p) {
                    freed_bytes += meta.len();
                }
                if let Some(name) = p.file_name().and_then(|n| n.to_str()) {
                    removed_files.push(name.to_string());
                }
            }
        }
    }
    std::fs::remove_dir_all(&dir)?;
    aicut_core::workspace::remove_empty_category(&vision_models_dir(kernel))?;

    Ok(VisionModelUninstallResult {
        model_id: model_id.to_string(),
        deleted: !removed_files.is_empty(),
        freed_bytes,
        removed_files,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use aicut_core::{GpuInfo, GpuVendor};

    fn fake_hw(accel: AccelBackend, vram_mb: Option<u64>, ram_mb: u64) -> HardwareProfile {
        HardwareProfile {
            os: "linux".into(),
            os_version: "Fedora Linux".into(),
            arch: "x86_64".into(),
            platform_tag: "linux-x64".into(),
            cpu_brand: "Test CPU".into(),
            physical_cores: 8,
            logical_cores: 16,
            total_memory_mb: ram_mb,
            available_memory_mb: ram_mb / 2,
            gpus: vram_mb
                .map(|v| {
                    vec![GpuInfo {
                        name: "NVIDIA GeForce RTX 3060".into(),
                        vendor: GpuVendor::Nvidia,
                        vram_mb: Some(v),
                        driver_version: Some("580.0".into()),
                    }]
                })
                .unwrap_or_default(),
            cuda_version: Some("13.5".into()),
            cuda_runtime_installed: true,
            vulkan_available: true,
            recommended_accel: accel,
            recommendation_summary: "test".into(),
        }
    }

    #[test]
    fn recommends_best_vision_model_for_rtx_3060_12gb() {
        let hw = fake_hw(AccelBackend::CudaNvenc, Some(12288), 32768);
        assert_eq!(recommended_vision_model_id(&hw), "qwen2.5-vl-7b-q4_k_m");
    }

    #[test]
    fn recommends_q8_for_16gb_gpu_and_3b_for_cpu() {
        let hw16 = fake_hw(AccelBackend::CudaNvenc, Some(16384), 32768);
        assert_eq!(recommended_vision_model_id(&hw16), "qwen2.5-vl-7b-q8_0");

        let cpu = fake_hw(AccelBackend::CpuOnly, None, 16384);
        assert_eq!(recommended_vision_model_id(&cpu), "qwen2.5-vl-3b-q4_k_m");
    }
}
