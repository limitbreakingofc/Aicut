use aicut_core::{
    AccelBackend, ActorKind, AicutError, AicutKernel, AicutResult, HardwareProfile, JobProgress,
};
use futures_util::StreamExt;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use std::io::Write;
use std::path::PathBuf;
use std::sync::Arc;

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct WhisperModelSpec {
    pub id: String,
    /// Model family ID (e.g. "large-v3-turbo", "large-v3", "large-v2", "medium", "small")
    pub family_id: String,
    pub family_name: String,
    /// Specific precision/quantization version (e.g. "FP16", "Q8_0", "Q5_0")
    pub version_tag: String,
    pub display_name: String,
    pub filename: String,
    pub size_mb: u64,
    pub min_vram_mb: u64,
    pub min_ram_mb: u64,
    /// Relative final transcription quality score out of 10.0
    pub quality_score: f32,
    pub quality_tier: String,
    pub speed_tier: String,
    /// Detailed Portuguese description explaining the final quality vs other versions
    pub quality_notes: String,
    pub description: String,
    pub download_url: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct WhisperModelStatus {
    pub id: String,
    pub family_id: String,
    pub family_name: String,
    pub version_tag: String,
    pub display_name: String,
    pub filename: String,
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
    /// True when this model is in the curated "best models for your GPU/hardware" set
    pub curated_for_hardware: bool,
    pub recommended: bool,
    /// True when this is the highest final-quality model compatible with the user's hardware
    pub best_quality_for_hardware: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub file_path: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub actual_size_bytes: Option<u64>,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct ModelUninstallResult {
    pub model_id: String,
    pub deleted: bool,
    pub freed_bytes: u64,
    pub removed_files: Vec<String>,
}

const HF_BASE: &str = "https://huggingface.co/ggerganov/whisper.cpp/resolve/main";

/// Verified official GGML models from `ggerganov/whisper.cpp` grouped by family & precision version.
pub fn catalog() -> Vec<WhisperModelSpec> {
    vec![
        // ---- Family 1: Whisper Large v3 Turbo (Best Speed + Top Quality) ----
        WhisperModelSpec {
            id: "large-v3-turbo".into(),
            family_id: "large-v3-turbo".into(),
            family_name: "Whisper Large v3 Turbo".into(),
            version_tag: "FP16 (Precisão Total)".into(),
            display_name: "Whisper Large v3 Turbo · FP16".into(),
            filename: "ggml-large-v3-turbo.bin".into(),
            size_mb: 1624,
            min_vram_mb: 3000,
            min_ram_mb: 6144,
            quality_score: 9.8,
            quality_tier: "9.8/10 · Altíssima".into(),
            speed_tier: "Muito Rápido (GPU)".into(),
            quality_notes: "Maior qualidade final da família Turbo (sem quantização). Superior ao Q8_0/Q5_0 em áudios com música de fundo, ruído ou fala sobreposta.".into(),
            description: "Precisão de topo com até 6× mais velocidade que o Large v3 clássico em 99 idiomas (excelente em PT-BR).".into(),
            download_url: format!("{HF_BASE}/ggml-large-v3-turbo.bin"),
        },
        WhisperModelSpec {
            id: "large-v3-turbo-q8_0".into(),
            family_id: "large-v3-turbo".into(),
            family_name: "Whisper Large v3 Turbo".into(),
            version_tag: "Q8_0 (8-bit Equilibrado)".into(),
            display_name: "Whisper Large v3 Turbo · Q8_0".into(),
            filename: "ggml-large-v3-turbo-q8_0.bin".into(),
            size_mb: 874,
            min_vram_mb: 2000,
            min_ram_mb: 4096,
            quality_score: 9.7,
            quality_tier: "9.7/10 · Quase FP16".into(),
            speed_tier: "Ultrarrápido".into(),
            quality_notes: "Ponto doce entre precisão e tamanho: perda praticamente imperceptível frente ao FP16 usando quase metade da VRAM/disco.".into(),
            description: "Ideal para uso diário rápido com qualidade praticamente idêntica ao FP16.".into(),
            download_url: format!("{HF_BASE}/ggml-large-v3-turbo-q8_0.bin"),
        },
        WhisperModelSpec {
            id: "large-v3-turbo-q5_0".into(),
            family_id: "large-v3-turbo".into(),
            family_name: "Whisper Large v3 Turbo".into(),
            version_tag: "Q5_0 (5-bit Compacto)".into(),
            display_name: "Whisper Large v3 Turbo · Q5_0".into(),
            filename: "ggml-large-v3-turbo-q5_0.bin".into(),
            size_mb: 574,
            min_vram_mb: 1400,
            min_ram_mb: 4096,
            quality_score: 9.4,
            quality_tier: "9.4/10 · Excelente".into(),
            speed_tier: "Ultrarrápido (GPU/CPU)".into(),
            quality_notes: "Versão mais leve do Turbo; pequena redução em termos técnicos raros comparado ao FP16/Q8_0, mas excelente em PT-BR e roda até em CPU.".into(),
            description: "Compacto (574 MB), baixa rápido e roda suave em GPUs de entrada ou CPU.".into(),
            download_url: format!("{HF_BASE}/ggml-large-v3-turbo-q5_0.bin"),
        },
        // ---- Family 2: Whisper Large v3 (Absolute Maximum Quality) ----
        WhisperModelSpec {
            id: "large-v3".into(),
            family_id: "large-v3".into(),
            family_name: "Whisper Large v3".into(),
            version_tag: "FP16 (Precisão Máxima)".into(),
            display_name: "Whisper Large v3 · FP16".into(),
            filename: "ggml-large-v3.bin".into(),
            size_mb: 3095,
            min_vram_mb: 5000,
            min_ram_mb: 8192,
            quality_score: 10.0,
            quality_tier: "10/10 · Qualidade Máxima".into(),
            speed_tier: "Pesado (GPU 6GB+)".into(),
            quality_notes: "Entrega a MAIOR qualidade final absoluta disponível localmente: máxima precisão em sotaques difíceis, termos técnicos, ruído alto e tradução.".into(),
            description: "Modelo completo de 1.55B parâmetros sem quantização para fidelidade absoluta.".into(),
            download_url: format!("{HF_BASE}/ggml-large-v3.bin"),
        },
        WhisperModelSpec {
            id: "large-v3-q5_0".into(),
            family_id: "large-v3".into(),
            family_name: "Whisper Large v3".into(),
            version_tag: "Q5_0 (5-bit Compacto)".into(),
            display_name: "Whisper Large v3 · Q5_0".into(),
            filename: "ggml-large-v3-q5_0.bin".into(),
            size_mb: 1080,
            min_vram_mb: 2400,
            min_ram_mb: 6144,
            quality_score: 9.6,
            quality_tier: "9.6/10 · Altíssima".into(),
            speed_tier: "Moderado".into(),
            quality_notes: "Retém a maior parte da fidelidade do Large v3 completo consumindo menos de 1/3 do espaço e VRAM.".into(),
            description: "Ótima opção de precisão máxima para GPUs de 4 GB ou CPU potente.".into(),
            download_url: format!("{HF_BASE}/ggml-large-v3-q5_0.bin"),
        },
        // ---- Family 3: Whisper Large v2 (Anti-Hallucination Classic) ----
        WhisperModelSpec {
            id: "large-v2-q8_0".into(),
            family_id: "large-v2".into(),
            family_name: "Whisper Large v2".into(),
            version_tag: "Q8_0 (8-bit Estável)".into(),
            display_name: "Whisper Large v2 · Q8_0".into(),
            filename: "ggml-large-v2-q8_0.bin".into(),
            size_mb: 1660,
            min_vram_mb: 3000,
            min_ram_mb: 6144,
            quality_score: 9.5,
            quality_tier: "9.5/10 · Muito Estável".into(),
            speed_tier: "Moderado".into(),
            quality_notes: "Muito apreciado para podcasts e lives longas por apresentar menos repetições/alucinações em trechos silenciosos.".into(),
            description: "Alternativa clássica de alta fidelidade para vídeos muito longos.".into(),
            download_url: format!("{HF_BASE}/ggml-large-v2-q8_0.bin"),
        },
        // ---- Family 4: Whisper Medium (Mid-range / CPU fallback) ----
        WhisperModelSpec {
            id: "medium-q8_0".into(),
            family_id: "medium".into(),
            family_name: "Whisper Medium".into(),
            version_tag: "Q8_0 (8-bit)".into(),
            display_name: "Whisper Medium · Q8_0".into(),
            filename: "ggml-medium-q8_0.bin".into(),
            size_mb: 823,
            min_vram_mb: 1800,
            min_ram_mb: 4096,
            quality_score: 8.8,
            quality_tier: "8.8/10 · Muito Boa".into(),
            speed_tier: "Rápido".into(),
            quality_notes: "Boa precisão geral, mas inferior ao Large v3 Turbo Q8_0 que tem tamanho parecido e qualidade bem superior.".into(),
            description: "Alternativa intermediária estável para PCs com memória limitada.".into(),
            download_url: format!("{HF_BASE}/ggml-medium-q8_0.bin"),
        },
        WhisperModelSpec {
            id: "medium-q5_0".into(),
            family_id: "medium".into(),
            family_name: "Whisper Medium".into(),
            version_tag: "Q5_0 (5-bit)".into(),
            display_name: "Whisper Medium · Q5_0".into(),
            filename: "ggml-medium-q5_0.bin".into(),
            size_mb: 539,
            min_vram_mb: 1200,
            min_ram_mb: 3500,
            quality_score: 8.5,
            quality_tier: "8.5/10 · Boa".into(),
            speed_tier: "Rápido (CPU)".into(),
            quality_notes: "Leve e confiável para máquinas mais antigas sem GPU dedicada.".into(),
            description: "Versão compacta do Medium para CPUs modestas.".into(),
            download_url: format!("{HF_BASE}/ggml-medium-q5_0.bin"),
        },
        // ---- Family 5: Whisper Small (Entry CPU / Low RAM) ----
        WhisperModelSpec {
            id: "small-q8_0".into(),
            family_id: "small".into(),
            family_name: "Whisper Small".into(),
            version_tag: "Q8_0 (8-bit)".into(),
            display_name: "Whisper Small · Q8_0".into(),
            filename: "ggml-small-q8_0.bin".into(),
            size_mb: 264,
            min_vram_mb: 800,
            min_ram_mb: 2048,
            quality_score: 7.9,
            quality_tier: "7.9/10 · Rápido".into(),
            speed_tier: "Instantâneo".into(),
            quality_notes: "Prioriza velocidade em PCs fracos; pode cometer erros em nomes próprios ou áudio ruidoso comparado ao Large v3 Turbo.".into(),
            description: "Indicado apenas para notebooks leves ou pouca memória RAM.".into(),
            download_url: format!("{HF_BASE}/ggml-small-q8_0.bin"),
        },
        WhisperModelSpec {
            id: "small-q5_1".into(),
            family_id: "small".into(),
            family_name: "Whisper Small".into(),
            version_tag: "Q5_1 (5-bit Ultraleve)".into(),
            display_name: "Whisper Small · Q5_1".into(),
            filename: "ggml-small-q5_1.bin".into(),
            size_mb: 190,
            min_vram_mb: 600,
            min_ram_mb: 1500,
            quality_score: 7.5,
            quality_tier: "7.5/10 · Ultraleve".into(),
            speed_tier: "Instantâneo".into(),
            quality_notes: "Menor consumo de memória possível mantendo português compreensível.".into(),
            description: "Opção mínima de emergência para hardware muito limitado.".into(),
            download_url: format!("{HF_BASE}/ggml-small-q5_1.bin"),
        },
    ]
}

fn detected_vram_mb(hw: &HardwareProfile) -> Option<u64> {
    hw.gpus.iter().filter_map(|g| g.vram_mb).max()
}

/// Evaluates whether a specific model version fits the user's GPU VRAM / system RAM cleanly.
pub fn evaluate_compatibility(spec: &WhisperModelSpec, hw: &HardwareProfile) -> (bool, String) {
    let has_dedicated_gpu = hw.recommended_accel != AccelBackend::CpuOnly;
    let vram = detected_vram_mb(hw);

    if has_dedicated_gpu {
        if let Some(vram_mb) = vram {
            if vram_mb >= spec.min_vram_mb {
                return (
                    true,
                    format!(
                        "Compatível com sua GPU ({:.1} GB VRAM ≥ {:.1} GB exigidos).",
                        vram_mb as f64 / 1024.0,
                        spec.min_vram_mb as f64 / 1024.0
                    ),
                );
            } else {
                return (
                    false,
                    format!(
                        "Exige ~{:.1} GB de VRAM, mas sua GPU tem {:.1} GB (pode ficar lento ou falhar por falta de VRAM).",
                        spec.min_vram_mb as f64 / 1024.0,
                        vram_mb as f64 / 1024.0
                    ),
                );
            }
        }
    }

    // CPU-only or GPU with unknown VRAM -> gate on system RAM
    if hw.total_memory_mb >= spec.min_ram_mb {
        let mode = if has_dedicated_gpu {
            "GPU/RAM"
        } else {
            "CPU/RAM"
        };
        (
            true,
            format!(
                "Compatível em modo {mode} ({:.1} GB RAM ≥ {:.1} GB exigidos).",
                hw.total_memory_mb as f64 / 1024.0,
                spec.min_ram_mb as f64 / 1024.0
            ),
        )
    } else {
        (
            false,
            format!(
                "Exige {:.1} GB de RAM (seu sistema possui {:.1} GB).",
                spec.min_ram_mb as f64 / 1024.0,
                hw.total_memory_mb as f64 / 1024.0
            ),
        )
    }
}

/// Decides which compatible models are part of the curated "Best for Your Hardware" view
/// (hiding weak Small/Medium models on strong GPUs so the user only sees top-quality options!).
pub fn is_curated_for_hardware(spec: &WhisperModelSpec, hw: &HardwareProfile, compatible: bool) -> bool {
    if !compatible {
        return false;
    }
    let vram = detected_vram_mb(hw).unwrap_or(0);
    let has_strong_gpu = hw.recommended_accel != AccelBackend::CpuOnly && vram >= 5000;
    let has_mid_gpu = hw.recommended_accel != AccelBackend::CpuOnly && vram >= 2000;

    if has_strong_gpu {
        // Strong GPU (6GB+ VRAM): only show top-tier Large v3 Turbo, Large v3, and Large v2!
        matches!(spec.family_id.as_str(), "large-v3-turbo" | "large-v3" | "large-v2")
    } else if has_mid_gpu {
        // Mid GPU (2–5GB VRAM): show Large v3 Turbo (all fitting quants) + Large v3 Q5_0 + Large v2
        matches!(spec.family_id.as_str(), "large-v3-turbo" | "large-v3" | "large-v2")
    } else if hw.total_memory_mb >= 6144 {
        // CPU with 8GB+ RAM: show Large v3 Turbo (compact/balanced) + Large v3 Q5_0 + Medium
        matches!(spec.family_id.as_str(), "large-v3-turbo" | "large-v3" | "medium")
    } else {
        // Low-RAM PC: show compact Turbo + Medium + Small
        matches!(spec.family_id.as_str(), "large-v3-turbo" | "medium" | "small")
    }
}

pub fn recommended_model_id(hw: &HardwareProfile) -> &'static str {
    let vram = detected_vram_mb(hw).unwrap_or(0);
    if hw.recommended_accel != AccelBackend::CpuOnly && vram >= 3500 {
        "large-v3-turbo"
    } else if hw.recommended_accel != AccelBackend::CpuOnly && vram >= 2000 {
        "large-v3-turbo-q8_0"
    } else if hw.total_memory_mb < 4096 {
        "small-q8_0"
    } else {
        "large-v3-turbo-q5_0"
    }
}

pub fn whisper_models_dir(kernel: &AicutKernel) -> PathBuf {
    kernel.workspace.models_dir.join("whisper")
}

pub fn resolve_model_path(kernel: &AicutKernel, model_id: &str) -> AicutResult<PathBuf> {
    let spec = catalog()
        .into_iter()
        .find(|m| m.id == model_id || m.filename == model_id)
        .ok_or_else(|| AicutError::not_found(format!("Unknown Whisper model '{model_id}'")))?;
    let path = whisper_models_dir(kernel).join(&spec.filename);
    if path.is_file() {
        Ok(path)
    } else {
        Err(AicutError::new(
            "MODEL_NOT_INSTALLED",
            format!("Whisper model '{}' is not downloaded yet.", spec.display_name),
        )
        .recoverable(format!(
            "Call transcribe.model_download with {{\"model_id\": \"{}\"}} first.",
            spec.id
        )))
    }
}

pub fn list_models(kernel: &AicutKernel) -> Vec<WhisperModelStatus> {
    let dir = whisper_models_dir(kernel);
    let rec_id = recommended_model_id(&kernel.hardware);

    let mut statuses: Vec<WhisperModelStatus> = catalog()
        .into_iter()
        .map(|spec| {
            let p = dir.join(&spec.filename);
            let meta = std::fs::metadata(&p).ok();
            let installed = meta
                .as_ref()
                .map(|m| m.is_file() && m.len() > 1024)
                .unwrap_or(false);
            let (compatible, compatibility_reason) = evaluate_compatibility(&spec, &kernel.hardware);
            let curated_for_hardware = is_curated_for_hardware(&spec, &kernel.hardware, compatible);

            WhisperModelStatus {
                recommended: spec.id == rec_id,
                best_quality_for_hardware: false,
                installed,
                compatible,
                compatibility_reason,
                curated_for_hardware,
                file_path: if installed {
                    Some(p.to_string_lossy().to_string())
                } else {
                    None
                },
                actual_size_bytes: if installed {
                    meta.map(|m| m.len())
                } else {
                    None
                },
                id: spec.id,
                family_id: spec.family_id,
                family_name: spec.family_name,
                version_tag: spec.version_tag,
                display_name: spec.display_name,
                filename: spec.filename,
                size_mb: spec.size_mb,
                min_vram_mb: spec.min_vram_mb,
                min_ram_mb: spec.min_ram_mb,
                quality_score: spec.quality_score,
                quality_tier: spec.quality_tier,
                speed_tier: spec.speed_tier,
                quality_notes: spec.quality_notes,
                description: spec.description,
            }
        })
        .collect();

    // Mark the single highest quality_score model among compatible ones as `best_quality_for_hardware`
    if let Some((best_idx, _)) = statuses
        .iter()
        .enumerate()
        .filter(|(_, m)| m.compatible)
        .max_by(|(_, a), (_, b)| {
            a.quality_score
                .partial_cmp(&b.quality_score)
                .unwrap_or(std::cmp::Ordering::Equal)
        })
    {
        statuses[best_idx].best_quality_for_hardware = true;
    }

    statuses
}

pub async fn download_model(
    kernel: Arc<AicutKernel>,
    model_id: &str,
    actor: ActorKind,
) -> AicutResult<WhisperModelStatus> {
    let spec = catalog()
        .into_iter()
        .find(|m| m.id == model_id)
        .ok_or_else(|| AicutError::not_found(format!("Unknown Whisper model '{model_id}'")))?;

    let dir = whisper_models_dir(&kernel);
    std::fs::create_dir_all(&dir)?;

    let _directory = aicut_core::workspace::lock_downloads(&kernel.workspace.models_dir)?;
    let stage = aicut_core::workspace::download_stage(&kernel.workspace.models_dir)?;
    let (job, cancel) = kernel.jobs.create_job(
        "transcribe.model_download",
        format!("Baixar modelo {}", spec.display_name),
        "transcribe",
        actor,
    );
    kernel.jobs.mark_running(&job.id, "downloading");
    kernel.jobs.append_log(
        &job.id,
        format!("Baixando {} de {}", spec.filename, spec.download_url),
    );

    let tmp_path = stage.path().join("weights");
    let final_path = dir.join(&spec.filename);

    let result: AicutResult<()> = async {
        let resp = kernel
            .tools
            .http_client()
            .get(&spec.download_url)
            .send()
            .await
            .map_err(|e| AicutError::internal(format!("Model download failed: {e}")))?;

        if !resp.status().is_success() {
            return Err(AicutError::internal(format!(
                "Model download failed with HTTP {}",
                resp.status()
            )));
        }

        let total = resp.content_length().or(Some(spec.size_mb * 1024 * 1024));
        let mut file = std::fs::File::create(&tmp_path)?;
        let mut stream = resp.bytes_stream();
        let mut downloaded: u64 = 0;
        let started = std::time::Instant::now();
        let mut last_emit = std::time::Instant::now();

        while let Some(chunk_res) = tokio::select! {
            _ = cancel.cancelled() => {
                let _ = std::fs::remove_file(&tmp_path);
                return Err(AicutError::cancelled("Model download cancelled"));
            }
            next = stream.next() => next,
        } {
            let chunk = chunk_res
                .map_err(|e| AicutError::internal(format!("Model download stream error: {e}")))?;
            file.write_all(&chunk)?;
            downloaded += chunk.len() as u64;

            if last_emit.elapsed().as_millis() >= 250 {
                let elapsed = started.elapsed().as_secs_f64().max(0.001);
                let speed = downloaded as f64 / elapsed;
                let percent = total
                    .map(|t| (downloaded as f64 / t as f64 * 99.0).min(99.0))
                    .unwrap_or(50.0);
                let eta = total.map(|t| t.saturating_sub(downloaded) as f64 / speed.max(1.0));
                kernel.jobs.update_progress(
                    &job.id,
                    JobProgress {
                        percent,
                        stage: "downloading".into(),
                        downloaded_bytes: Some(downloaded),
                        total_bytes: total,
                        speed_bytes_per_sec: Some(speed),
                        eta_seconds: eta,
                        message: Some(spec.display_name.clone()),
                    },
                );
                last_emit = std::time::Instant::now();
            }
        }
        file.flush()?;
        drop(file);

        std::fs::rename(&tmp_path, &final_path)?;
        Ok(())
    }
    .await;

    match result {
        Ok(()) => {
            let status = list_models(&kernel)
                .into_iter()
                .find(|m| m.id == spec.id)
                .ok_or_else(|| AicutError::internal("Model missing after install"))?;
            kernel.jobs.complete(
                &job.id,
                serde_json::to_value(&status).unwrap_or(serde_json::Value::Null),
            );
            Ok(status)
        }
        Err(err) => {
            let _ = std::fs::remove_file(&tmp_path);
            if cancel.is_cancelled() {
                kernel.jobs.mark_cancelled(&job.id);
            } else {
                kernel.jobs.fail(&job.id, err.clone());
            }
            Err(err)
        }
    }
}

/// Completely uninstalls a model from disk (`.bin` weights + any `.part` or companion files)
/// and reports exact freed bytes.
pub fn delete_model(kernel: &AicutKernel, model_id: &str) -> AicutResult<ModelUninstallResult> {
    let spec = catalog()
        .into_iter()
        .find(|m| m.id == model_id)
        .ok_or_else(|| AicutError::not_found(format!("Unknown Whisper model '{model_id}'")))?;
    let dir = whisper_models_dir(kernel);
    let _directory = aicut_core::workspace::lock_downloads(&kernel.workspace.models_dir)?;
    let candidates = [
        dir.join(&spec.filename),
        dir.join(format!(".{}.part", spec.filename)),
        dir.join(format!("{}.part", spec.filename)),
    ];

    let mut freed_bytes: u64 = 0;
    let mut removed_files = Vec::new();
    for path in candidates {
        match std::fs::symlink_metadata(&path) {
            Ok(meta) => {
                std::fs::remove_file(&path)?;
                freed_bytes += meta.len();
                removed_files.push(path.to_string_lossy().to_string());
            }
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => (),
            Err(e) => return Err(e.into()),
        }
    }
    kernel.repo.clear_selected_model("default_whisper_model", model_id)?;
    aicut_core::workspace::remove_empty_category(&dir)?;

    Ok(ModelUninstallResult {
        model_id: spec.id,
        deleted: !removed_files.is_empty(),
        freed_bytes,
        removed_files,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use aicut_core::{GpuInfo, GpuVendor};

    fn fake_hw(vram_mb: Option<u64>, ram_mb: u64, accel: AccelBackend) -> HardwareProfile {
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
                        name: "NVIDIA GeForce RTX".into(),
                        vendor: GpuVendor::Nvidia,
                        vram_mb: Some(v),
                        driver_version: Some("580.0".into()),
                    }]
                })
                .unwrap_or_default(),
            cuda_version: Some("13.2".into()),
            cuda_runtime_installed: true,
            vulkan_available: true,
            recommended_accel: accel,
            recommendation_summary: String::new(),
        }
    }

    #[test]
    fn curates_models_accurately_by_gpu_vram() {
        let rtx_12gb = fake_hw(Some(12288), 32768, AccelBackend::CudaNvenc);
        assert_eq!(recommended_model_id(&rtx_12gb), "large-v3-turbo");
        let large_v3 = catalog().into_iter().find(|m| m.id == "large-v3").unwrap();
        let small = catalog().into_iter().find(|m| m.id == "small-q8_0").unwrap();
        assert!(evaluate_compatibility(&large_v3, &rtx_12gb).0);
        assert!(is_curated_for_hardware(&large_v3, &rtx_12gb, true));
        assert!(!is_curated_for_hardware(&small, &rtx_12gb, true));

        let gtx_4gb = fake_hw(Some(4096), 16384, AccelBackend::CudaNvenc);
        assert!(!evaluate_compatibility(&large_v3, &gtx_4gb).0);
        let turbo_q8 = catalog().into_iter().find(|m| m.id == "large-v3-turbo-q8_0").unwrap();
        assert!(evaluate_compatibility(&turbo_q8, &gtx_4gb).0);
        assert!(is_curated_for_hardware(&turbo_q8, &gtx_4gb, true));
    }
}
