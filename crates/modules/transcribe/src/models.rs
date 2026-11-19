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
    pub display_name: String,
    pub filename: String,
    pub size_mb: u64,
    pub quality_tier: String,
    pub speed_tier: String,
    pub description: String,
    pub download_url: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct WhisperModelStatus {
    pub id: String,
    pub display_name: String,
    pub filename: String,
    pub size_mb: u64,
    pub quality_tier: String,
    pub speed_tier: String,
    pub description: String,
    pub installed: bool,
    pub recommended: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub file_path: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub actual_size_bytes: Option<u64>,
}

const HF_BASE: &str = "https://huggingface.co/ggerganov/whisper.cpp/resolve/main";

pub fn catalog() -> Vec<WhisperModelSpec> {
    vec![
        WhisperModelSpec {
            id: "large-v3-turbo-q5_0".into(),
            display_name: "Whisper Large v3 Turbo (Q5_0)".into(),
            filename: "ggml-large-v3-turbo-q5_0.bin".into(),
            size_mb: 574,
            quality_tier: "Excelente".into(),
            speed_tier: "Muito Rápido".into(),
            description: "Melhor equilíbrio geral para Português e 99 idiomas — qualidade de Large v3 com ~6x mais velocidade e baixo uso de memória.".into(),
            download_url: format!("{HF_BASE}/ggml-large-v3-turbo-q5_0.bin"),
        },
        WhisperModelSpec {
            id: "large-v3-turbo".into(),
            display_name: "Whisper Large v3 Turbo (FP16)".into(),
            filename: "ggml-large-v3-turbo.bin".into(),
            size_mb: 1624,
            quality_tier: "Altíssima".into(),
            speed_tier: "Rápido (GPU)".into(),
            description: "Versão completa sem quantização do Large v3 Turbo — ideal para GPUs NVIDIA/AMD com 4GB+ VRAM.".into(),
            download_url: format!("{HF_BASE}/ggml-large-v3-turbo.bin"),
        },
        WhisperModelSpec {
            id: "small".into(),
            display_name: "Whisper Small".into(),
            filename: "ggml-small.bin".into(),
            size_mb: 488,
            quality_tier: "Boa".into(),
            speed_tier: "Ultrarrápido".into(),
            description: "Leve e veloz para CPUs mais simples ou notebooks sem GPU dedicada.".into(),
            download_url: format!("{HF_BASE}/ggml-small.bin"),
        },
        WhisperModelSpec {
            id: "medium".into(),
            display_name: "Whisper Medium".into(),
            filename: "ggml-medium.bin".into(),
            size_mb: 1533,
            quality_tier: "Muito Boa".into(),
            speed_tier: "Moderado".into(),
            description: "Modelo clássico equilibrado para transcrição multilíngue.".into(),
            download_url: format!("{HF_BASE}/ggml-medium.bin"),
        },
        WhisperModelSpec {
            id: "large-v3-q5_0".into(),
            display_name: "Whisper Large v3 (Q5_0)".into(),
            filename: "ggml-large-v3-q5_0.bin".into(),
            size_mb: 1080,
            quality_tier: "Máxima".into(),
            speed_tier: "Moderado".into(),
            description: "Precisão máxima em áudios difíceis, sotaques fortes e ruído de fundo em formato compacto.".into(),
            download_url: format!("{HF_BASE}/ggml-large-v3-q5_0.bin"),
        },
        WhisperModelSpec {
            id: "large-v3".into(),
            display_name: "Whisper Large v3 (FP16)".into(),
            filename: "ggml-large-v3.bin".into(),
            size_mb: 3095,
            quality_tier: "Máxima Absoluta".into(),
            speed_tier: "Pesado (GPU 6GB+)".into(),
            description: "Modelo completo de máxima fidelidade para estações com GPU dedicada potente.".into(),
            download_url: format!("{HF_BASE}/ggml-large-v3.bin"),
        },
    ]
}

pub fn recommended_model_id(hw: &HardwareProfile) -> &'static str {
    let has_strong_gpu = hw.recommended_accel == AccelBackend::CudaNvenc
        && hw.gpus.first().and_then(|g| g.vram_mb).unwrap_or(0) >= 6000;
    if has_strong_gpu {
        "large-v3-turbo"
    } else if hw.total_memory_mb < 6000 {
        "small"
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
    catalog()
        .into_iter()
        .map(|spec| {
            let p = dir.join(&spec.filename);
            let meta = std::fs::metadata(&p).ok();
            let installed = meta.as_ref().map(|m| m.is_file() && m.len() > 1024).unwrap_or(false);
            WhisperModelStatus {
                recommended: spec.id == rec_id,
                installed,
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
                display_name: spec.display_name,
                filename: spec.filename,
                size_mb: spec.size_mb,
                quality_tier: spec.quality_tier,
                speed_tier: spec.speed_tier,
                description: spec.description,
            }
        })
        .collect()
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

    let (job, cancel) = kernel.jobs.create_job(
        "transcribe.model_download",
        format!("Baixar modelo {}", spec.display_name),
        "transcribe",
        actor,
    );
    kernel.jobs.mark_running(&job.id, "downloading");
    kernel
        .jobs
        .append_log(&job.id, format!("Baixando {} de {}", spec.filename, spec.download_url));

    let tmp_path = dir.join(format!(".{}.part", spec.filename));
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

        if final_path.exists() {
            let _ = std::fs::remove_file(&final_path);
        }
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

pub fn delete_model(kernel: &AicutKernel, model_id: &str) -> AicutResult<bool> {
    let spec = catalog()
        .into_iter()
        .find(|m| m.id == model_id)
        .ok_or_else(|| AicutError::not_found(format!("Unknown Whisper model '{model_id}'")))?;
    let path = whisper_models_dir(kernel).join(&spec.filename);
    if path.is_file() {
        std::fs::remove_file(&path)?;
        Ok(true)
    } else {
        Ok(false)
    }
}
