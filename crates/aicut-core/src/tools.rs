use futures_util::StreamExt;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::{Arc, RwLock};
use tokio::process::Command;

use crate::error::{AicutError, AicutResult};
use crate::events::{ActorKind, AicutEvent, EventBus};
use crate::hardware::{AccelBackend, HardwareProfile};
use crate::jobs::{JobManager, JobProgress};
use crate::process::configure_tokio_command;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum ToolInstallStrategy {
    /// Downloads the official yt-dlp standalone binary matching host OS & CPU arch.
    YtDlpOfficialRelease,
    /// Downloads the optimal FFmpeg/FFprobe build (including NVENC/AMF/QSV support on Windows x64).
    FfmpegHardwareBundle,
    /// Downloads the official ggml-org/whisper.cpp release matching GPU/CPU hardware (CUDA 12 / Vulkan / OpenBLAS / ARM64 / Linux).
    WhisperCppOfficialRelease,
    /// Downloads the official ggml-org/llama.cpp release matching GPU/CPU hardware (CUDA + cudart / Vulkan / Metal / CPU).
    LlamaCppOfficialRelease,
    /// Companion binary provided alongside another managed tool (e.g. ffprobe inside ffmpeg bundle).
    CompanionOf(String),
    /// Placeholder for future module-provided tools.
    Custom { homepage: String },
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct ToolDefinition {
    pub id: String,
    pub display_name: String,
    pub description: String,
    pub owner_module: String,
    pub category: String,
    /// Base binary name without .exe suffix (e.g. "yt-dlp", "ffmpeg", "ffprobe", "whisper-cli").
    pub binary_name: String,
    pub version_args: Vec<String>,
    pub strategy: ToolInstallStrategy,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct ToolStatus {
    pub id: String,
    pub display_name: String,
    pub description: String,
    pub owner_module: String,
    pub category: String,
    pub installed: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub installed_version: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub latest_version: Option<String>,
    pub update_available: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub binary_path: Option<String>,
    pub managed_by_aicut: bool,
    pub recommended_build: String,
    pub hardware_note: String,
}

#[derive(Debug, Clone)]
pub struct ResolvedDownload {
    pub url: String,
    pub companion_url: Option<String>,
    pub version: String,
    pub build_label: String,
    pub archive_kind: ArchiveKind,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ArchiveKind {
    SingleBinary,
    FfmpegZip,
    FfmpegTarGz,
    WhisperZip,
    WhisperTarGz,
    LlamaZip,
    LlamaTarGz,
}

pub struct ToolManager {
    bin_dir: PathBuf,
    hardware: HardwareProfile,
    definitions: RwLock<HashMap<String, ToolDefinition>>,
    latest_cache: RwLock<HashMap<String, String>>,
    events: Arc<EventBus>,
    jobs: Arc<JobManager>,
    http: reqwest::Client,
}

impl ToolManager {
    pub fn new(
        bin_dir: PathBuf,
        hardware: HardwareProfile,
        events: Arc<EventBus>,
        jobs: Arc<JobManager>,
    ) -> Self {
        let http = reqwest::Client::builder()
            .user_agent("Aicut/0.2.0 (+https://github.com/limitbreakingofc/Aicut)")
            .build()
            .unwrap_or_else(|_| reqwest::Client::new());

        Self {
            bin_dir,
            hardware,
            definitions: RwLock::new(HashMap::new()),
            latest_cache: RwLock::new(HashMap::new()),
            events,
            jobs,
            http,
        }
    }

    pub fn hardware(&self) -> &HardwareProfile {
        &self.hardware
    }

    pub fn http_client(&self) -> &reqwest::Client {
        &self.http
    }

    pub fn register_tool(&self, def: ToolDefinition) {
        if let Ok(mut guard) = self.definitions.write() {
            guard.insert(def.id.clone(), def);
        }
    }

    pub fn list_definitions(&self) -> Vec<ToolDefinition> {
        let mut items: Vec<ToolDefinition> = self
            .definitions
            .read()
            .map(|g| g.values().cloned().collect())
            .unwrap_or_default();
        items.sort_by(|a, b| a.id.cmp(&b.id));
        items
    }

    /// Resolves a binary path: checks Aicut's managed `bin_dir` (and `bin_dir/whisper`), then system PATH.
    pub fn resolve_binary(&self, binary_name: &str) -> Option<(PathBuf, bool)> {
        let managed_name = platform_binary_name(binary_name);
        let direct = self.bin_dir.join(&managed_name);
        if direct.is_file() {
            return Some((direct, true));
        }
        let in_whisper_dir = self.bin_dir.join("whisper").join(&managed_name);
        if in_whisper_dir.is_file() {
            return Some((in_whisper_dir, true));
        }
        let in_llama_dir = self.bin_dir.join("llama").join(&managed_name);
        if in_llama_dir.is_file() {
            return Some((in_llama_dir, true));
        }
        // Fallback for older whisper.cpp builds where the executable was named `main`
        if binary_name == "whisper-cli" {
            let legacy = self.bin_dir.join("whisper").join(platform_binary_name("main"));
            if legacy.is_file() {
                return Some((legacy, true));
            }
        }
        if let Ok(path) = which::which(binary_name) {
            return Some((path, false));
        }
        None
    }

    pub fn resolve_tool(&self, tool_id: &str) -> AicutResult<PathBuf> {
        let def = self
            .definitions
            .read()
            .ok()
            .and_then(|g| g.get(tool_id).cloned())
            .ok_or_else(|| AicutError::not_found(format!("Unknown tool '{tool_id}'")))?;

        self.resolve_binary(&def.binary_name)
            .map(|(p, _)| p)
            .ok_or_else(|| {
                AicutError::dependency_missing(
                    tool_id,
                    format!("Binary '{}' not found in managed bin dir or PATH", def.binary_name),
                )
            })
    }

    pub fn recommend_build_for(&self, def: &ToolDefinition) -> (String, String) {
        let tag = self.hardware.platform_tag.as_str();
        let accel = self.hardware.recommended_accel;

        match &def.strategy {
            ToolInstallStrategy::YtDlpOfficialRelease => {
                let build = match tag {
                    "win-x64" => "yt-dlp.exe (Windows x64 Official)",
                    "win-arm64" => "yt-dlp_arm64.exe (Windows ARM64 Official)",
                    "linux-x64" => "yt-dlp_linux (Linux x64 Standalone)",
                    "linux-arm64" => "yt-dlp_linux_aarch64 (Linux ARM64 Standalone)",
                    "macos-arm64" | "macos-x64" => "yt-dlp_macos (macOS Universal)",
                    _ => "yt-dlp (Generic)",
                };
                (
                    build.to_string(),
                    format!("Selecionado automaticamente para plataforma {tag}."),
                )
            }
            ToolInstallStrategy::FfmpegHardwareBundle | ToolInstallStrategy::CompanionOf(_) => {
                let build = match tag {
                    "win-x64" => "FFmpeg Win64 GPL (NVENC + AMF + QSV)",
                    "win-arm64" => "FFmpeg WinARM64 Native GPL",
                    "linux-x64" => "FFmpeg Linux64 GPL (VAAPI/NVENC)",
                    "linux-arm64" => "FFmpeg Linux ARM64 GPL",
                    _ => "FFmpeg System/Native Build",
                };
                let hw_note = match accel {
                    AccelBackend::CudaNvenc => {
                        "GPU NVIDIA detectada: build inclui suporte NVENC/NVDEC acelerado por hardware."
                    }
                    AccelBackend::AmfDirectml => {
                        "GPU AMD detectada: build inclui suporte AMF/D3D11VA acelerado por hardware."
                    }
                    AccelBackend::QsvOpenvino => {
                        "GPU Intel detectada: build inclui suporte QuickSync (QSV) acelerado por hardware."
                    }
                    AccelBackend::Metal => "Apple Silicon detectado: suporte VideoToolbox nativo.",
                    AccelBackend::CpuOnly => {
                        "Modo CPU otimizado: build multithread para todos os núcleos lógicos."
                    }
                };
                (build.to_string(), hw_note.to_string())
            }
            ToolInstallStrategy::WhisperCppOfficialRelease => match (tag, accel) {
                ("win-x64", AccelBackend::CudaNvenc) => {
                    let cuda_label = self
                        .hardware
                        .cuda_version
                        .as_deref()
                        .map(|v| format!("CUDA {v}"))
                        .unwrap_or_else(|| "CUDA".to_string());
                    (
                        format!("whisper.cpp Win64 {cuda_label} (GPU NVIDIA)"),
                        format!(
                            "GPU NVIDIA ({cuda_label}) detectada: seleciona automaticamente a maior build CUDA compatível (≤ {cuda_label}) publicada."
                        ),
                    )
                }
                ("win-arm64", AccelBackend::CudaNvenc) => {
                    let cuda_label = self
                        .hardware
                        .cuda_version
                        .as_deref()
                        .map(|v| format!("CUDA {v}"))
                        .unwrap_or_else(|| "CUDA".to_string());
                    (
                        format!("whisper.cpp Win ARM64 {cuda_label}"),
                        format!("Windows ARM64 com NVIDIA ({cuda_label}): seleciona a maior build CUDA compatível."),
                    )
                }
                ("win-x64", AccelBackend::AmfDirectml | AccelBackend::QsvOpenvino) => (
                    "whisper.cpp Win64 Vulkan / OpenBLAS".to_string(),
                    "GPU AMD/Intel detectada: prefere build Vulkan se disponível ou OpenBLAS SIMD otimizado."
                        .to_string(),
                ),
                ("win-x64", _) => (
                    "whisper.cpp Win64 OpenBLAS AVX2 (CPU)".to_string(),
                    "Build oficial OpenBLAS otimizada para múltiplos núcleos de CPU no Windows 11."
                        .to_string(),
                ),
                ("win-arm64", _) => (
                    "whisper.cpp Win ARM64 Native".to_string(),
                    "Build oficial nativa para Windows 11 ARM64.".to_string(),
                ),
                ("linux-x64", _) => (
                    "whisper.cpp Linux x64 Official".to_string(),
                    "Build oficial pré-compilada Linux x64.".to_string(),
                ),
                ("linux-arm64", _) => (
                    "whisper.cpp Linux ARM64 Official".to_string(),
                    "Build oficial pré-compilada Linux ARM64.".to_string(),
                ),
                _ => (
                    format!("whisper.cpp ({tag})"),
                    "Build oficial ggml-org/whisper.cpp.".to_string(),
                ),
            },
            ToolInstallStrategy::LlamaCppOfficialRelease => match (tag, accel) {
                ("win-x64", AccelBackend::CudaNvenc) => {
                    let cuda_label = self
                        .hardware
                        .cuda_version
                        .as_deref()
                        .map(|v| format!("CUDA {v}"))
                        .unwrap_or_else(|| "CUDA".to_string());
                    (
                        format!("llama.cpp Win64 {cuda_label} + cudart (GPU NVIDIA)"),
                        format!(
                            "GPU NVIDIA ({cuda_label}) detectada: baixa automaticamente a maior build CUDA compatível (≤ {cuda_label}) + bibliotecas cudart para rodar modelos visuais GGUF na GPU."
                        ),
                    )
                }
                ("win-x64", AccelBackend::AmfDirectml | AccelBackend::QsvOpenvino) => (
                    "llama.cpp Win64 Vulkan (GPU AMD/Intel)".to_string(),
                    "GPU AMD/Intel detectada: seleciona build oficial Vulkan para acelerar modelos visuais GGUF na GPU.".to_string(),
                ),
                ("win-x64", _) => (
                    "llama.cpp Win64 CPU AVX2".to_string(),
                    "Build oficial CPU otimizada para Windows 11.".to_string(),
                ),
                ("win-arm64", _) => (
                    "llama.cpp Win ARM64 Native".to_string(),
                    "Build oficial nativa para Windows 11 ARM64.".to_string(),
                ),
                ("macos-arm64", _) => (
                    "llama.cpp macOS ARM64 Metal".to_string(),
                    "Apple Silicon detectado: build oficial com aceleração Metal.".to_string(),
                ),
                ("linux-x64", _) => (
                    "llama.cpp Linux x64 Official".to_string(),
                    "Build oficial pré-compilada Linux x64.".to_string(),
                ),
                _ => (
                    format!("llama.cpp ({tag})"),
                    "Build oficial ggml-org/llama.cpp.".to_string(),
                ),
            },
            ToolInstallStrategy::Custom { homepage } => (
                format!("Hardware-matched ({tag})"),
                format!("Gerenciado pelo módulo proprietário ({homepage})."),
            ),
        }
    }

    pub async fn status_of(&self, tool_id: &str) -> AicutResult<ToolStatus> {
        let def = self
            .definitions
            .read()
            .ok()
            .and_then(|g| g.get(tool_id).cloned())
            .ok_or_else(|| AicutError::not_found(format!("Unknown tool '{tool_id}'")))?;

        let (recommended_build, hardware_note) = self.recommend_build_for(&def);
        let resolved = self.resolve_binary(&def.binary_name);
        let (installed, binary_path, managed_by_aicut, installed_version) = match resolved {
            Some((path, managed)) => {
                // For managed whisper.cpp or llama.cpp, read stored version marker if available
                let marker_ver = if def.id == "whisper" && managed {
                    std::fs::read_to_string(self.bin_dir.join("whisper").join(".version"))
                        .ok()
                        .map(|s| s.trim().to_string())
                        .filter(|s| !s.is_empty())
                } else if def.id == "llama-server" && managed {
                    std::fs::read_to_string(self.bin_dir.join("llama").join(".version"))
                        .ok()
                        .map(|s| s.trim().to_string())
                        .filter(|s| !s.is_empty())
                } else {
                    None
                };
                let ver = match marker_ver {
                    Some(v) => Some(v),
                    None => probe_binary_version(&path, &def.version_args).await,
                };
                (true, Some(path.to_string_lossy().to_string()), managed, ver)
            }
            None => (false, None, false, None),
        };

        let latest_version = self
            .latest_cache
            .read()
            .ok()
            .and_then(|g| g.get(&def.id).cloned());

        let update_available = match (&installed_version, &latest_version) {
            (Some(inst), Some(latest)) => !versions_match(inst, latest),
            (None, Some(_)) => true,
            _ => false,
        };

        Ok(ToolStatus {
            id: def.id,
            display_name: def.display_name,
            description: def.description,
            owner_module: def.owner_module,
            category: def.category,
            installed,
            installed_version,
            latest_version,
            update_available,
            binary_path,
            managed_by_aicut,
            recommended_build,
            hardware_note,
        })
    }

    pub async fn list_statuses(&self) -> Vec<ToolStatus> {
        let defs = self.list_definitions();
        let mut results = Vec::with_capacity(defs.len());
        for def in defs {
            if let Ok(st) = self.status_of(&def.id).await {
                results.push(st);
            }
        }
        results
    }

    /// Checks remote GitHub releases for the newest versions of registered tools.
    pub async fn check_updates(&self) -> AicutResult<Vec<ToolStatus>> {
        let defs = self.list_definitions();
        for def in &defs {
            match &def.strategy {
                ToolInstallStrategy::YtDlpOfficialRelease => {
                    if let Ok(resolved) = self.resolve_ytdlp_release().await {
                        if let Ok(mut cache) = self.latest_cache.write() {
                            cache.insert(def.id.clone(), resolved.version);
                        }
                    }
                }
                ToolInstallStrategy::FfmpegHardwareBundle => {
                    if let Ok(resolved) = self.resolve_ffmpeg_release().await {
                        if let Ok(mut cache) = self.latest_cache.write() {
                            cache.insert("ffmpeg".to_string(), resolved.version.clone());
                            cache.insert("ffprobe".to_string(), resolved.version);
                        }
                    }
                }
                ToolInstallStrategy::WhisperCppOfficialRelease => {
                    if let Ok(resolved) = self.resolve_whisper_release().await {
                        if let Ok(mut cache) = self.latest_cache.write() {
                            cache.insert(def.id.clone(), resolved.version);
                        }
                    }
                }
                ToolInstallStrategy::LlamaCppOfficialRelease => {
                    if let Ok(resolved) = self.resolve_llama_release().await {
                        if let Ok(mut cache) = self.latest_cache.write() {
                            cache.insert(def.id.clone(), resolved.version);
                        }
                    }
                }
                ToolInstallStrategy::CompanionOf(_) | ToolInstallStrategy::Custom { .. } => {}
            }
        }
        let statuses = self.list_statuses().await;
        for st in &statuses {
            self.events.publish(AicutEvent::ToolStatusChanged(st.clone()));
        }
        Ok(statuses)
    }

    /// Installs or updates a tool automatically choosing the optimal build for the detected hardware.
    pub async fn install_or_update(&self, tool_id: &str, actor: ActorKind) -> AicutResult<ToolStatus> {
        let def = self
            .definitions
            .read()
            .ok()
            .and_then(|g| g.get(tool_id).cloned())
            .ok_or_else(|| AicutError::not_found(format!("Unknown tool '{tool_id}'")))?;

        let target_id = match &def.strategy {
            ToolInstallStrategy::CompanionOf(parent) => parent.clone(),
            _ => def.id.clone(),
        };

        let target_def = self
            .definitions
            .read()
            .ok()
            .and_then(|g| g.get(&target_id).cloned())
            .unwrap_or(def.clone());

        std::fs::create_dir_all(&self.bin_dir)?;

        let (job, cancel) = self.jobs.create_job(
            "tool.install",
            format!("Instalar/Atualizar {}", target_def.display_name),
            "tools",
            actor,
        );
        self.jobs.mark_running(&job.id, "resolving_release");

        let outcome: AicutResult<ToolStatus> = async {
            let resolved = match &target_def.strategy {
                ToolInstallStrategy::YtDlpOfficialRelease => self.resolve_ytdlp_release().await?,
                ToolInstallStrategy::FfmpegHardwareBundle => self.resolve_ffmpeg_release().await?,
                ToolInstallStrategy::WhisperCppOfficialRelease => {
                    self.resolve_whisper_release().await?
                }
                ToolInstallStrategy::LlamaCppOfficialRelease => {
                    self.resolve_llama_release().await?
                }
                ToolInstallStrategy::CompanionOf(_) => {
                    return Err(AicutError::internal("Companion tool resolution loop"));
                }
                ToolInstallStrategy::Custom { homepage } => {
                    return Err(AicutError::new(
                        "UNSUPPORTED_AUTO_INSTALL",
                        format!(
                            "Tool '{}' must be installed via its module ({homepage}).",
                            target_def.id
                        ),
                    ));
                }
            };

            self.jobs.append_log(
                &job.id,
                format!(
                    "Hardware: {} ({:?}) -> Build selecionada: {} ({})",
                    self.hardware.platform_tag,
                    self.hardware.recommended_accel,
                    resolved.build_label,
                    resolved.version
                ),
            );
            self.jobs
                .append_log(&job.id, format!("Baixando de {}", resolved.url));

            let tmp_path = self.bin_dir.join(format!(".download-{}.tmp", job.id));
            self.download_file(&resolved.url, &tmp_path, &job.id, &cancel)
                .await?;

            self.jobs.update_progress(
                &job.id,
                JobProgress {
                    percent: 92.0,
                    stage: "installing".to_string(),
                    message: Some(format!("Instalando {}...", target_def.display_name)),
                    ..Default::default()
                },
            );

            match resolved.archive_kind {
                ArchiveKind::SingleBinary => {
                    let final_path = self.bin_dir.join(platform_binary_name(&target_def.binary_name));
                    replace_file(&tmp_path, &final_path)?;
                    make_executable(&final_path)?;
                }
                ArchiveKind::FfmpegZip => {
                    extract_ffmpeg_from_zip(&tmp_path, &self.bin_dir)?;
                    let _ = std::fs::remove_file(&tmp_path);
                }
                ArchiveKind::FfmpegTarGz => {
                    extract_ffmpeg_from_targz(&tmp_path, &self.bin_dir)?;
                    let _ = std::fs::remove_file(&tmp_path);
                }
                ArchiveKind::WhisperZip => {
                    let whisper_dir = self.bin_dir.join("whisper");
                    extract_whisper_from_zip(&tmp_path, &whisper_dir)?;
                    let _ = std::fs::write(whisper_dir.join(".version"), &resolved.version);
                    let _ = std::fs::remove_file(&tmp_path);
                }
                ArchiveKind::WhisperTarGz => {
                    let whisper_dir = self.bin_dir.join("whisper");
                    extract_whisper_from_targz(&tmp_path, &whisper_dir)?;
                    let _ = std::fs::write(whisper_dir.join(".version"), &resolved.version);
                    let _ = std::fs::remove_file(&tmp_path);
                }
                ArchiveKind::LlamaZip => {
                    let llama_dir = self.bin_dir.join("llama");
                    extract_llama_from_zip(&tmp_path, &llama_dir)?;
                    let _ = std::fs::remove_file(&tmp_path);
                    if let Some(ref comp_url) = resolved.companion_url {
                        self.jobs.append_log(
                            &job.id,
                            format!("Baixando bibliotecas CUDA runtime (cudart) de {comp_url}"),
                        );
                        let comp_tmp = self.bin_dir.join(format!(".download-{}-cudart.tmp", job.id));
                        self.download_file(comp_url, &comp_tmp, &job.id, &cancel)
                            .await?;
                        extract_llama_from_zip(&comp_tmp, &llama_dir)?;
                        let _ = std::fs::remove_file(&comp_tmp);
                    }
                    let _ = std::fs::write(llama_dir.join(".version"), &resolved.version);
                }
                ArchiveKind::LlamaTarGz => {
                    let llama_dir = self.bin_dir.join("llama");
                    extract_llama_from_targz(&tmp_path, &llama_dir)?;
                    let _ = std::fs::write(llama_dir.join(".version"), &resolved.version);
                    let _ = std::fs::remove_file(&tmp_path);
                }
            }

            if let Ok(mut cache) = self.latest_cache.write() {
                cache.insert(target_def.id.clone(), resolved.version.clone());
                if target_def.id == "ffmpeg" {
                    cache.insert("ffprobe".to_string(), resolved.version);
                }
            }

            let status = self.status_of(tool_id).await?;
            self.events
                .publish(AicutEvent::ToolStatusChanged(status.clone()));
            if target_def.id == "ffmpeg" {
                if let Ok(probe_st) = self.status_of("ffprobe").await {
                    self.events.publish(AicutEvent::ToolStatusChanged(probe_st));
                }
            }
            Ok(status)
        }
        .await;

        match outcome {
            Ok(status) => {
                self.jobs.complete(
                    &job.id,
                    serde_json::to_value(&status).unwrap_or(serde_json::Value::Null),
                );
                Ok(status)
            }
            Err(err) => {
                if cancel.is_cancelled() {
                    self.jobs.mark_cancelled(&job.id);
                } else {
                    self.jobs.fail(&job.id, err.clone());
                }
                Err(err)
            }
        }
    }

    async fn resolve_ytdlp_release(&self) -> AicutResult<ResolvedDownload> {
        let asset_name = match self.hardware.platform_tag.as_str() {
            "win-x64" => "yt-dlp.exe",
            "win-arm64" => "yt-dlp_arm64.exe",
            "linux-x64" => "yt-dlp_linux",
            "linux-arm64" => "yt-dlp_linux_aarch64",
            "macos-arm64" | "macos-x64" => "yt-dlp_macos",
            _ => "yt-dlp",
        };

        let resp = self
            .http
            .get("https://api.github.com/repos/yt-dlp/yt-dlp/releases/latest")
            .send()
            .await
            .map_err(|e| AicutError::internal(format!("Failed querying yt-dlp releases: {e}")))?;

        let json: serde_json::Value = resp
            .json()
            .await
            .map_err(|e| AicutError::internal(format!("Invalid GitHub release payload: {e}")))?;

        let tag_name = json
            .get("tag_name")
            .and_then(|v| v.as_str())
            .unwrap_or("latest")
            .to_string();

        let download_url = json
            .get("assets")
            .and_then(|v| v.as_array())
            .and_then(|assets| {
                assets.iter().find_map(|a| {
                    let name = a.get("name").and_then(|n| n.as_str())?;
                    if name == asset_name {
                        a.get("browser_download_url")
                            .and_then(|u| u.as_str())
                            .map(|s| s.to_string())
                    } else {
                        None
                    }
                })
            })
            .unwrap_or_else(|| {
                format!("https://github.com/yt-dlp/yt-dlp/releases/latest/download/{asset_name}")
            });

        Ok(ResolvedDownload {
            url: download_url,
            companion_url: None,
            version: tag_name,
            build_label: asset_name.to_string(),
            archive_kind: ArchiveKind::SingleBinary,
        })
    }

    async fn resolve_ffmpeg_release(&self) -> AicutResult<ResolvedDownload> {
        let tag = self.hardware.platform_tag.as_str();
        match tag {
            "win-x64" => Ok(ResolvedDownload {
                url: "https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-master-latest-win64-gpl.zip".to_string(),
                companion_url: None,
                version: "latest-win64-gpl".to_string(),
                build_label: "ffmpeg-master-latest-win64-gpl.zip (NVENC/AMF/QSV)".to_string(),
                archive_kind: ArchiveKind::FfmpegZip,
            }),
            "win-arm64" => Ok(ResolvedDownload {
                url: "https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-master-latest-winarm64-gpl.zip".to_string(),
                companion_url: None,
                version: "latest-winarm64-gpl".to_string(),
                build_label: "ffmpeg-master-latest-winarm64-gpl.zip".to_string(),
                archive_kind: ArchiveKind::FfmpegZip,
            }),
            "linux-x64" | "linux-arm64" => Err(AicutError::new(
                "LINUX_FFMPEG_SYSTEM",
                "On Linux, please install ffmpeg via your distribution package manager (e.g. apt/dnf/pacman) or place binaries in the Aicut bin directory.",
            )),
            _ => Err(AicutError::new(
                "UNSUPPORTED_PLATFORM",
                format!("Automatic FFmpeg download is not configured for platform '{tag}'."),
            )),
        }
    }

    /// Queries the latest releases of `ggml-org/whisper.cpp` (skipping source-only tags)
    /// and picks the optimal binary archive for the host OS/CPU/GPU profile.
    async fn resolve_whisper_release(&self) -> AicutResult<ResolvedDownload> {
        let resp = self
            .http
            .get("https://api.github.com/repos/ggml-org/whisper.cpp/releases?per_page=15")
            .send()
            .await
            .map_err(|e| {
                AicutError::internal(format!("Failed querying whisper.cpp releases: {e}"))
            })?;

        let releases: Vec<serde_json::Value> = resp
            .json()
            .await
            .map_err(|e| AicutError::internal(format!("Invalid whisper.cpp releases JSON: {e}")))?;

        let tag = self.hardware.platform_tag.as_str();
        let accel = self.hardware.recommended_accel;

        for rel in &releases {
            let Some(assets) = rel.get("assets").and_then(|v| v.as_array()) else {
                continue;
            };
            if assets.is_empty() {
                continue;
            }
            let version = rel
                .get("tag_name")
                .and_then(|v| v.as_str())
                .unwrap_or("latest")
                .to_string();

            if let Some((name, url, kind)) = select_best_whisper_asset(
                assets,
                tag,
                accel,
                self.hardware.cuda_major_minor(),
            ) {
                return Ok(ResolvedDownload {
                    url,
                    companion_url: None,
                    version,
                    build_label: name,
                    archive_kind: kind,
                });
            }
        }

        Err(AicutError::new(
            "WHISPER_ASSET_NOT_FOUND",
            format!("No prebuilt whisper.cpp binary found for platform '{tag}'."),
        ))
    }

    /// Queries the latest releases of `ggml-org/llama.cpp` and picks the highest-quality
    /// hardware-compatible binary archive (plus matching `cudart` runtime zip for Windows CUDA).
    async fn resolve_llama_release(&self) -> AicutResult<ResolvedDownload> {
        let resp = self
            .http
            .get("https://api.github.com/repos/ggml-org/llama.cpp/releases?per_page=10")
            .send()
            .await
            .map_err(|e| {
                AicutError::internal(format!("Failed querying llama.cpp releases: {e}"))
            })?;

        let releases: Vec<serde_json::Value> = resp
            .json()
            .await
            .map_err(|e| AicutError::internal(format!("Invalid llama.cpp releases JSON: {e}")))?;

        let tag = self.hardware.platform_tag.as_str();
        let accel = self.hardware.recommended_accel;

        for rel in &releases {
            let Some(assets) = rel.get("assets").and_then(|v| v.as_array()) else {
                continue;
            };
            if assets.is_empty() {
                continue;
            }
            let version = rel
                .get("tag_name")
                .and_then(|v| v.as_str())
                .unwrap_or("latest")
                .to_string();

            if let Some((name, url, companion_url, kind)) = select_best_llama_assets(
                assets,
                tag,
                accel,
                self.hardware.cuda_major_minor(),
            ) {
                return Ok(ResolvedDownload {
                    url,
                    companion_url,
                    version,
                    build_label: name,
                    archive_kind: kind,
                });
            }
        }

        Err(AicutError::new(
            "LLAMA_ASSET_NOT_FOUND",
            format!("No prebuilt llama.cpp binary found for platform '{tag}'."),
        ))
    }

    async fn download_file(
        &self,
        url: &str,
        dest: &Path,
        job_id: &str,
        cancel: &tokio_util::sync::CancellationToken,
    ) -> AicutResult<()> {
        let resp = self
            .http
            .get(url)
            .send()
            .await
            .map_err(|e| AicutError::internal(format!("Download failed: {e}")))?;

        if !resp.status().is_success() {
            return Err(AicutError::internal(format!(
                "Download failed with HTTP {}",
                resp.status()
            )));
        }

        let total = resp.content_length();
        let mut file = std::fs::File::create(dest)?;
        let mut stream = resp.bytes_stream();
        let mut downloaded: u64 = 0;
        let started = std::time::Instant::now();
        let mut last_emit = std::time::Instant::now();

        while let Some(chunk_res) = tokio::select! {
            _ = cancel.cancelled() => {
                let _ = std::fs::remove_file(dest);
                return Err(AicutError::cancelled("Tool installation cancelled"));
            }
            next = stream.next() => next,
        } {
            let chunk = chunk_res
                .map_err(|e| AicutError::internal(format!("Download stream error: {e}")))?;
            file.write_all(&chunk)?;
            downloaded += chunk.len() as u64;

            if last_emit.elapsed().as_millis() >= 200 {
                let elapsed = started.elapsed().as_secs_f64().max(0.001);
                let speed = downloaded as f64 / elapsed;
                let percent = total
                    .map(|t| (downloaded as f64 / t as f64 * 90.0).min(90.0))
                    .unwrap_or(45.0);
                let eta = total.map(|t| (t.saturating_sub(downloaded)) as f64 / speed.max(1.0));
                self.jobs.update_progress(
                    job_id,
                    JobProgress {
                        percent,
                        stage: "downloading".to_string(),
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
        Ok(())
    }
}

/// Extracts `(major, minor)` CUDA version from a release asset filename (e.g.
/// `whisper-bin-win-cuda-13.4.0-x64.zip` -> `(13, 4)`, `whisper-cublas-12.4.0-bin-x64.zip` -> `(12, 4)`).
pub fn parse_cuda_version_from_asset(filename: &str) -> Option<(u32, u32)> {
    let lower = filename.to_lowercase();
    for marker in ["cuda-", "cublas-"] {
        if let Some(idx) = lower.find(marker) {
            let rest = &lower[idx + marker.len()..];
            let ver_token = rest
                .split(|c: char| !c.is_ascii_digit() && c != '.')
                .next()?;
            let mut parts = ver_token.split('.');
            let major: u32 = parts.next()?.parse().ok()?;
            let minor: u32 = parts.next().and_then(|s| s.parse().ok()).unwrap_or(0);
            return Some((major, minor));
        }
    }
    None
}

/// Picks the optimal `whisper.cpp` binary asset from a GitHub release's asset list based on hardware
/// and the user's exact detected NVIDIA CUDA driver version (preferring the highest compatible CUDA <= driver_cuda).
pub fn select_best_whisper_asset(
    assets: &[serde_json::Value],
    platform_tag: &str,
    accel: AccelBackend,
    driver_cuda: Option<(u32, u32)>,
) -> Option<(String, String, ArchiveKind)> {
    let entries: Vec<(String, String)> = assets
        .iter()
        .filter_map(|a| {
            let name = a.get("name")?.as_str()?.to_string();
            let url = a.get("browser_download_url")?.as_str()?.to_string();
            Some((name, url))
        })
        .collect();

    let find_matching = |pred: &dyn Fn(&str) -> bool, kind: ArchiveKind| {
        entries
            .iter()
            .find(|(n, _)| pred(&n.to_lowercase()))
            .map(|(n, u)| (n.clone(), u.clone(), kind))
    };

    let pick_best_cuda = |arch_token: &str| -> Option<(String, String, ArchiveKind)> {
        let mut cuda_candidates: Vec<((u32, u32), &String, &String)> = entries
            .iter()
            .filter_map(|(n, u)| {
                let lower = n.to_lowercase();
                if lower.ends_with(".zip")
                    && lower.contains(arch_token)
                    && (lower.contains("cuda") || lower.contains("cublas"))
                {
                    let ver = parse_cuda_version_from_asset(&lower).unwrap_or((0, 0));
                    Some((ver, n, u))
                } else {
                    None
                }
            })
            .collect();

        if cuda_candidates.is_empty() {
            return None;
        }
        // Highest CUDA version first
        cuda_candidates.sort_by(|a, b| b.0.cmp(&a.0));

        if let Some(max_supported) = driver_cuda {
            if let Some((_, n, u)) = cuda_candidates
                .iter()
                .find(|(ver, _, _)| *ver == (0, 0) || *ver <= max_supported)
            {
                return Some(((*n).clone(), (*u).clone(), ArchiveKind::WhisperZip));
            }
            // Driver is older than all published CUDA builds -> fall back to CPU/OpenBLAS safely
            return None;
        }

        // Unknown driver version: pick the highest available CUDA build
        let (_, n, u) = cuda_candidates[0];
        Some((n.clone(), u.clone(), ArchiveKind::WhisperZip))
    };

    match platform_tag {
        "win-x64" => {
            if accel == AccelBackend::CudaNvenc {
                if let Some(m) = pick_best_cuda("x64") {
                    return Some(m);
                }
            }
            if matches!(accel, AccelBackend::AmfDirectml | AccelBackend::QsvOpenvino) {
                if let Some(m) = find_matching(
                    &|n| n.ends_with(".zip") && n.contains("x64") && n.contains("vulkan"),
                    ArchiveKind::WhisperZip,
                ) {
                    return Some(m);
                }
            }
            find_matching(
                &|n| n == "whisper-blas-bin-x64.zip",
                ArchiveKind::WhisperZip,
            )
            .or_else(|| find_matching(&|n| n == "whisper-bin-x64.zip", ArchiveKind::WhisperZip))
        }
        "win-arm64" => {
            if accel == AccelBackend::CudaNvenc {
                if let Some(m) = pick_best_cuda("arm64") {
                    return Some(m);
                }
            }
            find_matching(
                &|n| {
                    n.ends_with(".zip")
                        && n.contains("win")
                        && n.contains("arm64")
                        && n.contains("cpu")
                },
                ArchiveKind::WhisperZip,
            )
        }
        "linux-x64" => find_matching(
            &|n| n.ends_with(".tar.gz") && n.contains("ubuntu") && n.contains("x64"),
            ArchiveKind::WhisperTarGz,
        ),
        "linux-arm64" => find_matching(
            &|n| n.ends_with(".tar.gz") && n.contains("ubuntu") && n.contains("arm64"),
            ArchiveKind::WhisperTarGz,
        ),
        _ => None,
    }
}

/// Picks the optimal `llama.cpp` binary asset (plus optional matching `cudart` companion zip on
/// Windows CUDA) from a `ggml-org/llama.cpp` release's asset list based on hardware & CUDA driver.
pub fn select_best_llama_assets(
    assets: &[serde_json::Value],
    platform_tag: &str,
    accel: AccelBackend,
    driver_cuda: Option<(u32, u32)>,
) -> Option<(String, String, Option<String>, ArchiveKind)> {
    let entries: Vec<(String, String)> = assets
        .iter()
        .filter_map(|a| {
            let name = a.get("name")?.as_str()?.to_string();
            let url = a.get("browser_download_url")?.as_str()?.to_string();
            Some((name, url))
        })
        .collect();

    let find_matching = |pred: &dyn Fn(&str) -> bool, kind: ArchiveKind| {
        entries
            .iter()
            .find(|(n, _)| pred(&n.to_lowercase()))
            .map(|(n, u)| (n.clone(), u.clone(), None, kind))
    };

    let pick_best_win_cuda = |arch_token: &str| -> Option<(String, String, Option<String>, ArchiveKind)> {
        let mut cuda_candidates: Vec<((u32, u32), &String, &String)> = entries
            .iter()
            .filter_map(|(n, u)| {
                let lower = n.to_lowercase();
                if lower.ends_with(".zip")
                    && lower.starts_with("llama-")
                    && lower.contains("bin-win")
                    && lower.contains("cuda")
                    && lower.contains(arch_token)
                {
                    let ver = parse_cuda_version_from_asset(&lower).unwrap_or((0, 0));
                    Some((ver, n, u))
                } else {
                    None
                }
            })
            .collect();

        if cuda_candidates.is_empty() {
            return None;
        }
        cuda_candidates.sort_by(|a, b| b.0.cmp(&a.0));

        let chosen = if let Some(max_supported) = driver_cuda {
            cuda_candidates
                .iter()
                .find(|(ver, _, _)| *ver == (0, 0) || *ver <= max_supported)
                .copied()?
        } else {
            cuda_candidates[0]
        };

        let (chosen_ver, name, url) = chosen;
        let cudart_url = entries.iter().find_map(|(cn, cu)| {
            let cl = cn.to_lowercase();
            if cl.ends_with(".zip")
                && cl.starts_with("cudart-llama-")
                && cl.contains("win")
                && cl.contains(arch_token)
            {
                let cver = parse_cuda_version_from_asset(&cl).unwrap_or((0, 0));
                if cver == chosen_ver {
                    return Some(cu.clone());
                }
            }
            None
        });

        Some((name.clone(), url.clone(), cudart_url, ArchiveKind::LlamaZip))
    };

    match platform_tag {
        "win-x64" => {
            if accel == AccelBackend::CudaNvenc {
                if let Some(m) = pick_best_win_cuda("x64") {
                    return Some(m);
                }
            }
            if matches!(
                accel,
                AccelBackend::CudaNvenc | AccelBackend::AmfDirectml | AccelBackend::QsvOpenvino
            ) {
                if let Some(m) = find_matching(
                    &|n| {
                        n.ends_with(".zip")
                            && n.starts_with("llama-")
                            && n.contains("bin-win-vulkan-x64")
                    },
                    ArchiveKind::LlamaZip,
                ) {
                    return Some(m);
                }
            }
            find_matching(
                &|n| {
                    n.ends_with(".zip")
                        && n.starts_with("llama-")
                        && n.contains("bin-win-cpu-x64")
                },
                ArchiveKind::LlamaZip,
            )
            .or_else(|| {
                find_matching(
                    &|n| {
                        n.ends_with(".zip")
                            && n.starts_with("llama-")
                            && n.contains("bin-win")
                            && n.contains("x64")
                            && (n.contains("avx2") || n.contains("openblas"))
                    },
                    ArchiveKind::LlamaZip,
                )
            })
        }
        "win-arm64" => find_matching(
            &|n| {
                n.ends_with(".zip")
                    && n.starts_with("llama-")
                    && n.contains("bin-win")
                    && n.contains("arm64")
            },
            ArchiveKind::LlamaZip,
        ),
        "macos-arm64" => find_matching(
            &|n| {
                (n.ends_with(".zip") || n.ends_with(".tar.gz"))
                    && n.starts_with("llama-")
                    && n.contains("bin-macos-arm64")
            },
            ArchiveKind::LlamaZip,
        )
        .map(|(n, u, c, _)| {
            let kind = if n.ends_with(".tar.gz") {
                ArchiveKind::LlamaTarGz
            } else {
                ArchiveKind::LlamaZip
            };
            (n, u, c, kind)
        }),
        "linux-x64" => find_matching(
            &|n| {
                (n.ends_with(".tar.gz") || n.ends_with(".zip"))
                    && n.starts_with("llama-")
                    && n.contains("bin-ubuntu-x64")
            },
            ArchiveKind::LlamaTarGz,
        )
        .map(|(n, u, c, _)| {
            let kind = if n.ends_with(".zip") {
                ArchiveKind::LlamaZip
            } else {
                ArchiveKind::LlamaTarGz
            };
            (n, u, c, kind)
        }),
        _ => None,
    }
}

pub fn platform_binary_name(base: &str) -> String {
    if cfg!(windows) && !base.ends_with(".exe") {
        format!("{base}.exe")
    } else {
        base.to_string()
    }
}

async fn probe_binary_version(path: &Path, version_args: &[String]) -> Option<String> {
    let args: Vec<&str> = if version_args.is_empty() {
        vec!["--version"]
    } else {
        version_args.iter().map(|s| s.as_str()).collect()
    };
    let mut cmd = Command::new(path);
    configure_tokio_command(&mut cmd);
    let output = cmd.args(args).output().await.ok()?;
    let stdout = String::from_utf8_lossy(&output.stdout);
    let first_line = stdout.lines().next()?.trim();
    if first_line.is_empty() {
        return None;
    }
    if let Some(rest) = first_line.strip_prefix("ffmpeg version ") {
        return rest.split_whitespace().next().map(|s| s.to_string());
    }
    if let Some(rest) = first_line.strip_prefix("ffprobe version ") {
        return rest.split_whitespace().next().map(|s| s.to_string());
    }
    Some(first_line.to_string())
}

fn versions_match(installed: &str, latest: &str) -> bool {
    let a = installed.trim().trim_start_matches('v');
    let b = latest.trim().trim_start_matches('v');
    a == b || a.contains(b) || b.contains(a)
}

fn replace_file(src: &Path, dst: &Path) -> AicutResult<()> {
    if dst.exists() {
        let _ = std::fs::remove_file(dst);
    }
    if std::fs::rename(src, dst).is_err() {
        std::fs::copy(src, dst)?;
        let _ = std::fs::remove_file(src);
    }
    Ok(())
}

fn make_executable(path: &Path) -> AicutResult<()> {
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let mut perms = std::fs::metadata(path)?.permissions();
        perms.set_mode(0o755);
        std::fs::set_permissions(path, perms)?;
    }
    #[cfg(not(unix))]
    let _ = path;
    Ok(())
}

fn extract_ffmpeg_from_zip(zip_path: &Path, bin_dir: &Path) -> AicutResult<()> {
    let file = std::fs::File::open(zip_path)?;
    let mut archive = zip::ZipArchive::new(file)
        .map_err(|e| AicutError::internal(format!("Failed reading FFmpeg zip archive: {e}")))?;

    for i in 0..archive.len() {
        let mut entry = archive
            .by_index(i)
            .map_err(|e| AicutError::internal(format!("Zip entry error: {e}")))?;
        if entry.is_dir() {
            continue;
        }
        let name = entry.name().replace('\\', "/");
        let file_name = name.rsplit('/').next().unwrap_or("");
        if matches!(
            file_name,
            "ffmpeg.exe" | "ffprobe.exe" | "ffmpeg" | "ffprobe"
        ) {
            let out_path = bin_dir.join(file_name);
            let tmp_out = bin_dir.join(format!(".{file_name}.new"));
            let mut out = std::fs::File::create(&tmp_out)?;
            std::io::copy(&mut entry, &mut out)?;
            out.flush()?;
            replace_file(&tmp_out, &out_path)?;
            make_executable(&out_path)?;
        }
    }
    Ok(())
}

fn extract_ffmpeg_from_targz(archive_path: &Path, bin_dir: &Path) -> AicutResult<()> {
    let file = std::fs::File::open(archive_path)?;
    let dec = flate2::read::GzDecoder::new(file);
    let mut archive = tar::Archive::new(dec);
    for entry in archive.entries()? {
        let mut entry = entry?;
        let path = entry.path()?.into_owned();
        let file_name = path
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("")
            .to_string();
        if matches!(file_name.as_str(), "ffmpeg" | "ffprobe") {
            let out_path = bin_dir.join(&file_name);
            let tmp_out = bin_dir.join(format!(".{file_name}.new"));
            entry.unpack(&tmp_out)?;
            replace_file(&tmp_out, &out_path)?;
            make_executable(&out_path)?;
        }
    }
    Ok(())
}

/// Extracts whisper-cli and all companion DLLs/shared libraries into `whisper_dir`.
fn extract_whisper_from_zip(zip_path: &Path, whisper_dir: &Path) -> AicutResult<()> {
    std::fs::create_dir_all(whisper_dir)?;
    let file = std::fs::File::open(zip_path)?;
    let mut archive = zip::ZipArchive::new(file)
        .map_err(|e| AicutError::internal(format!("Failed reading whisper.cpp zip: {e}")))?;

    for i in 0..archive.len() {
        let mut entry = archive
            .by_index(i)
            .map_err(|e| AicutError::internal(format!("Zip entry error: {e}")))?;
        if entry.is_dir() {
            continue;
        }
        let name = entry.name().replace('\\', "/");
        let file_name = name.rsplit('/').next().unwrap_or("");
        if file_name.is_empty() {
            continue;
        }
        let lower = file_name.to_lowercase();
        let keep = lower.ends_with(".exe")
            || lower.ends_with(".dll")
            || lower.ends_with(".so")
            || lower.contains(".so.")
            || file_name == "whisper-cli";
        if keep {
            let out_path = whisper_dir.join(file_name);
            let tmp_out = whisper_dir.join(format!(".{file_name}.new"));
            let mut out = std::fs::File::create(&tmp_out)?;
            std::io::copy(&mut entry, &mut out)?;
            out.flush()?;
            replace_file(&tmp_out, &out_path)?;
            make_executable(&out_path)?;
        }
    }
    Ok(())
}

fn extract_whisper_from_targz(archive_path: &Path, whisper_dir: &Path) -> AicutResult<()> {
    std::fs::create_dir_all(whisper_dir)?;
    let file = std::fs::File::open(archive_path)?;
    let dec = flate2::read::GzDecoder::new(file);
    let mut archive = tar::Archive::new(dec);
    for entry in archive.entries()? {
        let mut entry = entry?;
        if !entry.header().entry_type().is_file() {
            continue;
        }
        let path = entry.path()?.into_owned();
        let file_name = path
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("")
            .to_string();
        if file_name.is_empty() {
            continue;
        }
        let keep = file_name == "whisper-cli"
            || file_name == "whisper-server"
            || file_name.contains(".so");
        if keep {
            let out_path = whisper_dir.join(&file_name);
            let tmp_out = whisper_dir.join(format!(".{file_name}.new"));
            entry.unpack(&tmp_out)?;
            replace_file(&tmp_out, &out_path)?;
            make_executable(&out_path)?;
        }
    }
    Ok(())
}

/// Extracts `llama-server` and all companion DLLs/shared libraries into `llama_dir`.
fn extract_llama_from_zip(zip_path: &Path, llama_dir: &Path) -> AicutResult<()> {
    std::fs::create_dir_all(llama_dir)?;
    let file = std::fs::File::open(zip_path)?;
    let mut archive = zip::ZipArchive::new(file)
        .map_err(|e| AicutError::internal(format!("Failed reading llama.cpp zip: {e}")))?;

    for i in 0..archive.len() {
        let mut entry = archive
            .by_index(i)
            .map_err(|e| AicutError::internal(format!("Zip entry error: {e}")))?;
        if entry.is_dir() {
            continue;
        }
        let name = entry.name().replace('\\', "/");
        let file_name = name.rsplit('/').next().unwrap_or("");
        if file_name.is_empty() {
            continue;
        }
        let lower = file_name.to_lowercase();
        let keep = lower == "llama-server.exe"
            || lower == "llama-server"
            || lower == "llama-mtmd-cli.exe"
            || lower == "llama-mtmd-cli"
            || lower.ends_with(".dll")
            || lower.ends_with(".dylib")
            || lower.ends_with(".so")
            || lower.contains(".so.");
        if keep {
            let out_path = llama_dir.join(file_name);
            let tmp_out = llama_dir.join(format!(".{file_name}.new"));
            let mut out = std::fs::File::create(&tmp_out)?;
            std::io::copy(&mut entry, &mut out)?;
            out.flush()?;
            replace_file(&tmp_out, &out_path)?;
            make_executable(&out_path)?;
        }
    }
    Ok(())
}

fn extract_llama_from_targz(archive_path: &Path, llama_dir: &Path) -> AicutResult<()> {
    std::fs::create_dir_all(llama_dir)?;
    let file = std::fs::File::open(archive_path)?;
    let dec = flate2::read::GzDecoder::new(file);
    let mut archive = tar::Archive::new(dec);
    for entry in archive.entries()? {
        let mut entry = entry?;
        if !entry.header().entry_type().is_file() {
            continue;
        }
        let path = entry.path()?.into_owned();
        let file_name = path
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("")
            .to_string();
        if file_name.is_empty() {
            continue;
        }
        let keep = file_name == "llama-server"
            || file_name == "llama-mtmd-cli"
            || file_name.contains(".so")
            || file_name.ends_with(".dylib");
        if keep {
            let out_path = llama_dir.join(&file_name);
            let tmp_out = llama_dir.join(format!(".{file_name}.new"));
            entry.unpack(&tmp_out)?;
            replace_file(&tmp_out, &out_path)?;
            make_executable(&out_path)?;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn selects_best_whisper_asset_by_hardware_and_cuda_version() {
        let assets = vec![
            serde_json::json!({"name": "whisper-bin-x64.zip", "browser_download_url": "https://x/cpu.zip"}),
            serde_json::json!({"name": "whisper-blas-bin-x64.zip", "browser_download_url": "https://x/blas.zip"}),
            serde_json::json!({"name": "whisper-bin-win-cuda-11.8.0-x64.zip", "browser_download_url": "https://x/cuda11.zip"}),
            serde_json::json!({"name": "whisper-bin-win-cuda-12.4.0-x64.zip", "browser_download_url": "https://x/cuda12.zip"}),
            serde_json::json!({"name": "whisper-bin-win-cuda-13.4.0-x64.zip", "browser_download_url": "https://x/cuda13.zip"}),
            serde_json::json!({"name": "whisper-bin-ubuntu-x64.tar.gz", "browser_download_url": "https://x/ubuntu.tar.gz"}),
        ];

        // CUDA 13.5 driver picks CUDA 13.4 build
        let nv13 = select_best_whisper_asset(&assets, "win-x64", AccelBackend::CudaNvenc, Some((13, 5))).unwrap();
        assert_eq!(nv13.0, "whisper-bin-win-cuda-13.4.0-x64.zip");

        // CUDA 13.2 driver picks CUDA 12.4 build (never picks incompatible 13.4!)
        let nv132 = select_best_whisper_asset(&assets, "win-x64", AccelBackend::CudaNvenc, Some((13, 2))).unwrap();
        assert_eq!(nv132.0, "whisper-bin-win-cuda-12.4.0-x64.zip");

        // CUDA 12.0 driver picks CUDA 11.8 build
        let nv120 = select_best_whisper_asset(&assets, "win-x64", AccelBackend::CudaNvenc, Some((12, 0))).unwrap();
        assert_eq!(nv120.0, "whisper-bin-win-cuda-11.8.0-x64.zip");

        let cpu = select_best_whisper_asset(&assets, "win-x64", AccelBackend::CpuOnly, None).unwrap();
        assert_eq!(cpu.0, "whisper-blas-bin-x64.zip");

        let lin = select_best_whisper_asset(&assets, "linux-x64", AccelBackend::CpuOnly, None).unwrap();
        assert_eq!(lin.0, "whisper-bin-ubuntu-x64.tar.gz");
    }

    #[test]
    fn selects_best_llama_assets_with_cudart_and_fallbacks() {
        let assets = vec![
            serde_json::json!({"name": "llama-b5500-bin-win-cpu-x64.zip", "browser_download_url": "https://x/cpu.zip"}),
            serde_json::json!({"name": "llama-b5500-bin-win-vulkan-x64.zip", "browser_download_url": "https://x/vulkan.zip"}),
            serde_json::json!({"name": "llama-b5500-bin-win-cuda-12.4-x64.zip", "browser_download_url": "https://x/cuda12.zip"}),
            serde_json::json!({"name": "cudart-llama-bin-win-cuda-12.4-x64.zip", "browser_download_url": "https://x/cudart12.zip"}),
            serde_json::json!({"name": "llama-b5500-bin-win-cuda-13.1-x64.zip", "browser_download_url": "https://x/cuda13.zip"}),
            serde_json::json!({"name": "cudart-llama-bin-win-cuda-13.1-x64.zip", "browser_download_url": "https://x/cudart13.zip"}),
            serde_json::json!({"name": "llama-b5500-bin-ubuntu-x64.zip", "browser_download_url": "https://x/ubuntu.zip"}),
        ];

        // NVIDIA CUDA 13.5 driver picks CUDA 13.1 + matching cudart 13.1
        let nv13 = select_best_llama_assets(&assets, "win-x64", AccelBackend::CudaNvenc, Some((13, 5))).unwrap();
        assert_eq!(nv13.0, "llama-b5500-bin-win-cuda-13.1-x64.zip");
        assert_eq!(nv13.2.as_deref(), Some("https://x/cudart13.zip"));

        // NVIDIA CUDA 12.6 driver picks CUDA 12.4 + matching cudart 12.4
        let nv12 = select_best_llama_assets(&assets, "win-x64", AccelBackend::CudaNvenc, Some((12, 6))).unwrap();
        assert_eq!(nv12.0, "llama-b5500-bin-win-cuda-12.4-x64.zip");
        assert_eq!(nv12.2.as_deref(), Some("https://x/cudart12.zip"));

        // Old NVIDIA driver (CUDA 11.2) safely falls back to Vulkan
        let old_nv = select_best_llama_assets(&assets, "win-x64", AccelBackend::CudaNvenc, Some((11, 2))).unwrap();
        assert_eq!(old_nv.0, "llama-b5500-bin-win-vulkan-x64.zip");
        assert!(old_nv.2.is_none());

        // AMD picks Vulkan
        let amd = select_best_llama_assets(&assets, "win-x64", AccelBackend::AmfDirectml, None).unwrap();
        assert_eq!(amd.0, "llama-b5500-bin-win-vulkan-x64.zip");

        // CPU picks win-cpu-x64
        let cpu = select_best_llama_assets(&assets, "win-x64", AccelBackend::CpuOnly, None).unwrap();
        assert_eq!(cpu.0, "llama-b5500-bin-win-cpu-x64.zip");

        // Linux picks ubuntu-x64
        let lin = select_best_llama_assets(&assets, "linux-x64", AccelBackend::CpuOnly, None).unwrap();
        assert_eq!(lin.0, "llama-b5500-bin-ubuntu-x64.zip");
    }
}
