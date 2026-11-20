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
use crate::process::{configure_linux_gpu_env_tokio, configure_std_command};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum ToolInstallStrategy {
    /// Downloads the official yt-dlp standalone binary matching host Linux CPU arch.
    YtDlpOfficialRelease,
    /// Downloads the optimal Linux GPL FFmpeg/FFprobe build (with NVIDIA NVENC/NVDEC & VAAPI).
    FfmpegHardwareBundle,
    /// Downloads the official ggml-org/whisper.cpp Linux release matching GPU/CPU hardware.
    WhisperCppOfficialRelease,
    /// Downloads the official ggml-org/llama.cpp Linux release matching GPU/CPU hardware (CUDA / Vulkan / CPU).
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
    FfmpegTarXz,
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
                    "linux-arm64" => "yt-dlp_linux_aarch64 (Linux ARM64 Standalone)",
                    _ => "yt-dlp_linux (Linux x64 Standalone)",
                };
                (
                    build.to_string(),
                    format!("Selecionado automaticamente para Linux ({tag})."),
                )
            }
            ToolInstallStrategy::FfmpegHardwareBundle | ToolInstallStrategy::CompanionOf(_) => {
                let build = match tag {
                    "linux-arm64" => "FFmpeg Linux ARM64 GPL (BtbN Static)",
                    _ => "FFmpeg Linux x64 GPL (NVENC + CUDA + VAAPI)",
                };
                let hw_note = match accel {
                    AccelBackend::CudaNvenc => {
                        "GPU NVIDIA detectada: suporte NVENC/NVDEC acelerado por hardware (também aproveita `pacman -S ffmpeg` se já instalado no sistema)."
                    }
                    AccelBackend::AmfDirectml => {
                        "GPU AMD detectada: build Linux GPL otimizada com VAAPI/libx264."
                    }
                    AccelBackend::QsvOpenvino => {
                        "GPU Intel detectada: build Linux GPL otimizada com QuickSync/VAAPI."
                    }
                    AccelBackend::Metal | AccelBackend::CpuOnly => {
                        "Modo CPU otimizado: build Linux multithread para todos os núcleos lógicos."
                    }
                };
                (build.to_string(), hw_note.to_string())
            }
            ToolInstallStrategy::WhisperCppOfficialRelease => match (tag, accel) {
                ("linux-x64", AccelBackend::CudaNvenc) => {
                    let cuda_label = self
                        .hardware
                        .cuda_version
                        .as_deref()
                        .map(|v| format!("CUDA {v}"))
                        .unwrap_or_else(|| "NVIDIA".to_string());
                    let note = if self.hardware.cuda_runtime_installed {
                        format!(
                            "GPU NVIDIA ({cuda_label}) + CUDA Runtime detectados: prioriza build Linux CUDA/Vulkan (ou `whisper-cli` nativo do Arch/CachyOS no PATH)."
                        )
                    } else {
                        format!(
                            "GPU NVIDIA ({cuda_label}) detectada: usa Vulkan/Linux oficial ou `whisper.cpp-cuda` nativo do Arch/CachyOS se instalado."
                        )
                    };
                    (format!("whisper.cpp Linux x64 ({cuda_label})"), note)
                }
                ("linux-arm64", _) => (
                    "whisper.cpp Linux ARM64 Official".to_string(),
                    "Build oficial pré-compilada Linux ARM64.".to_string(),
                ),
                _ => (
                    "whisper.cpp Linux x64 Official".to_string(),
                    "Build oficial pré-compilada Linux x64.".to_string(),
                ),
            },
            ToolInstallStrategy::LlamaCppOfficialRelease => match (tag, accel) {
                ("linux-x64", AccelBackend::CudaNvenc) => {
                    let cuda_label = self
                        .hardware
                        .cuda_version
                        .as_deref()
                        .map(|v| format!("CUDA {v}"))
                        .unwrap_or_else(|| "NVIDIA".to_string());
                    if self.hardware.cuda_runtime_installed {
                        (
                            format!("llama.cpp Linux x64 {cuda_label} (GPU NVIDIA)"),
                            format!(
                                "GPU NVIDIA ({cuda_label}) + CUDA Runtime detectados: prioriza build oficial CUDA (≤ {cuda_label}) com fallback automático para Vulkan."
                            ),
                        )
                    } else {
                        (
                            format!("llama.cpp Linux x64 Vulkan (GPU NVIDIA {cuda_label})"),
                            "GPU NVIDIA detectada no Linux: usa aceleração Vulkan na GPU sem precisar do toolkit CUDA completo (instale `sudo pacman -S cuda` para habilitar backend CUDA nativo).".to_string(),
                        )
                    }
                }
                ("linux-x64", AccelBackend::AmfDirectml | AccelBackend::QsvOpenvino) => (
                    "llama.cpp Linux x64 Vulkan (GPU)".to_string(),
                    "GPU dedicada/integrada detectada: seleciona build oficial Linux Vulkan para acelerar modelos visuais GGUF.".to_string(),
                ),
                ("linux-arm64", _) => (
                    "llama.cpp Linux ARM64 Official".to_string(),
                    "Build oficial pré-compilada Linux ARM64.".to_string(),
                ),
                _ => (
                    "llama.cpp Linux x64 CPU".to_string(),
                    "Build oficial Linux x64 otimizada para CPU.".to_string(),
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
                ArchiveKind::FfmpegTarXz => {
                    extract_ffmpeg_from_tarxz(&tmp_path, &self.bin_dir)?;
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
            "linux-arm64" => "yt-dlp_linux_aarch64",
            _ => "yt-dlp_linux",
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
            "linux-arm64" => Ok(ResolvedDownload {
                url: "https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-master-latest-linuxarm64-gpl.tar.xz".to_string(),
                companion_url: None,
                version: "latest-linuxarm64-gpl".to_string(),
                build_label: "ffmpeg-master-latest-linuxarm64-gpl.tar.xz".to_string(),
                archive_kind: ArchiveKind::FfmpegTarXz,
            }),
            _ => Ok(ResolvedDownload {
                url: "https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-master-latest-linux64-gpl.tar.xz".to_string(),
                companion_url: None,
                version: "latest-linux64-gpl".to_string(),
                build_label: "ffmpeg-master-latest-linux64-gpl.tar.xz (NVENC + CUDA + VAAPI)".to_string(),
                archive_kind: ArchiveKind::FfmpegTarXz,
            }),
        }
    }

    /// Queries the latest releases of `ggml-org/whisper.cpp` (skipping source-only tags)
    /// and picks the optimal Linux binary archive for the host CPU/GPU profile.
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
                self.hardware.cuda_runtime_installed,
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
            format!("No prebuilt whisper.cpp Linux binary found for platform '{tag}'."),
        ))
    }

    /// Queries the latest releases of `ggml-org/llama.cpp` and picks the highest-quality
    /// Linux GPU/CPU binary archive (CUDA when runtime is present, Vulkan for out-of-the-box NVIDIA/AMD/Intel GPU, or CPU fallback).
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
                self.hardware.cuda_runtime_installed,
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
            format!("No prebuilt llama.cpp Linux binary found for platform '{tag}'."),
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

/// Picks the optimal `whisper.cpp` Linux binary asset from a GitHub release's asset list based on
/// detected GPU hardware, CUDA driver version, and whether local CUDA runtime libraries exist.
pub fn select_best_whisper_asset(
    assets: &[serde_json::Value],
    platform_tag: &str,
    accel: AccelBackend,
    driver_cuda: Option<(u32, u32)>,
    has_cuda_runtime: bool,
) -> Option<(String, String, ArchiveKind)> {
    let entries: Vec<(String, String)> = assets
        .iter()
        .filter_map(|a| {
            let name = a.get("name")?.as_str()?.to_string();
            let url = a.get("browser_download_url")?.as_str()?.to_string();
            Some((name, url))
        })
        .collect();

    let kind_for = |name: &str| {
        if name.to_lowercase().ends_with(".zip") {
            ArchiveKind::WhisperZip
        } else {
            ArchiveKind::WhisperTarGz
        }
    };

    let find_matching = |pred: &dyn Fn(&str) -> bool| {
        entries
            .iter()
            .find(|(n, _)| pred(&n.to_lowercase()))
            .map(|(n, u)| (n.clone(), u.clone(), kind_for(n)))
    };

    let pick_best_linux_cuda = |arch_token: &str| -> Option<(String, String, ArchiveKind)> {
        let mut cuda_candidates: Vec<((u32, u32), &String, &String)> = entries
            .iter()
            .filter_map(|(n, u)| {
                let lower = n.to_lowercase();
                let is_archive = lower.ends_with(".tar.gz") || lower.ends_with(".zip");
                let is_linux = lower.contains("ubuntu") || lower.contains("linux");
                if is_archive
                    && is_linux
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
        cuda_candidates.sort_by(|a, b| b.0.cmp(&a.0));

        if let Some(max_supported) = driver_cuda {
            if let Some((_, n, u)) = cuda_candidates
                .iter()
                .find(|(ver, _, _)| *ver == (0, 0) || *ver <= max_supported)
            {
                return Some(((*n).clone(), (*u).clone(), kind_for(n)));
            }
            return None;
        }

        let (_, n, u) = cuda_candidates[0];
        Some((n.clone(), u.clone(), kind_for(n)))
    };

    match platform_tag {
        "linux-arm64" => find_matching(&|n| {
            (n.ends_with(".tar.gz") || n.ends_with(".zip"))
                && (n.contains("ubuntu") || n.contains("linux"))
                && n.contains("arm64")
        }),
        _ => {
            if accel == AccelBackend::CudaNvenc && has_cuda_runtime {
                if let Some(m) = pick_best_linux_cuda("x64") {
                    return Some(m);
                }
            }
            if matches!(
                accel,
                AccelBackend::CudaNvenc | AccelBackend::AmfDirectml | AccelBackend::QsvOpenvino
            ) {
                if let Some(m) = find_matching(&|n| {
                    (n.ends_with(".tar.gz") || n.ends_with(".zip"))
                        && (n.contains("ubuntu") || n.contains("linux"))
                        && n.contains("x64")
                        && n.contains("vulkan")
                }) {
                    return Some(m);
                }
            }
            find_matching(&|n| {
                (n.ends_with(".tar.gz") || n.ends_with(".zip"))
                    && (n.contains("ubuntu") || n.contains("linux"))
                    && n.contains("x64")
                    && !n.contains("cuda")
                    && !n.contains("cublas")
                    && !n.contains("vulkan")
                    && !n.contains("rocm")
            })
            .or_else(|| {
                find_matching(&|n| {
                    (n.ends_with(".tar.gz") || n.ends_with(".zip"))
                        && (n.contains("ubuntu") || n.contains("linux"))
                        && n.contains("x64")
                })
            })
        }
    }
}

/// Picks the optimal `llama.cpp` Linux binary asset from a `ggml-org/llama.cpp` release's
/// asset list based on detected GPU hardware, NVIDIA CUDA driver/runtime, and Vulkan support.
pub fn select_best_llama_assets(
    assets: &[serde_json::Value],
    platform_tag: &str,
    accel: AccelBackend,
    driver_cuda: Option<(u32, u32)>,
    has_cuda_runtime: bool,
) -> Option<(String, String, Option<String>, ArchiveKind)> {
    let entries: Vec<(String, String)> = assets
        .iter()
        .filter_map(|a| {
            let name = a.get("name")?.as_str()?.to_string();
            let url = a.get("browser_download_url")?.as_str()?.to_string();
            Some((name, url))
        })
        .collect();

    let kind_for = |name: &str| {
        if name.to_lowercase().ends_with(".zip") {
            ArchiveKind::LlamaZip
        } else {
            ArchiveKind::LlamaTarGz
        }
    };

    let find_matching = |pred: &dyn Fn(&str) -> bool| {
        entries
            .iter()
            .find(|(n, _)| pred(&n.to_lowercase()))
            .map(|(n, u)| (n.clone(), u.clone(), None, kind_for(n)))
    };

    let pick_best_linux_cuda = |arch_token: &str| -> Option<(String, String, Option<String>, ArchiveKind)> {
        let mut cuda_candidates: Vec<((u32, u32), &String, &String)> = entries
            .iter()
            .filter_map(|(n, u)| {
                let lower = n.to_lowercase();
                let is_archive = lower.ends_with(".tar.gz") || lower.ends_with(".zip");
                let is_linux = lower.contains("bin-ubuntu") || lower.contains("bin-linux");
                if is_archive
                    && lower.starts_with("llama-")
                    && is_linux
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

        let (_, name, url) = chosen;
        Some((name.clone(), url.clone(), None, kind_for(name)))
    };

    match platform_tag {
        "linux-arm64" => find_matching(&|n| {
            (n.ends_with(".tar.gz") || n.ends_with(".zip"))
                && n.starts_with("llama-")
                && (n.contains("bin-ubuntu") || n.contains("bin-linux"))
                && n.contains("arm64")
        }),
        _ => {
            // 1. NVIDIA + installed Linux CUDA runtime (`/opt/cuda` or `/usr/lib/libcudart.so`): highest compatible CUDA build.
            if accel == AccelBackend::CudaNvenc && has_cuda_runtime {
                if let Some(m) = pick_best_linux_cuda("x64") {
                    return Some(m);
                }
            }
            // 2. Any NVIDIA / AMD / Intel GPU on Linux: Vulkan build runs full GPU acceleration out-of-the-box with `nvidia-utils`/Mesa.
            if matches!(
                accel,
                AccelBackend::CudaNvenc | AccelBackend::AmfDirectml | AccelBackend::QsvOpenvino
            ) {
                if let Some(m) = find_matching(&|n| {
                    (n.ends_with(".tar.gz") || n.ends_with(".zip"))
                        && n.starts_with("llama-")
                        && (n.contains("bin-ubuntu-vulkan-x64") || n.contains("bin-linux-vulkan-x64"))
                }) {
                    return Some(m);
                }
            }
            // 3. Clean CPU fallback (explicitly excluding GPU/SYCL/ROCm variants).
            find_matching(&|n| {
                (n.ends_with(".tar.gz") || n.ends_with(".zip"))
                    && n.starts_with("llama-")
                    && (n.contains("bin-ubuntu-x64") || n.contains("bin-linux-x64"))
                    && !n.contains("cuda")
                    && !n.contains("vulkan")
                    && !n.contains("rocm")
                    && !n.contains("sycl")
                    && !n.contains("opencl")
            })
        }
    }
}

pub fn platform_binary_name(base: &str) -> String {
    base.to_string()
}

async fn probe_binary_version(path: &Path, version_args: &[String]) -> Option<String> {
    let args: Vec<&str> = if version_args.is_empty() {
        vec!["--version"]
    } else {
        version_args.iter().map(|s| s.as_str()).collect()
    };
    let mut cmd = Command::new(path);
    configure_linux_gpu_env_tokio(&mut cmd, path);
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
    if dst.exists() || dst.symlink_metadata().is_ok() {
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

fn find_file_recursive(dir: &Path, target_name: &str) -> Option<PathBuf> {
    let entries = std::fs::read_dir(dir).ok()?;
    for entry in entries.flatten() {
        let p = entry.path();
        if p.is_dir() {
            if let Some(found) = find_file_recursive(&p, target_name) {
                return Some(found);
            }
        } else if p.file_name().and_then(|n| n.to_str()) == Some(target_name) {
            return Some(p);
        }
    }
    None
}

fn extract_ffmpeg_from_tarxz(archive_path: &Path, bin_dir: &Path) -> AicutResult<()> {
    let tmp_dir = bin_dir.join(".ffmpeg-extract-tmp");
    if tmp_dir.exists() {
        let _ = std::fs::remove_dir_all(&tmp_dir);
    }
    std::fs::create_dir_all(&tmp_dir)?;

    let mut cmd = std::process::Command::new("tar");
    configure_std_command(&mut cmd);
    let output = cmd
        .arg("-xJf")
        .arg(archive_path)
        .arg("-C")
        .arg(&tmp_dir)
        .output()
        .map_err(|e| AicutError::internal(format!("Failed running tar to extract FFmpeg: {e}")))?;

    if !output.status.success() {
        let _ = std::fs::remove_dir_all(&tmp_dir);
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(AicutError::internal(format!(
            "tar failed extracting FFmpeg archive: {stderr}"
        )));
    }

    let mut extracted_any = false;
    for name in ["ffmpeg", "ffprobe"] {
        if let Some(found) = find_file_recursive(&tmp_dir, name) {
            let out_path = bin_dir.join(name);
            replace_file(&found, &out_path)?;
            make_executable(&out_path)?;
            extracted_any = true;
        }
    }

    let _ = std::fs::remove_dir_all(&tmp_dir);
    if !extracted_any {
        return Err(AicutError::internal(
            "FFmpeg archive extracted cleanly but neither ffmpeg nor ffprobe was found inside.",
        ));
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

/// Extracts `whisper-cli` and all companion `.so` shared libraries into `whisper_dir`.
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
        let keep = lower.ends_with(".so")
            || lower.contains(".so.")
            || file_name == "whisper-cli"
            || file_name == "whisper-server";
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

fn create_flat_symlink(target_name: &str, out_path: &Path) -> AicutResult<()> {
    if out_path.exists() || out_path.symlink_metadata().is_ok() {
        let _ = std::fs::remove_file(out_path);
    }
    #[cfg(unix)]
    {
        std::os::unix::fs::symlink(target_name, out_path)?;
    }
    #[cfg(not(unix))]
    {
        let _ = (target_name, out_path);
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
        let entry_type = entry.header().entry_type();
        if !entry_type.is_file() && !entry_type.is_symlink() {
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
            if entry_type.is_symlink() {
                if let Ok(Some(link_target)) = entry.link_name() {
                    if let Some(target_name) = link_target.file_name().and_then(|n| n.to_str()) {
                        create_flat_symlink(target_name, &out_path)?;
                    }
                }
            } else {
                let tmp_out = whisper_dir.join(format!(".{file_name}.new"));
                entry.unpack(&tmp_out)?;
                replace_file(&tmp_out, &out_path)?;
                make_executable(&out_path)?;
            }
        }
    }
    Ok(())
}

/// Extracts `llama-server` and all companion `.so` shared libraries into `llama_dir`.
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
        let keep = lower == "llama-server"
            || lower == "llama-mtmd-cli"
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
        let entry_type = entry.header().entry_type();
        if !entry_type.is_file() && !entry_type.is_symlink() {
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
            || file_name.contains(".so");
        if keep {
            let out_path = llama_dir.join(&file_name);
            if entry_type.is_symlink() {
                if let Ok(Some(link_target)) = entry.link_name() {
                    if let Some(target_name) = link_target.file_name().and_then(|n| n.to_str()) {
                        create_flat_symlink(target_name, &out_path)?;
                    }
                }
            } else {
                let tmp_out = llama_dir.join(format!(".{file_name}.new"));
                entry.unpack(&tmp_out)?;
                replace_file(&tmp_out, &out_path)?;
                make_executable(&out_path)?;
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn selects_best_whisper_linux_asset_by_nvidia_cuda_vulkan_and_cpu() {
        let assets = vec![
            serde_json::json!({"name": "whisper-bin-ubuntu-x64.tar.gz", "browser_download_url": "https://x/cpu.tar.gz"}),
            serde_json::json!({"name": "whisper-bin-ubuntu-vulkan-x64.tar.gz", "browser_download_url": "https://x/vulkan.tar.gz"}),
            serde_json::json!({"name": "whisper-bin-ubuntu-cuda-12.4-x64.tar.gz", "browser_download_url": "https://x/cuda12.tar.gz"}),
            serde_json::json!({"name": "whisper-bin-ubuntu-cuda-13.2-x64.tar.gz", "browser_download_url": "https://x/cuda13.tar.gz"}),
            serde_json::json!({"name": "whisper-bin-ubuntu-arm64.tar.gz", "browser_download_url": "https://x/arm64.tar.gz"}),
        ];

        // NVIDIA with CUDA runtime + driver 13.5 picks CUDA 13.2
        let nv13 = select_best_whisper_asset(&assets, "linux-x64", AccelBackend::CudaNvenc, Some((13, 5)), true).unwrap();
        assert_eq!(nv13.0, "whisper-bin-ubuntu-cuda-13.2-x64.tar.gz");

        // NVIDIA with driver 12.8 picks CUDA 12.4 (never incompatible 13.2!)
        let nv12 = select_best_whisper_asset(&assets, "linux-x64", AccelBackend::CudaNvenc, Some((12, 8)), true).unwrap();
        assert_eq!(nv12.0, "whisper-bin-ubuntu-cuda-12.4-x64.tar.gz");

        // NVIDIA without CUDA toolkit falls back cleanly to Vulkan GPU build
        let nv_vk = select_best_whisper_asset(&assets, "linux-x64", AccelBackend::CudaNvenc, Some((13, 0)), false).unwrap();
        assert_eq!(nv_vk.0, "whisper-bin-ubuntu-vulkan-x64.tar.gz");

        // CPU-only picks standard ubuntu-x64
        let cpu = select_best_whisper_asset(&assets, "linux-x64", AccelBackend::CpuOnly, None, false).unwrap();
        assert_eq!(cpu.0, "whisper-bin-ubuntu-x64.tar.gz");

        // ARM64 picks ubuntu-arm64
        let arm = select_best_whisper_asset(&assets, "linux-arm64", AccelBackend::CpuOnly, None, false).unwrap();
        assert_eq!(arm.0, "whisper-bin-ubuntu-arm64.tar.gz");
    }

    #[test]
    fn selects_best_llama_linux_assets_for_nvidia_cuda_vulkan_and_cpu() {
        let assets = vec![
            serde_json::json!({"name": "llama-b5500-bin-ubuntu-x64.tar.gz", "browser_download_url": "https://x/cpu.tar.gz"}),
            serde_json::json!({"name": "llama-b5500-bin-ubuntu-vulkan-x64.tar.gz", "browser_download_url": "https://x/vulkan.tar.gz"}),
            serde_json::json!({"name": "llama-b5500-bin-ubuntu-cuda-12.4-x64.tar.gz", "browser_download_url": "https://x/cuda12.tar.gz"}),
            serde_json::json!({"name": "llama-b5500-bin-ubuntu-cuda-13.1-x64.tar.gz", "browser_download_url": "https://x/cuda13.tar.gz"}),
        ];

        // NVIDIA CUDA 13.5 + CUDA runtime picks CUDA 13.1
        let nv13 = select_best_llama_assets(&assets, "linux-x64", AccelBackend::CudaNvenc, Some((13, 5)), true).unwrap();
        assert_eq!(nv13.0, "llama-b5500-bin-ubuntu-cuda-13.1-x64.tar.gz");
        assert_eq!(nv13.3, ArchiveKind::LlamaTarGz);

        // NVIDIA CUDA 12.6 + CUDA runtime picks CUDA 12.4
        let nv12 = select_best_llama_assets(&assets, "linux-x64", AccelBackend::CudaNvenc, Some((12, 6)), true).unwrap();
        assert_eq!(nv12.0, "llama-b5500-bin-ubuntu-cuda-12.4-x64.tar.gz");

        // NVIDIA without CUDA runtime (typical fresh CachyOS with `nvidia-utils`) picks Vulkan GPU build immediately!
        let nv_vk = select_best_llama_assets(&assets, "linux-x64", AccelBackend::CudaNvenc, Some((13, 0)), false).unwrap();
        assert_eq!(nv_vk.0, "llama-b5500-bin-ubuntu-vulkan-x64.tar.gz");

        // AMD picks Vulkan
        let amd = select_best_llama_assets(&assets, "linux-x64", AccelBackend::AmfDirectml, None, false).unwrap();
        assert_eq!(amd.0, "llama-b5500-bin-ubuntu-vulkan-x64.tar.gz");

        // CPU picks ubuntu-x64
        let cpu = select_best_llama_assets(&assets, "linux-x64", AccelBackend::CpuOnly, None, false).unwrap();
        assert_eq!(cpu.0, "llama-b5500-bin-ubuntu-x64.tar.gz");
    }
}
