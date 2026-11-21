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

const RELEASE_CACHE_TTL: std::time::Duration = std::time::Duration::from_secs(600);
const DEFAULT_RATE_LIMIT_COOLDOWN: std::time::Duration = std::time::Duration::from_secs(900);

struct CachedReleases {
    fetched_at: std::time::Instant,
    releases: Vec<serde_json::Value>,
}

struct CachedTag {
    fetched_at: std::time::Instant,
    release: serde_json::Value,
}

pub struct ToolManager {
    mutation: tokio::sync::Mutex<()>,
    bin_dir: PathBuf,
    hardware: HardwareProfile,
    definitions: RwLock<HashMap<String, ToolDefinition>>,
    latest_cache: RwLock<HashMap<String, String>>,
    releases_cache: RwLock<HashMap<String, CachedReleases>>,
    tag_cache: RwLock<HashMap<(String, String), CachedTag>>,
    github_api_blocked_until: RwLock<Option<std::time::Instant>>,
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
            .user_agent(concat!(
                "Aicut/",
                env!("CARGO_PKG_VERSION"),
                " (+https://github.com/limitbreakingofc/Aicut)"
            ))
            .connect_timeout(std::time::Duration::from_secs(20))
            .build()
            .unwrap_or_else(|_| reqwest::Client::new());

        Self {
            mutation: Default::default(),
            bin_dir,
            hardware,
            definitions: RwLock::new(HashMap::new()),
            latest_cache: RwLock::new(HashMap::new()),
            releases_cache: RwLock::new(HashMap::new()),
            tag_cache: RwLock::new(HashMap::new()),
            github_api_blocked_until: RwLock::new(None),
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
        let bundle = match binary_name {
            "ffmpeg" | "ffprobe" => Some("ffmpeg-bundle"),
            "yt-dlp" => Some("yt-dlp-bundle"),
            _ => None,
        };
        if let Some(dir) = bundle {
            let path = self.bin_dir.join(dir).join(&managed_name);
            if path.is_file() { return Some((path, true)); }
        }
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
                        "GPU NVIDIA detectada: suporte NVENC/NVDEC acelerado por hardware com um build do FFmpeg que inclua esses encoders."
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
                            "GPU NVIDIA ({cuda_label}) + CUDA Runtime detectados: prioriza build Linux CUDA/Vulkan (ou `whisper-cli` já disponível no PATH)."
                        )
                    } else {
                        format!(
                            "GPU NVIDIA ({cuda_label}) detectada: usa Vulkan/Linux oficial ou `whisper-cli` com suporte a CUDA, se instalado."
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
                            "Prioriza CUDA com o runtime fornecido pela release quando compatível com o driver. Usa Vulkan como alternativa.".to_string(),
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
                let marker_ver = path.parent().filter(|p| managed && *p != self.bin_dir)
                    .and_then(|dir| std::fs::read_to_string(dir.join(".version")).ok());
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
        self.install_version(tool_id, None, actor).await
    }

    pub async fn uninstall(&self, tool_id: &str) -> AicutResult<ToolStatus> {
        let _guard = self.mutation.lock().await;
        let _directory = crate::workspace::lock_downloads(&self.bin_dir)?;
        let names: &[&str] = match tool_id {
            "whisper" => &["whisper", "whisper-cli"],
            "llama-server" => &["llama", "llama-server"],
            "yt-dlp" => &["yt-dlp-bundle", "yt-dlp"],
            "ffmpeg" | "ffprobe" => &["ffmpeg-bundle", "ffmpeg", "ffprobe", ".ffmpeg-extract-tmp", ".ffmpeg.new", ".ffprobe.new"],
            _ => return Err(AicutError::invalid_input("Ferramenta desconhecida")),
        };
        for name in names {
            let path = self.bin_dir.join(name);
            match std::fs::symlink_metadata(&path) {
                Ok(meta) if meta.is_dir() && matches!(*name, "whisper" | "llama" | "yt-dlp-bundle" | "ffmpeg-bundle" | ".ffmpeg-extract-tmp") => std::fs::remove_dir_all(path)?,
                Ok(_) => std::fs::remove_file(path)?,
                Err(e) if e.kind() == std::io::ErrorKind::NotFound => (),
                Err(e) => return Err(e.into()),
            }
        }
        self.status_of(tool_id).await
    }

    pub async fn install_version(&self, tool_id: &str, version: Option<&str>, actor: ActorKind) -> AicutResult<ToolStatus> {
        let _guard = self.mutation.lock().await;
        let _directory = crate::workspace::lock_downloads(&self.bin_dir)?;
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
            let resolved = if let Some(tag) = version { self.tagged_release(&target_def.id, tag).await? } else { match &target_def.strategy {
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
            }};

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

            let stage = crate::workspace::download_stage(&self.bin_dir)?;
            let tmp_path = stage.path().join("download");
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

            let payload = stage.path().join("payload");
            std::fs::create_dir(&payload)?;
            let folder = match resolved.archive_kind {
                ArchiveKind::SingleBinary => {
                    let binary = payload.join(&target_def.binary_name);
                    std::fs::rename(&tmp_path, &binary)?;
                    make_executable(&binary)?;
                    "yt-dlp-bundle"
                }
                ArchiveKind::FfmpegTarXz => { extract_ffmpeg_from_tarxz(&tmp_path, &payload)?; "ffmpeg-bundle" }
                ArchiveKind::FfmpegTarGz => { extract_ffmpeg_from_targz(&tmp_path, &payload)?; "ffmpeg-bundle" }
                ArchiveKind::WhisperZip => { extract_whisper_from_zip(&tmp_path, &payload)?; "whisper" }
                ArchiveKind::WhisperTarGz => { extract_whisper_from_targz(&tmp_path, &payload)?; "whisper" }
                ArchiveKind::LlamaZip => { extract_llama_from_zip(&tmp_path, &payload)?; "llama" }
                ArchiveKind::LlamaTarGz => { extract_llama_from_targz(&tmp_path, &payload)?; "llama" }
            };
            if let Some(url) = &resolved.companion_url {
                let runtime = stage.path().join("cudart.tar.gz");
                self.download_file(url, &runtime, &job.id, &cancel).await?;
                extract_llama_from_targz(&runtime, &payload)?;
            }
            if !payload.join(&target_def.binary_name).is_file() || (folder == "ffmpeg-bundle" && !payload.join("ffprobe").is_file()) {
                return Err(AicutError::invalid_input("O pacote está incompleto."));
            }
            std::fs::write(payload.join(".version"), &resolved.version)?;
            let dest = self.bin_dir.join(folder);
            crate::workspace::publish_directory(&payload, &dest)?;

            if let (None, Ok(mut cache)) = (version, self.latest_cache.write()) {
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

    fn release_repo(tool: &str) -> AicutResult<&'static str> {
        Ok(match tool {
            "ffmpeg" => "FFmpeg-Builds/FFmpeg-Builds",
            "yt-dlp" => "yt-dlp/yt-dlp",
            "whisper" => "ggml-org/whisper.cpp",
            "llama-server" => "ggml-org/llama.cpp",
            _ => return Err(AicutError::invalid_input("Ferramenta sem releases.")),
        })
    }

    fn github_api_is_blocked(&self) -> bool {
        if let Ok(guard) = self.github_api_blocked_until.read() {
            if let Some(until) = *guard {
                return std::time::Instant::now() < until;
            }
        }
        false
    }

    fn mark_github_api_rate_limited(&self, cooldown: std::time::Duration) {
        if let Ok(mut guard) = self.github_api_blocked_until.write() {
            *guard = Some(std::time::Instant::now() + cooldown);
        }
    }

    /// Calls a GitHub REST API endpoint with status verification, optional `GITHUB_TOKEN`/`GH_TOKEN`
    /// authentication, and automatic rate-limit cooldown detection so callers can seamlessly fall back
    /// to `github.com` web feeds when unauthenticated quota (60 req/h) is exhausted.
    async fn github_api_get_json(&self, url: &str) -> Result<serde_json::Value, String> {
        if self.github_api_is_blocked() {
            return Err("GitHub API rate limit ativo; usando fallback web.".to_string());
        }

        let mut req = self
            .http
            .get(url)
            .header(reqwest::header::ACCEPT, "application/vnd.github+json")
            .header("X-GitHub-Api-Version", "2022-11-28");

        if let Some(token) = std::env::var("GITHUB_TOKEN")
            .ok()
            .or_else(|| std::env::var("GH_TOKEN").ok())
            .filter(|t| !t.trim().is_empty())
        {
            req = req.bearer_auth(token.trim());
        }

        let resp = req
            .send()
            .await
            .map_err(|e| format!("Falha de rede ao consultar {url}: {e}"))?;

        let status = resp.status();
        let remaining = resp
            .headers()
            .get("x-ratelimit-remaining")
            .and_then(|v| v.to_str().ok())
            .map(str::to_owned);
        let reset_header = resp
            .headers()
            .get("x-ratelimit-reset")
            .and_then(|v| v.to_str().ok())
            .map(str::to_owned);

        if !status.is_success() {
            let body = resp.text().await.unwrap_or_default();
            let is_rate_limited = status == reqwest::StatusCode::TOO_MANY_REQUESTS
                || (status == reqwest::StatusCode::FORBIDDEN
                    && (remaining.as_deref() == Some("0")
                        || body.to_lowercase().contains("rate limit")));
            if is_rate_limited {
                let cooldown = parse_rate_limit_cooldown(reset_header.as_deref());
                self.mark_github_api_rate_limited(cooldown);
                return Err(format!(
                    "Limite da API do GitHub atingido (HTTP {})",
                    status.as_u16()
                ));
            }
            let snippet: String = body.chars().take(180).collect();
            return Err(format!(
                "GitHub API retornou HTTP {} para {url}: {snippet}",
                status.as_u16()
            ));
        }

        resp.json::<serde_json::Value>()
            .await
            .map_err(|e| format!("Resposta JSON inválida da API do GitHub ({url}): {e}"))
    }

    fn cached_releases(&self, repo: &str) -> Option<Vec<serde_json::Value>> {
        let guard = self.releases_cache.read().ok()?;
        let entry = guard.get(repo)?;
        if entry.fetched_at.elapsed() < RELEASE_CACHE_TTL {
            Some(entry.releases.clone())
        } else {
            None
        }
    }

    fn store_releases_cache(&self, repo: &str, releases: Vec<serde_json::Value>) {
        if let Ok(mut guard) = self.releases_cache.write() {
            guard.insert(
                repo.to_string(),
                CachedReleases {
                    fetched_at: std::time::Instant::now(),
                    releases,
                },
            );
        }
    }

    fn cached_tag_release(&self, repo: &str, tag: &str) -> Option<serde_json::Value> {
        if let Ok(guard) = self.tag_cache.read() {
            if let Some(entry) = guard.get(&(repo.to_string(), tag.to_string())) {
                if entry.fetched_at.elapsed() < RELEASE_CACHE_TTL {
                    return Some(entry.release.clone());
                }
            }
        }
        if let Some(releases) = self.cached_releases(repo) {
            if let Some(found) = releases
                .into_iter()
                .find(|r| r.get("tag_name").and_then(|v| v.as_str()) == Some(tag))
            {
                return Some(found);
            }
        }
        None
    }

    fn store_tag_cache(&self, repo: &str, tag: &str, release: serde_json::Value) {
        if let Ok(mut guard) = self.tag_cache.write() {
            guard.insert(
                (repo.to_string(), tag.to_string()),
                CachedTag {
                    fetched_at: std::time::Instant::now(),
                    release,
                },
            );
        }
    }

    /// Tries cached releases first, then GitHub REST API `/releases?per_page=...`.
    async fn fetch_api_releases(
        &self,
        repo: &str,
        per_page: usize,
    ) -> Result<Vec<serde_json::Value>, String> {
        if let Some(cached) = self.cached_releases(repo) {
            return Ok(cached);
        }
        let url = format!("https://api.github.com/repos/{repo}/releases?per_page={per_page}");
        let val = self.github_api_get_json(&url).await?;
        let list = val
            .as_array()
            .cloned()
            .ok_or_else(|| format!("Payload inesperado da API do GitHub para {repo} (esperado array)."))?;
        self.store_releases_cache(repo, list.clone());
        Ok(list)
    }

    /// Fetches recent release tags directly from `https://github.com/{repo}/releases.atom`
    /// (does NOT use `api.github.com` and is immune to the 60 req/h unauthenticated API limit).
    async fn fetch_web_release_tags(&self, repo: &str) -> AicutResult<Vec<String>> {
        let url = format!("https://github.com/{repo}/releases.atom");
        let resp = self
            .http
            .get(&url)
            .header(reqwest::header::ACCEPT, "application/atom+xml,application/xml,text/xml")
            .send()
            .await
            .map_err(|e| {
                AicutError::internal(format!(
                    "Falha ao consultar feed de releases de {repo}: {e}"
                ))
            })?;
        if !resp.status().is_success() {
            return Err(AicutError::internal(format!(
                "Feed de releases de {repo} retornou HTTP {}",
                resp.status().as_u16()
            )));
        }
        let xml = resp.text().await.map_err(|e| {
            AicutError::internal(format!("Falha ao ler feed de releases de {repo}: {e}"))
        })?;
        let tags = parse_atom_release_tags(&xml);
        if tags.is_empty() {
            return Err(AicutError::not_found(format!(
                "Nenhuma release encontrada para {repo}."
            )));
        }
        Ok(tags)
    }

    /// Fetches the asset list for a specific release tag via GitHub's web `expanded_assets`
    /// fragment (`https://github.com/{repo}/releases/expanded_assets/{tag}`), which bypasses
    /// `api.github.com` rate limits completely.
    async fn fetch_web_release_for_tag(
        &self,
        repo: &str,
        tag: &str,
    ) -> AicutResult<serde_json::Value> {
        if let Some(cached) = self.cached_tag_release(repo, tag) {
            return Ok(cached);
        }
        let url = format!("https://github.com/{repo}/releases/expanded_assets/{tag}");
        let resp = self
            .http
            .get(&url)
            .header(reqwest::header::ACCEPT, "text/html")
            .send()
            .await
            .map_err(|e| {
                AicutError::internal(format!(
                    "Falha ao consultar arquivos da release '{tag}' de {repo}: {e}"
                ))
            })?;
        if !resp.status().is_success() {
            return Err(AicutError::internal(format!(
                "Release '{tag}' de {repo} retornou HTTP {}",
                resp.status().as_u16()
            )));
        }
        let html = resp.text().await.map_err(|e| {
            AicutError::internal(format!(
                "Falha ao ler arquivos da release '{tag}' de {repo}: {e}"
            ))
        })?;
        let assets = parse_expanded_assets_html(&html);
        let release = serde_json::json!({
            "tag_name": tag,
            "assets": assets,
        });
        self.store_tag_cache(repo, tag, release.clone());
        Ok(release)
    }

    async fn fetch_release_for_tag(
        &self,
        repo: &str,
        tag: &str,
    ) -> AicutResult<serde_json::Value> {
        if let Some(cached) = self.cached_tag_release(repo, tag) {
            return Ok(cached);
        }
        let api_url = format!("https://api.github.com/repos/{repo}/releases/tags/{tag}");
        if let Ok(release) = self.github_api_get_json(&api_url).await {
            if release.get("assets").and_then(|v| v.as_array()).is_some() {
                self.store_tag_cache(repo, tag, release.clone());
                return Ok(release);
            }
        }
        self.fetch_web_release_for_tag(repo, tag).await
    }

    pub async fn versions(&self, tool: &str) -> AicutResult<Vec<String>> {
        let repo = Self::release_repo(tool)?;
        if let Ok(releases) = self.fetch_api_releases(repo, 30).await {
            let compatible: Vec<String> = releases
                .iter()
                .filter(|r| self.release_asset(tool, r).is_ok())
                .filter_map(|r| r["tag_name"].as_str().map(str::to_owned))
                .collect();
            if !compatible.is_empty() {
                return Ok(compatible);
            }
        }

        // Web fallback via releases.atom (no GitHub REST API rate limit).
        let tags = self.fetch_web_release_tags(repo).await?;
        if tool == "whisper" {
            let mut compatible = Vec::new();
            for tag in tags.into_iter().take(10) {
                if let Ok(rel) = self.fetch_web_release_for_tag(repo, &tag).await {
                    if self.release_asset(tool, &rel).is_ok() {
                        compatible.push(tag);
                    }
                }
            }
            return Ok(compatible);
        }
        Ok(tags)
    }

    async fn tagged_release(&self, tool: &str, tag: &str) -> AicutResult<ResolvedDownload> {
        if tag.is_empty()
            || tag.len() > 100
            || !tag
                .bytes()
                .all(|b| b.is_ascii_alphanumeric() || b".-_".contains(&b))
            || tag == "."
            || tag == ".."
        {
            return Err(AicutError::invalid_input("Informe uma tag de release válida."));
        }
        let repo = Self::release_repo(tool)?;
        let release = self.fetch_release_for_tag(repo, tag).await?;
        self.release_asset(tool, &release)
    }

    fn release_asset(&self, tool: &str, release: &serde_json::Value) -> AicutResult<ResolvedDownload> {
        let tag = release["tag_name"]
            .as_str()
            .ok_or_else(|| AicutError::invalid_input("Sem tag"))?;
        let assets = release["assets"]
            .as_array()
            .ok_or_else(|| AicutError::invalid_input("Release sem arquivos."))?;
        let hw = &self.hardware;
        let asset = match tool {
            "whisper" => select_best_whisper_asset(
                assets,
                &hw.platform_tag,
                hw.recommended_accel,
                hw.cuda_major_minor(),
                hw.cuda_runtime_installed,
            )
            .map(|(n, u, k)| (n, u, None, k)),
            "llama-server" => select_best_llama_assets(
                assets,
                &hw.platform_tag,
                hw.recommended_accel,
                hw.cuda_major_minor(),
                hw.cuda_runtime_installed,
            ),
            _ => {
                let ytdlp_name = match hw.platform_tag.as_str() {
                    "linux-arm64" => "yt-dlp_linux_aarch64",
                    _ => "yt-dlp_linux",
                };
                let ffmpeg_suffix = match hw.platform_tag.as_str() {
                    "linux-arm64" => "linuxarm64-gpl.tar.xz",
                    _ => "linux64-gpl.tar.xz",
                };
                assets.iter().find_map(|a| {
                    let name = a["name"].as_str()?;
                    let kind = match tool {
                        "yt-dlp" if name == ytdlp_name => ArchiveKind::SingleBinary,
                        "ffmpeg" if name.ends_with(ffmpeg_suffix) => ArchiveKind::FfmpegTarXz,
                        _ => return None,
                    };
                    Some((
                        name.to_owned(),
                        a["browser_download_url"].as_str()?.to_owned(),
                        None,
                        kind,
                    ))
                })
            }
        };
        let (build_label, url, companion_url, archive_kind) = asset.ok_or_else(|| {
            AicutError::invalid_input("Release sem binário compatível com este hardware.")
        })?;
        Ok(ResolvedDownload {
            build_label,
            url,
            companion_url,
            archive_kind,
            version: tag.to_owned(),
        })
    }

    async fn resolve_latest_from_repo(
        &self,
        tool: &str,
        per_page: usize,
        prefer_tag: Option<fn(&str) -> bool>,
    ) -> AicutResult<ResolvedDownload> {
        let repo = Self::release_repo(tool)?;
        let api_err = match self.fetch_api_releases(repo, per_page).await {
            Ok(releases) => {
                for rel in &releases {
                    let Some(tag) = rel.get("tag_name").and_then(|v| v.as_str()) else {
                        continue;
                    };
                    if let Some(pred) = prefer_tag {
                        if !pred(tag) {
                            continue;
                        }
                    }
                    if let Ok(resolved) = self.release_asset(tool, rel) {
                        return Ok(resolved);
                    }
                }
                None
            }
            Err(err) => Some(err),
        };

        // Fallback via github.com Atom feed + expanded_assets HTML (zero REST API rate limit).
        match self.fetch_web_release_tags(repo).await {
            Ok(tags) => {
                let candidate_tags: Vec<String> = match prefer_tag {
                    Some(pred) => tags.into_iter().filter(|t| pred(t)).take(10).collect(),
                    None => tags.into_iter().take(10).collect(),
                };
                for tag in candidate_tags {
                    if let Ok(rel) = self.fetch_web_release_for_tag(repo, &tag).await {
                        if let Ok(resolved) = self.release_asset(tool, &rel) {
                            return Ok(resolved);
                        }
                    }
                }
            }
            Err(web_err) => {
                if let Some(api_reason) = api_err {
                    return Err(AicutError::internal(format!(
                        "Não foi possível obter releases de {tool} ({api_reason}; fallback web: {}).",
                        web_err.message
                    )));
                }
                return Err(web_err);
            }
        }

        let tag = self.hardware.platform_tag.as_str();
        let code = match tool {
            "whisper" => "WHISPER_ASSET_NOT_FOUND",
            "llama-server" => "LLAMA_ASSET_NOT_FOUND",
            _ => "ASSET_NOT_FOUND",
        };
        Err(AicutError::new(
            code,
            format!("Nenhum binário Linux pré-compilado de '{tool}' encontrado para a plataforma '{tag}'."),
        ))
    }

    async fn resolve_ytdlp_release(&self) -> AicutResult<ResolvedDownload> {
        if let Ok(resolved) = self.resolve_latest_from_repo("yt-dlp", 10, None).await {
            return Ok(resolved);
        }

        let asset_name = match self.hardware.platform_tag.as_str() {
            "linux-arm64" => "yt-dlp_linux_aarch64",
            _ => "yt-dlp_linux",
        };
        Ok(ResolvedDownload {
            url: format!("https://github.com/yt-dlp/yt-dlp/releases/latest/download/{asset_name}"),
            companion_url: None,
            version: "latest".to_string(),
            build_label: asset_name.to_string(),
            archive_kind: ArchiveKind::SingleBinary,
        })
    }

    async fn resolve_ffmpeg_release(&self) -> AicutResult<ResolvedDownload> {
        if let Ok(resolved) = self
            .resolve_latest_from_repo("ffmpeg", 15, Some(|tag: &str| tag.starts_with("autobuild-")))
            .await
        {
            return Ok(resolved);
        }
        if let Ok(resolved) = self.resolve_latest_from_repo("ffmpeg", 15, None).await {
            return Ok(resolved);
        }

        let asset_name = match self.hardware.platform_tag.as_str() {
            "linux-arm64" => "ffmpeg-master-latest-linuxarm64-gpl.tar.xz",
            _ => "ffmpeg-master-latest-linux64-gpl.tar.xz",
        };
        Ok(ResolvedDownload {
            url: format!(
                "https://github.com/FFmpeg-Builds/FFmpeg-Builds/releases/download/latest/{asset_name}"
            ),
            companion_url: None,
            version: "latest".to_string(),
            build_label: asset_name.to_string(),
            archive_kind: ArchiveKind::FfmpegTarXz,
        })
    }

    /// Queries the latest releases of `ggml-org/whisper.cpp` (skipping source-only tags)
    /// and picks the optimal Linux binary archive for the host CPU/GPU profile.
    async fn resolve_whisper_release(&self) -> AicutResult<ResolvedDownload> {
        self.resolve_latest_from_repo("whisper", 15, None).await
    }

    /// Queries the latest releases of `ggml-org/llama.cpp` and picks the highest-quality
    /// Linux GPU/CPU binary archive (CUDA when runtime is present, Vulkan for out-of-the-box NVIDIA/AMD/Intel GPU, or CPU fallback).
    async fn resolve_llama_release(&self) -> AicutResult<ResolvedDownload> {
        self.resolve_latest_from_repo("llama-server", 10, None).await
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
                    && (lower.contains(arch_token) || lower.contains("x86_64") || lower.contains("amd64"))
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
                    && (lower.contains(arch_token) || lower.contains("x86_64") || lower.contains("amd64"))
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
        let companion = entries.iter().find(|(candidate, _)| candidate == &format!("cudart-{name}")).map(|(_, url)| url.clone());
        if !has_cuda_runtime && (driver_cuda.is_none() || companion.is_none()) { return None; }
        Some((name.clone(), url.clone(), companion, kind_for(name)))
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
            if accel == AccelBackend::CudaNvenc {
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

fn parse_rate_limit_cooldown(reset_header: Option<&str>) -> std::time::Duration {
    let Some(raw) = reset_header else {
        return DEFAULT_RATE_LIMIT_COOLDOWN;
    };
    let Ok(reset_epoch) = raw.trim().parse::<u64>() else {
        return DEFAULT_RATE_LIMIT_COOLDOWN;
    };
    let Ok(now_epoch) = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
    else {
        return DEFAULT_RATE_LIMIT_COOLDOWN;
    };
    let wait_secs = reset_epoch.saturating_sub(now_epoch).clamp(60, 3600);
    std::time::Duration::from_secs(wait_secs)
}

fn percent_decode_simple(input: &str) -> String {
    let bytes = input.as_bytes();
    let mut out = Vec::with_capacity(bytes.len());
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'%' && i + 2 < bytes.len() {
            if let Ok(hex_str) = std::str::from_utf8(&bytes[i + 1..i + 3]) {
                if let Ok(val) = u8::from_str_radix(hex_str, 16) {
                    out.push(val);
                    i += 3;
                    continue;
                }
            }
        }
        out.push(bytes[i]);
        i += 1;
    }
    String::from_utf8(out).unwrap_or_else(|_| input.to_string())
}

/// Extracts ordered, deduplicated release tags from a GitHub `releases.atom` XML feed.
/// Only inspects `<link ... href="https://github.com/.../releases/tag/{tag}"/>` elements
/// so tags referenced inside `<content>` release notes are ignored.
pub fn parse_atom_release_tags(xml: &str) -> Vec<String> {
    let mut tags = Vec::new();
    let mut rest = xml;
    while let Some(link_start) = rest.find("<link ") {
        rest = &rest[link_start + 6..];
        let Some(tag_end) = rest.find('>') else {
            break;
        };
        let attrs = &rest[..tag_end];
        rest = &rest[tag_end + 1..];

        let Some(href_pos) = attrs.find("href=\"") else {
            continue;
        };
        let after_href = &attrs[href_pos + 6..];
        let Some(href_end) = after_href.find('"') else {
            continue;
        };
        let href = &after_href[..href_end];
        let Some(tag_pos) = href.find("/releases/tag/") else {
            continue;
        };
        let raw_tag = &href[tag_pos + "/releases/tag/".len()..];
        let clean = raw_tag
            .split(['?', '#', '/'])
            .next()
            .unwrap_or("")
            .trim();
        if clean.is_empty() {
            continue;
        }
        let decoded = percent_decode_simple(clean);
        if !decoded.is_empty() && !tags.iter().any(|t| t == &decoded) {
            tags.push(decoded);
        }
    }
    tags
}

/// Extracts GitHub release asset `{name, browser_download_url}` objects from the HTML
/// returned by `https://github.com/{owner}/{repo}/releases/expanded_assets/{tag}`.
pub fn parse_expanded_assets_html(html: &str) -> Vec<serde_json::Value> {
    let mut assets = Vec::new();
    let mut seen_names = std::collections::HashSet::new();
    let mut rest = html;

    while let Some(href_pos) = rest.find("href=\"") {
        rest = &rest[href_pos + 6..];
        let Some(end_pos) = rest.find('"') else {
            break;
        };
        let href = &rest[..end_pos];
        rest = &rest[end_pos + 1..];

        if !href.contains("/releases/download/") {
            continue;
        }
        let full_url = if href.starts_with("https://") || href.starts_with("http://") {
            href.to_string()
        } else if href.starts_with('/') {
            format!("https://github.com{href}")
        } else {
            continue;
        };

        let path_only = full_url.split(['?', '#']).next().unwrap_or(&full_url);
        let Some(raw_name) = path_only.rsplit('/').next() else {
            continue;
        };
        if raw_name.is_empty() {
            continue;
        }
        let name = percent_decode_simple(raw_name);
        if seen_names.insert(name.clone()) {
            assets.push(serde_json::json!({
                "name": name,
                "browser_download_url": full_url,
            }));
        }
    }

    assets
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

        // NVIDIA without CUDA runtime picks Vulkan GPU build immediately!
        let nv_vk = select_best_llama_assets(&assets, "linux-x64", AccelBackend::CudaNvenc, Some((13, 0)), false).unwrap();
        assert_eq!(nv_vk.0, "llama-b5500-bin-ubuntu-vulkan-x64.tar.gz");

        let mut bundled = assets.clone();
        bundled.push(serde_json::json!({"name": "cudart-llama-b5500-bin-ubuntu-cuda-12.4-x64.tar.gz", "browser_download_url": "https://x/runtime.tar.gz"}));
        let nv_bundled = select_best_llama_assets(&bundled, "linux-x64", AccelBackend::CudaNvenc, Some((12, 6)), false).unwrap();
        assert_eq!(nv_bundled.0, "llama-b5500-bin-ubuntu-cuda-12.4-x64.tar.gz");
        assert_eq!(nv_bundled.2.as_deref(), Some("https://x/runtime.tar.gz"));

        // AMD picks Vulkan
        let amd = select_best_llama_assets(&assets, "linux-x64", AccelBackend::AmfDirectml, None, false).unwrap();
        assert_eq!(amd.0, "llama-b5500-bin-ubuntu-vulkan-x64.tar.gz");

        // CPU picks ubuntu-x64
        let cpu = select_best_llama_assets(&assets, "linux-x64", AccelBackend::CpuOnly, None, false).unwrap();
        assert_eq!(cpu.0, "llama-b5500-bin-ubuntu-x64.tar.gz");
    }

    #[test]
    fn parses_github_atom_release_tags_ignoring_body_links() {
        let xml = r#"
          <feed>
            <link type="text/html" rel="alternate" href="https://github.com/ggml-org/whisper.cpp/releases"/>
            <entry>
              <link rel="alternate" type="text/html" href="https://github.com/ggml-org/whisper.cpp/releases/tag/b5907"/>
              <content>See https://github.com/ggml-org/ggml/releases/tag/v0.26.0 for notes</content>
            </entry>
            <entry>
              <link rel="alternate" type="text/html" href="https://github.com/ggml-org/whisper.cpp/releases/tag/v1.9.5"/>
            </entry>
            <entry>
              <link rel="alternate" type="text/html" href="https://github.com/ggml-org/whisper.cpp/releases/tag/b5907"/>
            </entry>
            <entry>
              <link rel="alternate" type="text/html" href="https://github.com/ggml-org/whisper.cpp/releases/tag/b5454"/>
            </entry>
          </feed>
        "#;
        let tags = parse_atom_release_tags(xml);
        assert_eq!(tags, vec!["b5907", "v1.9.5", "b5454"]);
    }

    #[test]
    fn parses_expanded_assets_html_for_llama_whisper_and_ffmpeg() {
        let html = r#"
          <ul>
            <li><a href="/ggml-org/llama.cpp/releases/download/b12606/cudart-llama-b12606-bin-ubuntu-cuda-13.5-x64.tar.gz">cudart</a></li>
            <li><a href="/ggml-org/llama.cpp/releases/download/b12606/llama-b12606-bin-ubuntu-cuda-13.5-x64.tar.gz">cuda</a></li>
            <li><a href="/ggml-org/llama.cpp/releases/download/b12606/llama-b12606-bin-ubuntu-vulkan-x64.tar.gz">vulkan</a></li>
            <li><a href="/ggml-org/llama.cpp/archive/refs/tags/b12606.tar.gz">Source code</a></li>
          </ul>
        "#;
        let assets = parse_expanded_assets_html(html);
        assert_eq!(assets.len(), 3);
        let picked = select_best_llama_assets(
            &assets,
            "linux-x64",
            AccelBackend::CudaNvenc,
            Some((13, 5)),
            false,
        )
        .unwrap();
        assert_eq!(picked.0, "llama-b12606-bin-ubuntu-cuda-13.5-x64.tar.gz");
        assert_eq!(
            picked.1,
            "https://github.com/ggml-org/llama.cpp/releases/download/b12606/llama-b12606-bin-ubuntu-cuda-13.5-x64.tar.gz"
        );
        assert_eq!(
            picked.2.as_deref(),
            Some("https://github.com/ggml-org/llama.cpp/releases/download/b12606/cudart-llama-b12606-bin-ubuntu-cuda-13.5-x64.tar.gz")
        );
    }

    #[test]
    fn rate_limit_cooldown_clamps_reasonable_bounds() {
        assert_eq!(parse_rate_limit_cooldown(None), DEFAULT_RATE_LIMIT_COOLDOWN);
        assert_eq!(
            parse_rate_limit_cooldown(Some("not-a-number")),
            DEFAULT_RATE_LIMIT_COOLDOWN
        );
        // Past timestamp clamps to 60s minimum
        assert_eq!(
            parse_rate_limit_cooldown(Some("1000")),
            std::time::Duration::from_secs(60)
        );
    }

    #[test]
    fn rate_limit_blocker_tracks_active_cooldown() {
        let tmp = tempfile::tempdir().unwrap();
        let events = Arc::new(EventBus::new(16));
        let jobs = Arc::new(JobManager::new(events.clone()));
        let hw = HardwareProfile::detect();
        let tm = ToolManager::new(tmp.path().to_path_buf(), hw, events, jobs);
        assert!(!tm.github_api_is_blocked());
        tm.mark_github_api_rate_limited(std::time::Duration::from_secs(60));
        assert!(tm.github_api_is_blocked());
    }

    #[tokio::test]
    #[ignore = "Requires live network access to github.com"]
    async fn live_web_fallback_resolves_all_tools_when_api_is_rate_limited() {
        let tmp = tempfile::tempdir().unwrap();
        let events = Arc::new(EventBus::new(16));
        let jobs = Arc::new(JobManager::new(events.clone()));
        let hw = HardwareProfile::detect();
        let tm = ToolManager::new(tmp.path().to_path_buf(), hw, events, jobs);

        // Simulate GitHub REST API 403 rate limit exceeded!
        tm.mark_github_api_rate_limited(std::time::Duration::from_secs(600));
        assert!(tm.github_api_is_blocked());

        let ytdlp = tm.resolve_ytdlp_release().await.expect("yt-dlp web fallback");
        assert!(ytdlp.url.contains("github.com/yt-dlp/yt-dlp/releases/"));

        let ffmpeg = tm.resolve_ffmpeg_release().await.expect("ffmpeg web fallback");
        assert!(ffmpeg.url.contains("FFmpeg-Builds/releases/download/"));
        assert!(ffmpeg.url.ends_with("gpl.tar.xz"));

        let whisper = tm.resolve_whisper_release().await.expect("whisper web fallback");
        assert!(whisper.url.contains("whisper.cpp/releases/download/"));
        assert_ne!(whisper.version, "v1.9.5"); // skipped source-only tags

        let llama = tm.resolve_llama_release().await.expect("llama web fallback");
        assert!(llama.url.contains("llama.cpp/releases/download/"));

        let whisper_versions = tm.versions("whisper").await.expect("whisper versions via web");
        assert!(!whisper_versions.is_empty());
    }
}


