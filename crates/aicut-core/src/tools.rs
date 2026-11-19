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

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum ToolInstallStrategy {
    /// Downloads the official yt-dlp standalone binary matching host OS & CPU arch.
    YtDlpOfficialRelease,
    /// Downloads the optimal FFmpeg/FFprobe build (including NVENC/AMF/QSV support on Windows x64).
    FfmpegHardwareBundle,
    /// Companion binary provided alongside another managed tool (e.g. ffprobe inside ffmpeg bundle).
    CompanionOf(String),
    /// Placeholder for future module-provided tools (Whisper, ONNX, etc.).
    Custom { homepage: String },
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct ToolDefinition {
    pub id: String,
    pub display_name: String,
    pub description: String,
    pub owner_module: String,
    pub category: String,
    /// Base binary name without .exe suffix (e.g. "yt-dlp", "ffmpeg", "ffprobe").
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
    pub version: String,
    pub build_label: String,
    pub archive_kind: ArchiveKind,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ArchiveKind {
    SingleBinary,
    Zip,
    TarXzOrGz,
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
            .user_agent("Aicut/0.1.0 (+https://github.com/limitbreakingofc/Aicut)")
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

    /// Resolves a binary path: checks Aicut's managed `bin_dir` first, then system PATH.
    pub fn resolve_binary(&self, binary_name: &str) -> Option<(PathBuf, bool)> {
        let managed_name = platform_binary_name(binary_name);
        let managed_path = self.bin_dir.join(&managed_name);
        if managed_path.is_file() {
            return Some((managed_path, true));
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
            ToolInstallStrategy::FfmpegHardwareBundle
            | ToolInstallStrategy::CompanionOf(_) => {
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
                    AccelBackend::Metal => {
                        "Apple Silicon detectado: suporte VideoToolbox nativo."
                    }
                    AccelBackend::CpuOnly => {
                        "Modo CPU otimizado: build multithread para todos os núcleos lógicos."
                    }
                };
                (build.to_string(), hw_note.to_string())
            }
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
                let ver = probe_binary_version(&path, &def.version_args).await;
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
                ToolInstallStrategy::CompanionOf(_) => {
                    return Err(AicutError::internal("Companion tool resolution loop"));
                }
                ToolInstallStrategy::Custom { homepage } => {
                    return Err(AicutError::new(
                        "UNSUPPORTED_AUTO_INSTALL",
                        format!("Tool '{}' must be installed via its module ({homepage}).", target_def.id),
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
                ArchiveKind::Zip => {
                    extract_ffmpeg_from_zip(&tmp_path, &self.bin_dir)?;
                    let _ = std::fs::remove_file(&tmp_path);
                }
                ArchiveKind::TarXzOrGz => {
                    extract_ffmpeg_from_targz(&tmp_path, &self.bin_dir)?;
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
                version: "latest-win64-gpl".to_string(),
                build_label: "ffmpeg-master-latest-win64-gpl.zip (NVENC/AMF/QSV)".to_string(),
                archive_kind: ArchiveKind::Zip,
            }),
            "win-arm64" => Ok(ResolvedDownload {
                url: "https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-master-latest-winarm64-gpl.zip".to_string(),
                version: "latest-winarm64-gpl".to_string(),
                build_label: "ffmpeg-master-latest-winarm64-gpl.zip".to_string(),
                archive_kind: ArchiveKind::Zip,
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
    let output = Command::new(path).args(args).output().await.ok()?;
    let stdout = String::from_utf8_lossy(&output.stdout);
    let first_line = stdout.lines().next()?.trim();
    if first_line.is_empty() {
        return None;
    }
    // Extract clean version token for ffmpeg ("ffmpeg version 6.1.1 ...") or yt-dlp ("2025.01.15")
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn selects_hardware_appropriate_build_labels() {
        let hw = HardwareProfile::detect();
        let events = Arc::new(EventBus::new(16));
        let jobs = Arc::new(JobManager::new(events.clone()));
        let tm = ToolManager::new(PathBuf::from("/tmp/aicut-test-bin"), hw, events, jobs);
        let ytdlp_def = ToolDefinition {
            id: "yt-dlp".into(),
            display_name: "yt-dlp".into(),
            description: "Video downloader".into(),
            owner_module: "media-ingest".into(),
            category: "media".into(),
            binary_name: "yt-dlp".into(),
            version_args: vec!["--version".into()],
            strategy: ToolInstallStrategy::YtDlpOfficialRelease,
        };
        let (build, note) = tm.recommend_build_for(&ytdlp_def);
        assert!(build.contains("yt-dlp"));
        assert!(!note.is_empty());
    }
}
