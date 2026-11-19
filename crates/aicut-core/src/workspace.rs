use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

use crate::error::AicutResult;

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct WorkspaceConfig {
    /// Root directory for user-facing media files (downloads & copied imports).
    pub workspace_dir: PathBuf,
    /// Directory where yt-dlp saves downloaded videos and audio.
    pub downloads_dir: PathBuf,
    /// Directory where copied local imports are stored.
    pub imports_dir: PathBuf,
    /// Directory for generated thumbnails and waveform previews.
    pub thumbnails_dir: PathBuf,
    /// Directory for Aicut-managed external binaries (yt-dlp, ffmpeg, future AI tools).
    pub bin_dir: PathBuf,
    /// Path to SQLite database storing library metadata, presets, jobs, and settings.
    pub db_path: PathBuf,
}

impl WorkspaceConfig {
    /// Builds default workspace paths suited for Windows 11 (Videos\Aicut + LocalAppData\Aicut)
    /// and Unix environments, with environment variable overrides for testing/portable mode.
    pub fn default_paths() -> Self {
        let env_workspace = std::env::var("AICUT_WORKSPACE_DIR").ok().map(PathBuf::from);
        let env_data = std::env::var("AICUT_DATA_DIR").ok().map(PathBuf::from);

        let workspace_dir = env_workspace.unwrap_or_else(|| {
            dirs::video_dir()
                .or_else(dirs::home_dir)
                .unwrap_or_else(|| PathBuf::from("."))
                .join("Aicut")
        });

        let data_dir = env_data.unwrap_or_else(|| {
            dirs::data_local_dir()
                .unwrap_or_else(|| workspace_dir.join(".aicut"))
                .join("Aicut")
        });

        Self {
            downloads_dir: workspace_dir.join("Downloads"),
            imports_dir: workspace_dir.join("Imports"),
            thumbnails_dir: data_dir.join("thumbnails"),
            bin_dir: data_dir.join("bin"),
            db_path: data_dir.join("aicut.db"),
            workspace_dir,
        }
    }

    /// Creates an isolated workspace rooted at `root` (ideal for tests and portable mode).
    pub fn from_root(root: impl AsRef<Path>) -> Self {
        let root = root.as_ref().to_path_buf();
        Self {
            downloads_dir: root.join("downloads"),
            imports_dir: root.join("imports"),
            thumbnails_dir: root.join("thumbnails"),
            bin_dir: root.join("bin"),
            db_path: root.join("aicut.db"),
            workspace_dir: root,
        }
    }

    pub fn ensure_dirs(&self) -> AicutResult<()> {
        for dir in [
            &self.workspace_dir,
            &self.downloads_dir,
            &self.imports_dir,
            &self.thumbnails_dir,
            &self.bin_dir,
        ] {
            std::fs::create_dir_all(dir)?;
        }
        if let Some(parent) = self.db_path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        Ok(())
    }
}
