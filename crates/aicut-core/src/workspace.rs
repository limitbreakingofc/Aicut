use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

use crate::error::AicutResult;

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct WorkspaceConfig {
    /// Root directory for user-facing media files (downloads, imports, exports).
    pub workspace_dir: PathBuf,
    /// Directory where yt-dlp saves downloaded videos and audio.
    pub downloads_dir: PathBuf,
    /// Directory where copied local imports are stored.
    pub imports_dir: PathBuf,
    /// Directory where generated subtitles, timelines, and transcripts are saved.
    pub exports_dir: PathBuf,
    /// Directory for generated thumbnails and waveform previews.
    pub thumbnails_dir: PathBuf,
    /// Directory for downloaded local AI models (e.g. Whisper GGML weights).
    pub models_dir: PathBuf,
    /// Directory for Aicut-managed external binaries (yt-dlp, ffmpeg, whisper-cli, future tools).
    pub bin_dir: PathBuf,
    /// Path to SQLite database storing library metadata, transcripts, presets, and settings.
    pub db_path: PathBuf,
}

impl WorkspaceConfig {
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
            exports_dir: workspace_dir.join("Exports"),
            thumbnails_dir: data_dir.join("thumbnails"),
            models_dir: data_dir.join("models"),
            bin_dir: data_dir.join("bin"),
            db_path: data_dir.join("aicut.db"),
            workspace_dir,
        }
    }

    pub fn from_root(root: impl AsRef<Path>) -> Self {
        let root = root.as_ref().to_path_buf();
        Self {
            downloads_dir: root.join("downloads"),
            imports_dir: root.join("imports"),
            exports_dir: root.join("exports"),
            thumbnails_dir: root.join("thumbnails"),
            models_dir: root.join("models"),
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
            &self.exports_dir,
            &self.thumbnails_dir,
            &self.models_dir,
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
