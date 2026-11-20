pub mod ffprobe;
pub mod formats;
pub mod importer;
pub mod ytdlp;

use aicut_core::{
    ActionAnnotations, ActionHandler, AicutError, AicutEvent, AicutKernel, AicutModule,
    DownloadPreset, EmptyInput, MediaAsset, MediaListFilter, ModuleHealth, ModuleManifest,
    ToolDefinition, ToolInstallStrategy, TypedAction,
};
use async_trait::async_trait;
use chrono::Utc;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::sync::Arc;
use uuid::Uuid;

use crate::importer::{import_local, ImportLocalRequest, ImportLocalResponse};
use crate::ytdlp::{probe_url, start_download, DownloadRequest, DownloadResponse, ProbeUrlRequest, ProbeUrlResult};

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct MediaIdInput {
    pub id: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct MediaUpdateInput {
    pub id: String,
    #[serde(default)]
    pub title: Option<String>,
    #[serde(default)]
    pub tags: Option<Vec<String>>,
    #[serde(default)]
    pub notes: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct MediaDeleteInput {
    pub id: String,
    /// When true, also deletes the underlying media file from disk. Defaults to false for safety.
    #[serde(default)]
    pub delete_file: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct MediaDeleteOutput {
    pub deleted_id: String,
    pub file_deleted: bool,
    pub moved_to_trash: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct MediaBulkDeleteInput {
    pub ids: Vec<String>,
    /// When false (default), moves items to the Aicut Trash keeping files on disk.
    /// When true, permanently deletes DB entries and removes files from disk.
    #[serde(default)]
    pub delete_file: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct MediaBulkDeleteOutput {
    pub processed_ids: Vec<String>,
    pub files_deleted: usize,
    pub freed_bytes: u64,
    pub moved_to_trash: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct TrashListOutput {
    pub total: usize,
    pub total_size_bytes: u64,
    pub items: Vec<MediaAsset>,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct MediaRestoreInput {
    pub ids: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct MediaRestoreOutput {
    pub restored_ids: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct TrashEmptyInput {
    /// Optional subset of trashed media IDs to purge; if omitted or empty, empties the entire Trash.
    #[serde(default)]
    pub ids: Option<Vec<String>>,
    /// Delete underlying files from disk as well (defaults to true when purging from Trash).
    #[serde(default = "default_true")]
    pub delete_files: bool,
}

fn default_true() -> bool {
    true
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct TrashEmptyOutput {
    pub purged_count: usize,
    pub files_deleted: usize,
    pub freed_bytes: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct MediaListOutput {
    pub total: usize,
    pub items: Vec<MediaAsset>,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct SavePresetInput {
    #[serde(default)]
    pub id: Option<String>,
    pub name: String,
    #[serde(default)]
    pub description: String,
    pub config: Value,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct PresetsListOutput {
    pub presets: Vec<DownloadPreset>,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct DeletePresetOutput {
    pub deleted: bool,
    pub id: String,
}

pub struct MediaIngestModule;

impl MediaIngestModule {
    pub fn new() -> Arc<Self> {
        Arc::new(Self)
    }
}

#[async_trait]
impl AicutModule for MediaIngestModule {
    fn manifest(&self) -> ModuleManifest {
        ModuleManifest {
            id: "media-ingest".to_string(),
            name: "Ingestão & Biblioteca de Mídia".to_string(),
            version: env!("CARGO_PKG_VERSION").to_string(),
            description: "Baixa vídeos e áudios com todas as opções do yt-dlp, importa arquivos locais do PC e gerencia a Biblioteca de Mídia.".to_string(),
            category: "media".to_string(),
            core: false,
            capabilities: vec![
                "yt-dlp-download".into(),
                "url-probe".into(),
                "local-video-import".into(),
                "local-audio-import".into(),
                "media-library".into(),
                "download-presets".into(),
            ],
        }
    }

    fn declared_tools(&self) -> Vec<ToolDefinition> {
        vec![
            ToolDefinition {
                id: "yt-dlp".to_string(),
                display_name: "yt-dlp".to_string(),
                description: "Motor de download para YouTube, TikTok, Twitch, Instagram e +1000 sites com controle total de formatos, legendas e recortes.".to_string(),
                owner_module: "media-ingest".to_string(),
                category: "download".to_string(),
                binary_name: "yt-dlp".to_string(),
                version_args: vec!["--version".to_string()],
                strategy: ToolInstallStrategy::YtDlpOfficialRelease,
            },
            ToolDefinition {
                id: "ffmpeg".to_string(),
                display_name: "FFmpeg".to_string(),
                description: "Motor multimídia para mesclagem de vídeo/áudio, extração de áudio, geração de thumbnails e aceleração por GPU.".to_string(),
                owner_module: "media-ingest".to_string(),
                category: "multimedia".to_string(),
                binary_name: "ffmpeg".to_string(),
                version_args: vec!["-version".to_string()],
                strategy: ToolInstallStrategy::FfmpegHardwareBundle,
            },
            ToolDefinition {
                id: "ffprobe".to_string(),
                display_name: "FFprobe".to_string(),
                description: "Analisador técnico de streams de vídeo e áudio (resolução, FPS, codecs, bitrate, duração).".to_string(),
                owner_module: "media-ingest".to_string(),
                category: "multimedia".to_string(),
                binary_name: "ffprobe".to_string(),
                version_args: vec!["-version".to_string()],
                strategy: ToolInstallStrategy::CompanionOf("ffmpeg".to_string()),
            },
        ]
    }

    async fn health(&self, kernel: &Arc<AicutKernel>) -> ModuleHealth {
        let has_ytdlp = kernel.tools.resolve_tool("yt-dlp").is_ok();
        let has_ffmpeg = kernel.tools.resolve_tool("ffmpeg").is_ok();
        match (has_ytdlp, has_ffmpeg) {
            (true, true) => ModuleHealth::healthy("yt-dlp e FFmpeg prontos"),
            (false, true) => ModuleHealth::degraded("yt-dlp não instalado — instale em Configurações > Ferramentas"),
            (true, false) => ModuleHealth::degraded("FFmpeg não instalado — necessário para merge de vídeo/áudio e thumbnails"),
            (false, false) => ModuleHealth::degraded("yt-dlp e FFmpeg pendentes — use Configurações > Ferramentas para instalar"),
        }
    }

    fn actions(&self) -> Vec<Arc<dyn ActionHandler>> {
        vec![
            Arc::new(TypedAction::<ProbeUrlRequest, ProbeUrlResult, _>::new(
                "media.probe_url",
                "media-ingest",
                "Analisar URL (yt-dlp Probe)",
                "Inspect a video/audio URL via yt-dlp without downloading. Returns title, uploader, duration, thumbnail, subtitles, chapters, and exact stream formats (video+audio, video_only, audio_only).",
                ActionAnnotations {
                    read_only: true,
                    idempotent: true,
                    ..Default::default()
                },
                |ctx, req| {
                    Box::pin(async move {
                        let res = probe_url(&ctx.kernel, &req).await?;
                        let summary = format!(
                            "Probed '{}' ({} formats available)",
                            res.title,
                            res.formats.len()
                        );
                        Ok((
                            res,
                            summary,
                            vec!["Call media.download with chosen mode/quality/format_id".into()],
                        ))
                    })
                },
            )),
            Arc::new(TypedAction::<DownloadRequest, DownloadResponse, _>::new(
                "media.download",
                "media-ingest",
                "Baixar Vídeo/Áudio (yt-dlp)",
                "Download video or audio from any yt-dlp supported URL with full visual/AI configurability (video+audio, video_only, audio_only, resolution, codec, container, subtitles, clip range, chapters, playlist, cookies, SponsorBlock, speed limit, concurrent fragments). Automatically indexes the result into the Library.",
                ActionAnnotations {
                    long_running: true,
                    ..Default::default()
                },
                |ctx, req| {
                    Box::pin(async move {
                        let res = start_download(ctx.kernel.clone(), req, ctx.actor).await?;
                        let summary = if res.assets.is_empty() {
                            format!("Started yt-dlp download job {}", res.job_id)
                        } else {
                            format!(
                                "Downloaded {} media item(s) and added to Library",
                                res.assets.len()
                            )
                        };
                        Ok((
                            res,
                            summary,
                            vec![
                                "Monitor progress with jobs.get or inspect the Library with media.list".into(),
                            ],
                        ))
                    })
                },
            )),
            Arc::new(TypedAction::<ImportLocalRequest, ImportLocalResponse, _>::new(
                "media.import_local",
                "media-ingest",
                "Importar Vídeos/Áudios do PC",
                "Import one or more local video/audio files or folders from the PC into the Aicut Library (supports all video and audio formats supported by yt-dlp/FFmpeg). Supports 'reference' (instant, no copy) and 'copy' modes.",
                ActionAnnotations {
                    long_running: true,
                    ..Default::default()
                },
                |ctx, req| {
                    Box::pin(async move {
                        let res = import_local(ctx.kernel.clone(), req, ctx.actor).await?;
                        let summary = format!(
                            "Imported {} local media item(s) ({} duplicate(s) skipped, {} error(s))",
                            res.imported.len(),
                            res.skipped_duplicates.len(),
                            res.errors.len()
                        );
                        Ok((
                            res,
                            summary,
                            vec!["Navigate to Library with ui.navigate {\"route\":\"library\"}".into()],
                        ))
                    })
                },
            )),
            Arc::new(TypedAction::<MediaListFilter, MediaListOutput, _>::new(
                "media.list",
                "media-ingest",
                "Listar Biblioteca de Mídia",
                "List, search, and filter all downloaded (yt-dlp) and locally imported video/audio assets in the Aicut Library.",
                ActionAnnotations {
                    read_only: true,
                    idempotent: true,
                    ..Default::default()
                },
                |ctx, filter| {
                    Box::pin(async move {
                        let items = ctx.kernel.repo.list_media(&filter)?;
                        let total = items.len();
                        let summary = format!("Found {total} media asset(s) in Library");
                        Ok((MediaListOutput { total, items }, summary, vec![]))
                    })
                },
            )),
            Arc::new(TypedAction::<MediaIdInput, MediaAsset, _>::new(
                "media.get",
                "media-ingest",
                "Obter Detalhes da Mídia",
                "Retrieve full metadata, file path, resolution, codecs, tags, and source URL for a Library media item by ID.",
                ActionAnnotations {
                    read_only: true,
                    idempotent: true,
                    ..Default::default()
                },
                |ctx, input| {
                    Box::pin(async move {
                        let asset = ctx
                            .kernel
                            .repo
                            .get_media(&input.id)?
                            .ok_or_else(|| AicutError::not_found(format!("Media '{}' not found", input.id)))?;
                        let summary = format!("Loaded media '{}' ({:?})", asset.title, asset.kind);
                        Ok((asset, summary, vec![]))
                    })
                },
            )),
            Arc::new(TypedAction::<MediaUpdateInput, MediaAsset, _>::new(
                "media.update",
                "media-ingest",
                "Atualizar Metadados da Mídia",
                "Update the title, tags, or notes of a Library media asset.",
                ActionAnnotations {
                    idempotent: true,
                    ..Default::default()
                },
                |ctx, input| {
                    Box::pin(async move {
                        let updated = ctx.kernel.repo.update_media(
                            &input.id,
                            input.title,
                            input.tags,
                            input.notes,
                        )?;
                        ctx.kernel
                            .events
                            .publish(AicutEvent::MediaUpdated(updated.clone()));
                        let summary = format!("Updated media '{}'", updated.title);
                        Ok((updated, summary, vec![]))
                    })
                },
            )),
            Arc::new(TypedAction::<MediaDeleteInput, MediaDeleteOutput, _>::new(
                "media.delete",
                "media-ingest",
                "Remover Mídia da Biblioteca",
                "Move a media item to the Aicut Trash (when delete_file=false, default) or permanently delete it from the Library and disk (when delete_file=true).",
                ActionAnnotations {
                    destructive: true,
                    ..Default::default()
                },
                |ctx, input| {
                    Box::pin(async move {
                        let (deleted, file_deleted, moved_to_trash) = if input.delete_file {
                            let item = ctx
                                .kernel
                                .repo
                                .delete_media(&input.id)?
                                .ok_or_else(|| AicutError::not_found(format!("Media '{}' not found", input.id)))?;
                            let mut removed = false;
                            let p = std::path::Path::new(&item.file_path);
                            if p.is_file() {
                                let _ = std::fs::remove_file(p);
                                removed = true;
                            }
                            if let Some(ref thumb) = item.thumbnail_path {
                                let _ = std::fs::remove_file(thumb);
                            }
                            (item, removed, false)
                        } else {
                            let item = ctx
                                .kernel
                                .repo
                                .trash_media(&input.id)?
                                .ok_or_else(|| AicutError::not_found(format!("Media '{}' not found", input.id)))?;
                            (item, false, true)
                        };
                        ctx.kernel
                            .events
                            .publish(AicutEvent::MediaDeleted { id: input.id.clone() });
                        let summary = if moved_to_trash {
                            format!("Moved '{}' to Trash (file kept on disk)", deleted.title)
                        } else if file_deleted {
                            format!("Permanently deleted '{}' and removed file from disk", deleted.title)
                        } else {
                            format!("Permanently removed '{}' from Library", deleted.title)
                        };
                        Ok((
                            MediaDeleteOutput {
                                deleted_id: input.id,
                                file_deleted,
                                moved_to_trash,
                            },
                            summary,
                            vec![],
                        ))
                    })
                },
            )),
            Arc::new(TypedAction::<MediaBulkDeleteInput, MediaBulkDeleteOutput, _>::new(
                "media.bulk_delete",
                "media-ingest",
                "Remover Múltiplas Mídias",
                "Move multiple selected media items to Trash (delete_file=false) or permanently delete them and their files from disk (delete_file=true).",
                ActionAnnotations {
                    destructive: true,
                    ..Default::default()
                },
                |ctx, input| {
                    Box::pin(async move {
                        let mut processed_ids = Vec::new();
                        let mut files_deleted = 0usize;
                        let mut freed_bytes = 0u64;

                        for id in &input.ids {
                            if input.delete_file {
                                if let Some(item) = ctx.kernel.repo.delete_media(id)? {
                                    let p = std::path::Path::new(&item.file_path);
                                    if p.is_file() {
                                        let sz = std::fs::metadata(p).map(|m| m.len()).unwrap_or(item.file_size_bytes);
                                        if std::fs::remove_file(p).is_ok() {
                                            files_deleted += 1;
                                            freed_bytes += sz;
                                        }
                                    }
                                    if let Some(ref thumb) = item.thumbnail_path {
                                        let _ = std::fs::remove_file(thumb);
                                    }
                                    ctx.kernel.events.publish(AicutEvent::MediaDeleted { id: id.clone() });
                                    processed_ids.push(id.clone());
                                }
                            } else if ctx.kernel.repo.trash_media(id)?.is_some() {
                                ctx.kernel.events.publish(AicutEvent::MediaDeleted { id: id.clone() });
                                processed_ids.push(id.clone());
                            }
                        }

                        let moved_to_trash = !input.delete_file;
                        let summary = if moved_to_trash {
                            format!("Moved {} item(s) to Trash", processed_ids.len())
                        } else {
                            format!(
                                "Permanently deleted {} item(s) ({files_deleted} file(s) removed from disk)",
                                processed_ids.len()
                            )
                        };
                        Ok((
                            MediaBulkDeleteOutput {
                                processed_ids,
                                files_deleted,
                                freed_bytes,
                                moved_to_trash,
                            },
                            summary,
                            vec![],
                        ))
                    })
                },
            )),
            Arc::new(TypedAction::<EmptyInput, TrashListOutput, _>::new(
                "media.trash_list",
                "media-ingest",
                "Listar Lixeira da Biblioteca",
                "List all soft-deleted media items currently in the Aicut Trash that can be restored or permanently purged.",
                ActionAnnotations {
                    read_only: true,
                    idempotent: true,
                    ..Default::default()
                },
                |ctx, _| {
                    Box::pin(async move {
                        let items = ctx.kernel.repo.list_trashed_media()?;
                        let total = items.len();
                        let total_size_bytes: u64 = items.iter().map(|i| i.file_size_bytes).sum();
                        let summary = format!("{total} item(s) in Trash");
                        Ok((
                            TrashListOutput {
                                total,
                                total_size_bytes,
                                items,
                            },
                            summary,
                            vec![],
                        ))
                    })
                },
            )),
            Arc::new(TypedAction::<MediaRestoreInput, MediaRestoreOutput, _>::new(
                "media.restore",
                "media-ingest",
                "Restaurar Mídias da Lixeira",
                "Restore one or more trashed media items back into the active Library.",
                ActionAnnotations {
                    idempotent: true,
                    ..Default::default()
                },
                |ctx, input| {
                    Box::pin(async move {
                        let mut restored_ids = Vec::new();
                        for id in &input.ids {
                            if let Some(asset) = ctx.kernel.repo.restore_media(id)? {
                                ctx.kernel.events.publish(AicutEvent::MediaCreated(asset));
                                restored_ids.push(id.clone());
                            }
                        }
                        let summary = format!("Restored {} item(s) from Trash", restored_ids.len());
                        Ok((MediaRestoreOutput { restored_ids }, summary, vec![]))
                    })
                },
            )),
            Arc::new(TypedAction::<TrashEmptyInput, TrashEmptyOutput, _>::new(
                "media.trash_empty",
                "media-ingest",
                "Esvaziar ou Excluir Permanentemente da Lixeira",
                "Permanently purge selected trashed items (or the entire Trash when ids is omitted) and optionally remove their files from disk.",
                ActionAnnotations {
                    destructive: true,
                    ..Default::default()
                },
                |ctx, input| {
                    Box::pin(async move {
                        let all_trashed = ctx.kernel.repo.list_trashed_media()?;
                        let filter_set: Option<std::collections::HashSet<&String>> = input
                            .ids
                            .as_ref()
                            .filter(|v| !v.is_empty())
                            .map(|v| v.iter().collect());

                        let mut purged_count = 0usize;
                        let mut files_deleted = 0usize;
                        let mut freed_bytes = 0u64;

                        for item in all_trashed {
                            if let Some(ref set) = filter_set {
                                if !set.contains(&item.id) {
                                    continue;
                                }
                            }
                            if ctx.kernel.repo.delete_media(&item.id)?.is_some() {
                                purged_count += 1;
                                if input.delete_files {
                                    let p = std::path::Path::new(&item.file_path);
                                    if p.is_file() {
                                        let sz = std::fs::metadata(p).map(|m| m.len()).unwrap_or(item.file_size_bytes);
                                        if std::fs::remove_file(p).is_ok() {
                                            files_deleted += 1;
                                            freed_bytes += sz;
                                        }
                                    }
                                }
                                if let Some(ref thumb) = item.thumbnail_path {
                                    let _ = std::fs::remove_file(thumb);
                                }
                            }
                        }

                        let summary = format!(
                            "Purged {purged_count} item(s) from Trash ({files_deleted} file(s) deleted from disk)"
                        );
                        Ok((
                            TrashEmptyOutput {
                                purged_count,
                                files_deleted,
                                freed_bytes,
                            },
                            summary,
                            vec![],
                        ))
                    })
                },
            )),
            Arc::new(TypedAction::<EmptyInput, PresetsListOutput, _>::new(
                "media.presets_list",
                "media-ingest",
                "Listar Presets do yt-dlp",
                "List saved yt-dlp download configuration presets.",
                ActionAnnotations {
                    read_only: true,
                    idempotent: true,
                    ..Default::default()
                },
                |ctx, _| {
                    Box::pin(async move {
                        let presets = ctx.kernel.repo.list_presets()?;
                        let summary = format!("Loaded {} download preset(s)", presets.len());
                        Ok((PresetsListOutput { presets }, summary, vec![]))
                    })
                },
            )),
            Arc::new(TypedAction::<SavePresetInput, DownloadPreset, _>::new(
                "media.presets_save",
                "media-ingest",
                "Salvar Preset do yt-dlp",
                "Create or update a reusable yt-dlp download preset.",
                ActionAnnotations {
                    idempotent: true,
                    ..Default::default()
                },
                |ctx, input| {
                    Box::pin(async move {
                        let preset = DownloadPreset {
                            id: input.id.unwrap_or_else(|| format!("preset-{}", Uuid::new_v4())),
                            name: input.name,
                            description: input.description,
                            config: input.config,
                            created_at: Utc::now(),
                        };
                        ctx.kernel.repo.save_preset(&preset)?;
                        let summary = format!("Saved download preset '{}'", preset.name);
                        Ok((preset, summary, vec![]))
                    })
                },
            )),
            Arc::new(TypedAction::<MediaIdInput, DeletePresetOutput, _>::new(
                "media.presets_delete",
                "media-ingest",
                "Excluir Preset do yt-dlp",
                "Delete a saved yt-dlp download preset by ID.",
                ActionAnnotations {
                    destructive: true,
                    ..Default::default()
                },
                |ctx, input| {
                    Box::pin(async move {
                        let deleted = ctx.kernel.repo.delete_preset(&input.id)?;
                        let summary = format!("Deleted preset '{}': {deleted}", input.id);
                        Ok((
                            DeletePresetOutput {
                                deleted,
                                id: input.id,
                            },
                            summary,
                            vec![],
                        ))
                    })
                },
            )),
        ]
    }
}

#[cfg(test)]
mod integration_tests {
    use super::*;
    use aicut_core::{ActorKind, MediaKind, WorkspaceConfig};
    use std::process::Command;

    #[tokio::test]
    async fn imports_real_video_video_only_and_audio_files_with_ffmpeg() {
        // Only run ffprobe/ffmpeg generation if ffmpeg is available in the test host
        if which::which("ffmpeg").is_err() {
            return;
        }
        let tmp = tempfile::tempdir().unwrap();
        let ws = WorkspaceConfig::from_root(tmp.path().join("workspace"));
        let kernel = AicutKernel::new(ws).unwrap();
        kernel.register_module(MediaIngestModule::new()).await.unwrap();

        let video_av = tmp.path().join("sample_av.mp4");
        let video_only = tmp.path().join("sample_vo.mp4");
        let audio_only = tmp.path().join("sample_audio.mp3");

        // 1. Video + Audio
        assert!(Command::new("ffmpeg")
            .args([
                "-y", "-f", "lavfi", "-i", "testsrc=duration=1:size=320x240:rate=24",
                "-f", "lavfi", "-i", "sine=frequency=440:duration=1",
                "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest"
            ])
            .arg(&video_av)
            .status()
            .unwrap()
            .success());

        // 2. Video Only
        assert!(Command::new("ffmpeg")
            .args([
                "-y", "-f", "lavfi", "-i", "testsrc=duration=1:size=320x240:rate=24",
                "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p"
            ])
            .arg(&video_only)
            .status()
            .unwrap()
            .success());

        // 3. Audio Only
        assert!(Command::new("ffmpeg")
            .args([
                "-y", "-f", "lavfi", "-i", "sine=frequency=440:duration=1",
                "-c:a", "libmp3lame"
            ])
            .arg(&audio_only)
            .status()
            .unwrap()
            .success());

        let out = kernel
            .dispatch(
                "media.import_local",
                serde_json::json!({
                    "paths": [
                        video_av.to_string_lossy(),
                        video_only.to_string_lossy(),
                        audio_only.to_string_lossy()
                    ],
                    "mode": "copy",
                    "tags": ["test", "e2e"],
                    "wait": true
                }),
                ActorKind::Ai,
            )
            .await
            .unwrap();

        let resp: ImportLocalResponse = serde_json::from_value(out.data).unwrap();
        assert_eq!(resp.imported.len(), 3);
        assert!(resp.errors.is_empty());

        let kinds: Vec<MediaKind> = resp.imported.iter().map(|a| a.kind).collect();
        assert!(kinds.contains(&MediaKind::Video));
        assert!(kinds.contains(&MediaKind::VideoOnly));
        assert!(kinds.contains(&MediaKind::Audio));

        let av_asset = resp.imported.iter().find(|a| a.kind == MediaKind::Video).unwrap();
        assert_eq!(av_asset.metadata.width, Some(320));
        assert_eq!(av_asset.metadata.height, Some(240));
        assert!(av_asset.thumbnail_path.is_some());

        // Verify media.list & media.delete
        let list_out = kernel
            .dispatch("media.list", serde_json::json!({"kind": "audio"}), ActorKind::Human)
            .await
            .unwrap();
        let list_resp: MediaListOutput = serde_json::from_value(list_out.data).unwrap();
        assert_eq!(list_resp.total, 1);
    }
}
