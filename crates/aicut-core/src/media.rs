use chrono::{DateTime, Utc};
use rusqlite::{params, Connection, OptionalExtension};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::path::Path;
use std::sync::Mutex;

use crate::error::{AicutError, AicutResult};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum MediaKind {
    Video,
    VideoOnly,
    Audio,
    Subtitle,
    Timeline,
    Transcript,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum MediaSourceType {
    Download,
    Local,
    Generated,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, JsonSchema)]
pub struct MediaMetadata {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub duration_sec: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub width: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub height: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub fps: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub video_codec: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub audio_codec: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub audio_channels: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sample_rate: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub bitrate_bps: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub format_name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub uploader: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub extractor: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub has_subtitles: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub language: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub transcript_id: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct MediaAsset {
    pub id: String,
    pub title: String,
    pub kind: MediaKind,
    pub source_type: MediaSourceType,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub source_url: Option<String>,
    /// Links generated subtitles, timelines, and transcripts back to their source video/audio asset.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub parent_media_id: Option<String>,
    pub file_path: String,
    pub file_name: String,
    pub extension: String,
    pub file_size_bytes: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub fingerprint: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub thumbnail_path: Option<String>,
    pub metadata: MediaMetadata,
    pub tags: Vec<String>,
    pub notes: String,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
    /// Set when the item was soft-deleted and moved to the Aicut Trash (file kept on disk).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub deleted_at: Option<DateTime<Utc>>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, JsonSchema)]
pub struct MediaListFilter {
    #[serde(default)]
    pub query: Option<String>,
    #[serde(default)]
    pub kind: Option<MediaKind>,
    #[serde(default)]
    pub source_type: Option<MediaSourceType>,
    #[serde(default)]
    pub parent_media_id: Option<String>,
    #[serde(default)]
    pub tag: Option<String>,
    #[serde(default)]
    pub sort_by: Option<String>,
    #[serde(default)]
    pub limit: Option<usize>,
    #[serde(default)]
    pub offset: Option<usize>,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct DownloadPreset {
    pub id: String,
    pub name: String,
    pub description: String,
    pub config: Value,
    pub created_at: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct TranscriptWord {
    pub start_ms: u64,
    pub end_ms: u64,
    pub text: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub probability: Option<f32>,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct TranscriptSegment {
    pub index: usize,
    pub start_ms: u64,
    pub end_ms: u64,
    pub text: String,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub words: Vec<TranscriptWord>,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct TranscriptRecord {
    pub id: String,
    pub media_id: String,
    pub media_title: String,
    pub model_id: String,
    pub language: String,
    pub duration_sec: Option<f64>,
    pub full_text: String,
    pub segments: Vec<TranscriptSegment>,
    pub created_at: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct ResolvedTranscript {
    pub transcript: TranscriptRecord,
    /// Where the transcript was found: "db_exact", "db_related_media", "db_video_key",
    /// "db_asset_ref", "library_json_file", "library_subtitle_file", or "sidecar_file".
    pub source: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source_asset_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source_file_path: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct SmartClipSegment {
    pub start_sec: f64,
    pub end_sec: f64,
    #[serde(default)]
    pub role: String,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, JsonSchema)]
pub struct ClipNarrativeArc {
    #[serde(default)]
    pub beginning: String,
    #[serde(default)]
    pub middle: String,
    #[serde(default)]
    pub end: String,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, JsonSchema)]
pub struct ClipScores {
    #[serde(default)]
    pub match_description: u8,
    #[serde(default)]
    pub visual_interest: u8,
    #[serde(default)]
    pub narrative_coherence: u8,
    #[serde(default)]
    pub hook_strength: u8,
}

impl ClipScores {
    pub fn overall(&self) -> f32 {
        (self.match_description as f32 * 0.35)
            + (self.narrative_coherence as f32 * 0.30)
            + (self.visual_interest as f32 * 0.20)
            + (self.hook_strength as f32 * 0.15)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct SmartClipPlan {
    pub id: String,
    pub title: String,
    #[serde(default)]
    pub hook_first_5s: String,
    #[serde(default)]
    pub visual_evidence: String,
    #[serde(default)]
    pub why_interesting: String,
    #[serde(default)]
    pub narrative_arc: ClipNarrativeArc,
    pub segments: Vec<SmartClipSegment>,
    pub total_duration_sec: f64,
    #[serde(default)]
    pub scores: ClipScores,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rendered_media_id: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct SmartClipRun {
    pub id: String,
    pub media_id: String,
    pub media_title: String,
    pub provider: String,
    pub model_id: String,
    pub description: String,
    pub max_duration_sec: f64,
    pub clips: Vec<SmartClipPlan>,
    pub created_at: DateTime<Utc>,
}

pub struct MediaRepository {
    conn: Mutex<Connection>,
}

impl MediaRepository {
    pub fn open(db_path: &Path) -> AicutResult<Self> {
        let conn = Connection::open(db_path)?;
        conn.execute_batch(
            "
            PRAGMA journal_mode = WAL;
            PRAGMA foreign_keys = ON;

            CREATE TABLE IF NOT EXISTS media_assets (
                id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                kind TEXT NOT NULL,
                source_type TEXT NOT NULL,
                source_url TEXT,
                parent_media_id TEXT,
                file_path TEXT NOT NULL,
                file_name TEXT NOT NULL,
                extension TEXT NOT NULL,
                file_size_bytes INTEGER NOT NULL,
                fingerprint TEXT,
                thumbnail_path TEXT,
                metadata_json TEXT NOT NULL,
                tags_json TEXT NOT NULL,
                notes TEXT NOT NULL DEFAULT '',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_media_created_at ON media_assets(created_at DESC);
            CREATE INDEX IF NOT EXISTS idx_media_fingerprint ON media_assets(fingerprint);

            CREATE TABLE IF NOT EXISTS transcripts (
                id TEXT PRIMARY KEY,
                media_id TEXT NOT NULL,
                media_title TEXT NOT NULL,
                model_id TEXT NOT NULL,
                language TEXT NOT NULL,
                duration_sec REAL,
                full_text TEXT NOT NULL,
                segments_json TEXT NOT NULL,
                created_at TEXT NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_transcripts_media_id ON transcripts(media_id);

            CREATE TABLE IF NOT EXISTS download_presets (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL UNIQUE,
                description TEXT NOT NULL DEFAULT '',
                config_json TEXT NOT NULL,
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS module_states (
                module_id TEXT PRIMARY KEY,
                enabled INTEGER NOT NULL DEFAULT 1,
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS kv_settings (
                key TEXT PRIMARY KEY,
                value_json TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS smart_clip_runs (
                id TEXT PRIMARY KEY,
                media_id TEXT NOT NULL,
                media_title TEXT NOT NULL,
                provider TEXT NOT NULL,
                model_id TEXT NOT NULL,
                description TEXT NOT NULL,
                max_duration_sec REAL NOT NULL,
                clips_json TEXT NOT NULL,
                created_at TEXT NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_smart_clip_runs_media_id ON smart_clip_runs(media_id);
            ",
        )?;

        // Safe upgrade migrations for older databases
        let _ = conn.execute(
            "ALTER TABLE media_assets ADD COLUMN parent_media_id TEXT",
            [],
        );
        let _ = conn.execute(
            "ALTER TABLE media_assets ADD COLUMN deleted_at TEXT",
            [],
        );

        let repo = Self {
            conn: Mutex::new(conn),
        };
        repo.seed_default_presets()?;
        Ok(repo)
    }

    fn seed_default_presets(&self) -> AicutResult<()> {
        let existing = self.list_presets()?;
        if !existing.is_empty() {
            return Ok(());
        }
        let defaults = vec![
            (
                "preset-best-mp4",
                "Melhor Qualidade (MP4)",
                "Vídeo + áudio na melhor qualidade disponível em MP4",
                serde_json::json!({
                    "mode": "video_audio",
                    "quality": "best",
                    "video_codec": "h264",
                    "container": "mp4",
                    "embed_metadata": true,
                    "embed_chapters": true,
                    "concurrent_fragments": 4
                }),
            ),
            (
                "preset-1080p-fast",
                "Full HD 1080p Equilibrado",
                "1080p H.264 compatível com qualquer editor de vídeo",
                serde_json::json!({
                    "mode": "video_audio",
                    "quality": "1080p",
                    "video_codec": "h264",
                    "container": "mp4",
                    "embed_metadata": true,
                    "concurrent_fragments": 4
                }),
            ),
            (
                "preset-video-only-4k",
                "Só Vídeo (Sem Áudio / B-Roll)",
                "Baixa apenas a trilha de vídeo na melhor resolução",
                serde_json::json!({
                    "mode": "video_only",
                    "quality": "best",
                    "video_codec": "auto",
                    "container": "mp4",
                    "concurrent_fragments": 4
                }),
            ),
            (
                "preset-audio-mp3",
                "Só Áudio (MP3 320k)",
                "Extrai apenas o áudio em MP3 alta qualidade com capa",
                serde_json::json!({
                    "mode": "audio_only",
                    "quality": "best",
                    "container": "mp3",
                    "audio_quality": "320k",
                    "embed_thumbnail": true,
                    "embed_metadata": true
                }),
            ),
        ];

        for (id, name, desc, cfg) in defaults {
            let preset = DownloadPreset {
                id: id.to_string(),
                name: name.to_string(),
                description: desc.to_string(),
                config: cfg,
                created_at: Utc::now(),
            };
            self.save_preset(&preset)?;
        }
        Ok(())
    }

    pub fn insert_media(&self, asset: &MediaAsset) -> AicutResult<()> {
        let conn = self
            .conn
            .lock()
            .map_err(|_| AicutError::internal("DB lock poisoned"))?;
        conn.execute(
            "INSERT INTO media_assets (
                id, title, kind, source_type, source_url, parent_media_id, file_path, file_name,
                extension, file_size_bytes, fingerprint, thumbnail_path, metadata_json, tags_json,
                notes, created_at, updated_at
            ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14, ?15, ?16, ?17)",
            params![
                asset.id,
                asset.title,
                serde_json::to_string(&asset.kind)?.trim_matches('"'),
                serde_json::to_string(&asset.source_type)?.trim_matches('"'),
                asset.source_url,
                asset.parent_media_id,
                asset.file_path,
                asset.file_name,
                asset.extension,
                asset.file_size_bytes as i64,
                asset.fingerprint,
                asset.thumbnail_path,
                serde_json::to_string(&asset.metadata)?,
                serde_json::to_string(&asset.tags)?,
                asset.notes,
                asset.created_at.to_rfc3339(),
                asset.updated_at.to_rfc3339(),
            ],
        )?;
        Ok(())
    }

    pub fn get_media(&self, id: &str) -> AicutResult<Option<MediaAsset>> {
        let conn = self
            .conn
            .lock()
            .map_err(|_| AicutError::internal("DB lock poisoned"))?;
        let row = conn
            .query_row(
                "SELECT id, title, kind, source_type, source_url, parent_media_id, file_path,
                        file_name, extension, file_size_bytes, fingerprint, thumbnail_path,
                        metadata_json, tags_json, notes, created_at, updated_at, deleted_at
                 FROM media_assets WHERE id = ?1",
                params![id],
                row_to_media,
            )
            .optional()?;
        Ok(row)
    }

    pub fn find_by_fingerprint(&self, fingerprint: &str) -> AicutResult<Option<MediaAsset>> {
        let conn = self
            .conn
            .lock()
            .map_err(|_| AicutError::internal("DB lock poisoned"))?;
        let row = conn
            .query_row(
                "SELECT id, title, kind, source_type, source_url, parent_media_id, file_path,
                        file_name, extension, file_size_bytes, fingerprint, thumbnail_path,
                        metadata_json, tags_json, notes, created_at, updated_at, deleted_at
                 FROM media_assets WHERE fingerprint = ?1 AND deleted_at IS NULL LIMIT 1",
                params![fingerprint],
                row_to_media,
            )
            .optional()?;
        Ok(row)
    }

    pub fn list_media(&self, filter: &MediaListFilter) -> AicutResult<Vec<MediaAsset>> {
        let conn = self
            .conn
            .lock()
            .map_err(|_| AicutError::internal("DB lock poisoned"))?;

        let mut stmt = conn.prepare(
            "SELECT id, title, kind, source_type, source_url, parent_media_id, file_path,
                    file_name, extension, file_size_bytes, fingerprint, thumbnail_path,
                    metadata_json, tags_json, notes, created_at, updated_at, deleted_at
             FROM media_assets
             WHERE deleted_at IS NULL
             ORDER BY created_at DESC",
        )?;

        let rows = stmt
            .query_map([], row_to_media)?
            .collect::<Result<Vec<_>, _>>()?;

        let q = filter.query.as_ref().map(|s| s.to_lowercase());
        let tag_filter = filter.tag.as_ref().map(|s| s.to_lowercase());

        let mut filtered: Vec<MediaAsset> = rows
            .into_iter()
            .filter(|m| {
                if let Some(kind) = filter.kind {
                    if m.kind != kind {
                        return false;
                    }
                }
                if let Some(source) = filter.source_type {
                    if m.source_type != source {
                        return false;
                    }
                }
                if let Some(ref pid) = filter.parent_media_id {
                    if m.parent_media_id.as_deref() != Some(pid.as_str()) {
                        return false;
                    }
                }
                if let Some(ref tag) = tag_filter {
                    if !m.tags.iter().any(|t| t.to_lowercase() == *tag) {
                        return false;
                    }
                }
                if let Some(ref query) = q {
                    let in_title = m.title.to_lowercase().contains(query);
                    let in_file = m.file_name.to_lowercase().contains(query);
                    let in_url = m
                        .source_url
                        .as_ref()
                        .map(|u| u.to_lowercase().contains(query))
                        .unwrap_or(false);
                    let in_uploader = m
                        .metadata
                        .uploader
                        .as_ref()
                        .map(|u| u.to_lowercase().contains(query))
                        .unwrap_or(false);
                    let in_tags = m.tags.iter().any(|t| t.to_lowercase().contains(query));
                    let in_notes = m.notes.to_lowercase().contains(query);
                    if !(in_title || in_file || in_url || in_uploader || in_tags || in_notes) {
                        return false;
                    }
                }
                true
            })
            .collect();

        match filter.sort_by.as_deref() {
            Some("title_asc") => filtered.sort_by(|a, b| a.title.to_lowercase().cmp(&b.title.to_lowercase())),
            Some("title_desc") => filtered.sort_by(|a, b| b.title.to_lowercase().cmp(&a.title.to_lowercase())),
            Some("size_desc") => filtered.sort_by(|a, b| b.file_size_bytes.cmp(&a.file_size_bytes)),
            Some("duration_desc") => filtered.sort_by(|a, b| {
                b.metadata
                    .duration_sec
                    .unwrap_or(0.0)
                    .partial_cmp(&a.metadata.duration_sec.unwrap_or(0.0))
                    .unwrap_or(std::cmp::Ordering::Equal)
            }),
            Some("oldest") => filtered.sort_by(|a, b| a.created_at.cmp(&b.created_at)),
            _ => filtered.sort_by(|a, b| b.created_at.cmp(&a.created_at)),
        }

        let offset = filter.offset.unwrap_or(0);
        let limit = filter.limit.unwrap_or(500);
        Ok(filtered.into_iter().skip(offset).take(limit).collect())
    }

    pub fn update_media(
        &self,
        id: &str,
        title: Option<String>,
        tags: Option<Vec<String>>,
        notes: Option<String>,
    ) -> AicutResult<MediaAsset> {
        let mut asset = self
            .get_media(id)?
            .ok_or_else(|| AicutError::not_found(format!("Media '{id}' not found")))?;
        if let Some(t) = title {
            asset.title = t;
        }
        if let Some(tg) = tags {
            asset.tags = tg;
        }
        if let Some(n) = notes {
            asset.notes = n;
        }
        asset.updated_at = Utc::now();

        let conn = self
            .conn
            .lock()
            .map_err(|_| AicutError::internal("DB lock poisoned"))?;
        conn.execute(
            "UPDATE media_assets SET title = ?1, tags_json = ?2, notes = ?3, updated_at = ?4 WHERE id = ?5",
            params![
                asset.title,
                serde_json::to_string(&asset.tags)?,
                asset.notes,
                asset.updated_at.to_rfc3339(),
                asset.id,
            ],
        )?;
        Ok(asset)
    }

    /// Soft-deletes a media asset by moving it into the Aicut Trash (`deleted_at = now`).
    pub fn trash_media(&self, id: &str) -> AicutResult<Option<MediaAsset>> {
        let existing = self.get_media(id)?;
        if let Some(mut asset) = existing {
            let now = Utc::now();
            let conn = self
                .conn
                .lock()
                .map_err(|_| AicutError::internal("DB lock poisoned"))?;
            conn.execute(
                "UPDATE media_assets SET deleted_at = ?1, updated_at = ?1 WHERE id = ?2",
                params![now.to_rfc3339(), id],
            )?;
            asset.deleted_at = Some(now);
            asset.updated_at = now;
            return Ok(Some(asset));
        }
        Ok(None)
    }

    /// Restores a soft-deleted media asset from the Trash back into the active Library.
    pub fn restore_media(&self, id: &str) -> AicutResult<Option<MediaAsset>> {
        let existing = self.get_media(id)?;
        if let Some(mut asset) = existing {
            let now = Utc::now();
            let conn = self
                .conn
                .lock()
                .map_err(|_| AicutError::internal("DB lock poisoned"))?;
            conn.execute(
                "UPDATE media_assets SET deleted_at = NULL, updated_at = ?1 WHERE id = ?2",
                params![now.to_rfc3339(), id],
            )?;
            asset.deleted_at = None;
            asset.updated_at = now;
            return Ok(Some(asset));
        }
        Ok(None)
    }

    /// Lists all items currently in the Aicut Trash (newest trashed first).
    pub fn list_trashed_media(&self) -> AicutResult<Vec<MediaAsset>> {
        let conn = self
            .conn
            .lock()
            .map_err(|_| AicutError::internal("DB lock poisoned"))?;
        let mut stmt = conn.prepare(
            "SELECT id, title, kind, source_type, source_url, parent_media_id, file_path,
                    file_name, extension, file_size_bytes, fingerprint, thumbnail_path,
                    metadata_json, tags_json, notes, created_at, updated_at, deleted_at
             FROM media_assets
             WHERE deleted_at IS NOT NULL
             ORDER BY deleted_at DESC",
        )?;
        let rows = stmt
            .query_map([], row_to_media)?
            .collect::<Result<Vec<_>, _>>()?;
        Ok(rows)
    }

    /// Permanently deletes a media record from SQLite.
    pub fn delete_media(&self, id: &str) -> AicutResult<Option<MediaAsset>> {
        let existing = self.get_media(id)?;
        if existing.is_some() {
            let conn = self
                .conn
                .lock()
                .map_err(|_| AicutError::internal("DB lock poisoned"))?;
            conn.execute("DELETE FROM media_assets WHERE id = ?1", params![id])?;
        }
        Ok(existing)
    }

    pub fn count_media(&self) -> AicutResult<usize> {
        let conn = self
            .conn
            .lock()
            .map_err(|_| AicutError::internal("DB lock poisoned"))?;
        let count: i64 = conn.query_row(
            "SELECT COUNT(*) FROM media_assets WHERE deleted_at IS NULL",
            [],
            |r| r.get(0),
        )?;
        Ok(count as usize)
    }

    // ---- Transcripts ----

    pub fn insert_transcript(&self, rec: &TranscriptRecord) -> AicutResult<()> {
        let conn = self
            .conn
            .lock()
            .map_err(|_| AicutError::internal("DB lock poisoned"))?;
        conn.execute(
            "INSERT INTO transcripts (
                id, media_id, media_title, model_id, language, duration_sec, full_text,
                segments_json, created_at
            ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9)",
            params![
                rec.id,
                rec.media_id,
                rec.media_title,
                rec.model_id,
                rec.language,
                rec.duration_sec,
                rec.full_text,
                serde_json::to_string(&rec.segments)?,
                rec.created_at.to_rfc3339(),
            ],
        )?;
        Ok(())
    }

    pub fn get_transcript(&self, id: &str) -> AicutResult<Option<TranscriptRecord>> {
        let conn = self
            .conn
            .lock()
            .map_err(|_| AicutError::internal("DB lock poisoned"))?;
        let row = conn
            .query_row(
                "SELECT id, media_id, media_title, model_id, language, duration_sec, full_text,
                        segments_json, created_at
                 FROM transcripts WHERE id = ?1",
                params![id],
                row_to_transcript,
            )
            .optional()?;
        Ok(row)
    }

    pub fn list_transcripts(&self, media_id: Option<&str>) -> AicutResult<Vec<TranscriptRecord>> {
        let conn = self
            .conn
            .lock()
            .map_err(|_| AicutError::internal("DB lock poisoned"))?;
        let items = if let Some(mid) = media_id {
            let mut stmt = conn.prepare(
                "SELECT id, media_id, media_title, model_id, language, duration_sec, full_text,
                        segments_json, created_at
                 FROM transcripts WHERE media_id = ?1 ORDER BY created_at DESC",
            )?;
            let rows = stmt
                .query_map(params![mid], row_to_transcript)?
                .collect::<Result<Vec<_>, _>>()?;
            rows
        } else {
            let mut stmt = conn.prepare(
                "SELECT id, media_id, media_title, model_id, language, duration_sec, full_text,
                        segments_json, created_at
                 FROM transcripts ORDER BY created_at DESC",
            )?;
            let rows = stmt
                .query_map([], row_to_transcript)?
                .collect::<Result<Vec<_>, _>>()?;
            rows
        };
        Ok(items)
    }

    pub fn delete_transcript(&self, id: &str) -> AicutResult<bool> {
        let conn = self
            .conn
            .lock()
            .map_err(|_| AicutError::internal("DB lock poisoned"))?;
        let changed = conn.execute("DELETE FROM transcripts WHERE id = ?1", params![id])?;
        Ok(changed > 0)
    }

    // ---- Unified Video Lineage & Full-Library Context Resolution ----

    /// Finds every Library asset linked to `media_id` via direct ID, `parent_media_id`,
    /// `video_id:<id>` tag, shared `source_url`, shared `fingerprint`, or canonical external
    /// video ID (such as YouTube `[fPQDxLwScrA]` or Twitch VOD ID).
    pub fn find_related_media(&self, media_id: &str) -> AicutResult<Vec<MediaAsset>> {
        let Some(target) = self.get_media(media_id)? else {
            return Ok(Vec::new());
        };
        let all = self.list_media(&MediaListFilter {
            limit: Some(5000),
            ..Default::default()
        })?;

        let root_id = target
            .parent_media_id
            .clone()
            .unwrap_or_else(|| target.id.clone());
        let root_asset = all.iter().find(|a| a.id == root_id).unwrap_or(&target);

        let target_url = normalize_url_opt(root_asset.source_url.as_deref())
            .or_else(|| normalize_url_opt(target.source_url.as_deref()));
        let target_fp = root_asset
            .fingerprint
            .as_deref()
            .or(target.fingerprint.as_deref())
            .map(str::to_string);
        let target_ext_key = extract_external_video_key(
            root_asset.source_url.as_deref().or(target.source_url.as_deref()),
            &root_asset.title,
            &root_asset.file_name,
        )
        .or_else(|| {
            extract_external_video_key(
                target.source_url.as_deref(),
                &target.title,
                &target.file_name,
            )
        });

        let video_tag_target = format!("video_id:{}", target.id);
        let video_tag_root = format!("video_id:{root_id}");

        let mut related: Vec<MediaAsset> = Vec::new();
        for item in all {
            let is_direct = item.id == target.id
                || item.id == root_id
                || item.parent_media_id.as_deref() == Some(target.id.as_str())
                || item.parent_media_id.as_deref() == Some(root_id.as_str())
                || item
                    .tags
                    .iter()
                    .any(|t| t == &video_tag_target || t == &video_tag_root);

            let same_url = target_url.is_some()
                && normalize_url_opt(item.source_url.as_deref()) == target_url;

            let same_fp = target_fp.is_some()
                && item.fingerprint.as_deref() == target_fp.as_deref();

            let same_ext_key = target_ext_key.is_some()
                && extract_external_video_key(
                    item.source_url.as_deref(),
                    &item.title,
                    &item.file_name,
                ) == target_ext_key;

            if is_direct || same_url || same_fp || same_ext_key {
                related.push(item);
            }
        }

        // Second pass: include children of any matched sibling asset
        let matched_ids: std::collections::HashSet<String> =
            related.iter().map(|a| a.id.clone()).collect();
        let all_again = self.list_media(&MediaListFilter {
            limit: Some(5000),
            ..Default::default()
        })?;
        for item in all_again {
            if matched_ids.contains(&item.id) {
                continue;
            }
            if let Some(ref pid) = item.parent_media_id {
                if matched_ids.contains(pid) {
                    related.push(item);
                }
            }
        }

        Ok(related)
    }

    /// Searches the entire Library & transcripts DB to find the best transcript for `media_id`:
    /// 1. Exact DB transcript by `media_id`
    /// 2. DB transcript linked to parent/sibling media (same video ID, URL, or fingerprint)
    /// 3. DB transcript whose title contains the same external video ID (e.g. `[fPQDxLwScrA]`)
    /// 4. Referenced via `metadata.transcript_id` on any related Library asset
    /// 5. Related `.json` (`TranscriptRecord`), `.srt`, or `.vtt` file registered in the Library
    /// 6. Sidecar `.srt`/`.vtt`/`.json` file sitting alongside the video on disk
    pub fn find_best_transcript_for_media(
        &self,
        media_id: &str,
    ) -> AicutResult<Option<ResolvedTranscript>> {
        // 1. Exact DB match
        if let Some(tr) = self.list_transcripts(Some(media_id))?.into_iter().next() {
            return Ok(Some(ResolvedTranscript {
                transcript: tr,
                source: "db_exact".to_string(),
                source_asset_id: Some(media_id.to_string()),
                source_file_path: None,
            }));
        }

        let Some(target) = self.get_media(media_id)? else {
            return Ok(None);
        };
        let related = self.find_related_media(media_id)?;

        // 2. DB transcript linked to any related asset ID (parent or sibling download)
        for asset in &related {
            if asset.id == media_id {
                continue;
            }
            if let Some(tr) = self.list_transcripts(Some(&asset.id))?.into_iter().next() {
                return Ok(Some(ResolvedTranscript {
                    transcript: tr,
                    source: "db_related_media".to_string(),
                    source_asset_id: Some(asset.id.clone()),
                    source_file_path: None,
                }));
            }
        }

        // 3. DB transcript referenced via metadata.transcript_id on any related Library asset
        for asset in &related {
            if let Some(ref tid) = asset.metadata.transcript_id {
                if let Some(tr) = self.get_transcript(tid)? {
                    return Ok(Some(ResolvedTranscript {
                        transcript: tr,
                        source: "db_asset_ref".to_string(),
                        source_asset_id: Some(asset.id.clone()),
                        source_file_path: Some(asset.file_path.clone()),
                    }));
                }
            }
        }

        // 4. DB transcript matching canonical external video key (e.g. YouTube [fPQDxLwScrA])
        let target_ext_key = extract_external_video_key(
            target.source_url.as_deref(),
            &target.title,
            &target.file_name,
        );
        if let Some(ref key) = target_ext_key {
            let all_tr = self.list_transcripts(None)?;
            if let Some(tr) = all_tr.into_iter().find(|t| {
                extract_external_video_key(None, &t.media_title, &t.media_title).as_deref()
                    == Some(key.as_str())
            }) {
                return Ok(Some(ResolvedTranscript {
                    transcript: tr,
                    source: "db_video_key".to_string(),
                    source_asset_id: None,
                    source_file_path: None,
                }));
            }
        }

        // 5. Parse related Library JSON transcript file (preserves word-level timestamps!)
        for asset in &related {
            let ext = asset.extension.to_lowercase();
            if ext == "json" && Path::new(&asset.file_path).is_file() {
                if let Ok(raw) = std::fs::read_to_string(&asset.file_path) {
                    if let Ok(tr) = serde_json::from_str::<TranscriptRecord>(&raw) {
                        if !tr.segments.is_empty() {
                            return Ok(Some(ResolvedTranscript {
                                transcript: tr,
                                source: "library_json_file".to_string(),
                                source_asset_id: Some(asset.id.clone()),
                                source_file_path: Some(asset.file_path.clone()),
                            }));
                        }
                    }
                }
            }
        }

        // 6. Parse related Library SRT / VTT subtitle file
        for asset in &related {
            let ext = asset.extension.to_lowercase();
            if matches!(ext.as_str(), "srt" | "vtt") && Path::new(&asset.file_path).is_file() {
                if let Ok(raw) = std::fs::read_to_string(&asset.file_path) {
                    if let Some(tr) = parse_subtitle_to_transcript(&raw, &target, &asset.id) {
                        return Ok(Some(ResolvedTranscript {
                            transcript: tr,
                            source: "library_subtitle_file".to_string(),
                            source_asset_id: Some(asset.id.clone()),
                            source_file_path: Some(asset.file_path.clone()),
                        }));
                    }
                }
            }
        }

        // 7. Sidecar subtitle/json sitting next to the video file on disk
        let video_path = Path::new(&target.file_path);
        if let (Some(parent), Some(stem)) = (video_path.parent(), video_path.file_stem().and_then(|s| s.to_str())) {
            if let Ok(entries) = std::fs::read_dir(parent) {
                for entry in entries.flatten() {
                    let p = entry.path();
                    if !p.is_file() {
                        continue;
                    }
                    let fname = p.file_name().and_then(|n| n.to_str()).unwrap_or("");
                    if !fname.starts_with(stem) {
                        continue;
                    }
                    let ext = p.extension().and_then(|e| e.to_str()).unwrap_or("").to_lowercase();
                    if ext == "json" {
                        if let Ok(raw) = std::fs::read_to_string(&p) {
                            if let Ok(tr) = serde_json::from_str::<TranscriptRecord>(&raw) {
                                if !tr.segments.is_empty() {
                                    return Ok(Some(ResolvedTranscript {
                                        transcript: tr,
                                        source: "sidecar_file".to_string(),
                                        source_asset_id: None,
                                        source_file_path: Some(p.to_string_lossy().to_string()),
                                    }));
                                }
                            }
                        }
                    } else if matches!(ext.as_str(), "srt" | "vtt") {
                        if let Ok(raw) = std::fs::read_to_string(&p) {
                            if let Some(tr) = parse_subtitle_to_transcript(&raw, &target, "sidecar") {
                                return Ok(Some(ResolvedTranscript {
                                    transcript: tr,
                                    source: "sidecar_file".to_string(),
                                    source_asset_id: None,
                                    source_file_path: Some(p.to_string_lossy().to_string()),
                                }));
                            }
                        }
                    }
                }
            }
        }

        Ok(None)
    }

    // ---- Smart Clip Runs ----

    pub fn insert_clip_run(&self, run: &SmartClipRun) -> AicutResult<()> {
        let conn = self
            .conn
            .lock()
            .map_err(|_| AicutError::internal("DB lock poisoned"))?;
        conn.execute(
            "INSERT INTO smart_clip_runs (
                id, media_id, media_title, provider, model_id, description,
                max_duration_sec, clips_json, created_at
            ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9)
            ON CONFLICT(id) DO UPDATE SET
                clips_json = excluded.clips_json",
            params![
                run.id,
                run.media_id,
                run.media_title,
                run.provider,
                run.model_id,
                run.description,
                run.max_duration_sec,
                serde_json::to_string(&run.clips)?,
                run.created_at.to_rfc3339(),
            ],
        )?;
        Ok(())
    }

    pub fn get_clip_run(&self, id: &str) -> AicutResult<Option<SmartClipRun>> {
        let conn = self
            .conn
            .lock()
            .map_err(|_| AicutError::internal("DB lock poisoned"))?;
        let item = conn
            .query_row(
                "SELECT id, media_id, media_title, provider, model_id, description,
                        max_duration_sec, clips_json, created_at
                 FROM smart_clip_runs WHERE id = ?1",
                params![id],
                row_to_clip_run,
            )
            .optional()?;
        Ok(item)
    }

    pub fn list_clip_runs(&self, media_id: Option<&str>) -> AicutResult<Vec<SmartClipRun>> {
        let conn = self
            .conn
            .lock()
            .map_err(|_| AicutError::internal("DB lock poisoned"))?;
        let items = if let Some(mid) = media_id {
            let mut stmt = conn.prepare(
                "SELECT id, media_id, media_title, provider, model_id, description,
                        max_duration_sec, clips_json, created_at
                 FROM smart_clip_runs WHERE media_id = ?1 ORDER BY created_at DESC",
            )?;
            let rows = stmt
                .query_map(params![mid], row_to_clip_run)?
                .collect::<Result<Vec<_>, _>>()?;
            rows
        } else {
            let mut stmt = conn.prepare(
                "SELECT id, media_id, media_title, provider, model_id, description,
                        max_duration_sec, clips_json, created_at
                 FROM smart_clip_runs ORDER BY created_at DESC",
            )?;
            let rows = stmt
                .query_map([], row_to_clip_run)?
                .collect::<Result<Vec<_>, _>>()?;
            rows
        };
        Ok(items)
    }

    pub fn delete_clip_run(&self, id: &str) -> AicutResult<bool> {
        let conn = self
            .conn
            .lock()
            .map_err(|_| AicutError::internal("DB lock poisoned"))?;
        let changed = conn.execute("DELETE FROM smart_clip_runs WHERE id = ?1", params![id])?;
        Ok(changed > 0)
    }

    // ---- Download Presets ----

    pub fn list_presets(&self) -> AicutResult<Vec<DownloadPreset>> {
        let conn = self
            .conn
            .lock()
            .map_err(|_| AicutError::internal("DB lock poisoned"))?;
        let mut stmt = conn.prepare(
            "SELECT id, name, description, config_json, created_at FROM download_presets ORDER BY created_at ASC",
        )?;
        let items = stmt
            .query_map([], |row| {
                let config_str: String = row.get(3)?;
                let created_str: String = row.get(4)?;
                Ok(DownloadPreset {
                    id: row.get(0)?,
                    name: row.get(1)?,
                    description: row.get(2)?,
                    config: serde_json::from_str(&config_str).unwrap_or(Value::Null),
                    created_at: DateTime::parse_from_rfc3339(&created_str)
                        .map(|d| d.with_timezone(&Utc))
                        .unwrap_or_else(|_| Utc::now()),
                })
            })?
            .collect::<Result<Vec<_>, _>>()?;
        Ok(items)
    }

    pub fn save_preset(&self, preset: &DownloadPreset) -> AicutResult<()> {
        let conn = self
            .conn
            .lock()
            .map_err(|_| AicutError::internal("DB lock poisoned"))?;
        conn.execute(
            "INSERT INTO download_presets (id, name, description, config_json, created_at)
             VALUES (?1, ?2, ?3, ?4, ?5)
             ON CONFLICT(id) DO UPDATE SET
                name = excluded.name,
                description = excluded.description,
                config_json = excluded.config_json",
            params![
                preset.id,
                preset.name,
                preset.description,
                serde_json::to_string(&preset.config)?,
                preset.created_at.to_rfc3339(),
            ],
        )?;
        Ok(())
    }

    pub fn delete_preset(&self, id: &str) -> AicutResult<bool> {
        let conn = self
            .conn
            .lock()
            .map_err(|_| AicutError::internal("DB lock poisoned"))?;
        let changed = conn.execute("DELETE FROM download_presets WHERE id = ?1", params![id])?;
        Ok(changed > 0)
    }

    // ---- Module States & KV Settings ----

    pub fn get_module_enabled(&self, module_id: &str) -> AicutResult<Option<bool>> {
        let conn = self
            .conn
            .lock()
            .map_err(|_| AicutError::internal("DB lock poisoned"))?;
        let enabled: Option<i64> = conn
            .query_row(
                "SELECT enabled FROM module_states WHERE module_id = ?1",
                params![module_id],
                |r| r.get(0),
            )
            .optional()?;
        Ok(enabled.map(|v| v != 0))
    }

    pub fn set_module_enabled(&self, module_id: &str, enabled: bool) -> AicutResult<()> {
        let conn = self
            .conn
            .lock()
            .map_err(|_| AicutError::internal("DB lock poisoned"))?;
        conn.execute(
            "INSERT INTO module_states (module_id, enabled, updated_at)
             VALUES (?1, ?2, ?3)
             ON CONFLICT(module_id) DO UPDATE SET enabled = excluded.enabled, updated_at = excluded.updated_at",
            params![module_id, if enabled { 1 } else { 0 }, Utc::now().to_rfc3339()],
        )?;
        Ok(())
    }

    pub fn get_setting(&self, key: &str) -> AicutResult<Option<Value>> {
        let conn = self
            .conn
            .lock()
            .map_err(|_| AicutError::internal("DB lock poisoned"))?;
        let raw: Option<String> = conn
            .query_row(
                "SELECT value_json FROM kv_settings WHERE key = ?1",
                params![key],
                |r| r.get(0),
            )
            .optional()?;
        match raw {
            Some(s) => Ok(Some(serde_json::from_str(&s)?)),
            None => Ok(None),
        }
    }

    pub fn set_setting(&self, key: &str, value: &Value) -> AicutResult<()> {
        let conn = self
            .conn
            .lock()
            .map_err(|_| AicutError::internal("DB lock poisoned"))?;
        conn.execute(
            "INSERT INTO kv_settings (key, value_json, updated_at)
             VALUES (?1, ?2, ?3)
             ON CONFLICT(key) DO UPDATE SET value_json = excluded.value_json, updated_at = excluded.updated_at",
            params![key, serde_json::to_string(value)?, Utc::now().to_rfc3339()],
        )?;
        Ok(())
    }

    pub fn list_settings(&self) -> AicutResult<serde_json::Map<String, Value>> {
        let conn = self
            .conn
            .lock()
            .map_err(|_| AicutError::internal("DB lock poisoned"))?;
        let mut stmt = conn.prepare("SELECT key, value_json FROM kv_settings")?;
        let mut map = serde_json::Map::new();
        let rows = stmt.query_map([], |row| {
            let k: String = row.get(0)?;
            let v: String = row.get(1)?;
            Ok((k, v))
        })?;
        for row in rows {
            let (k, v) = row?;
            if let Ok(parsed) = serde_json::from_str(&v) {
                map.insert(k, parsed);
            }
        }
        Ok(map)
    }
}

fn row_to_media(row: &rusqlite::Row<'_>) -> rusqlite::Result<MediaAsset> {
    let kind_str: String = row.get(2)?;
    let source_str: String = row.get(3)?;
    let meta_str: String = row.get(12)?;
    let tags_str: String = row.get(13)?;
    let created_str: String = row.get(15)?;
    let updated_str: String = row.get(16)?;
    let deleted_str: Option<String> = row.get(17).ok().flatten();

    let kind = match kind_str.as_str() {
        "video_only" => MediaKind::VideoOnly,
        "audio" => MediaKind::Audio,
        "subtitle" => MediaKind::Subtitle,
        "timeline" => MediaKind::Timeline,
        "transcript" => MediaKind::Transcript,
        _ => MediaKind::Video,
    };
    let source_type = match source_str.as_str() {
        "download" => MediaSourceType::Download,
        "generated" => MediaSourceType::Generated,
        _ => MediaSourceType::Local,
    };

    Ok(MediaAsset {
        id: row.get(0)?,
        title: row.get(1)?,
        kind,
        source_type,
        source_url: row.get(4)?,
        parent_media_id: row.get(5)?,
        file_path: row.get(6)?,
        file_name: row.get(7)?,
        extension: row.get(8)?,
        file_size_bytes: row.get::<_, i64>(9)? as u64,
        fingerprint: row.get(10)?,
        thumbnail_path: row.get(11)?,
        metadata: serde_json::from_str(&meta_str).unwrap_or_default(),
        tags: serde_json::from_str(&tags_str).unwrap_or_default(),
        notes: row.get(14)?,
        created_at: DateTime::parse_from_rfc3339(&created_str)
            .map(|d| d.with_timezone(&Utc))
            .unwrap_or_else(|_| Utc::now()),
        updated_at: DateTime::parse_from_rfc3339(&updated_str)
            .map(|d| d.with_timezone(&Utc))
            .unwrap_or_else(|_| Utc::now()),
        deleted_at: deleted_str.and_then(|s| {
            DateTime::parse_from_rfc3339(&s)
                .map(|d| d.with_timezone(&Utc))
                .ok()
        }),
    })
}

fn row_to_transcript(row: &rusqlite::Row<'_>) -> rusqlite::Result<TranscriptRecord> {
    let seg_str: String = row.get(7)?;
    let created_str: String = row.get(8)?;
    Ok(TranscriptRecord {
        id: row.get(0)?,
        media_id: row.get(1)?,
        media_title: row.get(2)?,
        model_id: row.get(3)?,
        language: row.get(4)?,
        duration_sec: row.get(5)?,
        full_text: row.get(6)?,
        segments: serde_json::from_str(&seg_str).unwrap_or_default(),
        created_at: DateTime::parse_from_rfc3339(&created_str)
            .map(|d| d.with_timezone(&Utc))
            .unwrap_or_else(|_| Utc::now()),
    })
}

fn row_to_clip_run(row: &rusqlite::Row<'_>) -> rusqlite::Result<SmartClipRun> {
    let clips_str: String = row.get(7)?;
    let created_str: String = row.get(8)?;
    Ok(SmartClipRun {
        id: row.get(0)?,
        media_id: row.get(1)?,
        media_title: row.get(2)?,
        provider: row.get(3)?,
        model_id: row.get(4)?,
        description: row.get(5)?,
        max_duration_sec: row.get(6)?,
        clips: serde_json::from_str(&clips_str).unwrap_or_default(),
        created_at: DateTime::parse_from_rfc3339(&created_str)
            .map(|d| d.with_timezone(&Utc))
            .unwrap_or_else(|_| Utc::now()),
    })
}

fn normalize_url_opt(url: Option<&str>) -> Option<String> {
    let raw = url?.trim();
    if raw.is_empty() {
        return None;
    }
    Some(raw.trim_end_matches('/').to_lowercase())
}

/// Extracts a canonical external video identifier (e.g. `yt:fPQDxLwScrA` or `tw:123456789`)
/// from a URL, title, or filename so all assets generated from the same video share identity.
pub fn extract_external_video_key(
    source_url: Option<&str>,
    title: &str,
    file_name: &str,
) -> Option<String> {
    if let Some(url) = source_url {
        if let Some(yt_id) = extract_youtube_id_from_url(url) {
            return Some(format!("yt:{yt_id}"));
        }
        if let Some(tw_id) = extract_twitch_vod_id(url) {
            return Some(format!("tw:{tw_id}"));
        }
    }
    for text in [title, file_name] {
        if let Some(id) = extract_bracketed_id(text) {
            return Some(format!("yt:{id}"));
        }
    }
    None
}

fn extract_youtube_id_from_url(url: &str) -> Option<String> {
    let lower = url.to_lowercase();
    if !lower.contains("youtube.com") && !lower.contains("youtu.be") {
        return None;
    }
    if let Some(idx) = url.find("v=") {
        let candidate: String = url[idx + 2..]
            .chars()
            .take_while(|c| c.is_ascii_alphanumeric() || *c == '-' || *c == '_')
            .collect();
        if candidate.len() == 11 {
            return Some(candidate);
        }
    }
    for marker in ["youtu.be/", "/shorts/", "/live/", "/embed/"] {
        if let Some(idx) = lower.find(marker) {
            let start = idx + marker.len();
            let candidate: String = url[start..]
                .chars()
                .take_while(|c| c.is_ascii_alphanumeric() || *c == '-' || *c == '_')
                .collect();
            if candidate.len() == 11 {
                return Some(candidate);
            }
        }
    }
    None
}

fn extract_twitch_vod_id(url: &str) -> Option<String> {
    let lower = url.to_lowercase();
    let idx = lower.find("twitch.tv/videos/")?;
    let digits: String = url[idx + "twitch.tv/videos/".len()..]
        .chars()
        .take_while(|c| c.is_ascii_digit())
        .collect();
    if digits.len() >= 6 {
        Some(digits)
    } else {
        None
    }
}

fn extract_bracketed_id(text: &str) -> Option<String> {
    let mut search = text;
    while let Some(open) = search.find('[') {
        let rest = &search[open + 1..];
        if let Some(close) = rest.find(']') {
            let inside = &rest[..close];
            if inside.len() == 11
                && inside
                    .chars()
                    .all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_')
            {
                return Some(inside.to_string());
            }
            search = &rest[close + 1..];
        } else {
            break;
        }
    }
    None
}

fn parse_subtitle_to_transcript(
    raw: &str,
    target: &MediaAsset,
    source_tag: &str,
) -> Option<TranscriptRecord> {
    let mut segments = Vec::new();
    let mut full_parts = Vec::new();
    let normalized = raw.replace("\r\n", "\n");
    let lines: Vec<&str> = normalized.lines().collect();
    let mut i = 0;

    while i < lines.len() {
        let line = lines[i].trim();
        if let Some((start_str, end_str)) = line.split_once("-->") {
            let start_ms = parse_sub_timestamp_ms(start_str.trim());
            let end_ms = parse_sub_timestamp_ms(
                end_str
                    .trim()
                    .split_whitespace()
                    .next()
                    .unwrap_or(end_str.trim()),
            );
            i += 1;
            let mut text_lines = Vec::new();
            while i < lines.len() && !lines[i].trim().is_empty() {
                let cleaned = strip_html_tags(lines[i].trim());
                if !cleaned.is_empty() {
                    text_lines.push(cleaned);
                }
                i += 1;
            }
            if let (Some(s), Some(e)) = (start_ms, end_ms) {
                let text = text_lines.join(" ").trim().to_string();
                if !text.is_empty() && e > s {
                    full_parts.push(text.clone());
                    segments.push(TranscriptSegment {
                        index: segments.len(),
                        start_ms: s,
                        end_ms: e,
                        text,
                        words: Vec::new(),
                    });
                }
            }
        } else {
            i += 1;
        }
    }

    if segments.is_empty() {
        return None;
    }

    Some(TranscriptRecord {
        id: format!("parsed-{source_tag}"),
        media_id: target.id.clone(),
        media_title: target.title.clone(),
        model_id: "library-subtitle".to_string(),
        language: target
            .metadata
            .language
            .clone()
            .unwrap_or_else(|| "auto".to_string()),
        duration_sec: target.metadata.duration_sec,
        full_text: full_parts.join(" "),
        segments,
        created_at: Utc::now(),
    })
}

fn parse_sub_timestamp_ms(ts: &str) -> Option<u64> {
    let clean = ts.replace(',', ".");
    let parts: Vec<&str> = clean.split(':').collect();
    let (h, m, s_part) = match parts.as_slice() {
        [m, s] => (0u64, m.parse::<u64>().ok()?, *s),
        [h, m, s] => (h.parse::<u64>().ok()?, m.parse::<u64>().ok()?, *s),
        _ => return None,
    };
    let (sec, ms) = if let Some((a, b)) = s_part.split_once('.') {
        let s = a.parse::<u64>().ok()?;
        let mut ms_str = b.chars().take(3).collect::<String>();
        while ms_str.len() < 3 {
            ms_str.push('0');
        }
        (s, ms_str.parse::<u64>().ok()?)
    } else {
        (s_part.parse::<u64>().ok()?, 0)
    };
    Some((h * 3600 + m * 60 + sec) * 1000 + ms)
}

fn strip_html_tags(input: &str) -> String {
    let mut out = String::with_capacity(input.len());
    let mut in_tag = false;
    for ch in input.chars() {
        match ch {
            '<' => in_tag = true,
            '>' => in_tag = false,
            _ if !in_tag => out.push(ch),
            _ => {}
        }
    }
    out.trim().to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extracts_youtube_and_bracketed_video_keys() {
        assert_eq!(
            extract_external_video_key(
                Some("https://www.youtube.com/watch?v=fPQDxLwScrA"),
                "Test",
                "test.mp4"
            )
            .as_deref(),
            Some("yt:fPQDxLwScrA")
        );
        assert_eq!(
            extract_external_video_key(
                None,
                "Instale isso no Claude Code [fPQDxLwScrA]",
                "video.mp4"
            )
            .as_deref(),
            Some("yt:fPQDxLwScrA")
        );
    }

    #[test]
    fn resolves_transcript_across_related_media_and_srt_files() {
        let dir = tempfile::tempdir().unwrap();
        let db = MediaRepository::open(&dir.path().join("test.db")).unwrap();

        let now = Utc::now();
        let video_asset = MediaAsset {
            id: "vid-1".into(),
            title: "Meu Video [fPQDxLwScrA]".into(),
            kind: MediaKind::Video,
            source_type: MediaSourceType::Download,
            source_url: Some("https://www.youtube.com/watch?v=fPQDxLwScrA".into()),
            parent_media_id: None,
            file_path: dir.path().join("video.mp4").to_string_lossy().to_string(),
            file_name: "video.mp4".into(),
            extension: "mp4".into(),
            file_size_bytes: 1000,
            fingerprint: None,
            thumbnail_path: None,
            metadata: MediaMetadata::default(),
            tags: vec![],
            notes: String::new(),
            created_at: now,
            updated_at: now,
            deleted_at: None,
        };
        db.insert_media(&video_asset).unwrap();

        let srt_path = dir.path().join("sub.srt");
        std::fs::write(
            &srt_path,
            "1\n00:00:01,000 --> 00:00:04,500\nFala galera, olha esse corte!\n\n",
        )
        .unwrap();

        let srt_asset = MediaAsset {
            id: "sub-1".into(),
            title: "Legenda SRT".into(),
            kind: MediaKind::Subtitle,
            source_type: MediaSourceType::Generated,
            source_url: None,
            parent_media_id: Some("vid-1".into()),
            file_path: srt_path.to_string_lossy().to_string(),
            file_name: "sub.srt".into(),
            extension: "srt".into(),
            file_size_bytes: 60,
            fingerprint: None,
            thumbnail_path: None,
            metadata: MediaMetadata::default(),
            tags: vec!["video_id:vid-1".into()],
            notes: String::new(),
            created_at: now,
            updated_at: now,
            deleted_at: None,
        };
        db.insert_media(&srt_asset).unwrap();

        let resolved = db.find_best_transcript_for_media("vid-1").unwrap().unwrap();
        assert_eq!(resolved.source, "library_subtitle_file");
        assert_eq!(resolved.transcript.segments.len(), 1);
        assert_eq!(resolved.transcript.segments[0].start_ms, 1000);
        assert!(resolved.transcript.segments[0].text.contains("olha esse corte"));

        // Verify Trash -> Restore -> Purge lifecycle
        assert_eq!(db.count_media().unwrap(), 2);
        db.trash_media("sub-1").unwrap();
        assert_eq!(db.count_media().unwrap(), 1);
        assert_eq!(db.list_trashed_media().unwrap().len(), 1);
        db.restore_media("sub-1").unwrap();
        assert_eq!(db.count_media().unwrap(), 2);
        assert!(db.list_trashed_media().unwrap().is_empty());
        db.delete_media("sub-1").unwrap();
        assert_eq!(db.count_media().unwrap(), 1);
    }
}
