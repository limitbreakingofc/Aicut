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
            ",
        )?;

        // Safe upgrade migration for v0.1.0 databases that lacked `parent_media_id`
        let _ = conn.execute(
            "ALTER TABLE media_assets ADD COLUMN parent_media_id TEXT",
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
                        metadata_json, tags_json, notes, created_at, updated_at
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
                        metadata_json, tags_json, notes, created_at, updated_at
                 FROM media_assets WHERE fingerprint = ?1 LIMIT 1",
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
                    metadata_json, tags_json, notes, created_at, updated_at
             FROM media_assets
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
        let count: i64 = conn.query_row("SELECT COUNT(*) FROM media_assets", [], |r| r.get(0))?;
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
