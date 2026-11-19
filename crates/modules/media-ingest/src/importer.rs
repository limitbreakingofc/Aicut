use aicut_core::{
    ActorKind, AicutError, AicutEvent, AicutKernel, AicutResult, JobProgress, MediaAsset,
    MediaKind, MediaSourceType,
};
use chrono::Utc;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use uuid::Uuid;
use walkdir::WalkDir;

use crate::ffprobe::{generate_thumbnail, probe_file};
use crate::formats::{compute_fast_fingerprint, extension_lower, is_supported_media_path};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema, Default)]
#[serde(rename_all = "snake_case")]
pub enum ImportMode {
    /// Keep original file in place without duplicating disk space (instant).
    #[default]
    Reference,
    /// Copy file into the managed Aicut Imports workspace directory.
    Copy,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct ImportLocalRequest {
    /// One or more local file paths or directory paths on the PC.
    pub paths: Vec<String>,
    #[serde(default)]
    pub mode: ImportMode,
    #[serde(default)]
    pub recursive: bool,
    #[serde(default)]
    pub tags: Vec<String>,
    #[serde(default = "default_true")]
    pub skip_duplicates: bool,
    #[serde(default = "default_true")]
    pub wait: bool,
}

fn default_true() -> bool {
    true
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct ImportLocalResponse {
    pub job_id: String,
    pub imported: Vec<MediaAsset>,
    pub skipped_duplicates: Vec<String>,
    pub errors: Vec<String>,
}

pub async fn import_local(
    kernel: Arc<AicutKernel>,
    req: ImportLocalRequest,
    actor: ActorKind,
) -> AicutResult<ImportLocalResponse> {
    if req.paths.is_empty() {
        return Err(AicutError::invalid_input(
            "Provide at least one local file or directory path to import.",
        ));
    }

    let (job, cancel) = kernel.jobs.create_job(
        "media.import_local",
        format!("Importar {} item(ns) do PC", req.paths.len()),
        "media-ingest",
        actor,
    );
    let job_id = job.id.clone();

    let wait = req.wait;
    let handle = tokio::spawn(execute_import_task(
        kernel.clone(),
        job_id.clone(),
        req,
        cancel,
    ));

    if wait {
        handle
            .await
            .map_err(|e| AicutError::internal(format!("Import task panicked: {e}")))?
    } else {
        Ok(ImportLocalResponse {
            job_id,
            imported: Vec::new(),
            skipped_duplicates: Vec::new(),
            errors: Vec::new(),
        })
    }
}

async fn execute_import_task(
    kernel: Arc<AicutKernel>,
    job_id: String,
    req: ImportLocalRequest,
    cancel: tokio_util::sync::CancellationToken,
) -> AicutResult<ImportLocalResponse> {
    kernel.jobs.mark_running(&job_id, "scanning_paths");

    let mut files_to_import: Vec<PathBuf> = Vec::new();
    let mut errors: Vec<String> = Vec::new();

    for raw in &req.paths {
        let p = PathBuf::from(raw.trim());
        if !p.exists() {
            errors.push(format!("Path does not exist: {}", p.display()));
            continue;
        }
        if p.is_file() {
            if is_supported_media_path(&p) {
                files_to_import.push(p);
            } else {
                errors.push(format!("Unsupported media format: {}", p.display()));
            }
        } else if p.is_dir() {
            let max_depth = if req.recursive { usize::MAX } else { 1 };
            for entry in WalkDir::new(&p).max_depth(max_depth).into_iter().flatten() {
                if entry.file_type().is_file() && is_supported_media_path(entry.path()) {
                    files_to_import.push(entry.into_path());
                }
            }
        }
    }

    let total = files_to_import.len();
    if total == 0 {
        let err = AicutError::invalid_input(if errors.is_empty() {
            "No supported video or audio files found in the specified path(s).".to_string()
        } else {
            errors.join("; ")
        });
        kernel.jobs.fail(&job_id, err.clone());
        return Err(err);
    }

    let ffmpeg_path = kernel.tools.resolve_tool("ffmpeg").ok();
    let ffprobe_path = kernel.tools.resolve_tool("ffprobe").ok();

    let mut imported = Vec::new();
    let mut skipped_duplicates = Vec::new();

    for (idx, src_path) in files_to_import.iter().enumerate() {
        if cancel.is_cancelled() {
            kernel.jobs.mark_cancelled(&job_id);
            return Err(AicutError::cancelled("Local import cancelled."));
        }

        let percent = (idx as f64 / total as f64) * 100.0;
        kernel.jobs.update_progress(
            &job_id,
            JobProgress {
                percent,
                stage: "importing".to_string(),
                message: Some(format!(
                    "Importando ({}/{}) {}",
                    idx + 1,
                    total,
                    src_path
                        .file_name()
                        .and_then(|n| n.to_str())
                        .unwrap_or("arquivo")
                )),
                ..Default::default()
            },
        );

        match import_single_file(
            &kernel,
            src_path,
            req.mode,
            &req.tags,
            req.skip_duplicates,
            ffmpeg_path.as_deref(),
            ffprobe_path.as_deref(),
        )
        .await
        {
            Ok(ImportFileOutcome::Imported(asset)) => imported.push(asset),
            Ok(ImportFileOutcome::Duplicate(existing_id)) => {
                skipped_duplicates.push(format!("{} (existing id: {existing_id})", src_path.display()));
            }
            Err(e) => errors.push(format!("{}: {}", src_path.display(), e.message)),
        }
    }

    let response = ImportLocalResponse {
        job_id: job_id.clone(),
        imported,
        skipped_duplicates,
        errors,
    };

    kernel.jobs.complete(
        &job_id,
        serde_json::to_value(&response).unwrap_or(serde_json::Value::Null),
    );
    Ok(response)
}

enum ImportFileOutcome {
    Imported(MediaAsset),
    Duplicate(String),
}

async fn import_single_file(
    kernel: &Arc<AicutKernel>,
    src_path: &Path,
    mode: ImportMode,
    tags: &[String],
    skip_duplicates: bool,
    ffmpeg_path: Option<&Path>,
    ffprobe_path: Option<&Path>,
) -> AicutResult<ImportFileOutcome> {
    let canonical = std::fs::canonicalize(src_path).unwrap_or_else(|_| src_path.to_path_buf());
    let fingerprint = compute_fast_fingerprint(&canonical).ok();

    if skip_duplicates {
        if let Some(ref fp) = fingerprint {
            if let Some(existing) = kernel.repo.find_by_fingerprint(fp)? {
                return Ok(ImportFileOutcome::Duplicate(existing.id));
            }
        }
    }

    let id = Uuid::new_v4().to_string();
    let ext = extension_lower(&canonical);
    let file_name = canonical
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("media")
        .to_string();
    let title = canonical
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or(&file_name)
        .to_string();

    let stored_path = match mode {
        ImportMode::Reference => canonical,
        ImportMode::Copy => {
            std::fs::create_dir_all(&kernel.workspace.imports_dir)?;
            let dest = kernel
                .workspace
                .imports_dir
                .join(format!("{}-{file_name}", &id[..8]));
            std::fs::copy(&canonical, &dest)?;
            dest
        }
    };

    let file_size_bytes = std::fs::metadata(&stored_path).map(|m| m.len()).unwrap_or(0);
    let probe = probe_file(ffprobe_path, &stored_path, &ext).await;

    let thumbnail_path = if probe.kind != MediaKind::Audio {
        generate_thumbnail(
            ffmpeg_path,
            &stored_path,
            &kernel.workspace.thumbnails_dir,
            &id,
            probe.metadata.duration_sec,
        )
        .await
        .map(|p| p.to_string_lossy().to_string())
    } else {
        None
    };

    let now = Utc::now();
    let asset = MediaAsset {
        id,
        title,
        kind: probe.kind,
        source_type: MediaSourceType::Local,
        source_url: None,
        parent_media_id: None,
        file_path: stored_path.to_string_lossy().to_string(),
        file_name,
        extension: ext,
        file_size_bytes,
        fingerprint,
        thumbnail_path,
        metadata: probe.metadata,
        tags: tags.to_vec(),
        notes: String::new(),
        created_at: now,
        updated_at: now,
    };

    kernel.repo.insert_media(&asset)?;
    kernel.events.publish(AicutEvent::MediaCreated(asset.clone()));
    Ok(ImportFileOutcome::Imported(asset))
}
