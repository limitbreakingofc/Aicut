use aicut_core::MediaKind;
use sha2::{Digest, Sha256};
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::Path;

pub const SUPPORTED_VIDEO_EXTENSIONS: &[&str] = &[
    "mp4", "mkv", "webm", "mov", "avi", "flv", "m4v", "ts", "mts", "m2ts", "wmv", "3gp", "ogv",
    "mpeg", "mpg",
];

pub const SUPPORTED_AUDIO_EXTENSIONS: &[&str] = &[
    "mp3", "m4a", "wav", "flac", "opus", "ogg", "aac", "wma", "aiff", "alac", "mka",
];

pub fn extension_lower(path: &Path) -> String {
    path.extension()
        .and_then(|e| e.to_str())
        .map(|s| s.to_lowercase())
        .unwrap_or_default()
}

pub fn is_supported_media_path(path: &Path) -> bool {
    let ext = extension_lower(path);
    SUPPORTED_VIDEO_EXTENSIONS.contains(&ext.as_str())
        || SUPPORTED_AUDIO_EXTENSIONS.contains(&ext.as_str())
}

pub fn infer_kind_from_extension(ext: &str) -> MediaKind {
    let lower = ext.to_lowercase();
    if SUPPORTED_AUDIO_EXTENSIONS.contains(&lower.as_str()) {
        MediaKind::Audio
    } else {
        MediaKind::Video
    }
}

/// Computes a fast, deterministic partial SHA-256 fingerprint (size + head 64KB + tail 64KB)
/// so multi-gigabyte 4K files are fingerprinted in milliseconds.
pub fn compute_fast_fingerprint(path: &Path) -> std::io::Result<String> {
    let mut file = File::open(path)?;
    let size = file.metadata()?.len();
    let mut hasher = Sha256::new();
    hasher.update(size.to_le_bytes());

    let chunk_size = 64 * 1024;
    let mut buf = vec![0u8; chunk_size];

    let n = file.read(&mut buf)?;
    hasher.update(&buf[..n]);

    if size > chunk_size as u64 {
        let seek_pos = size.saturating_sub(chunk_size as u64);
        file.seek(SeekFrom::Start(seek_pos))?;
        let n2 = file.read(&mut buf)?;
        hasher.update(&buf[..n2]);
    }

    let digest = hasher.finalize();
    Ok(digest.iter().map(|b| format!("{b:02x}")).collect())
}
