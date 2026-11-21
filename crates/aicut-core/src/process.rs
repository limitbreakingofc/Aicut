use std::path::{Path, PathBuf};

/// Removes AppImage mount prefixes (`$APPDIR` or `/tmp/.mount_*`) from `LD_LIBRARY_PATH`
/// so external host binaries (`nvidia-smi`, `ffmpeg`, `chromium`, `yt-dlp`) launched from
/// a legacy AppImage use system libraries.
pub fn sanitized_host_ld_library_path(raw: &str, appdir: Option<&str>) -> String {
    raw.split(':')
        .map(str::trim)
        .filter(|entry| {
            if entry.is_empty() {
                return false;
            }
            if let Some(app) = appdir.filter(|s| !s.is_empty()) {
                if entry.starts_with(app) {
                    return false;
                }
            }
            !entry.starts_with("/tmp/.mount_")
        })
        .collect::<Vec<_>>()
        .join(":")
}

/// Builds a clean Linux `LD_LIBRARY_PATH` for managed GPU/AI binaries (`llama-server`, `whisper-cli`)
/// by prepending the binary's own directory (for bundled `libllama.so`, `libggml.so`, `libwhisper.so`)
/// plus standard CUDA/NVIDIA library directories.
pub fn build_linux_gpu_ld_library_path(bin_path: &Path, existing_host_ld: Option<&str>) -> String {
    let mut dirs: Vec<String> = Vec::new();
    let mut push_unique = |p: String| {
        if !p.is_empty() && !dirs.iter().any(|d| d == &p) {
            dirs.push(p);
        }
    };

    if let Some(parent) = bin_path.parent() {
        push_unique(parent.to_string_lossy().to_string());
    }

    // Check toolkit and system driver library directories.
    for candidate in [
        "/opt/cuda/lib64",
        "/opt/cuda/targets/x86_64-linux/lib",
        "/usr/local/cuda/lib64",
        "/usr/lib64/nvidia",
        "/usr/lib/nvidia",
    ] {
        if PathBuf::from(candidate).is_dir() {
            push_unique(candidate.to_string());
        }
    }

    if let Some(host) = existing_host_ld {
        for part in host.split(':').map(str::trim).filter(|s| !s.is_empty()) {
            push_unique(part.to_string());
        }
    }

    dirs.join(":")
}

fn clean_env_ld_library_path() -> Option<String> {
    let raw = std::env::var("LD_LIBRARY_PATH").ok()?;
    let appdir = std::env::var("APPDIR").ok();
    let has_appimage = raw.contains("/tmp/.mount_")
        || appdir
            .as_deref()
            .map(|a| !a.is_empty() && raw.contains(a))
            .unwrap_or(false);
    if !has_appimage {
        return None;
    }
    Some(sanitized_host_ld_library_path(&raw, appdir.as_deref()))
}

/// Sanitizes child process environment on Linux (prevents AppImage `LD_LIBRARY_PATH` pollution
/// when invoking system binaries from a legacy AppImage).
pub fn configure_std_command(cmd: &mut std::process::Command) -> &mut std::process::Command {
    if let Some(clean_ld) = clean_env_ld_library_path() {
        if clean_ld.is_empty() {
            cmd.env_remove("LD_LIBRARY_PATH");
        } else {
            cmd.env("LD_LIBRARY_PATH", clean_ld);
        }
    }
    cmd
}

pub fn configure_tokio_command(cmd: &mut tokio::process::Command) -> &mut tokio::process::Command {
    if let Some(clean_ld) = clean_env_ld_library_path() {
        if clean_ld.is_empty() {
            cmd.env_remove("LD_LIBRARY_PATH");
        } else {
            cmd.env("LD_LIBRARY_PATH", clean_ld);
        }
    }
    cmd
}

/// Configures a native Linux GPU/AI child process (`llama-server`, `whisper-cli`) with
/// sanitized host `LD_LIBRARY_PATH` + local `.so` directory + CUDA library paths.
pub fn configure_linux_gpu_env_std<'a>(
    cmd: &'a mut std::process::Command,
    bin_path: &Path,
) -> &'a mut std::process::Command {
    configure_std_command(cmd);
    if let Some(parent) = bin_path.parent() {
        cmd.current_dir(parent);
    }
    let appdir = std::env::var("APPDIR").ok();
    let host_ld = std::env::var("LD_LIBRARY_PATH")
        .ok()
        .map(|raw| sanitized_host_ld_library_path(&raw, appdir.as_deref()));
    let ld = build_linux_gpu_ld_library_path(bin_path, host_ld.as_deref());
    if !ld.is_empty() {
        cmd.env("LD_LIBRARY_PATH", ld);
    }
    cmd
}

pub fn configure_linux_gpu_env_tokio<'a>(
    cmd: &'a mut tokio::process::Command,
    bin_path: &Path,
) -> &'a mut tokio::process::Command {
    configure_tokio_command(cmd);
    if let Some(parent) = bin_path.parent() {
        cmd.current_dir(parent);
    }
    let appdir = std::env::var("APPDIR").ok();
    let host_ld = std::env::var("LD_LIBRARY_PATH")
        .ok()
        .map(|raw| sanitized_host_ld_library_path(&raw, appdir.as_deref()));
    let ld = build_linux_gpu_ld_library_path(bin_path, host_ld.as_deref());
    if !ld.is_empty() {
        cmd.env("LD_LIBRARY_PATH", ld);
    }
    cmd
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn strips_appimage_mount_paths_but_preserves_host_cuda_paths() {
        let raw = "/tmp/.mount_AicutABC/usr/lib:/opt/cuda/lib64:/usr/lib";
        let clean = sanitized_host_ld_library_path(raw, Some("/tmp/.mount_AicutABC"));
        assert_eq!(clean, "/opt/cuda/lib64:/usr/lib");
    }

    #[test]
    fn prepends_binary_parent_dir_for_shared_libs() {
        let bin = Path::new("/home/user/.local/share/Aicut/bin/llama/llama-server");
        let ld = build_linux_gpu_ld_library_path(bin, Some("/usr/lib"));
        assert!(ld.starts_with("/home/user/.local/share/Aicut/bin/llama:"));
        assert!(ld.ends_with("/usr/lib"));
    }
}
