use aicut_core::WorkspaceConfig;
use aicut_server::ServerConfig;
use tauri_plugin_dialog::DialogExt;

#[tauri::command]
async fn open_download_folder(tools: bool) -> Result<(), String> {
    let workspace = WorkspaceConfig::default_paths();
    let path = if tools { workspace.bin_dir } else { workspace.models_dir };
    std::fs::create_dir_all(&path).map_err(|e| e.to_string())?;
    let status = tokio::process::Command::new("xdg-open").arg(path).status().await.map_err(|e| e.to_string())?;
    if status.success() { Ok(()) } else { Err("Falha ao abrir a pasta".into()) }
}

#[tauri::command]
async fn pick_media_paths(app: tauri::AppHandle, directory: bool) -> Result<Vec<String>, String> {
    tauri::async_runtime::spawn_blocking(move || {
        if directory {
            let folder = app.dialog().file().blocking_pick_folder();
            folder
                .and_then(|p| p.into_path().ok())
                .map(|p| vec![p.to_string_lossy().to_string()])
                .unwrap_or_default()
        } else {
            let files = app
                .dialog()
                .file()
                .add_filter(
                    "Vídeos e Áudios",
                    &[
                        "mp4", "mkv", "webm", "mov", "avi", "flv", "m4v", "ts", "wmv", "mp3", "m4a",
                        "wav", "flac", "opus", "ogg", "aac",
                    ],
                )
                .blocking_pick_files();
            files
                .unwrap_or_default()
                .into_iter()
                .filter_map(|fp| fp.into_path().ok())
                .map(|p| p.to_string_lossy().to_string())
                .collect()
        }
    })
    .await
    .map_err(|e| e.to_string())
}

/// Protects WebKitGTK on Fedora Linux (especially with NVIDIA proprietary or
/// open-kernel drivers on Wayland/XWayland) against blank windows and GBM/EGL protocol crashes.
/// Users can explicitly override by setting `WEBKIT_DISABLE_DMABUF_RENDERER=0` in their shell.
fn configure_linux_webkit_env() {
    #[cfg(target_os = "linux")]
    {
        if std::env::var_os("WEBKIT_DISABLE_DMABUF_RENDERER").is_none() {
            let has_nvidia = std::path::Path::new("/dev/nvidia0").exists()
                || std::path::Path::new("/proc/driver/nvidia/version").exists();
            let is_wayland = std::env::var_os("WAYLAND_DISPLAY").is_some();
            if has_nvidia || is_wayland {
                std::env::set_var("WEBKIT_DISABLE_DMABUF_RENDERER", "1");
                tracing::info!(
                    has_nvidia,
                    is_wayland,
                    "Configured WEBKIT_DISABLE_DMABUF_RENDERER=1 for stable Linux/NVIDIA WebKitGTK rendering"
                );
            }
        }
    }
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    let _ = tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .try_init();

    configure_linux_webkit_env();

    tauri::Builder::default()
        .plugin(tauri_plugin_dialog::init())
        .setup(|_app| {
            // Boot the embedded Aicut Rust kernel and local AI/Media server (127.0.0.1:43210)
            tauri::async_runtime::spawn(async {
                let ws = WorkspaceConfig::default_paths();
                match aicut_runtime::bootstrap_kernel(ws).await {
                    Ok(kernel) => {
                        if let Err(e) =
                            aicut_server::spawn_server(kernel, ServerConfig::default()).await
                        {
                            tracing::error!("Failed starting embedded Aicut server: {e}");
                        }
                    }
                    Err(e) => {
                        tracing::error!("Failed bootstrapping Aicut kernel: {e}");
                    }
                }
            });
            Ok(())
        })
        .invoke_handler(tauri::generate_handler![pick_media_paths, open_download_folder])
        .run(tauri::generate_context!())
        .expect("error while running Aicut desktop application");
}
