use aicut_core::WorkspaceConfig;
use aicut_server::ServerConfig;
use tauri_plugin_dialog::DialogExt;

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

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    let _ = tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .try_init();

    // Boot the embedded Aicut Rust kernel and local AI/Media server (127.0.0.1:43210)
    tauri::async_runtime::spawn(async {
        let ws = WorkspaceConfig::default_paths();
        match aicut_runtime::bootstrap_kernel(ws).await {
            Ok(kernel) => {
                if let Err(e) = aicut_server::spawn_server(kernel, ServerConfig::default()).await {
                    tracing::error!("Failed starting embedded Aicut server: {e}");
                }
            }
            Err(e) => {
                tracing::error!("Failed bootstrapping Aicut kernel: {e}");
            }
        }
    });

    tauri::Builder::default()
        .plugin(tauri_plugin_dialog::init())
        .invoke_handler(tauri::generate_handler![pick_media_paths])
        .run(tauri::generate_context!())
        .expect("error while running Aicut desktop application");
}
