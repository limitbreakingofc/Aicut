use aicut_core::{configure_tokio_command, ActorKind, AicutKernel, UiState};
use aicut_mcp::{action_id_to_mcp_name, handle_mcp_request, JsonRpcRequest};
use axum::body::Body;
use axum::extract::ws::{Message, WebSocket, WebSocketUpgrade};
use axum::extract::{Path, Query, State};
use axum::http::{header, HeaderMap, HeaderValue, Method, StatusCode};
use axum::response::sse::{Event as SseEvent, KeepAlive, Sse};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use futures_util::{SinkExt, StreamExt};
use serde::Deserialize;
use serde_json::{json, Value};
use std::collections::HashMap;
use std::convert::Infallible;
use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::{Arc, RwLock};
use tokio::io::{AsyncReadExt, AsyncSeekExt};
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;
use tower_http::cors::{Any, CorsLayer};
use tower_http::services::{ServeDir, ServeFile};
use uuid::Uuid;

pub const DEFAULT_HOST: &str = "127.0.0.1";
pub const DEFAULT_PORT: u16 = 43210;

#[derive(Clone)]
pub struct ServerConfig {
    pub host: String,
    pub port: u16,
    pub api_token: Option<String>,
    pub web_dir: Option<PathBuf>,
}

impl Default for ServerConfig {
    fn default() -> Self {
        let port = std::env::var("AICUT_PORT")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(DEFAULT_PORT);
        let api_token = std::env::var("AICUT_API_TOKEN")
            .ok()
            .filter(|s| !s.trim().is_empty());
        Self {
            host: DEFAULT_HOST.to_string(),
            port,
            api_token,
            web_dir: None,
        }
    }
}

#[derive(Clone)]
pub struct ServerState {
    pub kernel: Arc<AicutKernel>,
    pub api_token: Option<String>,
    pub base_url: String,
    sse_sessions: Arc<RwLock<HashMap<String, mpsc::Sender<String>>>>,
    preview_build_lock: Arc<tokio::sync::Mutex<()>>,
}

impl ServerState {
    pub fn new(kernel: Arc<AicutKernel>, config: &ServerConfig) -> Self {
        Self {
            kernel,
            api_token: config.api_token.clone(),
            base_url: format!("http://{}:{}", config.host, config.port),
            sse_sessions: Arc::new(RwLock::new(HashMap::new())),
            preview_build_lock: Arc::new(tokio::sync::Mutex::new(())),
        }
    }
}

pub fn build_router(kernel: Arc<AicutKernel>, config: &ServerConfig) -> Router {
    let state = ServerState::new(kernel, config);

    let cors = CorsLayer::new()
        .allow_origin(Any)
        .allow_methods([Method::GET, Method::POST, Method::PUT, Method::DELETE, Method::OPTIONS])
        .allow_headers(Any)
        .expose_headers(Any);

    let mut router = Router::new()
        .route("/health", get(health_handler))
        .route("/openapi.json", get(openapi_handler))
        .route("/api/v1/modules", get(list_modules_handler))
        .route("/api/v1/actions", get(list_actions_handler))
        .route("/api/v1/actions/{action_id}", post(execute_action_handler))
        .route("/api/v1/tools/openai", get(openai_tools_handler))
        .route("/api/v1/tools/anthropic", get(anthropic_tools_handler))
        .route("/api/v1/activity", get(activity_handler))
        .route("/api/v1/media/{id}/stream", get(stream_media_handler))
        .route("/api/v1/media/{id}/preview", get(stream_preview_media_handler))
        .route("/api/v1/media/{id}/thumbnail", get(thumbnail_handler))
        .route("/api/v1/media/{id}/text", get(media_text_handler))
        .route("/api/v1/events", get(ws_events_handler))
        .route("/mcp", post(mcp_http_handler).get(mcp_info_handler))
        .route("/mcp/sse", get(mcp_sse_handler))
        .route("/mcp/messages", post(mcp_sse_message_handler))
        .with_state(state)
        .layer(cors);

    if let Some(ref web_dir) = config.web_dir {
        if web_dir.is_dir() {
            let index = web_dir.join("index.html");
            let svc = ServeDir::new(web_dir).fallback(ServeFile::new(index));
            router = router.fallback_service(svc);
        }
    }

    router
}

pub async fn start_server(
    kernel: Arc<AicutKernel>,
    config: ServerConfig,
) -> std::io::Result<SocketAddr> {
    let addr: SocketAddr = format!("{}:{}", config.host, config.port)
        .parse()
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidInput, e))?;
    let router = build_router(kernel, &config);
    let listener = tokio::net::TcpListener::bind(addr).await?;
    let bound = listener.local_addr()?;
    tracing::info!("Aicut server listening on http://{bound}");
    axum::serve(listener, router).await?;
    Ok(bound)
}

/// Spawns the Aicut HTTP/WS/MCP server on a background Tokio task (used by Tauri desktop app).
pub async fn spawn_server(
    kernel: Arc<AicutKernel>,
    config: ServerConfig,
) -> std::io::Result<SocketAddr> {
    let addr: SocketAddr = format!("{}:{}", config.host, config.port)
        .parse()
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidInput, e))?;
    let router = build_router(kernel, &config);
    let listener = tokio::net::TcpListener::bind(addr).await?;
    let bound = listener.local_addr()?;
    tracing::info!("Aicut embedded server listening on http://{bound}");
    tokio::spawn(async move {
        if let Err(e) = axum::serve(listener, router).await {
            tracing::error!("Aicut embedded server stopped with error: {e}");
        }
    });
    Ok(bound)
}

fn check_auth(state: &ServerState, headers: &HeaderMap) -> Result<(), Response> {
    let Some(ref expected) = state.api_token else {
        return Ok(());
    };
    let supplied = headers
        .get(header::AUTHORIZATION)
        .and_then(|v| v.to_str().ok())
        .and_then(|s| s.strip_prefix("Bearer "));
    if supplied == Some(expected.as_str()) {
        Ok(())
    } else {
        Err((
            StatusCode::UNAUTHORIZED,
            Json(json!({ "error": { "code": "UNAUTHORIZED", "message": "Invalid or missing Bearer token." } })),
        )
            .into_response())
    }
}

fn parse_actor(headers: &HeaderMap) -> ActorKind {
    match headers
        .get("x-aicut-actor")
        .and_then(|v| v.to_str().ok())
        .map(|s| s.to_lowercase())
        .as_deref()
    {
        Some("human") => ActorKind::Human,
        Some("system") => ActorKind::System,
        _ => ActorKind::Ai,
    }
}

async fn health_handler(State(state): State<ServerState>) -> impl IntoResponse {
    Json(json!({
        "status": "ok",
        "app": "Aicut",
        "version": env!("CARGO_PKG_VERSION"),
        "platform": state.kernel.hardware.platform_tag,
        "mcp_url": format!("{}/mcp", state.base_url),
        "openapi_url": format!("{}/openapi.json", state.base_url)
    }))
}

async fn list_modules_handler(
    State(state): State<ServerState>,
    headers: HeaderMap,
) -> Response {
    if let Err(resp) = check_auth(&state, &headers) {
        return resp;
    }
    let modules = state.kernel.list_modules().await;
    Json(json!({ "modules": modules })).into_response()
}

async fn list_actions_handler(
    State(state): State<ServerState>,
    headers: HeaderMap,
) -> Response {
    if let Err(resp) = check_auth(&state, &headers) {
        return resp;
    }
    let actions = state.kernel.list_actions(false);
    Json(json!({ "actions": actions })).into_response()
}

async fn execute_action_handler(
    State(state): State<ServerState>,
    Path(action_id): Path<String>,
    headers: HeaderMap,
    payload: Option<Json<Value>>,
) -> Response {
    if let Err(resp) = check_auth(&state, &headers) {
        return resp;
    }
    let actor = parse_actor(&headers);
    let input = payload.map(|Json(v)| v).unwrap_or_else(|| json!({}));

    match state.kernel.dispatch(&action_id, input, actor).await {
        Ok(output) => Json(json!({ "ok": true, "output": output })).into_response(),
        Err(err) => {
            let status = match err.code.as_str() {
                "NOT_FOUND" => StatusCode::NOT_FOUND,
                "INVALID_INPUT" => StatusCode::BAD_REQUEST,
                "MODULE_DISABLED" => StatusCode::FORBIDDEN,
                "DEPENDENCY_MISSING" => StatusCode::FAILED_DEPENDENCY,
                _ => StatusCode::INTERNAL_SERVER_ERROR,
            };
            (status, Json(json!({ "ok": false, "error": err }))).into_response()
        }
    }
}

async fn activity_handler(State(state): State<ServerState>, headers: HeaderMap) -> Response {
    if let Err(resp) = check_auth(&state, &headers) {
        return resp;
    }
    let entries = state.kernel.events.recent_activity(100);
    Json(json!({ "entries": entries })).into_response()
}

async fn openai_tools_handler(State(state): State<ServerState>, headers: HeaderMap) -> Response {
    if let Err(resp) = check_auth(&state, &headers) {
        return resp;
    }
    let tools: Vec<Value> = state
        .kernel
        .list_actions(false)
        .into_iter()
        .map(|spec| {
            json!({
                "type": "function",
                "function": {
                    "name": action_id_to_mcp_name(&spec.id),
                    "description": format!("[{}] {}", spec.id, spec.description),
                    "parameters": spec.input_schema
                }
            })
        })
        .collect();
    Json(json!({ "tools": tools })).into_response()
}

async fn anthropic_tools_handler(State(state): State<ServerState>, headers: HeaderMap) -> Response {
    if let Err(resp) = check_auth(&state, &headers) {
        return resp;
    }
    let tools: Vec<Value> = state
        .kernel
        .list_actions(false)
        .into_iter()
        .map(|spec| {
            json!({
                "name": action_id_to_mcp_name(&spec.id),
                "description": format!("[{}] {}", spec.id, spec.description),
                "input_schema": spec.input_schema
            })
        })
        .collect();
    Json(json!({ "tools": tools })).into_response()
}

async fn openapi_handler(State(state): State<ServerState>) -> impl IntoResponse {
    let actions = state.kernel.list_actions(false);
    let mut paths = serde_json::Map::new();

    for spec in actions {
        let path_key = format!("/api/v1/actions/{}", spec.id);
        let op = json!({
            "post": {
                "operationId": action_id_to_mcp_name(&spec.id),
                "summary": spec.title,
                "description": spec.description,
                "tags": [spec.module_id],
                "requestBody": {
                    "required": false,
                    "content": {
                        "application/json": {
                            "schema": spec.input_schema
                        }
                    }
                },
                "responses": {
                    "200": {
                        "description": "Action executed successfully",
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "object",
                                    "properties": {
                                        "ok": { "type": "boolean" },
                                        "output": {
                                            "type": "object",
                                            "properties": {
                                                "summary": { "type": "string" },
                                                "data": spec.output_schema
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        });
        paths.insert(path_key, op);
    }

    Json(json!({
        "openapi": "3.1.0",
        "info": {
            "title": "Aicut Local AI & Media API",
            "version": env!("CARGO_PKG_VERSION"),
            "description": "Modular Windows 11 media studio API with full human-parity AI control over yt-dlp downloads, local video/audio imports, Library, Jobs, Tools/Hardware, and live UI."
        },
        "servers": [{ "url": state.base_url }],
        "paths": paths
    }))
}

/// Streams local video or audio files with full HTTP Range (`206 Partial Content`) support.
async fn stream_media_handler(
    State(state): State<ServerState>,
    Path(id): Path<String>,
    headers: HeaderMap,
) -> Response {
    let Ok(Some(asset)) = state.kernel.repo.get_media(&id) else {
        return (StatusCode::NOT_FOUND, "Media not found").into_response();
    };
    stream_file_with_ranges(&PathBuf::from(&asset.file_path), &headers, None).await
}

/// Streams a guaranteed WebView2/Chromium-playable H.264/AAC MP4 preview proxy on demand.
/// Used automatically as fallback when the original video uses an unsupported container/codec
/// (e.g. MKV, AV1, HEVC, ProRes) so Motion Studio preview NEVER breaks.
async fn stream_preview_media_handler(
    State(state): State<ServerState>,
    Path(id): Path<String>,
    headers: HeaderMap,
) -> Response {
    let Ok(Some(asset)) = state.kernel.repo.get_media(&id) else {
        return (StatusCode::NOT_FOUND, "Media not found").into_response();
    };
    let src_path = PathBuf::from(&asset.file_path);
    if !src_path.is_file() {
        return (StatusCode::NOT_FOUND, "Media file missing on disk").into_response();
    }

    let safe_id: String = id
        .chars()
        .filter(|c| c.is_ascii_alphanumeric() || *c == '-' || *c == '_')
        .collect();
    let cache_dir = std::env::temp_dir().join("aicut-preview-cache");
    let _ = std::fs::create_dir_all(&cache_dir);
    let proxy_path = cache_dir.join(format!("{safe_id}.mp4"));

    if !proxy_path.is_file() {
        let _guard = state.preview_build_lock.lock().await;
        if !proxy_path.is_file() {
            let Ok(ffmpeg_bin) = state.kernel.tools.resolve_tool("ffmpeg") else {
                return stream_file_with_ranges(&src_path, &headers, None).await;
            };
            let tmp_path = cache_dir.join(format!("{safe_id}.building.mp4"));
            let mut cmd = tokio::process::Command::new(&ffmpeg_bin);
            configure_tokio_command(&mut cmd);
            let ok = cmd
                .args(["-y", "-i"])
                .arg(&src_path)
                .args([
                    "-vf",
                    "scale='min(960,iw)':-2",
                    "-c:v",
                    "libx264",
                    "-preset",
                    "ultrafast",
                    "-crf",
                    "26",
                    "-pix_fmt",
                    "yuv420p",
                    "-c:a",
                    "aac",
                    "-b:a",
                    "128k",
                    "-movflags",
                    "+faststart",
                ])
                .arg(&tmp_path)
                .output()
                .await
                .map(|o| o.status.success() && tmp_path.is_file())
                .unwrap_or(false);
            if ok {
                let _ = std::fs::rename(&tmp_path, &proxy_path);
            } else {
                let _ = std::fs::remove_file(&tmp_path);
                return stream_file_with_ranges(&src_path, &headers, None).await;
            }
        }
    }

    stream_file_with_ranges(&proxy_path, &headers, Some("video/mp4")).await
}

async fn stream_file_with_ranges(
    path: &std::path::Path,
    headers: &HeaderMap,
    mime_override: Option<&str>,
) -> Response {
    let Ok(mut file) = tokio::fs::File::open(path).await else {
        return (StatusCode::NOT_FOUND, "Media file missing on disk").into_response();
    };
    let Ok(meta) = file.metadata().await else {
        return StatusCode::INTERNAL_SERVER_ERROR.into_response();
    };
    let file_size = meta.len();
    let mime = mime_override
        .map(|s| s.to_string())
        .unwrap_or_else(|| mime_guess::from_path(path).first_or_octet_stream().to_string());

    if let Some(range_val) = headers.get(header::RANGE).and_then(|v| v.to_str().ok()) {
        if let Some((start, end)) = parse_byte_range(range_val, file_size) {
            let chunk_len = end - start + 1;
            if file.seek(std::io::SeekFrom::Start(start)).await.is_err() {
                return StatusCode::INTERNAL_SERVER_ERROR.into_response();
            }
            let limited = file.take(chunk_len);
            let stream = tokio_util::io::ReaderStream::new(limited);
            let mut resp = Response::new(Body::from_stream(stream));
            *resp.status_mut() = StatusCode::PARTIAL_CONTENT;
            let h = resp.headers_mut();
            let _ = h.insert(header::CONTENT_TYPE, HeaderValue::from_str(&mime).unwrap_or(HeaderValue::from_static("application/octet-stream")));
            h.insert(header::ACCEPT_RANGES, HeaderValue::from_static("bytes"));
            let _ = h.insert(header::CONTENT_LENGTH, HeaderValue::from_str(&chunk_len.to_string()).unwrap());
            let _ = h.insert(
                header::CONTENT_RANGE,
                HeaderValue::from_str(&format!("bytes {start}-{end}/{file_size}")).unwrap(),
            );
            return resp;
        }
    }

    let stream = tokio_util::io::ReaderStream::new(file);
    let mut resp = Response::new(Body::from_stream(stream));
    let h = resp.headers_mut();
    let _ = h.insert(header::CONTENT_TYPE, HeaderValue::from_str(&mime).unwrap_or(HeaderValue::from_static("application/octet-stream")));
    h.insert(header::ACCEPT_RANGES, HeaderValue::from_static("bytes"));
    let _ = h.insert(header::CONTENT_LENGTH, HeaderValue::from_str(&file_size.to_string()).unwrap());
    resp
}

fn parse_byte_range(header_str: &str, file_size: u64) -> Option<(u64, u64)> {
    if file_size == 0 {
        return None;
    }
    let spec = header_str.strip_prefix("bytes=")?;
    let first = spec.split(',').next()?.trim();
    let (start_s, end_s) = first.split_once('-')?;
    let start: u64 = if start_s.is_empty() {
        let suffix: u64 = end_s.parse().ok()?;
        file_size.saturating_sub(suffix)
    } else {
        start_s.parse().ok()?
    };
    let end: u64 = if end_s.is_empty() {
        file_size - 1
    } else {
        end_s.parse::<u64>().ok()?.min(file_size - 1)
    };
    if start <= end && end < file_size {
        Some((start, end))
    } else {
        None
    }
}

async fn thumbnail_handler(
    State(state): State<ServerState>,
    Path(id): Path<String>,
) -> Response {
    let Ok(Some(asset)) = state.kernel.repo.get_media(&id) else {
        return StatusCode::NOT_FOUND.into_response();
    };
    let Some(ref thumb) = asset.thumbnail_path else {
        return StatusCode::NOT_FOUND.into_response();
    };
    match tokio::fs::read(thumb).await {
        Ok(bytes) => (
            [(header::CONTENT_TYPE, "image/jpeg"), (header::CACHE_CONTROL, "public, max-age=3600")],
            bytes,
        )
            .into_response(),
        Err(_) => StatusCode::NOT_FOUND.into_response(),
    }
}

async fn media_text_handler(
    State(state): State<ServerState>,
    Path(id): Path<String>,
) -> Response {
    let Ok(Some(asset)) = state.kernel.repo.get_media(&id) else {
        return StatusCode::NOT_FOUND.into_response();
    };
    match tokio::fs::read_to_string(&asset.file_path).await {
        Ok(text) => (
            [(header::CONTENT_TYPE, "text/plain; charset=utf-8")],
            text,
        )
            .into_response(),
        Err(_) => StatusCode::NOT_FOUND.into_response(),
    }
}

async fn ws_events_handler(
    ws: WebSocketUpgrade,
    State(state): State<ServerState>,
) -> impl IntoResponse {
    ws.on_upgrade(move |socket| handle_ws_connection(socket, state))
}

async fn handle_ws_connection(socket: WebSocket, state: ServerState) {
    let (mut sender, mut receiver) = socket.split();
    let mut rx = state.kernel.events.subscribe();

    // Send initial UI snapshot on connect
    let snapshot = json!({
        "type": "snapshot",
        "ui_state": state.kernel.ui.get(),
        "jobs": state.kernel.jobs.list(false),
        "activity": state.kernel.events.recent_activity(30)
    });
    if sender
        .send(Message::Text(snapshot.to_string().into()))
        .await
        .is_err()
    {
        return;
    }

    let send_task = tokio::spawn(async move {
        while let Ok(env) = rx.recv().await {
            if let Ok(text) = serde_json::to_string(&env) {
                if sender.send(Message::Text(text.into())).await.is_err() {
                    break;
                }
            }
        }
    });

    let kernel_ref = state.kernel.clone();
    while let Some(Ok(msg)) = receiver.next().await {
        if let Message::Text(text) = msg {
            if let Ok(val) = serde_json::from_str::<Value>(&text) {
                if val.get("type").and_then(|v| v.as_str()) == Some("ui_sync") {
                    if let Some(st_val) = val.get("state").cloned() {
                        if let Ok(next_state) = serde_json::from_value::<UiState>(st_val) {
                            kernel_ref.ui.replace(next_state);
                        }
                    }
                }
            }
        }
    }

    send_task.abort();
}

// ============================================================================
// MCP Streamable HTTP + Legacy SSE Endpoints
// ============================================================================

async fn mcp_info_handler(State(state): State<ServerState>) -> impl IntoResponse {
    Json(json!({
        "name": "Aicut MCP Server",
        "protocol": "Model Context Protocol (JSON-RPC 2.0)",
        "streamable_http": format!("{}/mcp", state.base_url),
        "legacy_sse": format!("{}/mcp/sse", state.base_url),
        "stdio_command": "aicut mcp"
    }))
}

async fn mcp_http_handler(
    State(state): State<ServerState>,
    headers: HeaderMap,
    Json(req): Json<JsonRpcRequest>,
) -> Response {
    if let Err(resp) = check_auth(&state, &headers) {
        return resp;
    }
    match handle_mcp_request(&state.kernel, req).await {
        Some(resp) => Json(resp).into_response(),
        None => StatusCode::ACCEPTED.into_response(),
    }
}

async fn mcp_sse_handler(
    State(state): State<ServerState>,
) -> Sse<impl futures_util::Stream<Item = Result<SseEvent, Infallible>>> {
    let session_id = Uuid::new_v4().to_string();
    let (tx, rx) = mpsc::channel::<String>(64);
    if let Ok(mut guard) = state.sse_sessions.write() {
        guard.insert(session_id.clone(), tx.clone());
    }

    let endpoint_url = format!("/mcp/messages?session_id={session_id}");
    let _ = tx.try_send(format!("ENDPOINT:{endpoint_url}"));

    let stream = ReceiverStream::new(rx).map(|msg| {
        if let Some(ep) = msg.strip_prefix("ENDPOINT:") {
            Ok(SseEvent::default().event("endpoint").data(ep.to_string()))
        } else {
            Ok(SseEvent::default().event("message").data(msg))
        }
    });

    Sse::new(stream).keep_alive(KeepAlive::default())
}

#[derive(Deserialize)]
struct SseQuery {
    session_id: String,
}

async fn mcp_sse_message_handler(
    State(state): State<ServerState>,
    Query(q): Query<SseQuery>,
    headers: HeaderMap,
    Json(req): Json<JsonRpcRequest>,
) -> Response {
    if let Err(resp) = check_auth(&state, &headers) {
        return resp;
    }
    let tx = state
        .sse_sessions
        .read()
        .ok()
        .and_then(|g| g.get(&q.session_id).cloned());
    let Some(tx) = tx else {
        return (StatusCode::NOT_FOUND, "Unknown SSE session_id").into_response();
    };
    if let Some(resp) = handle_mcp_request(&state.kernel, req).await {
        if let Ok(json_str) = serde_json::to_string(&resp) {
            let _ = tx.send(json_str).await;
        }
    }
    StatusCode::ACCEPTED.into_response()
}

#[cfg(test)]
mod tests {
    use super::*;
    use aicut_core::WorkspaceConfig;
    use axum::http::Request;
    use http_body_util::BodyExt;
    use tower::ServiceExt;

    #[tokio::test]
    async fn server_exposes_openapi_openai_tools_actions_and_mcp() {
        let tmp = tempfile::tempdir().unwrap();
        let kernel = aicut_runtime::bootstrap_kernel(WorkspaceConfig::from_root(tmp.path()))
            .await
            .unwrap();
        let app = build_router(kernel, &ServerConfig::default());

        // 1. OpenAPI 3.1
        let resp = app
            .clone()
            .oneshot(Request::builder().uri("/openapi.json").body(Body::empty()).unwrap())
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let bytes = resp.into_body().collect().await.unwrap().to_bytes();
        let openapi: Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(openapi["openapi"], "3.1.0");
        assert!(openapi["paths"].get("/api/v1/actions/media.download").is_some());

        // 2. Hardware info via REST Action API
        let resp = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/api/v1/actions/system.hardware_info")
                    .header("content-type", "application/json")
                    .body(Body::from("{}"))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let bytes = resp.into_body().collect().await.unwrap().to_bytes();
        let body: Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(body["ok"], true);
        assert!(body["output"]["data"]["platform_tag"].is_string());

        // 3. MCP over Streamable HTTP
        let resp = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/mcp")
                    .header("content-type", "application/json")
                    .body(Body::from(r#"{"jsonrpc":"2.0","id":1,"method":"tools/list"}"#))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let bytes = resp.into_body().collect().await.unwrap().to_bytes();
        let mcp_body: Value = serde_json::from_slice(&bytes).unwrap();
        assert!(mcp_body["result"]["tools"].as_array().unwrap().len() >= 15);
    }
}
