use aicut_core::{ActorKind, AicutKernel, MediaListFilter};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::sync::Arc;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};

pub const MCP_PROTOCOL_VERSION: &str = "2024-11-05";

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JsonRpcRequest {
    pub jsonrpc: String,
    #[serde(default)]
    pub id: Option<Value>,
    pub method: String,
    #[serde(default)]
    pub params: Option<Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JsonRpcResponse {
    pub jsonrpc: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub id: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub result: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<JsonRpcError>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JsonRpcError {
    pub code: i64,
    pub message: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub data: Option<Value>,
}

impl JsonRpcResponse {
    pub fn ok(id: Option<Value>, result: Value) -> Self {
        Self {
            jsonrpc: "2.0".to_string(),
            id,
            result: Some(result),
            error: None,
        }
    }

    pub fn err(id: Option<Value>, code: i64, message: impl Into<String>, data: Option<Value>) -> Self {
        Self {
            jsonrpc: "2.0".to_string(),
            id,
            result: None,
            error: Some(JsonRpcError {
                code,
                message: message.into(),
                data,
            }),
        }
    }
}

/// Converts an Aicut action ID ("media.download") to an MCP-safe tool name ("media_download").
pub fn action_id_to_mcp_name(action_id: &str) -> String {
    action_id.replace('.', "_")
}

/// Resolves an MCP tool name back to a registered Aicut action ID.
pub fn resolve_mcp_tool_name(kernel: &AicutKernel, tool_name: &str) -> Option<String> {
    let actions = kernel.list_actions(true);
    for spec in actions {
        if spec.id == tool_name || action_id_to_mcp_name(&spec.id) == tool_name {
            return Some(spec.id);
        }
    }
    None
}

/// Handles a single MCP JSON-RPC 2.0 request against an `AicutKernel`.
/// Returns `None` for notifications (messages without an `id`).
pub async fn handle_mcp_request(
    kernel: &Arc<AicutKernel>,
    req: JsonRpcRequest,
) -> Option<JsonRpcResponse> {
    let is_notification = req.id.is_none();
    let id = req.id.clone();

    let response = match req.method.as_str() {
        "initialize" => {
            let client_version = req
                .params
                .as_ref()
                .and_then(|p| p.get("protocolVersion"))
                .and_then(|v| v.as_str())
                .unwrap_or(MCP_PROTOCOL_VERSION);

            JsonRpcResponse::ok(
                id,
                json!({
                    "protocolVersion": client_version,
                    "capabilities": {
                        "tools": { "listChanged": true },
                        "resources": { "subscribe": false, "listChanged": true },
                        "prompts": { "listChanged": false }
                    },
                    "serverInfo": {
                        "name": "aicut",
                        "version": env!("CARGO_PKG_VERSION")
                    },
                    "instructions": "Aicut is a modular native Linux media & AI studio (Arch/CachyOS & NVIDIA CUDA/NVENC). You have full human-parity control over downloads (yt-dlp), local video/audio imports, the Media Library, transcription, Smart Clips, Motion Studio, background jobs, hardware/tool updates, and live desktop UI navigation."
                }),
            )
        }
        "notifications/initialized" | "notifications/cancelled" => return None,
        "ping" => JsonRpcResponse::ok(id, json!({})),
        "tools/list" => {
            let tools: Vec<Value> = kernel
                .list_actions(false)
                .into_iter()
                .map(|spec| {
                    json!({
                        "name": action_id_to_mcp_name(&spec.id),
                        "description": format!("[{}] {}", spec.id, spec.description),
                        "inputSchema": spec.input_schema,
                        "annotations": {
                            "title": spec.title,
                            "readOnlyHint": spec.annotations.read_only,
                            "destructiveHint": spec.annotations.destructive,
                            "idempotentHint": spec.annotations.idempotent
                        }
                    })
                })
                .collect();
            JsonRpcResponse::ok(id, json!({ "tools": tools }))
        }
        "tools/call" => {
            let params = req.params.unwrap_or(Value::Null);
            let Some(name) = params.get("name").and_then(|v| v.as_str()) else {
                return Some(JsonRpcResponse::err(
                    id,
                    -32602,
                    "Missing required parameter 'name'",
                    None,
                ));
            };
            let arguments = params.get("arguments").cloned().unwrap_or_else(|| json!({}));

            let Some(action_id) = resolve_mcp_tool_name(kernel, name) else {
                return Some(JsonRpcResponse::err(
                    id,
                    -32601,
                    format!("Unknown MCP tool '{name}'"),
                    None,
                ));
            };

            match kernel.dispatch(&action_id, arguments, ActorKind::Ai).await {
                Ok(output) => {
                    let payload = json!({
                        "summary": output.summary,
                        "data": output.data,
                        "next_suggestions": output.next_suggestions
                    });
                    JsonRpcResponse::ok(
                        id,
                        json!({
                            "content": [
                                {
                                    "type": "text",
                                    "text": serde_json::to_string_pretty(&payload).unwrap_or_default()
                                }
                            ],
                            "isError": false
                        }),
                    )
                }
                Err(err) => JsonRpcResponse::ok(
                    id,
                    json!({
                        "content": [
                            {
                                "type": "text",
                                "text": serde_json::to_string_pretty(&err).unwrap_or_else(|_| err.to_string())
                            }
                        ],
                        "isError": true
                    }),
                ),
            }
        }
        "resources/list" => {
            let resources = json!([
                {
                    "uri": "aicut://workspace/summary",
                    "name": "Workspace Summary",
                    "description": "High-level overview of Aicut workspace, hardware, media library count, and active jobs",
                    "mimeType": "application/json"
                },
                {
                    "uri": "aicut://media/library",
                    "name": "Media Library",
                    "description": "All videos and audio assets downloaded via yt-dlp or imported from the PC",
                    "mimeType": "application/json"
                },
                {
                    "uri": "aicut://jobs/active",
                    "name": "Active & Recent Jobs",
                    "description": "Real-time download, import, and tool update job queue",
                    "mimeType": "application/json"
                },
                {
                    "uri": "aicut://tools/status",
                    "name": "Tools & Hardware Status",
                    "description": "Detected hardware profile and status/versions of yt-dlp, ffmpeg, ffprobe, and future tools",
                    "mimeType": "application/json"
                },
                {
                    "uri": "aicut://ui/state",
                    "name": "Live Desktop UI State",
                    "description": "What the human user currently sees and has selected in the Aicut desktop window",
                    "mimeType": "application/json"
                }
            ]);
            JsonRpcResponse::ok(id, json!({ "resources": resources }))
        }
        "resources/read" => {
            let uri = req
                .params
                .as_ref()
                .and_then(|p| p.get("uri"))
                .and_then(|v| v.as_str())
                .unwrap_or("");

            let content = match uri {
                "aicut://workspace/summary" => json!({
                    "workspace": kernel.workspace,
                    "hardware": kernel.hardware,
                    "media_count": kernel.repo.count_media().unwrap_or(0),
                    "active_jobs": kernel.jobs.list(true).len(),
                    "ui_state": kernel.ui.get()
                }),
                "aicut://media/library" => {
                    let items = kernel
                        .repo
                        .list_media(&MediaListFilter::default())
                        .unwrap_or_default();
                    json!({ "total": items.len(), "items": items })
                }
                "aicut://jobs/active" => json!({ "jobs": kernel.jobs.list(false) }),
                "aicut://tools/status" => {
                    let tools = kernel.tools.list_statuses().await;
                    json!({ "hardware": kernel.hardware, "tools": tools })
                }
                "aicut://ui/state" => serde_json::to_value(kernel.ui.get()).unwrap_or(Value::Null),
                _ => {
                    return Some(JsonRpcResponse::err(
                        id,
                        -32602,
                        format!("Unknown resource URI '{uri}'"),
                        None,
                    ));
                }
            };

            JsonRpcResponse::ok(
                id,
                json!({
                    "contents": [
                        {
                            "uri": uri,
                            "mimeType": "application/json",
                            "text": serde_json::to_string_pretty(&content).unwrap_or_default()
                        }
                    ]
                }),
            )
        }
        "prompts/list" => {
            let prompts = json!([
                {
                    "name": "download_video_smart",
                    "description": "Inspect a URL with media_probe_url, pick optimal quality/mode, download it, and display it in the Library",
                    "arguments": [
                        { "name": "url", "description": "Video or audio URL to download", "required": true },
                        { "name": "preference", "description": "Desired mode/quality (e.g., 1080p mp4, video only 4k, mp3 audio)", "required": false }
                    ]
                },
                {
                    "name": "import_pc_media",
                    "description": "Import local video/audio files or a folder from the PC into the Aicut Library",
                    "arguments": [
                        { "name": "path", "description": "File or folder path on the PC", "required": true }
                    ]
                }
            ]);
            JsonRpcResponse::ok(id, json!({ "prompts": prompts }))
        }
        "prompts/get" => {
            let name = req
                .params
                .as_ref()
                .and_then(|p| p.get("name"))
                .and_then(|v| v.as_str())
                .unwrap_or("");
            let args = req
                .params
                .as_ref()
                .and_then(|p| p.get("arguments"))
                .cloned()
                .unwrap_or_else(|| json!({}));

            let text = match name {
                "download_video_smart" => format!(
                    "Use media_probe_url on '{}' first to inspect available formats and subtitles. If yt-dlp is missing, call tools_install with tool_id='yt-dlp'. Then call media_download with wait=true matching preference '{}', and finally call ui_select_media on the created asset so the user sees it in the Library.",
                    args.get("url").and_then(|v| v.as_str()).unwrap_or("<url>"),
                    args.get("preference").and_then(|v| v.as_str()).unwrap_or("best 1080p mp4")
                ),
                "import_pc_media" => format!(
                    "Call media_import_local with paths=['{}'], mode='reference', recursive=true, wait=true. Then call ui_navigate with route='library' so the user immediately sees the imported media.",
                    args.get("path").and_then(|v| v.as_str()).unwrap_or("<path>")
                ),
                _ => {
                    return Some(JsonRpcResponse::err(
                        id,
                        -32602,
                        format!("Unknown prompt '{name}'"),
                        None,
                    ));
                }
            };

            JsonRpcResponse::ok(
                id,
                json!({
                    "messages": [
                        {
                            "role": "user",
                            "content": { "type": "text", "text": text }
                        }
                    ]
                }),
            )
        }
        other => JsonRpcResponse::err(id, -32601, format!("Method not found: '{other}'"), None),
    };

    if is_notification {
        None
    } else {
        Some(response)
    }
}

/// Runs an MCP stdio server. If `remote_mcp_url` is reachable (e.g., the live Linux desktop
/// app at `http://127.0.0.1:43210/mcp`), requests are transparently proxied to the open GUI so
/// humans see every AI action live. Otherwise, requests execute against the embedded `kernel`.
pub async fn run_stdio_server(
    kernel: Arc<AicutKernel>,
    remote_mcp_url: Option<String>,
) -> std::io::Result<()> {
    let http = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(3600))
        .build()
        .unwrap_or_else(|_| reqwest::Client::new());

    let use_remote = if let Some(ref url) = remote_mcp_url {
        let health_url = url.trim_end_matches("/mcp").to_string() + "/health";
        http.get(&health_url)
            .timeout(std::time::Duration::from_millis(600))
            .send()
            .await
            .map(|r| r.status().is_success())
            .unwrap_or(false)
    } else {
        false
    };

    let stdin = BufReader::new(tokio::io::stdin());
    let mut lines = stdin.lines();
    let mut stdout = tokio::io::stdout();

    while let Some(line) = lines.next_line().await? {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        let req: JsonRpcRequest = match serde_json::from_str(trimmed) {
            Ok(r) => r,
            Err(e) => {
                let err_resp = JsonRpcResponse::err(None, -32700, format!("Parse error: {e}"), None);
                let out = serde_json::to_string(&err_resp).unwrap_or_default();
                stdout.write_all(out.as_bytes()).await?;
                stdout.write_all(b"\n").await?;
                stdout.flush().await?;
                continue;
            }
        };

        let resp_opt = if use_remote {
            if let Some(ref url) = remote_mcp_url {
                proxy_mcp_http(&http, url, &req).await
            } else {
                handle_mcp_request(&kernel, req).await
            }
        } else {
            handle_mcp_request(&kernel, req).await
        };

        if let Some(resp) = resp_opt {
            let out = serde_json::to_string(&resp).unwrap_or_default();
            stdout.write_all(out.as_bytes()).await?;
            stdout.write_all(b"\n").await?;
            stdout.flush().await?;
        }
    }

    Ok(())
}

async fn proxy_mcp_http(
    http: &reqwest::Client,
    url: &str,
    req: &JsonRpcRequest,
) -> Option<JsonRpcResponse> {
    let is_notif = req.id.is_none();
    match http.post(url).json(req).send().await {
        Ok(r) => {
            if is_notif || r.status() == reqwest::StatusCode::ACCEPTED || r.status() == reqwest::StatusCode::NO_CONTENT {
                None
            } else {
                r.json::<JsonRpcResponse>().await.ok()
            }
        }
        Err(e) => {
            if is_notif {
                None
            } else {
                Some(JsonRpcResponse::err(
                    req.id.clone(),
                    -32000,
                    format!("Failed forwarding MCP request to Aicut desktop daemon: {e}"),
                    None,
                ))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use aicut_core::WorkspaceConfig;

    #[tokio::test]
    async fn mcp_initialize_list_and_call_tools() {
        let tmp = tempfile::tempdir().unwrap();
        let kernel = aicut_runtime::bootstrap_kernel(WorkspaceConfig::from_root(tmp.path()))
            .await
            .unwrap();

        let init = handle_mcp_request(
            &kernel,
            JsonRpcRequest {
                jsonrpc: "2.0".into(),
                id: Some(json!(1)),
                method: "initialize".into(),
                params: Some(json!({"protocolVersion": "2024-11-05"})),
            },
        )
        .await
        .unwrap();
        assert!(init.error.is_none());
        assert_eq!(init.result.unwrap()["serverInfo"]["name"], "aicut");

        let list = handle_mcp_request(
            &kernel,
            JsonRpcRequest {
                jsonrpc: "2.0".into(),
                id: Some(json!(2)),
                method: "tools/list".into(),
                params: None,
            },
        )
        .await
        .unwrap();
        let tools = list.result.unwrap()["tools"].as_array().unwrap().clone();
        assert!(tools.iter().any(|t| t["name"] == "media_download"));
        assert!(tools.iter().any(|t| t["name"] == "media_import_local"));
        assert!(tools.iter().any(|t| t["name"] == "tools_list"));
        assert!(tools.iter().any(|t| t["name"] == "ui_navigate"));

        let call = handle_mcp_request(
            &kernel,
            JsonRpcRequest {
                jsonrpc: "2.0".into(),
                id: Some(json!(3)),
                method: "tools/call".into(),
                params: Some(json!({
                    "name": "ui_navigate",
                    "arguments": { "route": "library" }
                })),
            },
        )
        .await
        .unwrap();
        assert_eq!(call.result.unwrap()["isError"], false);
        assert_eq!(kernel.ui.get().active_route, "library");
    }
}
