use aicut_core::{ActorKind, WorkspaceConfig};
use aicut_server::{ServerConfig, DEFAULT_HOST, DEFAULT_PORT};
use clap::{Parser, Subcommand};
use serde_json::Value;
use std::path::PathBuf;

#[derive(Parser)]
#[command(
    name = "aicut",
    version,
    about = "Aicut — Modular AI-First Media Studio for Windows 11 (CLI, Daemon & MCP Server)"
)]
struct Cli {
    /// Optional custom workspace directory
    #[arg(long, global = true, env = "AICUT_WORKSPACE_DIR")]
    workspace: Option<PathBuf>,

    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Start the local Aicut HTTP/WebSocket/OpenAPI/MCP daemon
    Serve {
        #[arg(long, default_value = DEFAULT_HOST)]
        host: String,
        #[arg(long, default_value_t = DEFAULT_PORT, env = "AICUT_PORT")]
        port: u16,
        /// Optional static frontend directory to serve alongside the API
        #[arg(long)]
        web_dir: Option<PathBuf>,
    },
    /// Start a native MCP (Model Context Protocol) server over stdio
    Mcp {
        /// Optional running Aicut daemon MCP URL to proxy to so GUI stays in sync
        #[arg(long, default_value = "http://127.0.0.1:43210/mcp")]
        connect: String,
        /// Force standalone embedded kernel instead of proxying to running desktop app
        #[arg(long)]
        standalone: bool,
    },
    /// Execute any registered Aicut action with structured JSON input/output
    Run {
        /// Action ID (e.g. media.probe_url, media.download, media.import_local, tools.list)
        action_id: String,
        /// JSON input payload
        #[arg(long, default_value = "{}")]
        json: String,
    },
    /// List all registered actions (optionally with full JSON Schemas)
    Actions {
        #[arg(long)]
        module: Option<String>,
        #[arg(long)]
        schema: bool,
    },
    /// List all registered modules and their health
    Modules,
    /// Diagnose workspace, hardware profile, SQLite, yt-dlp, ffmpeg, and ffprobe
    Doctor,
}

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt()
        .with_writer(std::io::stderr)
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .init();

    let cli = Cli::parse();
    let ws = match cli.workspace {
        Some(dir) => WorkspaceConfig::from_root(dir),
        None => WorkspaceConfig::default_paths(),
    };

    let kernel = match aicut_runtime::bootstrap_kernel(ws).await {
        Ok(k) => k,
        Err(e) => {
            eprintln!("Failed to initialize Aicut kernel: {e}");
            std::process::exit(1);
        }
    };

    match cli.command {
        Commands::Serve {
            host,
            port,
            web_dir,
        } => {
            let mut cfg = ServerConfig::default();
            cfg.host = host;
            cfg.port = port;
            cfg.web_dir = web_dir;
            if let Err(e) = aicut_server::start_server(kernel, cfg).await {
                eprintln!("Server error: {e}");
                std::process::exit(1);
            }
        }
        Commands::Mcp {
            connect,
            standalone,
        } => {
            let remote = if standalone { None } else { Some(connect) };
            if let Err(e) = aicut_mcp::run_stdio_server(kernel, remote).await {
                eprintln!("MCP stdio server error: {e}");
                std::process::exit(1);
            }
        }
        Commands::Run { action_id, json } => {
            let input: Value = match serde_json::from_str(&json) {
                Ok(v) => v,
                Err(e) => {
                    eprintln!("Invalid JSON payload: {e}");
                    std::process::exit(2);
                }
            };
            match kernel.dispatch(&action_id, input, ActorKind::Ai).await {
                Ok(out) => {
                    println!(
                        "{}",
                        serde_json::to_string_pretty(&serde_json::json!({
                            "ok": true,
                            "output": out
                        }))
                        .unwrap()
                    );
                }
                Err(err) => {
                    println!(
                        "{}",
                        serde_json::to_string_pretty(&serde_json::json!({
                            "ok": false,
                            "error": err
                        }))
                        .unwrap()
                    );
                    std::process::exit(1);
                }
            }
        }
        Commands::Actions { module, schema } => {
            let actions: Vec<_> = kernel
                .list_actions(false)
                .into_iter()
                .filter(|a| module.as_deref().map(|m| a.module_id == m).unwrap_or(true))
                .collect();
            if schema {
                println!("{}", serde_json::to_string_pretty(&actions).unwrap());
            } else {
                let compact: Vec<_> = actions
                    .into_iter()
                    .map(|a| {
                        serde_json::json!({
                            "id": a.id,
                            "module": a.module_id,
                            "title": a.title,
                            "description": a.description
                        })
                    })
                    .collect();
                println!("{}", serde_json::to_string_pretty(&compact).unwrap());
            }
        }
        Commands::Modules => {
            let modules = kernel.list_modules().await;
            println!("{}", serde_json::to_string_pretty(&modules).unwrap());
        }
        Commands::Doctor => {
            let tools = kernel.tools.list_statuses().await;
            let modules = kernel.list_modules().await;
            let report = serde_json::json!({
                "app": "Aicut",
                "version": env!("CARGO_PKG_VERSION"),
                "workspace": kernel.workspace,
                "hardware": kernel.hardware,
                "media_count": kernel.repo.count_media().unwrap_or(0),
                "modules": modules.len(),
                "tools": tools
            });
            println!("{}", serde_json::to_string_pretty(&report).unwrap());
        }
    }
}
