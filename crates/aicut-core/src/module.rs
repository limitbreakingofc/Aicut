use async_trait::async_trait;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use std::sync::Arc;

use crate::action::ActionHandler;
use crate::error::AicutResult;
use crate::kernel::AicutKernel;
use crate::tools::ToolDefinition;

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct ModuleManifest {
    pub id: String,
    pub name: String,
    pub version: String,
    pub description: String,
    pub category: String,
    /// Core system modules cannot be disabled.
    pub core: bool,
    pub capabilities: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct ModuleHealth {
    pub healthy: bool,
    pub message: String,
}

impl ModuleHealth {
    pub fn healthy(message: impl Into<String>) -> Self {
        Self {
            healthy: true,
            message: message.into(),
        }
    }

    pub fn degraded(message: impl Into<String>) -> Self {
        Self {
            healthy: false,
            message: message.into(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct ModuleInfo {
    pub manifest: ModuleManifest,
    pub enabled: bool,
    pub health: ModuleHealth,
    pub action_ids: Vec<String>,
    pub tool_ids: Vec<String>,
}

/// Primary extensibility contract for every feature module in Aicut.
#[async_trait]
pub trait AicutModule: Send + Sync {
    fn manifest(&self) -> ModuleManifest;

    /// Actions exposed by this module to the UI, MCP, HTTP/OpenAPI, and CLI.
    fn actions(&self) -> Vec<Arc<dyn ActionHandler>>;

    /// External tools (yt-dlp, ffmpeg, future AI runtimes) owned or required by this module.
    fn declared_tools(&self) -> Vec<ToolDefinition> {
        Vec::new()
    }

    async fn init(&self, _kernel: &Arc<AicutKernel>) -> AicutResult<()> {
        Ok(())
    }

    async fn health(&self, _kernel: &Arc<AicutKernel>) -> ModuleHealth {
        ModuleHealth::healthy("Operational")
    }
}
