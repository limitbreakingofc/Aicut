use chrono::Utc;
use serde_json::Value;
use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use std::time::Instant;
use uuid::Uuid;

use crate::action::{ActionContext, ActionHandler, ActionOutput, ActionSpec};
use crate::error::{AicutError, AicutResult};
use crate::events::{ActivityEntry, ActorKind, AicutEvent, EventBus};
use crate::hardware::HardwareProfile;
use crate::jobs::JobManager;
use crate::media::MediaRepository;
use crate::module::{AicutModule, ModuleInfo};
use crate::tools::ToolManager;
use crate::ui_state::UiStateStore;
use crate::workspace::WorkspaceConfig;

struct RegisteredModule {
    module: Arc<dyn AicutModule>,
    enabled: bool,
    action_ids: Vec<String>,
    tool_ids: Vec<String>,
}

/// Central microkernel shared across GUI (Tauri), HTTP/WS server, MCP server, and CLI.
pub struct AicutKernel {
    pub workspace: WorkspaceConfig,
    pub hardware: HardwareProfile,
    pub events: Arc<EventBus>,
    pub jobs: Arc<JobManager>,
    pub repo: Arc<MediaRepository>,
    pub tools: Arc<ToolManager>,
    pub ui: Arc<UiStateStore>,
    modules: RwLock<HashMap<String, RegisteredModule>>,
    actions: RwLock<HashMap<String, Arc<dyn ActionHandler>>>,
}

impl AicutKernel {
    pub fn new(workspace: WorkspaceConfig) -> AicutResult<Arc<Self>> {
        workspace.ensure_dirs()?;
        let hardware = HardwareProfile::detect();
        let events = Arc::new(EventBus::new(512));
        let jobs = Arc::new(JobManager::new(events.clone()));
        let repo = Arc::new(MediaRepository::open(&workspace.db_path)?);
        let tools = Arc::new(ToolManager::new(
            workspace.bin_dir.clone(),
            hardware.clone(),
            events.clone(),
            jobs.clone(),
        ));
        let ui = Arc::new(UiStateStore::new(events.clone()));

        Ok(Arc::new(Self {
            workspace,
            hardware,
            events,
            jobs,
            repo,
            tools,
            ui,
            modules: RwLock::new(HashMap::new()),
            actions: RwLock::new(HashMap::new()),
        }))
    }

    pub async fn register_module(
        self: &Arc<Self>,
        module: Arc<dyn AicutModule>,
    ) -> AicutResult<()> {
        let manifest = module.manifest();
        let persisted_enabled = self.repo.get_module_enabled(&manifest.id)?;
        let enabled = if manifest.core {
            true
        } else {
            persisted_enabled.unwrap_or(true)
        };

        let declared_tools = module.declared_tools();
        let mut tool_ids = Vec::with_capacity(declared_tools.len());
        for tool in declared_tools {
            tool_ids.push(tool.id.clone());
            self.tools.register_tool(tool);
        }

        let handlers = module.actions();
        let mut action_ids = Vec::with_capacity(handlers.len());
        {
            let mut action_map = self
                .actions
                .write()
                .map_err(|_| AicutError::internal("Action registry lock poisoned"))?;
            for handler in handlers {
                let spec = handler.spec();
                action_ids.push(spec.id.clone());
                action_map.insert(spec.id, handler);
            }
        }

        module.init(self).await?;

        {
            let mut mod_map = self
                .modules
                .write()
                .map_err(|_| AicutError::internal("Module registry lock poisoned"))?;
            mod_map.insert(
                manifest.id,
                RegisteredModule {
                    module,
                    enabled,
                    action_ids,
                    tool_ids,
                },
            );
        }

        Ok(())
    }

    pub fn set_module_enabled(&self, module_id: &str, enabled: bool) -> AicutResult<bool> {
        {
            let mut guard = self
                .modules
                .write()
                .map_err(|_| AicutError::internal("Module lock poisoned"))?;
            let entry = guard
                .get_mut(module_id)
                .ok_or_else(|| AicutError::not_found(format!("Module '{module_id}' not found")))?;
            if entry.module.manifest().core && !enabled {
                return Err(AicutError::invalid_input(format!(
                    "Core module '{module_id}' cannot be disabled."
                )));
            }
            entry.enabled = enabled;
        }
        self.repo.set_module_enabled(module_id, enabled)?;
        self.events.publish(AicutEvent::ModuleStateChanged {
            module_id: module_id.to_string(),
            enabled,
        });
        Ok(enabled)
    }

    pub async fn list_modules(self: &Arc<Self>) -> Vec<ModuleInfo> {
        let snapshot: Vec<(Arc<dyn AicutModule>, bool, Vec<String>, Vec<String>)> = self
            .modules
            .read()
            .map(|guard| {
                guard
                    .values()
                    .map(|m| {
                        (
                            m.module.clone(),
                            m.enabled,
                            m.action_ids.clone(),
                            m.tool_ids.clone(),
                        )
                    })
                    .collect()
            })
            .unwrap_or_default();

        let mut infos = Vec::with_capacity(snapshot.len());
        for (module, enabled, action_ids, tool_ids) in snapshot {
            let health = module.health(self).await;
            infos.push(ModuleInfo {
                manifest: module.manifest(),
                enabled,
                health,
                action_ids,
                tool_ids,
            });
        }
        infos.sort_by(|a, b| a.manifest.id.cmp(&b.manifest.id));
        infos
    }

    pub fn is_module_enabled(&self, module_id: &str) -> bool {
        self.modules
            .read()
            .ok()
            .and_then(|g| g.get(module_id).map(|m| m.enabled))
            .unwrap_or(false)
    }

    pub fn list_actions(&self, include_disabled: bool) -> Vec<ActionSpec> {
        let mut specs: Vec<ActionSpec> = self
            .actions
            .read()
            .map(|guard| {
                guard
                    .values()
                    .map(|h| h.spec())
                    .filter(|s| include_disabled || self.is_module_enabled(&s.module_id))
                    .collect()
            })
            .unwrap_or_default();
        specs.sort_by(|a, b| a.id.cmp(&b.id));
        specs
    }

    pub fn get_action_spec(&self, action_id: &str) -> Option<ActionSpec> {
        self.actions
            .read()
            .ok()
            .and_then(|g| g.get(action_id).map(|h| h.spec()))
    }

    /// Executes any registered action through the unified Action Bus (used identically by GUI, MCP, HTTP, CLI).
    pub async fn dispatch(
        self: &Arc<Self>,
        action_id: &str,
        input: Value,
        actor: ActorKind,
    ) -> AicutResult<ActionOutput> {
        let started = Instant::now();
        let handler = self
            .actions
            .read()
            .ok()
            .and_then(|g| g.get(action_id).cloned())
            .ok_or_else(|| {
                AicutError::not_found(format!("Action '{action_id}' is not registered."))
                    .recoverable("Call system.list_modules or inspect available actions.")
            })?;

        let spec = handler.spec();
        if !self.is_module_enabled(&spec.module_id) {
            return Err(AicutError::module_disabled(&spec.module_id));
        }

        let ctx = ActionContext {
            kernel: self.clone(),
            actor,
        };
        let result = handler.execute(ctx, input).await;
        let duration_ms = started.elapsed().as_millis() as u64;

        let entry = ActivityEntry {
            id: Uuid::new_v4().to_string(),
            timestamp: Utc::now(),
            actor,
            action_id: action_id.to_string(),
            ok: result.is_ok(),
            summary: match &result {
                Ok(out) => out.summary.clone(),
                Err(err) => err.message.clone(),
            },
            duration_ms,
        };
        self.events.publish(AicutEvent::ActionExecuted(entry));

        result
    }
}
