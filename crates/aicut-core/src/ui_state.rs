use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use std::sync::{Arc, RwLock};

use crate::events::{AicutEvent, EventBus};

#[derive(Debug, Clone, Default, Serialize, Deserialize, JsonSchema)]
pub struct PlayerState {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub media_id: Option<String>,
    pub playing: bool,
    pub current_time_sec: f64,
    pub duration_sec: Option<f64>,
}

/// Synchronized live UI state readable and controllable by both humans and AI agents.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct UiState {
    /// Active navigation tab: "ingest" | "library" | "jobs" | "modules" | "ai-connect" | "settings"
    pub active_route: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub selected_media_id: Option<String>,
    pub search_query: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub active_filter_kind: Option<String>,
    pub player: PlayerState,
}

impl Default for UiState {
    fn default() -> Self {
        Self {
            active_route: "ingest".to_string(),
            selected_media_id: None,
            search_query: String::new(),
            active_filter_kind: None,
            player: PlayerState::default(),
        }
    }
}

pub struct UiStateStore {
    state: RwLock<UiState>,
    events: Arc<EventBus>,
}

impl UiStateStore {
    pub fn new(events: Arc<EventBus>) -> Self {
        Self {
            state: RwLock::new(UiState::default()),
            events,
        }
    }

    pub fn get(&self) -> UiState {
        self.state
            .read()
            .map(|g| g.clone())
            .unwrap_or_default()
    }

    pub fn update<F: FnOnce(&mut UiState)>(&self, f: F) -> UiState {
        let next = {
            let mut guard = self.state.write().expect("UiState lock poisoned");
            f(&mut guard);
            guard.clone()
        };
        self.events.publish(AicutEvent::UiStateChanged(next.clone()));
        next
    }

    pub fn replace(&self, next: UiState) -> UiState {
        self.update(|s| *s = next)
    }
}
