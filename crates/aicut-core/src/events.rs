use chrono::{DateTime, Utc};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use std::sync::RwLock;
use tokio::sync::broadcast;

use crate::jobs::JobRecord;
use crate::media::{MediaAsset, TranscriptRecord};
use crate::tools::ToolStatus;
use crate::ui_state::UiState;

/// Identifies who initiated an action or event — enables full AI vs Human observability.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum ActorKind {
    Human,
    Ai,
    System,
}

impl Default for ActorKind {
    fn default() -> Self {
        Self::System
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum NotificationLevel {
    Info,
    Success,
    Warning,
    Error,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct ActivityEntry {
    pub id: String,
    pub timestamp: DateTime<Utc>,
    pub actor: ActorKind,
    pub action_id: String,
    pub ok: bool,
    pub summary: String,
    pub duration_ms: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", content = "payload", rename_all = "snake_case")]
pub enum AicutEvent {
    JobCreated(JobRecord),
    JobUpdated(JobRecord),
    MediaCreated(MediaAsset),
    MediaUpdated(MediaAsset),
    MediaDeleted { id: String },
    TranscriptCreated(TranscriptRecord),
    TranscriptDeleted { id: String },
    UiStateChanged(UiState),
    UiNotification {
        level: NotificationLevel,
        title: String,
        message: String,
        actor: ActorKind,
    },
    ToolStatusChanged(ToolStatus),
    ModuleStateChanged {
        module_id: String,
        enabled: bool,
    },
    ActionExecuted(ActivityEntry),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EventEnvelope {
    pub seq: u64,
    pub timestamp: DateTime<Utc>,
    pub event: AicutEvent,
}

pub struct EventBus {
    tx: broadcast::Sender<EventEnvelope>,
    seq: std::sync::atomic::AtomicU64,
    recent_activity: RwLock<VecDeque<ActivityEntry>>,
}

impl EventBus {
    pub fn new(capacity: usize) -> Self {
        let (tx, _) = broadcast::channel(capacity);
        Self {
            tx,
            seq: std::sync::atomic::AtomicU64::new(1),
            recent_activity: RwLock::new(VecDeque::with_capacity(200)),
        }
    }

    pub fn publish(&self, event: AicutEvent) {
        if let AicutEvent::ActionExecuted(ref entry) = event {
            if let Ok(mut guard) = self.recent_activity.write() {
                if guard.len() >= 200 {
                    guard.pop_front();
                }
                guard.push_back(entry.clone());
            }
        }
        let seq = self.seq.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let envelope = EventEnvelope {
            seq,
            timestamp: Utc::now(),
            event,
        };
        let _ = self.tx.send(envelope);
    }

    pub fn subscribe(&self) -> broadcast::Receiver<EventEnvelope> {
        self.tx.subscribe()
    }

    pub fn recent_activity(&self, limit: usize) -> Vec<ActivityEntry> {
        match self.recent_activity.read() {
            Ok(guard) => guard.iter().rev().take(limit).cloned().collect(),
            Err(_) => Vec::new(),
        }
    }
}
