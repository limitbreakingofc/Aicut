use chrono::{DateTime, Utc};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use tokio_util::sync::CancellationToken;
use uuid::Uuid;

use crate::error::{AicutError, AicutResult};
use crate::events::{ActorKind, AicutEvent, EventBus};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum JobStatus {
    Queued,
    Running,
    Completed,
    Failed,
    Cancelled,
}

impl JobStatus {
    pub fn is_terminal(&self) -> bool {
        matches!(self, Self::Completed | Self::Failed | Self::Cancelled)
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, JsonSchema)]
pub struct JobProgress {
    /// Progress percentage in range 0.0..=100.0
    pub percent: f64,
    pub stage: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub downloaded_bytes: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub total_bytes: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub speed_bytes_per_sec: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub eta_seconds: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub message: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct JobRecord {
    pub id: String,
    pub kind: String,
    pub title: String,
    pub module_id: String,
    pub actor: ActorKind,
    pub status: JobStatus,
    pub progress: JobProgress,
    pub logs: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub result: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<AicutError>,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub finished_at: Option<DateTime<Utc>>,
}

struct ManagedJob {
    record: JobRecord,
    cancel_token: CancellationToken,
}

pub struct JobManager {
    jobs: RwLock<HashMap<String, ManagedJob>>,
    events: Arc<EventBus>,
}

impl JobManager {
    pub fn new(events: Arc<EventBus>) -> Self {
        Self {
            jobs: RwLock::new(HashMap::new()),
            events,
        }
    }

    pub fn create_job(
        &self,
        kind: impl Into<String>,
        title: impl Into<String>,
        module_id: impl Into<String>,
        actor: ActorKind,
    ) -> (JobRecord, CancellationToken) {
        let now = Utc::now();
        let id = Uuid::new_v4().to_string();
        let record = JobRecord {
            id: id.clone(),
            kind: kind.into(),
            title: title.into(),
            module_id: module_id.into(),
            actor,
            status: JobStatus::Queued,
            progress: JobProgress {
                percent: 0.0,
                stage: "queued".to_string(),
                ..Default::default()
            },
            logs: Vec::new(),
            result: None,
            error: None,
            created_at: now,
            updated_at: now,
            finished_at: None,
        };
        let cancel_token = CancellationToken::new();

        if let Ok(mut guard) = self.jobs.write() {
            guard.insert(
                id,
                ManagedJob {
                    record: record.clone(),
                    cancel_token: cancel_token.clone(),
                },
            );
        }
        self.events.publish(AicutEvent::JobCreated(record.clone()));
        (record, cancel_token)
    }

    pub fn mark_running(&self, job_id: &str, stage: impl Into<String>) {
        self.mutate(job_id, |job| {
            job.status = JobStatus::Running;
            job.progress.stage = stage.into();
        });
    }

    pub fn update_progress(&self, job_id: &str, progress: JobProgress) {
        self.mutate(job_id, |job| {
            if job.status == JobStatus::Queued {
                job.status = JobStatus::Running;
            }
            job.progress = progress;
        });
    }

    pub fn append_log(&self, job_id: &str, line: impl Into<String>) {
        let line = line.into();
        if line.trim().is_empty() {
            return;
        }
        self.mutate(job_id, |job| {
            if job.logs.len() >= 200 {
                job.logs.remove(0);
            }
            job.logs.push(line);
        });
    }

    pub fn complete(&self, job_id: &str, result: Value) {
        self.mutate(job_id, |job| {
            job.status = JobStatus::Completed;
            job.progress.percent = 100.0;
            job.progress.stage = "completed".to_string();
            job.result = Some(result);
            job.finished_at = Some(Utc::now());
        });
    }

    pub fn fail(&self, job_id: &str, error: AicutError) {
        self.mutate(job_id, |job| {
            job.status = JobStatus::Failed;
            job.progress.stage = "failed".to_string();
            job.error = Some(error);
            job.finished_at = Some(Utc::now());
        });
    }

    pub fn mark_cancelled(&self, job_id: &str) {
        self.mutate(job_id, |job| {
            job.status = JobStatus::Cancelled;
            job.progress.stage = "cancelled".to_string();
            job.finished_at = Some(Utc::now());
        });
    }

    pub fn cancel(&self, job_id: &str) -> AicutResult<JobRecord> {
        let record = {
            let mut guard = self
                .jobs
                .write()
                .map_err(|_| AicutError::internal("Job lock poisoned"))?;
            let entry = guard
                .get_mut(job_id)
                .ok_or_else(|| AicutError::not_found(format!("Job '{job_id}' not found")))?;
            if !entry.record.status.is_terminal() {
                entry.cancel_token.cancel();
                entry.record.status = JobStatus::Cancelled;
                entry.record.progress.stage = "cancelling".to_string();
                entry.record.updated_at = Utc::now();
                entry.record.finished_at = Some(Utc::now());
            }
            entry.record.clone()
        };
        self.events.publish(AicutEvent::JobUpdated(record.clone()));
        Ok(record)
    }

    pub fn get(&self, job_id: &str) -> Option<JobRecord> {
        self.jobs
            .read()
            .ok()
            .and_then(|guard| guard.get(job_id).map(|m| m.record.clone()))
    }

    pub fn list(&self, only_active: bool) -> Vec<JobRecord> {
        let mut items: Vec<JobRecord> = self
            .jobs
            .read()
            .map(|guard| {
                guard
                    .values()
                    .map(|m| m.record.clone())
                    .filter(|j| !only_active || !j.status.is_terminal())
                    .collect()
            })
            .unwrap_or_default();
        items.sort_by(|a, b| b.created_at.cmp(&a.created_at));
        items
    }

    pub fn clear_finished(&self) -> usize {
        if let Ok(mut guard) = self.jobs.write() {
            let before = guard.len();
            guard.retain(|_, v| !v.record.status.is_terminal());
            return before - guard.len();
        }
        0
    }

    fn mutate<F: FnOnce(&mut JobRecord)>(&self, job_id: &str, f: F) {
        let updated = {
            let Ok(mut guard) = self.jobs.write() else {
                return;
            };
            let Some(entry) = guard.get_mut(job_id) else {
                return;
            };
            f(&mut entry.record);
            entry.record.updated_at = Utc::now();
            entry.record.clone()
        };
        self.events.publish(AicutEvent::JobUpdated(updated));
    }
}
