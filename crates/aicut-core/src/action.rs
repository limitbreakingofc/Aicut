use async_trait::async_trait;
use schemars::{schema_for, JsonSchema};
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::future::Future;
use std::marker::PhantomData;
use std::pin::Pin;
use std::sync::Arc;

use crate::error::{AicutError, AicutResult};
use crate::events::ActorKind;
use crate::kernel::AicutKernel;

/// MCP-aligned behavioral hints so AI agents know whether an action is safe, read-only, or destructive.
#[derive(Debug, Clone, Default, Serialize, Deserialize, JsonSchema)]
pub struct ActionAnnotations {
    pub read_only: bool,
    pub destructive: bool,
    pub idempotent: bool,
    pub long_running: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct ActionSpec {
    /// Namespaced ID (e.g. "media.download", "ui.navigate", "tools.install").
    pub id: String,
    pub module_id: String,
    pub title: String,
    pub description: String,
    pub input_schema: Value,
    pub output_schema: Value,
    pub annotations: ActionAnnotations,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct ActionOutput {
    pub data: Value,
    /// Human- and LLM-readable concise summary of what happened.
    pub summary: String,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub next_suggestions: Vec<String>,
}

impl ActionOutput {
    pub fn new(data: Value, summary: impl Into<String>) -> Self {
        Self {
            data,
            summary: summary.into(),
            next_suggestions: Vec::new(),
        }
    }

    pub fn with_suggestions(mut self, suggestions: Vec<String>) -> Self {
        self.next_suggestions = suggestions;
        self
    }
}

#[derive(Clone)]
pub struct ActionContext {
    pub kernel: Arc<AicutKernel>,
    pub actor: ActorKind,
}

#[async_trait]
pub trait ActionHandler: Send + Sync {
    fn spec(&self) -> ActionSpec;
    async fn execute(&self, ctx: ActionContext, input: Value) -> AicutResult<ActionOutput>;
}

type BoxFuture<'a, T> = Pin<Box<dyn Future<Output = T> + Send + 'a>>;

/// Ergonomic typed action wrapper that automatically derives JSON Schema from Rust types.
pub struct TypedAction<I, O, F>
where
    I: DeserializeOwned + JsonSchema + Send + Sync + 'static,
    O: Serialize + JsonSchema + Send + Sync + 'static,
    F: Fn(ActionContext, I) -> BoxFuture<'static, AicutResult<(O, String, Vec<String>)>>
        + Send
        + Sync
        + 'static,
{
    spec: ActionSpec,
    handler: F,
    _marker: PhantomData<fn(I) -> O>,
}

impl<I, O, F> TypedAction<I, O, F>
where
    I: DeserializeOwned + JsonSchema + Send + Sync + 'static,
    O: Serialize + JsonSchema + Send + Sync + 'static,
    F: Fn(ActionContext, I) -> BoxFuture<'static, AicutResult<(O, String, Vec<String>)>>
        + Send
        + Sync
        + 'static,
{
    pub fn new(
        id: impl Into<String>,
        module_id: impl Into<String>,
        title: impl Into<String>,
        description: impl Into<String>,
        annotations: ActionAnnotations,
        handler: F,
    ) -> Self {
        let input_schema = serde_json::to_value(schema_for!(I)).unwrap_or(Value::Null);
        let output_schema = serde_json::to_value(schema_for!(O)).unwrap_or(Value::Null);
        Self {
            spec: ActionSpec {
                id: id.into(),
                module_id: module_id.into(),
                title: title.into(),
                description: description.into(),
                input_schema,
                output_schema,
                annotations,
            },
            handler,
            _marker: PhantomData,
        }
    }
}

#[async_trait]
impl<I, O, F> ActionHandler for TypedAction<I, O, F>
where
    I: DeserializeOwned + JsonSchema + Send + Sync + 'static,
    O: Serialize + JsonSchema + Send + Sync + 'static,
    F: Fn(ActionContext, I) -> BoxFuture<'static, AicutResult<(O, String, Vec<String>)>>
        + Send
        + Sync
        + 'static,
{
    fn spec(&self) -> ActionSpec {
        self.spec.clone()
    }

    async fn execute(&self, ctx: ActionContext, input: Value) -> AicutResult<ActionOutput> {
        let normalized = if input.is_null() {
            serde_json::json!({})
        } else {
            input
        };
        let typed_input: I = serde_json::from_value(normalized).map_err(|e| {
            AicutError::invalid_input(format!(
                "Invalid parameters for action '{}': {e}",
                self.spec.id
            ))
        })?;

        let (output, summary, suggestions) = (self.handler)(ctx, typed_input).await?;
        let data = serde_json::to_value(output)?;
        Ok(ActionOutput {
            data,
            summary,
            next_suggestions: suggestions,
        })
    }
}

/// Empty parameter struct for actions that require no inputs.
#[derive(Debug, Clone, Default, Serialize, Deserialize, JsonSchema)]
pub struct EmptyInput {}
