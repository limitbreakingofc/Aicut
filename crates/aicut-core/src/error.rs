use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::Value;

/// Structured, AI-friendly error returned by every Aicut action and core service.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, thiserror::Error)]
#[error("[{code}] {message}")]
pub struct AicutError {
    pub code: String,
    pub message: String,
    pub recoverable: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub hint: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub details: Option<Value>,
}

impl AicutError {
    pub fn new(code: impl Into<String>, message: impl Into<String>) -> Self {
        Self {
            code: code.into(),
            message: message.into(),
            recoverable: false,
            hint: None,
            details: None,
        }
    }

    pub fn recoverable(mut self, hint: impl Into<String>) -> Self {
        self.recoverable = true;
        self.hint = Some(hint.into());
        self
    }

    pub fn with_details(mut self, details: Value) -> Self {
        self.details = Some(details);
        self
    }

    pub fn invalid_input(message: impl Into<String>) -> Self {
        Self::new("INVALID_INPUT", message)
            .recoverable("Inspect the action's input_schema and retry with valid arguments.")
    }

    pub fn not_found(message: impl Into<String>) -> Self {
        Self::new("NOT_FOUND", message)
    }

    pub fn module_disabled(module_id: &str) -> Self {
        Self::new(
            "MODULE_DISABLED",
            format!("Module '{module_id}' is currently disabled."),
        )
        .recoverable(format!(
            "Enable module '{module_id}' via system.set_module_enabled before invoking its actions."
        ))
    }

    pub fn dependency_missing(tool_id: &str, reason: impl Into<String>) -> Self {
        Self::new(
            "DEPENDENCY_MISSING",
            format!("Required tool '{tool_id}' is not available: {}", reason.into()),
        )
        .recoverable(format!(
            "Call tools.install with {{\"tool_id\": \"{tool_id}\"}} to install the optimal version for this hardware automatically, then retry."
        ))
    }

    pub fn cancelled(message: impl Into<String>) -> Self {
        Self::new("CANCELLED", message)
    }

    pub fn internal(message: impl Into<String>) -> Self {
        Self::new("INTERNAL_ERROR", message)
    }
}

impl From<rusqlite::Error> for AicutError {
    fn from(err: rusqlite::Error) -> Self {
        Self::internal(format!("Database error: {err}"))
    }
}

impl From<std::io::Error> for AicutError {
    fn from(err: std::io::Error) -> Self {
        Self::internal(format!("I/O error: {err}"))
    }
}

impl From<serde_json::Error> for AicutError {
    fn from(err: serde_json::Error) -> Self {
        Self::invalid_input(format!("JSON serialization/deserialization error: {err}"))
    }
}

pub type AicutResult<T> = Result<T, AicutError>;
