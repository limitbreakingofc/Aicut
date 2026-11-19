pub mod action;
pub mod error;
pub mod events;
pub mod hardware;
pub mod jobs;
pub mod kernel;
pub mod media;
pub mod module;
pub mod tools;
pub mod ui_state;
pub mod workspace;

pub use action::{
    ActionAnnotations, ActionContext, ActionHandler, ActionOutput, ActionSpec, EmptyInput,
    TypedAction,
};
pub use error::{AicutError, AicutResult};
pub use events::{ActivityEntry, ActorKind, AicutEvent, EventBus, EventEnvelope, NotificationLevel};
pub use hardware::{AccelBackend, GpuInfo, GpuVendor, HardwareProfile};
pub use jobs::{JobManager, JobProgress, JobRecord, JobStatus};
pub use kernel::AicutKernel;
pub use media::{
    DownloadPreset, MediaAsset, MediaKind, MediaListFilter, MediaMetadata, MediaRepository,
    MediaSourceType,
};
pub use module::{AicutModule, ModuleHealth, ModuleInfo, ModuleManifest};
pub use tools::{ToolDefinition, ToolInstallStrategy, ToolManager, ToolStatus};
pub use ui_state::{PlayerState, UiState, UiStateStore};
pub use workspace::WorkspaceConfig;
