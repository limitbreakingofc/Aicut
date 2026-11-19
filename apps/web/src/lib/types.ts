export type ActorKind = "human" | "ai" | "system";
export type MediaKind = "video" | "video_only" | "audio" | "subtitle" | "timeline" | "transcript";
export type MediaSourceType = "download" | "local" | "generated";
export type JobStatus = "queued" | "running" | "completed" | "failed" | "cancelled";
export type RouteId =
  | "ingest"
  | "library"
  | "transcribe"
  | "smart-clips"
  | "jobs"
  | "modules"
  | "ai-connect"
  | "settings";

export interface AicutError {
  code: string;
  message: string;
  recoverable: boolean;
  hint?: string;
}

export interface ActionOutput<T = unknown> {
  data: T;
  summary: string;
  next_suggestions?: string[];
}

export interface MediaMetadata {
  duration_sec?: number;
  width?: number;
  height?: number;
  fps?: number;
  video_codec?: string;
  audio_codec?: string;
  audio_channels?: number;
  sample_rate?: number;
  bitrate_bps?: number;
  format_name?: string;
  uploader?: string;
  extractor?: string;
  has_subtitles?: boolean;
  language?: string;
  model_id?: string;
  transcript_id?: string;
}

export interface MediaAsset {
  id: string;
  title: string;
  kind: MediaKind;
  source_type: MediaSourceType;
  source_url?: string;
  parent_media_id?: string | null;
  file_path: string;
  file_name: string;
  extension: string;
  file_size_bytes: number;
  fingerprint?: string;
  thumbnail_path?: string;
  metadata: MediaMetadata;
  tags: string[];
  notes: string;
  created_at: string;
  updated_at: string;
}

export interface TranscriptWord {
  start_ms: number;
  end_ms: number;
  text: string;
  probability?: number;
}

export interface TranscriptSegment {
  index: number;
  start_ms: number;
  end_ms: number;
  text: string;
  words?: TranscriptWord[];
}

export interface TranscriptRecord {
  id: string;
  media_id: string;
  media_title: string;
  model_id: string;
  language: string;
  duration_sec?: number;
  full_text: string;
  segments: TranscriptSegment[];
  created_at: string;
}

export interface WhisperModelStatus {
  id: string;
  family_id: string;
  family_name: string;
  version_tag: string;
  display_name: string;
  filename: string;
  size_mb: number;
  min_vram_mb: number;
  min_ram_mb: number;
  quality_score: number;
  quality_tier: string;
  speed_tier: string;
  quality_notes: string;
  description: string;
  installed: boolean;
  compatible: boolean;
  compatibility_reason: string;
  curated_for_hardware: boolean;
  recommended: boolean;
  best_quality_for_hardware: boolean;
  file_path?: string;
  actual_size_bytes?: number;
}

export interface ExportFormatInfo {
  id: string;
  name: string;
  extension: string;
  category: "subtitle" | "timeline" | "transcript";
  description: string;
}

export interface ProbedFormat {
  format_id: string;
  ext: string;
  kind: "video_audio" | "video_only" | "audio_only";
  resolution?: string;
  height?: number;
  fps?: number;
  vcodec?: string;
  acodec?: string;
  filesize_bytes?: number;
  tbr_kbps?: number;
  note?: string;
}

export interface ProbeUrlResult {
  url: string;
  title: string;
  uploader?: string;
  extractor?: string;
  duration_sec?: number;
  thumbnail?: string;
  description?: string;
  is_playlist: boolean;
  playlist_count: number;
  chapters_count: number;
  available_subtitles: string[];
  formats: ProbedFormat[];
}

export interface DownloadPreset {
  id: string;
  name: string;
  description: string;
  config: Record<string, unknown>;
  created_at: string;
}

export interface JobProgress {
  percent: number;
  stage: string;
  downloaded_bytes?: number;
  total_bytes?: number;
  speed_bytes_per_sec?: number;
  eta_seconds?: number;
  message?: string;
}

export interface JobRecord {
  id: string;
  kind: string;
  title: string;
  module_id: string;
  actor: ActorKind;
  status: JobStatus;
  progress: JobProgress;
  logs: string[];
  result?: unknown;
  error?: AicutError;
  created_at: string;
  updated_at: string;
  finished_at?: string;
}

export interface GpuInfo {
  name: string;
  vendor: "nvidia" | "amd" | "intel" | "apple" | "unknown";
  vram_mb?: number;
  driver_version?: string;
}

export interface HardwareProfile {
  os: string;
  os_version: string;
  arch: string;
  platform_tag: string;
  cpu_brand: string;
  physical_cores: number;
  logical_cores: number;
  total_memory_mb: number;
  available_memory_mb: number;
  gpus: GpuInfo[];
  cuda_version?: string;
  recommended_accel: "cuda_nvenc" | "amf_directml" | "qsv_openvino" | "metal" | "cpu_only";
  recommendation_summary: string;
}

export interface ToolStatus {
  id: string;
  display_name: string;
  description: string;
  owner_module: string;
  category: string;
  installed: boolean;
  installed_version?: string;
  latest_version?: string;
  update_available: boolean;
  binary_path?: string;
  managed_by_aicut: boolean;
  recommended_build: string;
  hardware_note: string;
}

export interface ModuleManifest {
  id: string;
  name: string;
  version: string;
  description: string;
  category: string;
  core: boolean;
  capabilities: string[];
}

export interface ModuleInfo {
  manifest: ModuleManifest;
  enabled: boolean;
  health: { healthy: boolean; message: string };
  action_ids: string[];
  tool_ids: string[];
}

export interface ActionSpec {
  id: string;
  module_id: string;
  title: string;
  description: string;
  input_schema: Record<string, unknown>;
  output_schema: Record<string, unknown>;
  annotations: {
    read_only: boolean;
    destructive: boolean;
    idempotent: boolean;
    long_running: boolean;
  };
}

export interface ActivityEntry {
  id: string;
  timestamp: string;
  actor: ActorKind;
  action_id: string;
  ok: boolean;
  summary: string;
  duration_ms: number;
}

export interface UiState {
  active_route: RouteId;
  selected_media_id?: string | null;
  search_query: string;
  active_filter_kind?: string | null;
  player: {
    media_id?: string | null;
    playing: boolean;
    current_time_sec: number;
    duration_sec?: number | null;
  };
}

export interface WorkspaceConfig {
  workspace_dir: string;
  downloads_dir: string;
  imports_dir: string;
  exports_dir: string;
  thumbnails_dir: string;
  models_dir: string;
  bin_dir: string;
  db_path: string;
}

export interface VisionModelStatus {
  id: string;
  family_id: string;
  family_name: string;
  version_tag: string;
  display_name: string;
  hf_repo: string;
  model_filename: string;
  mmproj_filename: string;
  size_mb: number;
  min_vram_mb: number;
  min_ram_mb: number;
  quality_score: number;
  quality_tier: string;
  speed_tier: string;
  quality_notes: string;
  description: string;
  installed: boolean;
  compatible: boolean;
  compatibility_reason: string;
  curated_for_hardware: boolean;
  recommended: boolean;
  best_quality_for_hardware: boolean;
  model_path?: string;
  mmproj_path?: string;
  actual_size_bytes?: number;
}

export interface ClipsConfig {
  gemini_configured: boolean;
  gemini_key_preview?: string | null;
  default_provider: "gemini" | "local_free" | string;
  gemini_scan_model: string;
  gemini_director_model: string;
  local_vision_model: string;
}

export interface SmartClipSegment {
  start_sec: number;
  end_sec: number;
  role: string;
}

export interface ClipNarrativeArc {
  beginning: string;
  middle: string;
  end: string;
}

export interface ClipScores {
  match_description: number;
  visual_interest: number;
  narrative_coherence: number;
  hook_strength: number;
}

export interface SmartClipPlan {
  id: string;
  title: string;
  hook_first_5s: string;
  visual_evidence: string;
  why_interesting: string;
  narrative_arc: ClipNarrativeArc;
  segments: SmartClipSegment[];
  total_duration_sec: number;
  scores: ClipScores;
  rendered_media_id?: string | null;
}

export interface SmartClipRun {
  id: string;
  media_id: string;
  media_title: string;
  provider: string;
  model_id: string;
  description: string;
  max_duration_sec: number;
  clips: SmartClipPlan[];
  created_at: string;
}
