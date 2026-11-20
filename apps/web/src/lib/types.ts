export type ActorKind = "human" | "ai" | "system";
export type NotifyLevel = "info" | "success" | "warning" | "error";
/** Shows a toast notification (shared by every module view). */
export type NotifyFn = (level: NotifyLevel, title: string, message: string) => void;
export type MediaKind = "video" | "video_only" | "audio" | "subtitle" | "timeline" | "transcript";
export type MediaSourceType = "download" | "local" | "generated";
export type JobStatus = "queued" | "running" | "completed" | "failed" | "cancelled";
export type RouteId =
  | "ingest"
  | "library"
  | "transcribe"
  | "smart-clips"
  | "framing"
  | "motion-studio"
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
  deleted_at?: string | null;
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
  template_path?: string | null;
}

export interface NormalizedRect {
  x: number;
  y: number;
  w: number;
  h: number;
}

export type ContentRegionMode = "ai_director_in_region" | "fixed_region" | "fit_blur_region";

export interface FramingProfile {
  id: string;
  name: string;
  webcam_rect?: NormalizedRect | null;
  content_rect?: NormalizedRect | null;
  webcam_lock_static: boolean;
  content_mode: ContentRegionMode;
  border_trim_pct: number;
  face_zoom: number;
  updated_at: string;
}

export interface FramingGetOutput {
  active_profile: FramingProfile;
  has_media_override: boolean;
  default_profile: FramingProfile;
  presets: FramingProfile[];
}

export interface FramingPreviewOutput {
  media_id: string;
  timestamp_sec: number;
  duration_sec: number;
  width: number;
  height: number;
  image_data_url: string;
  detected_webcam_rect?: NormalizedRect | null;
  detected_content_rect?: NormalizedRect | null;
  detected_face_rect?: NormalizedRect | null;
}

export interface FramingReviewOutput {
  approved: boolean;
  score: number;
  verdict: "otimo" | "ajustar" | "problema" | string;
  reviewer_engine: string;
  has_border_leak: boolean;
  border_sides: string[];
  zoom_assessment: "ideal" | "muito_perto" | "muito_longe" | "sem_rosto" | string;
  head_or_face_clipped: boolean;
  face_height_pct?: number | null;
  summary: string;
  suggestions: string[];
  cropped_webcam_preview_data_url: string;
  suggested_webcam_rect?: NormalizedRect | null;
}

export type SceneLayoutPreset =
  | "split_branded"
  | "split_clean"
  | "content_hero_pip"
  | "content_fullscreen"
  | "camera_focus"
  | "camera_hero_pip"
  | "callout_zoom";

export type SceneTransition = "cut" | "fade" | "slide_up" | "wipe_left";
export type PipCorner = "top_right" | "top_left" | "bottom_right" | "bottom_left";

export interface PipConfig {
  corner: PipCorner;
  width_pct: number;
  border_hex: string;
  visible: boolean;
}

export interface ContentFocusConfig {
  focus_x: number;
  focus_y: number;
  zoom: number;
  fit_blur: boolean;
}

export interface SceneOverlay {
  headline?: string | null;
  badge?: string | null;
  callout_text?: string | null;
  accent_hex: string;
}

export interface CustomMotionCode {
  engine: string;
  code: string;
}

export interface SceneEffects {
  camera_zoom_end: number;
  impact_shake: number;
  vignette: boolean;
  color_mood: string;
}

export interface MotionScene {
  id: string;
  title: string;
  start_sec: number;
  end_sec: number;
  layout: SceneLayoutPreset;
  transition_in: SceneTransition;
  pip: PipConfig;
  content_focus: ContentFocusConfig;
  overlay: SceneOverlay;
  effects?: SceneEffects;
  rationale: string;
  custom_code?: CustomMotionCode | null;
}

export interface MotionChatMessage {
  role: "user" | "assistant" | string;
  text: string;
  created_at: string;
}

export interface AicutComposition {
  id: string;
  title: string;
  media_id: string;
  clip_start_sec: number;
  clip_end_sec: number;
  duration_sec: number;
  width: number;
  height: number;
  fps: number;
  webcam_rect: NormalizedRect;
  content_rect?: NormalizedRect | null;
  burn_subtitles: boolean;
  template_path?: string | null;
  engine_preference: string;
  motion_html?: string | null;
  creative_direction?: string;
  scenes: MotionScene[];
  chat_history: MotionChatMessage[];
  rendered_media_id?: string | null;
  updated_at: string;
}

export interface MotionExportCodeOutput {
  id: string;
  target: string;
  language: string;
  code: string;
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

export interface ResolvedTranscriptSummary {
  id: string;
  source: string;
  language: string;
  model_id: string;
  segment_count: number;
  word_count: number;
  source_asset_id?: string | null;
  source_file_path?: string | null;
}

export interface MediaContextInfo {
  media: MediaAsset;
  external_video_key?: string | null;
  related_assets: MediaAsset[];
  resolved_transcript?: ResolvedTranscriptSummary | null;
  previous_runs_count: number;
}
