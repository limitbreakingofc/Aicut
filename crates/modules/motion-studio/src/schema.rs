use aicut_core::{AicutError, AicutKernel, AicutResult};
use aicut_module_smart_clips::framing::NormalizedRect;
use chrono::Utc;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

pub const SETTING_MOTION_COMPOSITIONS: &str = "motion_studio_compositions";

/// Layout preset for an individual scene inside an `AicutComposition`.
/// AI can dynamically switch between these scene by scene as the video unfolds.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum SceneLayoutPreset {
    /// Branded 9:16 template (SMOKING33): clean webcam on TOP + content on BOTTOM.
    SplitBranded,
    /// Borderless 9:16 split: clean webcam on TOP + content on BOTTOM.
    SplitClean,
    /// Full-bleed 9:16 content hero + clean floating PiP webcam in a chosen corner.
    ContentHeroPip,
    /// 100% content focus (full screen, webcam temporarily hidden so nothing obscures key visuals).
    ContentFullscreen,
    /// Full-screen clean webcam focus — ideal for opening 3–4s hooks or intense reactions.
    CameraFocus,
    /// Large webcam on top (~68%) + compact content strip on bottom (~32%) for reaction moments.
    CameraHeroPip,
    /// Focal zoom onto a specific `(focus_x, focus_y)` region + callout highlight ring + optional PiP webcam.
    CalloutZoom,
}

impl Default for SceneLayoutPreset {
    fn default() -> Self {
        Self::SplitBranded
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum SceneTransition {
    Cut,
    Fade,
    SlideUp,
    WipeLeft,
}

impl Default for SceneTransition {
    fn default() -> Self {
        Self::Cut
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum PipCorner {
    TopRight,
    TopLeft,
    BottomRight,
    BottomLeft,
}

impl Default for PipCorner {
    fn default() -> Self {
        Self::TopRight
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct PipConfig {
    #[serde(default)]
    pub corner: PipCorner,
    /// Width of the floating webcam PiP relative to 1080px canvas (0.24..0.48, default 0.36).
    #[serde(default = "default_pip_width")]
    pub width_pct: f64,
    #[serde(default = "default_gold_hex")]
    pub border_hex: String,
    #[serde(default = "default_true")]
    pub visible: bool,
}

impl Default for PipConfig {
    fn default() -> Self {
        Self {
            corner: PipCorner::TopRight,
            width_pct: default_pip_width(),
            border_hex: default_gold_hex(),
            visible: true,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct ContentFocusConfig {
    /// Horizontal focus point in normalized `[0, 1]` coordinates (0.5 = center).
    #[serde(default = "default_half")]
    pub focus_x: f64,
    /// Vertical focus point in normalized `[0, 1]` coordinates (0.5 = center).
    #[serde(default = "default_half")]
    pub focus_y: f64,
    /// Focal zoom multiplier (`1.0` = normal, `1.5`..`2.6` = detail zoom for `callout_zoom`).
    #[serde(default = "default_one")]
    pub zoom: f64,
    /// Keep full 16:9 width visible over soft blurred background instead of cropping sides.
    #[serde(default)]
    pub fit_blur: bool,
}

impl Default for ContentFocusConfig {
    fn default() -> Self {
        Self {
            focus_x: 0.5,
            focus_y: 0.5,
            zoom: 1.0,
            fit_blur: false,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct SceneOverlay {
    /// Optional bold kinetic headline / hook title displayed near top of the scene.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub headline: Option<String>,
    /// Optional compact badge tag (e.g. "MOMENTO CHAVE", "DESTAQUE", "REAÇÃO").
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub badge: Option<String>,
    /// Optional callout label drawn next to the focused region on `callout_zoom` scenes.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub callout_text: Option<String>,
    #[serde(default = "default_gold_hex")]
    pub accent_hex: String,
}

impl Default for SceneOverlay {
    fn default() -> Self {
        Self {
            headline: None,
            badge: None,
            callout_text: None,
            accent_hex: default_gold_hex(),
        }
    }
}

/// Dynamic camera & visual editing techniques applied automatically by the AI per scene context.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct SceneEffects {
    /// Smooth zoom-in over the scene (1.0 = steady, 1.08 = subtle punch-in, 1.18 = dramatic zoom).
    #[serde(default = "default_one")]
    pub camera_zoom_end: f64,
    /// Brief energetic camera shake on scene entry (0.0 = off, 0.3..1.0 = impact shake).
    #[serde(default)]
    pub impact_shake: f64,
    /// Subtle dark cinema vignette for tension / storytelling emphasis.
    #[serde(default)]
    pub vignette: bool,
    /// Visual color mood: "natural" | "vibrant" | "dramatic" | "cyber"
    #[serde(default = "default_mood")]
    pub color_mood: String,
}

impl Default for SceneEffects {
    fn default() -> Self {
        Self {
            camera_zoom_end: 1.0,
            impact_shake: 0.0,
            vignette: false,
            color_mood: default_mood(),
        }
    }
}

fn default_mood() -> String {
    "natural".to_string()
}

/// Optional custom motion code slot for Phase 2 (HyperFrames HTML/CSS/GSAP) & Phase 3 (Remotion TSX).
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct CustomMotionCode {
    /// "hyperframes_html" | "remotion_tsx"
    pub engine: String,
    pub code: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct MotionScene {
    #[serde(default = "default_scene_id")]
    pub id: String,
    #[serde(default = "default_scene_title")]
    pub title: String,
    /// Clip-relative start timestamp in seconds (`0.0..duration_sec`).
    pub start_sec: f64,
    /// Clip-relative end timestamp in seconds (`0.0..duration_sec`).
    pub end_sec: f64,
    #[serde(default)]
    pub layout: SceneLayoutPreset,
    #[serde(default)]
    pub transition_in: SceneTransition,
    #[serde(default)]
    pub pip: PipConfig,
    #[serde(default)]
    pub content_focus: ContentFocusConfig,
    #[serde(default)]
    pub overlay: SceneOverlay,
    /// Context-aware camera & visual techniques (zoom punch-in, shake, vignette, mood).
    #[serde(default)]
    pub effects: SceneEffects,
    /// Short explanation of why the AI chose this layout for this moment.
    #[serde(default)]
    pub rationale: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub custom_code: Option<CustomMotionCode>,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct MotionChatMessage {
    /// "user" | "assistant"
    pub role: String,
    pub text: String,
    #[serde(default = "default_now")]
    pub created_at: String,
}

/// Canonical Aicut Motion Scene Graph:
/// Controls dynamic multi-scene layouts, transitions, overlays, and future HyperFrames/Remotion rendering.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct AicutComposition {
    #[serde(default = "default_comp_id")]
    pub id: String,
    pub title: String,
    pub media_id: String,
    /// Absolute start offset in the source video (seconds).
    #[serde(default)]
    pub clip_start_sec: f64,
    /// Absolute end offset in the source video (seconds).
    pub clip_end_sec: f64,
    /// Clip duration (`clip_end_sec - clip_start_sec`).
    pub duration_sec: f64,
    #[serde(default = "default_width")]
    pub width: u32,
    #[serde(default = "default_height")]
    pub height: u32,
    #[serde(default = "default_fps")]
    pub fps: u32,
    /// Clean borderless webcam region inherited from the Framing Studio tab (normalized 0..1).
    #[serde(default = "default_webcam_rect")]
    pub webcam_rect: NormalizedRect,
    /// Optional content bounding region (normalized 0..1).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub content_rect: Option<NormalizedRect>,
    #[serde(default = "default_true")]
    pub burn_subtitles: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub template_path: Option<String>,
    /// "hybrid_nvenc" | "hyperframes" | "remotion"
    #[serde(default = "default_engine_pref")]
    pub engine_preference: String,
    /// Bespoke HyperFrames HTML/CSS/SVG/JS motion overlay authored by the AI specifically for THIS video.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub motion_html: Option<String>,
    /// High-level creative editing direction & rationale crafted by the AI for this video.
    #[serde(default)]
    pub creative_direction: String,
    #[serde(default)]
    pub scenes: Vec<MotionScene>,
    #[serde(default)]
    pub chat_history: Vec<MotionChatMessage>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rendered_media_id: Option<String>,
    #[serde(default = "default_now")]
    pub updated_at: String,
}

impl AicutComposition {
    /// Enforces a rock-solid gapless, non-overlapping chronological timeline covering `[0.0, duration_sec]`.
    pub fn normalize(&mut self) {
        self.duration_sec = (self.clip_end_sec - self.clip_start_sec).max(1.0);
        self.webcam_rect = self.webcam_rect.clamped();
        self.content_rect = self.content_rect.map(NormalizedRect::clamped);
        if self.width < 64 {
            self.width = 1080;
        }
        if self.height < 64 {
            self.height = 1920;
        }
        if self.fps == 0 {
            self.fps = 30;
        }

        if self.scenes.is_empty() {
            self.scenes.push(MotionScene {
                id: default_scene_id(),
                title: "Cena principal".to_string(),
                start_sec: 0.0,
                end_sec: self.duration_sec,
                layout: SceneLayoutPreset::SplitBranded,
                transition_in: SceneTransition::Cut,
                pip: PipConfig::default(),
                content_focus: ContentFocusConfig::default(),
                overlay: SceneOverlay::default(),
                effects: SceneEffects::default(),
                rationale: "Layout padrão SMOKING33.".to_string(),
                custom_code: None,
            });
            return;
        }

        let dur = self.duration_sec;
        for s in &mut self.scenes {
            if s.id.trim().is_empty() {
                s.id = default_scene_id();
            }
            s.start_sec = s.start_sec.clamp(0.0, dur);
            s.end_sec = s.end_sec.clamp(0.0, dur);
            if s.end_sec < s.start_sec {
                std::mem::swap(&mut s.start_sec, &mut s.end_sec);
            }
            s.pip.width_pct = s.pip.width_pct.clamp(0.22, 0.52);
            s.content_focus.focus_x = s.content_focus.focus_x.clamp(0.05, 0.95);
            s.content_focus.focus_y = s.content_focus.focus_y.clamp(0.05, 0.95);
            s.content_focus.zoom = s.content_focus.zoom.clamp(1.0, 3.0);
        }

        self.scenes.sort_by(|a, b| {
            a.start_sec
                .partial_cmp(&b.start_sec)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Drop degenerate scenes (< 0.4s) if others exist
        if self.scenes.len() > 1 {
            self.scenes.retain(|s| (s.end_sec - s.start_sec) >= 0.4);
        }
        if self.scenes.is_empty() {
            self.scenes.push(MotionScene {
                id: default_scene_id(),
                title: "Cena principal".to_string(),
                start_sec: 0.0,
                end_sec: dur,
                layout: SceneLayoutPreset::SplitBranded,
                transition_in: SceneTransition::Cut,
                pip: PipConfig::default(),
                content_focus: ContentFocusConfig::default(),
                overlay: SceneOverlay::default(),
                effects: SceneEffects::default(),
                rationale: String::new(),
                custom_code: None,
            });
            return;
        }

        // Stitch boundaries seamlessly so timeline covers [0.0 .. duration_sec] with zero gaps/overlaps
        self.scenes[0].start_sec = 0.0;
        let len = self.scenes.len();
        for i in 0..len {
            if i + 1 < len {
                let min_end = (self.scenes[i].start_sec + 0.5).min(dur - 0.5 * (len - 1 - i) as f64);
                let boundary = self.scenes[i].end_sec.max(min_end).min(dur);
                self.scenes[i].end_sec = round2(boundary);
                self.scenes[i + 1].start_sec = self.scenes[i].end_sec;
            } else {
                self.scenes[i].end_sec = round2(dur);
            }
        }
        self.scenes.retain(|s| (s.end_sec - s.start_sec) >= 0.35);
        if let Some(last) = self.scenes.last_mut() {
            last.end_sec = round2(dur);
        }
        self.updated_at = Utc::now().to_rfc3339();
    }
}

fn round2(v: f64) -> f64 {
    (v * 100.0).round() / 100.0
}

fn default_comp_id() -> String {
    Uuid::new_v4().to_string()
}
fn default_scene_id() -> String {
    format!("sc-{}", &Uuid::new_v4().to_string()[..6])
}
fn default_scene_title() -> String {
    "Cena".to_string()
}
fn default_pip_width() -> f64 {
    0.36
}
fn default_gold_hex() -> String {
    "#EAB308".to_string()
}
fn default_half() -> f64 {
    0.5
}
fn default_one() -> f64 {
    1.0
}
fn default_true() -> bool {
    true
}
fn default_width() -> u32 {
    1080
}
fn default_height() -> u32 {
    1920
}
fn default_fps() -> u32 {
    30
}
fn default_engine_pref() -> String {
    "hybrid_nvenc".to_string()
}
fn default_now() -> String {
    Utc::now().to_rfc3339()
}
fn default_webcam_rect() -> NormalizedRect {
    NormalizedRect {
        x: 0.70,
        y: 0.64,
        w: 0.28,
        h: 0.33,
    }
}

// ---- Persistence in SQLite settings ----

pub fn load_all_compositions(kernel: &AicutKernel) -> Vec<AicutComposition> {
    kernel
        .repo
        .get_setting(SETTING_MOTION_COMPOSITIONS)
        .ok()
        .flatten()
        .and_then(|v| serde_json::from_value::<Vec<AicutComposition>>(v).ok())
        .unwrap_or_default()
}

pub fn save_composition(kernel: &AicutKernel, comp: &AicutComposition) -> AicutResult<()> {
    let mut list = load_all_compositions(kernel);
    if let Some(existing) = list.iter_mut().find(|c| c.id == comp.id) {
        *existing = comp.clone();
    } else {
        list.insert(0, comp.clone());
    }
    // Keep up to 40 recent compositions
    list.truncate(40);
    let val = serde_json::to_value(&list)
        .map_err(|e| AicutError::internal(format!("Failed serializing compositions: {e}")))?;
    kernel.repo.set_setting(SETTING_MOTION_COMPOSITIONS, &val)
}

pub fn get_composition(kernel: &AicutKernel, id: &str) -> Option<AicutComposition> {
    load_all_compositions(kernel).into_iter().find(|c| c.id == id)
}

pub fn delete_composition(kernel: &AicutKernel, id: &str) -> AicutResult<bool> {
    let mut list = load_all_compositions(kernel);
    let before = list.len();
    list.retain(|c| c.id != id);
    let removed = list.len() != before;
    if removed {
        let val = serde_json::to_value(&list)
            .map_err(|e| AicutError::internal(format!("Failed serializing compositions: {e}")))?;
        kernel.repo.set_setting(SETTING_MOTION_COMPOSITIONS, &val)?;
    }
    Ok(removed)
}

// ---- Forward-Compatible Compilers for HyperFrames (HTML/CSS/GSAP) & Remotion (Props JSON) ----

/// Compiles an `AicutComposition` into a clean, self-contained **HyperFrames** (HTML + CSS + GSAP)
/// composition ready for inspection, AI editing, or rendering with `npx hyperframes render`.
pub fn compile_to_hyperframes_html(comp: &AicutComposition) -> String {
    let mut scenes_html = String::new();
    for (idx, sc) in comp.scenes.iter().enumerate() {
        let dur = (sc.end_sec - sc.start_sec).max(0.5);
        let headline_html = sc
            .overlay
            .headline
            .as_deref()
            .filter(|s| !s.trim().is_empty())
            .map(|h| {
                format!(
                    "    <div class=\"hf-headline\" style=\"border-color:{}\">{}</div>\n",
                    sc.overlay.accent_hex,
                    escape_html(h)
                )
            })
            .unwrap_or_default();
        let callout_html = sc
            .overlay
            .callout_text
            .as_deref()
            .filter(|s| !s.trim().is_empty())
            .map(|c| {
                format!(
                    "    <div class=\"hf-callout\" style=\"left:{:.1}%;top:{:.1}%;border-color:{}\">{}</div>\n",
                    sc.content_focus.focus_x * 100.0,
                    sc.content_focus.focus_y * 100.0,
                    sc.overlay.accent_hex,
                    escape_html(c)
                )
            })
            .unwrap_or_default();
        let layout_str = serde_json::to_string(&sc.layout)
            .unwrap_or_else(|_| "\"split_branded\"".into())
            .trim_matches('"')
            .to_string();

        if let Some(ref custom) = sc.custom_code {
            if custom.engine == "hyperframes_html" && !custom.code.trim().is_empty() {
                scenes_html.push_str(&format!(
                    "  <!-- Scene #{}: {} (Custom HyperFrames Code) -->\n  <section class=\"hf-scene\" data-scene-id=\"{}\" data-start=\"{:.2}\" data-duration=\"{:.2}\">\n{}\n  </section>\n",
                    idx + 1,
                    escape_html(&sc.title),
                    sc.id,
                    sc.start_sec,
                    dur,
                    custom.code
                ));
                continue;
            }
        }

        scenes_html.push_str(&format!(
            "  <!-- Scene #{}: {} ({layout_str}) -->\n  <section class=\"hf-scene layout-{layout_str}\" data-scene-id=\"{}\" data-start=\"{:.2}\" data-duration=\"{:.2}\">\n    <div class=\"slot-content\" data-focus-x=\"{:.3}\" data-focus-y=\"{:.3}\" data-zoom=\"{:.2}\"></div>\n    <div class=\"slot-webcam\" data-pip-corner=\"{:?}\" data-pip-width=\"{:.2}\"></div>\n{headline_html}{callout_html}  </section>\n",
            idx + 1,
            escape_html(&sc.title),
            sc.id,
            sc.start_sec,
            dur,
            sc.content_focus.focus_x,
            sc.content_focus.focus_y,
            sc.content_focus.zoom,
            sc.pip.corner,
            sc.pip.width_pct,
        ));
    }

    format!(
        r#"<!DOCTYPE html>
<html lang="pt-BR">
<head>
  <meta charset="UTF-8" />
  <title>{title} — Aicut HyperFrames Composition</title>
  <script src="https://cdn.jsdelivr.net/npm/gsap@3.12.5/dist/gsap.min.js"></script>
  <style>
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{ width: {w}px; height: {h}px; overflow: hidden; background: #0a0b10; font-family: Inter, system-ui, sans-serif; color: #fff; }}
    .hf-scene {{ position: absolute; inset: 0; opacity: 0; }}
    .hf-headline {{ position: absolute; top: 68px; left: 50%; transform: translateX(-50%); padding: 14px 28px; border-radius: 14px; border: 3px solid #EAB308; background: rgba(10,11,16,0.86); font-size: 46px; font-weight: 800; text-transform: uppercase; letter-spacing: 0.02em; white-space: nowrap; }}
    .hf-callout {{ position: absolute; transform: translate(-50%, -50%); padding: 10px 20px; border-radius: 12px; border: 3px solid #EAB308; background: rgba(10,11,16,0.88); font-size: 32px; font-weight: 700; }}
  </style>
</head>
<body data-width="{w}" data-height="{h}" data-fps="{fps}" data-duration="{dur:.2}">
{scenes_html}  <script>
    // Deterministic GSAP master timeline ready for HyperFrames frame-by-frame capture
    const tl = gsap.timeline({{ paused: true }});
    document.querySelectorAll('.hf-scene').forEach((el) => {{
      const start = parseFloat(el.dataset.start || '0');
      const dur = parseFloat(el.dataset.duration || '1');
      tl.set(el, {{ opacity: 1 }}, start);
      const hl = el.querySelector('.hf-headline');
      if (hl) tl.fromTo(hl, {{ y: -28, scale: 0.9, opacity: 0 }}, {{ y: 0, scale: 1, opacity: 1, duration: 0.35, ease: 'back.out(1.7)' }}, start);
      tl.set(el, {{ opacity: 0 }}, start + dur);
    }});
    window.__aicutSeek = (sec) => tl.time(sec, false);
  </script>
</body>
</html>
"#,
        title = escape_html(&comp.title),
        w = comp.width,
        h = comp.height,
        fps = comp.fps,
        dur = comp.duration_sec,
    )
}

/// Exports the composition as strongly-typed **Remotion `inputProps`** JSON ready for `@remotion/player` or `remotion render`.
pub fn compile_to_remotion_props(comp: &AicutComposition) -> serde_json::Value {
    let fps = comp.fps.max(1);
    let duration_in_frames = (comp.duration_sec * fps as f64).round().max(1.0) as u64;
    let scenes: Vec<serde_json::Value> = comp
        .scenes
        .iter()
        .map(|s| {
            let from = (s.start_sec * fps as f64).round() as u64;
            let to = (s.end_sec * fps as f64).round() as u64;
            serde_json::json!({
                "id": s.id,
                "title": s.title,
                "fromFrame": from,
                "durationInFrames": to.saturating_sub(from).max(1),
                "layout": s.layout,
                "transitionIn": s.transition_in,
                "pip": s.pip,
                "contentFocus": s.content_focus,
                "overlay": s.overlay,
                "customCode": s.custom_code,
            })
        })
        .collect();

    serde_json::json!({
        "compositionId": "AicutDynamicShort",
        "width": comp.width,
        "height": comp.height,
        "fps": fps,
        "durationInFrames": duration_in_frames,
        "inputProps": {
            "mediaId": comp.media_id,
            "clipStartSec": comp.clip_start_sec,
            "clipEndSec": comp.clip_end_sec,
            "webcamRect": comp.webcam_rect,
            "contentRect": comp.content_rect,
            "burnSubtitles": comp.burn_subtitles,
            "scenes": scenes,
        }
    })
}

fn escape_html(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalize_stitches_gaps_and_clamps_scenes_to_exact_duration() {
        let mut comp = AicutComposition {
            id: "c1".into(),
            title: "Teste Dinâmico".into(),
            media_id: "m1".into(),
            clip_start_sec: 10.0,
            clip_end_sec: 40.0,
            duration_sec: 30.0,
            width: 1080,
            height: 1920,
            fps: 30,
            webcam_rect: default_webcam_rect(),
            content_rect: None,
            burn_subtitles: true,
            template_path: None,
            engine_preference: "hybrid_nvenc".into(),
            motion_html: None,
            creative_direction: String::new(),
            scenes: vec![
                MotionScene {
                    id: "s2".into(),
                    title: "Ação".into(),
                    start_sec: 12.0,
                    end_sec: 22.0,
                    layout: SceneLayoutPreset::ContentHeroPip,
                    transition_in: SceneTransition::Fade,
                    pip: PipConfig::default(),
                    content_focus: ContentFocusConfig::default(),
                    overlay: SceneOverlay {
                        headline: Some("MOMENTO INSANO!".into()),
                        ..Default::default()
                    },
                    effects: SceneEffects::default(),
                    rationale: String::new(),
                    custom_code: None,
                },
                MotionScene {
                    id: "s1".into(),
                    title: "Gancho".into(),
                    start_sec: 1.5,
                    end_sec: 8.0,
                    layout: SceneLayoutPreset::CameraFocus,
                    transition_in: SceneTransition::Cut,
                    pip: PipConfig::default(),
                    content_focus: ContentFocusConfig::default(),
                    overlay: SceneOverlay::default(),
                    effects: SceneEffects::default(),
                    rationale: String::new(),
                    custom_code: None,
                },
            ],
            chat_history: vec![],
            rendered_media_id: None,
            updated_at: default_now(),
        };

        comp.normalize();
        assert_eq!(comp.scenes.len(), 2);
        assert_eq!(comp.scenes[0].start_sec, 0.0);
        assert_eq!(comp.scenes[0].end_sec, comp.scenes[1].start_sec);
        assert_eq!(comp.scenes[1].end_sec, 30.0);

        let html = compile_to_hyperframes_html(&comp);
        assert!(html.contains("data-duration=\"30.00\""));
        assert!(html.contains("MOMENTO INSANO!"));
        assert!(html.contains("layout-content_hero_pip"));

        let remotion = compile_to_remotion_props(&comp);
        assert_eq!(remotion["durationInFrames"], 900);
        assert_eq!(remotion["inputProps"]["scenes"].as_array().unwrap().len(), 2);
    }
}
