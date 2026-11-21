use aicut_core::{configure_tokio_command, AicutError, AicutKernel, AicutResult, SmartClipSegment};
use image::{imageops, RgbImage};
use std::path::{Path, PathBuf};
use tokio::process::Command;
use tract_onnx::prelude::*;

use crate::renderer::WebcamCorner;

const ULTRAFACE_MODEL_FILENAME: &str = "ultraface-rfb-320.onnx";
const ULTRAFACE_DOWNLOAD_URLS: &[&str] = &[
    "https://raw.githubusercontent.com/Linzaer/Ultra-Light-Fast-Generic-Face-Detector-1MB/master/models/onnx/version-RFB-320.onnx",
    "https://huggingface.co/spaces/Linzaer/Ultra-Light-Fast-Generic-Face-Detector-1MB/resolve/main/models/onnx/version-RFB-320.onnx",
];

type RunnableFaceModel =
    SimplePlan<TypedFact, Box<dyn TypedOp>, Graph<TypedFact, Box<dyn TypedOp>>>;

#[derive(Debug, Clone, Copy)]
pub struct FaceBox {
    /// Normalized [0, 1] coordinates
    pub x1: f32,
    pub y1: f32,
    pub x2: f32,
    pub y2: f32,
    pub score: f32,
}

impl FaceBox {
    pub fn cx(&self) -> f32 {
        (self.x1 + self.x2) * 0.5
    }
    pub fn cy(&self) -> f32 {
        (self.y1 + self.y2) * 0.5
    }
    pub fn w(&self) -> f32 {
        (self.x2 - self.x1).max(0.0)
    }
    pub fn h(&self) -> f32 {
        (self.y2 - self.y1).max(0.0)
    }
    /// Estimated eye-line Y coordinate (roughly 35% down from top of face bbox).
    pub fn eye_y(&self) -> f32 {
        self.y1 + self.h() * 0.35
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PixelRect {
    pub x: f64,
    pub y: f64,
    pub w: f64,
    pub h: f64,
}

impl PixelRect {
    pub fn right(&self) -> f64 {
        self.x + self.w
    }
    pub fn bottom(&self) -> f64 {
        self.y + self.h
    }
}

#[derive(Debug, Clone)]
pub struct CameraTrackPlan {
    pub crop_w: u32,
    pub crop_h: u32,
    pub init_x: i32,
    pub init_y: i32,
    /// Chronological `(time_sec, x_px, y_px)` commands for FFmpeg `sendcmd`.
    pub keyframes: Vec<(f64, i32, i32)>,
    /// Named FFmpeg crop filter target (e.g. `"crop@cam"` or `"crop@content"`).
    pub filter_target: String,
    pub detected_faces_count: usize,
    pub is_pip_webcam: bool,
}

impl CameraTrackPlan {
    /// Writes FFmpeg `sendcmd` script driving `self.filter_target` smoothly over the entire clip.
    pub fn format_sendcmd_script(&self) -> String {
        let target = if self.filter_target.is_empty() {
            "crop@cam"
        } else {
            self.filter_target.as_str()
        };
        let mut out = String::new();
        for &(t, x, y) in &self.keyframes {
            out.push_str(&format!("{t:.3} [enter] {target} x {x}, {target} y {y};\n"));
        }
        out
    }

    pub fn write_sendcmd_file(&self, path: &Path) -> AicutResult<()> {
        std::fs::write(path, self.format_sendcmd_script())?;
        Ok(())
    }
}

/// Combined AI tracking output for a Short:
/// - `camera`: top webcam cameraman (`crop@cam`) with OBS border removal & face tracking
/// - `content`: bottom content director (`crop@content`) framing active speaker or peak visual action
#[derive(Debug, Clone, Default)]
pub struct DualTrackPlan {
    pub camera: Option<CameraTrackPlan>,
    pub content: Option<CameraTrackPlan>,
}

pub struct FaceDetector {
    model: RunnableFaceModel,
}

impl FaceDetector {
    pub fn load(model_path: &Path) -> AicutResult<Self> {
        let model = tract_onnx::onnx()
            .model_for_path(model_path)
            .map_err(|e| AicutError::internal(format!("Failed loading UltraFace ONNX: {e}")))?
            .with_input_fact(
                0,
                InferenceFact::dt_shape(f32::datum_type(), tvec!(1, 3, 240, 320)),
            )
            .map_err(|e| AicutError::internal(format!("Failed configuring UltraFace shape: {e}")))?
            .into_optimized()
            .map_err(|e| AicutError::internal(format!("Failed optimizing UltraFace model: {e}")))?
            .into_runnable()
            .map_err(|e| AicutError::internal(format!("Failed building UltraFace plan: {e}")))?;

        Ok(Self { model })
    }

    pub fn detect(&self, rgb: &RgbImage, score_threshold: f32) -> AicutResult<Vec<FaceBox>> {
        let resized = imageops::resize(rgb, 320, 240, imageops::FilterType::Triangle);
        let plane = 320usize * 240usize;
        let mut data = vec![0.0f32; 3 * plane];

        for y in 0..240usize {
            for x in 0..320usize {
                let p = resized.get_pixel(x as u32, y as u32);
                let idx = y * 320 + x;
                data[idx] = (p[0] as f32 - 127.0) / 128.0;
                data[plane + idx] = (p[1] as f32 - 127.0) / 128.0;
                data[2 * plane + idx] = (p[2] as f32 - 127.0) / 128.0;
            }
        }

        let tensor = Tensor::from_shape(&[1, 3, 240, 320], &data)
            .map_err(|e| AicutError::internal(format!("Failed creating input tensor: {e}")))?;
        let outputs = self
            .model
            .run(tvec!(tensor.into()))
            .map_err(|e| AicutError::internal(format!("UltraFace inference failed: {e}")))?;

        let mut scores_idx = None;
        let mut boxes_idx = None;
        for (i, out) in outputs.iter().enumerate() {
            let shape = out.shape();
            if shape.len() == 3 && shape[2] == 2 {
                scores_idx = Some(i);
            } else if shape.len() == 3 && shape[2] == 4 {
                boxes_idx = Some(i);
            }
        }

        let (Some(si), Some(bi)) = (scores_idx, boxes_idx) else {
            return Err(AicutError::internal(
                "Unexpected UltraFace ONNX output tensor shapes",
            ));
        };

        let scores_shape = outputs[si].shape().to_vec();
        let boxes_shape = outputs[bi].shape().to_vec();
        let scores = outputs[si]
            .as_slice::<f32>()
            .map_err(|e| AicutError::internal(format!("Failed reading scores tensor: {e}")))?;
        let boxes = outputs[bi]
            .as_slice::<f32>()
            .map_err(|e| AicutError::internal(format!("Failed reading boxes tensor: {e}")))?;

        let n = scores_shape[1]
            .min(boxes_shape[1])
            .min(scores.len() / 2)
            .min(boxes.len() / 4);
        let mut raw = Vec::new();
        for i in 0..n {
            let conf = scores[i * 2 + 1];
            if conf >= score_threshold {
                let x1 = boxes[i * 4].clamp(0.0, 1.0);
                let y1 = boxes[i * 4 + 1].clamp(0.0, 1.0);
                let x2 = boxes[i * 4 + 2].clamp(0.0, 1.0);
                let y2 = boxes[i * 4 + 3].clamp(0.0, 1.0);
                if x2 > x1 + 0.01 && y2 > y1 + 0.01 {
                    raw.push(FaceBox {
                        x1,
                        y1,
                        x2,
                        y2,
                        score: conf,
                    });
                }
            }
        }

        Ok(nms(raw, 0.35))
    }
}

pub fn face_model_path(kernel: &AicutKernel) -> PathBuf {
    kernel.workspace.models_dir.join("vision").join(ULTRAFACE_MODEL_FILENAME)
}
pub fn installed_face_model(kernel: &AicutKernel) -> AicutResult<PathBuf> {
    let path = face_model_path(kernel);
    if path.metadata().is_ok_and(|m| m.len() > 200_000) { Ok(path) }
    else { Err(AicutError::not_found("Instale UltraFace na aba Downloads.")) }
}

/// Ensures the tiny (~1.2 MB) UltraFace ONNX model exists locally, downloading it once automatically.
pub async fn ensure_face_detector_model(kernel: &AicutKernel) -> AicutResult<PathBuf> {
    let _directory = aicut_core::workspace::lock_downloads(&kernel.workspace.models_dir)?;
    let vision_dir = kernel.workspace.models_dir.join("vision");
    std::fs::create_dir_all(&vision_dir)?;
    let dest = vision_dir.join(ULTRAFACE_MODEL_FILENAME);
    if dest.is_file() && std::fs::metadata(&dest).map(|m| m.len() > 200_000).unwrap_or(false) {
        return Ok(dest);
    }

    let http = kernel.tools.http_client();
    let mut last_err = String::new();
    for url in ULTRAFACE_DOWNLOAD_URLS {
        match http.get(*url).send().await {
            Ok(resp) if resp.status().is_success() => {
                if let Ok(bytes) = resp.bytes().await {
                    if bytes.len() > 200_000 {
                        let stage = aicut_core::workspace::download_stage(&kernel.workspace.models_dir)?;
                        let tmp = stage.path().join("model");
                        std::fs::write(&tmp, &bytes)?;
                        if dest.exists() {
                            let _ = std::fs::remove_file(&dest);
                        }
                        std::fs::rename(&tmp, &dest)?;
                        return Ok(dest);
                    }
                }
            }
            Ok(resp) => {
                last_err = format!("HTTP {} from {url}", resp.status());
            }
            Err(e) => {
                last_err = e.to_string();
            }
        }
    }

    Err(AicutError::internal(format!(
        "Nao foi possivel baixar o modelo facial UltraFace (~1.2MB): {last_err}"
    )))
}

/// Configuration for top webcam slot tracking inside `compute_dual_tracks`.
#[derive(Debug, Clone, Copy)]
pub struct CameraSlotSpec {
    pub target_w: u32,
    pub target_h: u32,
    pub corner_hint: WebcamCorner,
    pub face_zoom: f32,
    pub border_trim_pct: f32,
    /// Optional user-saved precision webcam box (normalized 0..1). When set, overrides auto detection.
    pub custom_rect: Option<PixelRect>,
    /// When true with `custom_rect`, locks 100% cleanly to the saved area with zero zoom/pan drift.
    pub lock_static: bool,
}

/// Configuration for bottom content slot framing inside `compute_dual_tracks`.
#[derive(Debug, Clone, Copy)]
pub struct ContentSlotSpec {
    pub target_w: u32,
    pub target_h: u32,
    /// Optional user-saved precision content region (normalized 0..1).
    pub custom_rect: Option<PixelRect>,
    /// When true with `custom_rect`, shows the exact saved content region statically without panning.
    pub lock_static: bool,
}

/// Builds a crisp static `CameraTrackPlan` fitting the maximum `target_aspect` crop centered inside
/// `rect_norm` (normalized 0..1) with zero border leak and zero zoom glued to the face.
pub fn build_static_rect_plan(
    rect_norm: PixelRect,
    src_w: u32,
    src_h: u32,
    target_aspect: f64,
    filter_target: &str,
    is_pip_webcam: bool,
) -> CameraTrackPlan {
    let rx0 = (rect_norm.x * src_w as f64)
        .ceil()
        .clamp(0.0, src_w.saturating_sub(32) as f64);
    let ry0 = (rect_norm.y * src_h as f64)
        .ceil()
        .clamp(0.0, src_h.saturating_sub(32) as f64);
    let rx1 = (rect_norm.right() * src_w as f64)
        .floor()
        .clamp(rx0 + 32.0, src_w as f64);
    let ry1 = (rect_norm.bottom() * src_h as f64)
        .floor()
        .clamp(ry0 + 32.0, src_h as f64);
    let rw = (rx1 - rx0).max(32.0);
    let rh = (ry1 - ry0).max(32.0);

    let (cw_f, ch_f) = if rw / rh.max(1.0) >= target_aspect {
        (rh * target_aspect, rh)
    } else {
        (rw, rw / target_aspect)
    };
    let max_even_w = ((rw.floor() as u32) & !1).clamp(32, src_w & !1);
    let max_even_h = ((rh.floor() as u32) & !1).clamp(32, src_h & !1);
    let crop_w = ((cw_f.floor() as u32) & !1).clamp(32.min(max_even_w), max_even_w);
    let crop_h = ((ch_f.floor() as u32) & !1).clamp(32.min(max_even_h), max_even_h);

    let x = ((rx0 + (rw - crop_w as f64) * 0.5).round() as i32)
        .clamp(rx0 as i32, (rx1 as i32 - crop_w as i32).max(rx0 as i32));
    let y = ((ry0 + (rh - crop_h as f64) * 0.5).round() as i32)
        .clamp(ry0 as i32, (ry1 as i32 - crop_h as i32).max(ry0 as i32));

    CameraTrackPlan {
        crop_w,
        crop_h,
        init_x: x,
        init_y: y,
        keyframes: vec![(0.0, x, y)],
        filter_target: filter_target.to_string(),
        detected_faces_count: 0,
        is_pip_webcam,
    }
}

/// Runs full 2-stage AI face tracking & zero-border webcam edge detection across a clip.
/// Returns `Ok(Some(CameraTrackPlan))` on success, or `Ok(None)` if no face could be detected
/// (so the renderer can fall back cleanly to the static crop).
#[allow(clippy::too_many_arguments)]
pub async fn compute_camera_track(
    kernel: &AicutKernel,
    ffmpeg_bin: &Path,
    video_path: &Path,
    src_w: u32,
    src_h: u32,
    segments: &[SmartClipSegment],
    target_w: u32,
    target_h: u32,
    corner_hint: WebcamCorner,
    face_zoom: f32,
    border_trim_pct: f32,
) -> AicutResult<Option<CameraTrackPlan>> {
    let dual = compute_dual_tracks(
        kernel,
        ffmpeg_bin,
        video_path,
        src_w,
        src_h,
        segments,
        Some(CameraSlotSpec {
            target_w,
            target_h,
            corner_hint,
            face_zoom,
            border_trim_pct,
            custom_rect: None,
            lock_static: false,
        }),
        None,
    )
    .await?;
    Ok(dual.camera)
}

/// Runs unified single-pass AI tracking for BOTH Short windows in one frame extraction:
/// - `camera_slot`: top webcam cameraman (`crop@cam`) — respects user-saved precision rect if set
/// - `content_slot`: bottom content director (`crop@content`) — active speaker or peak action
#[allow(clippy::too_many_arguments)]
pub async fn compute_dual_tracks(
    kernel: &AicutKernel,
    ffmpeg_bin: &Path,
    video_path: &Path,
    src_w: u32,
    src_h: u32,
    segments: &[SmartClipSegment],
    camera_slot: Option<CameraSlotSpec>,
    content_slot: Option<ContentSlotSpec>,
) -> AicutResult<DualTrackPlan> {
    if (camera_slot.is_none() && content_slot.is_none())
        || src_w < 64
        || src_h < 64
        || segments.is_empty()
    {
        return Ok(DualTrackPlan::default());
    }

    // Fast path: if BOTH requested slots are static user-saved precision boxes, zero frame
    // extraction or ONNX inference is needed!
    let cam_is_pure_static = camera_slot
        .map(|c| c.custom_rect.is_some() && c.lock_static)
        .unwrap_or(true);
    let content_is_pure_static = content_slot
        .map(|c| c.custom_rect.is_some() && c.lock_static)
        .unwrap_or(true);
    if cam_is_pure_static && content_is_pure_static {
        let camera = camera_slot.and_then(|c| {
            c.custom_rect.map(|r| {
                build_static_rect_plan(
                    r,
                    src_w,
                    src_h,
                    c.target_w as f64 / c.target_h.max(1) as f64,
                    "crop@cam",
                    true,
                )
            })
        });
        let content = content_slot.and_then(|c| {
            c.custom_rect.map(|r| {
                build_static_rect_plan(
                    r,
                    src_w,
                    src_h,
                    c.target_w as f64 / c.target_h.max(1) as f64,
                    "crop@content",
                    false,
                )
            })
        });
        return Ok(DualTrackPlan { camera, content });
    }

    // Download UltraFace (~1.2 MB) once if missing. If offline/unreachable, content action tracking
    // or static custom rects can still proceed cleanly without failing the render.
    let model_path = installed_face_model(kernel).ok();

    let tmp_dir = std::env::temp_dir().join(format!("aicut-facetrack-{}", uuid::Uuid::new_v4()));
    std::fs::create_dir_all(&tmp_dir)?;

    let sample_fps = 3.0f64;
    let extract_res =
        extract_clip_analysis_frames(ffmpeg_bin, video_path, segments, sample_fps, &tmp_dir).await;
    let frames = match extract_res {
        Ok(f) if !f.is_empty() => f,
        _ => {
            let _ = std::fs::remove_dir_all(&tmp_dir);
            return Ok(DualTrackPlan::default());
        }
    };

    let total_clip_dur: f64 = segments
        .iter()
        .map(|s| (s.end_sec - s.start_sec).max(0.0))
        .sum();

    // Run CPU-bound ONNX inference + edge/motion/saliency analysis on blocking threadpool
    let plan = tokio::task::spawn_blocking(move || -> AicutResult<DualTrackPlan> {
        let detector = model_path
            .as_deref()
            .and_then(|p| FaceDetector::load(p).ok());
        let loaded: Vec<(f64, RgbImage)> = frames
            .iter()
            .filter_map(|(t, p)| image::open(p).ok().map(|img| (*t, img.to_rgb8())))
            .collect();

        if loaded.is_empty() {
            return Ok(DualTrackPlan::default());
        }

        let mut camera_plan: Option<CameraTrackPlan> = None;
        let mut webcam_exclude_norm: Option<PixelRect> = None;

        // ---- PART 1: Top Webcam Cameraman (User-Saved Precision Rect OR Zero-Border Auto) ----
        if let Some(cam) = camera_slot {
            let target_aspect = cam.target_w as f64 / cam.target_h.max(1) as f64;
            let zoom = cam.face_zoom.clamp(0.75, 1.50) as f64;
            let trim = cam.border_trim_pct.clamp(0.01, 0.12) as f64;

            if let Some(saved_rect) = cam.custom_rect {
                // User saved exact webcam interior in the Framing Studio tab!
                webcam_exclude_norm = Some(expand_norm_rect(saved_rect, 0.018));
                if cam.lock_static || detector.is_none() {
                    camera_plan = Some(build_static_rect_plan(
                        saved_rect,
                        src_w,
                        src_h,
                        target_aspect,
                        "crop@cam",
                        true,
                    ));
                } else if let Some(det) = detector.as_ref() {
                    let safe_box_px = PixelRect {
                        x: saved_rect.x * src_w as f64,
                        y: saved_rect.y * src_h as f64,
                        w: saved_rect.w * src_w as f64,
                        h: saved_rect.h * src_h as f64,
                    };
                    let mut tracked_samples: Vec<(f64, Option<FaceBox>)> =
                        Vec::with_capacity(loaded.len());
                    let mut prev_cx = (saved_rect.x + saved_rect.w * 0.5) as f32;
                    let mut prev_cy = (saved_rect.y + saved_rect.h * 0.45) as f32;
                    let mut detected_faces_count = 0usize;

                    for (t, rgb) in &loaded {
                        let fw = rgb.width() as f64;
                        let fh = rgb.height() as f64;
                        let cx0 = (saved_rect.x * fw).floor().clamp(0.0, fw - 2.0) as u32;
                        let cy0 = (saved_rect.y * fh).floor().clamp(0.0, fh - 2.0) as u32;
                        let cw = (saved_rect.w * fw)
                            .round()
                            .clamp(16.0, fw - cx0 as f64) as u32;
                        let ch = (saved_rect.h * fh)
                            .round()
                            .clamp(16.0, fh - cy0 as f64) as u32;

                        let cropped = imageops::crop_imm(rgb, cx0, cy0, cw, ch).to_image();
                        let picked = det.detect(&cropped, 0.55).ok().and_then(|faces| {
                            faces
                                .into_iter()
                                .map(|f| FaceBox {
                                    x1: (cx0 as f32 + f.x1 * cw as f32) / fw as f32,
                                    y1: (cy0 as f32 + f.y1 * ch as f32) / fh as f32,
                                    x2: (cx0 as f32 + f.x2 * cw as f32) / fw as f32,
                                    y2: (cy0 as f32 + f.y2 * ch as f32) / fh as f32,
                                    score: f.score,
                                })
                                .max_by(|a, b| {
                                    let sa = track_continuity_score(*a, prev_cx, prev_cy);
                                    let sb = track_continuity_score(*b, prev_cx, prev_cy);
                                    sa.partial_cmp(&sb).unwrap_or(std::cmp::Ordering::Equal)
                                })
                        });
                        if let Some(f) = picked {
                            prev_cx = f.cx();
                            prev_cy = f.cy();
                            detected_faces_count += 1;
                        }
                        tracked_samples.push((*t, picked));
                    }

                    if detected_faces_count > 0 {
                        camera_plan = Some(build_natural_camera_track(
                            &tracked_samples,
                            safe_box_px,
                            src_w,
                            src_h,
                            target_aspect,
                            zoom,
                            total_clip_dur,
                            true,
                        ));
                    } else {
                        camera_plan = Some(build_static_rect_plan(
                            saved_rect,
                            src_w,
                            src_h,
                            target_aspect,
                            "crop@cam",
                            true,
                        ));
                    }
                }
            } else if let Some(det) = detector.as_ref() {
                // PASS 1: Detect faces across clip & identify dominant human speaker (anti-NPC)
                let stride = (loaded.len() / 45).max(1);
                let mut pass1_faces: Vec<FaceBox> = Vec::new();
                for (idx, (_, rgb)) in loaded.iter().enumerate() {
                    if idx % stride != 0 {
                        continue;
                    }
                    if let Ok(found) = det.detect(rgb, 0.62) {
                        for f in found {
                            if matches_corner_hint(f, cam.corner_hint) {
                                pass1_faces.push(f);
                            }
                        }
                    }
                }

                if let Some(anchor_face) = select_dominant_face_cluster(&pass1_faces) {
                    let force_pip = matches!(
                        cam.corner_hint,
                        WebcamCorner::BottomRight
                            | WebcamCorner::BottomLeft
                            | WebcamCorner::TopRight
                            | WebcamCorner::TopLeft
                    );
                    let maybe_pip = match cam.corner_hint {
                        WebcamCorner::Center => false,
                        WebcamCorner::AutoFace => {
                            let near_edge = anchor_face.cx() < 0.36
                                || anchor_face.cx() > 0.64
                                || anchor_face.cy() < 0.34
                                || anchor_face.cy() > 0.66;
                            anchor_face.w() < 0.32 && (near_edge || anchor_face.w() < 0.22)
                        }
                        _ => true,
                    };

                    let (is_pip_webcam, safe_box_norm, outer_box_norm) = if maybe_pip {
                        let det_box =
                            detect_pip_webcam_zero_border(&loaded, anchor_face, trim, force_pip);
                        let confirmed_pip = force_pip || det_box.walls_detected >= 2;
                        if confirmed_pip {
                            (true, det_box.safe_rect, det_box.outer_rect)
                        } else {
                            let full = PixelRect {
                                x: 0.005,
                                y: 0.005,
                                w: 0.99,
                                h: 0.99,
                            };
                            (false, full, full)
                        }
                    } else {
                        let full = PixelRect {
                            x: 0.005,
                            y: 0.005,
                            w: 0.99,
                            h: 0.99,
                        };
                        (false, full, full)
                    };

                    if is_pip_webcam {
                        webcam_exclude_norm = Some(expand_norm_rect(outer_box_norm, 0.018));
                    }

                    let safe_box_px = PixelRect {
                        x: safe_box_norm.x * src_w as f64,
                        y: safe_box_norm.y * src_h as f64,
                        w: safe_box_norm.w * src_w as f64,
                        h: safe_box_norm.h * src_h as f64,
                    };

                    // PASS 2: High-precision face tracking INSIDE clean safe_box on EVERY frame
                    let mut tracked_samples: Vec<(f64, Option<FaceBox>)> =
                        Vec::with_capacity(loaded.len());
                    let mut prev_cx = anchor_face.cx();
                    let mut prev_cy = anchor_face.cy();
                    let mut detected_faces_count = 0usize;

                    for (t, rgb) in &loaded {
                        let fw = rgb.width() as f64;
                        let fh = rgb.height() as f64;
                        let cx0 = (safe_box_norm.x * fw).floor().clamp(0.0, fw - 2.0) as u32;
                        let cy0 = (safe_box_norm.y * fh).floor().clamp(0.0, fh - 2.0) as u32;
                        let cw = (safe_box_norm.w * fw)
                            .round()
                            .clamp(16.0, fw - cx0 as f64) as u32;
                        let ch = (safe_box_norm.h * fh)
                            .round()
                            .clamp(16.0, fh - cy0 as f64) as u32;

                        let cropped = imageops::crop_imm(rgb, cx0, cy0, cw, ch).to_image();
                        let picked = det.detect(&cropped, 0.58).ok().and_then(|faces| {
                            faces
                                .into_iter()
                                .map(|f| FaceBox {
                                    x1: (cx0 as f32 + f.x1 * cw as f32) / fw as f32,
                                    y1: (cy0 as f32 + f.y1 * ch as f32) / fh as f32,
                                    x2: (cx0 as f32 + f.x2 * cw as f32) / fw as f32,
                                    y2: (cy0 as f32 + f.y2 * ch as f32) / fh as f32,
                                    score: f.score,
                                })
                                .max_by(|a, b| {
                                    let sa = track_continuity_score(*a, prev_cx, prev_cy);
                                    let sb = track_continuity_score(*b, prev_cx, prev_cy);
                                    sa.partial_cmp(&sb).unwrap_or(std::cmp::Ordering::Equal)
                                })
                        });

                        if let Some(f) = picked {
                            prev_cx = f.cx();
                            prev_cy = f.cy();
                            detected_faces_count += 1;
                        }
                        tracked_samples.push((*t, picked));
                    }

                    if detected_faces_count > 0 {
                        camera_plan = Some(build_natural_camera_track(
                            &tracked_samples,
                            safe_box_px,
                            src_w,
                            src_h,
                            target_aspect,
                            zoom,
                            total_clip_dur,
                            is_pip_webcam,
                        ));
                    }
                }
            }
        }

        // ---- PART 2: Bottom Content Director (User-Saved Region OR Active Speaker + Action) ----
        let content_plan = content_slot.map(|cnt| {
            let content_aspect = cnt.target_w as f64 / cnt.target_h.max(1) as f64;
            if let Some(saved_content) = cnt.custom_rect {
                if cnt.lock_static {
                    return build_static_rect_plan(
                        saved_content,
                        src_w,
                        src_h,
                        content_aspect,
                        "crop@content",
                        false,
                    );
                }
                // Run AI Content Director strictly inside the user's saved content region
                return compute_content_director_track_in_region(
                    &loaded,
                    detector.as_ref(),
                    webcam_exclude_norm,
                    Some(saved_content),
                    src_w,
                    src_h,
                    content_aspect,
                    total_clip_dur,
                );
            }
            compute_content_director_track(
                &loaded,
                detector.as_ref(),
                webcam_exclude_norm,
                src_w,
                src_h,
                content_aspect,
                total_clip_dur,
            )
        });

        Ok(DualTrackPlan {
            camera: camera_plan,
            content: content_plan,
        })
    })
    .await
    .map_err(|e| AicutError::internal(format!("Dual camera/content tracking task failed: {e}")))??;

    let _ = std::fs::remove_dir_all(&tmp_dir);
    Ok(plan)
}

/// Extracts lightweight 640x360 frames across the clip's segments at `fps` into `out_dir`.
async fn extract_clip_analysis_frames(
    ffmpeg_bin: &Path,
    video_path: &Path,
    segments: &[SmartClipSegment],
    fps: f64,
    out_dir: &Path,
) -> AicutResult<Vec<(f64, PathBuf)>> {
    let mut cmd = Command::new(ffmpeg_bin);
    configure_tokio_command(&mut cmd);

    let pattern = out_dir.join("ft_%05d.jpg");

    if segments.len() == 1 {
        let seg = &segments[0];
        let dur = (seg.end_sec - seg.start_sec).max(0.5);
        let vf = format!("fps={fps:.3},scale=640:-2:flags=fast_bilinear");
        cmd.args([
            "-y",
            "-ss",
            &format!("{:.3}", seg.start_sec),
            "-t",
            &format!("{dur:.3}"),
            "-i",
        ])
        .arg(video_path)
        .args(["-vf", &vf, "-q:v", "6", "-fps_mode", "vfr"])
        .arg(&pattern);
    } else {
        cmd.args(["-y", "-i"]).arg(video_path);
        let mut parts = Vec::new();
        let mut concat_in = String::new();
        for (i, seg) in segments.iter().enumerate() {
            parts.push(format!(
                "[0:v]trim=start={:.3}:end={:.3},setpts=PTS-STARTPTS[v{i}]",
                seg.start_sec, seg.end_sec
            ));
            concat_in.push_str(&format!("[v{i}]"));
        }
        parts.push(format!(
            "{concat_in}concat=n={}:v=1:a=0,fps={fps:.3},scale=640:-2:flags=fast_bilinear[vout]",
            segments.len()
        ));
        cmd.args([
            "-filter_complex",
            &parts.join(";"),
            "-map",
            "[vout]",
            "-q:v",
            "6",
            "-fps_mode",
            "vfr",
        ])
        .arg(&pattern);
    }

    let out = cmd
        .output()
        .await
        .map_err(|e| AicutError::internal(format!("FFmpeg face-track frame extract failed: {e}")))?;
    if !out.status.success() {
        return Ok(Vec::new());
    }

    let mut files: Vec<PathBuf> = std::fs::read_dir(out_dir)?
        .flatten()
        .map(|e| e.path())
        .filter(|p| {
            p.file_name()
                .and_then(|n| n.to_str())
                .map(|s| s.starts_with("ft_") && s.ends_with(".jpg"))
                .unwrap_or(false)
        })
        .collect();
    files.sort();

    Ok(files
        .into_iter()
        .enumerate()
        .map(|(i, p)| (i as f64 / fps, p))
        .collect())
}

fn matches_corner_hint(face: FaceBox, hint: WebcamCorner) -> bool {
    match hint {
        WebcamCorner::AutoFace => true,
        WebcamCorner::BottomRight => face.cx() >= 0.45 && face.cy() >= 0.40,
        WebcamCorner::BottomLeft => face.cx() <= 0.55 && face.cy() >= 0.40,
        WebcamCorner::TopRight => face.cx() >= 0.45 && face.cy() <= 0.60,
        WebcamCorner::TopLeft => face.cx() <= 0.55 && face.cy() <= 0.60,
        WebcamCorner::Center => face.cx() >= 0.15 && face.cx() <= 0.85,
    }
}

/// Anti-NPC spatial clustering: groups detections on a coarse 6x6 grid and picks the most
/// persistent, high-confidence cluster (the real streamer rather than fleeting game NPCs).
pub(crate) fn select_dominant_face_cluster(faces: &[FaceBox]) -> Option<FaceBox> {
    if faces.is_empty() {
        return None;
    }
    let mut best_cluster: Vec<FaceBox> = Vec::new();
    let mut best_weight = -1.0f32;

    for seed in faces {
        let mut members = Vec::new();
        let mut weight = 0.0f32;
        for f in faces {
            let dx = (f.cx() - seed.cx()).abs();
            let dy = (f.cy() - seed.cy()).abs();
            if dx <= 0.14 && dy <= 0.18 {
                members.push(*f);
                weight += f.score * (1.0 + f.w() * 2.0);
            }
        }
        if weight > best_weight {
            best_weight = weight;
            best_cluster = members;
        }
    }

    if best_cluster.is_empty() {
        return None;
    }
    let n = best_cluster.len() as f32;
    let avg = |proj: fn(&FaceBox) -> f32| best_cluster.iter().map(proj).sum::<f32>() / n;
    Some(FaceBox {
        x1: avg(|f| f.x1),
        y1: avg(|f| f.y1),
        x2: avg(|f| f.x2),
        y2: avg(|f| f.y2),
        score: avg(|f| f.score),
    })
}

fn track_continuity_score(face: FaceBox, prev_cx: f32, prev_cy: f32) -> f32 {
    let dist = ((face.cx() - prev_cx).powi(2) + (face.cy() - prev_cy).powi(2)).sqrt();
    let proximity = (1.0 - dist * 2.5).clamp(0.0, 1.0);
    face.score * 0.55 + proximity * 0.35 + (face.w() * 2.0).min(0.2)
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct PipWebcamDetection {
    pub safe_rect: PixelRect,
    pub outer_rect: PixelRect,
    pub walls_detected: usize,
}

#[derive(Debug, Clone, Copy)]
struct WallScanResult {
    inner: usize,
    outer: usize,
    detected: bool,
}

/// Surgical 4-phase Zero-Border OBS PiP Webcam Detector:
/// 1. Builds 3-channel RGB temporal median + per-pixel temporal MAD across up to 18 frames.
/// 2. Scans OUTWARD from the face on ALL 4 sides (including screen edges!) to find continuous
///    straight OBS borders, pairing thick double-edge frames and locking onto the INNER edge.
/// 3. Runs an iterative inward static-border / nameplate / glow peeler on all 4 sides.
/// 4. Probes all 4 diagonals for rounded-corner masks + applies a hard zero-border safety inset
///    so 0 pixels of OBS border, glow, rounded corner, or gameplay ever appear in the top window.
pub(crate) fn detect_pip_webcam_zero_border(
    frames: &[(f64, RgbImage)],
    anchor_face: FaceBox,
    trim_pct: f64,
    force_pip: bool,
) -> PipWebcamDetection {
    let sample_count = frames.len().min(18).max(1);
    let step = (frames.len() / sample_count).max(1);
    let subset: Vec<&RgbImage> = frames
        .iter()
        .step_by(step)
        .take(sample_count)
        .map(|(_, img)| img)
        .collect();

    let w = subset[0].width() as usize;
    let h = subset[0].height() as usize;

    let mut median_rgb = vec![[0u8; 3]; w * h];
    let mut temporal_mad = vec![0.0f32; w * h];
    let mut r_buf = Vec::with_capacity(subset.len());
    let mut g_buf = Vec::with_capacity(subset.len());
    let mut b_buf = Vec::with_capacity(subset.len());
    let mid = subset.len() / 2;
    let inv_n = 1.0f32 / subset.len().max(1) as f32;

    for y in 0..h {
        for x in 0..w {
            r_buf.clear();
            g_buf.clear();
            b_buf.clear();
            for img in &subset {
                let p = img.get_pixel(x as u32, y as u32);
                r_buf.push(p[0]);
                g_buf.push(p[1]);
                b_buf.push(p[2]);
            }
            r_buf.sort_unstable();
            g_buf.sort_unstable();
            b_buf.sort_unstable();
            let mr = r_buf[mid];
            let mg = g_buf[mid];
            let mb = b_buf[mid];
            median_rgb[y * w + x] = [mr, mg, mb];

            let mut mad_sum = 0.0f32;
            for i in 0..subset.len() {
                mad_sum += ((r_buf[i] as i32 - mr as i32).abs()
                    + (g_buf[i] as i32 - mg as i32).abs()
                    + (b_buf[i] as i32 - mb as i32).abs()) as f32
                    / 3.0;
            }
            temporal_mad[y * w + x] = mad_sum * inv_n;
        }
    }

    let rgb_diff = |a: [u8; 3], b: [u8; 3]| -> f64 {
        ((a[0] as i32 - b[0] as i32).abs()
            + (a[1] as i32 - b[1] as i32).abs()
            + (a[2] as i32 - b[2] as i32).abs()) as f64
            / 3.0
    };

    let fx1 = (anchor_face.x1 as f64 * w as f64).round() as isize;
    let fy1 = (anchor_face.y1 as f64 * h as f64).round() as isize;
    let fx2 = (anchor_face.x2 as f64 * w as f64).round() as isize;
    let fy2 = (anchor_face.y2 as f64 * h as f64).round() as isize;
    let fw = (fx2 - fx1).max(10);
    let fh = (fy2 - fy1).max(10);
    let fcx = (fx1 + fx2) / 2;
    let fcy = (fy1 + fy2) / 2;

    // Generous outward corridors around the face (covers both tight 4:3 and wide 16:9 OBS cams)
    let min_left = (fx1 - (fw as f64 * 3.6) as isize).max(1) as usize;
    let max_left = (fx1 - (fw / 4).max(3)).clamp(1, w as isize - 2) as usize;
    let min_right = (fx2 + (fw / 4).max(3)).clamp(1, w as isize - 2) as usize;
    let max_right = (fx2 + (fw as f64 * 3.6) as isize).min(w as isize - 2) as usize;

    let min_top = (fy1 - (fh as f64 * 2.2) as isize).max(1) as usize;
    let max_top = (fy1 - (fh / 4).max(3)).clamp(1, h as isize - 2) as usize;
    let min_bot = (fy2 + (fh / 3).max(3)).clamp(1, h as isize - 2) as usize;
    let max_bot = (fy2 + (fh as f64 * 3.0) as isize).min(h as isize - 2) as usize;

    let y_band0 = (fcy - fh).clamp(2, h as isize - 3) as usize;
    let y_band1 = (fcy + fh).clamp(y_band0 as isize + 1, h as isize - 3) as usize;
    let x_band0 = (fcx - fw).clamp(2, w as isize - 3) as usize;
    let x_band1 = (fcx + fw).clamp(x_band0 as isize + 1, w as isize - 3) as usize;

    let collapse_and_pick_band = |hits: &[(usize, f64)], fallback: usize| -> WallScanResult {
        if hits.is_empty() {
            return WallScanResult {
                inner: fallback,
                outer: fallback,
                detected: false,
            };
        }
        // `hits` is ordered from nearest-to-face -> farthest-from-face.
        // Merge peaks within <= 26px (covers thick decorative OBS borders / glows up to ~78px in 1080p).
        let mut bands: Vec<(usize, usize, f64)> = Vec::new();
        for &(pos, strength) in hits {
            if let Some(last) = bands.last_mut() {
                if pos.abs_diff(last.1) <= 26 {
                    last.1 = pos;
                    if strength > last.2 {
                        last.2 = strength;
                    }
                    continue;
                }
            }
            bands.push((pos, pos, strength));
        }

        let max_strength = bands
            .iter()
            .map(|b| b.2)
            .fold(0.0f64, |a, b| a.max(b));
        // Stop at the FIRST strong border band near the face so we never jump past the webcam
        // into gameplay HUD lines, and ALWAYS lock onto its INNER edge (`band.0`).
        let picked = bands
            .iter()
            .find(|b| b.2 >= max_strength * 0.50 || b.2 >= 18.0)
            .copied()
            .unwrap_or(bands[0]);

        WallScanResult {
            inner: picked.0,
            outer: picked.1,
            detected: true,
        }
    };

    // Phase A: Scan OUTWARD from the face on ALL 4 sides (never skipping screen edges!)
    let scan_v_outward = |near_face: usize, far_limit: usize, go_left: bool, fallback: usize| -> WallScanResult {
        let mut hits: Vec<(usize, f64)> = Vec::new();
        let span = (y_band1 - y_band0 + 1) as f64;
        let mut x = near_face;
        loop {
            if x >= 2 && x + 2 < w {
                let mut g_sum = 0.0f64;
                let mut consistent = 0usize;
                let mut var_step_sum = 0.0f64;
                for y in y_band0..=y_band1 {
                    let d = rgb_diff(median_rgb[y * w + (x - 1)], median_rgb[y * w + (x + 1)]);
                    g_sum += d;
                    if d >= 9.0 {
                        consistent += 1;
                    }
                    let dv = (temporal_mad[y * w + (x - 2)] - temporal_mad[y * w + (x + 2)]).abs() as f64;
                    var_step_sum += dv;
                }
                let grad = g_sum / span;
                let consistency = consistent as f64 / span;
                let var_step = var_step_sum / span;
                let is_strong = (consistency >= 0.52 && grad >= 9.0)
                    || (consistency >= 0.42 && grad >= 13.5)
                    || (var_step >= 7.5 && consistency >= 0.36);
                if is_strong {
                    let strength = grad * (0.40 + 0.60 * consistency) + var_step * 0.85;
                    hits.push((x, strength));
                }
            }
            if go_left {
                if x <= far_limit || x == 0 {
                    break;
                }
                x -= 1;
            } else {
                if x >= far_limit || x + 1 >= w {
                    break;
                }
                x += 1;
            }
        }
        collapse_and_pick_band(&hits, fallback)
    };

    let scan_h_outward = |near_face: usize, far_limit: usize, go_up: bool, fallback: usize| -> WallScanResult {
        let mut hits: Vec<(usize, f64)> = Vec::new();
        let span = (x_band1 - x_band0 + 1) as f64;
        let mut y = near_face;
        loop {
            if y >= 2 && y + 2 < h {
                let mut g_sum = 0.0f64;
                let mut consistent = 0usize;
                let mut var_step_sum = 0.0f64;
                for x in x_band0..=x_band1 {
                    let d = rgb_diff(median_rgb[(y - 1) * w + x], median_rgb[(y + 1) * w + x]);
                    g_sum += d;
                    if d >= 9.0 {
                        consistent += 1;
                    }
                    let dv = (temporal_mad[(y - 2) * w + x] - temporal_mad[(y + 2) * w + x]).abs() as f64;
                    var_step_sum += dv;
                }
                let grad = g_sum / span;
                let consistency = consistent as f64 / span;
                let var_step = var_step_sum / span;
                let is_strong = (consistency >= 0.52 && grad >= 9.0)
                    || (consistency >= 0.42 && grad >= 13.5)
                    || (var_step >= 7.5 && consistency >= 0.36);
                if is_strong {
                    let strength = grad * (0.40 + 0.60 * consistency) + var_step * 0.85;
                    hits.push((y, strength));
                }
            }
            if go_up {
                if y <= far_limit || y == 0 {
                    break;
                }
                y -= 1;
            } else {
                if y >= far_limit || y + 1 >= h {
                    break;
                }
                y += 1;
            }
        }
        collapse_and_pick_band(&hits, fallback)
    };

    let def_left = (fx1 - (fw as f64 * 1.45) as isize).max(0) as usize;
    let def_right = (fx2 + (fw as f64 * 1.45) as isize).min(w as isize - 1) as usize;
    let def_top = (fy1 - (fh as f64 * 0.75) as isize).max(0) as usize;
    let def_bot = (fy2 + (fh as f64 * 1.35) as isize).min(h as isize - 1) as usize;

    let left_fb = if def_left <= 8 { 0 } else { def_left };
    let right_fb = if def_right >= w.saturating_sub(9) { w - 1 } else { def_right };
    let top_fb = if def_top <= 8 { 0 } else { def_top };
    let bot_fb = if def_bot >= h.saturating_sub(9) { h - 1 } else { def_bot };

    let l_scan = scan_v_outward(max_left, min_left, true, left_fb);
    let r_scan = scan_v_outward(min_right, max_right, false, right_fb);
    let t_scan = scan_h_outward(max_top, min_top, true, top_fb);
    let b_scan = scan_h_outward(min_bot, max_bot, false, bot_fb);

    let walls_detected = [l_scan.detected, r_scan.detected, t_scan.detected, b_scan.detected]
        .into_iter()
        .filter(|&d| d)
        .count();

    if !force_pip && walls_detected < 2 {
        let full = PixelRect {
            x: 0.005,
            y: 0.005,
            w: 0.99,
            h: 0.99,
        };
        return PipWebcamDetection {
            safe_rect: full,
            outer_rect: full,
            walls_detected,
        };
    }

    let mut inner_l = l_scan.inner.min(fx1.max(0) as usize);
    let mut inner_r = r_scan.inner.max(fx2.min(w as isize - 1) as usize);
    let mut inner_t = t_scan.inner.min(fy1.max(0) as usize);
    let mut inner_b = b_scan.inner.max(fy2.min(h as isize - 1) as usize);

    let outer_l = l_scan.outer.min(inner_l);
    let outer_r = r_scan.outer.max(inner_r);
    let outer_t = t_scan.outer.min(inner_t);
    let outer_b = b_scan.outer.max(inner_b);

    // Phase B: Iterative inward static-border / nameplate / glow peeler on ALL 4 sides
    let box_w_init = inner_r.saturating_sub(inner_l).max(32);
    let box_h_init = inner_b.saturating_sub(inner_t).max(32);
    let max_peel_x = ((box_w_init as f64 * 0.16).round() as usize).max(4);
    let max_peel_y = ((box_h_init as f64 * 0.16).round() as usize).max(4);

    let face_guard_l = (fx1 - (fw / 5).max(2)).max(0) as usize;
    let face_guard_r = (fx2 + (fw / 5).max(2)).min(w as isize - 1) as usize;
    let face_guard_t = (fy1 - (fh / 5).max(2)).max(0) as usize;
    let face_guard_b = (fy2 + (fh / 4).max(2)).min(h as isize - 1) as usize;

    let is_v_border_strip = |x: usize, inward_dx: isize, y0: usize, y1: usize| -> bool {
        if y1 <= y0 + 4 || x >= w {
            return false;
        }
        let x_in = (x as isize + inward_dx * 2).clamp(0, w as isize - 1) as usize;
        let mut tmads = Vec::with_capacity(y1 - y0 + 1);
        let mut uniform_sum = 0.0f64;
        let mut jump_cnt = 0usize;
        let mut total = 0usize;
        for y in y0..y1 {
            tmads.push(temporal_mad[y * w + x]);
            uniform_sum += rgb_diff(median_rgb[y * w + x], median_rgb[(y + 1) * w + x]);
            if rgb_diff(median_rgb[y * w + x], median_rgb[y * w + x_in]) >= 12.0 {
                jump_cnt += 1;
            }
            total += 1;
        }
        if total == 0 {
            return false;
        }
        tmads.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let med_tmad = tmads[tmads.len() / 2] as f64;
        let uniformity = uniform_sum / total as f64;
        let jump_frac = jump_cnt as f64 / total as f64;

        (med_tmad < 1.8 && uniformity < 10.5)
            || (med_tmad < 1.2 && jump_frac >= 0.45)
            || (jump_frac >= 0.50 && med_tmad < 4.2)
    };

    let is_h_border_strip = |y: usize, inward_dy: isize, x0: usize, x1: usize| -> bool {
        if x1 <= x0 + 4 || y >= h {
            return false;
        }
        let y_in = (y as isize + inward_dy * 2).clamp(0, h as isize - 1) as usize;
        let mut tmads = Vec::with_capacity(x1 - x0 + 1);
        let mut uniform_sum = 0.0f64;
        let mut jump_cnt = 0usize;
        let mut total = 0usize;
        for x in x0..x1 {
            tmads.push(temporal_mad[y * w + x]);
            uniform_sum += rgb_diff(median_rgb[y * w + x], median_rgb[y * w + (x + 1)]);
            if rgb_diff(median_rgb[y * w + x], median_rgb[y_in * w + x]) >= 12.0 {
                jump_cnt += 1;
            }
            total += 1;
        }
        if total == 0 {
            return false;
        }
        tmads.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let med_tmad = tmads[tmads.len() / 2] as f64;
        let uniformity = uniform_sum / total as f64;
        let jump_frac = jump_cnt as f64 / total as f64;

        (med_tmad < 1.8 && uniformity < 10.5)
            || (med_tmad < 1.2 && jump_frac >= 0.45)
            || (jump_frac >= 0.50 && med_tmad < 4.2)
    };

    let mid_y0 = inner_t + box_h_init / 7;
    let mid_y1 = inner_b.saturating_sub(box_h_init / 7).max(mid_y0 + 1);
    let mid_x0 = inner_l + box_w_init / 7;
    let mid_x1 = inner_r.saturating_sub(box_w_init / 7).max(mid_x0 + 1);

    let mut peeled_l = 0usize;
    let mut miss = 0usize;
    for step_i in 0..max_peel_x {
        let x = inner_l + step_i;
        if x >= face_guard_l {
            break;
        }
        if is_v_border_strip(x, 1, mid_y0, mid_y1) {
            peeled_l = step_i + 1;
            miss = 0;
        } else {
            miss += 1;
            if miss >= 2 {
                break;
            }
        }
    }
    inner_l += peeled_l;

    let mut peeled_r = 0usize;
    miss = 0;
    for step_i in 0..max_peel_x {
        let x = inner_r.saturating_sub(step_i);
        if x <= face_guard_r {
            break;
        }
        if is_v_border_strip(x, -1, mid_y0, mid_y1) {
            peeled_r = step_i + 1;
            miss = 0;
        } else {
            miss += 1;
            if miss >= 2 {
                break;
            }
        }
    }
    inner_r = inner_r.saturating_sub(peeled_r);

    let mut peeled_t = 0usize;
    miss = 0;
    for step_i in 0..max_peel_y {
        let y = inner_t + step_i;
        if y >= face_guard_t {
            break;
        }
        if is_h_border_strip(y, 1, mid_x0, mid_x1) {
            peeled_t = step_i + 1;
            miss = 0;
        } else {
            miss += 1;
            if miss >= 2 {
                break;
            }
        }
    }
    inner_t += peeled_t;

    let mut peeled_b = 0usize;
    miss = 0;
    for step_i in 0..max_peel_y {
        let y = inner_b.saturating_sub(step_i);
        if y <= face_guard_b {
            break;
        }
        if is_h_border_strip(y, -1, mid_x0, mid_x1) {
            peeled_b = step_i + 1;
            miss = 0;
        } else {
            miss += 1;
            if miss >= 2 {
                break;
            }
        }
    }
    inner_b = inner_b.saturating_sub(peeled_b);

    // Phase C: Rounded-corner mask probe (catches curved OBS corners exposing gameplay/border!)
    let bw = inner_r.saturating_sub(inner_l).max(32);
    let bh = inner_b.saturating_sub(inner_t).max(32);
    let mut core_tmads = Vec::new();
    for y in (inner_t + bh / 4)..(inner_b.saturating_sub(bh / 4)).max(inner_t + bh / 4 + 1) {
        for x in (inner_l + bw / 4)..(inner_r.saturating_sub(bw / 4)).max(inner_l + bw / 4 + 1) {
            if y < h && x < w {
                core_tmads.push(temporal_mad[y * w + x]);
            }
        }
    }
    core_tmads.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let core_tmad = core_tmads
        .get(core_tmads.len() / 2)
        .copied()
        .unwrap_or(2.0) as f64;

    let max_corner_r = ((bw.min(bh) as f64 * 0.12).round() as usize).min(20);
    let probe_corner = |sx: usize, sy: usize, dx: isize, dy: isize| -> usize {
        let mut depth = 0usize;
        for d in 0..max_corner_r {
            let cx = (sx as isize + dx * d as isize).clamp(0, w as isize - 1) as usize;
            let cy = (sy as isize + dy * d as isize).clamp(0, h as isize - 1) as usize;
            let ix = (sx as isize + dx * (d + 4) as isize).clamp(0, w as isize - 1) as usize;
            let iy = (sy as isize + dy * (d + 4) as isize).clamp(0, h as isize - 1) as usize;
            let tm = temporal_mad[cy * w + cx] as f64;
            let cd = rgb_diff(median_rgb[cy * w + cx], median_rgb[iy * w + ix]);
            let masked_out = (tm > core_tmad * 2.5 + 9.0) || (tm < 1.2 && cd > 22.0);
            if masked_out {
                depth = d + 1;
            } else {
                break;
            }
        }
        depth
    };

    let c_tl = probe_corner(inner_l, inner_t, 1, 1);
    let c_tr = probe_corner(inner_r, inner_t, -1, 1);
    let c_bl = probe_corner(inner_l, inner_b, 1, -1);
    let c_br = probe_corner(inner_r, inner_b, -1, -1);
    let corner_inset = c_tl.max(c_tr).max(c_bl).max(c_br);

    inner_l = (inner_l + corner_inset).min(face_guard_l);
    inner_r = inner_r.saturating_sub(corner_inset).max(face_guard_r);
    inner_t = (inner_t + corner_inset).min(face_guard_t);
    inner_b = inner_b.saturating_sub(corner_inset).max(face_guard_b);

    // Phase D: Guaranteed Zero-Border Safety Inset (minimum >= 3px in analysis space on every
    // detected/interior wall so anti-aliasing, shadows, or 1-2px glows can NEVER leak in!)
    let box_w = inner_r.saturating_sub(inner_l).max(32) as f64;
    let box_h = inner_b.saturating_sub(inner_t).max(32) as f64;

    let trim_l = if inner_l <= 2 && !l_scan.detected && peeled_l == 0 {
        (box_w * 0.02).max(2.0)
    } else {
        (box_w * trim_pct).max(3.0)
    };
    let trim_r = if inner_r >= w.saturating_sub(3) && !r_scan.detected && peeled_r == 0 {
        (box_w * 0.02).max(2.0)
    } else {
        (box_w * trim_pct).max(3.0)
    };
    let trim_t = if inner_t <= 2 && !t_scan.detected && peeled_t == 0 {
        (box_h * 0.02).max(2.0)
    } else {
        (box_h * trim_pct).max(3.0)
    };
    let trim_b = if inner_b >= h.saturating_sub(3) && !b_scan.detected && peeled_b == 0 {
        (box_h * 0.02).max(2.0)
    } else {
        (box_h * trim_pct).max(3.0)
    };

    let sx1 = (inner_l as f64 + trim_l).clamp(0.0, w as f64 - 16.0);
    let sy1 = (inner_t as f64 + trim_t).clamp(0.0, h as f64 - 16.0);
    let sx2 = (inner_r as f64 - trim_r).clamp(sx1 + 16.0, w as f64);
    let sy2 = (inner_b as f64 - trim_b).clamp(sy1 + 16.0, h as f64);

    let safe_rect = PixelRect {
        x: sx1 / w as f64,
        y: sy1 / h as f64,
        w: (sx2 - sx1) / w as f64,
        h: (sy2 - sy1) / h as f64,
    };
    let outer_rect = PixelRect {
        x: outer_l as f64 / w as f64,
        y: outer_t as f64 / h as f64,
        w: (outer_r.saturating_sub(outer_l)).max(16) as f64 / w as f64,
        h: (outer_b.saturating_sub(outer_t)).max(16) as f64 / h as f64,
    };
    PipWebcamDetection {
        safe_rect,
        outer_rect,
        walls_detected,
    }
}

fn expand_norm_rect(r: PixelRect, pad: f64) -> PixelRect {
    let x1 = (r.x - pad).clamp(0.0, 1.0);
    let y1 = (r.y - pad).clamp(0.0, 1.0);
    let x2 = (r.right() + pad).clamp(x1, 1.0);
    let y2 = (r.bottom() + pad).clamp(y1, 1.0);
    PixelRect {
        x: x1,
        y: y1,
        w: (x2 - x1).max(0.0),
        h: (y2 - y1).max(0.0),
    }
}

/// Converts raw per-frame face detections into a silky-smooth, natural cameraman trajectory
/// strictly clamped inside `safe_box_px` so ZERO borders can ever appear.
#[allow(clippy::too_many_arguments)]
pub fn build_natural_camera_track(
    samples: &[(f64, Option<FaceBox>)],
    safe_box_px: PixelRect,
    src_w: u32,
    src_h: u32,
    target_aspect: f64,
    face_zoom: f64,
    total_duration_sec: f64,
    is_pip_webcam: bool,
) -> CameraTrackPlan {
    // 1. Compute median face height in source pixels for ideal cameraman framing size
    let mut face_heights: Vec<f64> = samples
        .iter()
        .filter_map(|(_, f)| f.map(|b| b.h() as f64 * src_h as f64))
        .collect();
    face_heights.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let median_fh = face_heights
        .get(face_heights.len() / 2)
        .copied()
        .unwrap_or(safe_box_px.h * 0.35);

    // 2. Compute strict integer interior of safe_box_px so NOT EVEN 1px can ever leak out
    let safe_x0 = safe_box_px
        .x
        .ceil()
        .clamp(0.0, src_w.saturating_sub(32) as f64);
    let safe_y0 = safe_box_px
        .y
        .ceil()
        .clamp(0.0, src_h.saturating_sub(32) as f64);
    let safe_x1 = safe_box_px
        .right()
        .floor()
        .clamp(safe_x0 + 32.0, src_w as f64);
    let safe_y1 = safe_box_px
        .bottom()
        .floor()
        .clamp(safe_y0 + 32.0, src_h as f64);
    let safe_w_int = (safe_x1 - safe_x0).max(32.0);
    let safe_h_int = (safe_y1 - safe_y0).max(32.0);

    // Maximum aspect-locked crop that fits 100% inside the strict integer safe_box
    let (max_cw, max_ch) = if safe_w_int / safe_h_int.max(1.0) >= target_aspect {
        let ch = safe_h_int;
        (ch * target_aspect, ch)
    } else {
        let cw = safe_w_int;
        (cw, cw / target_aspect)
    };

    // For PiP webcams, show the FULL clean interior (no zoom glued to the face!) by default
    // (`face_zoom == 1.0` -> 100% of `max_ch`, only sliding gently within spare aspect margin).
    // Only zoom closer when the user explicitly requests `face_zoom > 1.0` (e.g. 1.15 / 1.30).
    let desired_ch = if is_pip_webcam {
        let zoom_divisor = face_zoom.clamp(1.0, 1.45);
        (max_ch / zoom_divisor).clamp(max_ch * 0.72, max_ch)
    } else {
        (median_fh / (0.28 * face_zoom)).clamp(max_ch * 0.55, max_ch * 0.96)
    };
    let desired_cw = (desired_ch * target_aspect).min(max_cw);
    let desired_ch = (desired_cw / target_aspect).min(max_ch);

    // Keep even integer dimensions strictly <= safe_w_int / safe_h_int for yuv420p compatibility
    let max_even_w = ((safe_w_int.floor() as u32) & !1).clamp(32, src_w & !1);
    let max_even_h = ((safe_h_int.floor() as u32) & !1).clamp(32, src_h & !1);
    let crop_w = ((desired_cw.floor() as u32) & !1).clamp(32.min(max_even_w), max_even_w);
    let crop_h = ((desired_ch.floor() as u32) & !1).clamp(32.min(max_even_h), max_even_h);

    let min_x = safe_x0.clamp(0.0, (src_w - crop_w) as f64);
    let max_x = (safe_x1 - crop_w as f64).clamp(min_x, (src_w - crop_w) as f64);
    let min_y = safe_y0.clamp(0.0, (src_h - crop_h) as f64);
    let max_y = (safe_y1 - crop_h as f64).clamp(min_y, (src_h - crop_h) as f64);

    let clamp_xy = |x: f64, y: f64| -> (f64, f64) {
        (x.clamp(min_x, max_x), y.clamp(min_y, max_y))
    };

    // 3. Map each detection to its ideal Rule-of-Thirds crop top-left (eyes at 38% from top)
    let first_face = samples.iter().find_map(|(_, f)| *f);
    let default_pos = if let Some(f) = first_face {
        let tx = f.cx() as f64 * src_w as f64 - crop_w as f64 * 0.5;
        let ty = f.eye_y() as f64 * src_h as f64 - crop_h as f64 * 0.38;
        clamp_xy(tx, ty)
    } else {
        clamp_xy((min_x + max_x) * 0.5, (min_y + max_y) * 0.5)
    };

    let mut raw_targets: Vec<(f64, f64, f64, bool)> = Vec::with_capacity(samples.len());
    let mut last_valid = default_pos;
    let mut last_raw_cx: Option<f64> = first_face.map(|f| f.cx() as f64);

    for &(t, face_opt) in samples {
        if let Some(f) = face_opt {
            let tx = f.cx() as f64 * src_w as f64 - crop_w as f64 * 0.5;
            let ty = f.eye_y() as f64 * src_h as f64 - crop_h as f64 * 0.38;
            let (cx, cy) = clamp_xy(tx, ty);
            let scene_cut = last_raw_cx
                .map(|prev| (f.cx() as f64 - prev).abs() > 0.28)
                .unwrap_or(false);
            last_valid = (cx, cy);
            last_raw_cx = Some(f.cx() as f64);
            raw_targets.push((t, cx, cy, scene_cut));
        } else {
            // Occlusion hold: stay rock-steady when face briefly leaves frame
            raw_targets.push((t, last_valid.0, last_valid.1, false));
        }
    }

    // 4. Simulate fluid-head tripod camera physics at 20 Hz (0.05s steps)
    let dt = 0.05f64;
    let deadzone_x = crop_w as f64 * 0.03;
    let deadzone_y = crop_h as f64 * 0.04;
    let max_step_x = (crop_w as f64 * 0.20) * dt;
    let max_step_y = (crop_h as f64 * 0.20) * dt;
    let damping = 0.14f64;

    let mut cur_x = default_pos.0;
    let mut cur_y = default_pos.1;
    let mut keyframes: Vec<(f64, i32, i32)> = Vec::new();
    let mut last_emitted = (cur_x.round() as i32, cur_y.round() as i32);
    keyframes.push((0.0, last_emitted.0, last_emitted.1));

    let steps = ((total_duration_sec.max(0.5) / dt).ceil() as usize).min(6000);
    let mut r_idx = 0usize;

    for s in 1..=steps {
        let t = s as f64 * dt;
        while r_idx + 1 < raw_targets.len() && raw_targets[r_idx + 1].0 <= t {
            r_idx += 1;
        }
        let (_, targ_x, targ_y, scene_cut) = raw_targets[r_idx];

        if scene_cut {
            cur_x = targ_x;
            cur_y = targ_y;
        } else {
            let dx = targ_x - cur_x;
            let dy = targ_y - cur_y;
            if dx.abs() > deadzone_x {
                let desired = dx * damping;
                cur_x += desired.clamp(-max_step_x, max_step_x);
            }
            if dy.abs() > deadzone_y {
                let desired = dy * damping;
                cur_y += desired.clamp(-max_step_y, max_step_y);
            }
        }

        let (clamped_x, clamped_y) = clamp_xy(cur_x, cur_y);
        cur_x = clamped_x;
        cur_y = clamped_y;

        let ix = cur_x.round() as i32;
        let iy = cur_y.round() as i32;
        if (ix - last_emitted.0).abs() >= 1 || (iy - last_emitted.1).abs() >= 1 {
            keyframes.push((t, ix, iy));
            last_emitted = (ix, iy);
        }
    }

    let detected_faces_count = samples.iter().filter(|(_, f)| f.is_some()).count();
    CameraTrackPlan {
        crop_w,
        crop_h,
        init_x: keyframes[0].1,
        init_y: keyframes[0].2,
        keyframes,
        filter_target: "crop@cam".to_string(),
        detected_faces_count,
        is_pip_webcam,
    }
}

const SALIENCY_GW: usize = 32;
const SALIENCY_GH: usize = 18;
const SALIENCY_CELLS: usize = SALIENCY_GW * SALIENCY_GH;

#[derive(Debug, Clone, Copy)]
pub struct ContentTarget {
    pub t: f64,
    pub target_x: f64,
    pub target_y: f64,
    pub hard_cut: bool,
    pub has_speaker: bool,
}

#[derive(Debug, Clone, Copy)]
struct SpeakerMemory {
    cx: f32,
    cy: f32,
    ema_mouth: f64,
}

fn rect_contains_point(r: PixelRect, x: f64, y: f64) -> bool {
    x >= r.x && x <= r.right() && y >= r.y && y <= r.bottom()
}

fn compute_content_crop_dims(src_w: u32, src_h: u32, content_aspect: f64) -> (u32, u32) {
    let src_aspect = src_w as f64 / src_h.max(1) as f64;
    let (desired_cw, desired_ch) = if src_aspect > content_aspect {
        // Landscape source into taller/squarer bottom slot: gentle 94% height gives
        // vertical framing headroom for speaker eyes + wide horizontal pan across 16:9.
        let ch = (src_h as f64 * 0.94).min(src_w as f64 / content_aspect);
        (ch * content_aspect, ch)
    } else {
        let cw = (src_w as f64 * 0.94).min(src_h as f64 * content_aspect);
        (cw, cw / content_aspect)
    };
    let crop_w = ((desired_cw.floor() as u32) & !1).clamp(64.min(src_w & !1), src_w & !1);
    let crop_h = ((desired_ch.floor() as u32) & !1).clamp(64.min(src_h & !1), src_h & !1);
    (crop_w.max(2), crop_h.max(2))
}

fn measure_global_frame_mad(prev: &RgbImage, curr: &RgbImage) -> f64 {
    if prev.width() != curr.width() || prev.height() != curr.height() {
        return 0.0;
    }
    let w = curr.width();
    let h = curr.height();
    let mut sum = 0.0f64;
    let mut count = 0usize;
    let mut y = 0;
    while y < h {
        let mut x = 0;
        while x < w {
            let a = prev.get_pixel(x, y);
            let b = curr.get_pixel(x, y);
            sum += ((a[0] as i32 - b[0] as i32).abs()
                + (a[1] as i32 - b[1] as i32).abs()
                + (a[2] as i32 - b[2] as i32).abs()) as f64
                / 3.0;
            count += 1;
            x += 16;
        }
        y += 16;
    }
    if count == 0 { 0.0 } else { sum / count as f64 }
}

/// Estimates speaking activity by measuring lower-face / mouth-jaw temporal pixel motion
/// inside a detected face box, compensated against global background camera motion.
fn measure_face_mouth_motion(
    prev: &RgbImage,
    curr: &RgbImage,
    face: FaceBox,
    global_mad: f64,
) -> f64 {
    if prev.width() != curr.width() || prev.height() != curr.height() {
        return 0.0;
    }
    let w = curr.width() as f64;
    let h = curr.height() as f64;
    let x1 = ((face.x1 + face.w() * 0.18) as f64 * w).clamp(0.0, (w - 2.0).max(0.0)) as u32;
    let x2_lo = x1 as f64 + 1.0;
    let x2 = ((face.x2 - face.w() * 0.18) as f64 * w).clamp(x2_lo, (w - 1.0).max(x2_lo)) as u32;
    let y1 = ((face.y1 + face.h() * 0.54) as f64 * h).clamp(0.0, (h - 2.0).max(0.0)) as u32;
    let y2_lo = y1 as f64 + 1.0;
    let y2 = ((face.y2 - face.h() * 0.04) as f64 * h).clamp(y2_lo, (h - 1.0).max(y2_lo)) as u32;

    let mut sum = 0.0f64;
    let mut count = 0usize;
    let mut y = y1;
    while y <= y2 && y < curr.height() {
        let mut x = x1;
        while x <= x2 && x < curr.width() {
            let a = prev.get_pixel(x, y);
            let b = curr.get_pixel(x, y);
            sum += ((a[0] as i32 - b[0] as i32).abs()
                + (a[1] as i32 - b[1] as i32).abs()
                + (a[2] as i32 - b[2] as i32).abs()) as f64
                / 3.0;
            count += 1;
            x += 2;
        }
        y += 2;
    }
    if count == 0 {
        return 0.0;
    }
    let mouth_mad = sum / count as f64;
    (mouth_mad - global_mad * 0.65).max(0.0)
}

/// Computes a fast 32x18 visual motion + spatial contrast/saturation saliency grid,
/// masking out `exclude_rect` (the streamer's PiP webcam already shown in the top Short window).
fn compute_saliency_grid(
    prev: Option<&RgbImage>,
    curr: &RgbImage,
    is_scene_cut: bool,
    exclude_rect: Option<PixelRect>,
) -> ([f64; SALIENCY_CELLS], [bool; SALIENCY_CELLS], f64) {
    let w = curr.width() as usize;
    let h = curr.height() as usize;
    let mut raw_motion = [0.0f64; SALIENCY_CELLS];
    let mut raw_edge = [0.0f64; SALIENCY_CELLS];
    let mut excluded = [false; SALIENCY_CELLS];
    let mut active_motions: Vec<f64> = Vec::with_capacity(SALIENCY_CELLS);

    let use_prev = prev.filter(|p| !is_scene_cut && p.width() as usize == w && p.height() as usize == h);

    for gy in 0..SALIENCY_GH {
        let py0 = (gy * h) / SALIENCY_GH;
        let py1 = (((gy + 1) * h) / SALIENCY_GH).min(h.saturating_sub(2));
        let cy_n = (gy as f64 + 0.5) / SALIENCY_GH as f64;

        for gx in 0..SALIENCY_GW {
            let idx = gy * SALIENCY_GW + gx;
            let cx_n = (gx as f64 + 0.5) / SALIENCY_GW as f64;
            if let Some(ex) = exclude_rect {
                if rect_contains_point(ex, cx_n, cy_n) {
                    excluded[idx] = true;
                    continue;
                }
            }

            let px0 = (gx * w) / SALIENCY_GW;
            let px1 = (((gx + 1) * w) / SALIENCY_GW).min(w.saturating_sub(2));
            let mut m_sum = 0.0f64;
            let mut e_sum = 0.0f64;
            let mut cnt = 0usize;

            let mut y = py0;
            while y < py1 {
                let mut x = px0;
                while x < px1 {
                    let p = curr.get_pixel(x as u32, y as u32);
                    let pr = curr.get_pixel((x + 2).min(w - 1) as u32, y as u32);
                    let pd = curr.get_pixel(x as u32, (y + 2).min(h - 1) as u32);

                    let gx_d = ((p[0] as i32 - pr[0] as i32).abs()
                        + (p[1] as i32 - pr[1] as i32).abs()
                        + (p[2] as i32 - pr[2] as i32).abs()) as f64
                        / 3.0;
                    let gy_d = ((p[0] as i32 - pd[0] as i32).abs()
                        + (p[1] as i32 - pd[1] as i32).abs()
                        + (p[2] as i32 - pd[2] as i32).abs()) as f64
                        / 3.0;
                    let max_c = p[0].max(p[1]).max(p[2]) as i32;
                    let min_c = p[0].min(p[1]).min(p[2]) as i32;
                    let sat = (max_c - min_c) as f64 * 0.18;
                    e_sum += (gx_d + gy_d) * 0.5 + sat;

                    if let Some(prev_img) = use_prev {
                        let pp = prev_img.get_pixel(x as u32, y as u32);
                        let md = ((p[0] as i32 - pp[0] as i32).abs()
                            + (p[1] as i32 - pp[1] as i32).abs()
                            + (p[2] as i32 - pp[2] as i32).abs()) as f64
                            / 3.0;
                        m_sum += md;
                    }
                    cnt += 1;
                    x += 2;
                }
                y += 2;
            }

            if cnt > 0 {
                let m_avg = m_sum / cnt as f64;
                raw_motion[idx] = m_avg;
                raw_edge[idx] = e_sum / cnt as f64;
                active_motions.push(m_avg);
            }
        }
    }

    // Camera-pan compensation: subtract median motion across non-excluded cells so uniform
    // camera panning doesn't drown out the actual moving subject / action highlight.
    active_motions.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let median_motion = active_motions
        .get(active_motions.len() / 2)
        .copied()
        .unwrap_or(0.0);
    let p95_motion = active_motions
        .get((active_motions.len() * 95) / 100)
        .copied()
        .unwrap_or(0.0);
    let peak_motion_strength = (p95_motion - median_motion).max(0.0);

    let mut energy = [0.0f64; SALIENCY_CELLS];
    for gy in 0..SALIENCY_GH {
        let dy = ((gy as f64 + 0.5) / SALIENCY_GH as f64 - 0.5) * 2.0;
        for gx in 0..SALIENCY_GW {
            let idx = gy * SALIENCY_GW + gx;
            if excluded[idx] {
                continue;
            }
            let dx = ((gx as f64 + 0.5) / SALIENCY_GW as f64 - 0.5) * 2.0;
            // Gentle center prior keeps calm scenes stable while real side action easily wins
            let prior = 0.72 + 0.28 * (-(dx * dx + dy * dy) * 0.75).exp();
            let comp_motion = (raw_motion[idx] - median_motion * 0.85).max(0.0);
            energy[idx] = (comp_motion * 2.4 + raw_edge[idx] * 0.36) * prior;
        }
    }

    (energy, excluded, peak_motion_strength)
}

/// Slides a crop-proportioned window across the 32x18 saliency grid to locate the exact
/// framing `(target_x, target_y)` that maximizes captured visual action & highlights while
/// avoiding excluded webcam areas.
#[allow(clippy::too_many_arguments)]
fn find_best_saliency_window(
    ema_grid: &[f64; SALIENCY_CELLS],
    excluded: &[bool; SALIENCY_CELLS],
    src_w: u32,
    src_h: u32,
    crop_w: u32,
    crop_h: u32,
) -> (f64, f64) {
    let win_cols = ((crop_w as f64 / src_w.max(1) as f64) * SALIENCY_GW as f64)
        .round()
        .clamp(1.0, SALIENCY_GW as f64) as usize;
    let win_rows = ((crop_h as f64 / src_h.max(1) as f64) * SALIENCY_GH as f64)
        .round()
        .clamp(1.0, SALIENCY_GH as f64) as usize;

    let mut sum_active = 0.0f64;
    let mut cnt_active = 0usize;
    for i in 0..SALIENCY_CELLS {
        if !excluded[i] {
            sum_active += ema_grid[i];
            cnt_active += 1;
        }
    }
    let avg_cell = if cnt_active > 0 {
        sum_active / cnt_active as f64
    } else {
        4.0
    };
    let overlap_penalty = avg_cell.max(4.0) * 0.45;

    let max_wx = SALIENCY_GW.saturating_sub(win_cols);
    let max_wy = SALIENCY_GH.saturating_sub(win_rows);
    let center_wx = max_wx as f64 * 0.5;
    let center_wy = max_wy as f64 * 0.5;

    let mut best_wx = max_wx / 2;
    let mut best_wy = max_wy / 2;
    let mut best_score = f64::NEG_INFINITY;

    for wy in 0..=max_wy {
        for wx in 0..=max_wx {
            let mut score = 0.0f64;
            for dy in 0..win_rows {
                let row_off = (wy + dy) * SALIENCY_GW;
                for dx in 0..win_cols {
                    let idx = row_off + wx + dx;
                    if excluded[idx] {
                        score -= overlap_penalty;
                    } else {
                        score += ema_grid[idx];
                    }
                }
            }
            // Tiny tie-breaker toward center when scene is completely uniform
            let dist_c = ((wx as f64 - center_wx).abs() + (wy as f64 - center_wy).abs()) * 0.01;
            score -= dist_c;
            if score > best_score {
                best_score = score;
                best_wx = wx;
                best_wy = wy;
            }
        }
    }

    let max_px_x = src_w.saturating_sub(crop_w) as f64;
    let max_px_y = src_h.saturating_sub(crop_h) as f64;
    let tx = if max_wx > 0 {
        (best_wx as f64 / max_wx as f64) * max_px_x
    } else {
        max_px_x * 0.5
    };
    let ty = if max_wy > 0 {
        (best_wy as f64 / max_wy as f64) * max_px_y
    } else {
        max_px_y * 0.5
    };
    (tx, ty)
}

/// Computes the AI Director track for the bottom content window (`crop@content`):
/// 1. Detects foreground speakers (excluding the streamer's corner PiP webcam) and tracks
///    who is actively speaking via lower-face/mouth motion + hysteresis + clean speaker cuts.
/// 2. When no foreground speaker dominates (gameplay, trailers, action moments), frames the
///    peak visual action/highlight via a 32x18 camera-pan-compensated saliency window.
pub fn compute_content_director_track(
    loaded: &[(f64, RgbImage)],
    detector: Option<&FaceDetector>,
    exclude_rect: Option<PixelRect>,
    src_w: u32,
    src_h: u32,
    content_aspect: f64,
    total_duration_sec: f64,
) -> CameraTrackPlan {
    compute_content_director_track_in_region(
        loaded,
        detector,
        exclude_rect,
        None,
        src_w,
        src_h,
        content_aspect,
        total_duration_sec,
    )
}

/// Runs the AI Content Director optionally bounded inside a user-saved `region_rect` (normalized 0..1).
#[allow(clippy::too_many_arguments)]
pub fn compute_content_director_track_in_region(
    loaded: &[(f64, RgbImage)],
    detector: Option<&FaceDetector>,
    exclude_rect: Option<PixelRect>,
    region_rect: Option<PixelRect>,
    src_w: u32,
    src_h: u32,
    content_aspect: f64,
    total_duration_sec: f64,
) -> CameraTrackPlan {
    let bounds_px = region_rect.map(|r| PixelRect {
        x: (r.x * src_w as f64).ceil().clamp(0.0, src_w.saturating_sub(64) as f64),
        y: (r.y * src_h as f64).ceil().clamp(0.0, src_h.saturating_sub(64) as f64),
        w: (r.w * src_w as f64).floor().clamp(64.0, src_w as f64),
        h: (r.h * src_h as f64).floor().clamp(64.0, src_h as f64),
    });
    let eff_w = bounds_px
        .map(|b| (b.w as u32).min(src_w))
        .unwrap_or(src_w);
    let eff_h = bounds_px
        .map(|b| (b.h as u32).min(src_h))
        .unwrap_or(src_h);
    let (crop_w, crop_h) = compute_content_crop_dims(eff_w, eff_h, content_aspect);
    let min_x = bounds_px.map(|b| b.x).unwrap_or(0.0);
    let min_y = bounds_px.map(|b| b.y).unwrap_or(0.0);
    let max_x = bounds_px
        .map(|b| (b.right() - crop_w as f64).clamp(min_x, src_w.saturating_sub(crop_w) as f64))
        .unwrap_or(src_w.saturating_sub(crop_w) as f64);
    let max_y = bounds_px
        .map(|b| (b.bottom() - crop_h as f64).clamp(min_y, src_h.saturating_sub(crop_h) as f64))
        .unwrap_or(src_h.saturating_sub(crop_h) as f64);
    let crop_w_norm = crop_w as f64 / src_w.max(1) as f64;

    let mut ema_grid = [0.0f64; SALIENCY_CELLS];
    let mut ema_initialized = false;

    let content_face_stride = if loaded.len() > 160 { 2 } else { 1 };
    let mut cached_faces: Vec<FaceBox> = Vec::new();
    let mut speaker_mem: Vec<SpeakerMemory> = Vec::new();
    let mut active_speaker_pos: Option<(f32, f32, f64)> = None; // (cx, cy, locked_since_t)

    let mut targets: Vec<ContentTarget> = Vec::with_capacity(loaded.len());
    let mut detected_speaker_frames = 0usize;

    for (idx, (t, rgb)) in loaded.iter().enumerate() {
        let prev_rgb = if idx > 0 { Some(&loaded[idx - 1].1) } else { None };
        let global_mad = prev_rgb
            .map(|p| measure_global_frame_mad(p, rgb))
            .unwrap_or(0.0);
        let is_scene_cut = idx > 0 && global_mad > 34.0;

        if is_scene_cut {
            speaker_mem.clear();
            active_speaker_pos = None;
            ema_initialized = false;
        }

        // 1. Visual Action & Highlight Saliency (32x18 grid, webcam masked out)
        let (frame_energy, excluded, peak_motion_strength) =
            compute_saliency_grid(prev_rgb, rgb, is_scene_cut, exclude_rect);
        if !ema_initialized {
            ema_grid = frame_energy;
            ema_initialized = true;
        } else {
            for i in 0..SALIENCY_CELLS {
                ema_grid[i] = ema_grid[i] * 0.65 + frame_energy[i] * 0.35;
            }
        }
        let (action_tx, action_ty) =
            find_best_saliency_window(&ema_grid, &excluded, src_w, src_h, crop_w, crop_h);

        // 2. Active Speaker Detection in Content Area (excluding streamer's PiP webcam)
        if let Some(det) = detector {
            if idx % content_face_stride == 0 || is_scene_cut {
                cached_faces = det
                    .detect(rgb, 0.56)
                    .unwrap_or_default()
                    .into_iter()
                    .filter(|f| {
                        let too_small = f.w() < 0.032 || f.h() < 0.045;
                        let inside_cam = exclude_rect
                            .map(|ex| rect_contains_point(ex, f.cx() as f64, f.cy() as f64))
                            .unwrap_or(false);
                        let inside_region = region_rect
                            .map(|rg| rect_contains_point(rg, f.cx() as f64, f.cy() as f64))
                            .unwrap_or(true);
                        !too_small && !inside_cam && inside_region
                    })
                    .collect();
            }
        } else {
            cached_faces.clear();
        }

        // Measure mouth activity for each detected content face & update temporal EMA
        let mut scored_faces: Vec<(FaceBox, f64, f64)> = Vec::with_capacity(cached_faces.len());
        let mut next_mem: Vec<SpeakerMemory> = Vec::with_capacity(cached_faces.len());

        for &f in &cached_faces {
            let raw_mouth = match prev_rgb {
                Some(p) if !is_scene_cut => measure_face_mouth_motion(p, rgb, f, global_mad),
                _ => 0.0,
            };
            let prev_ema = speaker_mem
                .iter()
                .filter(|m| ((m.cx - f.cx()).powi(2) + (m.cy - f.cy()).powi(2)).sqrt() < 0.14)
                .map(|m| m.ema_mouth)
                .next()
                .unwrap_or(raw_mouth);
            let ema_mouth = prev_ema * 0.55 + raw_mouth * 0.45;
            next_mem.push(SpeakerMemory {
                cx: f.cx(),
                cy: f.cy(),
                ema_mouth,
            });

            let mut score = f.score as f64 * 0.35
                + (f.h() as f64 * 2.2).min(0.85)
                + (ema_mouth / 14.0).min(1.85);

            // Hysteresis bonus for currently framed speaker prevents frantic ping-ponging
            if let Some((acx, acy, since_t)) = active_speaker_pos {
                let d = ((f.cx() - acx).powi(2) + (f.cy() - acy).powi(2)).sqrt();
                if d < 0.14 {
                    let hold_bonus = if (*t - since_t) < 1.5 { 0.42 } else { 0.18 };
                    score += hold_bonus;
                }
            }
            scored_faces.push((f, ema_mouth, score));
        }
        speaker_mem = next_mem;

        let best_speaker = scored_faces
            .iter()
            .max_by(|a, b| a.2.partial_cmp(&b.2).unwrap_or(std::cmp::Ordering::Equal))
            .copied();

        // 3. Fuse Speaker vs Action Highlight
        let use_speaker = best_speaker.filter(|(f, _, _)| {
            // If only a tiny distant background face is present during huge gameplay action, let action win
            !(f.h() < 0.075 && peak_motion_strength > 24.0)
        });

        if let Some((spk, _, _)) = use_speaker {
            detected_speaker_frames += 1;
            let speaker_switched_far = active_speaker_pos
                .map(|(acx, _, _)| (spk.cx() - acx).abs() > 0.20)
                .unwrap_or(false);
            let same_speaker = active_speaker_pos
                .map(|(acx, acy, _)| ((spk.cx() - acx).powi(2) + (spk.cy() - acy).powi(2)).sqrt() < 0.14)
                .unwrap_or(false);
            let since_t = if same_speaker {
                active_speaker_pos.map(|(_, _, s)| s).unwrap_or(*t)
            } else {
                *t
            };
            active_speaker_pos = Some((spk.cx(), spk.cy(), since_t));

            // Natural two-shot if a co-speaker fits comfortably alongside the active speaker
            let mut focus_cx = spk.cx() as f64;
            let mut focus_eye_y = spk.eye_y() as f64;
            if let Some((co, _, _)) = scored_faces
                .iter()
                .filter(|(other, _, _)| {
                    let span_x = other.x2.max(spk.x2) - other.x1.min(spk.x1);
                    let sep = (other.cx() - spk.cx()).abs();
                    sep > 0.06 && (span_x as f64) < crop_w_norm * 0.76
                })
                .max_by(|a, b| a.2.partial_cmp(&b.2).unwrap_or(std::cmp::Ordering::Equal))
            {
                focus_cx = spk.cx() as f64 * 0.65 + co.cx() as f64 * 0.35;
                focus_eye_y = spk.eye_y() as f64 * 0.65 + co.eye_y() as f64 * 0.35;
            }

            let tx = (focus_cx * src_w as f64 - crop_w as f64 * 0.5).clamp(min_x, max_x);
            let ty = (focus_eye_y * src_h as f64 - crop_h as f64 * 0.36).clamp(min_y, max_y);
            targets.push(ContentTarget {
                t: *t,
                target_x: tx,
                target_y: ty,
                hard_cut: is_scene_cut || speaker_switched_far,
                has_speaker: true,
            });
        } else {
            active_speaker_pos = None;
            targets.push(ContentTarget {
                t: *t,
                target_x: action_tx.clamp(min_x, max_x),
                target_y: action_ty.clamp(min_y, max_y),
                hard_cut: is_scene_cut,
                has_speaker: false,
            });
        }
    }

    build_content_camera_track(
        &targets,
        src_w,
        src_h,
        crop_w,
        crop_h,
        total_duration_sec,
        detected_speaker_frames,
    )
}

/// Converts per-frame content director targets into a smooth 20 Hz FFmpeg `sendcmd` trajectory
/// for `crop@content` — instant editorial cuts on scene changes / speaker switches, silky
/// fluid-head pan when tracking continuous action or speaker movement.
pub fn build_content_camera_track(
    targets: &[ContentTarget],
    src_w: u32,
    src_h: u32,
    crop_w: u32,
    crop_h: u32,
    total_duration_sec: f64,
    detected_speaker_frames: usize,
) -> CameraTrackPlan {
    let max_x = src_w.saturating_sub(crop_w) as f64;
    let max_y = src_h.saturating_sub(crop_h) as f64;
    let clamp_xy = |x: f64, y: f64| (x.clamp(0.0, max_x), y.clamp(0.0, max_y));

    let init = targets
        .first()
        .map(|t| clamp_xy(t.target_x, t.target_y))
        .unwrap_or((max_x * 0.5, max_y * 0.5));

    if targets.is_empty() || (max_x < 1.0 && max_y < 1.0) {
        let ix = init.0.round() as i32;
        let iy = init.1.round() as i32;
        return CameraTrackPlan {
            crop_w,
            crop_h,
            init_x: ix,
            init_y: iy,
            keyframes: vec![(0.0, ix, iy)],
            filter_target: "crop@content".to_string(),
            detected_faces_count: detected_speaker_frames,
            is_pip_webcam: false,
        };
    }

    let dt = 0.05f64;
    let deadzone_x = crop_w as f64 * 0.045;
    let deadzone_y = crop_h as f64 * 0.055;
    let max_step_x = (crop_w as f64 * 0.24) * dt;
    let max_step_y = (crop_h as f64 * 0.22) * dt;
    let damping = 0.13f64;

    let mut cur_x = init.0;
    let mut cur_y = init.1;
    let mut last_emitted = (cur_x.round() as i32, cur_y.round() as i32);
    let mut keyframes = vec![(0.0, last_emitted.0, last_emitted.1)];

    let steps = ((total_duration_sec.max(0.5) / dt).ceil() as usize).min(6000);
    let mut r_idx = 0usize;
    let mut applied_cut_idx: Option<usize> = None;

    for s in 1..=steps {
        let t = s as f64 * dt;
        while r_idx + 1 < targets.len() && targets[r_idx + 1].t <= t {
            r_idx += 1;
        }
        let ct = targets[r_idx];
        let (targ_x, targ_y) = clamp_xy(ct.target_x, ct.target_y);

        if ct.hard_cut && applied_cut_idx != Some(r_idx) {
            cur_x = targ_x;
            cur_y = targ_y;
            applied_cut_idx = Some(r_idx);
        } else {
            let dx = targ_x - cur_x;
            let dy = targ_y - cur_y;
            if dx.abs() > deadzone_x {
                cur_x += (dx * damping).clamp(-max_step_x, max_step_x);
            }
            if dy.abs() > deadzone_y {
                cur_y += (dy * damping).clamp(-max_step_y, max_step_y);
            }
        }

        let (cx, cy) = clamp_xy(cur_x, cur_y);
        cur_x = cx;
        cur_y = cy;

        let ix = cur_x.round() as i32;
        let iy = cur_y.round() as i32;
        if (ix - last_emitted.0).abs() >= 1 || (iy - last_emitted.1).abs() >= 1 {
            keyframes.push((t, ix, iy));
            last_emitted = (ix, iy);
        }
    }

    CameraTrackPlan {
        crop_w,
        crop_h,
        init_x: keyframes[0].1,
        init_y: keyframes[0].2,
        keyframes,
        filter_target: "crop@content".to_string(),
        detected_faces_count: detected_speaker_frames,
        is_pip_webcam: false,
    }
}

fn nms(mut boxes: Vec<FaceBox>, iou_thresh: f32) -> Vec<FaceBox> {
    boxes.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let mut kept: Vec<FaceBox> = Vec::new();
    for b in boxes {
        let suppressed = kept.iter().any(|k| box_iou(*k, b) > iou_thresh);
        if !suppressed {
            kept.push(b);
        }
    }
    kept
}

fn box_iou(a: FaceBox, b: FaceBox) -> f32 {
    let ix1 = a.x1.max(b.x1);
    let iy1 = a.y1.max(b.y1);
    let ix2 = a.x2.min(b.x2);
    let iy2 = a.y2.min(b.y2);
    let inter = (ix2 - ix1).max(0.0) * (iy2 - iy1).max(0.0);
    let union = (a.w() * a.h() + b.w() * b.h() - inter).max(1e-6);
    inter / union
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn camera_track_stays_strictly_inside_safe_box_with_zero_border_leak() {
        // Safe webcam region in bottom-right of 1920x1080
        let safe = PixelRect {
            x: 1320.0,
            y: 680.0,
            w: 560.0,
            h: 360.0,
        };
        // Simulate face moving from far left of webcam to far right of webcam
        let samples = vec![
            (0.0, Some(FaceBox { x1: 0.70, y1: 0.65, x2: 0.76, y2: 0.80, score: 0.95 })),
            (1.0, Some(FaceBox { x1: 0.78, y1: 0.66, x2: 0.84, y2: 0.81, score: 0.95 })),
            (2.0, Some(FaceBox { x1: 0.90, y1: 0.67, x2: 0.96, y2: 0.82, score: 0.95 })),
            (3.0, None), // brief occlusion -> should hold smoothly
            (4.0, Some(FaceBox { x1: 0.88, y1: 0.66, x2: 0.94, y2: 0.81, score: 0.95 })),
        ];

        let plan = build_natural_camera_track(
            &samples,
            safe,
            1920,
            1080,
            1020.0 / 582.0,
            1.0,
            4.5,
            true,
        );

        assert!(plan.crop_w as f64 <= safe.w);
        assert!(plan.crop_h as f64 <= safe.h);
        // At default zoom=1.0 on a PiP webcam, show the FULL clean webcam width (no tight face zoom!)
        assert!(
            plan.crop_w as f64 >= safe.w - 4.0,
            "expected full clean webcam without forced tight face zoom, got crop_w={} vs safe.w={}",
            plan.crop_w,
            safe.w
        );
        assert!(!plan.keyframes.is_empty());

        for &(_, x, y) in &plan.keyframes {
            assert!(x as f64 >= safe.x.floor());
            assert!(y as f64 >= safe.y.floor());
            assert!((x as f64 + plan.crop_w as f64) <= safe.right().ceil());
            assert!((y as f64 + plan.crop_h as f64) <= safe.bottom().ceil());
        }

        let script = plan.format_sendcmd_script();
        assert!(script.contains("[enter] crop@cam x "));
        assert!(script.contains(", crop@cam y "));
    }

    #[test]
    fn anti_npc_cluster_picks_persistent_streamer_over_fleeting_game_npcs() {
        let mut faces = Vec::new();
        // Streamer consistently in bottom-right webcam across 10 frames
        for _ in 0..10 {
            faces.push(FaceBox {
                x1: 0.78,
                y1: 0.70,
                x2: 0.86,
                y2: 0.86,
                score: 0.92,
            });
        }
        // Fleeting GTA NPCs in random places
        faces.push(FaceBox { x1: 0.20, y1: 0.30, x2: 0.32, y2: 0.50, score: 0.95 });
        faces.push(FaceBox { x1: 0.45, y1: 0.25, x2: 0.55, y2: 0.45, score: 0.94 });

        let dominant = select_dominant_face_cluster(&faces).unwrap();
        assert!(dominant.cx() > 0.75 && dominant.cy() > 0.65);
    }

    #[test]
    fn content_director_tracks_action_highlight_and_ignores_excluded_webcam() {
        let mut frames = Vec::new();
        for i in 0..8usize {
            let mut img = RgbImage::from_pixel(320, 180, image::Rgb([16, 16, 18]));
            // Bright colorful moving action highlight on the LEFT side
            let shift = (i * 4) as u32;
            for y in 40..120 {
                for x in (18 + shift)..(86 + shift) {
                    img.put_pixel(x, y, image::Rgb([255, 190, 20]));
                }
            }
            // Intense flashing streamer webcam in BOTTOM-RIGHT (must be masked out!)
            let cam_val = if i % 2 == 0 { 255 } else { 0 };
            for y in 112..178 {
                for x in 212..318 {
                    img.put_pixel(x, y, image::Rgb([cam_val, 255 - cam_val, cam_val]));
                }
            }
            frames.push((i as f64 / 3.0, img));
        }

        let exclude_webcam = PixelRect {
            x: 0.65,
            y: 0.60,
            w: 0.35,
            h: 0.40,
        };
        let plan = compute_content_director_track(
            &frames,
            None,
            Some(exclude_webcam),
            1920,
            1080,
            1020.0 / 889.0,
            2.5,
        );

        assert_eq!(plan.filter_target, "crop@content");
        assert!(plan.crop_w < 1920);
        // Should lock onto the LEFT side action rather than chasing the bottom-right webcam
        assert!(
            plan.init_x < 260,
            "expected left-aligned content crop, got init_x={}",
            plan.init_x
        );
        for &(_, x, y) in &plan.keyframes {
            assert!(x >= 0 && (x as u32 + plan.crop_w) <= 1920);
            assert!(y >= 0 && (y as u32 + plan.crop_h) <= 1080);
        }
        assert!(plan.format_sendcmd_script().contains("[enter] crop@content x "));
    }

    #[test]
    fn content_director_switches_between_left_and_right_speakers_with_clean_cut() {
        let targets = vec![
            ContentTarget { t: 0.0, target_x: 40.0, target_y: 20.0, hard_cut: false, has_speaker: true },
            ContentTarget { t: 1.0, target_x: 60.0, target_y: 20.0, hard_cut: false, has_speaker: true },
            ContentTarget { t: 2.0, target_x: 700.0, target_y: 30.0, hard_cut: true, has_speaker: true },
            ContentTarget { t: 3.0, target_x: 720.0, target_y: 30.0, hard_cut: false, has_speaker: true },
        ];
        let plan = build_content_camera_track(&targets, 1920, 1080, 1166, 1016, 3.5, 4);
        assert_eq!(plan.filter_target, "crop@content");
        assert_eq!(plan.init_x, 40);
        let at_cut = plan
            .keyframes
            .iter()
            .find(|&&(t, _, _)| (t - 2.0).abs() < 0.06)
            .copied()
            .expect("expected keyframe at speaker cut");
        assert!(at_cut.1 >= 690, "expected instant cut to right speaker, got x={}", at_cut.1);
    }

    #[test]
    fn measure_face_mouth_motion_detects_active_speaker_over_silent_listener() {
        let prev = RgbImage::from_pixel(320, 180, image::Rgb([60, 60, 60]));
        let mut curr = prev.clone();
        let left_face = FaceBox { x1: 0.10, y1: 0.20, x2: 0.30, y2: 0.60, score: 0.95 };
        let right_face = FaceBox { x1: 0.70, y1: 0.20, x2: 0.90, y2: 0.60, score: 0.95 };

        // Animate mouth/jaw region of RIGHT face only
        for y in 76..104 {
            for x in 236..276 {
                curr.put_pixel(x, y, image::Rgb([220, 120, 120]));
            }
        }
        let gmad = measure_global_frame_mad(&prev, &curr);
        let left_m = measure_face_mouth_motion(&prev, &curr, left_face, gmad);
        let right_m = measure_face_mouth_motion(&prev, &curr, right_face, gmad);
        assert!(right_m > left_m + 30.0, "expected right speaker mouth motion ({right_m}) >> left ({left_m})");
    }

    #[test]
    fn zero_border_detector_strips_thick_obs_border_even_on_screen_edge() {
        // 320x180 frames: moving gameplay + bottom-right webcam touching right/bottom screen edge
        // with a THICK 12px solid gold OBS border on ALL 4 sides (x: 180..320, y: 80..180,
        // so clean interior is x: 192..308, y: 92..168).
        let mut frames = Vec::new();
        for i in 0..10usize {
            let mut img = RgbImage::from_pixel(320, 180, image::Rgb([20, 30, 50]));
            // Dynamic moving gameplay outside webcam
            for y in 0..180u32 {
                for x in 0..320u32 {
                    let v = (((x + y + (i as u32 * 29)) * 7) % 180) as u8;
                    img.put_pixel(x, y, image::Rgb([v, (v / 2).wrapping_add(40), 200u8.saturating_sub(v)]));
                }
            }
            // Thick 12px static gold OBS border around x=180..320, y=80..180
            for y in 80..180u32 {
                for x in 180..320u32 {
                    img.put_pixel(x, y, image::Rgb([245, 195, 25]));
                }
            }
            // Clean live textured webcam interior: x=192..308, y=92..168
            for y in 92..168u32 {
                for x in 192..308u32 {
                    let jitter = ((i * 5 + (x as usize % 7)) % 12) as u8;
                    let r = (55u8).saturating_add(((x - 192) / 3) as u8).saturating_add(jitter);
                    let g = (48u8).saturating_add(((y - 92) / 3) as u8).saturating_add(jitter / 2);
                    let b = (62u8).saturating_add(jitter);
                    img.put_pixel(x, y, image::Rgb([r, g, b]));
                }
            }
            frames.push((i as f64 / 3.0, img));
        }

        // Streamer face centered inside the webcam interior (x: 235..265, y: 112..148)
        let face = FaceBox {
            x1: 235.0 / 320.0,
            y1: 112.0 / 180.0,
            x2: 265.0 / 320.0,
            y2: 148.0 / 180.0,
            score: 0.96,
        };

        let det = detect_pip_webcam_zero_border(&frames, face, 0.045, false);
        assert!(det.walls_detected >= 3, "expected OBS walls detected, got {}", det.walls_detected);

        let sx1 = det.safe_rect.x * 320.0;
        let sy1 = det.safe_rect.y * 180.0;
        let sx2 = det.safe_rect.right() * 320.0;
        let sy2 = det.safe_rect.bottom() * 180.0;

        // Every single side MUST sit strictly INSIDE the 12px gold border (zero gold pixels!)
        assert!(sx1 >= 192.0, "left border leaked! sx1={sx1:.1} < 192");
        assert!(sy1 >= 92.0, "top border leaked! sy1={sy1:.1} < 92");
        assert!(sx2 <= 308.0, "right screen-edge border leaked! sx2={sx2:.1} > 308");
        assert!(sy2 <= 168.0, "bottom screen-edge border leaked! sy2={sy2:.1} > 168");
    }

    #[test]
    fn zero_border_detector_peels_static_bottom_nameplate_and_rounded_corners() {
        // Bottom-left webcam at x=0..140, y=75..180 with:
        // - 14px static dark nameplate bar along the bottom (y=166..180)
        // - 8px rounded corners exposing fast-moving gameplay
        let mut frames = Vec::new();
        for i in 0..10usize {
            let mut img = RgbImage::from_pixel(320, 180, image::Rgb([10, 10, 10]));
            // Fast-moving gameplay everywhere
            for y in 0..180u32 {
                for x in 0..320u32 {
                    let v = (((x * 3 + y * 5 + i as u32 * 47) % 240)) as u8;
                    img.put_pixel(x, y, image::Rgb([v, 240 - v, (v / 2) + 50]));
                }
            }
            // Live textured webcam interior: x=0..136, y=79..166
            for y in 79..166u32 {
                for x in 0..136u32 {
                    // Cut out 8px rounded corners so moving gameplay shows through corners
                    let near_tl = x + (y - 79) < 9;
                    let near_tr = (135 - x) + (y - 79) < 9;
                    let near_bl = x + (165 - y) < 9;
                    let near_br = (135 - x) + (165 - y) < 9;
                    if near_tl || near_tr || near_bl || near_br {
                        continue;
                    }
                    let jitter = ((i * 4 + (x as usize % 5)) % 10) as u8;
                    img.put_pixel(
                        x,
                        y,
                        image::Rgb([
                            65u8.saturating_add((x / 4) as u8).saturating_add(jitter),
                            55u8.saturating_add(((y - 79) / 3) as u8),
                            70u8.saturating_add(jitter),
                        ]),
                    );
                }
            }
            // 14px static solid purple nameplate bar at bottom y=166..180
            for y in 166..180u32 {
                for x in 0..136u32 {
                    img.put_pixel(x, y, image::Rgb([110, 25, 165]));
                }
            }
            frames.push((i as f64 / 3.0, img));
        }

        let face = FaceBox {
            x1: 52.0 / 320.0,
            y1: 100.0 / 180.0,
            x2: 84.0 / 320.0,
            y2: 138.0 / 180.0,
            score: 0.95,
        };

        let det = detect_pip_webcam_zero_border(&frames, face, 0.045, true);
        let sy2 = det.safe_rect.bottom() * 180.0;
        let sx2 = det.safe_rect.right() * 320.0;
        assert!(sy2 <= 166.0, "bottom nameplate leaked! sy2={sy2:.1} > 166");
        assert!(sx2 <= 135.0, "right edge leaked! sx2={sx2:.1} > 135");
    }

    #[test]
    fn zero_border_detector_does_not_false_trigger_on_fullscreen_talking_head() {
        // Smoothly lit natural full-screen room with NO PiP rectangular frame
        let mut frames = Vec::new();
        for i in 0..8usize {
            let mut img = RgbImage::from_pixel(320, 180, image::Rgb([70, 65, 60]));
            for y in 0..180u32 {
                for x in 0..320u32 {
                    let jitter = ((i * 3) % 6) as u8;
                    let r = 60u8.saturating_add((x / 5) as u8).saturating_add(jitter);
                    let g = 55u8.saturating_add((y / 4) as u8);
                    let b = 65u8.saturating_add(jitter);
                    img.put_pixel(x, y, image::Rgb([r, g, b]));
                }
            }
            frames.push((i as f64 / 3.0, img));
        }

        let face = FaceBox {
            x1: 0.38,
            y1: 0.20,
            x2: 0.62,
            y2: 0.65,
            score: 0.97,
        };
        let det = detect_pip_webcam_zero_border(&frames, face, 0.045, false);
        assert!(det.walls_detected < 2, "expected fullscreen camera, got {} walls", det.walls_detected);
        assert!(det.safe_rect.w > 0.95 && det.safe_rect.h > 0.95);
    }
}
