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

/// Ensures the tiny (~1.2 MB) UltraFace ONNX model exists locally, downloading it once automatically.
pub async fn ensure_face_detector_model(kernel: &AicutKernel) -> AicutResult<PathBuf> {
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
                        let tmp = dest.with_extension("onnx.part");
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
        Some((target_w, target_h, corner_hint, face_zoom, border_trim_pct)),
        None,
    )
    .await?;
    Ok(dual.camera)
}

/// Runs unified single-pass AI tracking for BOTH Short windows in one frame extraction:
/// - `camera_slot`: `(target_w, target_h, corner_hint, face_zoom, border_trim_pct)` for the top webcam
/// - `content_slot`: `(target_w, target_h)` for the bottom content window (active speaker or peak action)
#[allow(clippy::too_many_arguments)]
pub async fn compute_dual_tracks(
    kernel: &AicutKernel,
    ffmpeg_bin: &Path,
    video_path: &Path,
    src_w: u32,
    src_h: u32,
    segments: &[SmartClipSegment],
    camera_slot: Option<(u32, u32, WebcamCorner, f32, f32)>,
    content_slot: Option<(u32, u32)>,
) -> AicutResult<DualTrackPlan> {
    if (camera_slot.is_none() && content_slot.is_none())
        || src_w < 64
        || src_h < 64
        || segments.is_empty()
    {
        return Ok(DualTrackPlan::default());
    }

    // Download UltraFace (~1.2 MB) once if missing. If offline/unreachable, content action tracking
    // can still proceed using pure visual motion + saliency without failing the render.
    let model_path = ensure_face_detector_model(kernel).await.ok();
    if model_path.is_none() && content_slot.is_none() {
        return Ok(DualTrackPlan::default());
    }

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

        // ---- PART 1: Top Webcam Cameraman (Zero-Border + Smooth Face Tracking) ----
        if let (Some((cam_w, cam_h, corner_hint, face_zoom, border_trim_pct)), Some(det)) =
            (camera_slot, detector.as_ref())
        {
            let target_aspect = cam_w as f64 / cam_h.max(1) as f64;
            let zoom = face_zoom.clamp(0.75, 1.50) as f64;
            let trim = border_trim_pct.clamp(0.01, 0.12) as f64;

            // PASS 1: Detect faces across clip & identify dominant human speaker (anti-NPC)
            let stride = (loaded.len() / 45).max(1);
            let mut pass1_faces: Vec<FaceBox> = Vec::new();
            for (idx, (_, rgb)) in loaded.iter().enumerate() {
                if idx % stride != 0 {
                    continue;
                }
                if let Ok(found) = det.detect(rgb, 0.62) {
                    for f in found {
                        if matches_corner_hint(f, corner_hint) {
                            pass1_faces.push(f);
                        }
                    }
                }
            }

            if let Some(anchor_face) = select_dominant_face_cluster(&pass1_faces) {
                let is_pip_webcam = match corner_hint {
                    WebcamCorner::BottomRight
                    | WebcamCorner::BottomLeft
                    | WebcamCorner::TopRight
                    | WebcamCorner::TopLeft => true,
                    WebcamCorner::Center => false,
                    WebcamCorner::AutoFace => {
                        let near_edge = anchor_face.cx() < 0.34
                            || anchor_face.cx() > 0.66
                            || anchor_face.cy() < 0.30
                            || anchor_face.cy() > 0.70;
                        anchor_face.w() < 0.18 && near_edge
                    }
                };

                let (safe_box_norm, outer_box_norm) = if is_pip_webcam {
                    detect_pip_webcam_safe_box(&loaded, anchor_face, trim)
                } else {
                    let full = PixelRect {
                        x: 0.005,
                        y: 0.005,
                        w: 0.99,
                        h: 0.99,
                    };
                    (full, full)
                };

                if is_pip_webcam {
                    // Mask out the streamer's corner webcam + OBS border from bottom content analysis
                    webcam_exclude_norm = Some(expand_norm_rect(outer_box_norm, 0.015));
                }

                let safe_box_px = PixelRect {
                    x: safe_box_norm.x * src_w as f64,
                    y: safe_box_norm.y * src_h as f64,
                    w: safe_box_norm.w * src_w as f64,
                    h: safe_box_norm.h * src_h as f64,
                };

                // PASS 2: High-precision zoomed face tracking INSIDE clean safe_box on EVERY frame
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

        // ---- PART 2: Bottom Content Director (Active Speaker + Visual Action / Highlight) ----
        let content_plan = content_slot.map(|(cnt_w, cnt_h)| {
            let content_aspect = cnt_w as f64 / cnt_h.max(1) as f64;
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
fn select_dominant_face_cluster(faces: &[FaceBox]) -> Option<FaceBox> {
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

/// Uses a temporal median image across the clip to locate the static 4 rectangular borders
/// of an OBS PiP webcam surrounding `anchor_face`.
/// Returns `(safe_box_norm, outer_box_norm)`:
/// - `safe_box_norm` shrinks inward by `trim_pct` so 0 pixels of OBS border leak into the top Short window.
/// - `outer_box_norm` is the full outer OBS webcam box so the bottom content director can mask it out.
fn detect_pip_webcam_safe_box(
    frames: &[(f64, RgbImage)],
    anchor_face: FaceBox,
    trim_pct: f64,
) -> (PixelRect, PixelRect) {
    let sample_count = frames.len().min(12).max(1);
    let step = (frames.len() / sample_count).max(1);
    let subset: Vec<&RgbImage> = frames
        .iter()
        .step_by(step)
        .take(sample_count)
        .map(|(_, img)| img)
        .collect();

    let w = subset[0].width() as usize;
    let h = subset[0].height() as usize;

    // Build 3-channel RGB temporal median image: moving gameplay blurs out while static
    // OBS webcam borders (including colored overlays with similar brightness) stay razor-sharp.
    let mut median_rgb = vec![[0u8; 3]; w * h];
    let mut r_buf = Vec::with_capacity(subset.len());
    let mut g_buf = Vec::with_capacity(subset.len());
    let mut b_buf = Vec::with_capacity(subset.len());
    let mid = subset.len() / 2;

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
            median_rgb[y * w + x] = [r_buf[mid], g_buf[mid], b_buf[mid]];
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

    // Realistic search corridor around the face for a PiP webcam box
    let min_left = (fx1 - fw * 3).max(0) as usize;
    let max_left = (fx1 - (fw / 3).max(4)).max(0) as usize;
    let min_right = (fx2 + (fw / 3).max(4)).min(w as isize - 1) as usize;
    let max_right = (fx2 + fw * 3).min(w as isize - 1) as usize;

    let min_top = (fy1 - fh * 2).max(0) as usize;
    let max_top = (fy1 - (fh / 3).max(4)).max(0) as usize;
    let min_bot = (fy2 + (fh / 2).max(4)).min(h as isize - 1) as usize;
    let max_bot = (fy2 + fh * 3).min(h as isize - 1) as usize;

    let y_band0 = (fcy - fh).clamp(1, h as isize - 2) as usize;
    let y_band1 = (fcy + fh).clamp(y_band0 as isize + 1, h as isize - 2) as usize;
    let x_band0 = (fcx - fw).clamp(1, w as isize - 2) as usize;
    let x_band1 = (fcx + fw).clamp(x_band0 as isize + 1, w as isize - 2) as usize;

    // Scan vertical RGB edge strength along x for left & right walls
    let scan_v_edge = |x_start: usize, x_end: usize, fallback: usize| -> usize {
        if x_end <= x_start + 2 {
            return fallback;
        }
        let mut best_x = fallback;
        let mut best_score = 0.0f64;
        for x in x_start.max(2)..x_end.min(w - 2) {
            let mut sum = 0.0f64;
            for y in y_band0..=y_band1 {
                sum += rgb_diff(median_rgb[y * w + (x - 1)], median_rgb[y * w + (x + 1)]);
            }
            let mean = sum / (y_band1 - y_band0 + 1) as f64;
            if mean > best_score {
                best_score = mean;
                best_x = x;
            }
        }
        if best_score >= 10.0 { best_x } else { fallback }
    };

    // Scan horizontal RGB edge strength along y for top & bottom walls
    let scan_h_edge = |y_start: usize, y_end: usize, fallback: usize| -> usize {
        if y_end <= y_start + 2 {
            return fallback;
        }
        let mut best_y = fallback;
        let mut best_score = 0.0f64;
        for y in y_start.max(2)..y_end.min(h - 2) {
            let mut sum = 0.0f64;
            for x in x_band0..=x_band1 {
                sum += rgb_diff(median_rgb[(y - 1) * w + x], median_rgb[(y + 1) * w + x]);
            }
            let mean = sum / (x_band1 - x_band0 + 1) as f64;
            if mean > best_score {
                best_score = mean;
                best_y = y;
            }
        }
        if best_score >= 10.0 { best_y } else { fallback }
    };

    let def_left = (fx1 - (fw as f64 * 1.45) as isize).max(0) as usize;
    let def_right = (fx2 + (fw as f64 * 1.45) as isize).min(w as isize - 1) as usize;
    let def_top = (fy1 - (fh as f64 * 0.75) as isize).max(0) as usize;
    let def_bot = (fy2 + (fh as f64 * 1.35) as isize).min(h as isize - 1) as usize;

    let mut left = if def_left <= 8 {
        0
    } else {
        scan_v_edge(min_left, max_left, def_left)
    };
    let mut right = if def_right >= w.saturating_sub(9) {
        w - 1
    } else {
        scan_v_edge(min_right, max_right, def_right)
    };
    let mut top = if def_top <= 8 {
        0
    } else {
        scan_h_edge(min_top, max_top, def_top)
    };
    let mut bot = if def_bot >= h.saturating_sub(9) {
        h - 1
    } else {
        scan_h_edge(min_bot, max_bot, def_bot)
    };

    // Ensure detected box comfortably encloses the anchor face
    left = left.min(fx1.max(0) as usize);
    right = right.max(fx2.min(w as isize - 1) as usize);
    top = top.min(fy1.max(0) as usize);
    bot = bot.max(fy2.min(h as isize - 1) as usize);

    let box_w = (right.saturating_sub(left)).max(32) as f64;
    let box_h = (bot.saturating_sub(top)).max(32) as f64;

    // Apply inward border trim (stronger on interior walls where OBS borders live)
    let trim_l = if left <= 4 { box_w * 0.015 } else { box_w * trim_pct };
    let trim_r = if right >= w - 5 { box_w * 0.015 } else { box_w * trim_pct };
    let trim_t = if top <= 4 { box_h * 0.015 } else { box_h * trim_pct };
    let trim_b = if bot >= h - 5 { box_h * 0.015 } else { box_h * trim_pct };

    let sx1 = (left as f64 + trim_l).clamp(0.0, w as f64 - 16.0);
    let sy1 = (top as f64 + trim_t).clamp(0.0, h as f64 - 16.0);
    let sx2 = (right as f64 - trim_r).clamp(sx1 + 16.0, w as f64);
    let sy2 = (bot as f64 - trim_b).clamp(sy1 + 16.0, h as f64);

    let safe_rect = PixelRect {
        x: sx1 / w as f64,
        y: sy1 / h as f64,
        w: (sx2 - sx1) / w as f64,
        h: (sy2 - sy1) / h as f64,
    };
    let outer_rect = PixelRect {
        x: left as f64 / w as f64,
        y: top as f64 / h as f64,
        w: (right.saturating_sub(left)).max(16) as f64 / w as f64,
        h: (bot.saturating_sub(top)).max(16) as f64 / h as f64,
    };
    (safe_rect, outer_rect)
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

    // 2. Compute maximum aspect-locked crop that fits 100% inside safe_box_px
    let (max_cw, max_ch) = if safe_box_px.w / safe_box_px.h.max(1.0) >= target_aspect {
        let ch = safe_box_px.h;
        (ch * target_aspect, ch)
    } else {
        let cw = safe_box_px.w;
        (cw, cw / target_aspect)
    };

    // Ideal crop height puts the face at ~38% of frame height (scaled by face_zoom),
    // leaving natural pan travel inside `safe_box_px` while never zooming in overly tight.
    let desired_ch = (median_fh / (0.38 * face_zoom)).clamp(
        max_ch * if is_pip_webcam { 0.62 } else { 0.35 },
        max_ch * if is_pip_webcam { 0.92 } else { 0.88 },
    );
    let desired_cw = (desired_ch * target_aspect).min(max_cw);
    let desired_ch = desired_cw / target_aspect;

    // Keep even integer dimensions for yuv420p compatibility
    let crop_w = ((desired_cw.floor() as u32) & !1).clamp(32, src_w & !1);
    let crop_h = ((desired_ch.floor() as u32) & !1).clamp(32, src_h & !1);

    let min_x = safe_box_px.x.ceil().clamp(0.0, (src_w - crop_w) as f64);
    let max_x = (safe_box_px.right() - crop_w as f64)
        .floor()
        .clamp(min_x, (src_w - crop_w) as f64);
    let min_y = safe_box_px.y.ceil().clamp(0.0, (src_h - crop_h) as f64);
    let max_y = (safe_box_px.bottom() - crop_h as f64)
        .floor()
        .clamp(min_y, (src_h - crop_h) as f64);

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
    let (crop_w, crop_h) = compute_content_crop_dims(src_w, src_h, content_aspect);
    let max_x = src_w.saturating_sub(crop_w) as f64;
    let max_y = src_h.saturating_sub(crop_h) as f64;
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
                        !too_small && !inside_cam
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

            let tx = (focus_cx * src_w as f64 - crop_w as f64 * 0.5).clamp(0.0, max_x);
            let ty = (focus_eye_y * src_h as f64 - crop_h as f64 * 0.36).clamp(0.0, max_y);
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
                target_x: action_tx.clamp(0.0, max_x),
                target_y: action_ty.clamp(0.0, max_y),
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
        assert!(plan.keyframes.len() >= 2);

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
}
