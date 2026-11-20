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
    pub detected_faces_count: usize,
    pub is_pip_webcam: bool,
}

impl CameraTrackPlan {
    /// Writes FFmpeg `sendcmd` script driving `crop@cam` smoothly over the entire clip.
    pub fn format_sendcmd_script(&self) -> String {
        let mut out = String::new();
        for &(t, x, y) in &self.keyframes {
            out.push_str(&format!("{t:.3} [enter] crop@cam x {x}, crop@cam y {y};\n"));
        }
        out
    }

    pub fn write_sendcmd_file(&self, path: &Path) -> AicutResult<()> {
        std::fs::write(path, self.format_sendcmd_script())?;
        Ok(())
    }
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
    if src_w < 64 || src_h < 64 || segments.is_empty() {
        return Ok(None);
    }

    let model_path = match ensure_face_detector_model(kernel).await {
        Ok(p) => p,
        Err(_) => return Ok(None),
    };

    let tmp_dir = std::env::temp_dir().join(format!("aicut-facetrack-{}", uuid::Uuid::new_v4()));
    std::fs::create_dir_all(&tmp_dir)?;

    let sample_fps = 3.0f64;
    let extract_res =
        extract_clip_analysis_frames(ffmpeg_bin, video_path, segments, sample_fps, &tmp_dir).await;
    let frames = match extract_res {
        Ok(f) if !f.is_empty() => f,
        _ => {
            let _ = std::fs::remove_dir_all(&tmp_dir);
            return Ok(None);
        }
    };

    let total_clip_dur: f64 = segments
        .iter()
        .map(|s| (s.end_sec - s.start_sec).max(0.0))
        .sum();
    let target_aspect = target_w as f64 / target_h.max(1) as f64;
    let zoom = face_zoom.clamp(0.75, 1.50) as f64;
    let trim = border_trim_pct.clamp(0.01, 0.12) as f64;

    // Run CPU-bound ONNX inference + edge detection on blocking threadpool
    let plan = tokio::task::spawn_blocking(move || -> AicutResult<Option<CameraTrackPlan>> {
        let detector = FaceDetector::load(&model_path)?;
        let loaded: Vec<(f64, RgbImage)> = frames
            .iter()
            .filter_map(|(t, p)| image::open(p).ok().map(|img| (*t, img.to_rgb8())))
            .collect();

        if loaded.is_empty() {
            return Ok(None);
        }

        // ---- PASS 1: Detect faces across clip & identify dominant human speaker (anti-NPC) ----
        let stride = (loaded.len() / 45).max(1);
        let mut pass1_faces: Vec<FaceBox> = Vec::new();
        for (idx, (_, rgb)) in loaded.iter().enumerate() {
            if idx % stride != 0 {
                continue;
            }
            if let Ok(found) = detector.detect(rgb, 0.62) {
                for f in found {
                    if matches_corner_hint(f, corner_hint) {
                        pass1_faces.push(f);
                    }
                }
            }
        }

        let Some(anchor_face) = select_dominant_face_cluster(&pass1_faces) else {
            return Ok(None);
        };

        // Decide if this is a corner PiP webcam or a full-screen face camera
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

        // ---- Detect exact 4 webcam walls via temporal median & apply inner zero-border trim ----
        let safe_box_norm = if is_pip_webcam {
            detect_pip_webcam_safe_box(&loaded, anchor_face, trim)
        } else {
            PixelRect {
                x: 0.005,
                y: 0.005,
                w: 0.99,
                h: 0.99,
            }
        };

        let safe_box_px = PixelRect {
            x: safe_box_norm.x * src_w as f64,
            y: safe_box_norm.y * src_h as f64,
            w: safe_box_norm.w * src_w as f64,
            h: safe_box_norm.h * src_h as f64,
        };

        // ---- PASS 2: High-precision zoomed face tracking INSIDE clean safe_box on EVERY frame ----
        let mut tracked_samples: Vec<(f64, Option<FaceBox>)> = Vec::with_capacity(loaded.len());
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
            let picked = detector
                .detect(&cropped, 0.58)
                .ok()
                .and_then(|faces| {
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

        if detected_faces_count == 0 {
            return Ok(None);
        }

        let plan = build_natural_camera_track(
            &tracked_samples,
            safe_box_px,
            src_w,
            src_h,
            target_aspect,
            zoom,
            total_clip_dur,
            is_pip_webcam,
        );
        Ok(Some(plan))
    })
    .await
    .map_err(|e| AicutError::internal(format!("Face tracking task failed: {e}")))??;

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
/// of an OBS PiP webcam surrounding `anchor_face`, then shrinks inward by `trim_pct` so
/// 0 pixels of OBS border or gameplay ever leak into the top Short window.
fn detect_pip_webcam_safe_box(
    frames: &[(f64, RgbImage)],
    anchor_face: FaceBox,
    trim_pct: f64,
) -> PixelRect {
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

    PixelRect {
        x: sx1 / w as f64,
        y: sy1 / h as f64,
        w: (sx2 - sx1) / w as f64,
        h: (sy2 - sy1) / h as f64,
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
        detected_faces_count,
        is_pip_webcam,
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
}
