use aicut_core::{AicutError, AicutKernel, AicutResult};
use std::io::Cursor;
use std::path::{Path, PathBuf};

/// Built-in default 9:16 RGBA Shorts template (camera on top, content on bottom, transparent rounded slots).
pub const DEFAULT_TEMPLATE_PNG: &[u8] = include_bytes!("../assets/smoking33-template.png");

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TemplateSlot {
    pub x: u32,
    pub y: u32,
    pub w: u32,
    pub h: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TemplateGeometry {
    pub top_camera: TemplateSlot,
    pub bottom_content: TemplateSlot,
}

impl Default for TemplateGeometry {
    fn default() -> Self {
        // Exact measured slots for smoking33-template.png scaled from 576x1024 -> 1080x1920 (+4px bleed behind gold borders)
        Self {
            top_camera: TemplateSlot {
                x: 30,
                y: 150,
                w: 1020,
                h: 582,
            },
            bottom_content: TemplateSlot {
                x: 30,
                y: 765,
                w: 1020,
                h: 889,
            },
        }
    }
}

/// Ensures the active PNG template file exists on disk (either user's custom PNG path or the built-in template).
pub fn resolve_template_png(
    kernel: &AicutKernel,
    custom_path: Option<&str>,
) -> AicutResult<PathBuf> {
    if let Some(p) = custom_path.map(str::trim).filter(|s| !s.is_empty()) {
        let path = PathBuf::from(p);
        if path.is_file() {
            return Ok(path);
        }
        return Err(AicutError::not_found(format!(
            "Moldura PNG personalizada nao encontrada: {p}"
        )));
    }

    let tpl_dir = kernel.workspace.thumbnails_dir.join("templates");
    std::fs::create_dir_all(&tpl_dir)?;
    let default_path = tpl_dir.join("default-shorts-template.png");
    if !default_path.is_file()
        || std::fs::metadata(&default_path)
            .map(|m| m.len() != DEFAULT_TEMPLATE_PNG.len() as u64)
            .unwrap_or(true)
    {
        std::fs::write(&default_path, DEFAULT_TEMPLATE_PNG)?;
    }
    Ok(default_path)
}

/// Automatically detects the top (camera) and bottom (content) transparent cutout slots in ANY
/// RGBA PNG template and scales them to 1080x1920 with a 4px bleed behind the rounded borders.
pub fn detect_template_geometry_from_path(png_path: &Path) -> TemplateGeometry {
    std::fs::read(png_path)
        .ok()
        .and_then(|bytes| detect_template_geometry_from_bytes(&bytes))
        .unwrap_or_default()
}

pub fn detect_template_geometry_from_bytes(bytes: &[u8]) -> Option<TemplateGeometry> {
    let mut decoder = png::Decoder::new(Cursor::new(bytes));
    decoder.set_transformations(png::Transformations::EXPAND | png::Transformations::STRIP_16);
    let mut reader = decoder.read_info().ok()?;
    let mut buf = vec![0u8; reader.output_buffer_size()];
    let info = reader.next_frame(&mut buf).ok()?;
    let pixels = &buf[..info.buffer_size()];

    let w = info.width as usize;
    let h = info.height as usize;
    if w < 64 || h < 64 {
        return None;
    }

    let bpp = match info.color_type {
        png::ColorType::Rgba => 4,
        png::ColorType::Rgb => 3,
        png::ColorType::GrayscaleAlpha => 2,
        png::ColorType::Grayscale => 1,
        _ => return None,
    };

    let is_cutout = |x: usize, y: usize| -> bool {
        let idx = (y * w + x) * bpp;
        match info.color_type {
            png::ColorType::Rgba => pixels[idx + 3] < 32,
            png::ColorType::GrayscaleAlpha => pixels[idx + 1] < 32,
            png::ColorType::Rgb => {
                pixels[idx] > 248 && pixels[idx + 1] > 248 && pixels[idx + 2] > 248
            }
            _ => false,
        }
    };

    // Scan center vertical column for continuous cutout bands
    let mid_x = w / 2;
    let mut y_bands: Vec<(usize, usize)> = Vec::new();
    let mut in_band = false;
    let mut band_start = 0usize;

    for y in 0..h {
        if is_cutout(mid_x, y) {
            if !in_band {
                in_band = true;
                band_start = y;
            }
        } else if in_band {
            in_band = false;
            if y - band_start >= h / 12 {
                y_bands.push((band_start, y - 1));
            }
        }
    }
    if in_band && h - band_start >= h / 12 {
        y_bands.push((band_start, h - 1));
    }

    if y_bands.len() < 2 {
        return None;
    }

    // Keep the two largest bands, sorted top-to-bottom
    y_bands.sort_by(|a, b| (b.1 - b.0).cmp(&(a.1 - a.0)));
    y_bands.truncate(2);
    y_bands.sort_by_key(|b| b.0);

    let sx = 1080.0 / w as f64;
    let sy = 1920.0 / h as f64;

    let build_slot = |(y0, y1): (usize, usize)| -> TemplateSlot {
        let probe_y = (y0 + y1) / 2;
        let mut x0 = 0usize;
        let mut x1 = w - 1;
        let mut found_x = false;
        for x in 0..w {
            if is_cutout(x, probe_y) {
                if !found_x {
                    x0 = x;
                    found_x = true;
                }
                x1 = x;
            }
        }

        let bleed = 4i32;
        let rx = ((x0 as f64 * sx).round() as i32 - bleed).clamp(0, 1076) as u32;
        let ry = ((y0 as f64 * sy).round() as i32 - bleed).clamp(0, 1916) as u32;
        let rx_end = (((x1 + 1) as f64 * sx).round() as i32 + bleed).clamp((rx + 4) as i32, 1080) as u32;
        let ry_end = (((y1 + 1) as f64 * sy).round() as i32 + bleed).clamp((ry + 4) as i32, 1920) as u32;

        // Keep width/height even for yuv420p codecs
        let rw = ((rx_end - rx) & !1).max(16);
        let rh = ((ry_end - ry) & !1).max(16);
        TemplateSlot {
            x: rx,
            y: ry,
            w: rw,
            h: rh,
        }
    };

    Some(TemplateGeometry {
        top_camera: build_slot(y_bands[0]),
        bottom_content: build_slot(y_bands[1]),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detects_smoking33_template_transparent_slots_accurately() {
        let geom = detect_template_geometry_from_bytes(DEFAULT_TEMPLATE_PNG)
            .expect("should detect two transparent boxes in built-in template");
        // Top camera box should be roughly x=30, y=150, w=1020, h=580
        assert!(geom.top_camera.x >= 20 && geom.top_camera.x <= 45);
        assert!(geom.top_camera.y >= 140 && geom.top_camera.y <= 165);
        assert!(geom.top_camera.w >= 1000 && geom.top_camera.w <= 1040);
        assert!(geom.top_camera.h >= 560 && geom.top_camera.h <= 600);

        // Bottom content box should sit below top camera box around y=765, h=888
        assert!(geom.bottom_content.y > geom.top_camera.y + geom.top_camera.h);
        assert!(geom.bottom_content.y >= 750 && geom.bottom_content.y <= 785);
        assert!(geom.bottom_content.h >= 870 && geom.bottom_content.h <= 910);
    }
}
