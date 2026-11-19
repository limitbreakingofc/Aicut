use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use std::process::Command;
use sysinfo::System;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum GpuVendor {
    Nvidia,
    Amd,
    Intel,
    Apple,
    Unknown,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum AccelBackend {
    CudaNvenc,
    AmfDirectml,
    QsvOpenvino,
    Metal,
    CpuOnly,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct GpuInfo {
    pub name: String,
    pub vendor: GpuVendor,
    pub vram_mb: Option<u64>,
    pub driver_version: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct HardwareProfile {
    pub os: String,
    pub os_version: String,
    pub arch: String,
    pub platform_tag: String,
    pub cpu_brand: String,
    pub physical_cores: usize,
    pub logical_cores: usize,
    pub total_memory_mb: u64,
    pub available_memory_mb: u64,
    pub gpus: Vec<GpuInfo>,
    pub recommended_accel: AccelBackend,
    pub recommendation_summary: String,
}

impl HardwareProfile {
    /// Detects the host OS, CPU architecture, memory, and GPUs to select
    /// the optimal builds for yt-dlp, FFmpeg, and future AI modules.
    pub fn detect() -> Self {
        let mut sys = System::new_all();
        sys.refresh_all();

        let os = std::env::consts::OS.to_string();
        let arch = std::env::consts::ARCH.to_string();
        let os_version = System::long_os_version()
            .or_else(System::os_version)
            .unwrap_or_else(|| os.clone());

        let platform_tag = match (os.as_str(), arch.as_str()) {
            ("windows", "x86_64") => "win-x64",
            ("windows", "aarch64") => "win-arm64",
            ("linux", "x86_64") => "linux-x64",
            ("linux", "aarch64") => "linux-arm64",
            ("macos", "aarch64") => "macos-arm64",
            ("macos", "x86_64") => "macos-x64",
            _ => "unknown",
        }
        .to_string();

        let cpu_brand = sys
            .cpus()
            .first()
            .map(|c| c.brand().trim().to_string())
            .filter(|s| !s.is_empty())
            .unwrap_or_else(|| "Unknown CPU".to_string());

        let physical_cores = sys.physical_core_count().unwrap_or(1);
        let logical_cores = sys.cpus().len().max(1);
        let total_memory_mb = sys.total_memory() / (1024 * 1024);
        let available_memory_mb = sys.available_memory() / (1024 * 1024);

        let gpus = detect_gpus(&os);
        let recommended_accel = pick_recommended_accel(&os, &gpus);
        let recommendation_summary = build_summary(&platform_tag, recommended_accel, &gpus);

        Self {
            os,
            os_version,
            arch,
            platform_tag,
            cpu_brand,
            physical_cores,
            logical_cores,
            total_memory_mb,
            available_memory_mb,
            gpus,
            recommended_accel,
            recommendation_summary,
        }
    }
}

fn classify_vendor(name: &str) -> GpuVendor {
    let lower = name.to_lowercase();
    if lower.contains("nvidia") || lower.contains("geforce") || lower.contains("rtx") || lower.contains("gtx") || lower.contains("quadro") {
        GpuVendor::Nvidia
    } else if lower.contains("amd") || lower.contains("radeon") || lower.contains("rx ") {
        GpuVendor::Amd
    } else if lower.contains("intel") || lower.contains("arc") || lower.contains("iris") || lower.contains("uhd") {
        GpuVendor::Intel
    } else if lower.contains("apple") {
        GpuVendor::Apple
    } else {
        GpuVendor::Unknown
    }
}

fn detect_gpus(os: &str) -> Vec<GpuInfo> {
    let mut gpus = Vec::new();

    // 1. Try nvidia-smi (cross-platform: Windows & Linux) for accurate NVIDIA VRAM & driver info.
    if let Ok(output) = Command::new("nvidia-smi")
        .args([
            "--query-gpu=name,memory.total,driver_version",
            "--format=csv,noheader,nounits",
        ])
        .output()
    {
        if output.status.success() {
            let text = String::from_utf8_lossy(&output.stdout);
            for line in text.lines() {
                let parts: Vec<&str> = line.split(',').map(|s| s.trim()).collect();
                if let Some(name) = parts.first().filter(|s| !s.is_empty()) {
                    let vram_mb = parts.get(1).and_then(|s| s.parse::<u64>().ok());
                    let driver_version = parts.get(2).map(|s| s.to_string());
                    gpus.push(GpuInfo {
                        name: (*name).to_string(),
                        vendor: GpuVendor::Nvidia,
                        vram_mb,
                        driver_version,
                    });
                }
            }
        }
    }

    // 2. Windows: query Win32_VideoController via PowerShell CIM (covers AMD, Intel, and NVIDIA fallback).
    if os == "windows" {
        if let Ok(output) = Command::new("powershell")
            .args([
                "-NoProfile",
                "-Command",
                "Get-CimInstance Win32_VideoController | Select-Object Name,AdapterRAM,DriverVersion | ConvertTo-Json -Compress",
            ])
            .output()
        {
            if output.status.success() {
                let raw = String::from_utf8_lossy(&output.stdout);
                if let Ok(val) = serde_json::from_str::<serde_json::Value>(raw.trim()) {
                    let items = match val {
                        serde_json::Value::Array(arr) => arr,
                        obj @ serde_json::Value::Object(_) => vec![obj],
                        _ => vec![],
                    };
                    for item in items {
                        let Some(name) = item.get("Name").and_then(|v| v.as_str()) else {
                            continue;
                        };
                        let name_trimmed = name.trim();
                        if name_trimmed.is_empty()
                            || gpus
                                .iter()
                                .any(|g| g.name.eq_ignore_ascii_case(name_trimmed))
                        {
                            continue;
                        }
                        let vram_mb = item
                            .get("AdapterRAM")
                            .and_then(|v| v.as_u64())
                            .map(|b| b / (1024 * 1024))
                            .filter(|&mb| mb > 0);
                        let driver_version = item
                            .get("DriverVersion")
                            .and_then(|v| v.as_str())
                            .map(|s| s.to_string());
                        gpus.push(GpuInfo {
                            name: name_trimmed.to_string(),
                            vendor: classify_vendor(name_trimmed),
                            vram_mb,
                            driver_version,
                        });
                    }
                }
            }
        }
    } else if os == "linux" && gpus.is_empty() {
        // 3. Linux fallback: inspect /sys/class/drm vendor IDs if available.
        if let Ok(entries) = std::fs::read_dir("/sys/class/drm") {
            for entry in entries.flatten() {
                let vendor_path = entry.path().join("device/vendor");
                if let Ok(vendor_hex) = std::fs::read_to_string(vendor_path) {
                    let (vendor, label) = match vendor_hex.trim() {
                        "0x10de" => (GpuVendor::Nvidia, "NVIDIA GPU"),
                        "0x1002" => (GpuVendor::Amd, "AMD Radeon GPU"),
                        "0x8086" => (GpuVendor::Intel, "Intel Integrated/Arc GPU"),
                        _ => continue,
                    };
                    if !gpus.iter().any(|g| g.vendor == vendor) {
                        gpus.push(GpuInfo {
                            name: label.to_string(),
                            vendor,
                            vram_mb: None,
                            driver_version: None,
                        });
                    }
                }
            }
        }
    } else if os == "macos" && std::env::consts::ARCH == "aarch64" && gpus.is_empty() {
        gpus.push(GpuInfo {
            name: "Apple Silicon GPU".to_string(),
            vendor: GpuVendor::Apple,
            vram_mb: None,
            driver_version: None,
        });
    }

    gpus
}

fn pick_recommended_accel(os: &str, gpus: &[GpuInfo]) -> AccelBackend {
    if gpus.iter().any(|g| g.vendor == GpuVendor::Nvidia) {
        AccelBackend::CudaNvenc
    } else if gpus.iter().any(|g| g.vendor == GpuVendor::Amd) {
        AccelBackend::AmfDirectml
    } else if gpus.iter().any(|g| g.vendor == GpuVendor::Intel) {
        AccelBackend::QsvOpenvino
    } else if os == "macos" && gpus.iter().any(|g| g.vendor == GpuVendor::Apple) {
        AccelBackend::Metal
    } else {
        AccelBackend::CpuOnly
    }
}

fn build_summary(platform_tag: &str, accel: AccelBackend, gpus: &[GpuInfo]) -> String {
    let gpu_label = gpus
        .first()
        .map(|g| g.name.as_str())
        .unwrap_or("No dedicated GPU detected");
    match accel {
        AccelBackend::CudaNvenc => format!(
            "Profile {platform_tag} with NVIDIA GPU ({gpu_label}): prefers NVENC/CUDA-enabled tool builds."
        ),
        AccelBackend::AmfDirectml => format!(
            "Profile {platform_tag} with AMD GPU ({gpu_label}): prefers AMF/DirectML-compatible tool builds."
        ),
        AccelBackend::QsvOpenvino => format!(
            "Profile {platform_tag} with Intel GPU ({gpu_label}): prefers QuickSync/OpenVINO-compatible tool builds."
        ),
        AccelBackend::Metal => format!(
            "Profile {platform_tag} with Apple Silicon ({gpu_label}): prefers native ARM64/Metal builds."
        ),
        AccelBackend::CpuOnly => format!(
            "Profile {platform_tag} (CPU mode): selects optimized native CPU tool builds."
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detects_hardware_profile_cleanly() {
        let profile = HardwareProfile::detect();
        assert!(!profile.os.is_empty());
        assert!(!profile.arch.is_empty());
        assert!(profile.logical_cores >= 1);
        assert!(profile.total_memory_mb > 0);
        assert!(!profile.recommendation_summary.is_empty());
    }

    #[test]
    fn classifies_gpu_vendors() {
        assert_eq!(classify_vendor("NVIDIA GeForce RTX 4080"), GpuVendor::Nvidia);
        assert_eq!(classify_vendor("AMD Radeon RX 7900 XTX"), GpuVendor::Amd);
        assert_eq!(classify_vendor("Intel(R) Arc(TM) A770"), GpuVendor::Intel);
    }
}
