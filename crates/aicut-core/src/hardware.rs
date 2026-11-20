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
    /// Maximum CUDA version supported by the installed NVIDIA driver (e.g. "13.2", "12.8").
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cuda_version: Option<String>,
    /// True when the local Linux host has a CUDA runtime library (`libcudart.so*`) installed
    /// (e.g. `sudo pacman -S cuda` on Arch/CachyOS).
    #[serde(default)]
    pub cuda_runtime_installed: bool,
    /// True when a Linux Vulkan ICD / loader is available (installed by default with `nvidia-utils` on CachyOS).
    #[serde(default)]
    pub vulkan_available: bool,
    pub recommended_accel: AccelBackend,
    pub recommendation_summary: String,
}

impl HardwareProfile {
    pub fn cuda_major_minor(&self) -> Option<(u32, u32)> {
        self.cuda_version.as_deref().and_then(parse_major_minor)
    }

    /// Detects the host Linux distribution, CPU architecture, memory, NVIDIA/AMD/Intel GPUs,
    /// CUDA driver/runtime, and Vulkan support to select optimal builds for FFmpeg, yt-dlp,
    /// whisper.cpp, and llama.cpp.
    pub fn detect() -> Self {
        let mut sys = System::new_all();
        sys.refresh_all();

        let os = std::env::consts::OS.to_string();
        let arch = std::env::consts::ARCH.to_string();
        let os_version = detect_linux_distro_name()
            .or_else(System::long_os_version)
            .or_else(System::os_version)
            .unwrap_or_else(|| os.clone());

        let platform_tag = match (os.as_str(), arch.as_str()) {
            ("linux", "x86_64") => "linux-x64",
            ("linux", "aarch64") => "linux-arm64",
            _ => "linux-x64",
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

        let gpus = detect_gpus();
        let cuda_version = if gpus.iter().any(|g| g.vendor == GpuVendor::Nvidia) {
            detect_cuda_version()
        } else {
            None
        };
        let cuda_runtime_installed = detect_linux_cuda_runtime();
        let vulkan_available = detect_linux_vulkan_available();
        let recommended_accel = pick_recommended_accel(&gpus);
        let recommendation_summary = build_summary(
            &platform_tag,
            recommended_accel,
            &gpus,
            cuda_version.as_deref(),
            cuda_runtime_installed,
            vulkan_available,
        );

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
            cuda_version,
            cuda_runtime_installed,
            vulkan_available,
            recommended_accel,
            recommendation_summary,
        }
    }
}

fn detect_linux_distro_name() -> Option<String> {
    let raw = std::fs::read_to_string("/etc/os-release").ok()?;
    for line in raw.lines() {
        if let Some(val) = line.strip_prefix("PRETTY_NAME=") {
            let clean = val.trim().trim_matches('"').trim_matches('\'').trim();
            if !clean.is_empty() {
                return Some(clean.to_string());
            }
        }
    }
    None
}

/// Checks Arch/CachyOS (`/opt/cuda`) and standard Linux paths for `libcudart.so*`.
pub fn detect_linux_cuda_runtime() -> bool {
    let dirs = [
        "/opt/cuda/lib64",
        "/opt/cuda/targets/x86_64-linux/lib",
        "/usr/local/cuda/lib64",
        "/usr/lib",
        "/usr/lib64",
        "/usr/lib/x86_64-linux-gnu",
    ];
    for dir in dirs {
        if let Ok(entries) = std::fs::read_dir(dir) {
            for entry in entries.flatten() {
                if let Some(name) = entry.file_name().to_str() {
                    if name.starts_with("libcudart.so") {
                        return true;
                    }
                }
            }
        }
    }
    false
}

/// Checks whether a Linux Vulkan ICD / loader is available (present out-of-the-box with NVIDIA drivers on CachyOS/Arch).
pub fn detect_linux_vulkan_available() -> bool {
    let paths = [
        "/usr/lib/libvulkan.so.1",
        "/usr/lib64/libvulkan.so.1",
        "/usr/lib/x86_64-linux-gnu/libvulkan.so.1",
        "/usr/share/vulkan/icd.d/nvidia_icd.json",
    ];
    paths.iter().any(|p| std::path::Path::new(p).exists())
}

#[allow(dead_code)]
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

fn detect_gpus() -> Vec<GpuInfo> {
    let mut gpus = Vec::new();

    // 1. Primary: `nvidia-smi` for exact NVIDIA GPU name, VRAM (MB), and driver version on Linux.
    let mut nv_cmd = Command::new("nvidia-smi");
    crate::process::configure_std_command(&mut nv_cmd);
    if let Ok(output) = nv_cmd
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

    // 2. Linux sysfs fallback: inspect `/sys/class/drm` PCI vendor IDs (NVIDIA / AMD / Intel).
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

    gpus
}

fn pick_recommended_accel(gpus: &[GpuInfo]) -> AccelBackend {
    if gpus.iter().any(|g| g.vendor == GpuVendor::Nvidia) {
        AccelBackend::CudaNvenc
    } else if gpus.iter().any(|g| g.vendor == GpuVendor::Amd) {
        AccelBackend::AmfDirectml
    } else if gpus.iter().any(|g| g.vendor == GpuVendor::Intel) {
        AccelBackend::QsvOpenvino
    } else {
        AccelBackend::CpuOnly
    }
}

fn detect_cuda_version() -> Option<String> {
    // 1. Primary: `nvidia-smi` reports the max CUDA version supported by the installed driver (e.g. "CUDA Version: 13.2")
    let mut smi = Command::new("nvidia-smi");
    crate::process::configure_std_command(&mut smi);
    if let Ok(output) = smi.output() {
        if output.status.success() {
            let stdout = String::from_utf8_lossy(&output.stdout);
            if let Some(ver) = parse_cuda_version_from_nvidia_smi(&stdout) {
                return Some(ver);
            }
        }
    }

    // 2. Fallback: `nvcc --version` on PATH or Arch/CachyOS (`/opt/cuda/bin/nvcc`) / standard Linux path
    for nvcc_bin in ["nvcc", "/opt/cuda/bin/nvcc", "/usr/local/cuda/bin/nvcc"] {
        let mut nvcc = Command::new(nvcc_bin);
        crate::process::configure_std_command(&mut nvcc);
        if let Ok(output) = nvcc.arg("--version").output() {
            if output.status.success() {
                let stdout = String::from_utf8_lossy(&output.stdout);
                if let Some(idx) = stdout.find("release ") {
                    let rest = &stdout[idx + "release ".len()..];
                    let token = rest
                        .split(|c: char| !c.is_ascii_digit() && c != '.')
                        .next()
                        .unwrap_or("")
                        .trim_end_matches('.');
                    if parse_major_minor(token).is_some() {
                        return Some(token.to_string());
                    }
                }
            }
        }
    }
    None
}

pub fn parse_cuda_version_from_nvidia_smi(text: &str) -> Option<String> {
    let idx = text.find("CUDA Version:")?;
    let rest = text[idx + "CUDA Version:".len()..].trim_start();
    let token = rest
        .split(|c: char| !c.is_ascii_digit() && c != '.')
        .next()?
        .trim_end_matches('.');
    parse_major_minor(token)?;
    Some(token.to_string())
}

pub fn parse_major_minor(version: &str) -> Option<(u32, u32)> {
    let mut parts = version.trim().split('.');
    let major: u32 = parts.next()?.parse().ok()?;
    let minor: u32 = parts.next().and_then(|s| s.parse().ok()).unwrap_or(0);
    Some((major, minor))
}

fn build_summary(
    platform_tag: &str,
    accel: AccelBackend,
    gpus: &[GpuInfo],
    cuda_version: Option<&str>,
    cuda_runtime_installed: bool,
    vulkan_available: bool,
) -> String {
    let gpu_label = gpus
        .first()
        .map(|g| g.name.as_str())
        .unwrap_or("Nenhuma GPU dedicada detectada");
    match accel {
        AccelBackend::CudaNvenc => {
            let cuda_tag = cuda_version
                .map(|v| format!(" · CUDA {v}"))
                .unwrap_or_default();
            let runtime_note = if cuda_runtime_installed {
                "CUDA Runtime + NVENC + Vulkan ativos"
            } else if vulkan_available {
                "NVENC + Vulkan GPU ativos (instale o pacote `cuda` para backend CUDA nativo)"
            } else {
                "NVENC ativo"
            };
            format!(
                "Perfil {platform_tag} com GPU NVIDIA ({gpu_label}{cuda_tag}): {runtime_note}."
            )
        }
        AccelBackend::AmfDirectml => format!(
            "Perfil {platform_tag} com GPU AMD ({gpu_label}): aceleração Vulkan para IA local e FFmpeg otimizado."
        ),
        AccelBackend::QsvOpenvino => format!(
            "Perfil {platform_tag} com GPU Intel ({gpu_label}): aceleração Vulkan/QuickSync para IA local e vídeo."
        ),
        AccelBackend::Metal => format!(
            "Perfil {platform_tag} ({gpu_label}): aceleração por hardware."
        ),
        AccelBackend::CpuOnly => format!(
            "Perfil {platform_tag} (modo CPU): seleciona binários Linux nativos multithread."
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

    #[test]
    fn parses_cuda_version_from_nvidia_smi_banner() {
        let banner = "| NVIDIA-SMI 580.12   Driver Version: 580.12   CUDA Version: 13.2     |";
        let parsed = parse_cuda_version_from_nvidia_smi(banner).unwrap();
        assert_eq!(parsed, "13.2");
        assert_eq!(parse_major_minor(&parsed), Some((13, 2)));
    }
}
