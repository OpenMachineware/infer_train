// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

// GPU capabilities detection
// Detects GPU type, memory, and supported features

use std::fmt;

/// GPU vendor
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GpuVendor {
    Apple,
    Nvidia,
    Amd,
    Intel,
    Unknown,
}

/// GPU capabilities
#[derive(Debug, Clone)]
pub struct GpuCapabilities {
    /// GPU vendor
    pub vendor: GpuVendor,
    /// Device name (e.g., "Apple M1 Max", "NVIDIA RTX 4090")
    pub name: String,
    /// Total memory in bytes (for discrete GPUs) or recommended working set (unified memory)
    pub memory: u64,
    /// SIMD group width (32 for Apple GPU, 32 for NVIDIA warp, 64 for AMD wavefront)
    pub simd_width: usize,
    /// Max threadgroup memory (bytes)
    pub max_threadgroup_memory: usize,
    /// Max SIMD groups per threadgroup
    pub max_simd_groups: usize,
    /// Supports SIMD dot product
    pub has_simd_dot: bool,
    /// Supports bfloat16
    pub has_bfloat: bool,
    /// Supports half-precision (FP16)
    pub has_fp16: bool,
    /// Supports int8 matrix multiplication
    pub has_int8_matmul: bool,
    /// Metal: GPU family
    pub metal_family: Option<u32>,
    /// CUDA: compute capability major
    pub cuda_compute_major: Option<u32>,
    /// CUDA: compute capability minor
    pub cuda_compute_minor: Option<u32>,
}

impl Default for GpuCapabilities {
    fn default() -> Self {
        Self {
            vendor: GpuVendor::Unknown,
            name: String::new(),
            memory: 0,
            simd_width: 32,
            max_threadgroup_memory: 32 * 1024,
            max_simd_groups: 8,
            has_simd_dot: false,
            has_bfloat: false,
            has_fp16: true,
            has_int8_matmul: false,
            metal_family: None,
            cuda_compute_major: None,
            cuda_compute_minor: None,
        }
    }
}

impl fmt::Display for GpuCapabilities {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{} {} ({} GB, {} threads)",
            match self.vendor {
                GpuVendor::Apple => "Apple",
                GpuVendor::Nvidia => "NVIDIA",
                GpuVendor::Amd => "AMD",
                GpuVendor::Intel => "Intel",
                GpuVendor::Unknown => "Unknown",
            },
            self.name,
            self.memory / (1024 * 1024 * 1024),
            self.simd_width
        )?;
        if self.has_simd_dot {
            write!(f, ", simd_dot")?;
        }
        if self.has_bfloat {
            write!(f, ", bf16")?;
        }
        Ok(())
    }
}

/// Detect GPU capabilities
pub fn detect() -> GpuCapabilities {
    // Try Metal first (macOS)
    #[cfg(all(target_os = "macos", feature = "metal"))]
    {
        if let Some(gpu) = detect_metal() {
            return gpu;
        }
    }

    // Fallback for macOS without Metal feature
    #[cfg(all(target_os = "macos", not(feature = "metal")))]
    {
        return detect_metal_fallback();
    }

    // Try CUDA (Linux/Windows with NVIDIA GPU)
    #[cfg(feature = "cuda")]
    {
        if let Some(gpu) = detect_cuda() {
            return gpu;
        }
    }

    // Try ROCm (Linux with AMD GPU)
    #[cfg(feature = "rocm")]
    {
        if let Some(gpu) = detect_rocm() {
            return gpu;
        }
    }

    // No GPU detected
    GpuCapabilities::default()
}

#[cfg(all(target_os = "macos", feature = "metal"))]
fn detect_metal() -> Option<GpuCapabilities> {
    use metal::{Device, MTLGPUFamily};

    let device = Device::system_default()?;
    let mut caps = GpuCapabilities::default();

    caps.vendor = GpuVendor::Apple;
    caps.name = device.name().to_string();

    // Memory
    caps.memory = device.recommended_max_working_set_size();

    // SIMD width
    caps.simd_width = device.max_threads_per_threadgroup() as usize;
    // Apple GPU has SIMD width of 32
    if caps.simd_width >= 1024 {
        caps.simd_width = 32; // max_threads_per_threadgroup returns max total threads
    }

    // Threadgroup memory
    caps.max_threadgroup_memory = device.max_threadgroup_memory_length();

    // Detect GPU family
    // Apple GPUs support family Apple7+ (M1+)
    if device.supports_family(MTLGPUFamily::Apple7) {
        caps.metal_family = Some(7);
        caps.has_simd_dot = true; // M2+ has simd_dot, M1 uses different path
        caps.has_fp16 = true;
    }
    if device.supports_family(MTLGPUFamily::Apple8) {
        caps.metal_family = Some(8);
        // M2+ features
        caps.has_simd_dot = true;
        caps.has_bfloat = true; // M3+ has bf16, M2 has partial
    }
    if device.supports_family(MTLGPUFamily::Apple9) {
        caps.metal_family = Some(9);
        caps.has_bfloat = true; // M3+ full bf16 support
    }

    // Infer GPU model from name
    if caps.name.contains("M1") {
        caps.has_simd_dot = true; // via simdfloat
        caps.has_bfloat = false;
    } else if caps.name.contains("M2") {
        caps.has_simd_dot = true;
        caps.has_bfloat = true;
    } else if caps.name.contains("M3") || caps.name.contains("M4") {
        caps.has_simd_dot = true;
        caps.has_bfloat = true;
        caps.has_int8_matmul = true;
    }

    // Max SIMD groups (typically 8 on Apple GPU)
    caps.max_simd_groups = 8;

    Some(caps)
}

#[cfg(all(target_os = "macos", not(feature = "metal")))]
fn detect_metal_fallback() -> GpuCapabilities {
    // Without Metal crate, use sysctl to get basic info
    let mut caps =
        GpuCapabilities { vendor: GpuVendor::Apple, ..Default::default() };

    // Assume Apple Silicon
    caps.name = "Apple GPU".to_string();
    caps.simd_width = 32;
    caps.max_threadgroup_memory = 32 * 1024;
    caps.has_simd_dot = true;
    caps.has_fp16 = true;

    // Try to get memory from sysctl
    #[cfg(target_os = "macos")]
    {
        if let Some(mem) = get_sysctl_int("hw.memsize") {
            caps.memory = mem as u64 / 2; // Rough estimate: half of RAM
        }
    }

    caps
}

#[cfg(feature = "cuda")]
fn detect_cuda() -> Option<GpuCapabilities> {
    // Placeholder: CUDA detection via rust-cuda or CUDA driver API
    // This would use cudaGetDeviceProperties
    None
}

#[cfg(feature = "rocm")]
fn detect_rocm() -> Option<GpuCapabilities> {
    // Placeholder: ROCm detection via HIP/ROCm API
    None
}

#[cfg(target_os = "macos")]
fn get_sysctl_int(name: &str) -> Option<i64> {
    use std::ffi::CString;
    use std::ptr;

    let name_c = CString::new(name).ok()?;
    let mut value: i64 = 0;
    let mut size = std::mem::size_of::<i64>();

    unsafe {
        if libc::sysctlbyname(
            name_c.as_ptr(),
            &mut value as *mut i64 as *mut libc::c_void,
            &mut size,
            ptr::null_mut(),
            0,
        ) != 0
        {
            return None;
        }
    }

    Some(value)
}
