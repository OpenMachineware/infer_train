// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

//! Device abstraction for heterogeneous compute.

use std::fmt;

/// Unique device identifier.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct DeviceId(pub usize);

impl fmt::Display for DeviceId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Device({})", self.0)
    }
}

/// Device type enumeration.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DeviceType {
    /// ARM CPU (NEON)
    ArmCpu,
    /// x86_64 CPU (AVX2/AVX512)
    X86Cpu,
    /// RISC-V CPU (V extension)
    RiscVCpu,
    /// Apple Metal GPU
    Metal,
    /// NVIDIA CUDA GPU
    Cuda,
    /// AMD ROCm GPU
    Rocm,
    /// Generic GPU (Vulkan)
    Vulkan,
}

impl DeviceType {
    /// Returns true if this is a CPU device.
    pub fn is_cpu(&self) -> bool {
        matches!(
            self,
            DeviceType::ArmCpu | DeviceType::X86Cpu | DeviceType::RiscVCpu
        )
    }

    /// Returns true if this is a GPU device.
    pub fn is_gpu(&self) -> bool {
        matches!(
            self,
            DeviceType::Metal
                | DeviceType::Cuda
                | DeviceType::Rocm
                | DeviceType::Vulkan
        )
    }

    /// Returns true if this device has unified memory.
    pub fn has_unified_memory(&self) -> bool {
        matches!(self, DeviceType::ArmCpu | DeviceType::Metal)
    }
}

impl fmt::Display for DeviceType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DeviceType::ArmCpu => write!(f, "ARM CPU"),
            DeviceType::X86Cpu => write!(f, "x86 CPU"),
            DeviceType::RiscVCpu => write!(f, "RISC-V CPU"),
            DeviceType::Metal => write!(f, "Metal GPU"),
            DeviceType::Cuda => write!(f, "CUDA GPU"),
            DeviceType::Rocm => write!(f, "ROCm GPU"),
            DeviceType::Vulkan => write!(f, "Vulkan GPU"),
        }
    }
}

/// Device information.
#[derive(Clone, Debug)]
pub struct DeviceInfo {
    /// Unique identifier.
    pub id: DeviceId,
    /// Device type.
    pub device_type: DeviceType,
    /// Human-readable name.
    pub name: String,
    /// Available memory in bytes.
    pub memory_size: u64,
    /// Memory bandwidth in GB/s.
    pub memory_bandwidth: f64,
    /// Number of compute units (cores, SMs, etc.).
    pub compute_units: usize,
    /// Maximum clock frequency in MHz.
    pub max_clock: u32,
}

/// Device handle for compute operations.
#[derive(Clone, Debug)]
pub struct Device {
    /// Device information.
    pub info: DeviceInfo,
}

impl Device {
    /// Create a new device with the given info.
    pub fn new(info: DeviceInfo) -> Self {
        Self { info }
    }

    /// Get the device ID.
    pub fn id(&self) -> DeviceId {
        self.info.id
    }

    /// Get the device type.
    pub fn device_type(&self) -> DeviceType {
        self.info.device_type
    }

    /// Get the device name.
    pub fn name(&self) -> &str {
        &self.info.name
    }

    /// Get available memory in bytes.
    pub fn memory_size(&self) -> u64 {
        self.info.memory_size
    }

    /// Check if this device has unified memory.
    pub fn has_unified_memory(&self) -> bool {
        self.info.device_type.has_unified_memory()
    }
}

#[cfg(target_os = "macos")]
impl Default for Device {
    fn default() -> Self {
        // Default to Metal on macOS
        Self::detect_metal_device()
    }
}

#[cfg(not(target_os = "macos"))]
impl Default for Device {
    fn default() -> Self {
        // Default to CPU on other platforms
        Self::detect_cpu_device()
    }
}

impl Device {
    /// Detect the best available device.
    #[cfg(target_os = "macos")]
    pub fn detect_best() -> Self {
        // On macOS, prefer Metal
        Self::detect_metal_device()
    }

    /// Detect the best available device.
    #[cfg(not(target_os = "macos"))]
    pub fn detect_best() -> Self {
        // On other platforms, use CPU for now
        Self::detect_cpu_device()
    }

    /// Detect CPU device.
    fn detect_cpu_device() -> Self {
        #[cfg(target_arch = "aarch64")]
        {
            Self::detect_arm_device()
        }
        #[cfg(target_arch = "x86_64")]
        {
            Self::detect_x86_device()
        }
        #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
        {
            Self::detect_generic_device()
        }
    }

    #[cfg(target_arch = "aarch64")]
    fn detect_arm_device() -> Self {
        let cores =
            std::thread::available_parallelism().map(|p| p.get()).unwrap_or(4);

        Self::new(DeviceInfo {
            id: DeviceId(0),
            device_type: DeviceType::ArmCpu,
            name: "ARM CPU".to_string(),
            memory_size: get_system_memory(),
            memory_bandwidth: 100.0, // Approximate for Apple Silicon
            compute_units: cores,
            max_clock: 3200,
        })
    }

    #[cfg(target_arch = "x86_64")]
    fn detect_x86_device() -> Self {
        let cores =
            std::thread::available_parallelism().map(|p| p.get()).unwrap_or(4);

        Self::new(DeviceInfo {
            id: DeviceId(0),
            device_type: DeviceType::X86Cpu,
            name: "x86 CPU".to_string(),
            memory_size: get_system_memory(),
            memory_bandwidth: 50.0, // DDR5 approximate
            compute_units: cores,
            max_clock: 4000,
        })
    }

    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
    fn detect_generic_device() -> Self {
        Self::new(DeviceInfo {
            id: DeviceId(0),
            device_type: DeviceType::ArmCpu, // Fallback
            name: "Generic CPU".to_string(),
            memory_size: get_system_memory(),
            memory_bandwidth: 30.0,
            compute_units: 4,
            max_clock: 2000,
        })
    }

    #[cfg(all(target_arch = "aarch64", feature = "metal"))]
    fn detect_metal_device() -> Self {
        // Metal device detection - placeholder for now
        // TODO: Use Metal API to get actual device info
        Self::detect_arm_device()
    }

    #[cfg(not(all(target_arch = "aarch64", feature = "metal")))]
    fn detect_metal_device() -> Self {
        // Fallback to CPU if Metal not available
        Self::detect_cpu_device()
    }
}

/// Get system memory in bytes.
pub fn get_system_memory() -> u64 {
    #[cfg(target_os = "macos")]
    {
        use std::mem;
        let mut size: u64 = 0;
        let mut len = mem::size_of::<u64>();
        unsafe {
            libc::sysctlbyname(
                b"hw.memsize\0".as_ptr() as *const i8,
                &mut size as *mut _ as *mut libc::c_void,
                &mut len as *mut usize,
                std::ptr::null_mut(),
                0,
            );
        }
        size
    }
    #[cfg(target_os = "linux")]
    {
        // Read from /proc/meminfo
        std::fs::read_to_string("/proc/meminfo")
            .ok()
            .and_then(|s| {
                s.lines()
                    .find(|l| l.starts_with("MemTotal:"))
                    .and_then(|l| {
                        l.split_whitespace()
                            .nth(1)
                            .and_then(|v| v.parse::<u64>().ok())
                    })
            })
            .unwrap_or(8 * 1024 * 1024 * 1024) // Default 8GB
            * 1024 // Convert KB to bytes
    }
    #[cfg(not(any(target_os = "macos", target_os = "linux")))]
    {
        8 * 1024 * 1024 * 1024 // Default 8GB
    }
}
