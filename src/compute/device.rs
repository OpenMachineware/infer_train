// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

// Device abstraction for cross-device tensor operations

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Device {
    Cpu,
    #[cfg(all(target_arch = "aarch64", feature = "metal"))]
    Metal,
    #[cfg(feature = "cuda")]
    Cuda(usize), // Device index
}

impl Device {
    pub fn is_cpu(&self) -> bool {
        matches!(self, Device::Cpu)
    }

    #[cfg(all(target_arch = "aarch64", feature = "metal"))]
    pub fn is_metal(&self) -> bool {
        matches!(self, Device::Metal)
    }

    #[cfg(feature = "cuda")]
    pub fn is_cuda(&self) -> bool {
        matches!(self, Device::Cuda(_))
    }
}

impl Default for Device {
    fn default() -> Self {
        Device::Cpu
    }
}

impl std::fmt::Display for Device {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Device::Cpu => write!(f, "CPU"),
            #[cfg(all(target_arch = "aarch64", feature = "metal"))]
            Device::Metal => write!(f, "Metal"),
            #[cfg(feature = "cuda")]
            Device::Cuda(idx) => write!(f, "CUDA:{}", idx),
        }
    }
}
