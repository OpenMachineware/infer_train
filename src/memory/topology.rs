// Memory topology detection

/// Memory region
#[derive(Debug, Clone)]
pub enum MemoryRegion {
    /// Unified memory (Apple Silicon, AMD AI Max)
    Unified {
        id: usize,
        cpu_access: bool,
        gpu_access: bool,
        bandwidth: usize, // GB/s
        capacity: usize,  // bytes
    },
    /// GPU device-local memory (NVIDIA dGPU)
    DeviceLocal {
        id: usize,
        bandwidth: usize, // GB/s
        capacity: usize,  // bytes
    },
    /// CPU memory
    HostVisible {
        id: usize,
        bandwidth: usize, // GB/s
        capacity: usize,
    },
}

/// System memory topology
#[derive(Debug, Clone)]
pub struct MemoryTopology {
    pub regions: Vec<MemoryRegion>,
}

impl MemoryTopology {
    /// Auto-detect system topology
    pub fn detect() -> Self {
        #[cfg(target_os = "macos")]
        {
            Self::apple_silicon()
        }

        #[cfg(target_os = "linux")]
        {
            Self::linux_detect()
        }

        #[cfg(not(any(target_os = "macos", target_os = "linux")))]
        {
            Self::default()
        }
    }

    #[cfg(target_os = "macos")]
    fn apple_silicon() -> Self {
        // TODO: Detect actual unified memory size
        Self {
            regions: vec![MemoryRegion::Unified {
                id: 0,
                cpu_access: true,
                gpu_access: true,
                bandwidth: 800, // ~800 GB/s for M1 Max
                capacity: 64 * 1024 * 1024 * 1024, // 64GB
            }],
        }
    }

    #[cfg(target_os = "linux")]
    fn linux_detect() -> Self {
        // TODO: Detect NVIDIA GPU + CPU
        Self::default()
    }
}

impl Default for MemoryTopology {
    fn default() -> Self {
        Self {
            regions: vec![MemoryRegion::HostVisible {
                id: 0,
                bandwidth: 50, // DDR5 ~50GB/s
                capacity: 16 * 1024 * 1024 * 1024,
            }],
        }
    }
}
