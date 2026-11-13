// 内存拓扑探测

/// 内存区域
#[derive(Debug, Clone)]
pub enum MemoryRegion {
    /// 统一内存（Apple Silicon, AMD AI Max）
    Unified {
        id: usize,
        cpu_access: bool,
        gpu_access: bool,
        bandwidth: usize, // GB/s
        capacity: usize,  // bytes
    },
    /// GPU 专用显存（NVIDIA dGPU）
    DeviceLocal {
        id: usize,
        bandwidth: usize, // GB/s
        capacity: usize,  // bytes
    },
    /// CPU 内存
    HostVisible {
        id: usize,
        bandwidth: usize, // GB/s
        capacity: usize,
    },
}

/// 系统内存拓扑
#[derive(Debug, Clone)]
pub struct MemoryTopology {
    pub regions: Vec<MemoryRegion>,
}

impl MemoryTopology {
    /// 自动探测系统拓扑
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
        // TODO: 实际检测统一内存大小
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
        // TODO: 检测 NVIDIA GPU + CPU
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
