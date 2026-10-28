// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

//! Memory topology abstraction for heterogeneous systems.

use std::fmt;
use std::time::Duration;

use super::device::DeviceId;

/// Memory bandwidth in GB/s.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Bandwidth(pub f64);

impl fmt::Display for Bandwidth {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:.1} GB/s", self.0)
    }
}

/// Unique memory region identifier.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct MemoryRegionId(pub usize);

impl fmt::Display for MemoryRegionId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "MemoryRegion({})", self.0)
    }
}

/// Memory region in the system.
#[derive(Clone, Debug)]
pub enum MemoryRegion {
    /// Unified memory (Apple Silicon, AMD AI Max).
    /// Accessible by both CPU and GPU with zero-copy.
    Unified {
        id: MemoryRegionId,
        device_id: DeviceId,
        cpu_access: bool,
        gpu_access: bool,
        bandwidth: Bandwidth,
        capacity: u64, // bytes
    },

    /// Device-local memory (NVIDIA dGPU, AMD dGPU).
    /// GPU-only access, requires DMA transfer.
    DeviceLocal {
        id: MemoryRegionId,
        device_id: DeviceId,
        bandwidth: Bandwidth,
        capacity: u64,
    },

    /// Host-visible memory (system RAM).
    /// CPU access, can be used as transfer buffer.
    HostVisible { id: MemoryRegionId, bandwidth: Bandwidth, capacity: u64 },

    /// Remote memory (network).
    /// Access via RDMA or other network protocol.
    Remote { id: MemoryRegionId, endpoint: String, bandwidth: Bandwidth },
}

impl MemoryRegion {
    /// Get the region ID.
    pub fn id(&self) -> MemoryRegionId {
        match self {
            MemoryRegion::Unified { id, .. } => *id,
            MemoryRegion::DeviceLocal { id, .. } => *id,
            MemoryRegion::HostVisible { id, .. } => *id,
            MemoryRegion::Remote { id, .. } => *id,
        }
    }

    /// Get the memory capacity in bytes.
    pub fn capacity(&self) -> u64 {
        match self {
            MemoryRegion::Unified { capacity, .. } => *capacity,
            MemoryRegion::DeviceLocal { capacity, .. } => *capacity,
            MemoryRegion::HostVisible { capacity, .. } => *capacity,
            MemoryRegion::Remote { .. } => 0, // Unknown
        }
    }

    /// Get the memory bandwidth.
    pub fn bandwidth(&self) -> Bandwidth {
        match self {
            MemoryRegion::Unified { bandwidth, .. } => *bandwidth,
            MemoryRegion::DeviceLocal { bandwidth, .. } => *bandwidth,
            MemoryRegion::HostVisible { bandwidth, .. } => *bandwidth,
            MemoryRegion::Remote { bandwidth, .. } => *bandwidth,
        }
    }

    /// Check if this region has unified memory (zero-copy).
    pub fn is_unified(&self) -> bool {
        matches!(self, MemoryRegion::Unified { .. })
    }

    /// Check if CPU can access this memory directly.
    pub fn cpu_accessible(&self) -> bool {
        match self {
            MemoryRegion::Unified { cpu_access, .. } => *cpu_access,
            MemoryRegion::HostVisible { .. } => true,
            MemoryRegion::DeviceLocal { .. } | MemoryRegion::Remote { .. } => {
                false
            }
        }
    }

    /// Check if GPU can access this memory directly.
    pub fn gpu_accessible(&self) -> bool {
        match self {
            MemoryRegion::Unified { gpu_access, .. } => *gpu_access,
            MemoryRegion::DeviceLocal { .. } => true,
            MemoryRegion::HostVisible { .. } | MemoryRegion::Remote { .. } => {
                false
            }
        }
    }
}

/// Transfer type between memory regions.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TransferType {
    /// Zero-copy (shared pointer in unified memory).
    ZeroCopy,
    /// DMA direct transfer (NVLink, HBM).
    DmaDirect,
    /// PCIe transfer.
    Pcie,
    /// Network transfer (RDMA, TCP).
    Network,
}

impl fmt::Display for TransferType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TransferType::ZeroCopy => write!(f, "Zero-Copy"),
            TransferType::DmaDirect => write!(f, "DMA Direct"),
            TransferType::Pcie => write!(f, "PCIe"),
            TransferType::Network => write!(f, "Network"),
        }
    }
}

/// Link between two memory regions.
#[derive(Clone, Debug)]
pub struct MemoryLink {
    /// Source region.
    pub from: MemoryRegionId,
    /// Destination region.
    pub to: MemoryRegionId,
    /// Transfer type.
    pub transfer_type: TransferType,
    /// Bandwidth in GB/s.
    pub bandwidth: Bandwidth,
    /// Transfer latency.
    pub latency: Duration,
}

impl MemoryLink {
    /// Calculate transfer time for a given size.
    pub fn transfer_time(&self, size_bytes: u64) -> Duration {
        if self.bandwidth.0 == 0.0 {
            return Duration::from_secs(0);
        }

        // time = size / bandwidth
        let seconds = (size_bytes as f64) / (self.bandwidth.0 * 1e9);
        let transfer_duration = Duration::from_secs_f64(seconds);

        // Add latency overhead
        transfer_duration + self.latency
    }
}

/// System memory topology.
#[derive(Clone, Debug, Default)]
pub struct MemoryTopology {
    /// All memory regions.
    pub regions: Vec<MemoryRegion>,
    /// Links between regions.
    pub links: Vec<MemoryLink>,
}

impl MemoryTopology {
    /// Create an empty topology.
    pub fn new() -> Self {
        Self { regions: Vec::new(), links: Vec::new() }
    }

    /// Add a region to the topology.
    pub fn add_region(&mut self, region: MemoryRegion) -> MemoryRegionId {
        let id = region.id();
        self.regions.push(region);
        id
    }

    /// Add a link between regions.
    pub fn add_link(&mut self, link: MemoryLink) {
        self.links.push(link);
    }

    /// Get a region by ID.
    pub fn get_region(&self, id: MemoryRegionId) -> Option<&MemoryRegion> {
        self.regions.iter().find(|r| r.id() == id)
    }

    /// Find a link between two regions.
    pub fn find_link(
        &self,
        from: MemoryRegionId,
        to: MemoryRegionId,
    ) -> Option<&MemoryLink> {
        self.links.iter().find(|l| l.from == from && l.to == to)
    }

    /// Detect the system memory topology.
    #[cfg(target_os = "macos")]
    pub fn detect() -> Self {
        Self::detect_apple_silicon()
    }

    #[cfg(not(target_os = "macos"))]
    pub fn detect() -> Self {
        Self::detect_generic()
    }

    /// Detect Apple Silicon memory topology.
    #[cfg(target_os = "macos")]
    fn detect_apple_silicon() -> Self {
        use super::device::get_system_memory;

        let total_memory = get_system_memory();

        // Apple Silicon: single unified memory region
        let region = MemoryRegion::Unified {
            id: MemoryRegionId(0),
            device_id: super::device::DeviceId(0),
            cpu_access: true,
            gpu_access: true,
            bandwidth: Bandwidth(100.0), // 100 GB/s for M1/M2
            capacity: total_memory,
        };

        let mut topology = Self::new();
        topology.add_region(region);
        topology
    }

    /// Detect generic system topology.
    #[cfg(not(target_os = "macos"))]
    fn detect_generic() -> Self {
        use super::device::get_system_memory;

        let total_memory = get_system_memory();

        // Generic system: host-visible memory only
        let region = MemoryRegion::HostVisible {
            id: MemoryRegionId(0),
            bandwidth: Bandwidth(50.0), // DDR5
            capacity: total_memory,
        };

        let mut topology = Self::new();
        topology.add_region(region);
        topology
    }
}
