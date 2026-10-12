// Platform detection: CPU and GPU capabilities
// Detects hardware features at startup for optimal operator selection

pub mod cpu;
pub mod gpu;

pub use cpu::{CpuArch, CpuCapabilities};
pub use gpu::{GpuCapabilities, GpuVendor};

use std::sync::OnceLock;

static CPU_CAPS: OnceLock<CpuCapabilities> = OnceLock::new();
static GPU_CAPS: OnceLock<GpuCapabilities> = OnceLock::new();

/// Initialize platform detection (call once at startup)
pub fn init() {
    let cpu = cpu::detect();
    let gpu = gpu::detect();

    CPU_CAPS.get_or_init(|| cpu);
    GPU_CAPS.get_or_init(|| gpu);

    // Log detected capabilities
    log::info!("CPU: {:?}", CPU_CAPS.get().unwrap());
    log::info!("GPU: {:?}", GPU_CAPS.get().unwrap());
}

/// Get CPU capabilities (call after init())
pub fn cpu() -> &'static CpuCapabilities {
    CPU_CAPS.get().expect("Platform not initialized, call init() first")
}

/// Get GPU capabilities (call after init())
pub fn gpu() -> &'static GpuCapabilities {
    GPU_CAPS.get().expect("Platform not initialized, call init() first")
}
