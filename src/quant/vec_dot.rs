#[cfg(target_arch = "aarch64")]
pub mod arm;

#[cfg(target_arch = "aarch64")]
pub mod gpu;

pub mod scalar;
pub mod tables;
