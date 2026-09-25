#[cfg(target_arch = "aarch64")]
pub mod rms_norm;

#[cfg(target_arch = "aarch64")]
pub mod rope;

#[cfg(target_arch = "aarch64")]
pub mod softmax;

#[cfg(target_arch = "aarch64")]
pub mod silu;

#[cfg(target_arch = "aarch64")]
pub mod gelu;

#[cfg(target_os = "macos")]
pub mod gpu;
