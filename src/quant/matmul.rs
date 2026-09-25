#[cfg(target_arch = "aarch64")]
pub mod k_quant;

#[cfg(target_arch = "aarch64")]
pub use k_quant::*;
