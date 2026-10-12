// NEON SIMD operators for ARM64
// NEON implementations are currently inline in the operator files
// (attention.rs, rms_norm.rs, etc.) using #[cfg(target_arch = "aarch64")]
// This module is for future organization

// TODO: Move NEON-specific code here for better organization

// Note: Current NEON functions are exported directly from their source modules
// via cfg(target_arch = "aarch64")
