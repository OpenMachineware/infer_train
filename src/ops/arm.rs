// ARM CPU operators (NEON, SVE)
// Re-exports platform-specific implementations

#[cfg(target_arch = "aarch64")]
pub mod neon;

// Placeholder for SVE (ARMv9)
