// Operator implementations organized by platform

// Platform-specific modules
pub mod arm;
pub mod common;
pub mod cuda;
pub mod metal;
pub mod x86;

// Existing operator modules
pub mod activation_tables;
pub mod attention;
pub mod attention_fused;
pub mod binary;
pub mod conv;
pub mod dup;
pub mod gelu;
pub mod moe;
pub mod rms_norm;
pub mod rope;
pub mod sampling;
pub mod scalar;
pub mod silu;
pub mod softmax;
pub mod tensor;

// Re-exports
pub use attention::*;
pub use binary::*;
pub use conv::*;
pub use dup::*;
pub use gelu::*;
pub use moe::*;
pub use rms_norm::*;
pub use rope::*;
pub use sampling::*;
pub use scalar::*;
pub use silu::*;
pub use softmax::*;
pub use tensor::*;
