// NEON SIMD implementations for ARM64

pub mod rms_norm;
pub mod silu;
pub mod softmax;

pub use rms_norm::rms_norm_f32 as rms_norm_f32_neon;
pub use rms_norm::rms_norm_fp16_to_f32 as rms_norm_fp16_to_f32_neon;
pub use silu::silu_backward_f32 as silu_backward_f32_neon;
pub use silu::silu_f32 as silu_f32_neon;
pub use silu::vexpq_f32;
pub use softmax::exp as neon_exp;
pub use softmax::softmax_f32 as softmax_f32_neon;
