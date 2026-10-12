// NEON SIMD implementations for ARM64

pub mod binary;
pub mod dup;
pub mod gelu;
pub mod rms_norm;
pub mod sampling;
pub mod silu;
pub mod softmax;
pub mod tensor;

pub use binary::{add_f32, div_f32, mul_f32, sub_f32};
pub use dup::dup_f16_to_f32 as dup_f16_to_f32_neon;
pub use dup::dup_f32_to_f16 as dup_f32_to_f16_neon;
pub use gelu::gelu_f32 as gelu_f32_neon;
pub use gelu::gelu_quick_f32 as gelu_quick_f32_neon;
pub use rms_norm::rms_norm_f32 as rms_norm_f32_neon;
pub use rms_norm::rms_norm_fp16_to_f32 as rms_norm_fp16_to_f32_neon;
pub use sampling::argmax_f32 as argmax_f32_neon;
pub use sampling::init_indices_neon;
pub use silu::silu_backward_f32 as silu_backward_f32_neon;
pub use silu::silu_f32 as silu_f32_neon;
pub use silu::vexpq_f32;
pub use softmax::exp as neon_exp;
pub use softmax::softmax_f32 as softmax_f32_neon;
pub use tensor::clamp_f32 as clamp_f32_neon;
pub use tensor::group_norm_f32 as group_norm_f32_neon;
pub use tensor::layer_norm_f32 as layer_norm_f32_neon;
