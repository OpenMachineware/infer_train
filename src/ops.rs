// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

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
// Explicit exports from binary to avoid FloatType conflict with dup
pub use binary::{
    AddOp, BinaryOp, DivOp, MulOp, SubOp, add_f16, add_f16_f32,
    add_f16_f32_to_f32, add_f32, binary_broadcast_scalar, binary_scalar,
    div_f16, div_f32, mul_f16, mul_f16_f32, mul_f16_f32_to_f32, mul_f32,
    sub_f16, sub_f32,
};
pub use conv::*;
// Explicit exports from dup to avoid FloatType conflict with binary
pub use dup::{dup, dup_f16_to_f32, dup_f32_to_f16, dup_same};
pub use gelu::*;
pub use moe::*;
pub use rms_norm::*;
pub use rope::*;
pub use sampling::*;
pub use scalar::*;
pub use silu::*;
pub use softmax::*;
pub use tensor::*;
