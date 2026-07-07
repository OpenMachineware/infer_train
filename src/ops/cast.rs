// src/ops/cast/mod.rs

pub mod cast;
pub mod quantize;

pub use cast::{cast, cast_backward, CastOp};
pub use quantize::{quantize, quantize_backward, dequantize, QuantizeOp, DequantizeOp};
