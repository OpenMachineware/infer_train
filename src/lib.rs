// Allow unsafe operations in unsafe functions without explicit blocks (Rust 2024)
#![allow(unsafe_op_in_unsafe_fn)]

pub mod ops;
pub mod quant;

pub const QK_K: usize = 256;  // Block size for K-quant formats
pub const K_SCALE_SIZE: usize = 12;  // Scale bytes for Q4_K, Q5_K
