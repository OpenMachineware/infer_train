// src/ops/linalg/mod.rs

pub mod matmul;
pub mod batch_matmul;
pub mod transpose;
pub mod permute;

pub use matmul::{matmul, matmul_backward, quantized_matmul, quantized_matmul_backward, MatMulOp, QuantizedMatMulOp};
pub use batch_matmul::{batch_matmul, batch_matmul_backward, BatchMatMulOp};
pub use transpose::{transpose, transpose_backward, quantized_transpose, TransposeOp, QuantizedTransposeOp};
pub use permute::{permute, permute_backward, PermuteOp};
