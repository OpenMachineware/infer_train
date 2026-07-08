// src/ops/linalg/mod.rs

pub mod batch_matmul;
pub mod matmul;
pub mod permute;
pub mod transpose;

pub use batch_matmul::{batch_matmul, batch_matmul_backward, BatchMatMulOp};
pub use matmul::{
    matmul, matmul_backward, quantized_matmul, quantized_matmul_backward,
    MatMulOp, QuantizedMatMulOp,
};
pub use permute::{permute, permute_backward, PermuteOp};
pub use transpose::{
    quantized_transpose, transpose, transpose_backward, QuantizedTransposeOp,
    TransposeOp,
};
