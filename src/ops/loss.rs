// src/ops/loss/mod.rs

pub mod cross_entropy;
pub mod mse;
pub mod l1;
pub mod bce;

pub use cross_entropy::{cross_entropy, cross_entropy_backward, CrossEntropyOp};
pub use mse::{mse, mse_backward, MseOp};
pub use l1::{l1, l1_backward, L1Op};
pub use bce::{bce, bce_backward, BceOp};
