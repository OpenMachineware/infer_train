// src/ops/data_gen/mod.rs

pub mod arange_linspace;
pub mod eye_diag;
pub mod rand;
pub mod zeros_ones;

pub use arange_linspace::{arange, linspace, ArangeOp, LinspaceOp};
pub use eye_diag::{diag, eye, DiagOp, EyeOp};
pub use rand::{rand, randn, RandOp, RandnOp};
pub use zeros_ones::{full, ones, zeros, FullOp, OnesOp, ZerosOp};
