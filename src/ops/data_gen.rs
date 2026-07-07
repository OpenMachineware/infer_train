// src/ops/data_gen/mod.rs

pub mod zeros_ones;
pub mod arange_linspace;
pub mod eye_diag;
pub mod rand;

pub use zeros_ones::{zeros, ones, full, ZerosOp, OnesOp, FullOp};
pub use arange_linspace::{arange, linspace, ArangeOp, LinspaceOp};
pub use eye_diag::{eye, diag, EyeOp, DiagOp};
pub use rand::{rand, randn, RandOp, RandnOp};
