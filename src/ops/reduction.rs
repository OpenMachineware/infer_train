// src/ops/reduction/mod.rs

pub mod sum;
pub mod mean;
pub mod max_min;
pub mod prod;
pub mod argmax_argmin;
pub mod norm;

pub use sum::{sum, sum_backward, SumOp};
pub use mean::{mean, mean_backward, MeanOp};
pub use max_min::{max, min, max_backward, min_backward, MaxOp, MinOp};
pub use prod::{prod, prod_backward, ProdOp};
pub use argmax_argmin::{argmax, argmin, ArgMaxOp, ArgMinOp};
pub use norm::{norm, norm_backward, NormOp};
