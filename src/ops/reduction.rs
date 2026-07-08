// src/ops/reduction/mod.rs

pub mod argmax_argmin;
pub mod max_min;
pub mod mean;
pub mod norm;
pub mod prod;
pub mod sum;

pub use argmax_argmin::{argmax, argmin, ArgMaxOp, ArgMinOp};
pub use max_min::{max, max_backward, min, min_backward, MaxOp, MinOp};
pub use mean::{mean, mean_backward, MeanOp};
pub use norm::{norm, norm_backward, NormOp};
pub use prod::{prod, prod_backward, ProdOp};
pub use sum::{sum, sum_backward, SumOp};
