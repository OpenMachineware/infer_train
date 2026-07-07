// src/ops/conv_pool/mod.rs

pub mod conv2d;
pub mod max_pool;
pub mod avg_pool;
pub mod conv1d;
pub mod conv3d;
pub mod conv_transpose;
pub mod adaptive_pool;
pub mod upsample;

pub use conv2d::{conv2d, conv2d_backward, Conv2dOp};
pub use max_pool::{max_pool, max_pool_backward, MaxPoolOp};
pub use avg_pool::{avg_pool, avg_pool_backward, AvgPoolOp};
pub use conv1d::{conv1d, conv1d_backward, Conv1dOp};
pub use conv3d::{conv3d, conv3d_backward, Conv3dOp};
pub use conv_transpose::{conv_transpose, conv_transpose_backward, ConvTransposeOp};
pub use adaptive_pool::{adaptive_avg_pool, adaptive_max_pool, AdaptivePoolOp};
pub use upsample::{upsample, upsample_backward, UpsampleOp};
