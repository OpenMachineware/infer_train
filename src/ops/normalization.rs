// src/ops/normalization/mod.rs

pub mod batch_norm;
pub mod layer_norm;
pub mod rms_norm;
pub mod instance_norm;
pub mod group_norm;

pub use batch_norm::{batch_norm, batch_norm_backward, BatchNormOp};
pub use layer_norm::{layer_norm, layer_norm_backward, LayerNormOp};
pub use rms_norm::{rms_norm, rms_norm_backward, RmsNormOp};
pub use instance_norm::{instance_norm, instance_norm_backward, InstanceNormOp};
pub use group_norm::{group_norm, group_norm_backward, GroupNormOp};
