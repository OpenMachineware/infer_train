// src/ops/mod.rs

pub mod registry;
pub mod math;
pub mod linalg;
pub mod activation;
pub mod normalization;
pub mod conv_pool;
pub mod reduction;
pub mod loss;
pub mod embedding_lookup;
pub mod attention;
pub mod control_flow;
pub mod data_gen;
pub mod cast;
pub mod tensor_manip;

pub use registry::{Operator, OpAttrs, DeviceType, OperatorRegistry};

// conv_pool
pub use conv_pool::conv2d::{conv2d, conv2d_backward};
pub use conv_pool::max_pool::{max_pool, max_pool_backward};
pub use conv_pool::avg_pool::{avg_pool, avg_pool_backward};

// normalization
pub use normalization::batch_norm::{batch_norm, batch_norm_backward};
pub use normalization::layer_norm::{layer_norm, layer_norm_backward};
pub use normalization::rms_norm::{rms_norm, rms_norm_backward};
