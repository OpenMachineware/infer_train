// src/ops/mod.rs

pub mod activation;
pub mod attention;
pub mod cast;
pub mod control_flow;
pub mod conv_pool;
pub mod data_gen;
pub mod embedding_lookup;
pub mod linalg;
pub mod loss;
pub mod math;
pub mod normalization;
pub mod reduction;
pub mod registry;
pub mod tensor_manip;

pub use registry::{DeviceType, OpAttrs, Operator, OperatorRegistry};

// conv_pool
pub use conv_pool::avg_pool::{avg_pool, avg_pool_backward};
pub use conv_pool::conv2d::{conv2d, conv2d_backward};
pub use conv_pool::max_pool::{max_pool, max_pool_backward};

// normalization
pub use normalization::batch_norm::{batch_norm, batch_norm_backward};
pub use normalization::layer_norm::{layer_norm, layer_norm_backward};
pub use normalization::rms_norm::{rms_norm, rms_norm_backward};
