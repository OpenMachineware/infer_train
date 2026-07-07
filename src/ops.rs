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
