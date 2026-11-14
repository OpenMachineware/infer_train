// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

// Inference engine

pub mod engine;
pub mod simple_model;

pub use engine::InferenceEngine;
pub use simple_model::{ModelConfig, SimpleModel};
