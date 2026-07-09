// InferTrain - A Unified Inference and Training Engine
//
// Copyright (c) 2026 Jia Liu & InferTrain Contributors
// SPDX-License-Identifier: Apache-2.0
//
// This file is part of InferTrain.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at:
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

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
