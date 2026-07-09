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

pub mod mha;
pub mod rotary;
pub mod sdpa;

pub use mha::{multi_head_attention, multi_head_attention_backward, MhaOp};
pub use rotary::{rotary_embedding, rotary_embedding_backward, RotaryOp};
pub use sdpa::{
    scaled_dot_product_attention, scaled_dot_product_attention_backward, SdpaOp,
};
