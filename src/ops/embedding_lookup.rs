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

pub mod embedding;
pub mod gather_scatter;
pub mod one_hot;
pub mod topk;

pub use embedding::{embedding, embedding_backward, EmbeddingOp};
pub use gather_scatter::{
    gather, gather_backward, scatter, scatter_backward, GatherOp, ScatterOp,
};
pub use one_hot::{one_hot, OneHotOp};
pub use topk::{topk, topk_backward, TopkOp};
