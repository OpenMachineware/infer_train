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

pub mod batch_matmul;
pub mod matmul;
pub mod permute;
pub mod transpose;

pub use batch_matmul::{batch_matmul, batch_matmul_backward, BatchMatMulOp};
pub use matmul::{
    matmul, matmul_backward, quantized_matmul, quantized_matmul_backward,
    MatMulOp, QuantizedMatMulOp,
};
pub use permute::{permute, permute_backward, PermuteOp};
pub use transpose::{
    quantized_transpose, transpose, transpose_backward, QuantizedTransposeOp,
    TransposeOp,
};
