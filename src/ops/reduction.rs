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

pub mod argmax_argmin;
pub mod max_min;
pub mod mean;
pub mod norm;
pub mod prod;
pub mod sum;

pub use argmax_argmin::{argmax, argmin, ArgMaxOp, ArgMinOp};
pub use max_min::{max, max_backward, min, min_backward, MaxOp, MinOp};
pub use mean::{mean, mean_backward, MeanOp};
pub use norm::{norm, norm_backward, NormOp};
pub use prod::{prod, prod_backward, ProdOp};
pub use sum::{sum, sum_backward, SumOp};
