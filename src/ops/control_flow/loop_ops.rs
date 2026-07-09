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

use crate::dtype::DType;
use crate::ops::registry::{OpAttrs, Operator};
use crate::tensor::Tensor;

// ============================================================
// While Loop (Conditional Loop)
// ============================================================

pub fn while_loop<T: DType + Send + Sync>(condition: &[i8], body: &[T]) -> T {
    // Simplified: iterate through values in body until condition is false
    // Actual implementation needs to work with graph execution
    let mut result = T::from_f32(0.0);
    for (i, &cond) in condition.iter().enumerate() {
        if cond != 0 {
            if i < body.len() {
                result = body[i];
            }
        } else {
            break;
        }
    }
    result
}

// ============================================================
// For Loop
// ============================================================

pub fn for_loop<T: DType + Send + Sync>(
    start: i64,
    end: i64,
    step: i64,
    body: &[T],
) -> T {
    // Simplified: iterate specified number of times
    let mut result = T::from_f32(0.0);
    let mut idx = 0;
    let mut i = start;
    while i < end {
        if idx < body.len() {
            result = body[idx];
        }
        idx += 1;
        i += step;
    }
    result
}

// ============================================================
// Loop Op - Placeholder   TODO: Implement
// ============================================================

pub struct LoopOp;

impl<T: DType + Send + Sync> Operator<T> for LoopOp {
    fn name(&self) -> &'static str {
        "loop"
    }
    fn forward(&self, inputs: &[&Tensor<T>], _attrs: &OpAttrs) -> Tensor<T> {
        // Simplified: return first input directly
        inputs[0].clone()
    }
    fn backward(
        &self,
        grad: &Tensor<T>,
        _inputs: &[&Tensor<T>],
        _attrs: &OpAttrs,
    ) -> Vec<Tensor<T>> {
        vec![grad.clone()]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_while_loop() {
        let condition = vec![1, 1, 1, 0, 1];
        let body = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let c = while_loop(&condition, &body);
        assert_eq!(c, 3.0);
    }

    #[test]
    fn test_for_loop() {
        let body = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let c = for_loop(0, 3, 1, &body);
        assert_eq!(c, 3.0);
    }
}
