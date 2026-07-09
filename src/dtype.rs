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

use half::{bf16, f16};
use std::ops::{Add, Div, Mul, Neg, Sub};

// ============================================================
// DType Trait
// ============================================================

pub trait DType:
    Copy
    + Clone
    + Send
    + Sync
    + PartialEq
    + std::fmt::Debug
    + Default
    + Add<Output = Self>
    + Sub<Output = Self>
    + Mul<Output = Self>
    + Div<Output = Self>
    + Neg<Output = Self>
    + 'static
{
    fn to_f32(self) -> f32;
    fn from_f32(val: f32) -> Self;
    fn zero() -> Self;
    fn one() -> Self;
    fn dtype_name() -> &'static str;
    fn is_quantized() -> bool {
        false
    }
}

// ============================================================
// Type Implementations
// ============================================================

// ---- F32 ----
impl DType for f32 {
    fn to_f32(self) -> f32 {
        self
    }
    fn from_f32(val: f32) -> Self {
        val
    }
    fn zero() -> Self {
        0.0
    }
    fn one() -> Self {
        1.0
    }
    fn dtype_name() -> &'static str {
        "f32"
    }
}

// ---- F64 ----
impl DType for f64 {
    fn to_f32(self) -> f32 {
        self as f32
    }
    fn from_f32(val: f32) -> Self {
        val as f64
    }
    fn zero() -> Self {
        0.0
    }
    fn one() -> Self {
        1.0
    }
    fn dtype_name() -> &'static str {
        "f64"
    }
}

// ---- F16 ----
impl DType for f16 {
    fn to_f32(self) -> f32 {
        self.to_f32()
    }
    fn from_f32(val: f32) -> Self {
        f16::from_f32(val)
    }
    fn zero() -> Self {
        f16::from_f32(0.0)
    }
    fn one() -> Self {
        f16::from_f32(1.0)
    }
    fn dtype_name() -> &'static str {
        "f16"
    }
}

// ---- BF16 ----
impl DType for bf16 {
    fn to_f32(self) -> f32 {
        self.to_f32()
    }
    fn from_f32(val: f32) -> Self {
        bf16::from_f32(val)
    }
    fn zero() -> Self {
        bf16::from_f32(0.0)
    }
    fn one() -> Self {
        bf16::from_f32(1.0)
    }
    fn dtype_name() -> &'static str {
        "bf16"
    }
}

// ---- I8 (Quantized) ----
impl DType for i8 {
    fn to_f32(self) -> f32 {
        self as f32
    }
    fn from_f32(val: f32) -> Self {
        val as i8
    }
    fn zero() -> Self {
        0
    }
    fn one() -> Self {
        1
    }
    fn dtype_name() -> &'static str {
        "i8"
    }
    fn is_quantized() -> bool {
        true
    }
}

// ---- I64 ----
impl DType for i64 {
    fn to_f32(self) -> f32 {
        self as f32
    }
    fn from_f32(val: f32) -> Self {
        val as i64
    }
    fn zero() -> Self {
        0
    }
    fn one() -> Self {
        1
    }
    fn dtype_name() -> &'static str {
        "i64"
    }
}
