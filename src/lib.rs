// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

// Allow unsafe operations in unsafe functions without explicit blocks (Rust 2024)
#![allow(unsafe_op_in_unsafe_fn)]

pub mod inference;
pub mod model;
pub mod ops;
pub mod platform;
pub mod quant;
