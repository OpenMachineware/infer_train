// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

// Allow unsafe operations in unsafe functions without explicit blocks (Rust 2024)
#![allow(unsafe_op_in_unsafe_fn)]

pub mod buffer;
pub mod compute;
pub mod dag;
pub mod executor;
pub mod inference;
pub mod model;
pub mod ops;
pub mod orchestrator;
pub mod platform;
pub mod quant;
pub mod tokenize;
