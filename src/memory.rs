// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

//! Memory management for inference
//!
//! This module provides:
//! - `arena`: Compute arena for zero-allocation temporary buffers
//! - `paged`: Paged KV cache for memory-efficient long context
//! - `topology`: System memory topology detection

pub mod arena;
pub mod paged;
pub mod topology;

pub use arena::{ComputeArena, calculate_work_size};
