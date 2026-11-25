// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

//! Memory management for inference
//!
//! This module provides:
//! - `arena`: Compute arena for zero-allocation temporary buffers
//! - `graph_alloc`: Graph allocator with lifetime tracking (llama.cpp style)
//! - `paged`: Paged KV cache for memory-efficient long context
//! - `topology`: System memory topology detection

pub mod arena;
pub mod graph_alloc;
pub mod paged;
pub mod topology;

pub use arena::{ComputeArena, calculate_work_size};
pub use graph_alloc::{
    ComputeGraph, GraphAllocator, GraphBuilder, LayerMemory, OpType,
};
