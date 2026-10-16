// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

//! Compute graph infrastructure for unified CPU/GPU execution.
//!
//! - Storage abstraction for CPU/Metal backends
//! - ComputeGraph for operation scheduling
//! - Backend trait for execution
//!
//! The goal is to minimize kernel dispatch overhead by batching
//! operations and reducing synchronization points.

pub mod backend;
pub mod graph;
pub mod storage;

#[cfg(all(target_arch = "aarch64", feature = "metal"))]
pub mod backend_metal;

pub use backend::{Backend, CpuBackend, ExecutionContext};
pub use graph::{ComputeGraph, GraphBuilder, OpNode, OpType};
pub use storage::{Device, Storage};

#[cfg(all(target_arch = "aarch64", feature = "metal"))]
pub use backend_metal::MetalBackend;
