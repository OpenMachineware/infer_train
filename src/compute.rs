// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

// Compute graph and backend abstraction

pub mod backend;
pub mod cpu_backend;
pub mod device;
pub mod graph;
pub mod op;
pub mod storage;
pub mod tensor;

#[cfg(all(target_arch = "aarch64", feature = "metal"))]
pub mod metal_backend;

pub use backend::{Backend, BackendError, BackendFactory, BackendResult};
pub use cpu_backend::CpuBackend;
pub use device::Device;
pub use graph::{ComputeGraph, GraphBuilder};
pub use op::{OpNode, OpParams, OpType};
pub use storage::{CpuStorage, StorageEnum};
pub use tensor::Tensor;
