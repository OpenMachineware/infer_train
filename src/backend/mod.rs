// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

//! Backend abstraction for unified execution across devices.
//!
//! This module provides a hardware-agnostic compute interface supporting:
//! - ARM CPU (NEON)
//! - x86 CPU (AVX2/AVX512)
//! - RISC-V CPU (V extension)
//! - Apple Metal
//! - NVIDIA CUDA
//! - AMD ROCm

mod device;
mod memory;
pub mod operator;
mod trait_impl;

pub use device::{Device, DeviceId, DeviceInfo, DeviceType};
pub use memory::{
    Bandwidth, MemoryLink, MemoryRegion, MemoryRegionId, MemoryTopology,
    TransferType,
};
pub use operator::{Cost, Operator, OperatorId, OperatorResult};
pub use trait_impl::{Backend, BackendRegistry, DataType, TensorRef};

#[cfg(target_arch = "aarch64")]
mod arm;

#[cfg(target_arch = "x86_64")]
mod x86;

#[cfg(all(target_arch = "aarch64", feature = "metal"))]
mod metal;

#[cfg(feature = "cuda")]
mod cuda;
