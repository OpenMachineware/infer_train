// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

#[cfg(all(target_arch = "aarch64", feature = "metal"))]
pub mod device;

#[cfg(all(target_arch = "aarch64", feature = "metal"))]
pub mod metal;

#[cfg(all(target_arch = "aarch64", feature = "metal"))]
pub use device::MetalDevice;
#[cfg(all(target_arch = "aarch64", feature = "metal"))]
pub use metal::MetalContext;

pub mod shader_util;

#[cfg(not(all(target_arch = "aarch64", feature = "metal")))]
pub mod metal_stub;
