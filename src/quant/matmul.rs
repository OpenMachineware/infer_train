// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

#[cfg(target_arch = "aarch64")]
pub mod fp16;
#[cfg(target_arch = "aarch64")]
pub mod fp32;
#[cfg(target_arch = "aarch64")]
pub mod k_quant;

#[cfg(target_arch = "aarch64")]
pub use fp16::*;
#[cfg(target_arch = "aarch64")]
pub use fp32::*;
#[cfg(target_arch = "aarch64")]
pub use k_quant::*;
