// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

#[cfg(target_arch = "aarch64")]
pub mod arm;

#[cfg(target_arch = "aarch64")]
pub mod gpu;

pub mod scalar;
pub mod tables;
