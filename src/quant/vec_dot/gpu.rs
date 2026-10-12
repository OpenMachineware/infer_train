// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

#[cfg(all(target_arch = "aarch64", target_os = "macos"))]
pub mod metal;
pub mod shader_util;

#[cfg(not(all(target_arch = "aarch64", target_os = "macos")))]
pub mod metal_stub;
