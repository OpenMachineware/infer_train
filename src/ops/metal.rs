// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

// Metal GPU operators for Apple Silicon

pub mod activation;
pub mod flash_attn;
pub mod kv_cache;
pub mod rms_norm;
pub mod rope;
pub mod softmax;

pub use activation::ActivationMetalContext;
pub use flash_attn::FlashAttnMetalContext;
pub use kv_cache::{GpuKVCache, GpuKVFormat};
pub use rms_norm::RmsNormMetalContext;
pub use rope::RopeMetalContext;
pub use softmax::SoftmaxMetalContext;
