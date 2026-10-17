// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

//! Operation execution layer (Layer 2 of three-layer architecture)
//!
//! Responsibilities:
//! - Execute compute operations across backends (CPU/GPU/Remote)
//! - Handle cross-device data dependencies
//! - Provide automatic fallback to CPU for unsupported formats
//! - Support all quantization formats and model architectures

mod executor;
mod op;

pub use executor::OperationExecutor;
pub use op::Op;
