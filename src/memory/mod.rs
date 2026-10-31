// SPDX-License-Identifier: AGPL-3.0-or-later
//
// Memory management for inference execution
//
// This module provides:
// - Request-level scratch buffers (zero-allocation)
// - Memory topology detection
// - Buffer pool management

mod scratch;

pub use scratch::{RequestScratch, RequestScratchPool};
