// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu
//
// Memory management for inference execution
//
// This module provides:
// - Request-level scratch buffers (zero-allocation)
// - Memory topology detection
// - Buffer pool management

mod scratch;

pub use scratch::{RequestScratch, RequestScratchPool};
