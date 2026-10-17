// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

//! Buffer management layer (Layer 1 of three-layer architecture)
//!
//! Responsibilities:
//! - Track buffer locations (Local GPU / Remote / Host)
//! - Manage buffer pool with reuse across tokens
//! - Handle transfers between devices (for heterogeneity)
//! - Prefetch support for expert scheduling

mod id;
mod location;
mod manager;
mod pool;

pub use id::BufferId;
pub use location::BufferLocation;
pub use manager::BufferManager;
pub use pool::BufferPool;
