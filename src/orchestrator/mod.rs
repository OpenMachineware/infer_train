// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

//! Forward Orchestrator (Layer 3)
//!
//! Coordinates forward pass execution across GPU and CPU backends.
//!
//! # Architecture
//!
//! - **architecture**: Detects model architecture (Standard/MQA/MLA/MoE)
//! - **forward**: Main forward pass orchestrator
//!
//! # Design
//!
//! The orchestrator sits on top of BufferManager (Layer 1) and OperationExecutor (Layer 2):
//!
//! ```text
//! ┌─────────────────────────────────────┐
//! │   ForwardOrchestrator (Layer 3)     │
//! │  - Architecture detection           │
//! │  - Path selection (GPU vs CPU)      │
//! │  - KV cache lifecycle               │
//! └────────────────┬────────────────────┘
//!                  │
//! ┌────────────────▼────────────────────┐
//! │  OperationExecutor (Layer 2)        │
//! │  - Execute operations               │
//! │  - Device selection                 │
//! │  - Fallback logic                   │
//! └────────────────┬────────────────────┘
//!                  │
//! ┌────────────────▼────────────────────┐
//! │    BufferManager (Layer 1)          │
//! │  - Buffer allocation                │
//! │  - Location tracking                │
//! │  - Transfer queue                   │
//! └─────────────────────────────────────┘
//! ```

pub mod architecture;
pub mod forward;

pub use architecture::{Architecture, AttentionType, FfnType};
pub use forward::ForwardOrchestrator;
