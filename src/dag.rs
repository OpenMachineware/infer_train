// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

//! DAG (Directed Acyclic Graph) - IR for high-level optimization
//!
//! Core philosophy: Load Model → Build DAG → Traverse DAG for Execution
//!
//! This module implements the intermediate representation (IR) that separates
//! execution planning from execution itself, enabling systematic optimization.

pub mod builder;
pub mod executor;
pub mod graph;
pub mod memory_ranges;
pub mod optimizer;

pub use builder::DagBuilder;
pub use executor::DagExecutor;
pub use graph::{DagEdge, DagNode, Operation, OptimizedDag};
pub use optimizer::DagOptimizer;
