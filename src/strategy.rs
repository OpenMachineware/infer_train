// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

//! Execution strategies for heterogeneous compute.

mod cpu_plan;
mod execution_strategy;
mod single;

pub use cpu_plan::CpuExecutionPlan;
pub use execution_strategy::{
    ExecutionStrategy, PipelineStage, TransferStrategy,
};
pub use single::SingleDeviceStrategy;
