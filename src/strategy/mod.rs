// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

//! Execution strategies for heterogeneous compute.

mod cpu_plan;
mod single;

pub use cpu_plan::CpuExecutionPlan;
pub use single::SingleDeviceStrategy;
