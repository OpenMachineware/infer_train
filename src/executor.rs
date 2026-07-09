// InferTrain - A Unified Inference and Training Engine
//
// Copyright (c) 2026 Jia Liu & InferTrain Contributors
// SPDX-License-Identifier: Apache-2.0
//
// This file is part of InferTrain.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at:
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

pub mod activation;
pub mod amp;
pub mod control;
pub mod executor;
pub mod index;
pub mod math;
pub mod memory_reuse;
pub mod nn;
pub mod parallel;
pub mod quantized;
pub mod scheduler;
pub mod tensor;
pub mod trainer;

pub use amp::{AmpConfig, AmpDtype, AmpGraphConverter};
pub use executor::{Executor, PyExecutor};
pub use memory_reuse::{MemoryConfig, MemoryPool};
pub use parallel::dispatch_op;
pub use scheduler::{Scheduler, SchedulerConfig};
pub use trainer::{
    AdamWOptimizerState, OptimizerState, OptimizerType, SGDOptimizerState,
    Trainer, TrainerConfig, TrainingState,
};
