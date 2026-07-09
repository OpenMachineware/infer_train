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

use pyo3::prelude::*;
use pyo3::types::{PyAny, PyList};

use crate::frontend::hook::HookTracer;
use crate::executor::PyExecutor;

/// Trace from PyTorch model and return DAG
#[pyfunction]
pub fn trace_model(
    _py: Python,
    model: Py<PyAny>,
    example_inputs: Vec<Py<PyAny>>,
) -> PyResult<Py<PyAny>> {
    let mut tracer = HookTracer::new();

    // Trace using HookTracer
    let dag = tracer.trace_to_dag(model, example_inputs)?;

    Ok(dag)
}

/// Trace from PyTorch model and export ITM
#[pyfunction]
#[pyo3(
    signature = (
        model,
        example_inputs,
        path,
        model_name=None,
        trainable=None
    )
)]
pub fn trace_and_export(
    _py: Python,
    model: Py<PyAny>,
    example_inputs: Vec<Py<PyAny>>,
    path: String,
    model_name: Option<String>,
    trainable: Option<bool>,
) -> PyResult<()> {
    let mut tracer = HookTracer::new();

    if let Some(name) = model_name {
        tracer.set_model_name(name);
    }

    let trainable = trainable.unwrap_or(false);

    tracer.trace_and_export_with_config(
        model,
        example_inputs,
        path,
        None,
        trainable,
        "adam".to_string(),
        0.001,
    )
}

/// Create trainable model from PyTorch model
#[pyfunction]
#[pyo3(
    signature = (
        model,
        example_inputs,
        path,
        model_name=None,
        optimizer=None,
        learning_rate=None
    )
)]
pub fn trace_trainable(
    _py: Python,
    model: Py<PyAny>,
    example_inputs: Vec<Py<PyAny>>,
    path: String,
    model_name: Option<String>,
    optimizer: Option<String>,
    learning_rate: Option<f32>,
) -> PyResult<()> {
    let mut tracer = HookTracer::new();

    if let Some(name) = model_name {
        tracer.set_model_name(name);
    }

    tracer.trace_and_export_trainable(
        model,
        example_inputs,
        path,
        None,
        optimizer.unwrap_or("adam".to_string()),
        learning_rate.unwrap_or(0.001),
    )
}

/// Load ITM model and execute inference
#[pyfunction]
pub fn load_and_infer(
    py: Python,
    model_path: String,
    inputs: Vec<Py<PyAny>>,
) -> PyResult<Vec<Py<PyAny>>> {
    use crate::ir::serialize::PyModelFile;
    let model_file = PyModelFile::load(&model_path)?;
    let dag = model_file.get_graph()?;

    // Create executor
    let mut executor = PyExecutor::new(dag)?;

    let list = PyList::new(py, inputs.iter().map(|x| x.bind(py)))?.unbind();
    let result = executor.execute(list)?;

    Ok(result)
}
