use pyo3::prelude::*;
// use pyo3::types::PyTuple;
use pyo3::types::{PyAny, PyList};

use crate::frontend::hook::HookTracer;
// use crate::ir::dag::DagGraph;
// use crate::ir::serialize::PyDagGraph;
use crate::executor::PyExecutor;

/// 从 PyTorch 模型追踪并返回 DAG
#[pyfunction]
pub fn trace_model(
    _py: Python,
    model: Py<PyAny>,
    example_inputs: Vec<Py<PyAny>>,
) -> PyResult<Py<PyAny>> {
    let mut tracer = HookTracer::new();

    // 使用 HookTracer 追踪
    let dag = tracer.trace_to_dag(model, example_inputs)?;

    Ok(dag)
}

/// 从 PyTorch 模型追踪并导出 ITM
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

/// 从 PyTorch 模型创建可训练模型
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

/// 加载 ITM 模型并执行推理
#[pyfunction]
pub fn load_and_infer(
    py: Python,
    model_path: String,
    inputs: Vec<Py<PyAny>>,
) -> PyResult<Vec<Py<PyAny>>> {
    use crate::ir::serialize::PyModelFile;
    let model_file = PyModelFile::load(&model_path)?;
    let dag = model_file.get_graph()?;

    // 创建执行器
    let mut executor = PyExecutor::new(dag)?;

    let list = PyList::new(py, inputs.iter().map(|x| x.bind(py)))?.unbind();
    let result = executor.execute(list)?;

    Ok(result)
}
