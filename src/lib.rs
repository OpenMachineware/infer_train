// src/lib.rs

pub mod ffi;
pub mod pytensor;
pub mod ir;
pub mod executor;
pub mod transform;
pub mod frontend;

use pyo3::prelude::*;

// ============================================================
// PyTorch 插件 (默认)
// ============================================================
#[cfg(feature = "torch")]
#[pymodule]
fn infer_train_torch(_py: Python, m: &Bound<PyModule>) -> PyResult<()> {
    // ---- 基础类型 ----
    m.add_class::<pytensor::PyTensor>()?;
    m.add_class::<pytensor::AdamState>()?;
    m.add_class::<pytensor::AdamWState>()?;
    m.add_function(wrap_pyfunction!(pytensor::sgd_update, m)?)?;

    // ---- IR 类型 ----
    m.add_class::<ir::serialize::PyModelFile>()?;
    m.add_class::<ir::serialize::PyDagGraph>()?;

    // ---- Hook 追踪器 ----
    m.add_class::<frontend::hook::HookTracer>()?;
    m.add_class::<frontend::hook::PyCfgGraph>()?;

    // ---- JIT Trace ----
    m.add_function(wrap_pyfunction!(frontend::jit_trace::trace_and_save, m)?)?;
    m.add_function(wrap_pyfunction!(frontend::jit_trace::test_trace_from_torch, m)?)?;
    m.add_function(wrap_pyfunction!(frontend::jit_trace::trace_with_weights_py, m)?)?;

    // ---- Executor ----
    m.add_class::<executor::PyExecutor>()?;

    // ---- 工具函数 ----
    m.add_function(wrap_pyfunction!(trace_from_torch_py, m)?)?;
    m.add_function(wrap_pyfunction!(execute_graph, m)?)?;
    m.add_function(wrap_pyfunction!(cfg_to_dag_py, m)?)?;
    m.add_function(wrap_pyfunction!(save_model_from_trace, m)?)?;
    m.add_function(wrap_pyfunction!(test_cfg_optimization, m)?)?;
    m.add_function(wrap_pyfunction!(test_full_pipeline, m)?)?;

    Ok(())
}

// ============================================================
// TensorFlow 插件
// ============================================================
#[cfg(feature = "tensorflow")]
#[pymodule]
fn infer_train_tf(_py: Python, m: &Bound<PyModule>) -> PyResult<()> {
    m.add_class::<pytensor::PyTensor>()?;
    m.add_class::<ir::serialize::PyModelFile>()?;
    m.add_class::<executor::PyExecutor>()?;
    Ok(())
}

// ============================================================
// JAX 插件
// ============================================================
#[cfg(feature = "jax")]
#[pymodule]
fn infer_train_jax(_py: Python, m: &Bound<PyModule>) -> PyResult<()> {
    m.add_class::<pytensor::PyTensor>()?;
    m.add_class::<ir::serialize::PyModelFile>()?;
    m.add_class::<executor::PyExecutor>()?;
    Ok(())
}

// ============================================================
// Python 工具函数
// ============================================================

#[pyfunction]
pub fn trace_from_torch_py(
    py: Python,
    model: &Bound<PyAny>,
    example_input: &Bound<PyAny>,
) -> PyResult<String> {
    let graph = frontend::jit_trace::trace_from_torch(py, model, example_input)?;
    Ok(format!("{:#?}", graph))
}

#[pyfunction]
pub fn execute_graph(py: Python, model: &Bound<PyAny>, x: &Bound<PyAny>) -> PyResult<Vec<f32>> {
    let graph = frontend::jit_trace::trace_from_torch(py, model, x)?;
    let mut executor = executor::Executor::new(graph);
    let input_tensor = ffi::Tensor::new_f32(&[], &[1, 10]);
    let result = executor.execute(&[input_tensor])
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))?;
    Ok(result[0].data_as_f32().to_vec())
}

#[pyfunction]
pub fn cfg_to_dag_py(cfg: &frontend::hook::PyCfgGraph) -> PyResult<ir::serialize::PyDagGraph> {
    let dag = transform::cfg_to_dag::CfgToDagConverter::convert(&cfg.inner)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))?;
    Ok(ir::serialize::PyDagGraph { inner: dag })
}

#[pyfunction]
pub fn save_model_from_trace(
    model_name: &str,
    cfg: &frontend::hook::PyCfgGraph,
    framework: &str,
    trainable: bool,
    path: &str,
) -> PyResult<()> {
    // 完整优化：CFG + DAG
    let mut cfg_clone = cfg.inner.clone();
    let dag = transform::FullOptimizer::optimize_full(&mut cfg_clone)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))?;

    let model_file = if trainable {
        ir::serialize::ModelFile::new_trainable(model_name, framework, dag, "adam", 0.001)
    } else {
        ir::serialize::ModelFile::new(model_name, framework, dag)
    };

    model_file.export(path)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))?;
    Ok(())
}

/// 测试 CFG 优化
#[pyfunction]
pub fn test_cfg_optimization() -> PyResult<String> {
    use crate::ir::cfg::CfgGraph;
    use crate::transform::CfgOptimizer;

    let mut cfg = CfgGraph::new("test_cfg");
    let entry = cfg.add_block("entry");
    let dead = cfg.add_block("dead");
    cfg.set_entry(entry);

    // 优化
    CfgOptimizer::optimize(&mut cfg)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))?;

    Ok(format!("CFG optimized, blocks: {}", cfg.blocks.len()))
}

/// 测试完整流程
#[pyfunction]
pub fn test_full_pipeline() -> PyResult<String> {
    use crate::ir::cfg::CfgGraph;
    use crate::transform::FullOptimizer;

    let mut cfg = CfgGraph::new("test_full");
    let entry = cfg.add_block("entry");
    cfg.set_entry(entry);

    // 完整优化
    let dag = FullOptimizer::optimize_full(&mut cfg)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))?;

    Ok(format!("Pipeline complete, DAG ops: {}", dag.ops.len()))
}
