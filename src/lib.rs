pub mod ffi;
pub mod pytensor;
pub mod ir;
pub mod transform;
pub mod frontend;

use pyo3::prelude::*;

#[cfg(feature = "torch")]
#[pymodule]
fn infer_train_torch(_py: Python, m: &Bound<PyModule>) -> PyResult<()> {
    m.add_class::<pytensor::PyTensor>()?;
    m.add_class::<pytensor::AdamState>()?;
    m.add_class::<pytensor::AdamWState>()?;
    m.add_function(wrap_pyfunction!(pytensor::sgd_update, m)?)?;

    // JIT Trace 导出
    m.add_function(wrap_pyfunction!(frontend::jit_trace::trace_and_save, m)?)?;

    // 调试工具（开发者使用）
    m.add_function(wrap_pyfunction!(frontend::jit_trace::test_trace_from_torch, m)?)?;
    m.add_function(wrap_pyfunction!(frontend::jit_trace::trace_with_weights_py, m)?)?;

    // 模型加载 + 执行
    m.add_class::<ir::executor::PyExecutor>()?;
    m.add_class::<ir::serialize::PyModelFile>()?;
    Ok(())
}

#[cfg(feature = "tensorflow")]
#[pymodule]
fn infer_train_tf(_py: Python, m: &Bound<PyModule>) -> PyResult<()> {
    m.add_class::<pytensor::PyTensor>()?;
    Ok(())
}

#[cfg(feature = "jax")]
#[pymodule]
fn infer_train_jax(_py: Python, m: &Bound<PyModule>) -> PyResult<()> {
    m.add_class::<pytensor::PyTensor>()?;
    Ok(())
}

// 测试函数
#[pyfunction]
pub fn execute_graph(py: Python, model: &Bound<PyAny>, x: &Bound<PyAny>) -> PyResult<Vec<f32>> {
    // 1. 捕获图
    let graph = frontend::jit_trace::trace_from_torch(py, model, x)?;

    // 2. 执行图
    let mut executor = ir::Executor::new(graph);

    // 3. 将输入转为 Tensor
    let input_tensor = ffi::Tensor::new_f32(&[], &[1, 10]); // TODO: 从 PyAny 提取数据

    let result = executor.execute(&[input_tensor]).map_err(|e| {
        pyo3::exceptions::PyRuntimeError::new_err(e)
    })?;

    // 4. 提取数据
    Ok(result[0].data_as_f32().to_vec())
}

#[pyfunction]
pub fn trace_from_torch_py(
    py: Python,
    model: &Bound<PyAny>,
    example_input: &Bound<PyAny>,
) -> PyResult<String> {
    let graph = frontend::jit_trace::trace_from_torch(py, model, example_input)?;
    Ok(format!("{:#?}", graph))
}
