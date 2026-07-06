// src/frontend/jit_trace.rs

use std::collections::HashMap;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyBytes, PyList};

use crate::ir::dag::{DagGraph, DataType, TensorType, AttrValue};
use crate::ir::serialize::{ModelFile, ModelHeader, PyModelFile};

// ============================================================
// 权重信息
// ============================================================
#[derive(Debug, Clone)]
pub struct WeightInfo {
    pub data: Vec<u8>,
    pub dtype: DataType,
    pub shape: Vec<i64>,
}

// ============================================================
// 主函数：带权重 trace
// ============================================================
pub fn trace_from_torch_with_weights(
    py: Python,
    model: &Bound<PyAny>,
    example_input: &Bound<PyAny>,
    weights: &HashMap<String, WeightInfo>,
) -> PyResult<DagGraph> {
    let mut graph = trace_from_torch(py, model, example_input)?;

    // 收集 conv2d 节点的权重输入
    let mut weight_infos: Vec<(u64, &WeightInfo)> = Vec::new();

    for (op_id, op) in &graph.ops {
        if op.op_type == "conv2d" {
            // conv2d inputs: [input, weight, bias, stride, padding, dilation, groups]
            if op.inputs.len() >= 3 {
                let weight_id = op.inputs[1];
                let bias_id = op.inputs[2];

                // 从 weights 中按名称匹配
                for (name, weight) in weights {
                    if name.contains("weight") {
                        weight_infos.push((weight_id, weight));
                    } else if name.contains("bias") {
                        weight_infos.push((bias_id, weight));
                    }
                }
            }
        }
    }

    for (value_id, weight) in weight_infos {
        graph.constants.insert(value_id, weight.data.clone());
    }

    Ok(graph)
}

// ============================================================
// 测试函数：接收 Python 字典（测试通过后删除）
// ============================================================
#[pyfunction]
pub fn test_trace_with_weights(
    py: Python,
    model: &Bound<PyAny>,
    example_input: &Bound<PyAny>,
    weights: &Bound<PyDict>,
) -> PyResult<String> {
    let mut weight_map = HashMap::new();

    for item in weights.iter() {
        let key = item.0;
        let value = item.1;

        let name: String = key.extract()?;
        let value_dict = value.downcast::<PyDict>()?;

        // 先处理 PyResult，再处理 Option
        let data_bytes: Vec<u8> = value_dict
            .get_item("data")?
            .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyKeyError, _>("missing 'data'"))?
            .extract()?;

        let dtype_str: String = value_dict
            .get_item("dtype")?
            .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyKeyError, _>("missing 'dtype'"))?
            .extract()?;

        let shape_list: Vec<i64> = value_dict
            .get_item("shape")?
            .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyKeyError, _>("missing 'shape'"))?
            .extract()?;

        let dtype = parse_python_dtype(&dtype_str);
        weight_map.insert(name, WeightInfo {
            data: data_bytes,
            dtype,
            shape: shape_list,
        });
    }

    let graph = trace_from_torch_with_weights(py, model, example_input, &weight_map)?;
    Ok(format!("{:#?}", graph))
}

fn parse_python_dtype(dtype: &str) -> DataType {
    if dtype.contains("float32") { DataType::F32 }
    else if dtype.contains("float64") { DataType::F64 }
    else if dtype.contains("float16") { DataType::F16 }
    else if dtype.contains("bfloat16") { DataType::BF16 }
    else if dtype.contains("int8") { DataType::I8 }
    else if dtype.contains("int32") { DataType::I32 }
    else if dtype.contains("int64") { DataType::I64 }
    else { DataType::F32 }
}

// ============================================================
// 从 Python 模型提取权重
// ============================================================
fn extract_weights_from_model(
    py: Python,
    model: &Bound<PyAny>,
) -> PyResult<Vec<(String, Vec<u8>, String, Vec<i64>)>> {
    let mut result = Vec::new();

    let named_params = model.call_method("named_parameters", (), None)?;
    let params_list = PyList::new(py, named_params.iter())?;

    for item in params_list.iter() {
        // 不能用 item?，直接使用 item
        let name = item.get_item(0)?.extract::<String>()?;
        let param = item.get_item(1)?;

        let data = param.call_method("detach", (), None)?;
        let data = data.call_method("numpy", (), None)?;
        let data = data.call_method("tobytes", (), None)?;
        let bytes_data: Vec<u8> = data.extract()?;

        let dtype = param.getattr("dtype")?.str()?.to_string();
        let shape = param.getattr("shape")?;
        let shape_list: Vec<i64> = shape.extract()?;

        result.push((name, bytes_data, dtype, shape_list));
    }

    Ok(result)
}

// ============================================================
// Python 导出的 trace_with_weights
// ============================================================
#[pyfunction]
pub fn trace_with_weights_py(
    py: Python,
    model: &Bound<PyAny>,
    example_input: &Bound<PyAny>,
    weights: &Bound<PyDict>,
) -> PyResult<String> {
    let mut weight_map = HashMap::new();

    for item in weights.iter() {
        let key = item.0;
        let value = item.1;

        let name: String = key.extract()?;
        let value_dict = value.downcast::<PyDict>()?;

        let data_bytes: Vec<u8> = value_dict
            .get_item("data")?
            .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyKeyError, _>("missing 'data'"))?
            .extract()?;

        let dtype_str: String = value_dict
            .get_item("dtype")?
            .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyKeyError, _>("missing 'dtype'"))?
            .extract()?;

        let shape_list: Vec<i64> = value_dict
            .get_item("shape")?
            .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyKeyError, _>("missing 'shape'"))?
            .extract()?;

        let dtype = parse_python_dtype(&dtype_str);
        weight_map.insert(name, WeightInfo {
            data: data_bytes,
            dtype,
            shape: shape_list,
        });
    }

    let graph = trace_from_torch_with_weights(py, model, example_input, &weight_map)?;
    Ok(format!("{:#?}", graph))
}

// ============================================================
// 导出模型文件
// ============================================================
// src/frontend/jit_trace.rs

#[pyfunction]
pub fn trace_and_save(
    py: Python,
    model: &Bound<PyAny>,
    example_input: &Bound<PyAny>,
    weights: &Bound<PyDict>,
    path: &str,
) -> PyResult<()> {
    // 复制 trace_with_weights_py 中的权重提取逻辑
    let mut weight_map = HashMap::new();

    for item in weights.iter() {
        let key = item.0;
        let value = item.1;

        let name: String = key.extract()?;
        let value_dict = value.downcast::<PyDict>()?;

        let data_bytes: Vec<u8> = value_dict
            .get_item("data")?
            .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyKeyError, _>("missing 'data'"))?
            .extract()?;

        let dtype_str: String = value_dict
            .get_item("dtype")?
            .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyKeyError, _>("missing 'dtype'"))?
            .extract()?;

        let shape_list: Vec<i64> = value_dict
            .get_item("shape")?
            .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyKeyError, _>("missing 'shape'"))?
            .extract()?;

        let dtype = parse_python_dtype(&dtype_str);
        weight_map.insert(name, WeightInfo {
            data: data_bytes,
            dtype,
            shape: shape_list,
        });
    }

    let graph = trace_from_torch_with_weights(py, model, example_input, &weight_map)?;

    // 创建 ModelFile 并保存
    let model_file = ModelFile::new("traced_model", "torch", graph);
    model_file.export(path)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e))
}

// ============================================================
// 主函数：从 PyTorch JIT 图捕获 IR
// ============================================================
pub fn trace_from_torch(
    py: Python,
    model: &Bound<PyAny>,
    example_input: &Bound<PyAny>,
) -> PyResult<DagGraph> {
    let torch = PyModule::import(py, "torch")?;
    let jit = torch.getattr("jit")?;

    // 1. 先 script
    let scripted = jit.call_method("script", (model,), None)?;

    // 2. 再 freeze（需要 eval 模式）
    // 注意：如果模型不是 eval 模式，freeze 会报错
    // 所以需要用户在调用前 model.eval()
    let frozen = jit.call_method("freeze", (scripted,), None)?;

    // 3. 获取图
    let graph = frozen.getattr("graph")?;
    let graph_str = graph.call_method("__str__", (), None)?.extract::<String>()?;
    // println!("Graph str:\n{}", graph_str);

    // let traced = jit.call_method("trace", (model, example_input), None)?;
    // let graph = traced.getattr("graph")?;
    // let graph_str = graph.call_method("__str__", (), None)?.extract::<String>()?;

    let mut ir_graph = DagGraph::new("traced_model");
    let mut value_map: HashMap<String, u64> = HashMap::new();
    let mut last_output = String::new();

    // 解析输入
    let input_names = parse_inputs(&graph_str);
    for (name, dtype, shape) in input_names {
        let dt = parse_dtype(&dtype);
        let ty = TensorType { dtype: dt, shape };
        let id = ir_graph.add_value(&name, ty);
        value_map.insert(name.clone(), id);
        ir_graph.set_inputs(vec![id]);
        last_output = name;
    }

    // 如果没解析到输入，创建一个默认输入
    if ir_graph.inputs.is_empty() {
        let id = ir_graph.add_value("input", TensorType {
            dtype: DataType::F32,
            shape: vec![1, 10],
        });
        value_map.insert("input".to_string(), id);
        value_map.insert("%x".to_string(), id);
        ir_graph.set_inputs(vec![id]);
    }

    let lines: Vec<&str> = graph_str.lines().collect();

    let mut constant_map: HashMap<String, AttrValue> = HashMap::new();

    for line in lines {
        let line = line.trim();
        if line.is_empty() || line.starts_with("return") {
            continue;
        }

        // 跳过 GetAttr
        if line.contains("prim::GetAttr") {
            continue;
        }

        // 处理 CallMethod
        if line.contains("prim::CallMethod") {
            if let Some((out_name, _)) = parse_output_name(line) {
                let method_name = extract_call_method_name(line);
                if method_name == "forward" {
                    let inputs = extract_inputs(line);
                    let input_ids = resolve_inputs(&inputs, &value_map);

                    if !input_ids.is_empty() {
                        // 判断是 linear 还是 conv2d
                        // 查看第一个输入（%conv 或 %fc）的类型
                        let op_type = if line.contains("Conv2d") || line.contains("conv") {
                            "conv2d"
                        } else {
                            "linear"
                        };

                        let out_id = create_value(&mut ir_graph, &out_name, DataType::F32, vec![1, 16, 16, 16]);
                        value_map.insert(out_name.clone(), out_id);
                        let attrs = HashMap::new();
                        ir_graph.add_op(op_type, input_ids, vec![out_id], attrs);
                        last_output = out_name;
                    }
                }
            }
            continue;
        }

        // 处理 Constant
        if line.contains("prim::Constant") {
            if let Some((name, value)) = parse_constant(line) {
                constant_map.insert(name.clone(), value);

                // 添加 Value 到 value_map（让 resolve_inputs 能找到）
                let ty = TensorType {
                    dtype: DataType::F32,
                    shape: vec![],
                };
                let id = ir_graph.add_value(&name, ty);
                value_map.insert(name, id);
            }
            continue;
        }

        // 处理 aten:: 算子
        if let Some((out_name, rest_line)) = parse_output_name(line) {
            let op_type = parse_aten_op(&rest_line);
            if op_type != "unknown" {
                // println!("conv2d rest_line: {}", rest_line);
                let inputs = extract_inputs(&rest_line);
                // println!("conv2d inputs: {:?}", inputs);
                let input_ids = resolve_inputs(&inputs, &value_map);

                // 创建输出 Value
                let out_id = create_value(&mut ir_graph, &out_name, DataType::F32, vec![1, 10]);
                value_map.insert(out_name.clone(), out_id);

                // 提取属性
                let mut attrs = HashMap::new();

                // ============================================================
                // 特殊处理：conv2d 从 constant_map 提取属性
                // ============================================================
                // println!("constant_map keys: {:?}", constant_map.keys().collect::<Vec<_>>());
                if op_type == "conv2d" && inputs.len() >= 7 {
                    // 用 trim 处理查找 key
                    let key_stride = inputs[3].trim();
                    let key_padding = inputs[4].trim();
                    let key_dilation = inputs[5].trim();
                    let key_groups = inputs[6].trim();

                    if let Some(attr) = constant_map.get(key_stride) {
                        attrs.insert("stride".to_string(), attr.clone());
                    }
                    if let Some(attr) = constant_map.get(key_padding) {
                        attrs.insert("padding".to_string(), attr.clone());
                    }
                    if let Some(attr) = constant_map.get(key_dilation) {
                        attrs.insert("dilation".to_string(), attr.clone());
                    }
                    if let Some(attr) = constant_map.get(key_groups) {
                        attrs.insert("groups".to_string(), attr.clone());
                    }
                }

                // println!("constant_map: {:?}", constant_map);

                if !input_ids.is_empty() {
                    ir_graph.add_op(op_type, input_ids, vec![out_id], attrs);
                }
                last_output = out_name;
            }
        }
    }

    // 设置输出
    if !ir_graph.outputs.is_empty() {
        // 保留已有输出
    } else {
        // 如果没有输出，用最后一个
        if let Some(id) = value_map.get(&last_output) {
            ir_graph.set_outputs(vec![*id]);
        }
    }

    Ok(ir_graph)
}

fn parse_constant(line: &str) -> Option<(String, AttrValue)> {
    if let Some(name_start) = line.find('%') {
        let rest = &line[name_start + 1..];
        let name_end = rest.find(':').unwrap_or(0);
        let name = format!("%{}", &rest[..name_end]).trim().to_string();

        if let Some(value_start) = line.find("[value=") {
            let value_str = &line[value_start + 7..];
            let mut depth = 0;
            let mut value_end = 0;
            for (i, ch) in value_str.char_indices() {
                if ch == '[' {
                    depth += 1;
                } else if ch == ']' {
                    if depth == 0 {
                        value_end = i;
                        break;
                    }
                    depth -= 1;
                }
            }
            if value_end > 0 {
                let raw = &value_str[..value_end].trim();

                // 处理 <Tensor>
                if raw.starts_with('<') && raw.ends_with('>') {
                    return Some((name, AttrValue::String("Tensor".to_string())));
                }
                // 处理数组: [2, 2]
                else if raw.starts_with('[') && raw.ends_with(']') {
                    let inner = &raw[1..raw.len()-1];
                    let values: Vec<i64> = inner
                        .split(',')
                        .filter_map(|s| s.trim().parse::<i64>().ok())
                        .collect();
                    return Some((name, AttrValue::IntList(values)));
                }
                // 处理整数: 1
                else if let Ok(v) = raw.parse::<i64>() {
                    return Some((name, AttrValue::Int(v)));
                }
            }
        }
    }
    None
}

// ============================================================
// 测试函数（Python 调用）
// 勿删，用于调试（看 IR 结构，不需要权重）
// ============================================================
#[pyfunction]
pub fn test_trace_from_torch(
    py: Python,
    model: &Bound<PyAny>,
    example_input: &Bound<PyAny>,
) -> PyResult<String> {
    let graph = trace_from_torch(py, model, example_input)?;
    Ok(format!("{:#?}", graph))
}

// ============================================================
// 输入解析
// ============================================================
fn parse_inputs(s: &str) -> Vec<(String, String, Vec<i64>)> {
    let mut result = Vec::new();
    // graph(%self.1 : __torch__.MyModel, %x : Float(1, 10, ...))
    if let Some(start) = s.find('(') {
        if let Some(end) = s.find(')') {
            let args = &s[start + 1..end];
            for part in args.split(',') {
                let part = part.trim();
                if let Some(name_pos) = part.find('%') {
                    let name = part[name_pos..].split(':').next().unwrap_or("").trim();
                    if name.is_empty() || name == "%self.1" {
                        continue;
                    }
                    // 提取 dtype 和 shape
                    let dtype = "Float";
                    let shape = vec![1, 10];
                    if !name.is_empty() {
                        result.push((name.to_string(), dtype.to_string(), shape));
                    }
                }
            }
        }
    }
    result
}

// ============================================================
// 属性提取
// ============================================================
fn extract_attrs(s: &str) -> HashMap<String, AttrValue> {
    let mut attrs = HashMap::new();

    // 从 JIT 节点中提取属性
    // aten::conv2d(%input, %weight, %bias, %stride, %padding, %dilation, %groups)
    // 这些参数在 JIT 中是以 %name 形式传递的，需要从上下文中解析

    // 简化：提取常量属性
    if let Some(start) = s.find('[') {
        if let Some(end) = s.rfind(']') {
            let attr_str = &s[start + 1..end];
            for part in attr_str.split(',') {
                let part = part.trim();
                if let Some(eq) = part.find('=') {
                    let key = part[..eq].trim();
                    let val = part[eq + 1..].trim();
                    if let Ok(v) = val.parse::<i64>() {
                        attrs.insert(key.to_string(), AttrValue::Int(v));
                    } else if let Ok(v) = val.parse::<f64>() {
                        attrs.insert(key.to_string(), AttrValue::Float(v));
                    } else {
                        attrs.insert(key.to_string(), AttrValue::String(val.to_string()));
                    }
                }
            }
        }
    }

    attrs
}

// ============================================================
// 值解析
// ============================================================
fn resolve_inputs(inputs: &[String], value_map: &HashMap<String, u64>) -> Vec<u64> {
    let mut result = Vec::new();
    for name in inputs {
        if let Some(&id) = value_map.get(name) {
            result.push(id);
        } else if let Some(&id) = value_map.get(&format!("%{}", name)) {
            result.push(id);
        }
    }
    result
}

fn create_value(
    graph: &mut DagGraph,
    name: &str,
    dtype: DataType,
    shape: Vec<i64>,
) -> u64 {
    let ty = TensorType { dtype, shape };
    graph.add_value(name, ty)
}

// ============================================================
// 解析算子类型
// ============================================================
fn parse_aten_op(s: &str) -> &'static str {
    if s.contains("aten::add") { return "add"; }
    if s.contains("aten::sub") { return "sub"; }
    if s.contains("aten::mul") { return "mul"; }
    if s.contains("aten::div") { return "div"; }
    if s.contains("aten::matmul") || s.contains("aten::mm") { return "matmul"; }
    if s.contains("aten::relu") { return "relu"; }
    if s.contains("aten::sigmoid") { return "sigmoid"; }
    if s.contains("aten::tanh") { return "tanh"; }
    if s.contains("aten::softmax") { return "softmax"; }
    if s.contains("aten::conv2d") || s.contains("aten::_convolution") { return "conv2d"; }
    if s.contains("aten::max_pool2d") { return "maxpool2d"; }
    if s.contains("aten::avg_pool2d") { return "avgpool2d"; }
    if s.contains("aten::batch_norm") { return "batchnorm2d"; }
    if s.contains("aten::layer_norm") { return "layernorm"; }
    if s.contains("aten::linear") { return "linear"; }
    if s.contains("aten::reshape") || s.contains("aten::view") { return "reshape"; }
    if s.contains("aten::transpose") { return "transpose"; }
    if s.contains("aten::cat") { return "cat"; }
    if s.contains("aten::dropout") { return "dropout"; }
    "unknown"
}

// ============================================================
// 通用辅助函数
// ============================================================
fn parse_output_name(line: &str) -> Option<(String, String)> {
    if let Some(pos) = line.find('=') {
        let left = line[..pos].trim();
        let right = line[pos + 1..].trim();
        if let Some(name_pos) = left.find('%') {
            let name = left[name_pos..].split(':').next().unwrap_or("").trim();
            if !name.is_empty() {
                return Some((name.to_string(), right.to_string()));
            }
        }
    }
    None
}

fn extract_inputs(s: &str) -> Vec<String> {
    if let Some(start) = s.find('(') {
        if let Some(end) = s.rfind(')') {
            let args = &s[start + 1..end];
            return args
                .split(',')
                .map(|x| x.trim().to_string())
                .filter(|x| !x.is_empty() && !x.contains("="))
                .collect();
        }
    }
    Vec::new()
}

fn extract_call_method_name(s: &str) -> String {
    if let Some(start) = s.find("[name=\"") {
        if let Some(end) = s[start + 7..].find('"') {
            return s[start + 7..start + 7 + end].to_string();
        }
    }
    "unknown".to_string()
}

fn parse_dtype(s: &str) -> DataType {
    if s.contains("Float") { DataType::F32 }
    else if s.contains("Double") { DataType::F64 }
    else if s.contains("Half") { DataType::F16 }
    else if s.contains("BFloat16") { DataType::BF16 }
    else if s.contains("Int") { DataType::I32 }
    else { DataType::F32 }
}
