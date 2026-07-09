// src/transform/constant_fold.rs

use crate::ir::dag::{AttrValue, DagGraph, DataType};
use std::collections::HashMap;

pub struct ConstantFoldingPass;

impl ConstantFoldingPass {
    pub fn apply(graph: &mut DagGraph) -> bool {
        let mut changed = false;
        let mut to_remove = Vec::new();
        let mut replacements: HashMap<u64, u64> = HashMap::new();

        let op_ids: Vec<u64> = graph.ops.keys().cloned().collect();

        for &op_id in &op_ids {
            let op = match graph.ops.get(&op_id) {
                Some(o) => o.clone(),
                None => continue,
            };

            // ============================================================
            // 量化传播 - 消除冗余 Cast
            // ============================================================
            if let Some(repl) = Self::propagate_quantization(graph, &op) {
                replacements.insert(op.outputs[0], repl);
                to_remove.push(op_id);
                changed = true;
                continue;
            }

            // ============================================================
            // 常量折叠
            // ============================================================
            if let Some(result) = Self::fold_constant_op(graph, &op) {
                let out_id = op.outputs[0];
                graph.constants.insert(out_id, result.0);
                if let Some(value) = graph.values.get_mut(&out_id) {
                    value.ty.shape = result.1;
                    value.ty.dtype = result.2;
                }
                to_remove.push(op_id);
                changed = true;
                continue;
            }
        }

        // 应用替换
        for (old_id, new_id) in &replacements {
            Self::apply_replacement(graph, *old_id, *new_id);
        }

        for id in to_remove {
            graph.ops.remove(&id);
        }

        changed
    }

    // ============================================================
    // 量化传播
    // ============================================================

    fn propagate_quantization(
        graph: &DagGraph,
        op: &crate::ir::dag::Op,
    ) -> Option<u64> {
        match op.op_type.as_str() {
            // ---------- Dequantize ----------
            "dequantize" => {
                if op.inputs.len() >= 1 {
                    let input_id = op.inputs[0];

                    if let Some(producer) =
                        Self::get_producer_op(graph, input_id)
                    {
                        if producer.op_type == "quantize" {
                            if producer.inputs.len() >= 1 {
                                return Some(producer.inputs[0]);
                            }
                        }
                    }

                    if graph.constants.contains_key(&input_id) {
                        return Some(input_id);
                    }
                }
                // 没匹配到，返回 None
                return None;
            }

            // ---------- Quantize ----------
            "quantize" => {
                if op.inputs.len() >= 1 {
                    let input_id = op.inputs[0];

                    if let Some(producer) =
                        Self::get_producer_op(graph, input_id)
                    {
                        if producer.op_type == "dequantize" {
                            if producer.inputs.len() >= 1 {
                                return Some(producer.inputs[0]);
                            }
                        }
                    }

                    if graph.constants.contains_key(&input_id) {
                        return None;
                    }
                }
                return None;
            }

            // ---------- Cast ----------
            "cast" => {
                if op.inputs.len() >= 1 {
                    let input_id = op.inputs[0];
                    if let Some(producer) =
                        Self::get_producer_op(graph, input_id)
                    {
                        if producer.op_type == "cast" {
                            // 两个 Cast 合并为一个
                            if producer.inputs.len() >= 1 {
                                // 直接返回 producer 的输入（相当于跳过中间 Cast）
                                // 因为最终 Cast 会决定类型，中间 Cast 可以省略
                                return Some(producer.inputs[0]);
                            }
                        }
                    }
                }
                return None;
            }

            _ => None,
        }
    }

    // ============================================================
    // 常量折叠
    // ============================================================
    fn fold_constant_op(
        graph: &DagGraph,
        op: &crate::ir::dag::Op,
    ) -> Option<(Vec<u8>, Vec<i64>, DataType)> {
        let mut all_constant = true;
        let mut const_inputs = Vec::new();
        let mut const_shapes = Vec::new();
        let mut const_dtypes = Vec::new();

        for &in_id in &op.inputs {
            if let Some(data) = graph.constants.get(&in_id) {
                const_inputs.push(data.clone());
                if let Some(value) = graph.values.get(&in_id) {
                    const_shapes.push(value.ty.shape.clone());
                    const_dtypes.push(value.ty.dtype);
                } else {
                    all_constant = false;
                    break;
                }
            } else {
                all_constant = false;
                break;
            }
        }

        if !all_constant || op.inputs.is_empty() {
            return None;
        }

        let dtype = const_dtypes[0];
        let all_same_dtype = const_dtypes.iter().all(|&d| d == dtype);
        if !all_same_dtype {
            return None;
        }

        // ============================================================
        // 量化常量折叠
        // ============================================================

        if op.op_type == "quantize" {
            // 量化常量
            if let Some(scale) = op.attrs.get("scale") {
                let scale_val = match scale {
                    AttrValue::Float(f) => *f as f32,
                    AttrValue::Int(i) => *i as f32,
                    _ => 1.0,
                };
                let zero_point = op
                    .attrs
                    .get("zero_point")
                    .and_then(|v| match v {
                        AttrValue::Int(i) => Some(*i as f32),
                        AttrValue::Float(f) => Some(*f as f32),
                        _ => None,
                    })
                    .unwrap_or(0.0);

                let data = Self::decode_tensor(
                    &const_inputs[0],
                    dtype,
                    &const_shapes[0],
                );
                let quantized_data: Vec<i8> = data
                    .iter()
                    .map(|&x| {
                        ((x as f32 / scale_val) + zero_point)
                            .round()
                            .clamp(-128.0, 127.0) as i8
                    })
                    .collect();

                let bytes: Vec<u8> = quantized_data
                    .iter()
                    .flat_map(|&v| v.to_le_bytes())
                    .collect();

                let out_shape = const_shapes[0].clone();
                return Some((bytes, out_shape, DataType::I8));
            }
        }

        if op.op_type == "dequantize" {
            // 反量化常量
            let dtype = const_dtypes[0];
            if dtype == DataType::I8 {
                let scale = op
                    .attrs
                    .get("scale")
                    .and_then(|v| match v {
                        AttrValue::Float(f) => Some(*f as f32),
                        _ => None,
                    })
                    .unwrap_or(1.0);
                let zero_point = op
                    .attrs
                    .get("zero_point")
                    .and_then(|v| match v {
                        AttrValue::Int(i) => Some(*i as f32),
                        AttrValue::Float(f) => Some(*f as f32),
                        _ => None,
                    })
                    .unwrap_or(0.0);

                let data: Vec<f32> = const_inputs[0]
                    .iter()
                    .map(|&b| (b as f32 - zero_point) * scale)
                    .collect();

                let bytes: Vec<u8> =
                    data.iter().flat_map(|&v| v.to_le_bytes()).collect();

                return Some((bytes, const_shapes[0].clone(), DataType::F32));
            }
        }

        // ============================================================
        // Cast 常量折叠
        // ============================================================
        if op.op_type == "cast" {
            if let Some(target_dtype) = Self::get_cast_dtype(op) {
                // 解码输入数据
                let data = Self::decode_tensor(
                    &const_inputs[0],
                    const_dtypes[0],
                    &const_shapes[0],
                );

                // 编码为目标数据类型
                let encoded = Self::encode_tensor(&data, target_dtype);

                // 返回折叠后的常量
                return Some((encoded, const_shapes[0].clone(), target_dtype));
            }
            return None;
        }

        // ============================================================
        // 原有的常量折叠
        // ============================================================

        Self::fold_op_tensor(
            &op.op_type,
            &const_inputs,
            &const_shapes,
            &const_dtypes,
            &op.attrs,
        )
    }

    fn fold_op_tensor(
        op_type: &str,
        inputs: &[Vec<u8>],
        shapes: &[Vec<i64>],
        dtypes: &[DataType],
        attrs: &HashMap<String, AttrValue>,
    ) -> Option<(Vec<u8>, Vec<i64>, DataType)> {
        if dtypes.is_empty() {
            return None;
        }

        let dtype = dtypes[0];
        let all_same_dtype = dtypes.iter().all(|&d| d == dtype);
        if !all_same_dtype {
            return None;
        }

        match op_type {
            "add" | "sub" | "mul" | "div" | "pow" | "maximum" | "minimum" => {
                if inputs.len() < 2 {
                    return None;
                }

                let broadcast_shape =
                    Self::broadcast_shapes(&shapes[0], &shapes[1])?;

                let data1 = Self::decode_tensor(&inputs[0], dtype, &shapes[0]);
                let data2 = Self::decode_tensor(&inputs[1], dtype, &shapes[1]);

                let broadcasted1 = Self::broadcast_data(
                    &data1,
                    &shapes[0],
                    &broadcast_shape,
                    dtype,
                );
                let broadcasted2 = Self::broadcast_data(
                    &data2,
                    &shapes[1],
                    &broadcast_shape,
                    dtype,
                );

                let result = match op_type {
                    "add" => Self::elementwise_op(
                        &broadcasted1,
                        &broadcasted2,
                        |a, b| a + b,
                        dtype,
                    ),
                    "sub" => Self::elementwise_op(
                        &broadcasted1,
                        &broadcasted2,
                        |a, b| a - b,
                        dtype,
                    ),
                    "mul" => Self::elementwise_op(
                        &broadcasted1,
                        &broadcasted2,
                        |a, b| a * b,
                        dtype,
                    ),
                    "div" => Self::elementwise_op(
                        &broadcasted1,
                        &broadcasted2,
                        |a, b| {
                            if b == 0.0 {
                                0.0
                            } else {
                                a / b
                            }
                        },
                        dtype,
                    ),
                    "pow" => Self::elementwise_op(
                        &broadcasted1,
                        &broadcasted2,
                        |a, b| a.powf(b),
                        dtype,
                    ),
                    "maximum" => Self::elementwise_op(
                        &broadcasted1,
                        &broadcasted2,
                        |a, b| a.max(b),
                        dtype,
                    ),
                    "minimum" => Self::elementwise_op(
                        &broadcasted1,
                        &broadcasted2,
                        |a, b| a.min(b),
                        dtype,
                    ),
                    _ => return None,
                };

                let encoded = Self::encode_tensor(&result, dtype);
                Some((encoded, broadcast_shape, dtype))
            }

            "exp" | "sqrt" | "log" | "log2" | "log10" | "abs" | "neg"
            | "floor" | "ceil" | "round" | "sin" | "cos" | "tan" => {
                if inputs.len() != 1 {
                    return None;
                }

                let data = Self::decode_tensor(&inputs[0], dtype, &shapes[0]);
                let result = match op_type {
                    "exp" => Self::unary_op(&data, |x| x.exp(), dtype),
                    "sqrt" => Self::unary_op(&data, |x| x.sqrt(), dtype),
                    "log" => Self::unary_op(&data, |x| x.ln(), dtype),
                    "log2" => Self::unary_op(&data, |x| x.log2(), dtype),
                    "log10" => Self::unary_op(&data, |x| x.log10(), dtype),
                    "abs" => Self::unary_op(&data, |x| x.abs(), dtype),
                    "neg" => Self::unary_op(&data, |x| -x, dtype),
                    "floor" => Self::unary_op(&data, |x| x.floor(), dtype),
                    "ceil" => Self::unary_op(&data, |x| x.ceil(), dtype),
                    "round" => Self::unary_op(&data, |x| x.round(), dtype),
                    "sin" => Self::unary_op(&data, |x| x.sin(), dtype),
                    "cos" => Self::unary_op(&data, |x| x.cos(), dtype),
                    "tan" => Self::unary_op(&data, |x| x.tan(), dtype),
                    _ => return None,
                };

                let encoded = Self::encode_tensor(&result, dtype);
                Some((encoded, shapes[0].clone(), dtype))
            }

            "sum" | "mean" | "max" | "min" | "prod" => {
                if inputs.len() != 1 {
                    return None;
                }

                let dim = attrs
                    .get("dim")
                    .and_then(|v| match v {
                        AttrValue::Int(i) => Some(*i as usize),
                        _ => None,
                    })
                    .unwrap_or(0);
                let keepdim = attrs
                    .get("keepdim")
                    .and_then(|v| match v {
                        AttrValue::Bool(b) => Some(*b),
                        _ => None,
                    })
                    .unwrap_or(false);

                let data = Self::decode_tensor(&inputs[0], dtype, &shapes[0]);
                let (result_data, result_shape) = Self::reduce_op(
                    &data, &shapes[0], dim, keepdim, op_type, dtype,
                );

                let encoded = Self::encode_tensor(&result_data, dtype);
                Some((encoded, result_shape, dtype))
            }

            "transpose" => {
                if inputs.len() != 1 {
                    return None;
                }

                let perm = attrs.get("perm").and_then(|v| match v {
                    AttrValue::IntList(list) => Some(list.clone()),
                    _ => None,
                });

                let (result_data, result_shape) = if let Some(perm) = perm {
                    Self::transpose_tensor(&inputs[0], &shapes[0], &perm, dtype)
                } else {
                    Self::transpose_tensor(
                        &inputs[0],
                        &shapes[0],
                        &[0, 1],
                        dtype,
                    )
                };

                Some((result_data, result_shape, dtype))
            }

            "reshape" | "view" => {
                if inputs.len() != 1 {
                    return None;
                }

                let new_shape = attrs.get("shape").and_then(|v| match v {
                    AttrValue::Shape(shape) => Some(shape.clone()),
                    _ => None,
                });

                if let Some(new_shape) = new_shape {
                    Some((inputs[0].clone(), new_shape, dtype))
                } else {
                    None
                }
            }

            "cat" => {
                if inputs.is_empty() {
                    return None;
                }

                let dim = attrs
                    .get("dim")
                    .and_then(|v| match v {
                        AttrValue::Int(i) => Some(*i as usize),
                        _ => None,
                    })
                    .unwrap_or(0);

                let mut result_data = Vec::new();
                let mut result_shape = shapes[0].clone();

                let _offset = 0;
                for (i, data) in inputs.iter().enumerate() {
                    let shape = &shapes[i];
                    if i == 0 {
                        result_shape[dim] = shape[dim];
                    } else {
                        result_shape[dim] += shape[dim];
                    }
                    result_data.extend_from_slice(data);
                }

                Some((result_data, result_shape, dtype))
            }

            _ => None,
        }
    }

    // ============================================================
    // 辅助函数
    // ============================================================

    fn get_producer_op<'a>(
        graph: &'a DagGraph,
        value_id: u64,
    ) -> Option<&'a crate::ir::dag::Op> {
        if let Some(value) = graph.values.get(&value_id) {
            if let Some(producer_id) = value.producer {
                return graph.ops.get(&producer_id);
            }
        }
        None
    }

    #[allow(dead_code)]
    fn get_cast_dtype(op: &crate::ir::dag::Op) -> Option<DataType> {
        op.attrs.get("dtype").and_then(|v| match v {
            AttrValue::String(s) => match s.as_str() {
                "F32" => Some(DataType::F32),
                "F64" => Some(DataType::F64),
                "F16" => Some(DataType::F16),
                "BF16" => Some(DataType::BF16),
                "I8" => Some(DataType::I8),
                "I16" => Some(DataType::I16),
                "I32" => Some(DataType::I32),
                "I64" => Some(DataType::I64),
                "Bool" => Some(DataType::Bool),
                _ => None,
            },
            AttrValue::Int(i) => match i {
                0 => Some(DataType::F32),
                1 => Some(DataType::F64),
                2 => Some(DataType::I32),
                3 => Some(DataType::I64),
                _ => None,
            },
            _ => None,
        })
    }

    #[allow(dead_code)]
    fn create_cast_op(input: u64, dtype: DataType) -> crate::ir::dag::Op {
        let dtype_str = match dtype {
            DataType::F32 => "F32",
            DataType::F64 => "F64",
            DataType::F16 => "F16",
            DataType::BF16 => "BF16",
            DataType::I8 => "I8",
            DataType::I16 => "I16",
            DataType::I32 => "I32",
            DataType::I64 => "I64",
            DataType::Bool => "Bool",
        };
        let mut attrs = HashMap::new();
        attrs.insert(
            "dtype".to_string(),
            AttrValue::String(dtype_str.to_string()),
        );
        crate::ir::dag::Op {
            id: 0,
            name: format!("cast_{}", input),
            op_type: "cast".to_string(),
            inputs: vec![input],
            outputs: vec![],
            attrs,
        }
    }

    fn apply_replacement(graph: &mut DagGraph, old_id: u64, new_id: u64) {
        for (_, op) in graph.ops.iter_mut() {
            for input in &mut op.inputs {
                if *input == old_id {
                    *input = new_id;
                }
            }
        }
        for output in &mut graph.outputs {
            if *output == old_id {
                *output = new_id;
            }
        }
        if let Some(value) = graph.values.get(&old_id) {
            if let Some(producer_id) = value.producer {
                if let Some(op) = graph.ops.get_mut(&producer_id) {
                    for output in &mut op.outputs {
                        if *output == old_id {
                            *output = new_id;
                        }
                    }
                }
            }
        }
    }

    fn decode_tensor(data: &[u8], dtype: DataType, shape: &[i64]) -> Vec<f64> {
        let num_elements: usize =
            shape.iter().filter(|&&d| d > 0).map(|&d| d as usize).product();

        match dtype {
            DataType::F32 => {
                let mut result = Vec::with_capacity(num_elements);
                let chunks = data.chunks_exact(4);
                for chunk in chunks {
                    let val = f32::from_le_bytes([
                        chunk[0], chunk[1], chunk[2], chunk[3],
                    ]);
                    result.push(val as f64);
                }
                result
            }
            DataType::F64 => {
                let mut result = Vec::with_capacity(num_elements);
                let chunks = data.chunks_exact(8);
                for chunk in chunks {
                    let val = f64::from_le_bytes([
                        chunk[0], chunk[1], chunk[2], chunk[3], chunk[4],
                        chunk[5], chunk[6], chunk[7],
                    ]);
                    result.push(val);
                }
                result
            }
            DataType::I32 => {
                let mut result = Vec::with_capacity(num_elements);
                let chunks = data.chunks_exact(4);
                for chunk in chunks {
                    let val = i32::from_le_bytes([
                        chunk[0], chunk[1], chunk[2], chunk[3],
                    ]);
                    result.push(val as f64);
                }
                result
            }
            _ => {
                let mut result = Vec::with_capacity(num_elements);
                let chunks = data.chunks_exact(4);
                for chunk in chunks {
                    let val = f32::from_le_bytes([
                        chunk[0], chunk[1], chunk[2], chunk[3],
                    ]);
                    result.push(val as f64);
                }
                result
            }
        }
    }

    fn encode_tensor(data: &[f64], dtype: DataType) -> Vec<u8> {
        match dtype {
            DataType::F32 => {
                let mut result = Vec::with_capacity(data.len() * 4);
                for &val in data {
                    let f = val as f32;
                    result.extend_from_slice(&f.to_le_bytes());
                }
                result
            }
            DataType::F64 => {
                let mut result = Vec::with_capacity(data.len() * 8);
                for &val in data {
                    result.extend_from_slice(&val.to_le_bytes());
                }
                result
            }
            DataType::I32 => {
                let mut result = Vec::with_capacity(data.len() * 4);
                for &val in data {
                    let i = val as i32;
                    result.extend_from_slice(&i.to_le_bytes());
                }
                result
            }
            _ => {
                let mut result = Vec::with_capacity(data.len() * 4);
                for &val in data {
                    let f = val as f32;
                    result.extend_from_slice(&f.to_le_bytes());
                }
                result
            }
        }
    }

    fn broadcast_shapes(s1: &[i64], s2: &[i64]) -> Option<Vec<i64>> {
        let max_rank = s1.len().max(s2.len());
        let mut result = Vec::with_capacity(max_rank);

        for i in 0..max_rank {
            let d1 = if i < s1.len() { s1[s1.len() - 1 - i] } else { 1 };
            let d2 = if i < s2.len() { s2[s2.len() - 1 - i] } else { 1 };

            if d1 != d2 && d1 != 1 && d2 != 1 && d1 > 0 && d2 > 0 {
                return None;
            }
            result.push(d1.max(d2));
        }
        result.reverse();
        Some(result)
    }

    fn broadcast_data(
        data: &[f64],
        from_shape: &[i64],
        to_shape: &[i64],
        _dtype: DataType,
    ) -> Vec<f64> {
        if from_shape == to_shape {
            return data.to_vec();
        }

        let mut result = data.to_vec();
        for (i, &dim) in to_shape.iter().enumerate().rev() {
            let from_dim = if i < from_shape.len() { from_shape[i] } else { 1 };
            if from_dim == 1 && dim > 1 {
                let original_len = result.len();
                let repeat = dim as usize;
                let mut new_data = Vec::with_capacity(original_len * repeat);
                for _ in 0..repeat {
                    new_data.extend_from_slice(&result);
                }
                result = new_data;
            }
        }
        result
    }

    fn elementwise_op<F>(
        a: &[f64],
        b: &[f64],
        op: F,
        _dtype: DataType,
    ) -> Vec<f64>
    where
        F: Fn(f64, f64) -> f64,
    {
        let len = a.len().min(b.len());
        let mut result = Vec::with_capacity(len);
        for i in 0..len {
            result.push(op(a[i], b[i]));
        }
        result
    }

    fn unary_op<F>(data: &[f64], op: F, _dtype: DataType) -> Vec<f64>
    where
        F: Fn(f64) -> f64,
    {
        data.iter().map(|&x| op(x)).collect()
    }

    fn reduce_op(
        data: &[f64],
        shape: &[i64],
        dim: usize,
        keepdim: bool,
        op_type: &str,
        _dtype: DataType,
    ) -> (Vec<f64>, Vec<i64>) {
        let dim_size = shape[dim] as usize;
        let num_elements: usize = shape.iter().map(|&d| d as usize).product();
        let stride = shape.iter().skip(dim + 1).product::<i64>() as usize;
        let batch = shape.iter().take(dim).product::<i64>() as usize;

        let out_size = num_elements / dim_size;
        let mut result = vec![0.0; out_size];

        for b in 0..batch {
            for i in 0..stride {
                let mut sum = 0.0;
                let base = b * dim_size * stride + i;
                for d in 0..dim_size {
                    let idx = base + d * stride;
                    sum += data[idx];
                }
                let value = match op_type {
                    "sum" => sum,
                    "mean" => sum / dim_size as f64,
                    "max" => {
                        let mut max_val = f64::NEG_INFINITY;
                        for d in 0..dim_size {
                            let idx = base + d * stride;
                            if data[idx] > max_val {
                                max_val = data[idx];
                            }
                        }
                        max_val
                    }
                    "min" => {
                        let mut min_val = f64::INFINITY;
                        for d in 0..dim_size {
                            let idx = base + d * stride;
                            if data[idx] < min_val {
                                min_val = data[idx];
                            }
                        }
                        min_val
                    }
                    "prod" => {
                        let mut prod = 1.0;
                        for d in 0..dim_size {
                            let idx = base + d * stride;
                            prod *= data[idx];
                        }
                        prod
                    }
                    _ => 0.0,
                };
                result[b * stride + i] = value;
            }
        }

        let mut out_shape = shape.to_vec();
        if keepdim {
            out_shape[dim] = 1;
        } else {
            out_shape.remove(dim);
        }

        (result, out_shape)
    }

    fn transpose_tensor(
        data: &[u8],
        shape: &[i64],
        perm: &[i64],
        dtype: DataType,
    ) -> (Vec<u8>, Vec<i64>) {
        let decoded = Self::decode_tensor(data, dtype, shape);
        let rank = shape.len();
        let dims: Vec<usize> = shape.iter().map(|&d| d as usize).collect();
        let perm: Vec<usize> = perm.iter().map(|&p| p as usize).collect();

        let total: usize = dims.iter().product();
        let mut result = vec![0.0; total];

        let mut strides = vec![1; rank];
        for i in (0..rank - 1).rev() {
            strides[i] = strides[i + 1] * dims[i + 1];
        }

        let out_strides: Vec<usize> =
            perm.iter().map(|&p| strides[p]).collect();
        let out_dims: Vec<usize> = perm.iter().map(|&p| dims[p]).collect();

        for i in 0..total {
            let mut idx = i;
            let mut in_pos = 0;
            for j in 0..rank {
                let dim_idx = idx / out_strides[j];
                idx %= out_strides[j];
                let orig_dim = perm[j];
                in_pos += dim_idx * strides[orig_dim];
            }
            result[i] = decoded[in_pos];
        }

        let out_shape: Vec<i64> = out_dims.iter().map(|&d| d as i64).collect();
        (Self::encode_tensor(&result, dtype), out_shape)
    }
}
