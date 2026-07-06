// src/transform/constant_fold.rs

use std::collections::HashMap;
use crate::ir::dag::{DagGraph, AttrValue, DataType};

pub struct ConstantFoldingPass;

impl ConstantFoldingPass {
    pub fn apply(&self, graph: &mut DagGraph) -> bool {
        let mut changed = false;
        let mut to_remove = Vec::new();

        let op_ids: Vec<u64> = graph.ops.keys().cloned().collect();

        for &op_id in &op_ids {
            let op = match graph.ops.get(&op_id) {
                Some(o) => o,
                None => continue,
            };

            // 检查是否所有输入都是常量
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
                continue;
            }

            // 尝试折叠
            if let Some((result_data, result_shape, result_dtype)) =
                self.fold_op_tensor(&op.op_type, &const_inputs, &const_shapes, &const_dtypes, &op.attrs)
            {
                let out_id = op.outputs[0];

                // 更新常量数据
                graph.constants.insert(out_id, result_data);

                // 更新 shape 和 dtype
                if let Some(value) = graph.values.get_mut(&out_id) {
                    value.ty.shape = result_shape;
                    value.ty.dtype = result_dtype;
                }

                to_remove.push(op_id);
                changed = true;
            }
        }

        for id in to_remove {
            graph.ops.remove(&id);
        }

        changed
    }

    fn fold_op_tensor(
        &self,
        op_type: &str,
        inputs: &[Vec<u8>],
        shapes: &[Vec<i64>],
        dtypes: &[DataType],
        attrs: &HashMap<String, AttrValue>,
    ) -> Option<(Vec<u8>, Vec<i64>, DataType)> {
        // 检查所有输入数据类型是否一致
        if dtypes.is_empty() {
            return None;
        }

        let dtype = dtypes[0];
        let all_same_dtype = dtypes.iter().all(|&d| d == dtype);
        if !all_same_dtype {
            return None; // 暂时不支持混合类型
        }

        match op_type {
            // ============================================================
            // 逐元素操作（支持张量）
            // ============================================================
            "add" | "sub" | "mul" | "div" | "pow" | "maximum" | "minimum" => {
                if inputs.len() < 2 {
                    return None;
                }

                // 广播 shape
                let broadcast_shape = self.broadcast_shapes(&shapes[0], &shapes[1])?;

                // 解码数据
                let data1 = self.decode_tensor(&inputs[0], dtype, &shapes[0]);
                let data2 = self.decode_tensor(&inputs[1], dtype, &shapes[1]);

                // 广播数据
                let broadcasted1 = self.broadcast_data(&data1, &shapes[0], &broadcast_shape, dtype);
                let broadcasted2 = self.broadcast_data(&data2, &shapes[1], &broadcast_shape, dtype);

                // 执行操作
                let result = match op_type {
                    "add" => self.elementwise_op(&broadcasted1, &broadcasted2, |a, b| a + b, dtype),
                    "sub" => self.elementwise_op(&broadcasted1, &broadcasted2, |a, b| a - b, dtype),
                    "mul" => self.elementwise_op(&broadcasted1, &broadcasted2, |a, b| a * b, dtype),
                    "div" => self.elementwise_op(&broadcasted1, &broadcasted2, |a, b| {
                        if b == 0.0 { 0.0 } else { a / b }
                    }, dtype),
                    "pow" => self.elementwise_op(&broadcasted1, &broadcasted2, |a, b| a.powf(b), dtype),
                    "maximum" => self.elementwise_op(&broadcasted1, &broadcasted2, |a, b| a.max(b), dtype),
                    "minimum" => self.elementwise_op(&broadcasted1, &broadcasted2, |a, b| a.min(b), dtype),
                    _ => return None,
                };

                let encoded = self.encode_tensor(&result, dtype);
                Some((encoded, broadcast_shape, dtype))
            }

            // ============================================================
            // 一元操作
            // ============================================================
            "exp" | "sqrt" | "log" | "log2" | "log10" |
            "abs" | "neg" | "floor" | "ceil" | "round" | "sin" | "cos" | "tan" => {
                if inputs.len() != 1 {
                    return None;
                }

                let data = self.decode_tensor(&inputs[0], dtype, &shapes[0]);
                let result = match op_type {
                    "exp" => self.unary_op(&data, |x| x.exp(), dtype),
                    "sqrt" => self.unary_op(&data, |x| x.sqrt(), dtype),
                    "log" => self.unary_op(&data, |x| x.ln(), dtype),
                    "log2" => self.unary_op(&data, |x| x.log2(), dtype),
                    "log10" => self.unary_op(&data, |x| x.log10(), dtype),
                    "abs" => self.unary_op(&data, |x| x.abs(), dtype),
                    "neg" => self.unary_op(&data, |x| -x, dtype),
                    "floor" => self.unary_op(&data, |x| x.floor(), dtype),
                    "ceil" => self.unary_op(&data, |x| x.ceil(), dtype),
                    "round" => self.unary_op(&data, |x| x.round(), dtype),
                    "sin" => self.unary_op(&data, |x| x.sin(), dtype),
                    "cos" => self.unary_op(&data, |x| x.cos(), dtype),
                    "tan" => self.unary_op(&data, |x| x.tan(), dtype),
                    _ => return None,
                };

                let encoded = self.encode_tensor(&result, dtype);
                Some((encoded, shapes[0].clone(), dtype))
            }

            // ============================================================
            // 归约操作
            // ============================================================
            "sum" | "mean" | "max" | "min" | "prod" => {
                if inputs.len() != 1 {
                    return None;
                }

                let dim = attrs.get("dim")
                    .and_then(|v| match v { AttrValue::Int(i) => Some(*i as usize), _ => None })
                    .unwrap_or(0);
                let keepdim = attrs.get("keepdim")
                    .and_then(|v| match v { AttrValue::Bool(b) => Some(*b), _ => None })
                    .unwrap_or(false);

                let data = self.decode_tensor(&inputs[0], dtype, &shapes[0]);
                let (result_data, result_shape) = self.reduce_op(&data, &shapes[0], dim, keepdim, op_type, dtype);

                let encoded = self.encode_tensor(&result_data, dtype);
                Some((encoded, result_shape, dtype))
            }

            // ============================================================
            // 转置
            // ============================================================
            "transpose" => {
                if inputs.len() != 1 {
                    return None;
                }

                let perm = attrs.get("perm")
                    .and_then(|v| match v { AttrValue::IntList(list) => Some(list.clone()), _ => None });

                let (result_data, result_shape) = if let Some(perm) = perm {
                    self.transpose_tensor(&inputs[0], &shapes[0], &perm, dtype)
                } else {
                    // 默认转置最后两维
                    self.transpose_tensor(&inputs[0], &shapes[0], &[0, 1], dtype)
                };

                Some((result_data, result_shape, dtype))
            }

            // ============================================================
            // Reshape
            // ============================================================
            "reshape" | "view" => {
                if inputs.len() != 1 {
                    return None;
                }

                let new_shape = attrs.get("shape")
                    .and_then(|v| match v { AttrValue::Shape(shape) => Some(shape.clone()), _ => None });

                if let Some(new_shape) = new_shape {
                    // 数据不变，只是改变 shape
                    Some((inputs[0].clone(), new_shape, dtype))
                } else {
                    None
                }
            }

            // ============================================================
            // Cat (拼接)
            // ============================================================
            "cat" => {
                if inputs.is_empty() {
                    return None;
                }

                let dim = attrs.get("dim")
                    .and_then(|v| match v { AttrValue::Int(i) => Some(*i as usize), _ => None })
                    .unwrap_or(0);

                let mut result_data = Vec::new();
                let mut result_shape = shapes[0].clone();

                // 在 dim 维度上拼接
                let mut offset = 0;
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
    // 辅助函数：张量操作
    // ============================================================

    fn decode_tensor(&self, data: &[u8], dtype: DataType, shape: &[i64]) -> Vec<f64> {
        let num_elements: usize = shape.iter().filter(|&&d| d > 0).map(|&d| d as usize).product();

        match dtype {
            DataType::F32 => {
                let mut result = Vec::with_capacity(num_elements);
                let chunks = data.chunks_exact(4);
                for chunk in chunks {
                    let val = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
                    result.push(val as f64);
                }
                result
            }
            DataType::F64 => {
                let mut result = Vec::with_capacity(num_elements);
                let chunks = data.chunks_exact(8);
                for chunk in chunks {
                    let val = f64::from_le_bytes([
                        chunk[0], chunk[1], chunk[2], chunk[3],
                        chunk[4], chunk[5], chunk[6], chunk[7]
                    ]);
                    result.push(val);
                }
                result
            }
            DataType::I32 => {
                let mut result = Vec::with_capacity(num_elements);
                let chunks = data.chunks_exact(4);
                for chunk in chunks {
                    let val = i32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
                    result.push(val as f64);
                }
                result
            }
            _ => {
                // 默认当作 f32
                let mut result = Vec::with_capacity(num_elements);
                let chunks = data.chunks_exact(4);
                for chunk in chunks {
                    let val = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
                    result.push(val as f64);
                }
                result
            }
        }
    }

    fn encode_tensor(&self, data: &[f64], dtype: DataType) -> Vec<u8> {
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
                // 默认用 f32
                let mut result = Vec::with_capacity(data.len() * 4);
                for &val in data {
                    let f = val as f32;
                    result.extend_from_slice(&f.to_le_bytes());
                }
                result
            }
        }
    }

    fn broadcast_shapes(&self, s1: &[i64], s2: &[i64]) -> Option<Vec<i64>> {
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

    fn broadcast_data(&self, data: &[f64], from_shape: &[i64], to_shape: &[i64], _dtype: DataType) -> Vec<f64> {
        if from_shape == to_shape {
            return data.to_vec();
        }

        // 简单广播：扩展维度
        let mut result = data.to_vec();
        for (i, &dim) in to_shape.iter().enumerate().rev() {
            let from_dim = if i < from_shape.len() { from_shape[i] } else { 1 };
            if from_dim == 1 && dim > 1 {
                // 重复数据
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

    fn elementwise_op<F>(&self, a: &[f64], b: &[f64], op: F, _dtype: DataType) -> Vec<f64>
    where F: Fn(f64, f64) -> f64 {
        let len = a.len().min(b.len());
        let mut result = Vec::with_capacity(len);
        for i in 0..len {
            result.push(op(a[i], b[i]));
        }
        result
    }

    fn unary_op<F>(&self, data: &[f64], op: F, _dtype: DataType) -> Vec<f64>
    where F: Fn(f64) -> f64 {
        data.iter().map(|&x| op(x)).collect()
    }

    fn reduce_op(
        &self,
        data: &[f64],
        shape: &[i64],
        dim: usize,
        keepdim: bool,
        op_type: &str,
        dtype: DataType,
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
                            if data[idx] > max_val { max_val = data[idx]; }
                        }
                        max_val
                    }
                    "min" => {
                        let mut min_val = f64::INFINITY;
                        for d in 0..dim_size {
                            let idx = base + d * stride;
                            if data[idx] < min_val { min_val = data[idx]; }
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

        // 计算输出 shape
        let mut out_shape = shape.to_vec();
        if keepdim {
            out_shape[dim] = 1;
        } else {
            out_shape.remove(dim);
        }

        (result, out_shape)
    }

    fn transpose_tensor(
        &self,
        data: &[u8],
        shape: &[i64],
        perm: &[i64],
        dtype: DataType,
    ) -> (Vec<u8>, Vec<i64>) {
        let decoded = self.decode_tensor(data, dtype, shape);
        let rank = shape.len();
        let dims: Vec<usize> = shape.iter().map(|&d| d as usize).collect();
        let perm: Vec<usize> = perm.iter().map(|&p| p as usize).collect();

        // 计算总元素数
        let total: usize = dims.iter().product();
        let mut result = vec![0.0; total];

        // 计算每个维度的步长
        let mut strides = vec![1; rank];
        for i in (0..rank-1).rev() {
            strides[i] = strides[i+1] * dims[i+1];
        }

        let out_strides: Vec<usize> = perm.iter().map(|&p| strides[p]).collect();
        let out_dims: Vec<usize> = perm.iter().map(|&p| dims[p]).collect();

        // 遍历所有元素
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
        (self.encode_tensor(&result, dtype), out_shape)
    }
}
