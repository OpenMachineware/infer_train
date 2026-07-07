// src/transform/shape_inference.rs

use std::collections::HashMap;
use crate::ir::dag::*;

pub struct ShapeInferencePass;

impl ShapeInferencePass {
    pub fn apply(graph: &mut DagGraph) -> Result<(), String> {
        // 按拓扑顺序遍历
        let topo = graph.topological_sort()?;

        for &op_id in &topo {
            let op = match graph.ops.get(&op_id) {
                Some(o) => o,
                None => continue,
            };

            // 收集输入 shape
            let input_shapes: Result<Vec<Vec<i64>>, String> = op.inputs.iter()
                .map(|&id| {
                    graph.values.get(&id)
                        .ok_or_else(|| format!("Value {} not found", id))
                        .map(|v| v.ty.shape.clone())
                })
                .collect();

            let input_shapes = input_shapes?;

            // 推导输出 shape
            let output_shapes = Self::infer_shapes(&op.op_type, &input_shapes, &op.attrs)?;

            // 更新 outputs 的 shape
            if output_shapes.len() != op.outputs.len() {
                return Err(format!(
                    "Shape count mismatch for op {}: expected {}, got {}",
                    op.op_type, op.outputs.len(), output_shapes.len()
                ));
            }

            for (out_id, shape) in op.outputs.iter().zip(output_shapes) {
                if let Some(value) = graph.values.get_mut(out_id) {
                    value.ty.shape = shape;
                }
            }
        }

        Ok(())
    }

    // ============================================================
    // 核心：Shape 推导规则
    // ============================================================
    fn infer_shapes(
        op_type: &str,
        input_shapes: &[Vec<i64>],
        attrs: &HashMap<String, AttrValue>,
    ) -> Result<Vec<Vec<i64>>, String> {
        match op_type {
            // ---------- 一元逐元素算子（shape不变） ----------
            "relu" | "relu6" | "leaky_relu" | "elu" | "gelu" | "silu" |
            "sigmoid" | "tanh" | "hard_swish" | "hard_sigmoid" |
            "softplus" | "softshrink" | "celu" |
            "exp" | "sqrt" | "log" | "log2" | "log10" |
            "abs" | "neg" | "floor" | "ceil" |
            "round" => {
                Self::check_input_count(input_shapes, 1)?;
                Ok(vec![input_shapes[0].clone()])
            }

            // ---------- 二元逐元素算子（广播） ----------
            "add" | "sub" | "mul" | "div" | "pow" |
            "maximum" | "minimum" | "equal" | "not_equal" |
            "less" | "less_equal" | "greater" | "greater_equal" => {
                Self::check_input_count(input_shapes, 2)?;
                let shape = Self::broadcast_shape(&input_shapes[0], &input_shapes[1])?;
                Ok(vec![shape])
            }

            // ---------- MatMul ----------
            "matmul" | "linear" => {
                Self::check_input_count(input_shapes, 2)?;
                let (s1, s2) = (&input_shapes[0], &input_shapes[1]);

                if s1.len() < 2 || s2.len() < 2 {
                    return Err(format!(
                        "MatMul requires at least 2D tensors, got {:?} and {:?}",
                        s1, s2
                    ));
                }

                let (m, k1) = (s1[s1.len()-2], s1[s1.len()-1]);
                let (k2, n) = (s2[s2.len()-2], s2[s2.len()-1]);

                if k1 != -1 && k2 != -1 && k1 != k2 {
                    return Err(format!(
                        "MatMul shape mismatch: {} != {}",
                        k1, k2
                    ));
                }

                // 广播 batch 维度
                let mut batch_dims = Vec::new();
                let max_batch = s1.len().max(s2.len()) - 2;
                for i in 0..max_batch {
                    let d1 = if i < s1.len() - 2 { s1[s1.len() - 2 - i] } else { 1 };
                    let d2 = if i < s2.len() - 2 { s2[s2.len() - 2 - i] } else { 1 };
                    if d1 != d2 && d1 != 1 && d2 != 1 {
                        return Err(format!(
                            "Batch dimension mismatch: {} vs {}",
                            d1, d2
                        ));
                    }
                    batch_dims.push(d1.max(d2));
                }
                batch_dims.reverse();

                let mut result = batch_dims;
                result.push(m);
                result.push(n);
                Ok(vec![result])
            }

            // ---------- Conv2d ----------
            "conv2d" => {
                Self::check_input_count(input_shapes, 2)?; // [x, weight] 或 [x, weight, bias]
                let x = &input_shapes[0];
                let w = &input_shapes[1];

                if x.len() != 4 {
                    return Err(format!("Conv2d input must be 4D, got {:?}", x));
                }
                if w.len() != 4 {
                    return Err(format!("Conv2d weight must be 4D, got {:?}", w));
                }

                let (n, c, h, w_in) = (x[0], x[1], x[2], x[3]);
                let (out_c, in_c, k_h, k_w) = (w[0], w[1], w[2], w[3]);

                if c != -1 && in_c != -1 && c != in_c {
                    return Err(format!(
                        "Conv2d channel mismatch: input {} vs weight {}",
                        c, in_c
                    ));
                }

                // 从 attrs 读取参数
                let stride = Self::get_int_list_attr(attrs, "stride", vec![1, 1]);
                let padding = Self::get_int_list_attr(attrs, "padding", vec![0, 0]);
                let dilation = Self::get_int_list_attr(attrs, "dilation", vec![1, 1]);
                let padding_mode = Self::get_string_attr(attrs, "padding_mode", "zeros");

                let (stride_h, stride_w) = (stride[0], stride[1]);
                let (padding_h, padding_w) = (padding[0], padding[1]);
                let (dilation_h, dilation_w) = (dilation[0], dilation[1]);

                let h_out = Self::conv_output_size(h, k_h, padding_h, stride_h, dilation_h, &padding_mode)?;
                let w_out = Self::conv_output_size(w_in, k_w, padding_w, stride_w, dilation_w, &padding_mode)?;

                Ok(vec![vec![n, out_c, h_out, w_out]])
            }

            // ---------- BatchNorm2d ----------
            "batchnorm2d" => {
                Self::check_input_count(input_shapes, 1)?;
                // shape 不变
                Ok(vec![input_shapes[0].clone()])
            }

            // ---------- 融合算子 ----------
            "fused_conv_bn" | "fused_conv_relu" | "fused_conv_bn_relu" => {
                // 与 conv2d 相同
                Self::infer_shapes("conv2d", input_shapes, attrs)
            }

            // ---------- Pooling ----------
            "maxpool2d" | "avgpool2d" => {
                Self::check_input_count(input_shapes, 1)?;
                let x = &input_shapes[0];
                if x.len() != 4 {
                    return Err(format!("Pooling input must be 4D, got {:?}", x));
                }

                let (n, c, h, w) = (x[0], x[1], x[2], x[3]);
                let kernel_size = Self::get_int_list_attr(attrs, "kernel_size", vec![1, 1]);
                let stride = Self::get_int_list_attr(attrs, "stride", kernel_size.clone());
                let padding = Self::get_int_list_attr(attrs, "padding", vec![0, 0]);
                let ceil_mode = Self::get_bool_attr(attrs, "ceil_mode", false);

                let (k_h, k_w) = (kernel_size[0], kernel_size[1]);
                let (s_h, s_w) = (stride[0], stride[1]);
                let (p_h, p_w) = (padding[0], padding[1]);

                let h_out = Self::pool_output_size(h, k_h, p_h, s_h, ceil_mode);
                let w_out = Self::pool_output_size(w, k_w, p_w, s_w, ceil_mode);

                Ok(vec![vec![n, c, h_out, w_out]])
            }

            // ---------- Reshape ----------
            "reshape" | "view" | "flatten" => {
                Self::check_input_count(input_shapes, 1)?;
                let shape_attr = Self::get_shape_attr(attrs, "shape")?;
                let x = &input_shapes[0];

                // 计算总元素数
                let total_elements = Self::total_elements(x)?;
                let mut inferred_shape = Vec::new();
                let mut has_neg = false;

                for &dim in &shape_attr {
                    if dim == -1 {
                        if has_neg {
                            return Err("Only one -1 allowed in reshape".to_string());
                        }
                        has_neg = true;
                        inferred_shape.push(0); // placeholder
                    } else {
                        inferred_shape.push(dim);
                    }
                }

                if has_neg {
                    // 计算 -1 维度的值
                    let known_elements: i64 = inferred_shape.iter()
                        .filter(|&&d| d != 0)
                        .product();
                    if known_elements == 0 {
                        return Err("Cannot infer -1 dimension with zero known dimensions".to_string());
                    }
                    if total_elements % known_elements != 0 {
                        return Err(format!(
                            "Cannot reshape {} elements into shape {:?}",
                            total_elements, shape_attr
                        ));
                    }
                    let neg_dim = total_elements / known_elements;
                    for dim in &mut inferred_shape {
                        if *dim == 0 {
                            *dim = neg_dim;
                        }
                    }
                }

                Ok(vec![inferred_shape])
            }

            // ---------- Transpose ----------
            "transpose" => {
                Self::check_input_count(input_shapes, 1)?;
                let x = &input_shapes[0];
                let perm = Self::get_int_list_attr(attrs, "perm",
                                                   (0..x.len() as i64).collect()
                );

                if perm.len() != x.len() {
                    return Err(format!(
                        "Perm length {} doesn't match input rank {}",
                        perm.len(), x.len()
                    ));
                }

                let mut result = Vec::new();
                for &idx in &perm {
                    let idx_usize = idx as usize;
                    if idx_usize >= x.len() {
                        return Err(format!("Perm index {} out of range", idx));
                    }
                    result.push(x[idx_usize]);
                }
                Ok(vec![result])
            }

            // ---------- Slice ----------
            "slice" => {
                Self::check_input_count(input_shapes, 1)?;
                let x = &input_shapes[0];
                let starts = Self::get_int_list_attr(attrs, "starts", vec![0]);
                let ends = Self::get_int_list_attr(attrs, "ends", vec![x.len() as i64]);
                let steps = Self::get_int_list_attr(attrs, "steps", vec![1]);

                if starts.len() != x.len() || ends.len() != x.len() {
                    return Err(format!(
                        "Slice dims mismatch: input {}, starts {}, ends {}",
                        x.len(), starts.len(), ends.len()
                    ));
                }

                let mut result = Vec::new();
                for i in 0..x.len() {
                    let start = starts.get(i).copied().unwrap_or(0);
                    let end = ends.get(i).copied().unwrap_or(x[i]);
                    let step = steps.get(i).copied().unwrap_or(1);

                    if x[i] == -1 {
                        result.push(-1);
                    } else {
                        let len = (end - start + step - 1) / step;
                        result.push(len);
                    }
                }
                Ok(vec![result])
            }

            // ---------- Concat ----------
            "cat" => {
                if input_shapes.is_empty() {
                    return Err("Concat requires at least one input".to_string());
                }
                let dim = Self::get_int_attr(attrs, "dim", 0) as usize;
                let rank = input_shapes[0].len();

                if dim >= rank {
                    return Err(format!(
                        "Concat dim {} out of range for rank {}",
                        dim, rank
                    ));
                }

                let mut result = input_shapes[0].clone();
                for i in 1..input_shapes.len() {
                    let shape = &input_shapes[i];
                    if shape.len() != rank {
                        return Err("All inputs to concat must have same rank".to_string());
                    }
                    for d in 0..rank {
                        if d != dim && shape[d] != -1 && result[d] != -1 && shape[d] != result[d] {
                            return Err(format!(
                                "Concat shape mismatch at dim {}: {} vs {}",
                                d, result[d], shape[d]
                            ));
                        }
                    }
                    // 在 concat 维度上相加
                    if result[dim] != -1 && shape[dim] != -1 {
                        result[dim] += shape[dim];
                    } else {
                        result[dim] = -1;
                    }
                }
                Ok(vec![result])
            }

            // ---------- Softmax ----------
            "softmax" | "log_softmax" => {
                Self::check_input_count(input_shapes, 1)?;
                // shape 不变
                Ok(vec![input_shapes[0].clone()])
            }

            // ---------- 常量/占位符 ----------
            "constant" => {
                // 直接从 attrs 读取 shape
                if let Some(shape) = Self::get_shape_attr_opt(attrs, "shape") {
                    Ok(vec![shape])
                } else if let Some(shape) = Self::get_int_list_attr_opt(attrs, "shape") {
                    Ok(vec![shape])
                } else {
                    // 默认标量
                    Ok(vec![vec![]])
                }
            }

            // ============================================================
            // Embedding
            // ============================================================
            "embedding" => {
                Self::check_input_count(input_shapes, 2)?; // [indices, weight]
                let indices = &input_shapes[0];
                let weight = &input_shapes[1];

                // 输出 shape = indices_shape + [weight_shape[-1]]
                let mut out_shape = indices.clone();
                if weight.len() >= 2 {
                    out_shape.push(weight[weight.len() - 1]);
                }
                Ok(vec![out_shape])
            }

            // ============================================================
            // Gather/Scatter
            // ============================================================
            "gather" => {
                Self::check_input_count(input_shapes, 2)?;
                let _dim = Self::get_int_attr(attrs, "dim", 0) as usize;

                // 输出 shape 与 indices 相同
                Ok(vec![input_shapes[1].clone()])
            }
            "scatter" => {
                Self::check_input_count(input_shapes, 3)?;
                // 输出 shape 与 input 相同
                Ok(vec![input_shapes[0].clone()])
            }

            // ============================================================
            // TopK
            // ============================================================
            "topk" => {
                Self::check_input_count(input_shapes, 1)?;
                let k = Self::get_int_attr(attrs, "k", 1);
                let dim = Self::get_int_attr(attrs, "dim", -1) as usize;

                let mut out_shape = input_shapes[0].clone();
                if dim < out_shape.len() {
                    out_shape[dim] = k;
                }
                // TopK 返回两个输出：values 和 indices
                Ok(vec![out_shape.clone(), out_shape])
            }

            // ============================================================
            // ArgMax/ArgMin
            // ============================================================
            "argmax" | "argmin" => {
                Self::check_input_count(input_shapes, 1)?;
                let dim = Self::get_int_attr(attrs, "dim", -1) as usize;
                let keepdim = Self::get_bool_attr(attrs, "keepdim", false);

                let mut out_shape = input_shapes[0].clone();
                if dim < out_shape.len() {
                    if keepdim {
                        out_shape[dim] = 1;
                    } else {
                        out_shape.remove(dim);
                    }
                }
                Ok(vec![out_shape])
            }

            // ============================================================
            // Where
            // ============================================================
            "where" => {
                Self::check_input_count(input_shapes, 3)?;
                let shape = Self::broadcast_shape(&input_shapes[0], &input_shapes[1])?;
                let shape = Self::broadcast_shape(&shape, &input_shapes[2])?;
                Ok(vec![shape])
            }

            // ============================================================
            // Sort
            // ============================================================
            "sort" => {
                Self::check_input_count(input_shapes, 1)?;
                let _dim = Self::get_int_attr(attrs, "dim", -1) as usize;
                let out_shape = input_shapes[0].clone();
                // Sort 返回 values 和 indices
                Ok(vec![out_shape.clone(), out_shape])
            }

            // ============================================================
            // 量化算子（shape 与对应 FP 算子相同）
            // ============================================================
            "quantized_add" | "quantized_sub" | "quantized_mul" | "quantized_div" => {
                Self::infer_shapes("add", input_shapes, attrs)
            }
            "quantized_matmul" => {
                Self::infer_shapes("matmul", input_shapes, attrs)
            }
            "quantized_relu" | "quantized_sigmoid" | "quantized_exp" |
            "quantized_sqrt" | "quantized_abs" | "quantized_neg" => {
                Self::infer_shapes("relu", input_shapes, attrs)
            }
            "quantized_conv2d" => {
                Self::infer_shapes("conv2d", input_shapes, attrs)
            }
            "quantized_linear" => {
                Self::infer_shapes("linear", input_shapes, attrs)
            }
            "quantized_maxpool2d" => {
                Self::infer_shapes("maxpool2d", input_shapes, attrs)
            }
            "quantized_avgpool2d" => {
                Self::infer_shapes("avgpool2d", input_shapes, attrs)
            }
            "quantized_batchnorm2d" => {
                Self::infer_shapes("batchnorm2d", input_shapes, attrs)
            }
            "quantized_clamp" => {
                Self::infer_shapes("clamp", input_shapes, attrs)
            }

            // ============================================================
            // Clamp
            // ============================================================
            "clamp" => {
                Self::check_input_count(input_shapes, 1)?;
                Ok(vec![input_shapes[0].clone()])
            }

            // ============================================================
            // Identity 相关
            // ============================================================
            "identity" => {
                Self::check_input_count(input_shapes, 1)?;
                Ok(vec![input_shapes[0].clone()])
            }

            // ============================================================
            // Dropout（推理时是 identity）
            // ============================================================
            "dropout" => {
                Self::check_input_count(input_shapes, 1)?;
                Ok(vec![input_shapes[0].clone()])
            }

            // ---------- 未知算子 ----------
            _ => {
                // 对于未知算子，假设 shape 不变或尝试从输入推断
                if !input_shapes.is_empty() {
                    Ok(vec![input_shapes[0].clone()])
                } else {
                    Err(format!("Unknown operator '{}' with no inputs", op_type))
                }
            }
        }
    }

    // ============================================================
    // 辅助函数：Shape 计算
    // ============================================================

    fn check_input_count(shapes: &[Vec<i64>], expected: usize) -> Result<(), String> {
        if shapes.len() != expected {
            Err(format!(
                "Expected {} inputs, got {}",
                expected, shapes.len()
            ))
        } else {
            Ok(())
        }
    }

    fn total_elements(shape: &[i64]) -> Result<i64, String> {
        let mut total = 1;
        for &dim in shape {
            if dim <= 0 && dim != -1 {
                return Err(format!("Invalid dimension: {}", dim));
            }
            if dim != -1 {
                total *= dim;
            }
        }
        Ok(total)
    }

    fn broadcast_shape(s1: &[i64], s2: &[i64]) -> Result<Vec<i64>, String> {
        let max_rank = s1.len().max(s2.len());
        let mut result = Vec::with_capacity(max_rank);

        for i in 0..max_rank {
            let d1 = if i < s1.len() { s1[s1.len() - 1 - i] } else { 1 };
            let d2 = if i < s2.len() { s2[s2.len() - 1 - i] } else { 1 };

            if d1 != d2 && d1 != 1 && d2 != 1 {
                return Err(format!(
                    "Cannot broadcast shapes {:?} and {:?}",
                    s1, s2
                ));
            }
            result.push(d1.max(d2));
        }
        result.reverse();
        Ok(result)
    }

    fn conv_output_size(
        input: i64,
        kernel: i64,
        padding: i64,
        stride: i64,
        dilation: i64,
        padding_mode: &str,
    ) -> Result<i64, String> {
        if input == -1 {
            return Ok(-1);
        }
        if stride <= 0 {
            return Err(format!("Invalid stride: {}", stride));
        }

        let effective_kernel = (kernel - 1) * dilation + 1;
        let padded = input + 2 * padding;
        let output = (padded - effective_kernel) / stride + 1;

        if padding_mode == "same" {
            // 对于 same padding，输出大小保持输入大小
            return Ok(input);
        }

        if output <= 0 {
            return Err(format!(
                "Conv output size <= 0: input={}, kernel={}, padding={}, stride={}",
                input, kernel, padding, stride
            ));
        }
        Ok(output)
    }

    fn pool_output_size(input: i64, kernel: i64, padding: i64, stride: i64, ceil_mode: bool) -> i64 {
        if input == -1 {
            return -1;
        }
        let output = (input + 2 * padding - kernel) as f64 / stride as f64 + 1.0;
        if ceil_mode {
            output.ceil() as i64
        } else {
            output.floor() as i64
        }
    }

    // ============================================================
    // 辅助函数：读取 Attr
    // ============================================================

    fn get_int_attr(attrs: &HashMap<String, AttrValue>, key: &str, default: i64) -> i64 {
        attrs.get(key)
            .and_then(|v| match v { AttrValue::Int(i) => Some(*i), _ => None })
            .unwrap_or(default)
    }

    fn get_bool_attr(attrs: &HashMap<String, AttrValue>, key: &str, default: bool) -> bool {
        attrs.get(key)
            .and_then(|v| match v { AttrValue::Bool(b) => Some(*b), _ => None })
            .unwrap_or(default)
    }

    fn get_string_attr(attrs: &HashMap<String, AttrValue>, key: &str, default: &str) -> String {
        attrs.get(key)
            .and_then(|v| match v { AttrValue::String(s) => Some(s.clone()), _ => None })
            .unwrap_or_else(|| default.to_string())
    }

    fn get_int_list_attr(attrs: &HashMap<String, AttrValue>, key: &str, default: Vec<i64>) -> Vec<i64> {
        attrs.get(key)
            .and_then(|v| match v { AttrValue::IntList(list) => Some(list.clone()), _ => None })
            .unwrap_or(default)
    }

    fn get_int_list_attr_opt(attrs: &HashMap<String, AttrValue>, key: &str) -> Option<Vec<i64>> {
        attrs.get(key)
            .and_then(|v| match v { AttrValue::IntList(list) => Some(list.clone()), _ => None })
    }

    fn get_shape_attr(attrs: &HashMap<String, AttrValue>, key: &str) -> Result<Vec<i64>, String> {
        attrs.get(key)
            .and_then(|v| match v { AttrValue::Shape(shape) => Some(shape.clone()), _ => None })
            .ok_or_else(|| format!("Attribute '{}' not found", key))
    }

    fn get_shape_attr_opt(attrs: &HashMap<String, AttrValue>, key: &str) -> Option<Vec<i64>> {
        attrs.get(key)
            .and_then(|v| match v { AttrValue::Shape(shape) => Some(shape.clone()), _ => None })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::dag::{DagGraph, DataType};

    #[test]
    fn test_shape_inference_add() {
        let mut graph = DagGraph::new("test");
        let v1 = graph.add_value("a", TensorType { dtype: DataType::F32, shape: vec![2, 3] });
        let v2 = graph.add_value("b", TensorType { dtype: DataType::F32, shape: vec![2, 3] });
        let v3 = graph.add_value("c", TensorType { dtype: DataType::F32, shape: vec![] });

        let op_id = graph.add_op("add", vec![v1, v2], vec![v3], HashMap::new());
        graph.set_outputs(vec![v3]);

        ShapeInferencePass::apply(&mut graph).unwrap();

        let out = graph.values.get(&v3).unwrap();
        assert_eq!(out.ty.shape, vec![2, 3]);
    }

    #[test]
    fn test_shape_inference_matmul() {
        let mut graph = DagGraph::new("test");
        let v1 = graph.add_value("a", TensorType { dtype: DataType::F32, shape: vec![2, 3] });
        let v2 = graph.add_value("b", TensorType { dtype: DataType::F32, shape: vec![3, 4] });
        let v3 = graph.add_value("c", TensorType { dtype: DataType::F32, shape: vec![] });

        let mut attrs = HashMap::new();
        let op_id = graph.add_op("matmul", vec![v1, v2], vec![v3], attrs);
        graph.set_outputs(vec![v3]);

        ShapeInferencePass::apply(&mut graph).unwrap();

        let out = graph.values.get(&v3).unwrap();
        assert_eq!(out.ty.shape, vec![2, 4]);
    }
}
