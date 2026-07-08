// src/autograd/backward.rs

use std::collections::HashMap;

use crate::tensor::Tensor;
use crate::ops::math::*;
use crate::ops::activation::*;
use crate::ops::tensor_manip::reshape::reshape;
use crate::ops::conv_pool::{conv2d_backward, max_pool_backward, avg_pool_backward};
use crate::ops::normalization::{batch_norm_backward, layer_norm_backward};
use crate::ops::rms_norm_backward;
use crate::ops::linalg::matmul_backward;

use super::tape::{Tape, TapeEntry};

// ============================================================
// 梯度函数 Trait
// ============================================================

pub trait GradFn {
    fn backward(&self, grad_output: &Tensor<f32>, tape: &Tape) -> Vec<(u64, Tensor<f32>)>;
}

// ============================================================
// 梯度计算
// ============================================================

pub fn backward(
    loss: &Tensor<f32>,
    tape: &Tape,
    values: &HashMap<u64, Tensor<f32>>,
) -> HashMap<u64, Tensor<f32>> {
    let mut grads: HashMap<u64, Tensor<f32>> = HashMap::new();

    if let Some(last_entry) = tape.entries().last() {
        let output_id = last_entry.output_id();
        let grad = Tensor::new(vec![1.0f32], &[1]);
        grads.insert(output_id, grad);

        let order = tape.reverse_order(output_id);
        for idx in order {
            let entry = &tape.entries()[idx];
            let output_id = entry.output_id();
            let grad = grads.get(&output_id);
            if grad.is_none() { continue; }
            let grad = grad.unwrap();

            let input_grads = backward_entry(entry, grad, values, tape);
            for (input_id, g) in input_grads {
                grads.entry(input_id)
                    .and_modify(|existing| {
                        // 梯度累积
                        let existing_data = existing.data_mut();
                        let grad_data = g.data();
                        for i in 0..existing_data.len().min(grad_data.len()) {
                            existing_data[i] += grad_data[i];
                        }
                    })
                    .or_insert(g);
            }
        }
    }

    grads
}

// ============================================================
// 单个 Entry 的反向传播
// ============================================================

fn backward_entry(
    entry: &TapeEntry,
    grad_output: &Tensor<f32>,
    values: &HashMap<u64, Tensor<f32>>,
    tape: &Tape,
) -> Vec<(u64, Tensor<f32>)> {
    match entry {
        // ============================================================
        // 数学算子
        // ============================================================
        TapeEntry::Add { input_a, input_b, .. } => {
            let grads = add_backward(grad_output,
                                     values.get(input_a).unwrap(),
                                     values.get(input_b).unwrap(),
            );
            vec![(*input_a, grads[0].clone()), (*input_b, grads[1].clone())]
        }
        TapeEntry::Sub { input_a, input_b, .. } => {
            let grads = sub_backward(grad_output,
                                     values.get(input_a).unwrap(),
                                     values.get(input_b).unwrap(),
            );
            vec![(*input_a, grads[0].clone()), (*input_b, grads[1].clone())]
        }
        TapeEntry::Mul { input_a, input_b, .. } => {
            let grads = mul_backward(grad_output,
                                     values.get(input_a).unwrap(),
                                     values.get(input_b).unwrap(),
            );
            vec![(*input_a, grads[0].clone()), (*input_b, grads[1].clone())]
        }
        TapeEntry::Div { input_a, input_b, .. } => {
            let grads = div_backward(grad_output,
                                     values.get(input_a).unwrap(),
                                     values.get(input_b).unwrap(),
            );
            vec![(*input_a, grads[0].clone()), (*input_b, grads[1].clone())]
        }

        TapeEntry::Pow { input, exponent, .. } => {
            let x = values.get(input).unwrap();
            // ∂(x^e)/∂x = e * x^(e-1)
            let e_minus_1 = *exponent - 1.0;
            let grad_input = mul(grad_output, &pow(x, &Tensor::new(vec![e_minus_1], &[1])));
            vec![(*input, mul(&Tensor::new(vec![*exponent], &[1]), &grad_input))]
        }
        TapeEntry::Pow { input, exponent, .. } => {
            let grads = pow_backward(grad_output,
                                     values.get(input).unwrap(),
                                     &Tensor::new(vec![*exponent], &[1]),
            );
            vec![(*input, grads[0].clone())]
        }
        TapeEntry::Exp { input, .. } => {
            let grads = exp_backward(grad_output, values.get(input).unwrap());
            vec![(*input, grads[0].clone())]
        }
        TapeEntry::Sqrt { input, .. } => {
            let grads = sqrt_backward(grad_output, values.get(input).unwrap());
            vec![(*input, grads[0].clone())]
        }
        TapeEntry::Log { input, .. } => {
            let grads = log_backward(grad_output, values.get(input).unwrap());
            vec![(*input, grads[0].clone())]
        }
        TapeEntry::Neg { input, .. } => {
            let grads = neg_backward(grad_output, values.get(input).unwrap());
            vec![(*input, grads[0].clone())]
        }

        // ============================================================
        // 线性代数
        // ============================================================
        TapeEntry::MatMul { input_a, input_b, .. } => {
            let grads = matmul_backward(grad_output,
                                        values.get(input_a).unwrap(),
                                        values.get(input_b).unwrap(),
            );
            vec![(*input_a, grads[0].clone()), (*input_b, grads[1].clone())]
        }

        // ============================================================
        // 激活函数
        // ============================================================
        TapeEntry::Relu { input, .. } => {
            let grads = relu_backward(grad_output, values.get(input).unwrap());
            vec![(*input, grads[0].clone())]
        }
        TapeEntry::Sigmoid { input, .. } => {
            let grads = sigmoid_backward(grad_output, values.get(input).unwrap());
            vec![(*input, grads[0].clone())]
        }
        TapeEntry::Tanh { input, .. } => {
            let grads = tanh_backward(grad_output, values.get(input).unwrap());
            vec![(*input, grads[0].clone())]
        }
        TapeEntry::Softmax { input, .. } => {
            let grads = softmax_backward(grad_output, values.get(input).unwrap());
            vec![(*input, grads[0].clone())]
        }
        // ============================================================
        // 卷积
        // ============================================================
        TapeEntry::Conv2d { input, weight, bias, stride, padding, dilation, groups, .. } => {
            let x = values.get(input).unwrap();
            let w = values.get(weight).unwrap();
            let grads = conv2d_backward(grad_output, x, w, *stride, *padding, *dilation, *groups);
            let mut result = vec![(*input, grads[0].clone()), (*weight, grads[1].clone())];
            if let Some(b) = bias {
                result.push((*b, grads[2].clone()));
            }
            result
        }

        // ============================================================
        // 池化
        // ============================================================
        TapeEntry::MaxPool { input, kernel_size, stride, padding, .. } => {
            let x = values.get(input).unwrap();
            let grad_input = max_pool_backward(grad_output, x, *kernel_size, *stride, *padding);
            vec![(*input, grad_input)]
        }
        TapeEntry::AvgPool { input, kernel_size, stride, padding, .. } => {
            let grad_input = avg_pool_backward(grad_output, *kernel_size, *stride, *padding);
            vec![(*input, grad_input)]
        }

        // ============================================================
        // 归一化
        // ============================================================
        TapeEntry::BatchNorm { input, weight, bias, running_mean, running_var, eps, .. } => {
            let x = values.get(input).unwrap();
            let w = values.get(weight).unwrap();
            let rm = values.get(running_mean).unwrap();
            let rv = values.get(running_var).unwrap();
            let grads = batch_norm_backward(grad_output, x, w, rm, rv, *eps);
            vec![
                (*input, grads[0].clone()),
                (*weight, grads[1].clone()),
                (*bias, grads[2].clone()),
            ]
        }
        TapeEntry::LayerNorm { input, weight, bias, eps, .. } => {
            let x = values.get(input).unwrap();
            let w = values.get(weight).unwrap();
            let grads = layer_norm_backward(grad_output, x, w, *eps);
            vec![
                (*input, grads[0].clone()),
                (*weight, grads[1].clone()),
                (*bias, grads[2].clone()),
            ]
        }
        TapeEntry::RMSNorm { input, weight, eps, .. } => {
            let x = values.get(input).unwrap();
            let w = values.get(weight).unwrap();
            let grads = rms_norm_backward(grad_output, x, w, *eps);
            vec![
                (*input, grads[0].clone()),
                (*weight, grads[1].clone()),
            ]
        }

        // ============================================================
        // 张量操作
        // ============================================================
        TapeEntry::Reshape { input, output, new_shape: _ } => {
            let original_shape = values.get(input).unwrap().shape();
            let grad_input = crate::ops::tensor_manip::reshape::reshape(grad_output, original_shape);
            vec![(*input, grad_input)]
        }
        TapeEntry::Concat { inputs, .. } => {
            let mut result = Vec::new();
            let mut offset = 0;
            for &input_id in inputs {
                let input_tensor = values.get(&input_id).unwrap();
                let len = input_tensor.len();
                let grad_slice = grad_output.data()[offset..offset + len].to_vec();
                let grad = Tensor::new(grad_slice, input_tensor.shape());
                result.push((input_id, grad));
                offset += len;
            }
            result
        }
        TapeEntry::Slice { input, dim, start, end, step, .. } => {
            // slice 的 backward：把梯度填回原始形状
            let original_shape = values.get(input).unwrap().shape();
            let mut grad_input = vec![0.0f32; original_shape.iter().product()];
            // TODO: 把 grad_output 填回对应的位置
            vec![(*input, Tensor::new(grad_input, original_shape))]
        }
        TapeEntry::Squeeze { input, dim, .. } => {
            let original_shape = values.get(input).unwrap().shape();
            let mut new_shape = original_shape.to_vec();
            if let Some(d) = dim {
                new_shape.insert(*d, 1);
            } else {
                // 如果 dim 是 None，无法恢复，返回原梯度
                return vec![(*input, grad_output.clone())];
            }
            let grad_input = reshape(grad_output, &new_shape);
            vec![(*input, grad_input)]
        }
        TapeEntry::Unsqueeze { input, dim, .. } => {
            let mut new_shape = values.get(input).unwrap().shape().to_vec();
            new_shape.remove(*dim);
            let grad_input = reshape(grad_output, &new_shape);
            vec![(*input, grad_input)]
        }
        TapeEntry::ReduceSum { input, dim, keepdim, .. } => {
            let x = values.get(input).unwrap();
            let mut grad_input = grad_output.clone();
            let shape = x.shape();
            if !keepdim {
                let mut new_shape = shape.to_vec();
                new_shape.insert(*dim, 1);
                grad_input = reshape(&grad_input, &new_shape);
            }
            // 广播到原始形状
            let mut broadcast_shape = shape.to_vec();
            for (i, &dim_size) in broadcast_shape.iter().enumerate() {
                if i == *dim && grad_input.shape()[i] != dim_size {
                    // 需要广播
                }
            }
            vec![(*input, grad_input)]
        }
        TapeEntry::ReduceMean { input, dim, keepdim, .. } => {
            let x = values.get(input).unwrap();
            let dim_size = x.shape()[*dim] as f32;
            let mut grad_input = grad_output.clone();
            for v in grad_input.data_mut() {
                *v /= dim_size;
            }
            let shape = x.shape();
            if !keepdim {
                let mut new_shape = shape.to_vec();
                new_shape.insert(*dim, 1);
                grad_input = reshape(&grad_input, &new_shape);
            }
            vec![(*input, grad_input)]
        }

        // ============================================================
        // 其他
        // ============================================================
        TapeEntry::Select { condition, true_val, false_val, .. } => {
            let cond = values.get(condition).unwrap();
            let grad_true = grad_output.clone();
            let grad_false = grad_output.clone();
            vec![
                (*condition, Tensor::new(vec![0.0f32], &[1])),
                (*true_val, grad_true),
                (*false_val, grad_false),
            ]
        }
        TapeEntry::Embedding { indices, weight, .. } => {
            // embedding 的 backward：梯度传给 weight
            let grad_weight = grad_output.clone();
            vec![(*weight, grad_weight)]
        }

        // ============================================================
        // 参数/输入/常量 (没有反向传播)
        // ============================================================
        TapeEntry::Parameter { .. } => vec![],
        TapeEntry::Input { .. } => vec![],
        TapeEntry::Constant { .. } => vec![],

        // ============================================================
        // 未实现的算子 (TODO)
        // ============================================================
        _ => {
            eprintln!("Warning: backward not implemented for {:?}", entry);
            vec![]
        }
    }
}
