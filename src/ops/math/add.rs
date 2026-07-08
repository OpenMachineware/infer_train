use crate::dtype::DType;
use crate::tensor::Tensor;
use rayon::prelude::*;
// use crate::ops::registry::{Operator, OpAttrs, DeviceType};
use crate::ops::registry::{OpAttrs, Operator};

// ============================================================
// 1. 浮点泛型 Forward
// ============================================================

pub fn add<T: DType + Send + Sync>(a: &Tensor<T>, b: &Tensor<T>) -> Tensor<T> {
    assert_eq!(a.shape(), b.shape(), "Shape mismatch in add");

    let data: Vec<T> = a
        .data()
        .par_iter()
        .zip(b.data().par_iter())
        .map(|(&x, &y)| x + y)
        .collect();

    Tensor::new(data, a.shape())
}

// ============================================================
// 2. 浮点泛型 Backward
// ============================================================

pub fn add_backward<T: DType>(
    grad_output: &Tensor<T>,
    _a: &Tensor<T>,
    _b: &Tensor<T>,
) -> Vec<Tensor<T>> {
    vec![grad_output.clone(), grad_output.clone()]
}

// ============================================================
// 3. 量化 Forward
// ============================================================

pub fn quantized_add(a: &Tensor<i8>, b: &Tensor<i8>) -> Tensor<i8> {
    assert_eq!(a.shape(), b.shape(), "Shape mismatch in quantized_add");

    let scale_a = a.scale().unwrap_or(1.0);
    let zero_a = a.zero_point().unwrap_or(0.0);
    let scale_b = b.scale().unwrap_or(1.0);
    let zero_b = b.zero_point().unwrap_or(0.0);

    let scale = scale_a;
    let zero = zero_a;

    let result_fp: Vec<f32> = a
        .data()
        .iter()
        .zip(b.data().iter())
        .map(|(&x, &y)| {
            let x_fp = (x as f32 - zero_a) * scale_a;
            let y_fp = (y as f32 - zero_b) * scale_b;
            x_fp + y_fp
        })
        .collect();

    let data: Vec<i8> = result_fp
        .iter()
        .map(|&v| ((v / scale) + zero).round().clamp(-128.0, 127.0) as i8)
        .collect();

    Tensor::<i8>::new_quantized(data, a.shape(), scale, zero)
}

// ============================================================
// 4. 量化 Backward
// ============================================================

pub fn quantized_add_backward(
    grad_output: &Tensor<i8>,
    _a: &Tensor<i8>,
    _b: &Tensor<i8>,
) -> Vec<Tensor<i8>> {
    vec![grad_output.clone(), grad_output.clone()]
}

// ============================================================
// 5. Operator Trait 实现（CPU 版本）
// ============================================================

pub struct AddOp;

impl<T: DType + Send + Sync> Operator<T> for AddOp {
    fn name(&self) -> &'static str {
        "add"
    }

    fn forward(&self, inputs: &[&Tensor<T>], _attrs: &OpAttrs) -> Tensor<T> {
        assert_eq!(inputs.len(), 2, "add requires 2 inputs");
        add(inputs[0], inputs[1])
    }

    fn backward(
        &self,
        grad_output: &Tensor<T>,
        inputs: &[&Tensor<T>],
        _attrs: &OpAttrs,
    ) -> Vec<Tensor<T>> {
        assert_eq!(inputs.len(), 2, "add requires 2 inputs");
        add_backward(grad_output, inputs[0], inputs[1])
    }
}

// ============================================================
// 6. 量化 Operator Trait 实现
// ============================================================

pub struct QuantizedAddOp;

impl Operator<i8> for QuantizedAddOp {
    fn name(&self) -> &'static str {
        "quantized_add"
    }

    fn forward(&self, inputs: &[&Tensor<i8>], _attrs: &OpAttrs) -> Tensor<i8> {
        assert_eq!(inputs.len(), 2, "quantized_add requires 2 inputs");
        quantized_add(inputs[0], inputs[1])
    }

    fn backward(
        &self,
        grad_output: &Tensor<i8>,
        inputs: &[&Tensor<i8>],
        _attrs: &OpAttrs,
    ) -> Vec<Tensor<i8>> {
        assert_eq!(inputs.len(), 2, "quantized_add requires 2 inputs");
        quantized_add_backward(grad_output, inputs[0], inputs[1])
    }

    fn supports_quantized(&self) -> bool {
        true
    }
}

// ============================================================
// 7. 测试
// ============================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_add_f32() {
        let a = Tensor::new(vec![1.0, 2.0, 3.0], &[3]);
        let b = Tensor::new(vec![4.0, 5.0, 6.0], &[3]);
        let c = add(&a, &b);
        assert_eq!(c.data(), &[5.0, 7.0, 9.0]);
    }

    #[test]
    fn test_add_bf16() {
        use half::bf16;
        let a =
            Tensor::new(vec![bf16::from_f32(1.0), bf16::from_f32(2.0)], &[2]);
        let b =
            Tensor::new(vec![bf16::from_f32(3.0), bf16::from_f32(4.0)], &[2]);
        let c = add(&a, &b);
        assert_eq!(c.data()[0].to_f32(), 4.0);
        assert_eq!(c.data()[1].to_f32(), 6.0);
    }

    #[test]
    fn test_quantized_add() {
        let a = Tensor::<i8>::new_quantized(vec![1, 2, 3], &[3], 0.1, 0.0);
        let b = Tensor::<i8>::new_quantized(vec![4, 5, 6], &[3], 0.1, 0.0);
        let c = quantized_add(&a, &b);
        assert_eq!(c.data(), &[5, 7, 9]);
        // println!("quantized_add test passed!");
    }

    #[test]
    fn test_add_backward() {
        let grad = Tensor::new(vec![1.0, 2.0, 3.0], &[3]);
        let a = Tensor::new(vec![1.0, 1.0, 1.0], &[3]);
        let b = Tensor::new(vec![2.0, 2.0, 2.0], &[3]);
        let grads = add_backward(&grad, &a, &b);
        assert_eq!(grads[0].data(), &[1.0, 2.0, 3.0]);
        assert_eq!(grads[1].data(), &[1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_add_op_trait() {
        let op = AddOp;
        let a = Tensor::new(vec![1.0, 2.0], &[2]);
        let b = Tensor::new(vec![3.0, 4.0], &[2]);
        let inputs = vec![&a, &b];
        let c = op.forward(&inputs, &OpAttrs::new());
        assert_eq!(c.data(), &[4.0, 6.0]);
    }
}
