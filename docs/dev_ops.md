# InferTrain Operator Development Guide


## 1. Overview

This guide explains how to add new operators to the InferTrain engine. An operator consists of three parts:
- Forward propagation (Forward)
- Backward propagation (Backward)
- Quantized version (Quantized)


## 2. Directory Structure

```
src/ops/
├── mod.rs              # Module exports
├── registry.rs         # Operator registry
├── math/               # Math operators
│   ├── mod.rs
│   ├── add.rs
│   └── ...
├── activation/         # Activation functions
├── conv_pool/          # Convolution and pooling
├── normalization/      # Normalization
├── linalg/             # Linear algebra
├── reduction/          # Reduction
├── loss/               # Loss functions
├── embedding_lookup/   # Embedding and lookup
├── attention/          # Attention
├── control_flow/       # Control flow
├── tensor_manip/       # Tensor manipulation
├── data_gen/           # Data generation
└── cast/               # Type conversion
```


## 3. Operator Template

### 3.1 Basic Operator Template

```rust
// src/ops/{category}/{name}.rs

use rayon::prelude::*;
use crate::dtype::DType;
use crate::tensor::Tensor;
use crate::ops::registry::{Operator, OpAttrs, DeviceType};

// ============================================================
// 1. Generic Forward (supports F32/F64/F16/BF16)
// ============================================================

pub fn {op_name}<T: DType + Send + Sync>(
    a: &Tensor<T>,
    b: &Tensor<T>,
) -> Tensor<T> {
    assert_eq!(a.shape(), b.shape(), "Shape mismatch in {op_name}");
    
    let data: Vec<T> = a.data()
        .par_iter()
        .zip(b.data().par_iter())
        .map(|(&x, &y)| x {operator} y)
        .collect();
    
    Tensor::new(data, a.shape())
}

// ============================================================
// 2. Generic Backward
// ============================================================

pub fn {op_name}_backward<T: DType>(
    grad_output: &Tensor<T>,
    a: &Tensor<T>,
    b: &Tensor<T>,
) -> Vec<Tensor<T>> {
    // Gradient formulas
    // ∂L/∂a = ...
    // ∂L/∂b = ...
    vec![grad_a, grad_b]
}

// ============================================================
// 3. Quantized Forward (I8)
// ============================================================

pub fn quantized_{op_name}(
    a: &Tensor<i8>,
    b: &Tensor<i8>,
) -> Tensor<i8> {
    let scale_a = a.scale().unwrap_or(1.0);
    let zero_a = a.zero_point().unwrap_or(0.0);
    let scale_b = b.scale().unwrap_or(1.0);
    let zero_b = b.zero_point().unwrap_or(0.0);

    let scale = scale_a;
    let zero = zero_a;

    let result_fp: Vec<f32> = a.data()
        .iter()
        .zip(b.data().iter())
        .map(|(&x, &y)| {
            let x_fp = (x as f32 - zero_a) * scale_a;
            let y_fp = (y as f32 - zero_b) * scale_b;
            x_fp {operator} y_fp
        })
        .collect();

    let data: Vec<i8> = result_fp.iter()
        .map(|&v| ((v / scale) + zero).round().clamp(-128.0, 127.0) as i8)
        .collect();

    Tensor::<i8>::new_quantized(data, a.shape(), scale, zero)
}

// ============================================================
// 4. Quantized Backward
// ============================================================

pub fn quantized_{op_name}_backward(
    grad_output: &Tensor<i8>,
    a: &Tensor<i8>,
    b: &Tensor<i8>,
) -> Vec<Tensor<i8>> {
    // Quantized gradient (simplified)
    vec![grad_output.clone(), grad_output.clone()]
}

// ============================================================
// 5. Operator Trait Implementation
// ============================================================

pub struct {OpName}Op;

impl<T: DType + Send + Sync> Operator<T> for {OpName}Op {
    fn name(&self) -> &'static str {
        "{op_name}"
    }

    fn forward(&self, inputs: &[&Tensor<T>], _attrs: &OpAttrs) -> Tensor<T> {
        assert_eq!(inputs.len(), 2);
        {op_name}(inputs[0], inputs[1])
    }

    fn backward(
        &self,
        grad_output: &Tensor<T>,
        inputs: &[&Tensor<T>],
        _attrs: &OpAttrs,
    ) -> Vec<Tensor<T>> {
        assert_eq!(inputs.len(), 2);
        {op_name}_backward(grad_output, inputs[0], inputs[1])
    }
}

pub struct Quantized{OpName}Op;

impl Operator<i8> for Quantized{OpName}Op {
    fn name(&self) -> &'static str {
        "quantized_{op_name}"
    }

    fn forward(&self, inputs: &[&Tensor<i8>], _attrs: &OpAttrs) -> Tensor<i8> {
        assert_eq!(inputs.len(), 2);
        quantized_{op_name}(inputs[0], inputs[1])
    }

    fn backward(
        &self,
        grad_output: &Tensor<i8>,
        inputs: &[&Tensor<i8>],
        _attrs: &OpAttrs,
    ) -> Vec<Tensor<i8>> {
        assert_eq!(inputs.len(), 2);
        quantized_{op_name}_backward(grad_output, inputs[0], inputs[1])
    }

    fn supports_quantized(&self) -> bool {
        true
    }
}

// ============================================================
// 6. Tests
// ============================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_{op_name}_f32() {
        let a = Tensor::new(vec![1.0, 2.0, 3.0], &[3]);
        let b = Tensor::new(vec![4.0, 5.0, 6.0], &[3]);
        let c = {op_name}(&a, &b);
        assert_eq!(c.data(), &[5.0, 7.0, 9.0]);
    }

    #[test]
    fn test_{op_name}_backward() {
        let grad = Tensor::new(vec![1.0, 2.0, 3.0], &[3]);
        let a = Tensor::new(vec![1.0, 1.0, 1.0], &[3]);
        let b = Tensor::new(vec![2.0, 2.0, 2.0], &[3]);
        let grads = {op_name}_backward(&grad, &a, &b);
        // Verify gradients
    }

    #[test]
    fn test_quantized_{op_name}() {
        let a = Tensor::<i8>::new_quantized(vec![1, 2, 3], &[3], 0.1, 0.0);
        let b = Tensor::<i8>::new_quantized(vec![4, 5, 6], &[3], 0.1, 0.0);
        let c = quantized_{op_name}(&a, &b);
        assert_eq!(c.data(), &[5, 7, 9]);
    }
}
```


## 4. Registering Operators

### 4.1 Update `mod.rs`

```rust
// src/ops/{category}/mod.rs

pub mod {name};

pub use {name}::{ 
    {name}, {name}_backward, 
    quantized_{name}, quantized_{name}_backward,
    {OpName}Op, Quantized{OpName}Op
};
```

### 4.2 Update Parent Module

```rust
// src/ops/mod.rs

pub mod {category};

pub use {category}::{name};
```


## 5. Operator Type Templates

### 5.1 Unary Operator

```rust
pub fn op_name<T: DType + Send + Sync>(a: &Tensor<T>) -> Tensor<T> {
    let data: Vec<T> = a.data()
        .par_iter()
        .map(|&x| T::from_f32(x.to_f32().func()))
        .collect();
    Tensor::new(data, a.shape())
}

pub fn op_name_backward<T: DType>(
    grad_output: &Tensor<T>,
    a: &Tensor<T>,
) -> Vec<Tensor<T>> {
    // ∂L/∂a = ∂L/∂output * derivative
    vec![grad]
}
```

### 5.2 Reduction Operator

```rust
pub fn op_name<T: DType + Send + Sync>(
    a: &Tensor<T>,
    dim: usize,
    keepdim: bool,
) -> Tensor<T> {
    // Reduction logic
}
```

### 5.3 Tensor Manipulation

```rust
pub fn op_name<T: DType + Send + Sync>(
    a: &Tensor<T>,
    new_shape: &[usize],
) -> Tensor<T> {
    // Shape transformation logic
}
```


## 6. Backward Propagation Formula Reference

| Operator | ∂L/∂x | ∂L/∂w |
|------|-------|-------|
| y = x + w | ∂L/∂y | ∂L/∂y |
| y = x * w | ∂L/∂y * w | ∂L/∂y * x |
| y = x / w | ∂L/∂y / w | -∂L/∂y * x / w² |
| y = relu(x) | ∂L/∂y if x > 0 else 0 | - |
| y = sigmoid(x) | ∂L/∂y * y * (1-y) | - |
| y = tanh(x) | ∂L/∂y * (1 - y²) | - |
| y = exp(x) | ∂L/∂y * y | - |
| y = log(x) | ∂L/∂y / x | - |
| y = x @ w | ∂L/∂y @ wᵀ | xᵀ @ ∂L/∂y |


## 7. Checklist

- [ ] Forward implementation (generic T: DType)
- [ ] Backward implementation (generic T: DType)
- [ ] Quantized Forward (i8 version)
- [ ] Quantized Backward (i8 version)
- [ ] Operator Trait implementation (both versions)
- [ ] Test cases (f32, bf16, quantized)
- [ ] Module exports (mod.rs)
- [ ] Documentation comments


## 8. FAQ

### Q: How to handle different DTypes?

A: Convert to f32 using `x.to_f32()`, compute, then convert back using `T::from_f32()`.

### Q: How many gradients should Backward return?

A: Return `Vec<Tensor<T>>` with the same order as inputs.

### Q: How to handle scale/zero_point in quantized version?

A: Dequantize → float computation → requantize.

### Q: What if operators need attributes?

A: Read from `attrs` in `forward`: `attrs.get_int("key").unwrap_or(default)`