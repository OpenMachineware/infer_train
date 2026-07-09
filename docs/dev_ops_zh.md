

# InferTrain 算子添加指南


## 1. 概述

本指南说明如何为 InferTrain 引擎添加新的算子。算子包括三个部分：
- 前向传播 (Forward)
- 反向传播 (Backward) 
- 量化版本 (Quantized)


## 2. 目录结构

```
src/ops/
├── mod.rs              # 模块导出
├── registry.rs         # 算子注册表
├── math/               # 数学算子
│   ├── mod.rs
│   ├── add.rs
│   └── ...
├── activation/         # 激活函数
├── conv_pool/          # 卷积与池化
├── normalization/      # 归一化
├── linalg/             # 线性代数
├── reduction/          # 归约
├── loss/               # 损失函数
├── embedding_lookup/   # 嵌入与查找
├── attention/          # 注意力
├── control_flow/       # 控制流
├── tensor_manip/       # 张量操作
├── data_gen/           # 数据生成
└── cast/               # 类型转换
```


## 3. 算子模板

### 3.1 基础算子模板

```rust
// src/ops/{category}/{name}.rs

use rayon::prelude::*;
use crate::dtype::DType;
use crate::tensor::Tensor;
use crate::ops::registry::{Operator, OpAttrs, DeviceType};

// ============================================================
// 1. 泛型 Forward (支持 F32/F64/F16/BF16)
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
// 2. 泛型 Backward
// ============================================================

pub fn {op_name}_backward<T: DType>(
    grad_output: &Tensor<T>,
    a: &Tensor<T>,
    b: &Tensor<T>,
) -> Vec<Tensor<T>> {
    // 梯度公式
    // ∂L/∂a = ...
    // ∂L/∂b = ...
    vec![grad_a, grad_b]
}

// ============================================================
// 3. 量化 Forward (I8)
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
// 4. 量化 Backward
// ============================================================

pub fn quantized_{op_name}_backward(
    grad_output: &Tensor<i8>,
    a: &Tensor<i8>,
    b: &Tensor<i8>,
) -> Vec<Tensor<i8>> {
    // 量化梯度（简化版）
    vec![grad_output.clone(), grad_output.clone()]
}

// ============================================================
// 5. Operator Trait 实现
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
// 6. 测试
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
        // 验证梯度
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


## 4. 注册算子

### 4.1 更新 `mod.rs`

```rust
// src/ops/{category}/mod.rs

pub mod {name};

pub use {name}::{ 
    {name}, {name}_backward, 
    quantized_{name}, quantized_{name}_backward,
    {OpName}Op, Quantized{OpName}Op
};
```

### 4.2 更新父模块

```rust
// src/ops/mod.rs

pub mod {category};

pub use {category}::{name};
```


## 5. 算子类型模板

### 5.1 一元算子 (Unary Op)

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

### 5.2 归约算子 (Reduction Op)

```rust
pub fn op_name<T: DType + Send + Sync>(
    a: &Tensor<T>,
    dim: usize,
    keepdim: bool,
) -> Tensor<T> {
    // 归约逻辑
}
```

### 5.3 张量操作 (Tensor Manipulation)

```rust
pub fn op_name<T: DType + Send + Sync>(
    a: &Tensor<T>,
    new_shape: &[usize],
) -> Tensor<T> {
    // 形状变换逻辑
}
```


## 6. 反向传播公式参考

| 算子 | ∂L/∂x | ∂L/∂w |
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


## 7. 检查清单

- [ ] Forward 实现（泛型 T: DType）
- [ ] Backward 实现（泛型 T: DType）
- [ ] Quantized Forward（i8 版本）
- [ ] Quantized Backward（i8 版本）
- [ ] Operator Trait 实现（两个版本）
- [ ] 测试用例（f32, bf16, quantized）
- [ ] 模块导出（mod.rs）
- [ ] 文档注释


## 8. 常见问题

### Q: 如何处理不同 DType？

A: 用 `x.to_f32()` 转为 f32 计算，再用 `T::from_f32()` 转回。

### Q: Backward 需要返回几个梯度？

A: 返回 Vec<Tensor<T>>，顺序与 inputs 相同。

### Q: 量化版本的 scale/zero_point 如何处理？

A: 反量化 → 浮点计算 → 重新量化。

### Q: 算子需要属性 (attrs) 怎么办？

A: 在 `forward` 中从 `attrs` 读取：`attrs.get_int("key").unwrap_or(default)`

