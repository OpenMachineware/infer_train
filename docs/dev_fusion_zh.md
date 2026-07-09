# InferTrain 算子融合开发指南

## 1. 概述

算子融合（Operator Fusion）是优化推理和训练性能的关键技术，通过将多个连续的算子合并为一个组合算子，减少内存访问和内核启动开销。

本指南说明如何为新算子添加融合支持，以及如何设计融合规则。


## 2. 融合框架架构

```
┌─────────────────────────────────────────────────────────────┐
│                    FusionPass                              │
│  ┌─────────────────────────────────────────────────────┐   │
│  │             融合规则注册表                          │   │
│  │  ┌──────────────┐  ┌──────────────┐              │   │
│  │  │ Conv + BN    │  │ Conv + ReLU  │  ...         │   │
│  │  └──────────────┘  └──────────────┘              │   │
│  └─────────────────────────────────────────────────────┘   │
│                          │                                  │
│                          ▼                                  │
│  ┌─────────────────────────────────────────────────────┐   │
│  │              模式匹配引擎                           │   │
│  │  (匹配 DAG 中的连续算子模式)                        │   │
│  └─────────────────────────────────────────────────────┘   │
│                          │                                  │
│                          ▼                                  │
│  ┌─────────────────────────────────────────────────────┐   │
│  │              权重合并引擎                           │   │
│  │  (融合后的权重计算)                                 │   │
│  └─────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```


## 3. 核心概念

### 3.1 融合规则

一个融合规则包含三部分：

| 组件 | 说明 | 示例 |
|------|------|------|
| 模式 (Pattern) | 匹配的算子序列 | Conv2d → BatchNorm2d |
| 条件 (Condition) | 是否可以融合 | 必须同一通道数 |
| 转换 (Transform) | 如何融合 | 合并 BN 参数到 Conv |

### 3.2 融合类型

| 类型 | 说明 | 示例 |
|------|------|------|
| **垂直融合** | 前后算子融合 | Conv + BN → FusedConvBN |
| **水平融合** | 同层多个算子融合 | 多个同形状的 MatMul |
| **算子内融合** | 算子内部操作合并 | Conv 的 bias 和 BN 的 bias |


## 4. 融合规则开发

### 4.1 基本结构

```rust
// src/transform/fusion.rs

pub struct FusionPass;

impl FusionPass {
    pub fn apply(graph: &mut DagGraph) -> bool {
        let mut changed = false;
        let mut to_remove = Vec::new();
        let mut new_ops = Vec::new();

        // 1. 遍历所有算子
        let op_ids: Vec<u64> = graph.ops.keys().cloned().collect();

        for &op_id in &op_ids {
            if to_remove.contains(&op_id) {
                continue;
            }

            let op = match graph.ops.get(&op_id) {
                Some(o) => o,
                None => continue,
            };

            // 2. 尝试匹配融合规则
            if let Some(fused) = try_fuse(graph, op) {
                // 3. 生成融合后算子
                new_ops.push(fused);
                // 4. 标记被融合的算子
                to_remove.push(op_id);
                changed = true;
            }
        }

        // 5. 删除旧算子，添加新算子
        for id in to_remove {
            graph.ops.remove(&id);
        }
        for op in new_ops {
            graph.insert_op(op);
        }

        changed
    }
}
```

### 4.2 融合规则模板

```rust
// src/transform/fusion_rules.rs

/// 融合规则 Trait
pub trait FusionRule {
    /// 匹配模式
    fn matches(&self, graph: &DagGraph, op: &Op) -> Option<FusionMatch>;

    /// 执行融合
    fn fuse(&self, graph: &mut DagGraph, match_info: FusionMatch) -> Result<Op, String>;

    /// 规则名称
    fn name(&self) -> &'static str;
}

/// 匹配结果
pub struct FusionMatch {
    pub target_op_id: u64,           // 主算子 ID
    pub matched_ops: Vec<u64>,       // 匹配到的所有算子 ID
    pub inputs: Vec<u64>,            // 融合后的输入
    pub outputs: Vec<u64>,           // 融合后的输出
    pub attrs: HashMap<String, AttrValue>, // 融合后的属性
    pub weight_map: HashMap<u64, u64>,     // 权重映射
}
```

### 4.3 实现一个新融合规则

以 `Linear + ReLU` 融合为例：

```rust
// src/transform/fusion_rules/linear_relu.rs

use crate::ir::dag::{DagGraph, Op, AttrValue};
use super::{FusionRule, FusionMatch};

pub struct LinearReluFusion;

impl FusionRule for LinearReluFusion {
    fn name(&self) -> &'static str {
        "linear_relu"
    }

    fn matches(&self, graph: &DagGraph, op: &Op) -> Option<FusionMatch> {
        // 1. 检查当前算子是否是 Linear
        if op.op_type != "linear" {
            return None;
        }

        // 2. 检查 Linear 的输出是否只有一个使用者
        let linear_out = op.outputs[0];
        let users = graph.get_users(linear_out);
        if users.len() != 1 {
            return None;
        }

        // 3. 检查下一个算子是否是 ReLU
        let next_id = users[0];
        let next_op = match graph.ops.get(&next_id) {
            Some(o) => o,
            None => return None,
        };

        if next_op.op_type != "relu" {
            return None;
        }

        // 4. 返回匹配结果
        Some(FusionMatch {
            target_op_id: op.id,
            matched_ops: vec![op.id, next_id],
            inputs: op.inputs.clone(),
            outputs: next_op.outputs.clone(),
            attrs: op.attrs.clone(),
            weight_map: HashMap::new(),
        })
    }

    fn fuse(&self, graph: &mut DagGraph, match_info: FusionMatch) -> Result<Op, String> {
        // 创建融合算子
        let fused_op = Op {
            id: 0,
            name: format!("fused_{}_relu", match_info.target_op_id),
            op_type: "fused_linear_relu".to_string(),
            inputs: match_info.inputs,
            outputs: match_info.outputs,
            attrs: match_info.attrs,
        };

        Ok(fused_op)
    }
}
```


## 5. 权重合并

### 5.1 Conv + BN 权重合并公式

```rust
// src/transform/fusion_weight.rs

/// Conv + BN 权重合并
///
/// 原公式:
///   conv_out = conv(x, w, b)
///   bn_out = (conv_out - mean) * gamma / sqrt(var + eps) + beta
///
/// 合并后:
///   w_fused = w * gamma / sqrt(var + eps)
///   b_fused = (b - mean) * gamma / sqrt(var + eps) + beta
///
pub fn fuse_conv_bn_weights(
    conv_weight: &[f32],
    conv_bias: Option<&[f32]>,
    bn_weight: &[f32],
    bn_bias: &[f32],
    bn_mean: &[f32],
    bn_var: &[f32],
    eps: f32,
) -> (Vec<f32>, Vec<f32>) {
    let out_channels = bn_weight.len();
    let elements_per_channel = conv_weight.len() / out_channels;

    let mut fused_weight = conv_weight.to_vec();
    let mut fused_bias = if let Some(b) = conv_bias {
        b.to_vec()
    } else {
        vec![0.0; out_channels]
    };

    for c in 0..out_channels {
        let scale = bn_weight[c] / (bn_var[c] + eps).sqrt();
        let shift = (fused_bias[c] - bn_mean[c]) * scale + bn_bias[c];

        fused_bias[c] = shift;

        let start = c * elements_per_channel;
        let end = start + elements_per_channel;
        for i in start..end {
            fused_weight[i] *= scale;
        }
    }

    (fused_weight, fused_bias)
}
```

### 5.2 Linear + BN 权重合并

```rust
/// Linear + BN 权重合并
///
/// 原公式:
///   linear_out = x @ w.T + b
///   bn_out = (linear_out - mean) * gamma / sqrt(var + eps) + beta
///
/// 合并后:
///   w_fused = w * gamma / sqrt(var + eps)
///   b_fused = (b - mean) * gamma / sqrt(var + eps) + beta
///
pub fn fuse_linear_bn_weights(
    linear_weight: &[f32],
    linear_bias: Option<&[f32]>,
    bn_weight: &[f32],
    bn_bias: &[f32],
    bn_mean: &[f32],
    bn_var: &[f32],
    eps: f32,
) -> (Vec<f32>, Vec<f32>) {
    // 与 Conv + BN 相同
    fuse_conv_bn_weights(linear_weight, linear_bias, bn_weight, bn_bias, bn_mean, bn_var, eps)
}
```

### 5.3 量化权重重合

```rust
/// 量化 Conv + 量化 BN 融合 (保持量化精度)
pub fn fuse_quantized_conv_bn(
    conv_weight: &[i8],
    conv_scale: f32,
    conv_zero: f32,
    bn_weight: &[f32],
    bn_bias: &[f32],
    bn_mean: &[f32],
    bn_var: &[f32],
    eps: f32,
) -> (Vec<i8>, f32, f32) {
    // 1. 反量化
    let fp_weight: Vec<f32> = conv_weight.iter()
        .map(|&x| (x as f32 - conv_zero) * conv_scale)
        .collect();

    // 2. 融合
    let (mut fused_fp_weight, fused_bias) = fuse_conv_bn_weights(
        &fp_weight,
        None,
        bn_weight,
        bn_bias,
        bn_mean,
        bn_var,
        eps,
    );

    // 3. 重新量化
    let max_val = fused_fp_weight.iter()
        .fold(0.0f32, |a, &b| a.max(b.abs()));
    let scale = max_val / 127.0;

    let q_weight: Vec<i8> = fused_fp_weight.iter()
        .map(|&x| (x / scale).round().clamp(-127.0, 127.0) as i8)
        .collect();

    (q_weight, scale, 0.0)
}
```


## 6. 支持的融合模式

| 模式 | 实现状态 | 权重合并 |
|------|----------|----------|
| Conv2d + BatchNorm2d | ✅ 已实现 | ✅ |
| Conv2d + ReLU | ✅ 已实现 | ❌ (无权重) |
| Conv2d + BatchNorm2d + ReLU | ✅ 已实现 | ✅ |
| Linear + ReLU | ⬜ 待实现 | ❌ |
| Linear + BatchNorm | ⬜ 待实现 | ✅ |
| Conv2d + BatchNorm + SiLU | ⬜ 待实现 | ✅ |
| Conv2d + BatchNorm + GELU | ⬜ 待实现 | ✅ |
| MatMul + Add (Bias) | ⬜ 待实现 | ❌ |
| Conv2d + Add (Residual) | ⬜ 待实现 | ❌ |


## 7. 添加新融合模式的步骤

### Step 1: 识别融合模式

```rust
// 示例：Conv2d + SiLU
// 在 graph 中识别 Conv2d → SiLU 模式
```

### Step 2: 检查融合条件

```rust
// 检查条件
fn can_fuse_conv_silu(graph: &DagGraph, conv_id: u64, silu_id: u64) -> bool {
    let conv = graph.get_op(conv_id).unwrap();
    let silu = graph.get_op(silu_id).unwrap();

    // 1. Conv 的输出只被 SiLU 使用
    let users = graph.get_users(conv.outputs[0]);
    if users.len() != 1 || users[0] != silu_id {
        return false;
    }

    // 2. 没有其他依赖
    // 3. 数据类型匹配
    // 4. 设备一致

    true
}
```

### Step 3: 实现融合逻辑

```rust
fn fuse_conv_silu(graph: &mut DagGraph, conv_id: u64, silu_id: u64) -> Result<Op, String> {
    let conv = graph.get_op(conv_id).unwrap();
    let silu = graph.get_op(silu_id).unwrap();

    let fused_op = Op {
        id: 0,
        name: format!("fused_conv_silu_{}", conv_id),
        op_type: "fused_conv_silu".to_string(),
        inputs: conv.inputs.clone(),
        outputs: silu.outputs.clone(),
        attrs: conv.attrs.clone(),
    };

    // 标记删除
    // ...

    Ok(fused_op)
}
```

### Step 4: 注册融合规则

```rust
// src/transform/fusion.rs

fn try_fuse(graph: &mut DagGraph, op_id: u64, op: &Op) -> Option<Op> {
    match op.op_type.as_str() {
        "conv2d" => {
            // 尝试 Conv + BN
            if let Some(fused) = Self::try_fuse_conv_bn(graph, op_id, op) {
                return Some(fused);
            }
            // 尝试 Conv + SiLU (新增)
            if let Some(fused) = Self::try_fuse_conv_silu(graph, op_id, op) {
                return Some(fused);
            }
            // 尝试 Conv + ReLU
            if let Some(fused) = Self::try_fuse_conv_relu(graph, op_id, op) {
                return Some(fused);
            }
        }
        "linear" => {
            // 尝试 Linear + ReLU
            if let Some(fused) = Self::try_fuse_linear_relu(graph, op_id, op) {
                return Some(fused);
            }
        }
        _ => {}
    }
    None
}
```


## 8. 测试融合

```rust
// tests/test_fusion.rs

#[test]
fn test_conv_bn_fusion() {
    let mut graph = create_test_graph();

    // 1. 添加 Conv2d
    let conv_id = graph.add_op("conv2d", vec![input], vec![conv_out], attrs);

    // 2. 添加 BatchNorm2d
    let bn_id = graph.add_op("batchnorm2d", vec![conv_out], vec![out], bn_attrs);

    // 3. 执行融合
    let changed = FusionPass::apply(&mut graph);

    // 4. 验证
    assert!(changed);
    assert!(graph.ops.contains_key("fused_conv_bn"));
    assert!(!graph.ops.contains_key(conv_id));
    assert!(!graph.ops.contains_key(bn_id));

    // 5. 验证权重合并正确
    let fused_weight = graph.constants.get(&fused_weight_id).unwrap();
    // 验证 fused_weight == conv_weight * bn_scale
}
```


## 9. 性能验证

```rust
// benches/bench_fusion.rs

use criterion::*;

fn bench_conv_bn(c: &mut Criterion) {
    c.bench_function("conv_bn_unfused", |b| {
        b.iter(|| {
            // 执行未融合的 Conv + BN
        })
    });

    c.bench_function("conv_bn_fused", |b| {
        b.iter(|| {
            // 执行融合后的 Conv_BN
        })
    });
}
```


## 10. 常见问题

### Q: 融合后精度会下降吗？

A: 理论上不会。数学上等价，但浮点运算顺序不同可能带来微小误差（通常 < 1e-5）。

### Q: 量化模型能融合吗？

A: 可以，但需要先反量化 → 融合 → 重新量化，可能引入额外误差。建议用 QAT (量化感知训练)。

### Q: 融合后的算子如何反向传播？

A: 需要实现 fused op 的 backward，否则训练时无法使用。

### Q: 如何确定哪些算子可以融合？

A: 参考常见融合模式，或通过 profiling 找出热点算子。

### Q: 融合后内存占用如何变化？

A: 通常减少（不需要存储中间结果），但也可能增加（需要存储更多权重）。


## 11. 开发检查清单

- [ ] 识别融合模式（两个或多个连续算子）
- [ ] 检查融合条件（输出只被一个算子使用）
- [ ] 验证数据类型兼容
- [ ] 权重合并逻辑（如果有）
- [ ] 属性合并（如果有）
- [ ] 创建融合后算子
- [ ] 标记删除旧算子
- [ ] 注册到 FusionPass
- [ ] 单元测试验证
- [ ] 性能测试验证
- [ ] 精度验证


## 12. 融合实现模板

```rust
// src/transform/fusion_rules/op1_op2.rs

/// A + B 融合规则
pub struct Op1Op2Fusion;

impl FusionRule for Op1Op2Fusion {
    fn name(&self) -> &'static str {
        "op1_op2"
    }

    fn matches(&self, graph: &DagGraph, op: &Op) -> Option<FusionMatch> {
        // 1. 匹配 Op1
        if op.op_type != "op1" {
            return None;
        }

        // 2. 检查后继是否是 Op2
        let out = op.outputs[0];
        let users = graph.get_users(out);
        if users.len() != 1 {
            return None;
        }

        let next = users[0];
        let next_op = graph.get_op(next)?;
        if next_op.op_type != "op2" {
            return None;
        }

        // 3. 返回匹配
        Some(FusionMatch {
            target_op_id: op.id,
            matched_ops: vec![op.id, next],
            inputs: op.inputs.clone(),
            outputs: next_op.outputs.clone(),
            attrs: op.attrs.clone(),
            weight_map: HashMap::new(),
        })
    }

    fn fuse(&self, graph: &mut DagGraph, match_info: FusionMatch) -> Result<Op, String> {
        // 1. 如果有权重，合并
        // 2. 创建融合算子
        let fused = Op {
            id: 0,
            name: format!("fused_{}", match_info.target_op_id),
            op_type: "fused_op1_op2".to_string(),
            inputs: match_info.inputs,
            outputs: match_info.outputs,
            attrs: match_info.attrs,
        };

        Ok(fused)
    }
}
```

---

## 13. 相关文档

- [dev_ops_zh.md](dev_ops_zh.md) - 算子开发指南
- `src/transform/fusion.rs` - 融合实现
- `tests/test_fusion.rs` - 融合测试
