# InferTrain Operator Fusion Development Guide

## 1. Overview

Operator Fusion is a key technique to optimize inference and training performance. By merging multiple consecutive operators into a combined operator, it reduces memory access and kernel launch overhead.

This guide explains how to add fusion support for new operators and how to design fusion rules.


## 2. Fusion Framework Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    FusionPass                              │
│  ┌─────────────────────────────────────────────────────┐   │
│  │             Fusion Rule Registry                    │   │
│  │  ┌──────────────┐  ┌──────────────┐              │   │
│  │  │ Conv + BN    │  │ Conv + ReLU  │  ...         │   │
│  │  └──────────────┘  └──────────────┘              │   │
│  └─────────────────────────────────────────────────────┘   │
│                          │                                  │
│                          ▼                                  │
│  ┌─────────────────────────────────────────────────────┐   │
│  │              Pattern Matching Engine                │   │
│  │  (Match consecutive operator patterns in DAG)       │   │
│  └─────────────────────────────────────────────────────┘   │
│                          │                                  │
│                          ▼                                  │
│  ┌─────────────────────────────────────────────────────┐   │
│  │              Weight Merging Engine                  │   │
│  │  (Weight computation after fusion)                 │   │
│  └─────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```


## 3. Core Concepts

### 3.1 Fusion Rule

A fusion rule consists of three parts:

| Component | Description | Example |
|------|------|------|
| Pattern | Matched operator sequence | Conv2d → BatchNorm2d |
| Condition | Whether fusion is possible | Must have same channel count |
| Transform | How to fuse | Merge BN parameters into Conv |

### 3.2 Fusion Types

| Type | Description | Example |
|------|------|------|
| **Vertical Fusion** | Chain operators | Conv + BN → FusedConvBN |
| **Horizontal Fusion** | Multiple operators at same level | Multiple MatMul with same shape |
| **Intra-operator Fusion** | Merge operations within operator | Conv bias and BN bias |


## 4. Fusion Rule Development

### 4.1 Basic Structure

```rust
// src/transform/fusion.rs

pub struct FusionPass;

impl FusionPass {
    pub fn apply(graph: &mut DagGraph) -> bool {
        let mut changed = false;
        let mut to_remove = Vec::new();
        let mut new_ops = Vec::new();

        // 1. Iterate all operators
        let op_ids: Vec<u64> = graph.ops.keys().cloned().collect();

        for &op_id in &op_ids {
            if to_remove.contains(&op_id) {
                continue;
            }

            let op = match graph.ops.get(&op_id) {
                Some(o) => o,
                None => continue,
            };

            // 2. Try matching fusion rules
            if let Some(fused) = try_fuse(graph, op) {
                // 3. Generate fused operator
                new_ops.push(fused);
                // 4. Mark fused operators for removal
                to_remove.push(op_id);
                changed = true;
            }
        }

        // 5. Remove old operators, add new operators
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

### 4.2 Fusion Rule Template

```rust
// src/transform/fusion_rules.rs

/// Fusion Rule Trait
pub trait FusionRule {
    /// Match pattern
    fn matches(&self, graph: &DagGraph, op: &Op) -> Option<FusionMatch>;

    /// Execute fusion
    fn fuse(&self, graph: &mut DagGraph, match_info: FusionMatch) -> Result<Op, String>;

    /// Rule name
    fn name(&self) -> &'static str;
}

/// Match result
pub struct FusionMatch {
    pub target_op_id: u64,           // Main operator ID
    pub matched_ops: Vec<u64>,       // All matched operator IDs
    pub inputs: Vec<u64>,            // Inputs after fusion
    pub outputs: Vec<u64>,           // Outputs after fusion
    pub attrs: HashMap<String, AttrValue>, // Attributes after fusion
    pub weight_map: HashMap<u64, u64>,     // Weight mapping
}
```

### 4.3 Implementing a New Fusion Rule

Taking `Linear + ReLU` fusion as an example:

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
        // 1. Check if current operator is Linear
        if op.op_type != "linear" {
            return None;
        }

        // 2. Check if Linear output has only one user
        let linear_out = op.outputs[0];
        let users = graph.get_users(linear_out);
        if users.len() != 1 {
            return None;
        }

        // 3. Check if next operator is ReLU
        let next_id = users[0];
        let next_op = match graph.ops.get(&next_id) {
            Some(o) => o,
            None => return None,
        };

        if next_op.op_type != "relu" {
            return None;
        }

        // 4. Return match result
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
        // Create fused operator
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


## 5. Weight Merging

### 5.1 Conv + BN Weight Merging Formula

```rust
// src/transform/fusion_weight.rs

/// Conv + BN weight merging
///
/// Original formula:
///   conv_out = conv(x, w, b)
///   bn_out = (conv_out - mean) * gamma / sqrt(var + eps) + beta
///
/// After merging:
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

### 5.2 Linear + BN Weight Merging

```rust
/// Linear + BN weight merging
///
/// Original formula:
///   linear_out = x @ w.T + b
///   bn_out = (linear_out - mean) * gamma / sqrt(var + eps) + beta
///
/// After merging:
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
    // Same as Conv + BN
    fuse_conv_bn_weights(linear_weight, linear_bias, bn_weight, bn_bias, bn_mean, bn_var, eps)
}
```

### 5.3 Quantized Weight Merging

```rust
/// Quantized Conv + Quantized BN fusion (maintain quantization accuracy)
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
    // 1. Dequantize
    let fp_weight: Vec<f32> = conv_weight.iter()
        .map(|&x| (x as f32 - conv_zero) * conv_scale)
        .collect();

    // 2. Fuse
    let (mut fused_fp_weight, fused_bias) = fuse_conv_bn_weights(
        &fp_weight,
        None,
        bn_weight,
        bn_bias,
        bn_mean,
        bn_var,
        eps,
    );

    // 3. Requantize
    let max_val = fused_fp_weight.iter()
        .fold(0.0f32, |a, &b| a.max(b.abs()));
    let scale = max_val / 127.0;

    let q_weight: Vec<i8> = fused_fp_weight.iter()
        .map(|&x| (x / scale).round().clamp(-127.0, 127.0) as i8)
        .collect();

    (q_weight, scale, 0.0)
}
```


## 6. Supported Fusion Patterns

| Pattern | Implementation Status | Weight Merging |
|------|----------|----------|
| Conv2d + BatchNorm2d | ✅ Implemented | ✅ |
| Conv2d + ReLU | ✅ Implemented | ❌ (no weights) |
| Conv2d + BatchNorm2d + ReLU | ✅ Implemented | ✅ |
| Linear + ReLU | ⬜ To be implemented | ❌ |
| Linear + BatchNorm | ⬜ To be implemented | ✅ |
| Conv2d + BatchNorm + SiLU | ⬜ To be implemented | ✅ |
| Conv2d + BatchNorm + GELU | ⬜ To be implemented | ✅ |
| MatMul + Add (Bias) | ⬜ To be implemented | ❌ |
| Conv2d + Add (Residual) | ⬜ To be implemented | ❌ |


## 7. Steps to Add New Fusion Pattern

### Step 1: Identify Fusion Pattern

```rust
// Example: Conv2d + SiLU
// Identify Conv2d → SiLU pattern in graph
```

### Step 2: Check Fusion Conditions

```rust
// Check conditions
fn can_fuse_conv_silu(graph: &DagGraph, conv_id: u64, silu_id: u64) -> bool {
    let conv = graph.get_op(conv_id).unwrap();
    let silu = graph.get_op(silu_id).unwrap();

    // 1. Conv output is only used by SiLU
    let users = graph.get_users(conv.outputs[0]);
    if users.len() != 1 || users[0] != silu_id {
        return false;
    }

    // 2. No other dependencies
    // 3. Data types match
    // 4. Devices match

    true
}
```

### Step 3: Implement Fusion Logic

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

    // Mark for removal
    // ...

    Ok(fused_op)
}
```

### Step 4: Register Fusion Rule

```rust
// src/transform/fusion.rs

fn try_fuse(graph: &mut DagGraph, op_id: u64, op: &Op) -> Option<Op> {
    match op.op_type.as_str() {
        "conv2d" => {
            // Try Conv + BN
            if let Some(fused) = Self::try_fuse_conv_bn(graph, op_id, op) {
                return Some(fused);
            }
            // Try Conv + SiLU (new)
            if let Some(fused) = Self::try_fuse_conv_silu(graph, op_id, op) {
                return Some(fused);
            }
            // Try Conv + ReLU
            if let Some(fused) = Self::try_fuse_conv_relu(graph, op_id, op) {
                return Some(fused);
            }
        }
        "linear" => {
            // Try Linear + ReLU
            if let Some(fused) = Self::try_fuse_linear_relu(graph, op_id, op) {
                return Some(fused);
            }
        }
        _ => {}
    }
    None
}
```


## 8. Testing Fusion

```rust
// tests/test_fusion.rs

#[test]
fn test_conv_bn_fusion() {
    let mut graph = create_test_graph();

    // 1. Add Conv2d
    let conv_id = graph.add_op("conv2d", vec![input], vec![conv_out], attrs);

    // 2. Add BatchNorm2d
    let bn_id = graph.add_op("batchnorm2d", vec![conv_out], vec![out], bn_attrs);

    // 3. Execute fusion
    let changed = FusionPass::apply(&mut graph);

    // 4. Verify
    assert!(changed);
    assert!(graph.ops.contains_key("fused_conv_bn"));
    assert!(!graph.ops.contains_key(conv_id));
    assert!(!graph.ops.contains_key(bn_id));

    // 5. Verify weight merging correctness
    let fused_weight = graph.constants.get(&fused_weight_id).unwrap();
    // Verify fused_weight == conv_weight * bn_scale
}
```


## 9. Performance Validation

```rust
// benches/bench_fusion.rs

use criterion::*;

fn bench_conv_bn(c: &mut Criterion) {
    c.bench_function("conv_bn_unfused", |b| {
        b.iter(|| {
            // Execute unfused Conv + BN
        })
    });

    c.bench_function("conv_bn_fused", |b| {
        b.iter(|| {
            // Execute fused Conv_BN
        })
    });
}
```


## 10. FAQ

### Q: Does fusion reduce accuracy?

A: Theoretically no. Mathematically equivalent, but different floating-point operation order may introduce minor errors (typically < 1e-5).

### Q: Can quantized models be fused?

A: Yes, but need dequantize → fuse → requantize, which may introduce additional errors. QAT (Quantization-Aware Training) is recommended.

### Q: How does backward propagation work for fused operators?

A: Need to implement backward for fused op, otherwise training is not possible.

### Q: How to determine which operators can be fused?

A: Refer to common fusion patterns, or find hotspots through profiling.

### Q: How does memory usage change after fusion?

A: Usually decreases (no need to store intermediate results), but may increase (need to store more weights).


## 11. Development Checklist

- [ ] Identify fusion pattern (two or more consecutive operators)
- [ ] Check fusion conditions (output used by only one operator)
- [ ] Verify data type compatibility
- [ ] Weight merging logic (if applicable)
- [ ] Attribute merging (if applicable)
- [ ] Create fused operator
- [ ] Mark old operators for removal
- [ ] Register to FusionPass
- [ ] Unit test verification
- [ ] Performance test verification
- [ ] Accuracy verification


## 12. Fusion Implementation Template

```rust
// src/transform/fusion_rules/op1_op2.rs

/// A + B fusion rule
pub struct Op1Op2Fusion;

impl FusionRule for Op1Op2Fusion {
    fn name(&self) -> &'static str {
        "op1_op2"
    }

    fn matches(&self, graph: &DagGraph, op: &Op) -> Option<FusionMatch> {
        // 1. Match Op1
        if op.op_type != "op1" {
            return None;
        }

        // 2. Check if successor is Op2
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

        // 3. Return match
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
        // 1. Merge weights if any
        // 2. Create fused operator
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

## 13. Related Documentation

- [dev_ops.md](dev_ops.md) - Operator development guide
- `src/transform/fusion.rs` - Fusion implementation
- `tests/test_fusion.rs` - Fusion tests