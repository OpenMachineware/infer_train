// src/transform/verify.rs

use crate::ir::dag::DagGraph;

pub struct VerifyPass;

impl VerifyPass {
    pub fn verify(graph: &DagGraph) -> Result<(), String> {
        // 1. 检查所有 Op 的 inputs 都存在
        for (op_id, op) in &graph.ops {
            for &in_id in &op.inputs {
                if !graph.values.contains_key(&in_id) {
                    return Err(format!(
                        "Op {} input value {} not found",
                        op_id, in_id
                    ));
                }
            }
            for &out_id in &op.outputs {
                if !graph.values.contains_key(&out_id) {
                    return Err(format!(
                        "Op {} output value {} not found",
                        op_id, out_id
                    ));
                }
            }
        }

        // 2. 检查所有 Value 都有 producer 或者是 input/constant
        for (&id, value) in &graph.values {
            if !graph.inputs.contains(&id) &&
                !graph.constants.contains_key(&id) &&
                !graph.outputs.contains(&id) {
                if let Some(producer) = value.producer {
                    if !graph.ops.contains_key(&producer) {
                        return Err(format!(
                            "Value {} has missing producer {}",
                            id, producer
                        ));
                    }
                } else {
                    return Err(format!(
                        "Value {} has no producer and is not input/constant/output",
                        id
                    ));
                }
            }
        }

        // 3. 检查没有循环依赖
        if let Err(e) = graph.topological_sort() {
            return Err(format!("Graph has cycles: {}", e));
        }

        // 4. 检查所有 shape 都是有效的
        for (&id, value) in &graph.values {
            for &dim in &value.ty.shape {
                if dim < -1 {
                    return Err(format!(
                        "Value {} has invalid shape dimension: {}",
                        id, dim
                    ));
                }
            }
        }

        // 5. 检查所有常量数据大小匹配 shape
        for (&id, data) in &graph.constants {
            if let Some(value) = graph.values.get(&id) {
                let expected_size = Self::compute_tensor_size(&value.ty);
                if data.len() != expected_size {
                    return Err(format!(
                        "Constant {} size mismatch: expected {} bytes, got {}",
                        id, expected_size, data.len()
                    ));
                }
            }
        }

        Ok(())
    }

    fn compute_tensor_size(ty: &crate::ir::dag::TensorType) -> usize {
        let num_elements: usize = ty.shape.iter()
            .filter(|&&d| d != -1)
            .map(|&d| d as usize)
            .product();

        let elem_size = match ty.dtype {
            crate::ir::dag::DataType::F32 => 4,
            crate::ir::dag::DataType::F64 => 8,
            crate::ir::dag::DataType::F16 => 2,
            crate::ir::dag::DataType::BF16 => 2,
            crate::ir::dag::DataType::I8 => 1,
            crate::ir::dag::DataType::I32 => 4,
            crate::ir::dag::DataType::I64 => 8,
            crate::ir::dag::DataType::Bool => 1,
        };

        num_elements * elem_size
    }
}
