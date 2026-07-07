use std::collections::HashMap;
use crate::ir::cfg::CfgGraph;
use crate::ir::dag::{DagGraph, Op, TensorType, DataType, AttrValue};

pub struct CfgToDagConverter;

impl CfgToDagConverter {
    pub fn convert(cfg: &CfgGraph) -> Result<DagGraph, String> {
        let mut dag = DagGraph::new(&cfg.name);

        let block_order = cfg.topological_sort()?;
        let mut value_map: HashMap<u64, u64> = HashMap::new();

        // 处理输入
        for &input_id in &cfg.inputs {
            if let Some((dtype, shape)) = cfg.value_types.get(&input_id) {
                let dag_id = dag.add_value(
                    &format!("input_{}", input_id),
                    TensorType {
                        dtype: *dtype,
                        shape: shape.clone(),
                    }
                );
                value_map.insert(input_id, dag_id);
            }
        }

        for &block_id in &block_order {
            let block = cfg.blocks.get(&block_id)
                .ok_or_else(|| format!("Block {} not found", block_id))?;

            Self::convert_block(&mut dag, block, &mut value_map, cfg)?;

            if let Some(branch_info) = &block.branch_info {
                Self::convert_branch(&mut dag, branch_info, &mut value_map, cfg)?;
            }
        }

        // 处理输出
        let mut output_ids = Vec::new();
        for &out_id in &cfg.outputs {
            if let Some(&dag_id) = value_map.get(&out_id) {
                output_ids.push(dag_id);
            } else {
                if let Some((dtype, shape)) = cfg.value_types.get(&out_id) {
                    let dag_id = dag.add_value(
                        &format!("output_{}", out_id),
                        TensorType {
                            dtype: *dtype,
                            shape: shape.clone(),
                        }
                    );
                    output_ids.push(dag_id);
                }
            }
        }
        dag.set_outputs(output_ids);

        Ok(dag)
    }

    fn convert_block(
        dag: &mut DagGraph,
        block: &crate::ir::cfg::CfgBlock,
        value_map: &mut HashMap<u64, u64>,
        cfg: &CfgGraph,
    ) -> Result<(), String> {
        for op in &block.ops {
            // 映射输入
            let mut inputs = Vec::new();
            for &in_id in &op.inputs {
                if let Some(&dag_id) = value_map.get(&in_id) {
                    inputs.push(dag_id);
                } else {
                    if let Some((dtype, shape)) = cfg.value_types.get(&in_id) {
                        let dag_id = dag.add_value(
                            &format!("v_{}", in_id),
                            TensorType {
                                dtype: *dtype,
                                shape: shape.clone(),
                            }
                        );
                        value_map.insert(in_id, dag_id);
                        inputs.push(dag_id);
                    } else {
                        return Err(format!("Value {} not found", in_id));
                    }
                }
            }

            // 创建输出
            let mut outputs = Vec::new();
            for &out_id in &op.outputs {
                let out_name = format!("{}_{}", op.op_type, out_id);
                let (dtype, shape) = if let Some(info) = cfg.value_types.get(&out_id) {
                    (info.0.clone(), info.1.clone())
                } else {
                    (DataType::F32, vec![])
                };

                let dag_id = dag.add_value(
                    &out_name,
                    TensorType {
                        dtype,  // DataType 是 Copy
                        shape: shape.clone(),
                    }
                );
                value_map.insert(out_id, dag_id);
                outputs.push(dag_id);
            }

            // 创建 Op
            let op_id = dag.insert_op(Op {
                id: 0,
                name: op.name.clone(),
                op_type: op.op_type.clone(),
                inputs,
                outputs: outputs.clone(),
                attrs: op.attrs.clone(),
            });

            // 更新 producer
            for &out_id in &op.outputs {
                if let Some(&dag_id) = value_map.get(&out_id) {
                    if let Some(value) = dag.values.get_mut(&dag_id) {
                        value.producer = Some(op_id);
                    }
                }
            }
        }

        Ok(())
    }

    fn convert_branch(
        dag: &mut DagGraph,
        branch_info: &crate::ir::cfg::BranchInfo,
        value_map: &mut HashMap<u64, u64>,
        cfg: &CfgGraph,
    ) -> Result<(), String> {
        let cond_dag_id = match value_map.get(&branch_info.condition_value) {
            Some(&id) => id,
            None => return Err(format!(
                "Condition value {} not found",
                branch_info.condition_value
            )),
        };

        let true_outputs = Self::find_block_outputs(cfg, branch_info.true_branch)?;
        let false_outputs = Self::find_block_outputs(cfg, branch_info.false_branch)?;

        if true_outputs.len() != false_outputs.len() {
            return Err(format!(
                "Branch output mismatch: true has {}, false has {}",
                true_outputs.len(), false_outputs.len()
            ));
        }

        for (idx, (&true_id, &false_id)) in true_outputs.iter().zip(false_outputs.iter()).enumerate() {
            let true_dag_id = *value_map.get(&true_id)
                .ok_or_else(|| format!("True value {} not found", true_id))?;
            let false_dag_id = *value_map.get(&false_id)
                .ok_or_else(|| format!("False value {} not found", false_id))?;

            let dtype = if let Some((dtype, _)) = cfg.value_types.get(&true_id) {
                *dtype
            } else {
                DataType::F32
            };

            let select_id = dag.add_value(
                &format!("select_{}_{}", branch_info.merge_block, idx),
                TensorType {
                    dtype,
                    shape: vec![],
                }
            );

            let mut attrs = HashMap::new();
            attrs.insert("condition".to_string(), AttrValue::String("branch".to_string()));

            dag.insert_op(Op {
                id: 0,
                name: format!("select_{}", branch_info.merge_block),
                op_type: "select".to_string(),
                inputs: vec![cond_dag_id, true_dag_id, false_dag_id],
                outputs: vec![select_id],
                attrs,
            });
        }

        Ok(())
    }

    fn find_block_outputs(cfg: &CfgGraph, block_id: u64) -> Result<Vec<u64>, String> {
        let block = cfg.blocks.get(&block_id)
            .ok_or_else(|| format!("Block {} not found", block_id))?;

        let mut outputs = Vec::new();

        if let Some(last_op) = block.ops.last() {
            outputs.extend_from_slice(&last_op.outputs);
        }

        if outputs.is_empty() {
            for &succ_id in &block.successors {
                let succ_outputs = Self::find_block_outputs(cfg, succ_id)?;
                outputs.extend(succ_outputs);
            }
        }

        Ok(outputs)
    }
}

// ============================================================
// 测试
// ============================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::cfg::{CfgGraph, CfgOp};

    #[test]
    fn test_cfg_to_dag_simple() {
        let mut cfg = CfgGraph::new("test");
        let entry = cfg.add_block("entry");
        cfg.set_entry(entry);

        // 添加一个简单的算子
        let op = CfgOp {
            id: 0,
            op_type: "add".to_string(),
            inputs: vec![1, 2],
            outputs: vec![3],
            attrs: HashMap::new(),
            name: "add_0".to_string(),
        };
        cfg.add_op(entry, op).unwrap();

        // 添加 value types
        cfg.value_types.insert(1, (DataType::F32, vec![2, 3]));
        cfg.value_types.insert(2, (DataType::F32, vec![2, 3]));
        cfg.value_types.insert(3, (DataType::F32, vec![2, 3]));

        cfg.add_input(1, DataType::F32, vec![2, 3]);
        cfg.add_input(2, DataType::F32, vec![2, 3]);
        cfg.add_output(3);

        let dag = CfgToDagConverter::convert(&cfg).unwrap();

        assert_eq!(dag.ops.len(), 1);
        assert_eq!(dag.values.len(), 3);
        assert_eq!(dag.inputs.len(), 2);
        assert_eq!(dag.outputs.len(), 1);
    }
}
