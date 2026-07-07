// src/executor/parallel.rs

use std::collections::HashMap;
use crate::tensor::Tensor;
use crate::ir::dag::{DagGraph, Op};
use super::memory_reuse::MemoryPool;

pub struct ParallelExecutor<'a> {
    graph: &'a DagGraph,
    values: &'a mut HashMap<u64, Tensor<f32>>,
    memory_pool: &'a mut MemoryPool,
    _num_threads: usize,
}

impl<'a> ParallelExecutor<'a> {
    pub fn new(
        graph: &'a DagGraph,
        values: &'a mut HashMap<u64, Tensor<f32>>,
        memory_pool: &'a mut MemoryPool,
        num_threads: usize,
    ) -> Self {
        ParallelExecutor {
            graph,
            values,
            memory_pool,
            _num_threads: num_threads,
        }
    }

    pub fn execute(&mut self, order: &[u64]) -> Result<(), String> {
        let deps = self.build_dependency_graph(order);
        let levels = self.compute_levels(order, &deps);

        for level in levels {
            self.execute_level(&level)?;
        }

        Ok(())
    }

    fn build_dependency_graph(&self, order: &[u64]) -> HashMap<u64, Vec<u64>> {
        let mut deps = HashMap::new();
        let mut producer_map: HashMap<u64, u64> = HashMap::new();

        for &op_id in order {
            if let Some(op) = self.graph.get_op(op_id) {
                let mut dependencies = Vec::new();
                for &in_id in &op.inputs {
                    if let Some(&producer) = producer_map.get(&in_id) {
                        dependencies.push(producer);
                    }
                }
                deps.insert(op_id, dependencies);
                for &out_id in &op.outputs {
                    producer_map.insert(out_id, op_id);
                }
            }
        }

        deps
    }

    fn compute_levels(&self, order: &[u64], deps: &HashMap<u64, Vec<u64>>) -> Vec<Vec<u64>> {
        let mut levels = Vec::new();
        let mut remaining: Vec<u64> = order.to_vec();
        let mut completed = std::collections::HashSet::new();

        while !remaining.is_empty() {
            let mut current_level = Vec::new();
            let mut next_remaining = Vec::new();

            for &op_id in &remaining {
                let all_deps_done = deps.get(&op_id)
                    .map(|deps| deps.iter().all(|d| completed.contains(d)))
                    .unwrap_or(true);

                if all_deps_done {
                    current_level.push(op_id);
                } else {
                    next_remaining.push(op_id);
                }
            }

            if current_level.is_empty() {
                break;
            }

            levels.push(current_level);
            for &op_id in levels.last().unwrap() {
                completed.insert(op_id);
            }
            remaining = next_remaining;
        }

        levels
    }

    fn execute_level(&mut self, level: &[u64]) -> Result<(), String> {
        for &op_id in level {
            if let Some(op) = self.graph.get_op(op_id) {
                Self::execute_op_direct(self.graph, self.values, self.memory_pool, op)?;
            }
        }
        Ok(())
    }

    fn execute_op_direct(
        graph: &DagGraph,
        values: &mut HashMap<u64, Tensor<f32>>,
        memory_pool: &mut MemoryPool,
        op: &Op,
    ) -> Result<(), String> {
        let mut input_tensors = Vec::new();
        for &in_id in &op.inputs {
            if let Some(t) = values.get(&in_id) {
                input_tensors.push(t.clone());
            } else {
                return Err(format!("Input value {} not found for op {}", in_id, op.id));
            }
        }

        let outputs = dispatch_op(&op.op_type, &input_tensors, &op.attrs)?;

        for (i, &out_id) in op.outputs.iter().enumerate() {
            if i < outputs.len() {
                let tensor = memory_pool.allocate_or_use(outputs[i].clone());
                values.insert(out_id, tensor);
            }
        }

        for &in_id in &op.inputs {
            let users = graph.get_users(in_id);
            if users.len() == 1 && users[0] == op.id {
                if let Some(tensor) = values.get(&in_id) {
                    memory_pool.mark_reusable(tensor);
                }
            }
        }

        Ok(())
    }
}

pub use super::executor::dispatch_op;
