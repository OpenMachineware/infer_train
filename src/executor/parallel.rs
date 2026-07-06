// src/executor/parallel.rs

use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use rayon::prelude::*;
use crate::ffi::Tensor;
use crate::ir::dag::{DagGraph, Op};
use super::memory_reuse::MemoryPool;

/// 并行执行器
pub struct ParallelExecutor<'a> {
    graph: &'a DagGraph,
    values: &'a mut HashMap<u64, Tensor>,
    memory_pool: &'a mut MemoryPool,
    num_threads: usize,
}

impl<'a> ParallelExecutor<'a> {
    pub fn new(
        graph: &'a DagGraph,
        values: &'a mut HashMap<u64, Tensor>,
        memory_pool: &'a mut MemoryPool,
        num_threads: usize,
    ) -> Self {
        ParallelExecutor {
            graph,
            values,
            memory_pool,
            num_threads,
        }
    }

    pub fn execute(&mut self, order: &[u64]) -> Result<(), String> {
        // 构建依赖图
        let deps = self.build_dependency_graph(order);

        // 将 ops 分组为可并行执行的层级
        let levels = self.compute_levels(order, &deps);

        // 按层级执行
        for level in levels {
            self.execute_level(&level)?;
        }

        Ok(())
    }

    /// 构建依赖图：每个 op 依赖哪些输入
    fn build_dependency_graph(&self, order: &[u64]) -> HashMap<u64, Vec<u64>> {
        let mut deps = HashMap::new();

        // 记录每个 value 的最新 producer
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

                // 更新 producer
                for &out_id in &op.outputs {
                    producer_map.insert(out_id, op_id);
                }
            }
        }

        deps
    }

    /// 计算并行层级
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
                // 如果没有可执行的 op，说明有循环依赖
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

    /// 执行一个层级（并行）
    fn execute_level(&mut self, level: &[u64]) -> Result<(), String> {
        if level.len() <= 1 {
            // 只有一个 op，直接执行
            for &op_id in level {
                if let Some(op) = self.graph.get_op(op_id) {
                    self.execute_single_op(op)?;
                }
            }
            return Ok(());
        }

        // 多个 op 并行执行
        let graph = self.graph;
        let values = self.values;
        let memory_pool = self.memory_pool;

        // 使用 Arc 包装可共享的数据
        let results: Vec<Result<(), String>> = level.par_iter()
            .map(|&op_id| {
                let op = graph.get_op(op_id)
                    .ok_or_else(|| format!("Op {} not found", op_id))?;

                // 每个线程独立执行
                Self::execute_op_parallel(graph, values, memory_pool, op)
            })
            .collect();

        // 检查错误
        for result in results {
            result?;
        }

        Ok(())
    }

    fn execute_single_op(&mut self, op: &Op) -> Result<(), String> {
        Self::execute_op_parallel(self.graph, self.values, self.memory_pool, op)
    }

    fn execute_op_parallel(
        graph: &DagGraph,
        values: &mut HashMap<u64, Tensor>,
        memory_pool: &mut MemoryPool,
        op: &Op,
    ) -> Result<(), String> {
        // 收集输入
        let mut input_tensors = Vec::new();
        for &in_id in &op.inputs {
            if let Some(t) = values.get(&in_id) {
                input_tensors.push(t.clone());
            } else {
                return Err(format!("Input value {} not found for op {}", in_id, op.id));
            }
        }

        // 执行算子（这里需要调用 dispatch_op）
        // 注意：dispatch_op 需要是独立函数或 static 方法
        let outputs = super::executor::dispatch_op(&op.op_type, &input_tensors, &op.attrs)?;

        // 存储输出
        for (i, &out_id) in op.outputs.iter().enumerate() {
            if i < outputs.len() {
                let tensor = memory_pool.allocate_or_use(outputs[i].clone());
                values.insert(out_id, tensor);
            }
        }

        // 标记可复用的 tensor
        for &in_id in &op.inputs {
            let users = graph.get_users(in_id);
            if users.len() == 1 && users[0] == op.id {
                if let Some(tensor) = values.get(&in_id) {
                    memory_pool.mark_reusable(tensor.clone());
                }
            }
        }

        Ok(())
    }
}

/// 导出 dispatch_op 供并行执行使用
pub fn dispatch_op(
    op_type: &str,
    inputs: &[Tensor],
    attrs: &HashMap<String, crate::ir::dag::AttrValue>,
) -> Result<Vec<Tensor>, String> {
    // 这里复用 executor 的 dispatch 逻辑
    // 或者直接内联
    if op_type.starts_with("quantized_") {
        return super::quantized::dispatch_quantized(op_type, inputs, attrs);
    }

    if let Ok(result) = super::math::dispatch_math(op_type, inputs, attrs) {
        return Ok(result);
    }
    if let Ok(result) = super::nn::dispatch_nn(op_type, inputs, attrs) {
        return Ok(result);
    }
    if let Ok(result) = super::activation::dispatch_activation(op_type, inputs, attrs) {
        return Ok(result);
    }
    if let Ok(result) = super::tensor::dispatch_tensor(op_type, inputs, attrs) {
        return Ok(result);
    }
    if let Ok(result) = super::index::dispatch_index(op_type, inputs, attrs) {
        return Ok(result);
    }
    if let Ok(result) = super::control::dispatch_control(op_type, inputs, attrs) {
        return Ok(result);
    }

    Err(format!("Unknown operator: {}", op_type))
}
