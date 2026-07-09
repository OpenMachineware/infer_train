// src/executor/scheduler.rs

use std::collections::{HashMap, VecDeque, HashSet};
use std::cmp::Ordering;
use crate::ir::dag::DagGraph;

// ============================================================
// 调度器配置
// ============================================================

#[derive(Debug, Clone)]
pub struct SchedulerConfig {
    pub enable_parallel: bool,
    pub enable_prefetch: bool,
    pub max_concurrent_ops: usize,
    pub memory_limit: usize,
}

impl Default for SchedulerConfig {
    fn default() -> Self {
        SchedulerConfig {
            enable_parallel: true,
            enable_prefetch: true,
            max_concurrent_ops: 16,
            memory_limit: 1024 * 1024 * 1024, // 1GB
        }
    }
}

// ============================================================
// 调度器
// ============================================================

pub struct Scheduler {
    config: SchedulerConfig,
    execution_order: Vec<u64>,
    node_weights: HashMap<u64, NodeWeight>,
}

#[derive(Debug, Clone)]
pub struct NodeWeight {
    pub op_id: u64,
    pub compute_cost: f32,      // 计算开销 (FLOPs)
    pub memory_cost: usize,     // 内存开销 (bytes)
    pub priority: f32,          // 优先级 (越高越先执行)
}

impl Scheduler {
    pub fn new(config: SchedulerConfig) -> Self {
        Scheduler {
            config,
            execution_order: Vec::new(),
            node_weights: HashMap::new(),
        }
    }

    // ============================================================
    // 生成执行顺序
    // ============================================================

    pub fn schedule(&mut self, graph: &DagGraph) -> Vec<u64> {
        self.node_weights.clear();
        self.execution_order.clear();

        // 计算每个节点的权重
        self.compute_node_weights(graph);

        // 拓扑排序 + 按优先级调度
        let topo = self.topological_sort_with_priority(graph);

        // 启发式重排 (考虑内存和计算)
        self.heuristic_reorder(graph, &topo);

        self.execution_order.clone()
    }

    // ============================================================
    // 计算节点权重
    // ============================================================

    fn compute_node_weights(&mut self, graph: &DagGraph) {
        for (&op_id, op) in &graph.ops {
            let (compute_cost, memory_cost) = self.estimate_cost(op);

            let priority = self.calculate_priority(&compute_cost, &memory_cost);

            self.node_weights.insert(op_id, NodeWeight {
                op_id,
                compute_cost,
                memory_cost,
                priority,
            });
        }
    }

    fn estimate_cost(&self, op: &crate::ir::dag::Op) -> (f32, usize) {
        let compute_cost = match op.op_type.as_str() {
            "matmul" => 10.0,
            "conv2d" => 20.0,
            "add" | "sub" | "mul" | "div" => 1.0,
            "relu" | "sigmoid" | "tanh" => 0.5,
            "reshape" | "transpose" => 0.1,
            _ => 1.0,
        };

        let memory_cost = match op.op_type.as_str() {
            "matmul" => 1024 * 1024,
            "conv2d" => 2048 * 1024,
            _ => 1024,
        };

        (compute_cost, memory_cost)
    }

    fn calculate_priority(&self, compute: &f32, memory: &usize) -> f32 {
        // 计算密集型 + 内存密集型 = 高优先级
        let compute_factor = compute / 10.0;
        let memory_factor = (*memory as f32) / (1024.0 * 1024.0);
        compute_factor + memory_factor * 0.5
    }

    // ============================================================
    // 拓扑排序 + 优先级
    // ============================================================

    fn topological_sort_with_priority(&self, graph: &DagGraph) -> Vec<u64> {
        let mut in_degree: HashMap<u64, usize> = HashMap::new();
        let mut adj: HashMap<u64, Vec<u64>> = HashMap::new();

        for (&id, _) in &graph.ops {
            in_degree.entry(id).or_insert(0);
            adj.entry(id).or_insert(Vec::new());
        }

        for (op_id, op) in &graph.ops {
            for &out_id in &op.outputs {
                for (next_id, next_op) in &graph.ops {
                    if next_op.inputs.contains(&out_id) {
                        adj.entry(*op_id).or_insert(Vec::new()).push(*next_id);
                        *in_degree.entry(*next_id).or_insert(0) += 1;
                    }
                }
            }
        }

        // 按优先级排序的队列
        let mut queue: Vec<u64> = in_degree.iter()
            .filter_map(|(&id, &deg)| {
                if deg == 0 {
                    Some(id)
                } else {
                    None
                }
            })
            .collect();

        // 按优先级降序排序 (高优先级先执行)
        queue.sort_by(|a, b| {
            let pa = self.node_weights.get(a).map(|w| w.priority).unwrap_or(0.0);
            let pb = self.node_weights.get(b).map(|w| w.priority).unwrap_or(0.0);
            pb.partial_cmp(&pa).unwrap_or(Ordering::Equal)
        });

        let mut result = Vec::new();
        let mut queue_deque: VecDeque<u64> = queue.into_iter().collect();

        while let Some(id) = queue_deque.pop_front() {
            result.push(id);

            if let Some(neighbors) = adj.get(&id) {
                for &next in neighbors {
                    let deg = in_degree.get_mut(&next).unwrap();
                    *deg -= 1;
                    if *deg == 0 {
                        // 插入时保持优先级
                        let priority = self.node_weights.get(&next)
                            .map(|w| w.priority)
                            .unwrap_or(0.0);

                        let pos = queue_deque.iter()
                            .position(|&x| {
                                let p = self.node_weights.get(&x)
                                    .map(|w| w.priority)
                                    .unwrap_or(0.0);
                                p < priority
                            })
                            .unwrap_or(queue_deque.len());

                        queue_deque.insert(pos, next);
                    }
                }
            }
        }

        result
    }

    // ============================================================
    // 启发式重排
    // ============================================================

    fn heuristic_reorder(&self, graph: &DagGraph, order: &[u64]) -> Vec<u64> {
        let mut result = order.to_vec();

        if !self.config.enable_parallel {
            return result;
        }

        // 按内存访问模式重排: 优先安排能复用内存的算子
        let mut memory_map: HashMap<u64, Vec<u64>> = HashMap::new();

        for &op_id in order {
            if let Some(op) = graph.get_op(op_id) {
                for &in_id in &op.inputs {
                    memory_map.entry(in_id)
                        .or_insert(Vec::new())
                        .push(op_id);
                }
            }
        }

        // 对结果按内存复用分组
        let mut grouped = Vec::new();
        let mut visited = HashSet::new();

        for &op_id in order {
            if visited.contains(&op_id) {
                continue;
            }

            let mut group = Vec::new();
            let mut queue = VecDeque::new();
            queue.push_back(op_id);

            while let Some(id) = queue.pop_front() {
                if visited.contains(&id) {
                    continue;
                }
                visited.insert(id);
                group.push(id);

                // 找共享内存的邻居
                if let Some(op) = graph.get_op(id) {
                    for &in_id in &op.inputs {
                        if let Some(users) = memory_map.get(&in_id) {
                            for &user in users {
                                if !visited.contains(&user) {
                                    queue.push_back(user);
                                }
                            }
                        }
                    }
                }
            }

            if !group.is_empty() {
                // 按内存大小排序 (先分配大的，再分配小的，减少碎片)
                group.sort_by(|a, b| {
                    let mem_a = self.node_weights.get(a)
                        .map(|w| w.memory_cost)
                        .unwrap_or(0);
                    let mem_b = self.node_weights.get(b)
                        .map(|w| w.memory_cost)
                        .unwrap_or(0);
                    mem_b.cmp(&mem_a)
                });
                grouped.extend(group);
            }
        }

        if !grouped.is_empty() {
            result = grouped;
        }

        result
    }

    // ============================================================
    // 获取执行计划
    // ============================================================

    pub fn get_execution_order(&self) -> &[u64] {
        &self.execution_order
    }

    pub fn get_node_weight(&self, op_id: u64) -> Option<&NodeWeight> {
        self.node_weights.get(&op_id)
    }

    // ============================================================
    // 内存规划
    // ============================================================

    pub fn plan_memory(&self, graph: &DagGraph) -> HashMap<u64, (usize, usize)> {
        let mut plan = HashMap::new();
        let mut live_ranges = HashMap::new();

        // 计算每个值的生命周期
        for (pos, &op_id) in self.execution_order.iter().enumerate() {
            if let Some(op) = graph.get_op(op_id) {
                for &out_id in &op.outputs {
                    live_ranges.insert(out_id, (pos, pos));
                }
                for &in_id in &op.inputs {
                    live_ranges.entry(in_id)
                        .and_modify(|(_start, end)| {
                            *end = pos;
                        })
                        .or_insert((pos, pos));
                }
            }
        }

        // 分配内存地址
        let mut current_offset = 0;
        for (&id, (_start, _end)) in &live_ranges {
            if let Some(value) = graph.get_value(id) {
                let size: usize = value.ty.shape.iter()
                    .filter(|&&d| d != -1)
                    .map(|&d| d as usize)
                    .product();
                let size_bytes = size * 4; // f32
                plan.insert(id, (current_offset, size_bytes));
                current_offset += size_bytes;
            }
        }

        plan
    }
}
