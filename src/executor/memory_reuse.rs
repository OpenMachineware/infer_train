// src/executor/memory_reuse.rs

use std::collections::{HashMap, HashSet, VecDeque};
use crate::ffi::Tensor;

/// 内存池 - 实现 tensor 内存复用
#[derive(Clone)]
pub struct MemoryPool {
    /// 可复用的 tensor 池（按 size 分组）
    reusable: HashMap<usize, Vec<Tensor>>,
    /// 当前活跃的 tensor
    active: HashSet<usize>,  // 存储 tensor 的 id
    /// 总分配次数
    allocations: usize,
    /// 总复用次数
    reuse_count: usize,
}

impl MemoryPool {
    pub fn new() -> Self {
        MemoryPool {
            reusable: HashMap::new(),
            active: HashSet::new(),
            allocations: 0,
            reuse_count: 0,
        }
    }

    pub fn reset(&mut self) {
        self.reusable.clear();
        self.active.clear();
        self.allocations = 0;
        self.reuse_count = 0;
    }

    /// 分配或复用 tensor
    pub fn allocate_or_use(&mut self, tensor: Tensor) -> Tensor {
        let size = tensor.size_in_bytes();

        // 尝试从池中获取可复用的 tensor
        if let Some(pool) = self.reusable.get_mut(&size) {
            if let Some(reused) = pool.pop() {
                self.reuse_count += 1;
                return reused;
            }
        }

        self.allocations += 1;
        tensor
    }

    /// 标记 tensor 为可复用
    pub fn mark_reusable(&mut self, tensor: Tensor) {
        let size = tensor.size_in_bytes();
        self.reusable.entry(size).or_insert_with(Vec::new).push(tensor);
    }

    /// 获取统计信息
    pub fn stats(&self) -> (usize, usize) {
        (self.allocations, self.reuse_count)
    }

    /// 清理所有可复用的 tensor
    pub fn clear(&mut self) {
        self.reusable.clear();
    }
}

/// 生命周期分析器 - 计算每个 value 的活跃区间
pub struct LivenessAnalyzer {
    /// 每个 value 的最后使用位置
    last_use: HashMap<u64, usize>,
    /// 每个 value 的第一次使用位置
    first_use: HashMap<u64, usize>,
}

impl LivenessAnalyzer {
    pub fn new() -> Self {
        LivenessAnalyzer {
            last_use: HashMap::new(),
            first_use: HashMap::new(),
        }
    }

    /// 分析 liveness
    pub fn analyze(&mut self, order: &[u64], graph: &crate::ir::dag::DagGraph) {
        // 从后往前扫描，记录每个 value 的最后使用位置
        for (pos, &op_id) in order.iter().enumerate() {
            if let Some(op) = graph.get_op(op_id) {
                // 记录 outputs 的第一次使用（从后往前 = 最后使用）
                for &out_id in &op.outputs {
                    self.last_use.insert(out_id, pos);
                    self.first_use.entry(out_id).or_insert(pos);
                }
            }
        }

        // 再扫描一遍，更新 last_use
        for (pos, &op_id) in order.iter().enumerate() {
            if let Some(op) = graph.get_op(op_id) {
                for &in_id in &op.inputs {
                    self.last_use.insert(in_id, pos);
                }
            }
        }
    }

    /// 检查某个 value 在指定位置是否活跃
    pub fn is_live_at(&self, value_id: u64, pos: usize) -> bool {
        if let Some(&first) = self.first_use.get(&value_id) {
            if let Some(&last) = self.last_use.get(&value_id) {
                return pos >= first && pos <= last;
            }
        }
        false
    }

    /// 获取 value 的生命周期
    pub fn lifespan(&self, value_id: u64) -> Option<(usize, usize)> {
        match (self.first_use.get(&value_id), self.last_use.get(&value_id)) {
            (Some(&first), Some(&last)) => Some((first, last)),
            _ => None,
        }
    }

    /// 计算内存峰值
    pub fn peak_memory(&self, order: &[u64], graph: &crate::ir::dag::DagGraph) -> usize {
        let mut peak = 0;
        let mut current = 0;
        let mut sizes: HashMap<u64, usize> = HashMap::new();

        // 计算每个 value 的大小
        for (&id, value) in &graph.values {
            let size: usize = value.ty.shape.iter()
                .filter(|&&d| d != -1)
                .map(|&d| d as usize)
                .product();
            let dtype_size = match value.ty.dtype {
                crate::ir::dag::DataType::F32 => 4,
                crate::ir::dag::DataType::F64 => 8,
                crate::ir::dag::DataType::F16 => 2,
                crate::ir::dag::DataType::BF16 => 2,
                crate::ir::dag::DataType::I8 => 1,
                crate::ir::dag::DataType::I32 => 4,
                crate::ir::dag::DataType::I64 => 8,
                crate::ir::dag::DataType::Bool => 1,
            };
            sizes.insert(id, size * dtype_size);
        }

        for (pos, &op_id) in order.iter().enumerate() {
            if let Some(op) = graph.get_op(op_id) {
                // 添加 outputs
                for &out_id in &op.outputs {
                    if let Some(&size) = sizes.get(&out_id) {
                        current += size;
                    }
                }

                peak = peak.max(current);

                // 移除不再活跃的 inputs
                for &in_id in &op.inputs {
                    if let Some(&last) = self.last_use.get(&in_id) {
                        if last == pos {
                            if let Some(&size) = sizes.get(&in_id) {
                                current = current.saturating_sub(size);
                            }
                        }
                    }
                }
            }
        }

        peak
    }
}
