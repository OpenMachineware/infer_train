// src/executor/memory_reuse.rs

use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap};

use crate::tensor::Tensor;

// ============================================================
// 内存块
// ============================================================

#[derive(Debug, Clone, Copy)]
pub struct MemoryBlock {
    pub index: usize,
    pub offset: usize,
    pub size: usize,
    pub is_free: bool,
}

// ============================================================
// 空闲块 (用于 BinaryHeap)
// ============================================================

#[derive(Debug, Clone, Copy)]
struct FreeBlock {
    pub offset: usize,
    pub size: usize,
}

impl Eq for FreeBlock {}

impl PartialEq for FreeBlock {
    fn eq(&self, other: &Self) -> bool {
        self.size == other.size
    }
}

impl PartialOrd for FreeBlock {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for FreeBlock {
    fn cmp(&self, other: &Self) -> Ordering {
        // 按 size 降序 (先分配最大的，减少碎片)
        other.size.cmp(&self.size)
    }
}

// ============================================================
// 内存配置
// ============================================================

#[derive(Debug, Clone, Copy)]
pub struct MemoryConfig {
    pub block_size: usize,     // 内存块大小 (2^n)
    pub total_size: usize,     // 总内存大小
    pub enable_defrag: bool,   // 是否启用碎片整理
    pub defrag_threshold: f32, // 碎片率阈值
}

impl Default for MemoryConfig {
    fn default() -> Self {
        MemoryConfig {
            block_size: 4096,
            total_size: 64 * 1024 * 1024, // 64MB
            enable_defrag: false,
            defrag_threshold: 0.3,
        }
    }
}

impl MemoryConfig {
    /// 推理模式配置
    pub fn inference() -> Self {
        MemoryConfig {
            block_size: 4096,
            total_size: 32 * 1024 * 1024, // 32MB
            enable_defrag: false,
            defrag_threshold: 0.5,
        }
    }

    /// 训练模式配置
    pub fn training() -> Self {
        MemoryConfig {
            block_size: 8192,
            total_size: 256 * 1024 * 1024, // 256MB
            enable_defrag: true,
            defrag_threshold: 0.2,
        }
    }
}

// ============================================================
// 字节池
// ============================================================

pub struct BytePool {
    // 整个内存池 (连续内存)
    arena: Vec<u8>,
    // 每个块的大小 (2^n)
    block_size: usize,
    // 空闲块列表 (按大小排序)
    free_blocks: BinaryHeap<FreeBlock>,
    // 分配表: tensor_id -> (block_offset, size)
    allocations: HashMap<u64, (usize, usize)>,
    // 总分配次数
    allocations_count: usize,
    // 总复用次数
    reuse_count: usize,
    // 当前总分配大小
    allocated_size: usize,
}

impl BytePool {
    pub fn new(block_size: usize, total_size: usize) -> Self {
        let arena = vec![0u8; total_size];
        let mut free_blocks = BinaryHeap::new();
        free_blocks.push(FreeBlock { offset: 0, size: total_size });

        BytePool {
            arena,
            block_size,
            free_blocks,
            allocations: HashMap::new(),
            allocations_count: 0,
            reuse_count: 0,
            allocated_size: 0,
        }
    }

    pub fn new_with_blocks(block_size: usize, num_blocks: usize) -> Self {
        Self::new(block_size, block_size * num_blocks)
    }

    // ============================================================
    // 分配
    // ============================================================

    pub fn allocate(&mut self, size: usize) -> Option<(usize, usize)> {
        // 对齐到 block_size
        let aligned_size =
            ((size + self.block_size - 1) / self.block_size) * self.block_size;

        // 找最佳空闲块 (First-Fit + Best-Fit 混合)
        let mut best_block: Option<FreeBlock> = None;
        let mut best_idx = 0;

        let blocks: Vec<FreeBlock> = self.free_blocks.drain().collect();
        for (i, block) in blocks.iter().enumerate() {
            if block.size >= aligned_size {
                if let Some(ref best) = best_block {
                    if block.size < best.size {
                        best_block = Some(*block);
                        best_idx = i;
                    }
                } else {
                    best_block = Some(*block);
                    best_idx = i;
                }
            }
        }

        // 放回非选中的块
        for (i, block) in blocks.into_iter().enumerate() {
            if i != best_idx {
                self.free_blocks.push(block);
            }
        }

        let block = match best_block {
            Some(b) => b,
            None => return None,
        };

        // 分割块
        if block.size > aligned_size {
            let remaining = FreeBlock {
                offset: block.offset + aligned_size,
                size: block.size - aligned_size,
            };
            self.free_blocks.push(remaining);
        }

        let id = self.allocations.len() as u64;
        self.allocations.insert(id, (block.offset, aligned_size));
        self.allocated_size += aligned_size;
        self.allocations_count += 1;

        Some((block.offset, aligned_size))
    }

    // ============================================================
    // 释放
    // ============================================================

    pub fn free(&mut self, id: u64) {
        if let Some((offset, size)) = self.allocations.remove(&id) {
            self.allocated_size -= size;

            // 合并相邻的空闲块
            let new_free = FreeBlock { offset, size };

            // 收集所有空闲块
            let mut blocks: Vec<FreeBlock> = self.free_blocks.drain().collect();
            blocks.push(new_free);

            // 按偏移排序
            blocks.sort_by_key(|b| b.offset);

            // 合并相邻块
            let mut merged: Vec<FreeBlock> = Vec::new();
            let mut current = blocks[0];
            for block in blocks.into_iter().skip(1) {
                if current.offset + current.size == block.offset {
                    // 相邻，合并
                    current.size += block.size;
                } else {
                    merged.push(current);
                    current = block;
                }
            }
            merged.push(current);

            // 放回堆
            for block in merged {
                self.free_blocks.push(block);
            }

            self.reuse_count += 1;
        }
    }

    // ============================================================
    // 获取数据指针
    // ============================================================

    pub fn get_data(&self, id: u64) -> Option<&[u8]> {
        self.allocations
            .get(&id)
            .map(|&(offset, size)| &self.arena[offset..offset + size])
    }

    pub fn get_data_mut(&mut self, id: u64) -> Option<&mut [u8]> {
        if let Some(&(offset, size)) = self.allocations.get(&id) {
            Some(&mut self.arena[offset..offset + size])
        } else {
            None
        }
    }

    // ============================================================
    // 统计
    // ============================================================

    pub fn stats(&self) -> (usize, usize, usize, usize) {
        let total_size = self.arena.len();
        let free_size: usize = self.free_blocks.iter().map(|b| b.size).sum();
        (
            self.allocations_count,
            self.reuse_count,
            self.allocated_size,
            total_size - free_size, // 实际使用
        )
    }

    pub fn fragmentation(&self) -> f32 {
        let total_free: usize = self.free_blocks.iter().map(|b| b.size).sum();
        let num_blocks = self.free_blocks.len();
        if num_blocks == 0 || total_free == 0 {
            return 0.0;
        }
        let avg_free = total_free / num_blocks;
        let max_free =
            self.free_blocks.iter().map(|b| b.size).max().unwrap_or(0);
        if max_free == 0 {
            return 0.0;
        }
        1.0 - (avg_free as f32 / max_free as f32)
    }

    // ============================================================
    // 重置
    // ============================================================

    pub fn reset(&mut self) {
        self.allocations.clear();
        self.allocated_size = 0;
        self.allocations_count = 0;
        self.reuse_count = 0;
        self.free_blocks.clear();
        self.free_blocks.push(FreeBlock { offset: 0, size: self.arena.len() });
    }

    pub fn clear(&mut self) {
        self.reset();
        self.arena.fill(0);
    }

    // ============================================================
    // 压缩 (Defragmentation)
    // ============================================================

    pub fn defragment(&mut self) {
        // 收集所有已分配块
        let mut allocs: Vec<(usize, usize, u64)> = self
            .allocations
            .iter()
            .map(|(&id, &(offset, size))| (offset, size, id))
            .collect();
        allocs.sort_by_key(|(offset, _, _)| *offset);

        // 重新排列到连续区域
        let mut new_offset = 0;
        let mut new_allocs = HashMap::new();

        for (old_offset, size, id) in allocs {
            if new_offset != old_offset {
                // 移动数据
                let src = old_offset;
                let dst = new_offset;
                for i in 0..size {
                    self.arena[dst + i] = self.arena[src + i];
                }
            }
            new_allocs.insert(id, (new_offset, size));
            new_offset += size;
        }

        // 更新分配表
        self.allocations = new_allocs;

        // 重置空闲块
        self.free_blocks.clear();
        if new_offset < self.arena.len() {
            self.free_blocks.push(FreeBlock {
                offset: new_offset,
                size: self.arena.len() - new_offset,
            });
        }
    }
}

// ============================================================
// Tensor 专用的内存池适配器
// ============================================================

pub struct MemoryPool {
    pool: BytePool,
    // tensor_id -> pool_id
    tensor_map: HashMap<u64, u64>,
    next_tensor_id: u64,
}

impl MemoryPool {
    pub fn new(block_size: usize, total_size: usize) -> Self {
        MemoryPool {
            pool: BytePool::new(block_size, total_size),
            tensor_map: HashMap::new(),
            next_tensor_id: 0,
        }
    }

    pub fn with_config(config: MemoryConfig) -> Self {
        MemoryPool::new(config.block_size, config.total_size)
    }

    pub fn new_with_blocks(block_size: usize, num_blocks: usize) -> Self {
        Self::new(block_size, block_size * num_blocks)
    }

    pub fn allocate(&mut self, data: &[u8]) -> Option<u64> {
        let id = self.next_tensor_id;
        self.next_tensor_id += 1;

        let (_offset, _size) = self.pool.allocate(data.len())?;
        let arena = self.pool.get_data_mut(id)?;
        arena[..data.len()].copy_from_slice(data);

        self.tensor_map.insert(id, id);
        Some(id)
    }

    pub fn get(&self, id: u64) -> Option<&[u8]> {
        self.pool.get_data(id)
    }

    pub fn get_mut(&mut self, id: u64) -> Option<&mut [u8]> {
        self.pool.get_data_mut(id)
    }

    pub fn free(&mut self, id: u64) {
        self.pool.free(id);
        self.tensor_map.remove(&id);
    }

    pub fn stats(&self) -> (usize, usize, usize, usize) {
        self.pool.stats()
    }

    pub fn fragmentation(&self) -> f32 {
        self.pool.fragmentation()
    }

    pub fn reset(&mut self) {
        self.pool.reset();
        self.tensor_map.clear();
        self.next_tensor_id = 0;
    }

    pub fn defragment(&mut self) {
        self.pool.defragment();
    }

    // ============================================================
    // 兼容旧接口（用于 Executor）
    // ============================================================

    pub fn allocate_or_use(&mut self, value_id: u64, tensor: Tensor<f32>) -> Tensor<f32> {
        let data_bytes = tensor.data();
        let bytes = bytemuck::cast_slice(data_bytes);

        if let Some(id) = self.allocate(bytes) {
            self.tensor_map.insert(value_id, id);
            let pool_data = self.get(id).unwrap();
            let float_data: &[f32] = bytemuck::cast_slice(pool_data);
            Tensor::new(float_data.to_vec(), tensor.shape())
        } else {
            tensor
        }
    }

    pub fn mark_reusable(&mut self, value_id: u64) {
        if let Some(&id) = self.tensor_map.get(&value_id) {
            self.pool.free(id);
            self.tensor_map.remove(&value_id);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_byte_pool_allocate_free() {
        let mut pool = BytePool::new_with_blocks(64, 10);

        let (offset, size) = pool.allocate(100).unwrap();
        assert_eq!(size, 128);
        assert_eq!(offset, 0);

        let (offset2, size2) = pool.allocate(50).unwrap();
        assert_eq!(offset2, 128);
        assert_eq!(size2, 64);

        // 释放第一个块：用 id=0（第一次分配返回的 id）
        pool.free(0);

        // 分配第三个块，应该复用第一个块
        let (offset3, _) = pool.allocate(60).unwrap();
        assert_eq!(offset3, 0);
    }

    #[test]
    fn test_byte_pool_defragment() {
        let mut pool = BytePool::new_with_blocks(64, 10);

        let (o1, _) = pool.allocate(100).unwrap(); // id=0, offset=0
        let (o2, _) = pool.allocate(50).unwrap(); // id=1, offset=128
        let (o3, _) = pool.allocate(80).unwrap(); // id=2, offset=192

        // 用 id 释放
        pool.free(0); // 释放第一个块
        pool.free(2); // 释放第三个块

        // 碎片化
        assert!(pool.fragmentation() > 0.0);

        pool.defragment();

        // 压缩后碎片减少
        assert!(pool.fragmentation() < 0.1);
    }
}
