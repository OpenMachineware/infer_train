// src/executor/memory_reuse.rs

use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap};

use crate::tensor::Tensor;

// ============================================================
// Memory Block
// ============================================================

#[derive(Debug, Clone, Copy)]
pub struct MemoryBlock {
    pub index: usize,
    pub offset: usize,
    pub size: usize,
    pub is_free: bool,
}

// ============================================================
// Free Block (for BinaryHeap)
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
        // Sort by size descending
        // (allocate largest first, reduce fragmentation)
        other.size.cmp(&self.size)
    }
}

// ============================================================
// Memory Configuration
// ============================================================

#[derive(Debug, Clone, Copy)]
pub struct MemoryConfig {
    pub block_size: usize,     // Memory block size (2^n)
    pub total_size: usize,     // Total memory size
    pub enable_defrag: bool,   // Enable defragmentation
    pub defrag_threshold: f32, // Fragmentation threshold
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
    /// Inference mode configuration
    pub fn inference() -> Self {
        MemoryConfig {
            block_size: 4096,
            total_size: 32 * 1024 * 1024, // 32MB
            enable_defrag: false,
            defrag_threshold: 0.5,
        }
    }

    /// Training mode configuration
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
// Byte Pool
// ============================================================

pub struct BytePool {
    // The entire memory pool (contiguous memory)
    arena: Vec<u8>,
    // Size of each block (2^n)
    block_size: usize,
    // Free block list (sorted by size)
    free_blocks: BinaryHeap<FreeBlock>,
    // Allocation table: tensor_id -> (block_offset, size)
    allocations: HashMap<u64, (usize, usize)>,
    // Total allocation count
    allocations_count: usize,
    // Total reuse count
    reuse_count: usize,
    // Current total allocated size
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
    // Allocate
    // ============================================================

    pub fn allocate(&mut self, size: usize) -> Option<(usize, usize)> {
        // Align to block_size
        let aligned_size =
            ((size + self.block_size - 1) / self.block_size) * self.block_size;

        // Find best free block (First-Fit + Best-Fit hybrid)
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

        // Put back non-selected blocks
        for (i, block) in blocks.into_iter().enumerate() {
            if i != best_idx {
                self.free_blocks.push(block);
            }
        }

        let block = match best_block {
            Some(b) => b,
            None => return None,
        };

        // Split block
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
    // Free
    // ============================================================

    pub fn free(&mut self, id: u64) {
        if let Some((offset, size)) = self.allocations.remove(&id) {
            self.allocated_size -= size;

            // Merge adjacent free blocks
            let new_free = FreeBlock { offset, size };

            // Collect all free blocks
            let mut blocks: Vec<FreeBlock> = self.free_blocks.drain().collect();
            blocks.push(new_free);

            // Sort by offset
            blocks.sort_by_key(|b| b.offset);

            // Merge adjacent blocks
            let mut merged: Vec<FreeBlock> = Vec::new();
            let mut current = blocks[0];
            for block in blocks.into_iter().skip(1) {
                if current.offset + current.size == block.offset {
                    // Adjacent, merge
                    current.size += block.size;
                } else {
                    merged.push(current);
                    current = block;
                }
            }
            merged.push(current);

            // Put back to heap
            for block in merged {
                self.free_blocks.push(block);
            }

            self.reuse_count += 1;
        }
    }

    // ============================================================
    // Get Data Pointer
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
    // Statistics
    // ============================================================

    pub fn stats(&self) -> (usize, usize, usize, usize) {
        let total_size = self.arena.len();
        let free_size: usize = self.free_blocks.iter().map(|b| b.size).sum();
        (
            self.allocations_count,
            self.reuse_count,
            self.allocated_size,
            total_size - free_size, // Actual usage
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
    // Reset
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
    // Compression (Defragmentation)
    // ============================================================

    pub fn defragment(&mut self) {
        // Collect all allocated blocks
        let mut allocs: Vec<(usize, usize, u64)> = self
            .allocations
            .iter()
            .map(|(&id, &(offset, size))| (offset, size, id))
            .collect();
        allocs.sort_by_key(|(offset, _, _)| *offset);

        // Reorganize to contiguous area
        let mut new_offset = 0;
        let mut new_allocs = HashMap::new();

        for (old_offset, size, id) in allocs {
            if new_offset != old_offset {
                // Move data
                let src = old_offset;
                let dst = new_offset;
                for i in 0..size {
                    self.arena[dst + i] = self.arena[src + i];
                }
            }
            new_allocs.insert(id, (new_offset, size));
            new_offset += size;
        }

        // Update allocation table
        self.allocations = new_allocs;

        // Reset free blocks
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
// Tensor-specific Memory Pool Adapter
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
    // Compatibility with old interface (for Executor)
    // ============================================================

    pub fn allocate_or_use(
        &mut self,
        value_id: u64,
        tensor: Tensor<f32>,
    ) -> Tensor<f32> {
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

        // Free first block: use id=0 (id returned by first allocation)
        pool.free(0);

        // Allocate third block, should reuse first block
        let (offset3, _) = pool.allocate(60).unwrap();
        assert_eq!(offset3, 0);
    }

    #[test]
    fn test_byte_pool_defragment() {
        let mut pool = BytePool::new_with_blocks(64, 10);

        let (o1, _) = pool.allocate(100).unwrap(); // id=0, offset=0
        let (o2, _) = pool.allocate(50).unwrap(); // id=1, offset=128
        let (o3, _) = pool.allocate(80).unwrap(); // id=2, offset=192

        // Free using id
        pool.free(0); // Free first block
        pool.free(2); // Free third block

        // Fragmentation
        assert!(pool.fragmentation() > 0.0);

        pool.defragment();

        // Fragmentation reduced after compression
        assert!(pool.fragmentation() < 0.1);
    }
}