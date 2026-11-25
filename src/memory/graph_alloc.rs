// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

//! Graph allocator with lifetime tracking (llama.cpp ggml_gallocr design)
//!
//! Key optimizations from llama.cpp:
//! 1. Track n_children (consumer count) for each tensor
//! 2. Track n_views (view reference count) for view tensors
//! 3. Free tensor memory when n_children == 0 && n_views == 0
//! 4. In-place operation detection for supported ops
//! 5. Memory reuse across layers via free block management
//!
//! This is a direct port of llama.cpp's ggml-alloc.c design with
//! the same allocation strategy and lifetime tracking algorithm.

use std::collections::HashMap;
use std::ptr;

// ============================================================================
// Core Types (matching llama.cpp)
// ============================================================================

/// Maximum number of free blocks to track (same as llama.cpp's MAX_FREE_BLOCKS)
const MAX_FREE_BLOCKS: usize = 256;

/// Relative memory address within an allocation
/// Matches llama.cpp's `struct buffer_address`
#[derive(Clone, Copy, Debug, Default)]
struct BufferAddress {
    chunk: i32,    // -1 means invalid
    offset: usize, // Local memory offset within the buffer
}

impl BufferAddress {
    const INVALID: Self = Self { chunk: -1, offset: usize::MAX };
}

/// Free block in the allocator
/// Matches llama.cpp's `struct free_block`
#[derive(Clone, Copy, Debug)]
struct FreeBlock {
    offset: usize,
    size: usize,
}

/// Hash node for tracking tensor lifetime
/// Matches llama.cpp's hash table for tensor metadata
#[derive(Clone, Debug, Default)]
struct HashNode {
    n_children: u32, // Number of consumers
    n_views: u32,    // Number of view references
    allocated: bool, // Allocated by this allocator
    addr: BufferAddress,
    buffer_id: i32,
}

/// Tensor allocation record
#[derive(Clone, Copy, Debug)]
struct TensorAlloc {
    buffer_id: i32,
    addr: BufferAddress,
    size_max: usize,
}

/// Node allocation record (for graph nodes and their inputs)
#[derive(Clone, Copy, Debug)]
struct NodeAlloc {
    dst: TensorAlloc,
    src: [TensorAlloc; 4], // GGML_MAX_SRC = 4
}

// ============================================================================
// Dynamic Tensor Allocator (ggml_dyn_tallocr)
// ============================================================================

/// Dynamic allocator for a single buffer
/// Matches llama.cpp's `struct ggml_dyn_tallocr`
struct DynTAlloc {
    alignment: usize,
    max_chunk_size: usize,
    free_blocks: Vec<FreeBlock>,
    max_size: usize,
}

impl DynTAlloc {
    fn new(alignment: usize, max_size: usize) -> Self {
        let mut alloc = Self {
            alignment,
            max_chunk_size: max_size,
            free_blocks: Vec::with_capacity(MAX_FREE_BLOCKS),
            max_size: 0,
        };
        // Initialize with one free block covering entire buffer
        alloc.free_blocks.push(FreeBlock { offset: 0, size: max_size });
        alloc
    }

    fn aligned_offset(&self, offset: usize) -> usize {
        (offset + self.alignment - 1) & !(self.alignment - 1)
    }

    /// Allocate bytes, returning offset
    /// Matches llama.cpp's `ggml_dyn_tallocr_alloc`
    fn alloc(&mut self, size: usize, tensor_name: &str) -> usize {
        let size = self.aligned_offset(size);

        // Find best fitting free block
        let mut best_fit_idx: Option<usize> = None;
        let mut best_fit_size = usize::MAX;
        let mut max_avail = 0usize;

        // Search all blocks except the last one (which can grow)
        for i in 0..self.free_blocks.len().saturating_sub(1) {
            let block = &self.free_blocks[i];
            max_avail = max_avail.max(block.size);
            if block.size >= size && block.size < best_fit_size {
                best_fit_idx = Some(i);
                best_fit_size = block.size;
            }
        }

        // If no suitable block found, try the last block
        if best_fit_idx.is_none() && !self.free_blocks.is_empty() {
            let last_idx = self.free_blocks.len() - 1;
            let block = &self.free_blocks[last_idx];
            max_avail = max_avail.max(block.size);
            if block.size >= size {
                best_fit_idx = Some(last_idx);
            }
        }

        let (offset, remaining_size) = if let Some(idx) = best_fit_idx {
            let block = self.free_blocks[idx];
            let remaining = block.size - size;

            if remaining > 0 {
                // Shrink the block
                self.free_blocks[idx].offset = block.offset + size;
                self.free_blocks[idx].size = remaining;
            } else {
                // Remove the block
                self.free_blocks.remove(idx);
            }
            (block.offset, remaining)
        } else {
            // Out of memory
            panic!(
                "DynTAlloc: not enough space for tensor '{}' (need {} bytes, max avail {} bytes)",
                tensor_name, size, max_avail
            );
        };

        // Update max_size for statistics
        self.max_size = self.max_size.max(offset + size);

        offset
    }

    /// Free bytes at the given address
    /// Matches llama.cpp's `ggml_dyn_tallocr_free_bytes`
    fn free(&mut self, addr: usize, size: usize) {
        let size = self.aligned_offset(size);

        // Find the appropriate position and merge/insert
        let mut merge_idx: Option<usize> = None;
        let mut merge_type: i32 = 0; // 0: end, 1: beginning

        for i in 0..self.free_blocks.len() {
            let block = self.free_blocks[i];

            // Check if addr is at the end of this block
            if block.offset + block.size == addr {
                merge_idx = Some(i);
                merge_type = 0;
                break;
            }

            // Check if addr is at the beginning of this block
            if addr + size == block.offset {
                merge_idx = Some(i);
                merge_type = 1;
                break;
            }
        }

        if let Some(idx) = merge_idx {
            if merge_type == 0 {
                // Extend block at the end
                self.free_blocks[idx].size += size;

                // Try to merge with next block
                if idx < self.free_blocks.len() - 1 {
                    let block_end = self.free_blocks[idx].offset
                        + self.free_blocks[idx].size;
                    let next_start = self.free_blocks[idx + 1].offset;
                    if block_end == next_start {
                        let merged_size = self.free_blocks[idx].size
                            + self.free_blocks[idx + 1].size;
                        self.free_blocks[idx].size = merged_size;
                        self.free_blocks.remove(idx + 1);
                    }
                }
            } else {
                // Extend block at the beginning
                self.free_blocks[idx].offset = addr;
                self.free_blocks[idx].size += size;

                // Try to merge with previous block
                if idx > 0 {
                    let prev_end = self.free_blocks[idx - 1].offset
                        + self.free_blocks[idx - 1].size;
                    let block_start = self.free_blocks[idx].offset;
                    if prev_end == block_start {
                        let merged_size = self.free_blocks[idx - 1].size
                            + self.free_blocks[idx].size;
                        self.free_blocks[idx - 1].size = merged_size;
                        self.free_blocks.remove(idx);
                    }
                }
            }
        } else {
            // No adjacent block found, insert new free block
            if self.free_blocks.len() < MAX_FREE_BLOCKS {
                self.insert_block_sorted(addr, size);
            }
        }
    }

    /// Insert a new free block in sorted order (by offset)
    fn insert_block_sorted(&mut self, offset: usize, size: usize) {
        let insert_pos = self
            .free_blocks
            .iter()
            .position(|b| b.offset > offset)
            .unwrap_or(self.free_blocks.len());

        self.free_blocks.insert(insert_pos, FreeBlock { offset, size });
    }

    fn reset(&mut self, max_size: usize) {
        self.free_blocks.clear();
        self.free_blocks.push(FreeBlock { offset: 0, size: max_size });
        self.max_size = 0;
    }
}

// ============================================================================
// Graph Allocator (ggml_gallocr)
// ============================================================================

/// Graph allocator with lifetime tracking
/// Matches llama.cpp's `struct ggml_gallocr`
pub struct GraphAllocator {
    /// Memory buffer
    buffer: Vec<u8>,
    /// Dynamic allocator
    talloc: DynTAlloc,
    /// Hash table for tensor metadata
    hash_set: HashMap<usize, HashNode>,
    /// Node allocation records
    node_allocs: Vec<NodeAlloc>,
    /// Leaf allocation records
    leaf_allocs: Vec<TensorAlloc>,
    /// Peak memory usage
    peak_usage: usize,
    /// Debug mode
    debug: bool,
}

/// Operation type for in-place detection
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OpType {
    // In-place operations (can reuse input memory)
    Fill,
    Scale,
    Add,
    Sub,
    Mul,
    Div,
    Sqr,
    Sqrt,
    Log,
    Unary, // sin, cos, gelu, silu, etc.
    Rope,
    RopeBack,
    SiluBack,
    RmsNorm,
    RmsNormBack,
    Clamp,
    SoftMax,
    SoftMaxBack,

    // Non-inplace operations
    MatMul,
    MatMulId,
    MulMat,
    MulMatId,
    GetRows,
    View,
    Reshape,
    Permute,
    Transpose,
    Cont,
    Cpy,
    Dup,
    AddId,
    Other,
}

impl OpType {
    /// Check if this operation can be done in-place
    /// Matches llama.cpp's `ggml_op_can_inplace`
    pub fn can_inplace(&self) -> bool {
        matches!(
            self,
            OpType::Fill
                | OpType::Scale
                | OpType::Add
                | OpType::AddId
                | OpType::Sub
                | OpType::Mul
                | OpType::Div
                | OpType::Sqr
                | OpType::Sqrt
                | OpType::Log
                | OpType::Unary
                | OpType::Rope
                | OpType::RopeBack
                | OpType::SiluBack
                | OpType::RmsNorm
                | OpType::RmsNormBack
                | OpType::Clamp
                | OpType::SoftMax
                | OpType::SoftMaxBack
        )
    }
}

/// Tensor descriptor for graph allocation
#[derive(Clone, Debug)]
pub struct TensorDesc {
    pub id: usize,
    pub name: String,
    pub size: usize,         // Size in bytes
    pub op_type: OpType,     // Operation type
    pub src_ids: Vec<usize>, // Source tensor IDs
    pub is_view: bool,
    pub view_src_id: Option<usize>,
    pub is_input: bool, // Is this an input tensor?
    pub is_leaf: bool,  // Is this a leaf tensor?
}

/// Computation graph for allocation
pub struct ComputeGraph {
    pub nodes: Vec<TensorDesc>, // Computation nodes (in order)
    pub leafs: Vec<TensorDesc>, // Input tensors
}

impl GraphAllocator {
    /// Create a new graph allocator with the specified capacity
    pub fn new(capacity: usize) -> Self {
        Self {
            buffer: vec![0u8; capacity],
            talloc: DynTAlloc::new(64, capacity), // 64-byte alignment
            hash_set: HashMap::new(),
            node_allocs: Vec::new(),
            leaf_allocs: Vec::new(),
            peak_usage: 0,
            debug: false,
        }
    }

    pub fn with_debug(mut self, debug: bool) -> Self {
        self.debug = debug;
        self
    }

    /// Reset the allocator for a new graph
    pub fn reset(&mut self) {
        self.talloc.reset(self.buffer.len());
        self.hash_set.clear();
        self.node_allocs.clear();
        self.leaf_allocs.clear();
    }

    /// Get hash node for a tensor, creating if needed
    fn get_hash_node(&mut self, tensor_id: usize) -> &mut HashNode {
        self.hash_set.entry(tensor_id).or_insert_with(HashNode::default)
    }

    /// Check if a tensor is already allocated
    fn is_allocated(&self, tensor_id: usize) -> bool {
        self.hash_set.get(&tensor_id).map_or(false, |n| n.allocated)
    }

    /// Check if this allocator owns a tensor
    fn is_own(&self, tensor_id: usize) -> bool {
        self.hash_set.get(&tensor_id).map_or(false, |n| n.allocated)
    }

    /// Allocate a tensor node
    /// Matches llama.cpp's `ggml_gallocr_allocate_node`
    fn allocate_node(
        &mut self,
        tensor_id: usize,
        size: usize,
        name: &str,
        buffer_id: i32,
    ) {
        // Check if already allocated
        if self.hash_set.get(&tensor_id).map_or(false, |n| n.allocated) {
            return;
        }

        // Allocate memory
        let offset = self.talloc.alloc(size, name);

        // Update hash node
        let node =
            self.hash_set.entry(tensor_id).or_insert_with(HashNode::default);
        node.allocated = true;
        node.addr = BufferAddress { chunk: buffer_id.max(0) as i32, offset };
        node.buffer_id = buffer_id;

        if self.debug {
            eprintln!("[ALLOC] tensor {} @ {} ({} bytes)", name, offset, size);
        }
    }

    /// Free a tensor node
    /// Matches llama.cpp's `ggml_gallocr_free_node`
    fn free_node(&mut self, tensor_id: usize, size: usize, name: &str) {
        let node = self.hash_set.get(&tensor_id);
        if let Some(node) = node {
            if node.allocated {
                self.talloc.free(node.addr.offset, size);
                if self.debug {
                    eprintln!(
                        "[FREE] tensor {} @ {} ({} bytes)",
                        name, node.addr.offset, size
                    );
                }
            }
        }
    }

    /// Reserve memory for a graph (analysis pass)
    /// Matches llama.cpp's `ggml_gallocr_reserve`
    pub fn reserve(&mut self, graph: &ComputeGraph) -> bool {
        // Reset state
        self.reset();

        // First pass: count children and views for each tensor
        // Matches llama.cpp's `ggml_gallocr_alloc_graph_impl` first part

        // Process leafs (inputs)
        for (i, leaf) in graph.leafs.iter().enumerate() {
            let node = self.get_hash_node(leaf.id);
            node.n_children = 0; // Will be incremented by consumers
            node.allocated = false;
        }

        // Process nodes and count references
        for node_desc in &graph.nodes {
            // Count views
            if node_desc.is_view {
                if let Some(view_src_id) = node_desc.view_src_id {
                    let src_node = self.get_hash_node(view_src_id);
                    src_node.n_views += 1;
                }
            }

            // Mark input tensors
            if node_desc.is_input {
                let node = self.get_hash_node(node_desc.id);
                node.allocated = true; // External tensor, not allocated by us
            }

            // Count children for source tensors
            for &src_id in &node_desc.src_ids {
                let src_node = self.get_hash_node(src_id);
                src_node.n_children += 1;
            }
        }

        // Second pass: allocate tensors in order, free when no longer needed
        // Matches llama.cpp's `ggml_gallocr_alloc_graph_impl` second part

        // Allocate leafs
        for (i, leaf) in graph.leafs.iter().enumerate() {
            if !leaf.is_view && !self.is_allocated(leaf.id) {
                self.allocate_node(leaf.id, leaf.size, &leaf.name, 0);
            }
        }

        // Allocate nodes in order
        for node_desc in &graph.nodes {
            // Allocate parents (sources) first
            for &src_id in &node_desc.src_ids {
                if !self.is_allocated(src_id) {
                    // Find the source tensor to get its size
                    let src_desc =
                        graph.nodes.iter().find(|n| n.id == src_id).or_else(
                            || graph.leafs.iter().find(|l| l.id == src_id),
                        );

                    if let Some(src) = src_desc {
                        if !src.is_view {
                            self.allocate_node(src_id, src.size, &src.name, 0);
                        }
                    }
                }
            }

            // Allocate this node
            if !node_desc.is_view && !self.is_allocated(node_desc.id) {
                // Check for in-place opportunity
                if node_desc.op_type.can_inplace()
                    && !node_desc.src_ids.is_empty()
                {
                    // Try to reuse first source's memory if it's the only consumer
                    let src_id = node_desc.src_ids[0];

                    // Check conditions first (read-only)
                    let can_inplace =
                        self.hash_set.get(&src_id).map_or(false, |src_node| {
                            src_node.n_children == 1
                                && src_node.n_views == 0
                                && src_node.allocated
                        });

                    if can_inplace {
                        // Copy src_node info first
                        let src_node_copy = self.hash_set.get(&src_id).cloned();
                        if let Some(src_node) = src_node_copy {
                            // Update current node
                            let node = self.get_hash_node(node_desc.id);
                            *node = src_node;
                            node.n_children = 0;
                            if self.debug {
                                eprintln!(
                                    "[INPLACE] tensor {} reuses {}'s memory",
                                    node_desc.name,
                                    graph
                                        .nodes
                                        .iter()
                                        .find(|n| n.id == src_id)
                                        .map(|n| n.name.as_str())
                                        .unwrap_or("?")
                                );
                            }
                            continue;
                        }
                    }
                }

                self.allocate_node(
                    node_desc.id,
                    node_desc.size,
                    &node_desc.name,
                    0,
                );
            }

            // Decrement children count for sources and free if needed
            for &src_id in &node_desc.src_ids {
                let src_node = self.get_hash_node(src_id);
                if src_node.n_children > 0 {
                    src_node.n_children -= 1;
                }

                // Free if no more consumers and no views
                if src_node.n_children == 0
                    && src_node.n_views == 0
                    && src_node.allocated
                {
                    // Find the source tensor to get its size
                    let src_desc =
                        graph.nodes.iter().find(|n| n.id == src_id).or_else(
                            || graph.leafs.iter().find(|l| l.id == src_id),
                        );

                    if let Some(src) = src_desc {
                        if !src.is_view {
                            self.free_node(src_id, src.size, &src.name);
                        }
                    }
                }
            }
        }

        // Store allocation records
        self.node_allocs = vec![
            NodeAlloc {
                dst: TensorAlloc {
                    buffer_id: -1,
                    addr: BufferAddress::INVALID,
                    size_max: 0
                },
                src: [TensorAlloc {
                    buffer_id: -1,
                    addr: BufferAddress::INVALID,
                    size_max: 0,
                }; 4],
            };
            graph.nodes.len()
        ];

        self.leaf_allocs = vec![
            TensorAlloc {
                buffer_id: -1,
                addr: BufferAddress::INVALID,
                size_max: 0,
            };
            graph.leafs.len()
        ];

        // Copy allocation info from hash table
        for (i, node_desc) in graph.nodes.iter().enumerate() {
            if let Some(hash_node) = self.hash_set.get(&node_desc.id) {
                self.node_allocs[i].dst.buffer_id = hash_node.buffer_id;
                self.node_allocs[i].dst.addr = hash_node.addr;
                self.node_allocs[i].dst.size_max = node_desc.size;
            }
        }

        for (i, leaf) in graph.leafs.iter().enumerate() {
            if let Some(hash_node) = self.hash_set.get(&leaf.id) {
                self.leaf_allocs[i].buffer_id = hash_node.buffer_id;
                self.leaf_allocs[i].addr = hash_node.addr;
                self.leaf_allocs[i].size_max = leaf.size;
            }
        }

        self.peak_usage = self.talloc.max_size;
        true
    }

    /// Get memory for a tensor
    pub fn get_tensor_memory<T>(&self, tensor_id: usize) -> Option<&mut [T]> {
        self.hash_set.get(&tensor_id).and_then(|node| {
            if node.allocated && node.addr.chunk >= 0 {
                let offset = node.addr.offset;
                let size = self.buffer.len() - offset; // Max possible size
                unsafe {
                    let ptr = self.buffer.as_ptr().add(offset) as *mut T;
                    Some(std::slice::from_raw_parts_mut(
                        ptr,
                        size / std::mem::size_of::<T>(),
                    ))
                }
            } else {
                None
            }
        })
    }

    /// Allocate memory for a tensor directly (for runtime allocation)
    pub fn alloc_tensor<T: Copy>(
        &mut self,
        name: &str,
        count: usize,
    ) -> &mut [T] {
        let size = count * std::mem::size_of::<T>();
        let offset = self.talloc.alloc(size, name);
        self.peak_usage = self.peak_usage.max(offset + size);

        unsafe {
            let ptr = self.buffer.as_mut_ptr().add(offset) as *mut T;
            let slice = std::slice::from_raw_parts_mut(ptr, count);
            ptr::write_bytes(slice.as_mut_ptr(), 0, count);
            slice
        }
    }

    /// Free memory for a tensor by name
    pub fn free_tensor(&mut self, offset: usize, size: usize) {
        self.talloc.free(offset, size);
    }

    /// Get peak memory usage
    pub fn peak_usage(&self) -> usize {
        self.peak_usage
    }

    /// Get current allocation offset
    pub fn current_offset(&self) -> usize {
        self.talloc.max_size
    }

    /// Get buffer capacity
    pub fn capacity(&self) -> usize {
        self.buffer.len()
    }

    /// Get raw buffer pointer
    pub fn buffer_ptr(&mut self) -> *mut u8 {
        self.buffer.as_mut_ptr()
    }
}

// ============================================================================
// Helper for building compute graphs
// ============================================================================

/// Builder for compute graphs
pub struct GraphBuilder {
    nodes: Vec<TensorDesc>,
    leafs: Vec<TensorDesc>,
    next_id: usize,
}

impl GraphBuilder {
    pub fn new() -> Self {
        Self { nodes: Vec::new(), leafs: Vec::new(), next_id: 0 }
    }

    /// Add a leaf tensor (input)
    pub fn add_leaf(
        &mut self,
        name: impl Into<String>,
        size: usize,
        is_input: bool,
    ) -> usize {
        let id = self.next_id;
        self.next_id += 1;
        self.leafs.push(TensorDesc {
            id,
            name: name.into(),
            size,
            op_type: OpType::Other,
            src_ids: Vec::new(),
            is_view: false,
            view_src_id: None,
            is_input,
            is_leaf: true,
        });
        id
    }

    /// Add a computation node
    pub fn add_node(
        &mut self,
        name: impl Into<String>,
        size: usize,
        op_type: OpType,
        src_ids: Vec<usize>,
    ) -> usize {
        let id = self.next_id;
        self.next_id += 1;
        self.nodes.push(TensorDesc {
            id,
            name: name.into(),
            size,
            op_type,
            src_ids,
            is_view: false,
            view_src_id: None,
            is_input: false,
            is_leaf: false,
        });
        id
    }

    /// Add a view tensor
    pub fn add_view(
        &mut self,
        name: impl Into<String>,
        size: usize,
        view_src_id: usize,
    ) -> usize {
        let id = self.next_id;
        self.next_id += 1;
        self.nodes.push(TensorDesc {
            id,
            name: name.into(),
            size,
            op_type: OpType::View,
            src_ids: vec![view_src_id],
            is_view: true,
            view_src_id: Some(view_src_id),
            is_input: false,
            is_leaf: false,
        });
        id
    }

    /// Build the compute graph
    pub fn build(self) -> ComputeGraph {
        ComputeGraph { nodes: self.nodes, leafs: self.leafs }
    }
}

impl Default for GraphBuilder {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Layer-specific allocation helpers
// ============================================================================

impl GraphAllocator {
    /// Allocate memory for a transformer layer forward pass
    /// Returns memory offsets for each tensor (not the actual slices)
    ///
    /// This matches llama.cpp's approach where:
    /// 1. normed_attn is freed after Q, K, V projections
    /// 2. Q, K, V are freed after attention
    /// 3. attn_output is freed after output projection
    /// 4. normed_ffn is freed after gate/up projections
    /// 5. gate, up are freed after SwiGLU
    pub fn allocate_layer(
        &mut self,
        hidden_size: usize,
        intermediate_size: usize,
        n_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
    ) -> LayerMemory {
        let f32_size = std::mem::size_of::<f32>();

        // Build the compute graph for one layer
        let mut builder = GraphBuilder::new();

        // Leaf: hidden state input (external, not allocated)
        let hidden_id =
            builder.add_leaf("hidden", hidden_size * f32_size, true);

        // Attention phase
        let normed_attn_id = builder.add_node(
            "normed_attn",
            hidden_size * f32_size,
            OpType::RmsNorm,
            vec![hidden_id],
        );

        // Q, K, V projections (all consume normed_attn)
        let q_id = builder.add_node(
            "q",
            n_heads * head_dim * f32_size,
            OpType::MulMat,
            vec![normed_attn_id],
        );
        let k_id = builder.add_node(
            "k",
            n_kv_heads * head_dim * f32_size,
            OpType::MulMat,
            vec![normed_attn_id],
        );
        let v_id = builder.add_node(
            "v",
            n_kv_heads * head_dim * f32_size,
            OpType::MulMat,
            vec![normed_attn_id],
        );

        // Q norm (in-place on q)
        let q_normed_id = builder.add_node(
            "q_normed",
            n_heads * head_dim * f32_size,
            OpType::RmsNorm,
            vec![q_id],
        );

        // K norm (in-place on k)
        let k_normed_id = builder.add_node(
            "k_normed",
            n_kv_heads * head_dim * f32_size,
            OpType::RmsNorm,
            vec![k_id],
        );

        // Attention output (consumes q_normed, k_normed, v)
        // Note: This is simplified - actual attention involves KV cache
        let attn_output_id = builder.add_node(
            "attn_output",
            n_heads * head_dim * f32_size,
            OpType::Other,
            vec![q_normed_id, k_normed_id, v_id],
        );

        // Output projection
        let attn_out_id = builder.add_node(
            "attn_out",
            hidden_size * f32_size,
            OpType::MulMat,
            vec![attn_output_id],
        );

        // Residual add (in-place on hidden)
        let hidden2_id = builder.add_node(
            "hidden_after_attn",
            hidden_size * f32_size,
            OpType::Add,
            vec![hidden_id, attn_out_id],
        );

        // FFN phase
        let normed_ffn_id = builder.add_node(
            "normed_ffn",
            hidden_size * f32_size,
            OpType::RmsNorm,
            vec![hidden2_id],
        );

        // Gate and Up projections
        let gate_id = builder.add_node(
            "gate",
            intermediate_size * f32_size,
            OpType::MulMat,
            vec![normed_ffn_id],
        );
        let up_id = builder.add_node(
            "up",
            intermediate_size * f32_size,
            OpType::MulMat,
            vec![normed_ffn_id],
        );

        // SwiGLU (in-place on gate, consumes up)
        let hidden_ffn_id = builder.add_node(
            "hidden_ffn",
            intermediate_size * f32_size,
            OpType::Mul,
            vec![gate_id, up_id],
        );

        // Down projection
        let ffn_out_id = builder.add_node(
            "ffn_out",
            hidden_size * f32_size,
            OpType::MulMat,
            vec![hidden_ffn_id],
        );

        // Final residual add
        let _hidden3_id = builder.add_node(
            "hidden_final",
            hidden_size * f32_size,
            OpType::Add,
            vec![hidden2_id, ffn_out_id],
        );

        // Build and reserve
        let graph = builder.build();
        self.reserve(&graph);

        // Extract allocation offsets
        LayerMemory {
            normed_attn: self.get_offset("normed_attn", &graph),
            q: self.get_offset("q", &graph),
            k: self.get_offset("k", &graph),
            v: self.get_offset("v", &graph),
            q_normed: self.get_offset("q_normed", &graph),
            k_normed: self.get_offset("k_normed", &graph),
            attn_output: self.get_offset("attn_output", &graph),
            attn_out: self.get_offset("attn_out", &graph),
            normed_ffn: self.get_offset("normed_ffn", &graph),
            gate: self.get_offset("gate", &graph),
            up: self.get_offset("up", &graph),
            ffn_out: self.get_offset("ffn_out", &graph),
        }
    }

    fn get_offset(&self, name: &str, graph: &ComputeGraph) -> usize {
        for node in &graph.nodes {
            if node.name == name {
                if let Some(hash_node) = self.hash_set.get(&node.id) {
                    return hash_node.addr.offset;
                }
            }
        }
        usize::MAX
    }
}

/// Memory layout for a layer
#[derive(Clone, Debug, Default)]
pub struct LayerMemory {
    pub normed_attn: usize,
    pub q: usize,
    pub k: usize,
    pub v: usize,
    pub q_normed: usize,
    pub k_normed: usize,
    pub attn_output: usize,
    pub attn_out: usize,
    pub normed_ffn: usize,
    pub gate: usize,
    pub up: usize,
    pub ffn_out: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dyn_talloc_basic() {
        let mut alloc = DynTAlloc::new(64, 1024);

        // First allocation
        let off1 = alloc.alloc(100, "test1");
        assert_eq!(off1, 0);

        // Second allocation (aligned)
        let off2 = alloc.alloc(100, "test2");
        assert!(off2 >= 100);

        // Free first
        alloc.free(off1, 100);

        // Third allocation should reuse freed memory
        let off3 = alloc.alloc(50, "test3");
        assert_eq!(off3, 0); // Reused
    }

    #[test]
    fn test_graph_allocator_layer() {
        let mut alloc = GraphAllocator::new(10 * 1024 * 1024).with_debug(true);

        // Qwen3-0.6B dimensions
        let layer_mem = alloc.allocate_layer(1024, 3072, 16, 8, 128);

        println!("Layer memory layout:");
        println!("  normed_attn: {}", layer_mem.normed_attn);
        println!("  q: {}", layer_mem.q);
        println!("  k: {}", layer_mem.k);
        println!("  v: {}", layer_mem.v);
        println!("  attn_output: {}", layer_mem.attn_output);
        println!("  peak usage: {} bytes", alloc.peak_usage());

        // Memory should be reused
        // normed_attn memory should be reused for later tensors
    }

    #[test]
    fn test_inplace_detection() {
        assert!(OpType::RmsNorm.can_inplace());
        assert!(OpType::Add.can_inplace());
        assert!(OpType::Mul.can_inplace());
        assert!(!OpType::MatMul.can_inplace());
        assert!(!OpType::MulMat.can_inplace());
    }
}
