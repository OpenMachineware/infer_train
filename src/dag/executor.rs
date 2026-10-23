// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

//! DAG Executor - Traverse DAG and dispatch operations
//!
//! Executes operations by traversing optimized DAG nodes

use super::graph::{DagNode, Operation, OptimizedDag};
use crate::model::LlamaModel;

#[cfg(all(target_arch = "aarch64", feature = "metal"))]
use crate::buffer::ForwardBufferPool;

#[cfg(all(target_arch = "aarch64", feature = "metal"))]
use crate::inference::KVCacheGPU;

#[cfg(all(target_arch = "aarch64", feature = "metal"))]
use std::cell::RefCell;

#[cfg(all(target_arch = "aarch64", feature = "metal"))]
use std::collections::HashMap;

#[cfg(all(target_arch = "aarch64", feature = "metal"))]
use std::rc::Rc;

#[cfg(all(target_arch = "aarch64", feature = "metal"))]
use metal::Buffer as MetalBuffer;

#[cfg(all(target_arch = "aarch64", feature = "metal"))]
use metal::CommandBuffer;

/// Tensor registry: maps tensor IDs to GPU buffers
/// Uses Vec for O(1) direct indexing instead of HashMap
#[cfg(all(target_arch = "aarch64", feature = "metal"))]
pub struct TensorRegistry {
    buffers: Vec<Option<Rc<MetalBuffer>>>,
    max_tensor_id: usize,
    /// Cached dtypes for weight tensors (avoid HashMap lookups)
    dtypes: Vec<Option<crate::model::DataType>>,
}

#[cfg(all(target_arch = "aarch64", feature = "metal"))]
impl TensorRegistry {
    pub fn new() -> Self {
        Self { buffers: Vec::new(), max_tensor_id: 0, dtypes: Vec::new() }
    }

    /// Pre-allocate space for expected number of tensors
    pub fn with_capacity(capacity: usize) -> Self {
        let mut buffers = Vec::with_capacity(capacity);
        buffers.resize(capacity, None);
        let mut dtypes = Vec::with_capacity(capacity);
        dtypes.resize(capacity, None);
        Self { buffers, max_tensor_id: 0, dtypes }
    }

    pub fn get(&self, id: usize) -> Option<Rc<MetalBuffer>> {
        self.buffers.get(id).and_then(|b| b.clone())
    }

    /// Get cached dtype for a tensor
    pub fn get_dtype(&self, id: usize) -> Option<crate::model::DataType> {
        self.dtypes.get(id).and_then(|d| *d)
    }

    pub fn insert(&mut self, id: usize, buffer: Rc<MetalBuffer>) {
        // Extend vector if needed
        if id >= self.buffers.len() {
            // Double the size for amortized O(1) growth
            let new_len = (id + 1).next_power_of_two();
            self.buffers.resize(new_len, None);
            self.dtypes.resize(new_len, None);
        }
        self.buffers[id] = Some(buffer);
        if id > self.max_tensor_id {
            self.max_tensor_id = id;
        }
    }

    /// Insert buffer with dtype (for weight tensors)
    pub fn insert_with_dtype(
        &mut self,
        id: usize,
        buffer: Rc<MetalBuffer>,
        dtype: crate::model::DataType,
    ) {
        self.insert(id, buffer);
        if id < self.dtypes.len() {
            self.dtypes[id] = Some(dtype);
        }
    }
}

/// Execution context for DAG traversal
#[cfg(all(target_arch = "aarch64", feature = "metal"))]
pub struct ExecutionContext<'a> {
    pub model: &'a LlamaModel,
    pub kv_cache: &'a mut KVCacheGPU,
    pub gpu_ctx: &'a crate::quant::vec_dot::gpu::metal::MetalContext,
    pub tensor_registry: TensorRegistry,
    pub weight_bindings: &'a HashMap<usize, crate::model::Tensor>,
    pub forward_pool: &'a mut ForwardBufferPool,
    pub layer_idx: usize,
    pub position: usize,
    pub use_mma: bool,    // Use MMA kernel for Flash Attention
    pub use_mma_v2: bool, // Use MMA v2 kernel
}

/// Async command buffer state for overlapping CPU-GPU execution
///
/// Async command buffer for CPU-GPU pipeline overlap.
/// Stores command buffer reference for deferred wait.
#[cfg(all(target_arch = "aarch64", feature = "metal"))]
pub struct AsyncCommandBuffer {
    /// Output buffer containing logits
    output_buffer: Rc<MetalBuffer>,
    vocab_size: usize,
    /// Raw pointer to command buffer (for async wait)
    /// We use raw pointer because CommandBufferRef is a reference type
    command_buffer: Option<*mut metal::CommandBufferRef>,
    /// Flag indicating whether GPU execution has completed
    completed: bool,
}

#[cfg(all(target_arch = "aarch64", feature = "metal"))]
impl AsyncCommandBuffer {
    pub fn new(
        output_buffer: Rc<MetalBuffer>,
        vocab_size: usize,
        command_buffer: &metal::CommandBufferRef,
    ) -> Self {
        // Retain the command buffer to keep it alive until we wait
        // Using Objective-C retain semantics
        let cb_ptr = command_buffer as *const metal::CommandBufferRef
            as *mut metal::CommandBufferRef;
        unsafe {
            // Increment reference count
            let _ = metal::objc::rc::autoreleasepool(|| {
                std::mem::forget(command_buffer);
            });
        }

        Self {
            output_buffer,
            vocab_size,
            command_buffer: Some(cb_ptr),
            completed: false,
        }
    }

    /// Wait for command buffer completion
    pub fn wait(&mut self) {
        if self.completed {
            return;
        }

        if let Some(cb_ptr) = self.command_buffer {
            unsafe {
                let cb = &*cb_ptr;
                cb.wait_until_completed();
            }
            self.completed = true;
        }
    }

    /// Check if command buffer has completed
    pub fn is_completed(&self) -> bool {
        self.completed
    }

    /// Get logits from output buffer
    /// Must call wait() first to ensure GPU has finished
    pub fn get_logits(&self) -> Vec<f32> {
        if !self.completed {
            eprintln!(
                "[WARN] get_logits called before wait() - data may be incomplete"
            );
        }

        let mut logits = vec![0.0f32; self.vocab_size];
        unsafe {
            std::ptr::copy_nonoverlapping(
                self.output_buffer.contents() as *const f32,
                logits.as_mut_ptr(),
                self.vocab_size,
            );
        }
        logits
    }
}

#[cfg(all(target_arch = "aarch64", feature = "metal"))]
impl Drop for AsyncCommandBuffer {
    fn drop(&mut self) {
        // Command buffer will be released automatically when this drops
        // The output_buffer Rc will keep the data alive
        self.command_buffer.take();
    }
}

pub struct DagExecutor;

#[cfg(all(target_arch = "aarch64", feature = "metal"))]
pub struct DagForwardResult {
    pub logits: Option<Vec<f32>>,
    pub pending_cb: Option<AsyncCommandBuffer>,
}

impl DagExecutor {
    /// Execute forward pass by traversing DAG nodes
    ///
    /// This is the synchronous version - waits for GPU completion before returning.
    /// For async execution, use `forward_dag_async` instead.
    #[cfg(all(target_arch = "aarch64", feature = "metal"))]
    pub fn forward_dag(
        model: &LlamaModel,
        tokens: &[u32],
        kv_cache: &mut KVCacheGPU,
        gpu_ctx: &crate::quant::vec_dot::gpu::metal::MetalContext,
    ) -> Option<Vec<f32>> {
        let mut result =
            Self::forward_dag_async(model, tokens, kv_cache, gpu_ctx, None);

        // Wait for GPU completion and get logits
        if let Some(ref mut async_cb) = result.pending_cb {
            async_cb.wait();
            Some(async_cb.get_logits())
        } else {
            result.logits
        }
    }

    /// Execute forward pass asynchronously
    ///
    /// Returns a DagForwardResult containing:
    /// - `logits`: None (will be populated after wait)
    /// - `pending_cb`: The async command buffer that's executing on GPU
    ///
    /// The caller can:
    /// 1. Immediately start preparing the next token (CPU work overlaps GPU execution)
    /// 2. Call `result.wait_and_get_logits()` when logits are needed
    ///
    /// IMPORTANT: The returned AsyncCommandBuffer must be kept alive until completion.
    /// It retains the output buffer needed for logits.
    #[cfg(all(target_arch = "aarch64", feature = "metal"))]
    pub fn forward_dag_async(
        model: &LlamaModel,
        tokens: &[u32],
        kv_cache: &mut KVCacheGPU,
        gpu_ctx: &crate::quant::vec_dot::gpu::metal::MetalContext,
        _pending_cb: Option<&AsyncCommandBuffer>, // Previous buffer (for dependency tracking)
    ) -> DagForwardResult {
        if tokens.is_empty() || model.tok_embeddings.is_none() {
            return DagForwardResult { logits: None, pending_cb: None };
        }

        let dag = match model.optimized_dag.as_ref() {
            Some(dag) => dag,
            None => return DagForwardResult { logits: None, pending_cb: None },
        };

        let hidden_size = model.config.hidden_size;
        let device = gpu_ctx.device();
        let queue = gpu_ctx.queue();

        // Create buffer pool
        let reuse_pool: Rc<RefCell<Vec<Rc<MetalBuffer>>>> =
            Rc::new(RefCell::new(Vec::new()));
        let mut forward_pool =
            ForwardBufferPool::new(device.clone(), reuse_pool.clone());

        // Process first token (simplified - handle single token for now)
        let token = tokens[0];

        // 1. Embedding lookup
        let hidden = match model.embedding_lookup_impl(token) {
            Some(h) => h,
            None => return DagForwardResult { logits: None, pending_cb: None },
        };

        // Use cached embed buffer to avoid allocation on every decode
        let hidden_buffer =
            Rc::new(gpu_ctx.get_or_create_embed_buffer(hidden_size * 4));
        unsafe {
            std::ptr::copy_nonoverlapping(
                hidden.as_ptr() as *const std::ffi::c_void,
                hidden_buffer.contents(),
                hidden_size * 4,
            );
        }

        // Debug: check embedding for NaN
        if std::env::var("INFER_TRACE_NAN").is_ok() {
            let has_nan = hidden.iter().any(|&x| x.is_nan() || x.is_infinite());
            eprintln!("[NAN] Embedding has NaN/Inf: {}", has_nan);
            if has_nan {
                eprintln!(
                    "[NAN] First 10 embedding values: {:?}",
                    &hidden[..10]
                );
            }
        }

        // Create tensor registry
        let mut tensor_registry = TensorRegistry::new();

        // Bind initial input tensor
        tensor_registry.insert(0, hidden_buffer);

        // Bind all weight tensors to GPU buffers (with dtype caching)
        for (tensor_id, tensor) in &dag.dag.weight_bindings {
            let weight_buffer = gpu_ctx.get_or_create_buffer(tensor);
            tensor_registry.insert_with_dtype(
                *tensor_id,
                weight_buffer,
                tensor.dtype,
            );
        }

        // Check if MMA kernel is enabled via environment variable
        // MMA kernels disabled by default due to NaN issues with GQA models
        // TODO: Fix MMA kernels and re-enable
        let use_mma = std::env::var("INFER_USE_MMA")
            .map(|v| v == "1" || v == "true" || v == "v2")
            .unwrap_or(false); // Disabled by default
        let use_mma_v2 =
            std::env::var("INFER_USE_MMA").map(|v| v == "v2").unwrap_or(false); // Disabled by default

        // Create execution context
        let mut exec_ctx = ExecutionContext {
            model,
            kv_cache,
            gpu_ctx,
            tensor_registry,
            weight_bindings: &dag.dag.weight_bindings,
            forward_pool: &mut forward_pool,
            layer_idx: 0,
            position: 0,
            use_mma,
            use_mma_v2,
        };

        // Use single command buffer for entire graph
        // Use retained command buffer for async execution
        let command_buffer = queue.new_command_buffer();

        let encoder = command_buffer.new_compute_command_encoder();

        // Traverse DAG nodes in execution order
        for node_id in &dag.execution_order {
            let node = &dag.dag.nodes[*node_id];

            // Debug: trace all operations
            if std::env::var("INFER_TRACE_DAG").is_ok() {
                eprintln!(
                    "[TRACE] Node {}: {:?}",
                    node_id,
                    std::mem::discriminant(&node.op)
                );
            }

            // Debug: show registry state before executing node 534
            if std::env::var("INFER_DEBUG_DAG").is_ok() && *node_id == 534 {
                let count = exec_ctx
                    .tensor_registry
                    .buffers
                    .iter()
                    .filter(|b| b.is_some())
                    .count();
                eprintln!(
                    "[DAG] Before node 534: registry has {} entries",
                    count
                );

                // Show the actual FusedQKV operation details
                if let Operation::FusedQKV {
                    input,
                    q_weight,
                    k_weight,
                    v_weight,
                    output_q,
                    output_k,
                    output_v,
                    ..
                } = &node.op
                {
                    eprintln!(
                        "[DAG] FusedQKV details: input={}, outputs={}/{}/{}",
                        input, output_q, output_k, output_v
                    );
                }
            }

            // Execute node
            let result = Self::execute_node(node, &mut exec_ctx, &encoder);

            // Trace exit point execution
            if std::env::var("INFER_DEBUG_DAG").is_ok()
                && dag.dag.exit_points.contains(&node_id)
            {
                eprintln!(
                    "[DAG] Exit point node {} executed: success={}",
                    node_id,
                    result.is_some()
                );
            }

            if result.is_none() {
                if std::env::var("INFER_DEBUG_DAG").is_ok() {
                    eprintln!(
                        "[DAG] Node {} ({:?}) failed! Op details: {:?}",
                        node_id,
                        std::mem::discriminant(&node.op),
                        node.op
                    );
                    // Check which tensors are missing
                    match &node.op {
                        Operation::FusedGateUp {
                            input,
                            output_gate,
                            output_up,
                            ..
                        } => {
                            eprintln!(
                                "[DAG] FusedGateUp needs tensors: input={}, outputs={}/{}",
                                input, output_gate, output_up
                            );
                            if exec_ctx.tensor_registry.get(*input).is_none() {
                                eprintln!(
                                    "[DAG] Missing input tensor {}",
                                    input
                                );
                            }
                        }
                        Operation::FusedSwiGLU { gate, up, .. } => {
                            eprintln!(
                                "[DAG] FusedSwiGLU needs tensors: gate={}, up={}",
                                gate, up
                            );
                            if exec_ctx.tensor_registry.get(*gate).is_none() {
                                eprintln!("[DAG] Missing gate tensor {}", gate);
                            }
                            if exec_ctx.tensor_registry.get(*up).is_none() {
                                eprintln!("[DAG] Missing up tensor {}", up);
                            }
                        }
                        Operation::FusedAddNorm {
                            lhs,
                            rhs,
                            norm_weight,
                            ..
                        } => {
                            eprintln!(
                                "[DAG] FusedAddNorm needs tensors: lhs={}, rhs={}, norm_weight={}",
                                lhs, rhs, norm_weight
                            );
                            if exec_ctx.tensor_registry.get(*lhs).is_none() {
                                eprintln!("[DAG] Missing lhs tensor {}", lhs);
                            }
                            if exec_ctx.tensor_registry.get(*rhs).is_none() {
                                eprintln!("[DAG] Missing rhs tensor {}", rhs);
                            }
                            if exec_ctx
                                .tensor_registry
                                .get(*norm_weight)
                                .is_none()
                            {
                                eprintln!(
                                    "[DAG] Missing norm_weight tensor {}",
                                    norm_weight
                                );
                            }
                        }
                        _ => {}
                    }
                }
                // Clean up encoder before returning
                encoder.end_encoding();
                return DagForwardResult { logits: None, pending_cb: None };
            }
        }

        // Commit command buffer for async execution (like llama.cpp)
        // DO NOT wait - let the caller decide when to wait
        // This enables CPU-GPU overlap (software pipelining)
        encoder.end_encoding();

        // Get output tensor from the exit node
        let exit_node_id = dag.dag.exit_points[0];
        let exit_node = &dag.dag.nodes[exit_node_id];

        if std::env::var("INFER_DEBUG_DAG").is_ok() {
            eprintln!(
                "[DAG] Exit node ID: {}, op: {:?}",
                exit_node_id,
                std::mem::discriminant(&exit_node.op)
            );
        }

        // Extract output tensor ID from the exit node's operation
        let output_tensor_id = match &exit_node.op {
            Operation::Gemv { output, .. } => *output,
            Operation::FusedNormLinear { output, .. } => *output,
            Operation::Add { output, .. } => *output,
            Operation::RmsNorm { output, .. } => *output,
            Operation::Mul { output, .. } => *output,
            _ => {
                eprintln!(
                    "[DAG] Exit node has unsupported operation: {:?}",
                    exit_node.op
                );
                return DagForwardResult { logits: None, pending_cb: None };
            }
        };

        if std::env::var("INFER_DEBUG_DAG").is_ok() {
            eprintln!("[DAG] Output tensor ID: {}", output_tensor_id);
        }

        let output_buffer = exec_ctx.tensor_registry.get(output_tensor_id);

        if output_buffer.is_none() {
            eprintln!(
                "[DAG] Output tensor {} not found in registry",
                output_tensor_id
            );
            return DagForwardResult { logits: None, pending_cb: None };
        }

        let output_buffer = output_buffer.unwrap();

        // Get vocab size for logits extraction
        let output_tensor = model.output.as_ref();
        if output_tensor.is_none() {
            eprintln!("[DAG] Model has no output tensor");
            return DagForwardResult { logits: None, pending_cb: None };
        }
        let vocab_size = output_tensor.unwrap().shape[0];

        // Create async command buffer with output buffer retained
        let async_cb =
            AsyncCommandBuffer::new(output_buffer, vocab_size, command_buffer);

        // Commit the command buffer for async execution
        command_buffer.commit();

        // CRITICAL: Advance KV cache position
        // We advance here because the kernel encoding is complete.
        // The actual GPU write will happen asynchronously, but Metal's
        // command queue ordering ensures the next CB won't start
        // until this one completes.
        exec_ctx.kv_cache.advance();

        // Return async result - caller must call wait() before get_logits()
        DagForwardResult {
            logits: None, // No logits yet - must wait
            pending_cb: Some(async_cb),
        }
    }

    /// Check tensors for NaN values (debug utility)
    #[cfg(all(target_arch = "aarch64", feature = "metal"))]
    fn check_tensors_for_nan(
        dag: &super::graph::OptimizedDag,
        tensor_registry: &TensorRegistry,
    ) {
        if std::env::var("INFER_TRACE_NAN").is_err() {
            return;
        }

        let mut first_nan_node: Option<usize> = None;

        for node_id in &dag.execution_order {
            let node = &dag.dag.nodes[*node_id];
            let outputs = Self::get_node_outputs(&node.op);

            for output_id in &outputs {
                if let Some(buffer) = tensor_registry.get(*output_id) {
                    let ptr = buffer.contents() as *const f32;
                    let len = buffer.length() as usize / 4;

                    let has_nan = (0..std::cmp::min(10, len)).any(|i| {
                        let val = unsafe { *ptr.add(i) };
                        val.is_nan() || val.is_infinite()
                    });

                    let all_zeros = (0..std::cmp::min(10, len)).all(|i| {
                        let val = unsafe { *ptr.add(i) };
                        val == 0.0
                    });

                    if has_nan && first_nan_node.is_none() {
                        first_nan_node = Some(*node_id);
                        eprintln!(
                            "[NAN] FIRST NaN at node {} ({:?}) output tensor {}",
                            node_id,
                            std::mem::discriminant(&node.op),
                            output_id
                        );

                        // Print input tensors
                        let inputs = Self::get_node_inputs(&node.op);
                        for input_id in &inputs {
                            if let Some(buf) = tensor_registry.get(*input_id) {
                                let ptr = buf.contents() as *const f32;
                                let len = buf.length() as usize / 4;
                                let input_has_nan = (0..std::cmp::min(10, len))
                                    .any(|i| {
                                        let val = unsafe { *ptr.add(i) };
                                        val.is_nan() || val.is_infinite()
                                    });
                                let input_all_zeros =
                                    (0..std::cmp::min(10, len)).all(|i| {
                                        let val = unsafe { *ptr.add(i) };
                                        val == 0.0
                                    });
                                eprintln!(
                                    "  Input tensor {}: has_nan={}, all_zeros={}",
                                    input_id, input_has_nan, input_all_zeros
                                );
                            }
                        }
                    }

                    // Trace FlashAttention outputs specifically
                    if matches!(&node.op, Operation::FlashAttention { .. }) {
                        if let Operation::FlashAttention { layer_idx, .. } =
                            &node.op
                        {
                            eprintln!(
                                "[CHECK] FlashAttention node {} (layer {}): tensor {}, has_nan={}, all_zeros={}, first_vals={:?}",
                                node_id,
                                layer_idx,
                                output_id,
                                has_nan,
                                all_zeros,
                                (0..5)
                                    .map(|i| unsafe { *ptr.add(i) })
                                    .collect::<Vec<_>>()
                            );
                        }
                    }
                }
            }
        }
    }

    /// Execute a single DAG node
    #[cfg(all(target_arch = "aarch64", feature = "metal"))]
    fn execute_node(
        node: &DagNode,
        ctx: &mut ExecutionContext,
        encoder: &metal::ComputeCommandEncoderRef,
    ) -> Option<()> {
        // Debug: trace all nodes in range
        if std::env::var("INFER_TRACE_NAN").is_ok()
            && node.id >= 520
            && node.id <= 545
        {
            eprintln!(
                "[NODE] Node {}: {:?}",
                node.id,
                std::mem::discriminant(&node.op)
            );
        }

        match &node.op {
            Operation::EmbeddingLookup {
                token_id: _,
                embedding_weight: _,
                output: _,
            } => {
                // Embedding lookup already done before DAG traversal
                // This node is just a placeholder
                Some(())
            }

            Operation::RmsNorm { input, weight, output, epsilon } => {
                let input_buf = ctx.tensor_registry.get(*input)?;
                let weight_buf = ctx.tensor_registry.get(*weight)?;
                let output_buf =
                    ctx.forward_pool.get(ctx.model.config.hidden_size * 4);

                if std::env::var("INFER_TRACE_NAN").is_ok()
                    && *output >= 725
                    && *output <= 745
                {
                    eprintln!(
                        "[TRACE] RmsNorm node {}: input={}, output={}",
                        node.id, input, output
                    );
                    let input_ptr = input_buf.contents() as *const f32;
                    let input_len = (input_buf.length() / 4) as usize;
                    let input_vals: Vec<f32> = unsafe {
                        std::slice::from_raw_parts(
                            input_ptr,
                            std::cmp::min(5, input_len),
                        )
                        .to_vec()
                    };
                    eprintln!("  input tensor {}: {:?}", input, input_vals);
                }

                if std::env::var("INFER_TRACE_DAG").is_ok() {
                    eprintln!(
                        "[TRACE] RmsNorm: input={}, output={}",
                        input, output
                    );
                }

                ctx.gpu_ctx.rms_norm_encode(
                    encoder,
                    &input_buf,
                    &weight_buf,
                    &output_buf,
                    1,
                    ctx.model.config.hidden_size,
                    *epsilon,
                );

                ctx.tensor_registry.insert(*output, output_buf);
                Some(())
            }

            Operation::Gemv { input, weight, output, out_size } => {
                let input_buf = ctx.tensor_registry.get(*input)?;
                let weight_buf = ctx.tensor_registry.get(*weight)?;
                let output_buf = ctx.forward_pool.get(*out_size * 4);

                // Get weight tensor dtype from cached registry (avoid HashMap lookup)
                let dtype = ctx.tensor_registry.get_dtype(*weight)?;

                // Debug: print Gemv parameters
                if std::env::var("INFER_TRACE_NAN").is_ok() {
                    let weight_tensor = ctx.weight_bindings.get(weight)?;
                    eprintln!(
                        "[TRACE] Gemv node {}: input={}, weight={}, output={}, dtype={:?}, weight_shape={:?}",
                        node.id,
                        input,
                        weight,
                        output,
                        dtype,
                        weight_tensor.shape
                    );

                    // Check if weight buffer has valid data
                    let weight_ptr = weight_buf.contents() as *const u8;
                    let weight_len = weight_buf.length() as usize;
                    eprintln!(
                        "[TRACE] Weight buffer length: {} bytes",
                        weight_len
                    );

                    // Check if input buffer has valid size
                    let input_ptr = input_buf.contents() as *const f32;
                    let input_len = input_buf.length() as usize / 4;
                    eprintln!(
                        "[TRACE] Input buffer length: {} floats",
                        input_len
                    );

                    // Check first few input values
                    let first_vals: Vec<f32> = (0..std::cmp::min(5, input_len))
                        .map(|i| unsafe { *input_ptr.add(i) })
                        .collect();
                    eprintln!("[TRACE] First input values: {:?}", first_vals);
                }

                ctx.gpu_ctx.mv_encode(
                    encoder,
                    &weight_buf,
                    &input_buf,
                    &output_buf,
                    *out_size,
                    ctx.model.config.hidden_size,
                    dtype,
                );

                ctx.tensor_registry.insert(*output, output_buf);
                Some(())
            }

            Operation::FusedQKV {
                input,
                q_weight,
                k_weight,
                v_weight,
                output_q,
                output_k,
                output_v,
                q_out_size,
                k_out_size,
                v_out_size,
                layer_idx: _,
            } => {
                if std::env::var("INFER_DEBUG_DAG").is_ok() {
                    eprintln!(
                        "[DAG] FusedQKV: input={}, q_weight={}, k_weight={}, v_weight={}",
                        input, q_weight, k_weight, v_weight
                    );
                }

                let input_buf = match ctx.tensor_registry.get(*input) {
                    Some(buf) => buf,
                    None => {
                        if std::env::var("INFER_DEBUG_DAG").is_ok() {
                            eprintln!(
                                "[DAG] FusedQKV: input {} not in tensor_registry",
                                input
                            );
                        }
                        return None;
                    }
                };

                // Get weight buffers - check each one
                let q_weight_buf = match ctx.tensor_registry.get(*q_weight) {
                    Some(buf) => buf,
                    None => {
                        if std::env::var("INFER_DEBUG_DAG").is_ok() {
                            eprintln!(
                                "[DAG] FusedQKV: q_weight {} not in tensor_registry",
                                q_weight
                            );
                        }
                        return None;
                    }
                };
                let k_weight_buf = match ctx.tensor_registry.get(*k_weight) {
                    Some(buf) => buf,
                    None => {
                        if std::env::var("INFER_DEBUG_DAG").is_ok() {
                            eprintln!(
                                "[DAG] FusedQKV: k_weight {} not in tensor_registry",
                                k_weight
                            );
                        }
                        return None;
                    }
                };
                let v_weight_buf = match ctx.tensor_registry.get(*v_weight) {
                    Some(buf) => buf,
                    None => {
                        if std::env::var("INFER_DEBUG_DAG").is_ok() {
                            eprintln!(
                                "[DAG] FusedQKV: v_weight {} not in tensor_registry",
                                v_weight
                            );
                        }
                        return None;
                    }
                };

                // Allocate output buffers
                let q_buf = ctx.forward_pool.get(*q_out_size * 4);
                let k_buf = ctx.forward_pool.get(*k_out_size * 4);
                let v_buf = ctx.forward_pool.get(*v_out_size * 4);

                // Get weight dtypes from cached registry (avoid HashMap lookups)
                let q_dtype = match ctx.tensor_registry.get_dtype(*q_weight) {
                    Some(d) => d,
                    None => {
                        if std::env::var("INFER_DEBUG_DAG").is_ok() {
                            eprintln!(
                                "[DAG] FusedQKV: q_weight {} dtype not cached",
                                q_weight
                            );
                        }
                        return None;
                    }
                };
                let k_dtype = match ctx.tensor_registry.get_dtype(*k_weight) {
                    Some(d) => d,
                    None => {
                        if std::env::var("INFER_DEBUG_DAG").is_ok() {
                            eprintln!(
                                "[DAG] FusedQKV: k_weight {} dtype not cached",
                                k_weight
                            );
                        }
                        return None;
                    }
                };
                let v_dtype = match ctx.tensor_registry.get_dtype(*v_weight) {
                    Some(d) => d,
                    None => {
                        if std::env::var("INFER_DEBUG_DAG").is_ok() {
                            eprintln!(
                                "[DAG] FusedQKV: v_weight {} dtype not cached",
                                v_weight
                            );
                        }
                        return None;
                    }
                };

                let hidden_size = ctx.model.config.hidden_size;

                // Run three GEMV operations
                // TODO: Optimize to single GEMM when weights are contiguous
                if std::env::var("INFER_TRACE_NAN").is_ok() {
                    let input_ptr = input_buf.contents() as *const f32;
                    let has_nan =
                        (0..std::cmp::min(10, *q_out_size)).any(|i| {
                            let val = unsafe { *input_ptr.add(i) };
                            val.is_nan() || val.is_infinite()
                        });
                    eprintln!(
                        "[TRACE] FusedQKV input NaN: {}, dtype: Q={:?}, K={:?}, V={:?}",
                        has_nan, q_dtype, k_dtype, v_dtype
                    );
                }

                ctx.gpu_ctx.mv_encode(
                    encoder,
                    &q_weight_buf,
                    &input_buf,
                    &q_buf,
                    *q_out_size,
                    hidden_size,
                    q_dtype,
                );
                ctx.gpu_ctx.mv_encode(
                    encoder,
                    &k_weight_buf,
                    &input_buf,
                    &k_buf,
                    *k_out_size,
                    hidden_size,
                    k_dtype,
                );
                ctx.gpu_ctx.mv_encode(
                    encoder,
                    &v_weight_buf,
                    &input_buf,
                    &v_buf,
                    *v_out_size,
                    hidden_size,
                    v_dtype,
                );

                ctx.tensor_registry.insert(*output_q, q_buf);
                ctx.tensor_registry.insert(*output_k, k_buf);
                ctx.tensor_registry.insert(*output_v, v_buf);

                if std::env::var("INFER_TRACE_DAG").is_ok() {
                    eprintln!(
                        "[TRACE] FusedQKV outputs: q={}, k={}, v={}",
                        output_q, output_k, output_v
                    );
                }

                Some(())
            }

            Operation::FusedQKVWeight {
                input,
                qkv_weight,
                output_q,
                output_k,
                output_v,
                q_out_size,
                k_out_size,
                v_out_size,
                layer_idx: _,
            } => {
                // Fused QKV with merged weight tensor
                // Single GEMV operation, result is split into Q, K, V
                if std::env::var("INFER_DEBUG_DAG").is_ok() {
                    eprintln!(
                        "[DAG] FusedQKVWeight: input={}, qkv_weight={}",
                        input, qkv_weight
                    );
                }

                let input_buf = match ctx.tensor_registry.get(*input) {
                    Some(buf) => buf,
                    None => {
                        if std::env::var("INFER_DEBUG_DAG").is_ok() {
                            eprintln!(
                                "[DAG] FusedQKVWeight: input {} not in tensor_registry",
                                input
                            );
                        }
                        return None;
                    }
                };

                let qkv_weight_buf = match ctx.tensor_registry.get(*qkv_weight)
                {
                    Some(buf) => buf,
                    None => {
                        if std::env::var("INFER_DEBUG_DAG").is_ok() {
                            eprintln!(
                                "[DAG] FusedQKVWeight: qkv_weight {} not in tensor_registry",
                                qkv_weight
                            );
                        }
                        return None;
                    }
                };

                // Allocate output buffer for the fused QKV result
                let total_out_size = q_out_size + k_out_size + v_out_size;
                let qkv_buf = ctx.forward_pool.get(total_out_size * 4);

                // Get weight dtype from cached registry
                let qkv_dtype = match ctx.tensor_registry.get_dtype(*qkv_weight)
                {
                    Some(d) => d,
                    None => {
                        if std::env::var("INFER_DEBUG_DAG").is_ok() {
                            eprintln!(
                                "[DAG] FusedQKVWeight: qkv_weight {} dtype not cached",
                                qkv_weight
                            );
                        }
                        return None;
                    }
                };

                let hidden_size = ctx.model.config.hidden_size;

                // Single GEMV for the fused weight
                ctx.gpu_ctx.mv_encode(
                    encoder,
                    &qkv_weight_buf,
                    &input_buf,
                    &qkv_buf,
                    total_out_size,
                    hidden_size,
                    qkv_dtype,
                );

                // Allocate separate output buffers for Q, K, V
                let q_buf = ctx.forward_pool.get(*q_out_size * 4);
                let k_buf = ctx.forward_pool.get(*k_out_size * 4);
                let v_buf = ctx.forward_pool.get(*v_out_size * 4);

                // Split the fused QKV result into separate Q, K, V buffers
                ctx.gpu_ctx.split_qkv_encode(
                    encoder,
                    &qkv_buf,
                    &q_buf,
                    &k_buf,
                    &v_buf,
                    *q_out_size,
                    *k_out_size,
                    *v_out_size,
                );

                ctx.tensor_registry.insert(*output_q, q_buf);
                ctx.tensor_registry.insert(*output_k, k_buf);
                ctx.tensor_registry.insert(*output_v, v_buf);

                Some(())
            }

            Operation::FusedGateUp {
                input,
                gate_weight,
                up_weight,
                output_gate,
                output_up,
                intermediate_size,
                ..
            } => {
                let input_buf = ctx.tensor_registry.get(*input)?;
                let gate_weight_buf = ctx.tensor_registry.get(*gate_weight)?;
                let up_weight_buf = ctx.tensor_registry.get(*up_weight)?;

                let gate_buf = ctx.forward_pool.get(*intermediate_size * 4);
                let up_buf = ctx.forward_pool.get(*intermediate_size * 4);

                let hidden_size = ctx.model.config.hidden_size;

                // Get weight dtypes from cached registry
                let gate_dtype = ctx.tensor_registry.get_dtype(*gate_weight)?;
                let up_dtype = ctx.tensor_registry.get_dtype(*up_weight)?;

                if std::env::var("INFER_TRACE_NAN").is_ok()
                    && node.id >= 520
                    && node.id <= 545
                {
                    eprintln!(
                        "[TRACE] FusedGateUp node {}: input={}, outputs={}/{}",
                        node.id, input, output_gate, output_up
                    );
                }

                if std::env::var("INFER_TRACE_DAG").is_ok() {
                    eprintln!(
                        "[TRACE] FusedGateUp: input={}, outputs={}/{}",
                        input, output_gate, output_up
                    );
                }

                // Use fused kernel for Q4_K, Q5_K, Q6_K (optimized to read input once)
                // Fallback to separate GEMV for other formats
                match gate_dtype {
                    crate::model::tensor::DataType::Q4_K
                    | crate::model::tensor::DataType::Q5_K
                    | crate::model::tensor::DataType::Q6_K => {
                        // Fused kernel: read input once, compute both gate and up
                        ctx.gpu_ctx.fused_gate_up_encode(
                            encoder,
                            &gate_weight_buf,
                            &up_weight_buf,
                            &input_buf,
                            &gate_buf,
                            &up_buf,
                            *intermediate_size,
                            hidden_size,
                            gate_dtype,
                        );
                    }
                    _ => {
                        // Fallback: two separate GEMV calls for unsupported formats
                        ctx.gpu_ctx.mv_encode(
                            encoder,
                            &gate_weight_buf,
                            &input_buf,
                            &gate_buf,
                            *intermediate_size,
                            hidden_size,
                            gate_dtype,
                        );
                        ctx.gpu_ctx.mv_encode(
                            encoder,
                            &up_weight_buf,
                            &input_buf,
                            &up_buf,
                            *intermediate_size,
                            hidden_size,
                            up_dtype,
                        );
                    }
                }

                ctx.tensor_registry.insert(*output_gate, gate_buf);
                ctx.tensor_registry.insert(*output_up, up_buf);
                Some(())
            }

            Operation::RoPE { input, output, pos: _, n_heads, head_dim } => {
                let input_buf = ctx.tensor_registry.get(*input)?;
                let output_buf = ctx.forward_pool.get(*n_heads * *head_dim * 4);

                // Trace RoPE output tensor ID
                if std::env::var("INFER_TRACE_DAG").is_ok() {
                    eprintln!(
                        "[TRACE] RoPE: input={}, output={}",
                        input, output
                    );
                }

                // Debug: trace RoPE for layers 30-31
                if std::env::var("INFER_TRACE_NAN").is_ok()
                    && ctx.layer_idx >= 30
                {
                    let ptr = input_buf.contents() as *const f32;
                    let len = input_buf.length() as usize / 4;
                    let first_vals: Vec<f32> =
                        (0..5).map(|i| unsafe { *ptr.add(i) }).collect();
                    let has_nan = (0..std::cmp::min(10, len))
                        .any(|i| unsafe { (*ptr.add(i)).is_nan() });
                    eprintln!(
                        "[ROPE] Layer {} input tensor {}: has_nan={}, first_vals={:?}",
                        ctx.layer_idx, input, has_nan, first_vals
                    );
                }

                // Create position slice
                let pos_slice = [ctx.position as i32];

                ctx.gpu_ctx.rope_neox_encode(
                    encoder,
                    &input_buf,
                    &pos_slice,
                    1, // n_seqs
                    *n_heads,
                    *head_dim,
                    10000.0, // freq_base
                    1.0,     // freq_scale
                    &output_buf,
                );

                ctx.tensor_registry.insert(*output, output_buf);
                Some(())
            }

            Operation::FlashAttention {
                q,
                output,
                layer_idx,
                n_heads,
                n_kv_heads,
                head_dim,
            } => {
                let q_buf = ctx.tensor_registry.get(*q)?;

                // Trace FlashAttention input/output
                if std::env::var("INFER_TRACE_DAG").is_ok() {
                    eprintln!(
                        "[TRACE] FlashAttention: q={}, output={}, layer={}",
                        q, output, layer_idx
                    );
                }

                // Get KV cache buffers directly from kv_cache object
                let k_buf = ctx.kv_cache.k_buffer(*layer_idx);
                let v_buf = ctx.kv_cache.v_buffer(*layer_idx);
                let output_buf = ctx.forward_pool.get(*n_heads * *head_dim * 4);

                // Get actual sequence length from kv_cache
                let seq_len = ctx.kv_cache.seq_len() + 1; // +1 for current token
                let kv_stride = ctx.kv_cache.max_seq();

                // Debug: trace FlashAttention inputs for layers 30-31
                if std::env::var("INFER_TRACE_NAN").is_ok() && *layer_idx >= 30
                {
                    let q_ptr = q_buf.contents() as *const f32;
                    let q_len = q_buf.length() as usize / 4;
                    let q_has_nan = (0..std::cmp::min(10, q_len))
                        .any(|i| unsafe { (*q_ptr.add(i)).is_nan() });

                    let k_ptr = k_buf.contents() as *const f32;
                    let k_len = k_buf.length() as usize / 4;
                    let k_has_nan = (0..std::cmp::min(10, k_len))
                        .any(|i| unsafe { (*k_ptr.add(i)).is_nan() });

                    let v_ptr = v_buf.contents() as *const f32;
                    let v_len = v_buf.length() as usize / 4;
                    let v_has_nan = (0..std::cmp::min(10, v_len))
                        .any(|i| unsafe { (*v_ptr.add(i)).is_nan() });

                    let q_first: Vec<f32> =
                        (0..3).map(|i| unsafe { *q_ptr.add(i) }).collect();
                    let k_first: Vec<f32> =
                        (0..3).map(|i| unsafe { *k_ptr.add(i) }).collect();
                    let v_first: Vec<f32> =
                        (0..3).map(|i| unsafe { *v_ptr.add(i) }).collect();

                    eprintln!(
                        "[FA] Layer {}: Q(has_nan={}, {:?}), K(has_nan={}, {:?}), V(has_nan={}, {:?}), seq_len={}",
                        layer_idx,
                        q_has_nan,
                        q_first,
                        k_has_nan,
                        k_first,
                        v_has_nan,
                        v_first,
                        seq_len
                    );
                }

                // Use multi-head kernel if enabled (INFER_USE_MULTIHEAD=1)
                // This reduces dispatch overhead by 32x for models with n_heads >= 32
                // Note: For n_heads < 32, single-head approach is more efficient
                let use_multihead = std::env::var("INFER_USE_MULTIHEAD")
                    .is_ok()
                    && *head_dim == 128
                    && *n_heads >= 32;

                if use_multihead {
                    ctx.gpu_ctx.flash_attn_multihead_encode(
                        encoder,
                        1, // n_seqs
                        *n_heads,
                        *n_kv_heads,
                        *head_dim,
                        seq_len,
                        kv_stride,
                        &q_buf,
                        k_buf,
                        v_buf,
                        &output_buf,
                    );
                } else if ctx.use_mma && *head_dim == 128 {
                    // Use batched MMA kernel for better performance
                    ctx.gpu_ctx.flash_attn_mma_batched_encode(
                        encoder,
                        *n_heads,
                        *n_kv_heads,
                        seq_len,
                        kv_stride,
                        &q_buf,
                        k_buf,
                        v_buf,
                        &output_buf,
                    );
                } else {
                    ctx.gpu_ctx.flash_attn_f16_encode(
                        encoder,
                        1, // n_seqs
                        *n_heads,
                        *n_kv_heads,
                        *head_dim,
                        seq_len,
                        kv_stride,
                        &q_buf,
                        k_buf,
                        v_buf,
                        &output_buf,
                    );
                }

                ctx.tensor_registry.insert(*output, output_buf);
                Some(())
            }

            Operation::KVCacheUpdate { input, layer_idx, is_key } => {
                let input_buf = ctx.tensor_registry.get(*input)?;

                // Get model config
                let n_kv_heads = ctx
                    .model
                    .layers
                    .get(*layer_idx)
                    .and_then(|l| l.wk.as_ref())
                    .map(|w| w.shape[1] / ctx.model.config.head_dim)
                    .unwrap_or(ctx.model.config.n_heads);
                let head_dim = ctx.model.config.head_dim;
                let pos = ctx.kv_cache.seq_len();
                let max_seq = ctx.kv_cache.max_seq();

                // Debug: trace KV cache operations for layers 30-31
                if std::env::var("INFER_TRACE_NAN").is_ok() && *layer_idx >= 30
                {
                    let ptr = input_buf.contents() as *const f32;
                    let len = input_buf.length() as usize / 4;
                    let first_vals: Vec<f32> = (0..std::cmp::min(5, len))
                        .map(|i| unsafe { *ptr.add(i) })
                        .collect();
                    let has_nan = (0..std::cmp::min(10, len))
                        .any(|i| unsafe { (*ptr.add(i)).is_nan() });

                    // Check buffer addresses for overlap detection
                    let cache_buf = if *is_key {
                        ctx.kv_cache.k_buffer(*layer_idx)
                    } else {
                        ctx.kv_cache.v_buffer(*layer_idx)
                    };
                    let layer_30_buf = if *is_key {
                        ctx.kv_cache.k_buffer(30)
                    } else {
                        ctx.kv_cache.v_buffer(30)
                    };
                    let layer_31_buf = if *is_key {
                        ctx.kv_cache.k_buffer(31)
                    } else {
                        ctx.kv_cache.v_buffer(31)
                    };

                    eprintln!(
                        "[KV] Layer {} {}: input tensor {}, has_nan={}, first_vals={:?}, pos={}, n_kv_heads={}, head_dim={}, max_seq={}, buf_ptr={:p}, L30_buf={:p}, L31_buf={:p}",
                        layer_idx,
                        if *is_key { "K" } else { "V" },
                        input,
                        has_nan,
                        first_vals,
                        pos,
                        n_kv_heads,
                        head_dim,
                        max_seq,
                        cache_buf.contents(),
                        layer_30_buf.contents(),
                        layer_31_buf.contents()
                    );
                }

                if *is_key {
                    // Update K cache
                    let k_cache_buf = ctx.kv_cache.k_buffer(*layer_idx);
                    ctx.gpu_ctx.kv_update_k_encode(
                        encoder,
                        &input_buf,
                        k_cache_buf,
                        pos,
                        max_seq,
                        n_kv_heads,
                        head_dim,
                    );
                } else {
                    // Update V cache
                    let v_cache_buf = ctx.kv_cache.v_buffer(*layer_idx);
                    ctx.gpu_ctx.kv_update_v_encode(
                        encoder,
                        &input_buf,
                        v_cache_buf,
                        pos,
                        max_seq,
                        n_kv_heads,
                        head_dim,
                    );
                }

                Some(())
            }

            Operation::SiLU { input, output } => {
                let input_buf = ctx.tensor_registry.get(*input)?;
                let output_buf =
                    ctx.forward_pool.get(ctx.model.config.hidden_size * 4);

                ctx.gpu_ctx.silu_encode(
                    encoder,
                    &input_buf,
                    &output_buf,
                    ctx.model.config.hidden_size,
                );

                ctx.tensor_registry.insert(*output, output_buf);
                Some(())
            }

            Operation::Mul { lhs, rhs, output } => {
                let lhs_buf = ctx.tensor_registry.get(*lhs)?;
                let rhs_buf = ctx.tensor_registry.get(*rhs)?;
                let output_buf =
                    ctx.forward_pool.get(ctx.model.config.hidden_size * 4);

                ctx.gpu_ctx.mul_encode(
                    encoder,
                    &lhs_buf,
                    &rhs_buf,
                    &output_buf,
                    ctx.model.config.hidden_size,
                );

                ctx.tensor_registry.insert(*output, output_buf);
                Some(())
            }

            Operation::Add { lhs, rhs, output } => {
                let lhs_buf = ctx.tensor_registry.get(*lhs)?;
                let rhs_buf = ctx.tensor_registry.get(*rhs)?;
                let output_buf =
                    ctx.forward_pool.get(ctx.model.config.hidden_size * 4);

                if std::env::var("INFER_TRACE_NAN").is_ok()
                    && *output >= 725
                    && *output <= 745
                {
                    eprintln!(
                        "[TRACE] Add node {}: lhs={}, rhs={}, output={}",
                        node.id, lhs, rhs, output
                    );
                    let lhs_ptr = lhs_buf.contents() as *const f32;
                    let lhs_len = (lhs_buf.length() / 4) as usize;
                    let lhs_vals: Vec<f32> = unsafe {
                        std::slice::from_raw_parts(
                            lhs_ptr,
                            std::cmp::min(5, lhs_len),
                        )
                        .to_vec()
                    };
                    let rhs_ptr = rhs_buf.contents() as *const f32;
                    let rhs_len = (rhs_buf.length() / 4) as usize;
                    let rhs_vals: Vec<f32> = unsafe {
                        std::slice::from_raw_parts(
                            rhs_ptr,
                            std::cmp::min(5, rhs_len),
                        )
                        .to_vec()
                    };
                    eprintln!("  lhs: {:?}, rhs: {:?}", lhs_vals, rhs_vals);
                }

                if std::env::var("INFER_TRACE_DAG").is_ok() {
                    eprintln!(
                        "[TRACE] Add: lhs={}, rhs={}, output={}",
                        lhs, rhs, output
                    );
                }

                ctx.gpu_ctx.add_encode(
                    encoder,
                    &lhs_buf,
                    &rhs_buf,
                    &output_buf,
                    ctx.model.config.hidden_size,
                );

                ctx.tensor_registry.insert(*output, output_buf);
                Some(())
            }

            // Fused operations
            Operation::FusedSwiGLU { gate, up, output, layer_idx: _ } => {
                let gate_buf = ctx.tensor_registry.get(*gate)?;
                let up_buf = ctx.tensor_registry.get(*up)?;

                // Get intermediate_size from gate_buf length
                let intermediate_size = (gate_buf.length() / 4) as usize;
                let output_buf = ctx.forward_pool.get(intermediate_size * 4);

                if std::env::var("INFER_TRACE_DAG").is_ok() {
                    eprintln!(
                        "[TRACE] FusedSwiGLU: gate={}, up={}, output={}, intermediate_size={}",
                        gate, up, output, intermediate_size
                    );
                }

                ctx.gpu_ctx.fused_swiglu_encode(
                    encoder,
                    &gate_buf,
                    &up_buf,
                    &output_buf,
                    intermediate_size,
                );

                ctx.tensor_registry.insert(*output, output_buf);
                Some(())
            }

            Operation::FusedNormLinear {
                input,
                norm_weight,
                linear_weight,
                output,
                out_size,
                in_size: _,
                epsilon,
                layer_idx: _,
            } => {
                let input_buf = ctx.tensor_registry.get(*input)?;
                let norm_weight_buf = ctx.tensor_registry.get(*norm_weight)?;
                let linear_weight_buf =
                    ctx.tensor_registry.get(*linear_weight)?;
                let output_buf = ctx.forward_pool.get(*out_size * 4);

                // Get weight dtype from cached registry
                let dtype = ctx.tensor_registry.get_dtype(*linear_weight)?;

                // Use F16 fused kernel for F16 weights, fallback otherwise
                match dtype {
                    crate::model::tensor::DataType::F16 => {
                        ctx.gpu_ctx.fused_norm_linear_f16_encode(
                            encoder,
                            &input_buf,
                            &norm_weight_buf,
                            &linear_weight_buf,
                            &output_buf,
                            *out_size,
                            ctx.model.config.hidden_size,
                            *epsilon,
                        );
                    }
                    crate::model::tensor::DataType::Q4_K => {
                        // Q4_K fused kernel - each block is 144 bytes for 256 elements
                        let nb = ctx.model.config.hidden_size / 256;
                        let nb01 = (nb * 144) as u64;
                        ctx.gpu_ctx.fused_norm_linear_q4_k_encode(
                            encoder,
                            &input_buf,
                            &norm_weight_buf,
                            &linear_weight_buf,
                            &output_buf,
                            *out_size,
                            ctx.model.config.hidden_size,
                            *epsilon,
                            nb01,
                        );
                    }
                    crate::model::tensor::DataType::Q5_K => {
                        // Q5_K fused kernel - each block is 176 bytes for 256 elements
                        let nb = ctx.model.config.hidden_size / 256;
                        let nb01 = (nb * 176) as u64;
                        ctx.gpu_ctx.fused_norm_linear_q5_k_encode(
                            encoder,
                            &input_buf,
                            &norm_weight_buf,
                            &linear_weight_buf,
                            &output_buf,
                            *out_size,
                            ctx.model.config.hidden_size,
                            *epsilon,
                            nb01,
                        );
                    }
                    crate::model::tensor::DataType::Q6_K => {
                        // Q6_K fused kernel - each block is 210 bytes for 256 elements
                        let nb = ctx.model.config.hidden_size / 256;
                        let nb01 = (nb * 210) as u64;
                        ctx.gpu_ctx.fused_norm_linear_q6_k_encode(
                            encoder,
                            &input_buf,
                            &norm_weight_buf,
                            &linear_weight_buf,
                            &output_buf,
                            *out_size,
                            ctx.model.config.hidden_size,
                            *epsilon,
                            nb01,
                        );
                    }
                    _ => {
                        // Fallback: separate RMSNorm + GEMV for other dtypes
                        // First, do RMSNorm
                        let norm_out_buf = ctx
                            .forward_pool
                            .get(ctx.model.config.hidden_size * 4);
                        ctx.gpu_ctx.rms_norm_encode(
                            encoder,
                            &input_buf,
                            &norm_weight_buf,
                            &norm_out_buf,
                            1,
                            ctx.model.config.hidden_size,
                            *epsilon,
                        );

                        // Then, do GEMV
                        ctx.gpu_ctx.mv_encode(
                            encoder,
                            &linear_weight_buf,
                            &norm_out_buf,
                            &output_buf,
                            *out_size,
                            ctx.model.config.hidden_size,
                            dtype,
                        );
                    }
                }

                ctx.tensor_registry.insert(*output, output_buf);
                Some(())
            }

            Operation::FusedAddNorm {
                lhs,
                rhs,
                norm_weight,
                output,
                epsilon,
                layer_idx: _,
            } => {
                let lhs_buf = ctx.tensor_registry.get(*lhs)?;
                let rhs_buf = ctx.tensor_registry.get(*rhs)?;
                let norm_weight_buf = ctx.tensor_registry.get(*norm_weight)?;

                let hidden_size = ctx.model.config.hidden_size;
                let output_buf = ctx.forward_pool.get(hidden_size * 4);

                // Use fused kernel
                ctx.gpu_ctx.fused_add_norm_encode(
                    encoder,
                    &lhs_buf,
                    &rhs_buf,
                    &norm_weight_buf,
                    &output_buf,
                    hidden_size,
                    *epsilon,
                );

                ctx.tensor_registry.insert(*output, output_buf);
                Some(())
            }

            Operation::FusedGemvAdd {
                input,
                weight,
                residual,
                output,
                out_size,
                in_size,
                layer_idx: _,
            } => {
                let input_buf = ctx.tensor_registry.get(*input)?;
                let weight_buf = ctx.tensor_registry.get(*weight)?;
                let residual_buf = ctx.tensor_registry.get(*residual)?;

                // Output buffer - GEMV result + residual add
                let output_buf = ctx.forward_pool.get(*out_size * 4);

                // Get weight dtype from cached registry
                let dtype = ctx.tensor_registry.get_dtype(*weight)?;

                // Use fused kernel if available (K-quant formats)
                let fused_success = ctx.gpu_ctx.fused_gemv_add_encode(
                    encoder,
                    &weight_buf,
                    &input_buf,
                    &residual_buf,
                    &output_buf,
                    *out_size,
                    *in_size,
                    dtype,
                );

                if !fused_success {
                    // Fallback to separate operations for unsupported formats
                    // Get weight tensor from bindings for fallback path
                    let weight_tensor = ctx.weight_bindings.get(weight)?;
                    ctx.model.gemv_encode(
                        encoder,
                        &input_buf,
                        *in_size,
                        weight_tensor,
                        *out_size,
                        ctx.gpu_ctx,
                        &output_buf,
                    )?;

                    ctx.gpu_ctx.add_encode(
                        encoder,
                        &residual_buf,
                        &output_buf,
                        &output_buf,
                        *out_size,
                    );
                }

                ctx.tensor_registry.insert(*output, output_buf);
                #[cfg(debug_assertions)]
                eprintln!(
                    "[FusedGemvAdd] input={}, weight={}, residual={}, output={}, out_size={}, fused={}",
                    input, weight, residual, output, out_size, fused_success
                );
                Some(())
            }

            Operation::ExpertRouting { .. } => {
                // MoE - not implemented yet
                None
            }

            Operation::ExpertCompute { .. } => {
                // MoE - not implemented yet
                None
            }

            Operation::NoOp => {
                // Placeholder for fused/removed operations - skip
                Some(())
            }
        }
    }

    /// Get layer index from DAG node (if it belongs to a specific layer)
    fn get_node_layer_idx(node: &DagNode) -> Option<usize> {
        // Use the node's layer_idx field
        node.layer_idx
    }

    /// Get output tensor IDs from a DAG node operation
    fn get_node_outputs(op: &Operation) -> Vec<usize> {
        match op {
            Operation::EmbeddingLookup { output, .. } => vec![*output],
            Operation::RmsNorm { output, .. } => vec![*output],
            Operation::Gemv { output, .. } => vec![*output],
            Operation::FusedQKV { output_q, output_k, output_v, .. } => {
                vec![*output_q, *output_k, *output_v]
            }
            Operation::FusedQKVWeight {
                output_q, output_k, output_v, ..
            } => {
                vec![*output_q, *output_k, *output_v]
            }
            Operation::FusedGateUp { output_gate, output_up, .. } => {
                vec![*output_gate, *output_up]
            }
            Operation::RoPE { output, .. } => vec![*output],
            Operation::FlashAttention { output, .. } => vec![*output],
            Operation::KVCacheUpdate { .. } => vec![],
            Operation::SiLU { output, .. } => vec![*output],
            Operation::Mul { output, .. } => vec![*output],
            Operation::Add { output, .. } => vec![*output],
            Operation::FusedSwiGLU { output, .. } => vec![*output],
            Operation::FusedNormLinear { output, .. } => vec![*output],
            Operation::FusedAddNorm { output, .. } => vec![*output],
            Operation::FusedGemvAdd { output, .. } => vec![*output],
            Operation::ExpertRouting {
                selected_experts,
                routing_weights,
                ..
            } => {
                vec![*selected_experts, *routing_weights]
            }
            Operation::ExpertCompute { output, .. } => vec![*output],
            Operation::NoOp => vec![],
        }
    }

    /// Get input tensor IDs from a DAG node operation
    fn get_node_inputs(op: &Operation) -> Vec<usize> {
        match op {
            Operation::EmbeddingLookup { .. } => vec![],
            Operation::RmsNorm { input, weight, .. } => vec![*input, *weight],
            Operation::Gemv { input, weight, .. } => vec![*input, *weight],
            Operation::FusedQKV {
                input, q_weight, k_weight, v_weight, ..
            } => {
                vec![*input, *q_weight, *k_weight, *v_weight]
            }
            Operation::FusedQKVWeight { input, qkv_weight, .. } => {
                vec![*input, *qkv_weight]
            }
            Operation::FusedGateUp {
                input, gate_weight, up_weight, ..
            } => {
                vec![*input, *gate_weight, *up_weight]
            }
            Operation::RoPE { input, .. } => vec![*input],
            Operation::FlashAttention { q, .. } => vec![*q],
            Operation::KVCacheUpdate { input, .. } => vec![*input],
            Operation::SiLU { input, .. } => vec![*input],
            Operation::Mul { lhs, rhs, .. } => vec![*lhs, *rhs],
            Operation::Add { lhs, rhs, .. } => vec![*lhs, *rhs],
            Operation::FusedSwiGLU { gate, up, .. } => vec![*gate, *up],
            Operation::FusedNormLinear {
                input,
                norm_weight,
                linear_weight,
                ..
            } => {
                vec![*input, *norm_weight, *linear_weight]
            }
            Operation::FusedAddNorm { lhs, rhs, norm_weight, .. } => {
                vec![*lhs, *rhs, *norm_weight]
            }
            Operation::FusedGemvAdd { input, weight, residual, .. } => {
                vec![*input, *weight, *residual]
            }
            Operation::ExpertRouting { .. } => vec![],
            Operation::ExpertCompute { .. } => vec![],
            Operation::NoOp => vec![],
        }
    }
}
