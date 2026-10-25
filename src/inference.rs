// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

// Inference runtime: connect model layer with operators

pub mod dequant;
#[cfg(all(target_arch = "aarch64", feature = "metal"))]
pub mod gpu_tensor;
pub mod kv_cache;
#[cfg(all(target_arch = "aarch64", feature = "metal"))]
pub mod kv_cache_gpu;

#[cfg(all(target_arch = "aarch64", feature = "metal"))]
pub use gpu_tensor::GpuTensor;
pub use kv_cache::{KVCache, KVCacheBase, KVCacheFormat};
#[cfg(all(target_arch = "aarch64", feature = "metal"))]
pub use kv_cache_gpu::KVCacheGPU;

use crate::model::config::ModelConfig;
use crate::model::llama::LlamaModel;
use crate::model::tensor::DataType;
use crate::model::weights::WeightStore;
use crate::ops::sampling::{SamplingStrategy, sample};
use crate::quant::kv_cache::blocks::QK;
use crate::quant::types::{
    BlockIQ4NL, BlockQ4_0, BlockQ4_1, BlockQ5_0, BlockQ5_1, BlockQ8_0,
};
use half::f16;

/// Inference context
pub struct InferenceContext {
    pub config: ModelConfig,
    pub model: LlamaModel,
    pub device: Device,
    #[cfg(all(target_arch = "aarch64", feature = "metal"))]
    pub gpu_ctx: Option<crate::quant::vec_dot::gpu::metal::MetalContext>,
    #[cfg(all(target_arch = "aarch64", feature = "metal"))]
    pub gpu_kv_cache: Option<KVCacheGPU>,
    /// Multiple pending command buffers for GPU parallelism
    /// n_cb = number of concurrent CBs = bandwidth multiplier
    #[cfg(all(target_arch = "aarch64", feature = "metal"))]
    pending_cbs: Vec<crate::dag::executor::AsyncCommandBuffer>,
    /// Cached logits from completed CBs
    #[cfg(all(target_arch = "aarch64", feature = "metal"))]
    cached_logits: Option<Vec<f32>>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Device {
    Cpu,
    Gpu,
}

impl InferenceContext {
    pub fn from_gguf(path: &str, device: Device) -> Result<Self, String> {
        let store = WeightStore::from_gguf(path).map_err(|e| e.to_string())?;
        let gguf =
            crate::model::GgufFile::open(path).map_err(|e| e.to_string())?;
        let config = ModelConfig::from_gguf(&gguf);
        let model = LlamaModel::from_weights(&store, config);

        // Use model's config which may have been updated from actual weight shapes
        let config = model.config.clone();

        #[cfg(all(target_arch = "aarch64", feature = "metal"))]
        let (gpu_ctx, gpu_kv_cache) = if device == Device::Gpu {
            match crate::quant::vec_dot::gpu::metal::MetalContext::new() {
                Ok(ctx) => {
                    let kv = KVCacheGPU::new(ctx.device(), &config, 2048);
                    (Some(ctx), Some(kv))
                }
                Err(e) => {
                    eprintln!("MetalContext::new() failed: {}", e);
                    (None, None)
                }
            }
        } else {
            (None, None)
        };

        #[cfg(not(all(target_arch = "aarch64", feature = "metal")))]
        let _ = device; // Silence unused warning

        Ok(InferenceContext {
            config,
            model,
            device,
            #[cfg(all(target_arch = "aarch64", feature = "metal"))]
            gpu_ctx,
            #[cfg(all(target_arch = "aarch64", feature = "metal"))]
            gpu_kv_cache,
            #[cfg(all(target_arch = "aarch64", feature = "metal"))]
            pending_cbs: Vec::new(),
            #[cfg(all(target_arch = "aarch64", feature = "metal"))]
            cached_logits: None,
        })
    }

    /// Load model with a shared MetalContext (avoids GPU state corruption)
    ///
    /// IMPORTANT: Always use this when creating multiple InferenceContexts.
    /// Creating multiple MetalContexts on the same Metal device causes corruption.
    #[cfg(all(target_arch = "aarch64", feature = "metal"))]
    pub fn from_gguf_with_gpu_ctx(
        path: &str,
        gpu_ctx: crate::quant::vec_dot::gpu::metal::MetalContext,
    ) -> Result<Self, String> {
        let store = WeightStore::from_gguf(path).map_err(|e| e.to_string())?;
        let gguf =
            crate::model::GgufFile::open(path).map_err(|e| e.to_string())?;
        let config = ModelConfig::from_gguf(&gguf);
        let model = LlamaModel::from_weights(&store, config);

        // Use model's config which may have been updated from actual weight shapes
        let config = model.config.clone();

        let kv = KVCacheGPU::new(gpu_ctx.device(), &config, 2048);

        Ok(InferenceContext {
            config,
            model,
            device: Device::Gpu,
            gpu_ctx: Some(gpu_ctx),
            gpu_kv_cache: Some(kv),
            pending_cbs: Vec::new(),
            cached_logits: None,
        })
    }

    /// Get layer count
    pub fn layer_count(&self) -> usize {
        self.model.layers.len()
    }

    /// Get parameter count
    pub fn param_count(&self) -> usize {
        self.model.param_count()
    }

    /// Check if model is loaded
    pub fn is_loaded(&self) -> bool {
        self.model.tok_embeddings.is_some() && !self.model.layers.is_empty()
    }

    /// Run forward pass with appropriate device
    pub fn forward(
        &mut self,
        tokens: &[u32],
        kv_cache: &mut KVCache,
    ) -> Option<Vec<f32>> {
        match self.device {
            Device::Gpu => {
                #[cfg(all(target_arch = "aarch64", feature = "metal"))]
                {
                    if let Some(ref gpu_ctx) = self.gpu_ctx {
                        self.model.forward_gpu(tokens, kv_cache, gpu_ctx)
                    } else {
                        self.model.forward(tokens, kv_cache)
                    }
                }
                #[cfg(not(all(target_arch = "aarch64", feature = "metal")))]
                {
                    self.model.forward(tokens, kv_cache)
                }
            }
            Device::Cpu => self.model.forward(tokens, kv_cache),
        }
    }

    /// Run forward pass with GPU KV cache (zero-copy Flash Attention)
    #[cfg(all(target_arch = "aarch64", feature = "metal"))]
    pub fn forward_gpu_zerocopy(&mut self, tokens: &[u32]) -> Option<Vec<f32>> {
        if let (Some(ref gpu_ctx), Some(ref mut kv_cache)) =
            (self.gpu_ctx.as_ref(), self.gpu_kv_cache.as_mut())
        {
            self.model.forward_gpu_zerocopy(tokens, kv_cache, gpu_ctx)
        } else {
            self.model.forward(
                tokens,
                &mut KVCache::new(&self.config, 2048, KVCacheFormat::F32),
            )
        }
    }

    /// Fully GPU-native forward pass with async pipeline (CPU-GPU overlap)
    /// For decode, uses forward_gpu_batched to overlap CPU encoding with GPU execution
    #[cfg(all(target_arch = "aarch64", feature = "metal"))]
    pub fn forward_gpu_native(&mut self, tokens: &[u32]) -> Option<Vec<f32>> {
        // Fall back to CPU for MQA/MLA architecture (not yet supported on GPU)
        if self.model.uses_mqa() {
            eprintln!("[DEBUG] Model uses MQA/MLA, falling back to CPU");
            return self.model.forward(
                tokens,
                &mut KVCache::new(&self.config, 2048, KVCacheFormat::F32),
            );
        }

        // For decode (single token), use async pipeline
        if tokens.len() == 1 {
            return self.forward_gpu_batched(tokens);
        }

        // For prefill, use synchronous batch implementation
        if let (Some(ref gpu_ctx), Some(ref mut kv_cache)) =
            (self.gpu_ctx.as_ref(), self.gpu_kv_cache.as_mut())
        {
            self.model.forward_gpu_native_batched(tokens, kv_cache, gpu_ctx)
        } else {
            self.model.forward(
                tokens,
                &mut KVCache::new(&self.config, 2048, KVCacheFormat::F32),
            )
        }
    }

    /// Run forward pass with batched GPU encoding (async pipeline)
    ///
    /// Implements llama.cpp style CPU-GPU pipeline:
    /// 1. Wait for previous CB to complete and get its logits
    /// 2. Submit new CB for current tokens (async)
    /// 3. Return previous tokens' logits
    ///
    /// For first call, waits synchronously (no overlap).
    /// For subsequent calls, overlaps CPU encoding with GPU execution.
    #[cfg(all(target_arch = "aarch64", feature = "metal"))]
    pub fn forward_gpu_batched(&mut self, tokens: &[u32]) -> Option<Vec<f32>> {
        // Fall back to CPU for MQA/MLA architecture (not yet supported on GPU)
        if self.model.uses_mqa() {
            return self.model.forward(
                tokens,
                &mut KVCache::new(&self.config, 2048, KVCacheFormat::F32),
            );
        }

        // Step 1: Wait for oldest CB in pipeline and get its logits
        // This implements multi-pass pipelining: N CBs can execute in parallel on GPU
        // Default pipeline depth = 2 (balance between parallelism and latency)
        let pipeline_depth = std::env::var("INFER_PIPELINE_DEPTH")
            .map(|v| v.parse().unwrap_or(2))
            .unwrap_or(2);

        if self.pending_cbs.len() >= pipeline_depth {
            // Pipeline full: wait for oldest CB
            if let Some(mut oldest_cb) = self.pending_cbs.pop() {
                oldest_cb.wait();
                let logits = oldest_cb.get_logits();
                self.cached_logits = Some(logits);
            }
        }

        // Step 2: Submit new CB asynchronously
        // GPU starts encoding immediately, CPU continues
        // Multiple CBs can execute in parallel on GPU
        if let (Some(ref gpu_ctx), Some(ref mut kv_cache)) =
            (self.gpu_ctx.as_ref(), self.gpu_kv_cache.as_mut())
        {
            use crate::dag::executor::DagExecutor;

            let result = DagExecutor::forward_dag_async(
                &self.model,
                tokens,
                kv_cache,
                gpu_ctx,
                None, // No dependency tracking needed for sequential decode
            );

            // Add new CB to pipeline (GPU can execute multiple CBs in parallel)
            if let Some(cb) = result.pending_cb {
                self.pending_cbs.insert(0, cb); // Insert at front (newest)
            }

            // Return cached logits from completed CB
            // For first calls, cached_logits is None (pipeline filling)
            self.cached_logits.take()
        } else {
            // Fallback to CPU
            self.model.forward(
                tokens,
                &mut KVCache::new(&self.config, 2048, KVCacheFormat::F32),
            )
        }
    }

    /// Batch prefill using GPU GEMM (more efficient for multiple tokens)
    #[cfg(all(target_arch = "aarch64", feature = "metal"))]
    pub fn forward_prefill_batch(
        &mut self,
        tokens: &[u32],
    ) -> Option<Vec<f32>> {
        if let (Some(ref gpu_ctx), Some(ref mut kv_cache)) =
            (self.gpu_ctx.as_ref(), self.gpu_kv_cache.as_mut())
        {
            self.model.forward_prefill_batch(tokens, kv_cache, gpu_ctx)
        } else {
            self.model.forward(
                tokens,
                &mut KVCache::new(&self.config, 2048, KVCacheFormat::F32),
            )
        }
    }

    /// Generate tokens from prompt
    pub fn generate(
        &mut self,
        prompt_tokens: &[u32],
        max_tokens: usize,
        strategy: &SamplingStrategy,
    ) -> Vec<u32> {
        let mut kv_cache = KVCache::new(
            &self.config,
            prompt_tokens.len() + max_tokens,
            KVCacheFormat::F32,
        );
        let mut generated = Vec::with_capacity(max_tokens);

        // Prefill: process prompt
        if let Some(logits) = self.forward(prompt_tokens, &mut kv_cache) {
            let first_token = sample(&logits, strategy) as u32;
            generated.push(first_token);
        }

        // Decode: generate remaining tokens
        for _ in 1..max_tokens {
            let eos = self.config.eos_token_id;
            if generated.last() == Some(&eos) {
                break;
            }

            let last_token =
                generated.last().copied().unwrap_or(self.config.bos_token_id);

            if let Some(logits) = self.forward(&[last_token], &mut kv_cache) {
                let next_token = sample(&logits, strategy) as u32;
                generated.push(next_token);
            } else {
                break;
            }
        }

        generated
    }

    /// Get reference to model (for debugging)
    pub fn model(&self) -> &LlamaModel {
        &self.model
    }

    /// Get weight store (for debugging)
    pub fn weight_store(&self) -> &WeightStore {
        // We need to store this - for now just return an error
        unimplemented!("Need to store weight store in InferenceContext")
    }
}

/// Dequantize quantized/FP data to F32
/// Handles all supported data types including K-quant and standard quantization
pub fn dequant_to_f32(data: &[u8], dtype: DataType, count: usize) -> Vec<f32> {
    let mut result = vec![0.0f32; count];

    match dtype {
        DataType::F32 => {
            for i in 0..count {
                let offset = i * 4;
                if offset + 4 <= data.len() {
                    let bytes: [u8; 4] =
                        data[offset..offset + 4].try_into().unwrap();
                    result[i] = f32::from_le_bytes(bytes);
                }
            }
        }
        DataType::F16 => {
            for i in 0..count {
                let offset = i * 2;
                if offset + 2 <= data.len() {
                    let bytes: [u8; 2] =
                        data[offset..offset + 2].try_into().unwrap();
                    result[i] = f16::from_le_bytes(bytes).to_f32();
                }
            }
        }
        DataType::BF16 => {
            for i in 0..count {
                let offset = i * 2;
                if offset + 2 <= data.len() {
                    let bytes: [u8; 2] =
                        data[offset..offset + 2].try_into().unwrap();
                    // BF16 to F32: shift left by 16 bits
                    let bits = u16::from_le_bytes(bytes) as u32;
                    result[i] = f32::from_bits(bits << 16);
                }
            }
        }
        DataType::Q4_0 => {
            dequant_q4_0(data, &mut result, count);
        }
        DataType::Q4_1 => {
            dequant_q4_1(data, &mut result, count);
        }
        DataType::Q5_0 => {
            dequant_q5_0(data, &mut result, count);
        }
        DataType::Q5_1 => {
            dequant_q5_1(data, &mut result, count);
        }
        DataType::Q8_0 => {
            dequant_q8_0(data, &mut result, count);
        }
        DataType::Q2_K => {
            dequant::dequant_q2_k(data, &mut result, count);
        }
        DataType::Q3_K => {
            dequant::dequant_q3_k(data, &mut result, count);
        }
        DataType::Q4_K => {
            dequant::dequant_q4_k(data, &mut result, count);
        }
        DataType::Q5_K => {
            dequant::dequant_q5_k(data, &mut result, count);
        }
        DataType::Q6_K => {
            dequant::dequant_q6_k(data, &mut result, count);
        }
        DataType::IQ4_NL => {
            dequant_iq4_nl(data, &mut result, count);
        }
        DataType::IQ4_XS => {
            dequant_iq4_xs(data, &mut result, count);
        }
        // IQ formats - placeholder implementation
        // These formats require complex dequantization logic
        // For now, return zeros to prevent NaN propagation
        DataType::IQ3_XXS
        | DataType::IQ3_S
        | DataType::IQ2_XXS
        | DataType::IQ2_XS
        | DataType::IQ2_S
        | DataType::IQ1_S => {
            // Placeholder: return zeros for unsupported formats
            // TODO: Implement proper dequantization for these formats
            log::warn!(
                "Dequantization for {:?} not implemented, returning zeros",
                dtype
            );
            // Result is already initialized to zeros
        }
    }

    result
}

// Standard Q formats (32 elements per block)

fn dequant_q4_0(data: &[u8], out: &mut [f32], count: usize) {
    let n_blocks = count.div_ceil(QK);
    for i in 0..n_blocks {
        let block_offset = i * std::mem::size_of::<BlockQ4_0>();
        if block_offset + std::mem::size_of::<BlockQ4_0>() > data.len() {
            break;
        }
        let block: &BlockQ4_0 =
            unsafe { &*(data.as_ptr().add(block_offset) as *const BlockQ4_0) };
        let d = f16::from_bits(block.d).to_f32();

        for j in 0..QK / 2 {
            let q = block.qs[j];
            let q0 = ((q & 0x0F) as i32) - 8;
            let q1 = (((q >> 4) & 0x0F) as i32) - 8;

            let idx0 = i * QK + j;
            let idx1 = i * QK + j + QK / 2;
            if idx0 < count {
                out[idx0] = q0 as f32 * d;
            }
            if idx1 < count {
                out[idx1] = q1 as f32 * d;
            }
        }
    }
}

fn dequant_q4_1(data: &[u8], out: &mut [f32], count: usize) {
    let n_blocks = count.div_ceil(QK);
    for i in 0..n_blocks {
        let block_offset = i * std::mem::size_of::<BlockQ4_1>();
        if block_offset + std::mem::size_of::<BlockQ4_1>() > data.len() {
            break;
        }
        let block: &BlockQ4_1 =
            unsafe { &*(data.as_ptr().add(block_offset) as *const BlockQ4_1) };
        let d = f16::from_bits(block.d).to_f32();
        let m = f16::from_bits(block.m).to_f32();

        for j in 0..QK / 2 {
            let q = block.qs[j];
            let q0 = (q & 0x0F) as i32;
            let q1 = ((q >> 4) & 0x0F) as i32;

            let idx0 = i * QK + j;
            let idx1 = i * QK + j + QK / 2;
            if idx0 < count {
                out[idx0] = q0 as f32 * d + m;
            }
            if idx1 < count {
                out[idx1] = q1 as f32 * d + m;
            }
        }
    }
}

fn dequant_q5_0(data: &[u8], out: &mut [f32], count: usize) {
    let n_blocks = count.div_ceil(QK);
    for i in 0..n_blocks {
        let block_offset = i * std::mem::size_of::<BlockQ5_0>();
        if block_offset + std::mem::size_of::<BlockQ5_0>() > data.len() {
            break;
        }
        let block: &BlockQ5_0 =
            unsafe { &*(data.as_ptr().add(block_offset) as *const BlockQ5_0) };
        let d = f16::from_bits(block.d).to_f32();

        let qh = block.qh[0] as u32
            | ((block.qh[1] as u32) << 8)
            | ((block.qh[2] as u32) << 16)
            | ((block.qh[3] as u32) << 24);

        for j in 0..QK / 2 {
            let xh0 = ((qh >> j) & 1) << 4;
            let xh1 = ((qh >> (j + 16)) & 1) << 4;

            let q0 = ((block.qs[j] & 0x0F) as i32 | xh0 as i32) - 16;
            let q1 = (((block.qs[j] >> 4) & 0x0F) as i32 | xh1 as i32) - 16;

            let idx0 = i * QK + j;
            let idx1 = i * QK + j + QK / 2;
            if idx0 < count {
                out[idx0] = q0 as f32 * d;
            }
            if idx1 < count {
                out[idx1] = q1 as f32 * d;
            }
        }
    }
}

fn dequant_q5_1(data: &[u8], out: &mut [f32], count: usize) {
    let n_blocks = count.div_ceil(QK);
    for i in 0..n_blocks {
        let block_offset = i * std::mem::size_of::<BlockQ5_1>();
        if block_offset + std::mem::size_of::<BlockQ5_1>() > data.len() {
            break;
        }
        let block: &BlockQ5_1 =
            unsafe { &*(data.as_ptr().add(block_offset) as *const BlockQ5_1) };
        let d = f16::from_bits(block.d).to_f32();
        let m = f16::from_bits(block.m).to_f32();

        let qh = block.qh[0] as u32
            | ((block.qh[1] as u32) << 8)
            | ((block.qh[2] as u32) << 16)
            | ((block.qh[3] as u32) << 24);

        for j in 0..QK / 2 {
            let xh0 = ((qh >> j) & 1) << 4;
            let xh1 = ((qh >> (j + 16)) & 1) << 4;

            let q0 = (block.qs[j] & 0x0F) as i32 | xh0 as i32;
            let q1 = ((block.qs[j] >> 4) & 0x0F) as i32 | xh1 as i32;

            let idx0 = i * QK + j;
            let idx1 = i * QK + j + QK / 2;
            if idx0 < count {
                out[idx0] = q0 as f32 * d + m;
            }
            if idx1 < count {
                out[idx1] = q1 as f32 * d + m;
            }
        }
    }
}

fn dequant_q8_0(data: &[u8], out: &mut [f32], count: usize) {
    let n_blocks = count.div_ceil(QK);
    for i in 0..n_blocks {
        let block_offset = i * std::mem::size_of::<BlockQ8_0>();
        if block_offset + std::mem::size_of::<BlockQ8_0>() > data.len() {
            break;
        }
        let block: &BlockQ8_0 =
            unsafe { &*(data.as_ptr().add(block_offset) as *const BlockQ8_0) };
        let d = f16::from_bits(block.d).to_f32();

        for j in 0..QK {
            let idx = i * QK + j;
            if idx < count {
                out[idx] = block.qs[j] as f32 * d;
            }
        }
    }
}

fn dequant_iq4_nl(data: &[u8], out: &mut [f32], count: usize) {
    const KVALUES: [f32; 16] = [
        -127.0, -104.0, -83.0, -65.0, -49.0, -35.0, -22.0, -10.0, 1.0, 13.0,
        25.0, 38.0, 53.0, 69.0, 89.0, 113.0,
    ];

    let n_blocks = count.div_ceil(QK);
    for i in 0..n_blocks {
        let block_offset = i * std::mem::size_of::<BlockIQ4NL>();
        if block_offset + std::mem::size_of::<BlockIQ4NL>() > data.len() {
            break;
        }
        let block: &BlockIQ4NL =
            unsafe { &*(data.as_ptr().add(block_offset) as *const BlockIQ4NL) };
        let d = f16::from_bits(block.d).to_f32();

        for j in 0..QK / 2 {
            let q = block.qs[j];
            let idx0 = (q & 0x0F) as usize;
            let idx1 = ((q >> 4) & 0x0F) as usize;

            let out_idx0 = i * QK + j;
            let out_idx1 = i * QK + j + QK / 2;
            if out_idx0 < count {
                out[out_idx0] = KVALUES[idx0] * d;
            }
            if out_idx1 < count {
                out[out_idx1] = KVALUES[idx1] * d;
            }
        }
    }
}

fn dequant_iq4_xs(data: &[u8], out: &mut [f32], count: usize) {
    use crate::quant::types::BlockIQ4XS;
    const QK_K: usize = 256;

    const KVALUES: [f32; 16] = [
        -127.0, -104.0, -83.0, -65.0, -49.0, -35.0, -22.0, -10.0, 1.0, 13.0,
        25.0, 38.0, 53.0, 69.0, 89.0, 113.0,
    ];

    let n_blocks = count.div_ceil(QK_K);
    for block_idx in 0..n_blocks {
        let block_offset = block_idx * std::mem::size_of::<BlockIQ4XS>();
        if block_offset + std::mem::size_of::<BlockIQ4XS>() > data.len() {
            break;
        }
        let block: &BlockIQ4XS =
            unsafe { &*(data.as_ptr().add(block_offset) as *const BlockIQ4XS) };

        let d = f16::from_bits(block.d).to_f32();

        // Process 8 sub-blocks of 32 elements each
        for ib in 0..8 {
            // Extract scale for this sub-block (6-bit value from scales_l + scales_h)
            // ls = ((scales_l[ib/2] >> 4*(ib%2)) & 0xf) | (((scales_h >> 2*ib) & 3) << 4)
            let ls = ((block.scales_l[ib / 2] >> (4 * (ib % 2))) & 0xF) as i32
                | ((((block.scales_h >> (2 * ib)) & 3) << 4) as i32);
            let dl = d * (ls - 32) as f32;

            // Each sub-block has 32 elements stored in 16 bytes
            for j in 0..16 {
                let byte = block.qs[ib * 16 + j];
                let idx0 = (byte & 0x0F) as usize;
                let idx1 = ((byte >> 4) & 0x0F) as usize;

                let out_idx0 = block_idx * QK_K + ib * 32 + j;
                let out_idx1 = block_idx * QK_K + ib * 32 + j + 16;

                if out_idx0 < count {
                    out[out_idx0] = KVALUES[idx0] * dl;
                }
                if out_idx1 < count {
                    out[out_idx1] = KVALUES[idx1] * dl;
                }
            }
        }
    }
}
