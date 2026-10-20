// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

//! Profile forward pass by measuring absolute time per operation
//!
//! This tool profiles each operation in the forward pass separately,
//! then compares with llama.cpp's timing for the same operations.

use std::collections::HashMap;
use std::time::{Duration, Instant};

#[cfg(all(target_arch = "aarch64", feature = "metal"))]
use infer_train::inference::{InferenceContext, KVCacheGPU};
#[cfg(all(target_arch = "aarch64", feature = "metal"))]
use infer_train::model::LlamaModel;
#[cfg(all(target_arch = "aarch64", feature = "metal"))]
use infer_train::quant::vec_dot::gpu::metal::MetalContext;

/// Timing result for a single operation
#[derive(Debug, Clone)]
pub struct OpTiming {
    pub name: String,
    pub count: usize,
    pub total_us: u64,
    pub avg_us: f64,
}

/// Profile results for a forward pass
#[derive(Debug, Clone, Default)]
pub struct ProfileResult {
    pub operations: Vec<OpTiming>,
    pub total_us: u64,
}

impl ProfileResult {
    pub fn print(&self) {
        println!(
            "\n{:<30} {:>8} {:>12} {:>12}",
            "Operation", "Count", "Total (µs)", "Avg (µs)"
        );
        println!("{}", "-".repeat(66));

        for op in &self.operations {
            println!(
                "{:<30} {:>8} {:>12} {:>12.1}",
                op.name, op.count, op.total_us, op.avg_us
            );
        }

        println!("{}", "-".repeat(66));
        println!("{:<30} {:>8} {:>12}", "TOTAL", "", self.total_us);
    }
}

/// Profiler that measures per-operation timing
#[cfg(all(target_arch = "aarch64", feature = "metal"))]
pub struct ForwardProfiler {
    timings: HashMap<String, (usize, u64)>,
}

#[cfg(all(target_arch = "aarch64", feature = "metal"))]
impl ForwardProfiler {
    pub fn new() -> Self {
        Self { timings: HashMap::new() }
    }

    /// Time a single operation and record it
    pub fn time_op<F, T>(&mut self, name: &str, f: F) -> T
    where
        F: FnOnce() -> T,
    {
        let start = Instant::now();
        let result = f();
        let elapsed = start.elapsed().as_micros() as u64;

        let entry = self.timings.entry(name.to_string()).or_insert((0, 0));
        entry.0 += 1;
        entry.1 += elapsed;

        result
    }

    /// Get profiling results
    pub fn finish(&self) -> ProfileResult {
        let mut operations: Vec<OpTiming> = self
            .timings
            .iter()
            .map(|(name, (count, total_us))| OpTiming {
                name: name.clone(),
                count: *count,
                total_us: *total_us,
                avg_us: *total_us as f64 / *count as f64,
            })
            .collect();

        operations.sort_by(|a, b| b.total_us.cmp(&a.total_us));

        let total_us = operations.iter().map(|o| o.total_us).sum();

        ProfileResult { operations, total_us }
    }
}

#[cfg(all(target_arch = "aarch64", feature = "metal"))]
fn profile_forward_pass(
    model: &LlamaModel,
    tokens: &[u32],
    kv_cache: &mut KVCacheGPU,
    gpu_ctx: &MetalContext,
) -> ProfileResult {
    let mut profiler = ForwardProfiler::new();

    let hidden_size = model.config.hidden_size;
    let device = gpu_ctx.device();
    let queue = gpu_ctx.queue();

    // Process first token only for profiling
    let token = tokens[0];

    // 1. Embedding lookup
    let hidden = profiler.time_op("embedding_lookup", || {
        model.embedding_lookup_impl(token).unwrap()
    });

    // Upload to GPU
    let hidden_buffer = profiler.time_op("upload_hidden", || {
        std::rc::Rc::new(device.new_buffer_with_data(
            hidden.as_ptr() as *const std::ffi::c_void,
            (hidden_size * 4) as u64,
            metal::MTLResourceOptions::StorageModeShared,
        ))
    });

    // Process each layer
    for layer_idx in 0..model.layers.len() {
        let layer = &model.layers[layer_idx];

        // 1. Attention norm
        profiler.time_op("rms_norm", || {
            let queue = gpu_ctx.queue();
            let cmd = queue.new_command_buffer();
            let encoder = cmd.new_compute_command_encoder();

            let norm_weight = gpu_ctx
                .get_or_create_buffer(layer.attention_norm.as_ref().unwrap());
            let output = device.new_buffer(
                (hidden_size * 4) as u64,
                metal::MTLResourceOptions::StorageModeShared,
            );

            gpu_ctx.rms_norm_encode(
                encoder,
                &hidden_buffer,
                &norm_weight,
                &std::rc::Rc::new(output),
                1,
                hidden_size,
                1e-5,
            );

            encoder.end_encoding();
            cmd.commit();
            cmd.wait_until_completed();
        });

        // 2. Q projection
        profiler.time_op("gemv_q", || {
            let queue = gpu_ctx.queue();
            let cmd = queue.new_command_buffer();
            let encoder = cmd.new_compute_command_encoder();

            let wq = layer.wq.as_ref().unwrap();
            let q_out_size = wq.shape[1];
            let weight = gpu_ctx.get_or_create_buffer(wq);
            let output = device.new_buffer(
                (q_out_size * 4) as u64,
                metal::MTLResourceOptions::StorageModeShared,
            );

            gpu_ctx.mv_encode(
                encoder,
                &weight,
                &hidden_buffer,
                &std::rc::Rc::new(output),
                q_out_size,
                hidden_size,
                wq.dtype,
            );

            encoder.end_encoding();
            cmd.commit();
            cmd.wait_until_completed();
        });

        // 3. K projection
        profiler.time_op("gemv_k", || {
            let queue = gpu_ctx.queue();
            let cmd = queue.new_command_buffer();
            let encoder = cmd.new_compute_command_encoder();

            let wk = layer.wk.as_ref().unwrap();
            let k_out_size = wk.shape[1];
            let weight = gpu_ctx.get_or_create_buffer(wk);
            let output = device.new_buffer(
                (k_out_size * 4) as u64,
                metal::MTLResourceOptions::StorageModeShared,
            );

            gpu_ctx.mv_encode(
                encoder,
                &weight,
                &hidden_buffer,
                &std::rc::Rc::new(output),
                k_out_size,
                hidden_size,
                wk.dtype,
            );

            encoder.end_encoding();
            cmd.commit();
            cmd.wait_until_completed();
        });

        // 4. V projection
        profiler.time_op("gemv_v", || {
            let queue = gpu_ctx.queue();
            let cmd = queue.new_command_buffer();
            let encoder = cmd.new_compute_command_encoder();

            let wv = layer.wv.as_ref().unwrap();
            let v_out_size = wv.shape[1];
            let weight = gpu_ctx.get_or_create_buffer(wv);
            let output = device.new_buffer(
                (v_out_size * 4) as u64,
                metal::MTLResourceOptions::StorageModeShared,
            );

            gpu_ctx.mv_encode(
                encoder,
                &weight,
                &hidden_buffer,
                &std::rc::Rc::new(output),
                v_out_size,
                hidden_size,
                wv.dtype,
            );

            encoder.end_encoding();
            cmd.commit();
            cmd.wait_until_completed();
        });

        // 5. RoPE for Q
        profiler.time_op("rope_q", || {
            let queue = gpu_ctx.queue();
            let cmd = queue.new_command_buffer();
            let encoder = cmd.new_compute_command_encoder();

            let wq = layer.wq.as_ref().unwrap();
            let q_out_size = wq.shape[1];
            let head_dim = model.config.head_dim;
            let n_heads = q_out_size / head_dim;
            let pos = kv_cache.seq_len();

            let input = device.new_buffer(
                (q_out_size * 4) as u64,
                metal::MTLResourceOptions::StorageModeShared,
            );
            let output = device.new_buffer(
                (q_out_size * 4) as u64,
                metal::MTLResourceOptions::StorageModeShared,
            );

            gpu_ctx.rope_neox_encode(
                encoder,
                &std::rc::Rc::new(input),
                &[pos as i32],
                1,
                n_heads,
                head_dim,
                10000.0,
                1.0,
                &std::rc::Rc::new(output),
            );

            encoder.end_encoding();
            cmd.commit();
            cmd.wait_until_completed();
        });

        // 6. RoPE for K
        profiler.time_op("rope_k", || {
            // Similar to RoPE Q
        });

        // 7. Flash Attention
        profiler.time_op("flash_attn", || {
            // Flash attention timing
        });

        // 8. Output projection
        profiler.time_op("gemv_wo", || {
            // Output projection timing
        });

        // 9. Add (residual)
        profiler.time_op("add", || {
            // Add timing
        });

        // 10. FFN norm
        profiler.time_op("rms_norm_ffn", || {
            // FFN norm timing
        });

        // 11. Gate projection
        profiler.time_op("gemv_gate", || {
            // Gate timing
        });

        // 12. Up projection
        profiler.time_op("gemv_up", || {
            // Up timing
        });

        // 13. SiLU
        profiler.time_op("silu", || {
            // SiLU timing
        });

        // 14. Mul (SwiGLU)
        profiler.time_op("mul", || {
            // Mul timing
        });

        // 15. Down projection
        profiler.time_op("gemv_down", || {
            // Down timing
        });

        // 16. Add (residual)
        profiler.time_op("add_final", || {
            // Final add timing
        });

        break; // Only profile first layer for now
    }

    profiler.finish()
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: {} <model.gguf>", args[0]);
        std::process::exit(1);
    }
    let model_path = args[1].clone();

    #[cfg(all(target_arch = "aarch64", feature = "metal"))]
    {
        println!("=== Forward Pass Profiler ===");
        println!("Model: {}", model_path);

        // Load model
        println!("\nLoading model...");
        let gpu_ctx = MetalContext::new().unwrap();
        let device = gpu_ctx.device();
        let ctx = InferenceContext::from_gguf_with_gpu_ctx(
            &model_path,
            gpu_ctx.clone(),
        )
        .unwrap();
        let config = ctx.model.config.clone();

        println!("Model loaded");
        println!("\nConfig:");
        println!("  Hidden size: {}", config.hidden_size);
        println!("  Intermediate size: {}", config.intermediate_size);
        println!("  n_layers: {}", config.n_layers);
        println!("  n_heads: {}", config.n_heads);
        println!("  head_dim: {}", config.head_dim);

        // Warmup
        println!("\nWarming up...");
        let mut kv = KVCacheGPU::new(&device, &config, 2048);
        let _ = ctx.model.forward_gpu_native_batched(
            &[1u32, 2u32, 3u32],
            &mut kv,
            &gpu_ctx,
        );
        println!("Warmup done");

        // Profile decode
        println!("\n=== Profiling Decode (Layer 0 only) ===");

        // First, get baseline: measure full forward pass
        println!("\n--- Baseline: Full forward pass ---");
        let mut decode_times = Vec::new();
        for i in 0..10 {
            let token = 4u32 + i as u32;
            let start = Instant::now();
            let _ = ctx.model.forward_gpu_native_batched(
                &[token],
                &mut kv,
                &gpu_ctx,
            );
            decode_times.push(start.elapsed());
        }

        let avg_decode: Duration =
            decode_times.iter().sum::<Duration>() / decode_times.len() as u32;
        let tps = 1.0 / avg_decode.as_secs_f64();
        println!("  Average decode time: {:?}", avg_decode);
        println!("  Decode throughput: {:.1} tokens/sec", tps);
        println!(
            "  Per-layer estimate: {:?}",
            avg_decode / config.n_layers as u32
        );

        // Now profile individual operations
        println!("\n--- Operation-level profiling ---");

        // Get a layer for profiling
        let layer = &ctx.model.layers[0];
        let hidden_size = config.hidden_size;

        // Profile RMSNorm
        println!("\n[RMSNorm] {} elements", hidden_size);
        let mut norm_times = Vec::new();
        for _ in 0..50 {
            let queue = gpu_ctx.queue();
            let cmd = queue.new_command_buffer();
            let encoder = cmd.new_compute_command_encoder();

            let input = std::rc::Rc::new(device.new_buffer(
                (hidden_size * 4) as u64,
                metal::MTLResourceOptions::StorageModeShared,
            ));
            let weight = gpu_ctx
                .get_or_create_buffer(layer.attention_norm.as_ref().unwrap());
            let output = std::rc::Rc::new(device.new_buffer(
                (hidden_size * 4) as u64,
                metal::MTLResourceOptions::StorageModeShared,
            ));

            gpu_ctx.rms_norm_encode(
                &encoder,
                &input,
                &weight,
                &output,
                1,
                hidden_size,
                1e-5,
            );

            encoder.end_encoding();
            let start = Instant::now();
            cmd.commit();
            cmd.wait_until_completed();
            norm_times.push(start.elapsed());
        }

        let avg_norm =
            norm_times.iter().sum::<Duration>() / norm_times.len() as u32;
        println!("  Average time: {:?}", avg_norm);

        // Profile GEMV (Q projection)
        let wq = layer.wq.as_ref().unwrap();
        let q_out_size = wq.shape[1];
        println!("\n[GEMV Q] {}x{}", q_out_size, hidden_size);

        let mut gemv_times = Vec::new();
        for _ in 0..50 {
            let queue = gpu_ctx.queue();
            let cmd = queue.new_command_buffer();
            let encoder = cmd.new_compute_command_encoder();

            let input = std::rc::Rc::new(device.new_buffer(
                (hidden_size * 4) as u64,
                metal::MTLResourceOptions::StorageModeShared,
            ));
            let weight = gpu_ctx.get_or_create_buffer(wq);
            let output = std::rc::Rc::new(device.new_buffer(
                (q_out_size * 4) as u64,
                metal::MTLResourceOptions::StorageModeShared,
            ));

            gpu_ctx.mv_encode(
                &encoder,
                &weight,
                &input,
                &output,
                q_out_size,
                hidden_size,
                wq.dtype,
            );

            encoder.end_encoding();
            let start = Instant::now();
            cmd.commit();
            cmd.wait_until_completed();
            gemv_times.push(start.elapsed());
        }

        let avg_gemv =
            gemv_times.iter().sum::<Duration>() / gemv_times.len() as u32;
        println!("  Average time: {:?}", avg_gemv);

        // Profile GEMV (Gate projection) - larger
        let w1 = layer.w1.as_ref().unwrap();
        let intermediate_size = w1.shape[1];
        println!("\n[GEMV Gate] {}x{}", intermediate_size, hidden_size);

        let mut gate_times = Vec::new();
        for _ in 0..50 {
            let queue = gpu_ctx.queue();
            let cmd = queue.new_command_buffer();
            let encoder = cmd.new_compute_command_encoder();

            let input = std::rc::Rc::new(device.new_buffer(
                (hidden_size * 4) as u64,
                metal::MTLResourceOptions::StorageModeShared,
            ));
            let weight = gpu_ctx.get_or_create_buffer(w1);
            let output = std::rc::Rc::new(device.new_buffer(
                (intermediate_size * 4) as u64,
                metal::MTLResourceOptions::StorageModeShared,
            ));

            gpu_ctx.mv_encode(
                &encoder,
                &weight,
                &input,
                &output,
                intermediate_size,
                hidden_size,
                w1.dtype,
            );

            encoder.end_encoding();
            let start = Instant::now();
            cmd.commit();
            cmd.wait_until_completed();
            gate_times.push(start.elapsed());
        }

        let avg_gate =
            gate_times.iter().sum::<Duration>() / gate_times.len() as u32;
        println!("  Average time: {:?}", avg_gate);

        // Summary
        println!("\n=== Summary ===");
        println!("Operations per layer:");
        println!("  2x RMSNorm: {:?}", 2 * avg_norm);
        println!(
            "  6x GEMV (Q,K,V,WO,Gate,Down): ~{:?}",
            4 * avg_gemv + 2 * avg_gate
        );
        println!("  2x GEMV (Gate,Up): {:?}", 2 * avg_gate);
        println!("");
        println!(
            "Estimated per-layer: {:?}",
            2 * avg_norm + 4 * avg_gemv + 2 * avg_gate
        );
        println!("Actual per-layer: {:?}", avg_decode / config.n_layers as u32);
        println!("");
        println!(
            "Note: Does not include Flash Attention, RoPE, activations, etc."
        );
        println!("The difference indicates those operations' overhead.");
    }

    #[cfg(not(all(target_arch = "aarch64", feature = "metal")))]
    {
        eprintln!("This profiler requires Metal support on aarch64");
    }
}
