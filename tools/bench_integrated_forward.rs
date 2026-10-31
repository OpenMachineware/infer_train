// SPDX-License-Identifier: AGPL-3.0-or-later
//
// Benchmark integrated forward pass with per-layer GEMV batching

use std::env;
use std::time::Instant;

use infer_train::cpu::{
    CpuWorkBuffers, ForwardOp, ForwardPassWork, global_forward_executor,
};
use infer_train::inference::{KVCache, KVCacheBase};
use infer_train::model::config::ModelConfig;
use infer_train::model::gguf::GgufFile;
use infer_train::model::llama::LlamaModel;
use infer_train::model::tensor::DataType;
use infer_train::model::weights::WeightStore;
use infer_train::quant::types::{BlockQ8K, QK_K};

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: {} <model.gguf>", args[0]);
        std::process::exit(1);
    }

    let model_path = &args[1];
    println!("Loading model from {}", model_path);

    let gguf = GgufFile::open(model_path).expect("Failed to open GGUF");
    let store =
        WeightStore::from_gguf(model_path).expect("Failed to load model");
    let config = ModelConfig::from_gguf(&gguf);
    let model = LlamaModel::from_weights(&store, config);

    println!(
        "Model: {} layers, hidden_size={}",
        model.layers.len(),
        model.config.hidden_size
    );

    // Check executor threads
    let executor = global_forward_executor();
    let n_threads = executor.n_threads();
    println!("Executor threads: {}", n_threads);

    let hidden_size = model.config.hidden_size;
    let intermediate_size = model.config.intermediate_size;
    let n_heads = model.config.n_heads;
    let n_kv_heads = model.config.n_kv_heads;
    let head_dim = model.config.head_dim;
    let n_layers = model.layers.len();
    let vocab_size = model.config.vocab_size;

    let mut kv_cache = KVCache::new(
        &model.config,
        2048,
        infer_train::inference::KVCacheFormat::F32,
    );
    let mut buffers = CpuWorkBuffers::new(
        hidden_size,
        intermediate_size,
        n_heads,
        n_kv_heads,
        head_dim,
        head_dim,
        head_dim,
        vocab_size,
        n_heads * head_dim + n_kv_heads * head_dim * 2,
        2048,
        n_layers,
    );
    buffers.init_norm_weights(&model);

    // Check dtypes
    let first_layer = &model.layers[0];
    let wq_dtype = first_layer.wq.as_ref().unwrap().dtype;
    let wk_dtype = first_layer.wk.as_ref().unwrap().dtype;
    let wv_dtype = first_layer.wv.as_ref().unwrap().dtype;
    let wo_dtype = first_layer.wo.as_ref().unwrap().dtype;
    let w1_dtype = first_layer.w1.as_ref().unwrap().dtype;
    let w3_dtype = first_layer.w3.as_ref().unwrap().dtype;
    let w2_dtype = first_layer.w2.as_ref().unwrap().dtype;
    let output_dtype = model.output.as_ref().unwrap().dtype;

    let is_k_quant = |dt: DataType| {
        matches!(dt, DataType::Q4_K | DataType::Q5_K | DataType::Q6_K)
    };
    let dtype_to_code = |dt: DataType| -> u8 {
        match dt {
            DataType::Q4_K => 0,
            DataType::Q5_K => 1,
            DataType::Q6_K => 2,
            DataType::F16 => 3,
            _ => panic!("Unsupported dtype: {:?}", dt),
        }
    };

    let qkv_all_kq =
        is_k_quant(wq_dtype) && is_k_quant(wk_dtype) && is_k_quant(wv_dtype);
    let ffn_gate_up_kq = is_k_quant(w1_dtype) && is_k_quant(w3_dtype);

    println!(
        "\nDtypes: QKV K-quant={}, FFN gate+up K-quant={}",
        qkv_all_kq, ffn_gate_up_kq
    );
    println!("Output dtype: {:?}", output_dtype);

    // Pre-quantize input buffer
    let in_size_padded = ((hidden_size + QK_K - 1) / QK_K) * QK_K;
    let nb = in_size_padded / QK_K;
    let x_q8k_blocks = unsafe {
        std::slice::from_raw_parts_mut(
            buffers.q8k_input.as_mut_ptr() as *mut BlockQ8K,
            nb.max(intermediate_size / QK_K),
        )
    };

    // ========================================
    // Baseline
    // ========================================
    println!("\n=== Baseline (forward_pooled) ===");
    let iterations = 30;
    let mut times_baseline: Vec<f64> = Vec::with_capacity(iterations);

    for _ in 0..5 {
        kv_cache.clear();
        let _ = model.forward_pooled(&[1], &mut kv_cache, &mut buffers);
    }

    for _ in 0..iterations {
        kv_cache.clear();
        let start = Instant::now();
        let _ = model.forward_pooled(&[1], &mut kv_cache, &mut buffers);
        times_baseline.push(start.elapsed().as_micros() as f64);
    }

    let avg_baseline: f64 =
        times_baseline.iter().sum::<f64>() / iterations as f64;
    println!(
        "Baseline: {:.2} µs ({:.2} t/s)",
        avg_baseline,
        1e6 / avg_baseline
    );

    // ========================================
    // Integrated forward with executor
    // ========================================
    println!("\n=== Integrated forward ===");

    let mut times_integrated: Vec<f64> = Vec::with_capacity(iterations);

    for _ in 0..5 {
        kv_cache.clear();
        let _ = integrated_forward(
            &model,
            &mut kv_cache,
            &mut buffers,
            &executor,
            n_threads,
            qkv_all_kq,
            ffn_gate_up_kq,
            &dtype_to_code,
        );
    }

    for _ in 0..iterations {
        kv_cache.clear();
        let start = Instant::now();
        let _ = integrated_forward(
            &model,
            &mut kv_cache,
            &mut buffers,
            &executor,
            n_threads,
            qkv_all_kq,
            ffn_gate_up_kq,
            &dtype_to_code,
        );
        times_integrated.push(start.elapsed().as_micros() as f64);
    }

    let avg_integrated: f64 =
        times_integrated.iter().sum::<f64>() / iterations as f64;
    println!(
        "Integrated: {:.2} µs ({:.2} t/s)",
        avg_integrated,
        1e6 / avg_integrated
    );
    println!("Speedup: {:.2}x", avg_baseline / avg_integrated);
}

fn integrated_forward(
    model: &LlamaModel,
    kv_cache: &mut KVCache,
    buffers: &mut CpuWorkBuffers,
    executor: &infer_train::cpu::ForwardPassExecutor,
    n_threads: usize,
    qkv_all_kq: bool,
    ffn_gate_up_kq: bool,
    dtype_to_code: &impl Fn(DataType) -> u8,
) -> Option<Vec<f32>> {
    let hidden_size = model.config.hidden_size;
    let intermediate_size = model.config.intermediate_size;
    let n_heads = model.config.n_heads;
    let n_kv_heads = model.config.n_kv_heads;
    let head_dim = model.config.head_dim;
    let vocab_size = model.config.vocab_size;

    let q_size = n_heads * head_dim;
    let k_size = n_kv_heads * head_dim;
    let v_size = n_kv_heads * head_dim;
    let pos = kv_cache.seq_len();

    // 1. Embedding lookup
    model.embedding_lookup_into(1, &mut buffers.hidden)?;

    // Process each layer
    for (layer_idx, layer) in model.layers.iter().enumerate() {
        // 2. Attention norm
        let attn_norm_offset = layer_idx * 2 * hidden_size;
        model.apply_rms_norm_predequant(
            &buffers.hidden,
            &buffers.norm_weights
                [attn_norm_offset..attn_norm_offset + hidden_size],
            &mut buffers.hidden_alt,
        );

        // 3. QKV projections
        if qkv_all_kq {
            // Use ForwardPassExecutor for batched QKV
            let wq = layer.wq.as_ref().unwrap();
            let wk = layer.wk.as_ref().unwrap();
            let wv = layer.wv.as_ref().unwrap();

            // Quantize input once
            let in_size_padded = ((hidden_size + QK_K - 1) / QK_K) * QK_K;
            let nb = in_size_padded / QK_K;
            let x_q8k_blocks = unsafe {
                std::slice::from_raw_parts_mut(
                    buffers.q8k_input.as_mut_ptr() as *mut BlockQ8K,
                    nb,
                )
            };
            LlamaModel::quantize_input_to_q8k(
                &buffers.hidden_alt,
                x_q8k_blocks,
                &mut buffers.x_padded,
            );

            let ops = vec![
                ForwardOp::Gemv {
                    weights: wq.bytes().as_ptr(),
                    input: x_q8k_blocks.as_ptr() as *const u8,
                    output: buffers.q_buffer[..q_size].as_mut_ptr(),
                    k: hidden_size,
                    m: q_size,
                    dtype: dtype_to_code(wq.dtype),
                },
                ForwardOp::Gemv {
                    weights: wk.bytes().as_ptr(),
                    input: x_q8k_blocks.as_ptr() as *const u8,
                    output: buffers.k_buffer[..k_size].as_mut_ptr(),
                    k: hidden_size,
                    m: k_size,
                    dtype: dtype_to_code(wk.dtype),
                },
                ForwardOp::Gemv {
                    weights: wv.bytes().as_ptr(),
                    input: x_q8k_blocks.as_ptr() as *const u8,
                    output: buffers.v_buffer[..v_size].as_mut_ptr(),
                    k: hidden_size,
                    m: v_size,
                    dtype: dtype_to_code(wv.dtype),
                },
            ];

            let work = ForwardPassWork::with_ops(ops, n_threads);
            executor.execute(work);
        } else {
            // Fallback to single-thread for F16
            model.gemv_into_with_buffers_single_thread(
                &buffers.hidden_alt,
                layer.wq.as_ref().unwrap(),
                &mut buffers.q_buffer[..q_size],
                &mut buffers.q8k_input,
                &mut buffers.x_padded,
            );
            model.gemv_into_with_buffers_single_thread(
                &buffers.hidden_alt,
                layer.wk.as_ref().unwrap(),
                &mut buffers.k_buffer[..k_size],
                &mut buffers.q8k_input,
                &mut buffers.x_padded,
            );
            model.gemv_into_with_buffers_single_thread(
                &buffers.hidden_alt,
                layer.wv.as_ref().unwrap(),
                &mut buffers.v_buffer[..v_size],
                &mut buffers.q8k_input,
                &mut buffers.x_padded,
            );
        }

        // 4. RoPE
        model.apply_rope_into(
            &buffers.q_buffer[..q_size],
            pos,
            n_heads,
            head_dim,
            &mut buffers.rope_buffer,
        )?;
        buffers.q_buffer[..q_size]
            .copy_from_slice(&buffers.rope_buffer[..q_size]);
        model.apply_rope_into(
            &buffers.k_buffer[..k_size],
            pos,
            n_kv_heads,
            head_dim,
            &mut buffers.rope_buffer,
        )?;
        buffers.k_buffer[..k_size]
            .copy_from_slice(&buffers.rope_buffer[..k_size]);

        // 5. Attention
        model.attention_zero_alloc(
            &buffers.q_buffer[..q_size],
            &buffers.k_buffer[..k_size],
            &buffers.v_buffer[..v_size],
            layer_idx,
            kv_cache,
            n_heads,
            head_dim,
            n_kv_heads,
            &mut buffers.attn_out,
            &mut buffers.kv_dequant,
        )?;

        // 6. wo projection
        model.gemv_into_with_buffers_single_thread(
            &buffers.attn_out,
            layer.wo.as_ref().unwrap(),
            &mut buffers.hidden_alt,
            &mut buffers.q8k_input,
            &mut buffers.x_padded,
        );

        // 7. Residual
        for i in 0..hidden_size {
            buffers.hidden[i] += buffers.hidden_alt[i];
        }

        // 8. FFN norm
        let ffn_norm_offset = layer_idx * 2 * hidden_size + hidden_size;
        model.apply_rms_norm_predequant(
            &buffers.hidden,
            &buffers.norm_weights
                [ffn_norm_offset..ffn_norm_offset + hidden_size],
            &mut buffers.hidden_alt,
        );

        // 9. FFN
        if ffn_gate_up_kq {
            // Batch gate+up with executor
            let w1 = layer.w1.as_ref().unwrap();
            let w3 = layer.w3.as_ref().unwrap();

            // Quantize input
            let in_size_padded = ((hidden_size + QK_K - 1) / QK_K) * QK_K;
            let nb = in_size_padded / QK_K;
            let x_q8k_blocks = unsafe {
                std::slice::from_raw_parts_mut(
                    buffers.q8k_input.as_mut_ptr() as *mut BlockQ8K,
                    nb,
                )
            };
            LlamaModel::quantize_input_to_q8k(
                &buffers.hidden_alt,
                x_q8k_blocks,
                &mut buffers.x_padded,
            );

            let ops = vec![
                ForwardOp::Gemv {
                    weights: w1.bytes().as_ptr(),
                    input: x_q8k_blocks.as_ptr() as *const u8,
                    output: buffers.ffn_gate.as_mut_ptr(),
                    k: hidden_size,
                    m: intermediate_size,
                    dtype: dtype_to_code(w1.dtype),
                },
                ForwardOp::Gemv {
                    weights: w3.bytes().as_ptr(),
                    input: x_q8k_blocks.as_ptr() as *const u8,
                    output: buffers.ffn_up.as_mut_ptr(),
                    k: hidden_size,
                    m: intermediate_size,
                    dtype: dtype_to_code(w3.dtype),
                },
            ];

            let work = ForwardPassWork::with_ops(ops, n_threads);
            executor.execute(work);

            // Activation
            for i in 0..intermediate_size {
                let silu =
                    buffers.ffn_gate[i] / (1.0 + buffers.ffn_gate[i].exp());
                buffers.ffn_up[i] = silu * buffers.ffn_up[i];
            }

            // w2 with ForwardPassExecutor
            let w2 = layer.w2.as_ref().unwrap();

            // Quantize ffn_up for w2
            let in_size_padded = ((intermediate_size + QK_K - 1) / QK_K) * QK_K;
            let nb = in_size_padded / QK_K;
            let x_q8k_blocks = unsafe {
                std::slice::from_raw_parts_mut(
                    buffers.q8k_input.as_mut_ptr() as *mut BlockQ8K,
                    nb,
                )
            };
            LlamaModel::quantize_input_to_q8k(
                &buffers.ffn_up,
                x_q8k_blocks,
                &mut buffers.x_padded,
            );

            let ops = vec![ForwardOp::Gemv {
                weights: w2.bytes().as_ptr(),
                input: x_q8k_blocks.as_ptr() as *const u8,
                output: buffers.hidden.as_mut_ptr(),
                k: intermediate_size,
                m: hidden_size,
                dtype: dtype_to_code(w2.dtype),
            }];
            let work = ForwardPassWork::with_ops(ops, n_threads);
            executor.execute(work);
        } else {
            model.compute_ffn_zero_alloc(
                layer,
                &buffers.hidden_alt,
                &mut buffers.ffn_gate,
                &mut buffers.ffn_up,
                &mut buffers.hidden,
                &mut buffers.q8k_input,
                &mut buffers.x_padded,
            )?;
        }

        kv_cache.advance();
    }

    // Final norm
    model.apply_rms_norm_predequant(
        &buffers.hidden,
        &buffers.norm_weight,
        &mut buffers.hidden_alt,
    );

    // Output projection - use ForwardPassExecutor for F16 (2.7x faster)
    let output_weight = model.output.as_ref().unwrap();

    let ops = vec![ForwardOp::Gemv {
        weights: output_weight.bytes().as_ptr(),
        input: buffers.hidden_alt.as_ptr() as *const u8,
        output: buffers.logits[..vocab_size].as_mut_ptr(),
        k: hidden_size,
        m: vocab_size,
        dtype: dtype_to_code(output_weight.dtype),
    }];
    let work = ForwardPassWork::with_ops(ops, n_threads);
    executor.execute(work);

    Some(buffers.logits[..vocab_size].to_vec())
}
